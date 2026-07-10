# ===----------------------------------------------------------------------=== #
# tests/test_lowp_bwd.mojo — Chunk E gate (docs/ai/fp8_training_design.md §6,
# "Chunk E — Backward FP8 integration"):
#
# Exercises the fp8 sibling functions added to `llmm/matmul.mojo` (mirroring
# Chunk D's `matmul_fwd`/`matmul_fwd_lowp` split rather than branching inside
# the existing bf16 functions — see the module comment above
# `matmul_d_input_bwd_lowp` in llmm/matmul.mojo):
#   - `matmul_d_input_bwd_lowp` (dgrad: E4M3 weight x E5M2 d_output -> bf16
#     d_input, transpose_a=True/transpose_b=False)
#   - `matmul_d_weight_bwd_lowp` (wgrad: E4M3 input x E5M2 d_output -> bf16
#     d_weight, transpose_a=True/transpose_b=True)
#   - `matmul_bwd_lowp` (bundles bias grad + both of the above, matching
#     `matmul_bwd`'s own bundling, and owns the ONE-PER-STEP `doutput_state`
#     update both sub-GEMMs then read read-only)
#
# All three read scale/scale_inv from real Chunk-C `AmaxState`s via Chunk D's
# device-pointer-scale `lowp_gemm_devscale` primitive (no host readback on the
# critical path) — this exercises the actual AmaxState -> matmul_bwd_lowp
# coupling train_gpt2.mojo's backward call sites use.
#
# Compares each fp8 sibling's output against the existing bf16 function
# (`matmul_d_input_bwd`/`matmul_d_weight_bwd`/`matmul_bwd`) on identical
# inputs — per-tensor cosine similarity > 0.99 and relative L2 < 0.1,
# matching Chunk E's gate metric (MEMORY.md `weak-gates-overrule-nothing`:
# no flat atol).
#
# GPU-only, guarded by `has_nvidia_gpu_accelerator()`. Run under
# `flock -w 10800 /tmp/llmm-gpu.lock -c 'pixi run -e cuda mojo run -I . tests/test_lowp_bwd.mojo'`.
# ===----------------------------------------------------------------------=== #

from std.random import seed
from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import TestSuite

from llmm.lowp import FP8_SPEC
from llmm.amax import (
    AmaxState,
    compute_amax,
    kernel_ptr_as_immut,
    device_buf_mut_ptr,
)
from llmm.matmul import (
    matmul_d_input_bwd,
    matmul_d_weight_bwd,
    matmul_bwd,
    matmul_d_input_bwd_lowp,
    matmul_d_weight_bwd_lowp,
    matmul_bwd_lowp,
)
from llmm.memory import MutKernelPtr, ImmutKernelPtr

from _lowp_test_common import (
    random_bf16 as _random_bf16,
    zeros_bf16 as _zeros_bf16,
    clone_bf16 as _clone_bf16,
    cosine_and_rel_l2 as _cosine_and_rel_l2,
)


def _prime_state[
    fmt_dtype: DType
](
    mut state: AmaxState[FP8_SPEC],
    data: ImmutKernelPtr[DType.bfloat16],
    n: Int,
    ctx: DeviceContext,
) raises -> None:
    """Compute this tensor's amax and push it into `state` via
    `update_scale` (warmup semantics: `AmaxState`'s calling contract in
    llmm/amax.mojo -- the first `amax_history_len` calls use the just-
    computed amax directly, exactly right for a single standalone-test
    tensor). Standalone-test stand-in for what `matmul_fwd_lowp` (weight/
    input) and `matmul_bwd_lowp` (doutput) do in the real training loop.
    """
    var amax_buf = ctx.enqueue_create_buffer[DType.float32](1)
    compute_amax[FP8_SPEC, DType.bfloat16](
        device_buf_mut_ptr(amax_buf), data, n, ctx
    )
    ctx.synchronize()
    state.update_scale[fmt_dtype](
        kernel_ptr_as_immut(device_buf_mut_ptr(amax_buf)), ctx
    )


# ===----------------------------------------------------------------------=== #
# matmul_d_input_bwd_lowp (dgrad): E4M3 weight x E5M2 d_output -> bf16 d_input.
#
# weight_state/doutput_state are pre-"warmed" AmaxStates: one `update_scale`
# call each before the timed comparison, mimicking `matmul_fwd_lowp` having
# already updated `weight_state` this step (dgrad's calling contract, per
# llmm/matmul.mojo's module comment above `matmul_d_input_bwd_lowp`) and
# `matmul_bwd_lowp` having already updated `doutput_state` once before this
# sub-GEMM runs -- this standalone test calls `matmul_d_input_bwd_lowp`
# directly (not through `matmul_bwd_lowp`), so it does that priming itself.
# ===----------------------------------------------------------------------=== #


def _run_d_input_bwd_case(use_gelu_case: Bool) raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime ROWS = 128
    comptime C = 64  # in_channels
    comptime OC = 96  # out_channels

    seed(4242)
    var d_output = _random_bf16(ctx, ROWS * OC, Float32(1.0))
    var weight = _random_bf16(ctx, OC * C, Float32(1.0))
    var pre_gelu = _random_bf16(ctx, ROWS * C, Float32(1.0))

    var d_input_ref = _zeros_bf16(ctx, ROWS * C)
    var d_input_lowp = _zeros_bf16(ctx, ROWS * C)

    var weight_state = AmaxState[FP8_SPEC](ctx)
    var doutput_state = AmaxState[FP8_SPEC](ctx)

    # Prime both states from THIS tensor's amax before the fp8 call (mirrors
    # matmul_fwd_lowp's weight update / matmul_bwd_lowp's doutput update --
    # see the module docstring above).
    _prime_state[FP8_SPEC.fwd_dtype](
        weight_state,
        weight.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        OC * C,
        ctx,
    )
    _prime_state[FP8_SPEC.bwd_dtype](
        doutput_state,
        d_output.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        ROWS * OC,
        ctx,
    )

    # bf16 reference: the existing, already-validated matmul_d_input_bwd.
    if use_gelu_case:
        matmul_d_input_bwd[DType.bfloat16, "gpu", use_gelu=True](
            device_buf_mut_ptr(d_input_ref),
            kernel_ptr_as_immut_bf16(d_output),
            kernel_ptr_as_immut_bf16(weight),
            kernel_ptr_as_immut_bf16(pre_gelu),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            ctx,
        )
    else:
        matmul_d_input_bwd[DType.bfloat16, "gpu", use_gelu=False](
            device_buf_mut_ptr(d_input_ref),
            kernel_ptr_as_immut_bf16(d_output),
            kernel_ptr_as_immut_bf16(weight),
            kernel_ptr_as_immut_bf16(pre_gelu),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            ctx,
        )
    ctx.synchronize()

    # fp8 sibling.
    if use_gelu_case:
        matmul_d_input_bwd_lowp[DType.bfloat16, "gpu", use_gelu=True](
            device_buf_mut_ptr(d_input_lowp),
            kernel_ptr_as_immut_bf16(d_output),
            kernel_ptr_as_immut_bf16(weight),
            kernel_ptr_as_immut_bf16(pre_gelu),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            weight_state,
            doutput_state,
            ctx,
        )
    else:
        matmul_d_input_bwd_lowp[DType.bfloat16, "gpu", use_gelu=False](
            device_buf_mut_ptr(d_input_lowp),
            kernel_ptr_as_immut_bf16(d_output),
            kernel_ptr_as_immut_bf16(weight),
            kernel_ptr_as_immut_bf16(pre_gelu),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            weight_state,
            doutput_state,
            ctx,
        )
    ctx.synchronize()

    _cosine_and_rel_l2(
        ctx,
        d_input_lowp,
        d_input_ref,
        ROWS * C,
        "dgrad" + ("_gelu" if use_gelu_case else "_nogelu"),
    )


def test_matmul_d_input_bwd_lowp_no_gelu() raises:
    _run_d_input_bwd_case(False)


def test_matmul_d_input_bwd_lowp_gelu() raises:
    _run_d_input_bwd_case(True)


# ===----------------------------------------------------------------------=== #
# matmul_d_weight_bwd_lowp (wgrad): E4M3 input x E5M2 d_output -> bf16 d_weight.
# ===----------------------------------------------------------------------=== #


def _run_d_weight_bwd_case(accumulate_case: Bool) raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime ROWS = 128
    comptime C = 64  # in_channels
    comptime OC = 96  # out_channels

    seed(9797)
    var d_output = _random_bf16(ctx, ROWS * OC, Float32(1.0))
    var input = _random_bf16(ctx, ROWS * C, Float32(1.0))

    var seed_vals = _random_bf16(ctx, OC * C, Float32(0.5))
    var d_weight_ref = _clone_bf16(ctx, seed_vals, OC * C)
    var d_weight_lowp = _clone_bf16(ctx, seed_vals, OC * C)

    var input_state = AmaxState[FP8_SPEC](ctx)
    var doutput_state = AmaxState[FP8_SPEC](ctx)
    _prime_state[FP8_SPEC.fwd_dtype](
        input_state,
        input.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        ROWS * C,
        ctx,
    )
    _prime_state[FP8_SPEC.bwd_dtype](
        doutput_state,
        d_output.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        ROWS * OC,
        ctx,
    )

    var scratch = _zeros_bf16(ctx, ROWS * OC)  # unused by the cuBLASLt path
    if accumulate_case:
        matmul_d_weight_bwd[DType.bfloat16, "gpu", accumulate=True](
            device_buf_mut_ptr(d_weight_ref),
            kernel_ptr_as_immut_bf16(d_output),
            kernel_ptr_as_immut_bf16(input),
            device_buf_mut_ptr(scratch),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            ctx,
        )
    else:
        matmul_d_weight_bwd[DType.bfloat16, "gpu", accumulate=False](
            device_buf_mut_ptr(d_weight_ref),
            kernel_ptr_as_immut_bf16(d_output),
            kernel_ptr_as_immut_bf16(input),
            device_buf_mut_ptr(scratch),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            ctx,
        )
    ctx.synchronize()

    if accumulate_case:
        matmul_d_weight_bwd_lowp[DType.bfloat16, "gpu", accumulate=True](
            device_buf_mut_ptr(d_weight_lowp),
            kernel_ptr_as_immut_bf16(d_output),
            kernel_ptr_as_immut_bf16(input),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            input_state,
            doutput_state,
            ctx,
        )
    else:
        matmul_d_weight_bwd_lowp[DType.bfloat16, "gpu", accumulate=False](
            device_buf_mut_ptr(d_weight_lowp),
            kernel_ptr_as_immut_bf16(d_output),
            kernel_ptr_as_immut_bf16(input),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            input_state,
            doutput_state,
            ctx,
        )
    ctx.synchronize()

    _cosine_and_rel_l2(
        ctx,
        d_weight_lowp,
        d_weight_ref,
        OC * C,
        "wgrad" + ("_accum" if accumulate_case else "_noaccum"),
    )


def test_matmul_d_weight_bwd_lowp_no_accum() raises:
    _run_d_weight_bwd_case(False)


def test_matmul_d_weight_bwd_lowp_accum() raises:
    _run_d_weight_bwd_case(True)


# ===----------------------------------------------------------------------=== #
# matmul_bwd_lowp: the bundled entry point (bias grad + dgrad + wgrad) used
# from train_gpt2.mojo's backward call sites -- exercises the ONE-PER-STEP
# doutput_state update this function owns (see its docstring in
# llmm/matmul.mojo) end-to-end, not just the two sub-GEMMs individually.
# ===----------------------------------------------------------------------=== #


def test_matmul_bwd_lowp_end_to_end() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime ROWS = 128
    comptime C = 64
    comptime OC = 96

    seed(1234)
    var d_output = _random_bf16(ctx, ROWS * OC, Float32(1.0))
    var input = _random_bf16(ctx, ROWS * C, Float32(1.0))
    var weight = _random_bf16(ctx, OC * C, Float32(1.0))

    var d_input_ref = _zeros_bf16(ctx, ROWS * C)
    var d_input_lowp = _zeros_bf16(ctx, ROWS * C)
    var d_weight_ref = _zeros_bf16(ctx, OC * C)
    var d_weight_lowp = _zeros_bf16(ctx, OC * C)
    var d_bias_ref = _zeros_bf16(ctx, OC)
    var d_bias_lowp = _zeros_bf16(ctx, OC)
    var scratch = _zeros_bf16(ctx, ROWS * OC)

    var input_state = AmaxState[FP8_SPEC](ctx)
    var weight_state = AmaxState[FP8_SPEC](ctx)
    var doutput_state = AmaxState[FP8_SPEC](ctx)
    # Prime input/weight from a prior ("forward-equivalent") update, mirroring
    # matmul_fwd_lowp having already run this step -- matmul_bwd_lowp itself
    # only updates doutput_state (see its docstring).
    _prime_state[FP8_SPEC.fwd_dtype](
        input_state,
        input.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        ROWS * C,
        ctx,
    )
    _prime_state[FP8_SPEC.fwd_dtype](
        weight_state,
        weight.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        OC * C,
        ctx,
    )

    matmul_bwd[DType.bfloat16, "gpu", use_gelu=False](
        device_buf_mut_ptr(d_input_ref),
        device_buf_mut_ptr(d_weight_ref),
        device_buf_mut_ptr(d_bias_ref),
        kernel_ptr_as_immut_bf16(d_output),
        kernel_ptr_as_immut_bf16(input),
        kernel_ptr_as_immut_bf16(weight),
        kernel_ptr_as_immut_bf16(input),  # dummy pre_gelu (use_gelu=False)
        device_buf_mut_ptr(scratch),
        Int64(ROWS),
        Int64(1),
        Int64(C),
        Int64(OC),
        ctx,
    )
    ctx.synchronize()

    matmul_bwd_lowp[DType.bfloat16, "gpu", use_gelu=False](
        device_buf_mut_ptr(d_input_lowp),
        device_buf_mut_ptr(d_weight_lowp),
        device_buf_mut_ptr(d_bias_lowp),
        kernel_ptr_as_immut_bf16(d_output),
        kernel_ptr_as_immut_bf16(input),
        kernel_ptr_as_immut_bf16(weight),
        kernel_ptr_as_immut_bf16(input),  # dummy pre_gelu (use_gelu=False)
        Int64(ROWS),
        Int64(1),
        Int64(C),
        Int64(OC),
        input_state,
        weight_state,
        doutput_state,
        ctx,
    )
    ctx.synchronize()

    _cosine_and_rel_l2(
        ctx, d_input_lowp, d_input_ref, ROWS * C, "bwd_lowp_d_input"
    )
    _cosine_and_rel_l2(
        ctx, d_weight_lowp, d_weight_ref, OC * C, "bwd_lowp_d_weight"
    )
    _cosine_and_rel_l2(ctx, d_bias_lowp, d_bias_ref, OC, "bwd_lowp_d_bias")


@always_inline
def kernel_ptr_as_immut_bf16(
    buf: DeviceBuffer[DType.bfloat16],
) -> ImmutKernelPtr[DType.bfloat16]:
    return buf.unsafe_ptr().as_immutable().as_unsafe_any_origin()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
