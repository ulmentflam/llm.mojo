# ===----------------------------------------------------------------------=== #
# tests/test_lowp_bwd.mojo — Chunk E gate (docs/ai/fp8_training_design.md §6,
# "Chunk E — Backward FP8 integration"), Phase-1 standalone unit level:
#
# Exercises the `use_lowp=True` branches added to `llmm/matmul.mojo`'s
# `matmul_d_input_bwd` (dgrad: E4M3 weight x E5M2 d_output -> bf16 d_input,
# transpose_a=True/transpose_b=False) and `matmul_d_weight_bwd` (wgrad: E4M3
# input x E5M2 d_output -> bf16 d_weight, transpose_a=True/transpose_b=True),
# both wired through `lowp_gemm` (Chunk B) with real Chunk-C `AmaxState`
# scales (not host-computed constants) — i.e. this exercises the actual
# AmaxState -> matmul_bwd coupling those two chunks hand off across.
#
# Compares each fp8-branch (`use_lowp=True`) output against the SAME
# function's existing bf16 path (`use_lowp=False`) on identical inputs —
# per-tensor cosine similarity > 0.99 and relative L2 < 0.1, matching Chunk
# E's gate metric (MEMORY.md `weak-gates-overrule-nothing`: no flat atol).
#
# This is Chunk E's PHASE 1 gate: standalone, locally-constructed AmaxStates,
# not yet wired into train_gpt2.mojo's training loop (that's Phase 2, after
# Chunk D's shared per-tensor scaling-state container lands — see this
# worktree's final report). GELU-backward fusion (DGELU) is NOT part of the
# fp8 GEMM epilogue (cuBLASLt fp8 has restricted epilogue support, design §5
# item 2) — it runs as the existing standalone bf16 kernel afterward, same as
# the `USE_GELU_FUSION=False` bf16 path; `test_matmul_d_input_bwd_lowp_gelu`
# below covers that composition explicitly.
#
# GPU-only, guarded by `has_nvidia_gpu_accelerator()`. Run under
# `flock -w 10800 /tmp/llmm-gpu.lock -c 'pixi run -e cuda mojo run -I . tests/test_lowp_bwd.mojo'`.
# ===----------------------------------------------------------------------=== #

from std.math import sqrt
from std.random import random_float64, seed
from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import TestSuite, assert_true

from llmm.lowp import FP8_SPEC
from llmm.amax import (
    AmaxState,
    compute_amax,
    kernel_ptr_as_immut,
    device_buf_mut_ptr,
)
from llmm.matmul import matmul_d_input_bwd, matmul_d_weight_bwd
from llmm.memory import MutKernelPtr, ImmutKernelPtr


# ===----------------------------------------------------------------------=== #
# Small host<->device helpers (mirrors tests/test_amax.mojo / test_lowp_gemm.mojo).
# ===----------------------------------------------------------------------=== #


def _read_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32]
) raises -> Float32:
    var host = ctx.enqueue_create_host_buffer[DType.float32](1)
    buf.enqueue_copy_to(host)
    ctx.synchronize()
    return host.unsafe_ptr()[0]


def _random_bf16(
    ctx: DeviceContext, n: Int, scale: Float32
) raises -> DeviceBuffer[DType.bfloat16]:
    var host = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    for i in range(n):
        var v = Float32((random_float64() * 2.0 - 1.0)) * scale
        host.unsafe_ptr()[i] = v.cast[DType.bfloat16]()
    var dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    dev.enqueue_copy_from(host)
    ctx.synchronize()
    return dev


def _zeros_bf16(
    ctx: DeviceContext, n: Int
) raises -> DeviceBuffer[DType.bfloat16]:
    var host = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    for i in range(n):
        host.unsafe_ptr()[i] = Float32(0.0).cast[DType.bfloat16]()
    var dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    dev.enqueue_copy_from(host)
    ctx.synchronize()
    return dev


def _clone_bf16(
    ctx: DeviceContext, src: DeviceBuffer[DType.bfloat16], n: Int
) raises -> DeviceBuffer[DType.bfloat16]:
    var host = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    src.enqueue_copy_to(host)
    ctx.synchronize()
    var dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    dev.enqueue_copy_from(host)
    ctx.synchronize()
    return dev


def _cosine_and_rel_l2(
    ctx: DeviceContext,
    got: DeviceBuffer[DType.bfloat16],
    want: DeviceBuffer[DType.bfloat16],
    n: Int,
    label: String,
) raises -> None:
    var host_got = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    var host_want = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    got.enqueue_copy_to(host_got)
    want.enqueue_copy_to(host_want)
    ctx.synchronize()

    var l2_err = Float32(0.0)
    var dot = Float32(0.0)
    var norm_got = Float32(0.0)
    var norm_want = Float32(0.0)
    for i in range(n):
        var g = host_got.unsafe_ptr()[i].cast[DType.float32]()
        var w = host_want.unsafe_ptr()[i].cast[DType.float32]()
        assert_true(g == g, label + ": NaN in fp8 output at " + String(i))
        assert_true(
            g > Float32(-1e30) and g < Float32(1e30),
            label + ": Inf/overflow in fp8 output at " + String(i),
        )
        var err = g - w
        l2_err += err * err
        dot += g * w
        norm_got += g * g
        norm_want += w * w
    var rel_l2 = sqrt(l2_err / (norm_want + Float32(1e-12)))
    var cosine = dot / (sqrt(norm_got) * sqrt(norm_want) + Float32(1e-12))
    assert_true(
        rel_l2 < Float32(0.1),
        label + ": relative L2 " + String(rel_l2) + " >= 0.1",
    )
    assert_true(
        cosine > Float32(0.99),
        label + ": cosine similarity " + String(cosine) + " <= 0.99",
    )


# ===----------------------------------------------------------------------=== #
# AmaxState-driven scale helper — mirrors the real calling contract Chunk
# D/E's training-loop integration will use (llmm/amax.mojo's AmaxState
# docstring): compute_amax on the operand about to be quantized, then
# update_scale (warmup at step 0 uses that same amax -- exactly right for a
# single standalone GEMM call, no history needed).
# ===----------------------------------------------------------------------=== #


struct _Scale[fmt_dtype: DType]:
    var state: AmaxState[FP8_SPEC]
    var host_scale: Float32

    def __init__(
        out self,
        ctx: DeviceContext,
        data: ImmutKernelPtr[DType.bfloat16],
        n: Int,
    ) raises:
        self.state = AmaxState[FP8_SPEC](ctx)
        var amax_buf = ctx.enqueue_create_buffer[DType.float32](1)
        compute_amax[FP8_SPEC, DType.bfloat16](
            device_buf_mut_ptr(amax_buf), data, n, ctx
        )
        ctx.synchronize()
        self.state.update_scale[Self.fmt_dtype](
            kernel_ptr_as_immut(device_buf_mut_ptr(amax_buf)), ctx
        )
        self.host_scale = _read_f32(ctx, self.state.scale)

    def scale_inv_ptr(self) raises -> ImmutKernelPtr[DType.float32]:
        return kernel_ptr_as_immut(device_buf_mut_ptr(self.state.scale_inv))


# ===----------------------------------------------------------------------=== #
# matmul_d_input_bwd (dgrad): E4M3 weight x E5M2 d_output -> bf16 d_input.
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

    var weight_scale_state = _Scale[FP8_SPEC.fwd_dtype](
        ctx,
        weight.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        OC * C,
    )
    var d_output_scale_state = _Scale[FP8_SPEC.bwd_dtype](
        ctx,
        d_output.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        ROWS * OC,
    )

    var weight_t_scratch = ctx.enqueue_create_buffer[DType.uint8](C * OC)
    var d_output_scratch = ctx.enqueue_create_buffer[DType.uint8](ROWS * OC)
    ctx.synchronize()

    # bf16 reference (use_lowp=False -- the existing, already-validated path).
    if use_gelu_case:
        matmul_d_input_bwd[DType.bfloat16, "gpu", use_gelu=True](
            device_buf_mut_ptr(d_input_ref),
            kernel_ptr_as_immut(device_buf_mut_ptr(d_output)),
            kernel_ptr_as_immut(device_buf_mut_ptr(weight)),
            kernel_ptr_as_immut(device_buf_mut_ptr(pre_gelu)),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            ctx,
        )
    else:
        matmul_d_input_bwd[DType.bfloat16, "gpu", use_gelu=False](
            device_buf_mut_ptr(d_input_ref),
            kernel_ptr_as_immut(device_buf_mut_ptr(d_output)),
            kernel_ptr_as_immut(device_buf_mut_ptr(weight)),
            kernel_ptr_as_immut(device_buf_mut_ptr(pre_gelu)),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            ctx,
        )
    ctx.synchronize()

    # fp8 path (use_lowp=True).
    if use_gelu_case:
        matmul_d_input_bwd[DType.bfloat16, "gpu", use_gelu=True, use_lowp=True](
            device_buf_mut_ptr(d_input_lowp),
            kernel_ptr_as_immut(device_buf_mut_ptr(d_output)),
            kernel_ptr_as_immut(device_buf_mut_ptr(weight)),
            kernel_ptr_as_immut(device_buf_mut_ptr(pre_gelu)),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            ctx,
            device_buf_mut_ptr(weight_t_scratch),
            device_buf_mut_ptr(d_output_scratch),
            weight_scale_state.host_scale,
            weight_scale_state.scale_inv_ptr(),
            d_output_scale_state.host_scale,
            d_output_scale_state.scale_inv_ptr(),
        )
    else:
        matmul_d_input_bwd[
            DType.bfloat16, "gpu", use_gelu=False, use_lowp=True
        ](
            device_buf_mut_ptr(d_input_lowp),
            kernel_ptr_as_immut(device_buf_mut_ptr(d_output)),
            kernel_ptr_as_immut(device_buf_mut_ptr(weight)),
            kernel_ptr_as_immut(device_buf_mut_ptr(pre_gelu)),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            ctx,
            device_buf_mut_ptr(weight_t_scratch),
            device_buf_mut_ptr(d_output_scratch),
            weight_scale_state.host_scale,
            weight_scale_state.scale_inv_ptr(),
            d_output_scale_state.host_scale,
            d_output_scale_state.scale_inv_ptr(),
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
# matmul_d_weight_bwd (wgrad): E4M3 input x E5M2 d_output -> bf16 d_weight.
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
    var scratch = _zeros_bf16(ctx, ROWS * OC)  # unused by the cuBLASLt path

    var seed_vals = _random_bf16(ctx, OC * C, Float32(0.5))
    var d_weight_ref = _clone_bf16(ctx, seed_vals, OC * C)
    var d_weight_lowp = _clone_bf16(ctx, seed_vals, OC * C)

    var input_scale_state = _Scale[FP8_SPEC.fwd_dtype](
        ctx,
        input.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        ROWS * C,
    )
    var d_output_scale_state = _Scale[FP8_SPEC.bwd_dtype](
        ctx,
        d_output.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
        ROWS * OC,
    )

    var input_t_scratch = ctx.enqueue_create_buffer[DType.uint8](C * ROWS)
    var d_output_t_scratch = ctx.enqueue_create_buffer[DType.uint8](OC * ROWS)
    ctx.synchronize()

    if accumulate_case:
        matmul_d_weight_bwd[DType.bfloat16, "gpu", accumulate=True](
            device_buf_mut_ptr(d_weight_ref),
            kernel_ptr_as_immut(device_buf_mut_ptr(d_output)),
            kernel_ptr_as_immut(device_buf_mut_ptr(input)),
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
            kernel_ptr_as_immut(device_buf_mut_ptr(d_output)),
            kernel_ptr_as_immut(device_buf_mut_ptr(input)),
            device_buf_mut_ptr(scratch),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            ctx,
        )
    ctx.synchronize()

    if accumulate_case:
        matmul_d_weight_bwd[
            DType.bfloat16, "gpu", accumulate=True, use_lowp=True
        ](
            device_buf_mut_ptr(d_weight_lowp),
            kernel_ptr_as_immut(device_buf_mut_ptr(d_output)),
            kernel_ptr_as_immut(device_buf_mut_ptr(input)),
            device_buf_mut_ptr(scratch),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            ctx,
            device_buf_mut_ptr(input_t_scratch),
            device_buf_mut_ptr(d_output_t_scratch),
            input_scale_state.host_scale,
            input_scale_state.scale_inv_ptr(),
            d_output_scale_state.host_scale,
            d_output_scale_state.scale_inv_ptr(),
        )
    else:
        matmul_d_weight_bwd[
            DType.bfloat16, "gpu", accumulate=False, use_lowp=True
        ](
            device_buf_mut_ptr(d_weight_lowp),
            kernel_ptr_as_immut(device_buf_mut_ptr(d_output)),
            kernel_ptr_as_immut(device_buf_mut_ptr(input)),
            device_buf_mut_ptr(scratch),
            Int64(ROWS),
            Int64(1),
            Int64(C),
            Int64(OC),
            ctx,
            device_buf_mut_ptr(input_t_scratch),
            device_buf_mut_ptr(d_output_t_scratch),
            input_scale_state.host_scale,
            input_scale_state.scale_inv_ptr(),
            d_output_scale_state.host_scale,
            d_output_scale_state.scale_inv_ptr(),
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
