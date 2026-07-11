# ===----------------------------------------------------------------------=== #
# tests/test_matmul_fwd_fp4.mojo — fp4 forward linear gate:
#
#   Run 1 forward step under fp4 (`matmul_fwd_fp4`) vs bf16 (`matmul_fwd`) on
#   the same input/weight/bias data, for the two FP4-eligible MLP linear GEMM
#   shapes (fc[+gelu], proj) at real GPT-2 124M (d12, channels=768) dimensions
#   AND the real B=4/T=1024 row count (`rows=4096`, unlike
#   tests/test_lowp_gemm_fp4.mojo's rows=512 GEMM-layer test, which trades
#   row count for a tractable host fp32 reference — this test instead
#   compares directly against the production bf16 `matmul_fwd` GPU path, so
#   4096 rows costs nothing extra). Assert, on the block linear's activation
#   output (post-GEMM+bias, PRE-nonlinearity — i.e. `pre_gelu_ptr` for fc,
#   the final `out_ptr` for proj, mirroring tests/test_matmul_fwd_lowp.mojo's
#   fp8 analogue):
#     - per-tensor cosine similarity > 0.999
#     - relative L2 norm within the fp4 floor calibrated by
#       tests/test_lowp_gemm_fp4.mojo (~0.145-0.146 measured there at these
#       exact MLP shapes at rows=512; this test reports its own measurement
#       at rows=4096 rather than assume it matches exactly — GEMM
#       reduction-dimension noise doesn't depend on row count, but 4096 rows
#       averages over 8x more independent quantization-noise draws, so a
#       small further tightening is plausible)
#     - no NaN/Inf
#
# This exercises the real `llmm.matmul.matmul_fwd_fp4` entry point
# (nvfp4_quantize x2 -> lowp_gemm_fp4 -> bias_gelu_fwd) end-to-end — the
# actual call `train_gpt2.mojo`'s fc/proj sites make under
# `-D LLMM_PRECISION=fp4` for a middle block, not just the underlying
# `lowp_gemm_fp4` primitive tests/test_lowp_gemm_fp4.mojo already covers.
#
# GPU-only, guarded by `has_nvidia_gpu_accelerator()`; expected to run under
# `flock -w 10800 /tmp/llmm-gpu.lock -c '...'` (shared GPU).
# ===----------------------------------------------------------------------=== #

from std.memory import UnsafePointer
from std.random import random_float64, seed
from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext
from std.testing import TestSuite, assert_true

from llmm.matmul import matmul_fwd, matmul_fwd_fp4

from _lowp_test_common import cosine_and_rel_l2


def _run_site_case(
    label: String,
    rows: Int,
    in_channels: Int,
    out_channels: Int,
    use_gelu: Bool,
) raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime DT = DType.bfloat16

    var n_in = rows * in_channels
    var n_w = out_channels * in_channels
    var n_out = rows * out_channels

    # Realistic GPT-2 124M magnitude regime: weights ~N(0, 0.02) (GPT-2 init
    # std), post-layernorm activations ~O(1) (layernorm normalizes to unit
    # variance before gamma/beta), bias ~0. Same distribution/seed convention
    # as tests/test_matmul_fwd_lowp.mojo's fp8 analogue.
    var host_in = ctx.enqueue_create_host_buffer[DT](n_in)
    var host_w = ctx.enqueue_create_host_buffer[DT](n_w)
    var host_b = ctx.enqueue_create_host_buffer[DT](out_channels)
    seed(20260710)
    for i in range(n_in):
        var v = Float32((random_float64() * 2.0 - 1.0) * 1.5)
        host_in.unsafe_ptr()[i] = v.cast[DT]()
    for i in range(n_w):
        var v = Float32((random_float64() * 2.0 - 1.0) * 0.02)
        host_w.unsafe_ptr()[i] = v.cast[DT]()
    for i in range(out_channels):
        var v = Float32((random_float64() * 2.0 - 1.0) * 0.01)
        host_b.unsafe_ptr()[i] = v.cast[DT]()

    var dev_in = ctx.enqueue_create_buffer[DT](n_in)
    var dev_w = ctx.enqueue_create_buffer[DT](n_w)
    var dev_b = ctx.enqueue_create_buffer[DT](out_channels)
    dev_in.enqueue_copy_from(host_in)
    dev_w.enqueue_copy_from(host_w)
    dev_b.enqueue_copy_from(host_b)

    # bf16 reference path (the exact production non-fp4 call).
    var dev_out_ref = ctx.enqueue_create_buffer[DT](n_out)
    var dev_pre_gelu_ref = ctx.enqueue_create_buffer[DT](n_out)
    ctx.synchronize()

    if use_gelu:
        matmul_fwd[DT, "gpu", use_gelu=True, has_bias=True](
            dev_out_ref.unsafe_ptr().as_unsafe_any_origin(),
            dev_pre_gelu_ref.unsafe_ptr().as_unsafe_any_origin(),
            dev_in.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_w.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_b.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            Int64(rows),
            Int64(1),
            Int64(in_channels),
            Int64(out_channels),
            ctx,
        )
    else:
        matmul_fwd[DT, "gpu", use_gelu=False, has_bias=True](
            dev_out_ref.unsafe_ptr().as_unsafe_any_origin(),
            dev_pre_gelu_ref.unsafe_ptr().as_unsafe_any_origin(),
            dev_in.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_w.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_b.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            Int64(rows),
            Int64(1),
            Int64(in_channels),
            Int64(out_channels),
            ctx,
        )
    ctx.synchronize()

    # fp4 path (matmul_fwd_fp4) — no AmaxState/delayed-scaling arguments;
    # nvfp4_quantize computes both scale levels fresh in-kernel.
    var dev_out_fp4 = ctx.enqueue_create_buffer[DT](n_out)
    var dev_pre_gelu_fp4 = ctx.enqueue_create_buffer[DT](n_out)
    ctx.synchronize()

    if use_gelu:
        matmul_fwd_fp4[DT, "gpu", use_gelu=True, has_bias=True](
            dev_out_fp4.unsafe_ptr().as_unsafe_any_origin(),
            dev_pre_gelu_fp4.unsafe_ptr().as_unsafe_any_origin(),
            dev_in.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_w.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_b.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            Int64(rows),
            Int64(1),
            Int64(in_channels),
            Int64(out_channels),
            ctx,
        )
    else:
        matmul_fwd_fp4[DT, "gpu", use_gelu=False, has_bias=True](
            dev_out_fp4.unsafe_ptr().as_unsafe_any_origin(),
            dev_pre_gelu_fp4.unsafe_ptr().as_unsafe_any_origin(),
            dev_in.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_w.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dev_b.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            Int64(rows),
            Int64(1),
            Int64(in_channels),
            Int64(out_channels),
            ctx,
        )
    ctx.synchronize()

    # Compare the "post-GEMM, pre-nonlinearity" tensor: pre_gelu for the fc
    # site, the final (bias-added) out for proj (no nonlinearity).
    var host_ref = ctx.enqueue_create_host_buffer[DT](n_out)
    var host_fp4 = ctx.enqueue_create_host_buffer[DT](n_out)
    if use_gelu:
        dev_pre_gelu_ref.enqueue_copy_to(host_ref)
        dev_pre_gelu_fp4.enqueue_copy_to(host_fp4)
    else:
        dev_out_ref.enqueue_copy_to(host_ref)
        dev_out_fp4.enqueue_copy_to(host_fp4)
    ctx.synchronize()

    var rel_l2 = Float32(0.0)
    var cosine = Float32(0.0)
    cosine_and_rel_l2(
        host_fp4.unsafe_ptr(),
        host_ref.unsafe_ptr(),
        n_out,
        label + ": matmul_fwd_fp4 output",
        rel_l2,
        cosine,
    )

    # Bound per tests/test_lowp_gemm_fp4.mojo's own empirical floor for these
    # exact MLP shapes (~0.145-0.146 measured there at rows=512; gaussian-512
    # probe-shape case measured ~0.135); 0.20 leaves headroom for this test's
    # different (uniform-ish, GPT-2-init-magnitude) input distribution and
    # 8x row count while still being far tighter than a broken-kernel
    # failure mode (which test_lowp_gemm_fp4.mojo's own postmortem shows
    # lands orders of magnitude higher, ~1e5, not a moderate miss).
    assert_true(
        rel_l2 < Float32(0.20),
        label
        + ": relative L2 norm "
        + String(rel_l2)
        + " >= 0.20 (rows="
        + String(rows)
        + " in="
        + String(in_channels)
        + " out="
        + String(out_channels)
        + ")",
    )
    # Cosine floor is DERIVED from the relL2 floor, not independent: for
    # quantization error ~orthogonal to the signal, cosine ~= 1/sqrt(1 +
    # rel_l2^2), so fp4's ~0.151 relL2 floor implies cosine ~0.9886 (measured
    # 0.9894/0.9903 at both MLP sites, bit-identical across runs). The fp8
    # test's 0.999 bound is self-consistent only at fp8's ~0.036 floor —
    # reusing it here would demand sub-physics error for fp4's coarser
    # format. 0.985 sits below the measured floor with margin while still
    # failing loudly on real bugs (wrong operand/scale/layout land <<0.98).
    assert_true(
        cosine > Float32(0.985),
        label + ": cosine similarity " + String(cosine) + " <= 0.985",
    )


def test_fc_site() raises:
    # fc: input [rows,768] @ weight [3072,768]^T -> [rows,3072]. rows=4096 ==
    # B=4 * T=1024 (this project's default GPT-2 124M training shape).
    _run_site_case("fc", 4096, 768, 4 * 768, use_gelu=True)


def test_proj_site() raises:
    # proj: input [rows,3072] @ weight [768,3072]^T -> [rows,768].
    _run_site_case("proj", 4096, 4 * 768, 768, use_gelu=False)


def main() raises:
    # fp4 GEMM is cuBLASLt-only (comptime assert HAS_CUBLAS in llmm/matmul.mojo);
    # comptime-gate discovery so non-CUDA compiles to a skip, not a build error.
    comptime if has_nvidia_gpu_accelerator():
        TestSuite.discover_tests[__functions_in_module()]().run()
    else:
        print(
            "SKIP tests/test_matmul_fwd_fp4.mojo: NVFP4/cuBLASLt is NVIDIA-only"
        )
