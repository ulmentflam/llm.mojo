# ===----------------------------------------------------------------------=== #
# tests/test_lowp_gemm_fp4.mojo — FP4-GEMM chunk gate
# (docs/ai/fp4_readiness_summary.md, docs/ai/fp4_training_recipes_research.md):
#
#   1. `lowp_gemm_fp4` (llmm/matmul.mojo) vs a plain fp32 reference GEMM on
#      gaussian data, matching tests/probe_fp4/RESULTS.md's own methodology
#      (M=N=K=512, target rel L2 ~0.1445) — plus a `bf16_control_gemm` arm
#      through the identical cuBLASLt call path/D-readback convention, which
#      must land at ~bf16-rounding-only error (~0.003): the probe's own
#      "comparison-harness bug" lesson (RESULTS.md, "First-pass version of
#      this probe had a column-major-vs-row-major output indexing bug...").
#   2. Real GPT-2 MLP shapes (768x3072, 3072x768) at a representative row
#      count, weight operand quantized with 2D 16x16 block scaling
#      (`b_block_rows=16`) per the recipe.
#   3. Ill-scaled inputs (tiny/huge magnitude) — no NaN/Inf, no collapse.
#   4. RHT (llmm/hadamard.mojo) composed with quantize+GEMM: verifies the
#      documented `(H@a)^T @ (H@b) == 16 * a^T @ b` contract survives FP4
#      quantization noise.
#   5. Stochastic-rounding bracketing (host, no GPU) + determinism under a
#      fixed seed (GPU, `nvfp4_quantize` run twice byte-for-byte).
#
# GPU-only tests are guarded by `has_nvidia_gpu_accelerator()` and expected to
# run under `flock -w 10800 /tmp/llmm-gpu.lock -c '...'` (shared GPU).
# ===----------------------------------------------------------------------=== #

from std.math import sqrt
from std.memory import UnsafePointer
from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext
from std.testing import TestSuite, assert_true, assert_equal

from llmm.rand import MT19937
from llmm.memory import MutKernelPtr, ImmutKernelPtr, MutMemPtr
from llmm.hadamard import hadamard16_fwd_gpu
from llmm.nvfp4_quant import (
    encode_e2m1,
    decode_e2m1,
    ROUND_MODE_STOCHASTIC,
    nvfp4_packed_size,
    nvfp4_scale_buffer_size,
    nvfp4_quantize,
)
from llmm.matmul import lowp_gemm_fp4, bf16_control_gemm


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def _pseudo_gaussian_fill(
    mut rng: MT19937,
    data: MutMemPtr[DType.float32],
    numel: Int,
    std: Float32,
) -> None:
    """Irwin-Hall approximate-normal fill (see
    tests/test_nvfp4_quant.mojo's identical helper for the derivation/link
    dependency rationale: avoids `llmm.rand.normal_`'s Box-Muller, which
    needs `-lm`, not passed by the generic `make test-mojo` loop).
    """
    for i in range(numel):
        var s = Float32(0.0)
        for _ in range(12):
            s += rng.randfloat32()
        data[i] = (s - 6.0) * std


# Host fp32 GEMM reference, forward orientation only (matches lowp_gemm_fp4's
# scope): `a_host` is `[m,k]` row-major, `b_host` is `[n,k]` row-major (the
# "weight" role, e.g. `[OC,C]`), `out_host[j*m+i] = sum_p a[i,p]*b[j,p]` —
# same col-major-D flat-indexing convention `_matmul_cublaslt_fp4`/
# `bf16_control_gemm` write, so no row/col un-transposition is needed when
# comparing (tests/test_lowp_gemm.mojo's `_host_gemm_ref` uses the identical
# convention; ported here rather than imported since it is file-local there).
def _host_gemm_ref(
    a_host: UnsafePointer[Float32, MutUntrackedOrigin],
    b_host: UnsafePointer[Float32, MutUntrackedOrigin],
    out_host: UnsafePointer[Float32, MutUntrackedOrigin],
    m: Int,
    n: Int,
    k: Int,
) -> None:
    for i in range(m):
        for j in range(n):
            var acc = Float32(0.0)
            for p in range(k):
                acc += a_host[i * k + p] * b_host[j * k + p]
            out_host[j * m + i] = acc


@always_inline
def _rel_l2(
    got: UnsafePointer[Float32, MutUntrackedOrigin],
    want: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
) -> Float64:
    var sq_err = Float64(0.0)
    var sq_want = Float64(0.0)
    for i in range(n):
        var e = Float64(got[i]) - Float64(want[i])
        sq_err += e * e
        sq_want += Float64(want[i]) * Float64(want[i])
    return sqrt(sq_err / (sq_want if sq_want > 0.0 else 1.0))


@always_inline
def _assert_finite(
    d: UnsafePointer[Float32, MutUntrackedOrigin], n: Int, label: String
) raises:
    for i in range(n):
        var v = d[i]
        assert_true(v == v, label + ": NaN at " + String(i))
        assert_true(
            v > Float32(-1e30) and v < Float32(1e30),
            label + ": Inf/overflow at " + String(i),
        )


# ===----------------------------------------------------------------------=== #
# 1. lowp_gemm_fp4 vs fp32 reference (gaussian, probe-matching shape) +
#    bf16 control arm through the identical harness.
# ===----------------------------------------------------------------------=== #


def _run_fp4_gemm_case[
    a_block_rows: Int = 1,
    b_block_rows: Int = 1,
](
    m: Int,
    n: Int,
    k: Int,
    std: Float32,
    label: String,
    max_rel_l2: Float64,
    max_bf16_rel_l2: Float64 = 0.02,
) raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime IN_DT = DType.bfloat16

    var a_n = m * k
    var b_n = n * k

    var host_a_f32 = ctx.enqueue_create_host_buffer[DType.float32](a_n)
    var host_b_f32 = ctx.enqueue_create_host_buffer[DType.float32](b_n)
    var rng = MT19937(UInt32(2026))
    _pseudo_gaussian_fill(rng, host_a_f32.unsafe_ptr(), a_n, std)
    _pseudo_gaussian_fill(rng, host_b_f32.unsafe_ptr(), b_n, std)
    ctx.synchronize()

    # bf16-round the inputs once, host-side, so the fp32 reference is against
    # the SAME bf16-rounded data every GEMM (fp4 and bf16-control) actually
    # consumes -- otherwise bf16 rounding of the inputs themselves would
    # inflate the "bf16 control" error above pure GEMM-precision noise.
    var host_a_bf16 = ctx.enqueue_create_host_buffer[IN_DT](a_n)
    var host_b_bf16 = ctx.enqueue_create_host_buffer[IN_DT](b_n)
    var host_a_ref = ctx.enqueue_create_host_buffer[DType.float32](a_n)
    var host_b_ref = ctx.enqueue_create_host_buffer[DType.float32](b_n)
    for i in range(a_n):
        var bf = host_a_f32.unsafe_ptr()[i].cast[IN_DT]()
        host_a_bf16.unsafe_ptr()[i] = bf
        host_a_ref.unsafe_ptr()[i] = bf.cast[DType.float32]()
    for i in range(b_n):
        var bf = host_b_f32.unsafe_ptr()[i].cast[IN_DT]()
        host_b_bf16.unsafe_ptr()[i] = bf
        host_b_ref.unsafe_ptr()[i] = bf.cast[DType.float32]()

    var dev_a = ctx.enqueue_create_buffer[IN_DT](a_n)
    var dev_b = ctx.enqueue_create_buffer[IN_DT](b_n)
    dev_a.enqueue_copy_from(host_a_bf16)
    dev_b.enqueue_copy_from(host_b_bf16)

    var a_q = ctx.enqueue_create_buffer[DType.uint8](nvfp4_packed_size(m, k))
    var a_scale = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_scale_buffer_size(m, k, a_block_rows)
    )
    var a_tscale = ctx.enqueue_create_buffer[DType.float32](1)
    var b_q = ctx.enqueue_create_buffer[DType.uint8](nvfp4_packed_size(n, k))
    var b_scale = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_scale_buffer_size(n, k, b_block_rows)
    )
    var b_tscale = ctx.enqueue_create_buffer[DType.float32](1)
    var dev_d = ctx.enqueue_create_buffer[IN_DT](m * n)
    var dev_d_bf16 = ctx.enqueue_create_buffer[IN_DT](m * n)
    ctx.synchronize()

    lowp_gemm_fp4[IN_DT, IN_DT, "gpu", a_block_rows, b_block_rows](
        rebind[MutKernelPtr[IN_DT]](dev_d.unsafe_ptr().as_unsafe_any_origin()),
        rebind[ImmutKernelPtr[IN_DT]](
            dev_a.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[IN_DT]](
            dev_b.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            a_q.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            a_scale.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            a_tscale.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            b_q.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            b_scale.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            b_tscale.unsafe_ptr().as_unsafe_any_origin()
        ),
        m,
        n,
        k,
        False,
        ctx,
    )

    bf16_control_gemm["gpu"](
        rebind[MutKernelPtr[DType.bfloat16]](
            dev_d_bf16.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DType.bfloat16]](
            dev_a.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DType.bfloat16]](
            dev_b.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        m,
        n,
        k,
        ctx,
    )
    ctx.synchronize()

    var host_d = ctx.enqueue_create_host_buffer[IN_DT](m * n)
    var host_d_bf16 = ctx.enqueue_create_host_buffer[IN_DT](m * n)
    dev_d.enqueue_copy_to(host_d)
    dev_d_bf16.enqueue_copy_to(host_d_bf16)
    ctx.synchronize()

    var host_got = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    var host_got_bf16 = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    for i in range(m * n):
        host_got.unsafe_ptr()[i] = host_d.unsafe_ptr()[i].cast[DType.float32]()
        host_got_bf16.unsafe_ptr()[i] = host_d_bf16.unsafe_ptr()[i].cast[
            DType.float32
        ]()

    var host_ref = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    _host_gemm_ref(
        host_a_ref.unsafe_ptr(),
        host_b_ref.unsafe_ptr(),
        host_ref.unsafe_ptr(),
        m,
        n,
        k,
    )

    _assert_finite(host_got.unsafe_ptr(), m * n, label + " (fp4)")
    _assert_finite(host_got_bf16.unsafe_ptr(), m * n, label + " (bf16)")

    var rel_l2 = _rel_l2(host_got.unsafe_ptr(), host_ref.unsafe_ptr(), m * n)
    var rel_l2_bf16 = _rel_l2(
        host_got_bf16.unsafe_ptr(), host_ref.unsafe_ptr(), m * n
    )
    print(label, "fp4 rel_l2 =", rel_l2, " bf16-control rel_l2 =", rel_l2_bf16)

    assert_true(
        rel_l2 < max_rel_l2,
        label + ": fp4 rel_l2 " + String(rel_l2) + " >= " + String(max_rel_l2),
    )
    assert_true(
        rel_l2_bf16 < max_bf16_rel_l2,
        label
        + ": bf16-control rel_l2 "
        + String(rel_l2_bf16)
        + " >= "
        + String(max_bf16_rel_l2)
        + " -- if this fails while fp4's rel_l2 also looks wrong, suspect"
        " the comparison harness (D layout/readback), not the fp4 kernel"
        " (see tests/probe_fp4/RESULTS.md's own postmortem)",
    )


def test_fp4_gemm_gaussian_probe_shape() raises:
    # Matches tests/probe_fp4/RESULTS.md's M=N=K=512 setup as closely as a
    # bf16-input (vs the probe's raw fp32) harness allows. std chosen so
    # magnitudes land in a similar dynamic-range regime to the probe's
    # uniform[-3,3] (std=1.7 gaussian has comparable spread); target rel_l2
    # a bit looser than the probe's exact 0.1445 to allow for the
    # gaussian-vs-uniform distribution difference and bf16 (vs fp32) input
    # rounding, while still being tight enough to catch a broken kernel.
    _run_fp4_gemm_case(
        512, 512, 512, Float32(1.7), "gaussian-512", max_rel_l2=0.20
    )


def test_fp4_gemm_ill_scaled_tiny() raises:
    # Tiny magnitudes (~2e-4) -- would flush toward e2m1's zero/subnormal
    # grid points without the per-tensor fp32 scale compensating. Proves the
    # scale mechanism (no NaN/Inf/collapse), looser bound (mirrors
    # tests/test_lowp_gemm.mojo's fp8 ill-scaled-tiny case). Shape matches
    # the probe's own 512^3 (tests/probe_fp4/RESULTS.md's confirmed-dispatch
    # size, `cutlass...128x128x256...vs16`) rather than a smaller ad hoc
    # shape, since cuBLASLt's NVFP4 block-scaled kernels are tuned for that
    # tile and a much smaller M/N/K risks `cublasLtMatmulAlgoGetHeuristic`
    # returning zero candidates for reasons unrelated to this test's actual
    # purpose (scale-mechanism correctness, not kernel-dispatch coverage).
    _run_fp4_gemm_case(
        512,
        512,
        512,
        Float32(2.0e-4),
        "ill-scaled-tiny",
        max_rel_l2=0.30,
        max_bf16_rel_l2=0.05,
    )


def test_fp4_gemm_ill_scaled_huge() raises:
    # Huge magnitudes (~5e3) -- would saturate e2m1 without a compensating
    # (small) tensor scale. Same probe-matching shape rationale as above.
    _run_fp4_gemm_case(
        512, 512, 512, Float32(5.0e3), "ill-scaled-huge", max_rel_l2=0.20
    )


# ===----------------------------------------------------------------------=== #
# 2. Real GPT-2 MLP shapes, weight operand 2D (16x16) block-scaled.
# ===----------------------------------------------------------------------=== #


def test_fp4_gemm_mlp_fc_up_shape() raises:
    # fc_up: input [rows, 768] @ weight [3072, 768]^T -> [rows, 3072].
    # rows=512 (not the full 4096 of a real B*T) keeps the O(m*n*k) host fp32
    # reference tractable in a unit test; the channel/contraction dims (768,
    # 3072 -- 48 and 192 sixteen-element blocks respectively) are the real
    # GPT-2 124M MLP shapes and are what exercises realistic block-scaling
    # statistics. b_block_rows=16 -> weight operand uses 2D 16x16 scaling
    # per the recipe.
    _run_fp4_gemm_case[b_block_rows=16](
        512,
        3072,
        768,
        Float32(0.05),
        "mlp-fc-up-768x3072",
        max_rel_l2=0.20,
    )


def test_fp4_gemm_mlp_fc_down_shape() raises:
    # fc_down: input [rows, 3072] @ weight [768, 3072]^T -> [rows, 768].
    _run_fp4_gemm_case[b_block_rows=16](
        512,
        768,
        3072,
        Float32(0.05),
        "mlp-fc-down-3072x768",
        max_rel_l2=0.20,
    )


# ===----------------------------------------------------------------------=== #
# 3. RHT composed with quantize+GEMM: (H@a)^T @ (H@b) == 16 * a^T @ b.
# ===----------------------------------------------------------------------=== #


def test_fp4_rht_quantize_gemm_contract() raises:
    # llmm/hadamard.mojo's forward RHT (`y = H16 @ (s ⊙ x)`) is deliberately
    # NOT its own transpose in the signed sense -- but H16 is symmetric and
    # H16 @ H16 == 16*I, so applying the SAME forward transform to both GEMM
    # operands along the contracted (k) dimension scales the exact-precision
    # GEMM result by a constant 16 (module docstring's math derivation).
    # This test verifies that identity survives FP4 quantization: RHT(a) and
    # RHT(b) quantized+GEMM'd should match `16 * fp32_ref(a, b)` within
    # ordinary FP4 quantization noise, not some unrelated result -- proving
    # the transform composes correctly with the quantize/GEMM pipeline
    # rather than merely round-tripping in isolation.
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime IN_DT = DType.bfloat16
    # Shape matches the probe's own 512^3 (tests/probe_fp4/RESULTS.md) --
    # see test_fp4_gemm_ill_scaled_tiny's comment for why a much smaller
    # shape risks a dispatch-heuristic false negative unrelated to this
    # test's actual purpose. 512 is also a multiple of both NVFP4_BLOCK and
    # HADAMARD_BLOCK (16).
    var m = 512
    var n = 512
    var k = 512

    var host_a_f32 = ctx.enqueue_create_host_buffer[DType.float32](m * k)
    var host_b_f32 = ctx.enqueue_create_host_buffer[DType.float32](n * k)
    var rng = MT19937(UInt32(4242))
    _pseudo_gaussian_fill(rng, host_a_f32.unsafe_ptr(), m * k, Float32(1.0))
    _pseudo_gaussian_fill(rng, host_b_f32.unsafe_ptr(), n * k, Float32(1.0))
    ctx.synchronize()

    var host_a_bf16 = ctx.enqueue_create_host_buffer[IN_DT](m * k)
    var host_b_bf16 = ctx.enqueue_create_host_buffer[IN_DT](n * k)
    var host_a_ref = ctx.enqueue_create_host_buffer[DType.float32](m * k)
    var host_b_ref = ctx.enqueue_create_host_buffer[DType.float32](n * k)
    for i in range(m * k):
        var bf = host_a_f32.unsafe_ptr()[i].cast[IN_DT]()
        host_a_bf16.unsafe_ptr()[i] = bf
        host_a_ref.unsafe_ptr()[i] = bf.cast[DType.float32]()
    for i in range(n * k):
        var bf = host_b_f32.unsafe_ptr()[i].cast[IN_DT]()
        host_b_bf16.unsafe_ptr()[i] = bf
        host_b_ref.unsafe_ptr()[i] = bf.cast[DType.float32]()

    var dev_a = ctx.enqueue_create_buffer[IN_DT](m * k)
    var dev_b = ctx.enqueue_create_buffer[IN_DT](n * k)
    dev_a.enqueue_copy_from(host_a_bf16)
    dev_b.enqueue_copy_from(host_b_bf16)
    var dev_a_rht = ctx.enqueue_create_buffer[IN_DT](m * k)
    var dev_b_rht = ctx.enqueue_create_buffer[IN_DT](n * k)
    ctx.synchronize()

    hadamard16_fwd_gpu[IN_DT, "gpu"](
        rebind[MutKernelPtr[IN_DT]](
            dev_a_rht.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[IN_DT]](
            dev_a.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        m,
        k,
        ctx,
    )
    hadamard16_fwd_gpu[IN_DT, "gpu"](
        rebind[MutKernelPtr[IN_DT]](
            dev_b_rht.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[IN_DT]](
            dev_b.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        n,
        k,
        ctx,
    )
    ctx.synchronize()

    var a_q = ctx.enqueue_create_buffer[DType.uint8](nvfp4_packed_size(m, k))
    var a_scale = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_scale_buffer_size(m, k, 1)
    )
    var a_tscale = ctx.enqueue_create_buffer[DType.float32](1)
    var b_q = ctx.enqueue_create_buffer[DType.uint8](nvfp4_packed_size(n, k))
    var b_scale = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_scale_buffer_size(n, k, 1)
    )
    var b_tscale = ctx.enqueue_create_buffer[DType.float32](1)
    var dev_d = ctx.enqueue_create_buffer[IN_DT](m * n)
    ctx.synchronize()

    lowp_gemm_fp4[IN_DT, IN_DT, "gpu", 1, 1](
        rebind[MutKernelPtr[IN_DT]](dev_d.unsafe_ptr().as_unsafe_any_origin()),
        rebind[ImmutKernelPtr[IN_DT]](
            dev_a_rht.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[IN_DT]](
            dev_b_rht.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            a_q.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            a_scale.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            a_tscale.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            b_q.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.uint8]](
            b_scale.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            b_tscale.unsafe_ptr().as_unsafe_any_origin()
        ),
        m,
        n,
        k,
        False,
        ctx,
    )
    ctx.synchronize()

    var host_d = ctx.enqueue_create_host_buffer[IN_DT](m * n)
    dev_d.enqueue_copy_to(host_d)
    ctx.synchronize()

    var host_got = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    for i in range(m * n):
        host_got.unsafe_ptr()[i] = host_d.unsafe_ptr()[i].cast[DType.float32]()

    var host_ref = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    _host_gemm_ref(
        host_a_ref.unsafe_ptr(),
        host_b_ref.unsafe_ptr(),
        host_ref.unsafe_ptr(),
        m,
        n,
        k,
    )
    var host_ref_scaled = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    for i in range(m * n):
        host_ref_scaled.unsafe_ptr()[i] = host_ref.unsafe_ptr()[i] * 16.0

    _assert_finite(host_got.unsafe_ptr(), m * n, "rht-quantize-gemm")
    var rel_l2 = _rel_l2(
        host_got.unsafe_ptr(), host_ref_scaled.unsafe_ptr(), m * n
    )
    print("rht-quantize-gemm rel_l2 (vs 16*fp32 ref) =", rel_l2)
    assert_true(
        rel_l2 < 0.25,
        "RHT+quantize+GEMM rel_l2 vs 16*fp32 ref = "
        + String(rel_l2)
        + " -- the RHT-composition contract ((H@a)^T@(H@b) == 16*a^T@b)"
        " should hold within ordinary FP4 quantization noise",
    )


# ===----------------------------------------------------------------------=== #
# 4. Stochastic-rounding bracketing (host) + determinism (GPU).
# ===----------------------------------------------------------------------=== #


def test_sr_e2m1_bracket_and_unbiasedness() raises:
    # x=0.6 lies in the [0.5, 1.0) bracket (grid indices 1, 2); p_up =
    # (0.6-0.5)/0.5 = 0.2. A rand comfortably below p_up must round UP
    # (idx=2, decodes to 1.0); a rand comfortably above must round DOWN
    # (idx=1, decodes to 0.5) -- i.e. SR always produces one of the two
    # bracketing grid values, never anything else.
    var x = Float32(0.6)
    var up = encode_e2m1[ROUND_MODE_STOCHASTIC](x, Float32(0.05))
    var down = encode_e2m1[ROUND_MODE_STOCHASTIC](x, Float32(0.95))
    assert_equal(decode_e2m1(up), Float32(1.0))
    assert_equal(decode_e2m1(down), Float32(0.5))

    # Exact grid points are deterministic under SR too (p_up == 0 exactly at
    # the lower endpoint of its bracket), regardless of `rand`.
    var exact = Float32(2.0)
    for i in range(5):
        var r = Float32(i) * 0.2499
        assert_equal(
            decode_e2m1(encode_e2m1[ROUND_MODE_STOCHASTIC](exact, r)),
            Float32(2.0),
        )

    # Unbiasedness: sweep `rand` densely over [0,1) (deterministic, not
    # actually random) and check the empirical mean of the decoded value
    # converges to x -- SR's whole point is E[decode(encode(x))] == x
    # exactly, by construction of the linear-interpolation probability.
    var xs: List[Float32] = [0.1, 0.6, 1.2, 1.9, 2.7, 3.6, 5.0]
    comptime N = 2000
    for i in range(len(xs)):
        var xv = xs[i]
        var total = Float32(0.0)
        for j in range(N):
            var r = (Float32(j) + 0.5) / Float32(N)
            total += decode_e2m1(encode_e2m1[ROUND_MODE_STOCHASTIC](xv, r))
        var mean = total / Float32(N)
        var err = mean - xv
        var abs_err = err if err >= 0.0 else -err
        assert_true(
            abs_err < 0.01,
            "SR mean for x="
            + String(xv)
            + " was "
            + String(mean)
            + " (err "
            + String(abs_err)
            + ")",
        )


def test_sr_nvfp4_quantize_deterministic_under_fixed_seed() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    comptime IN_DT = DType.bfloat16
    var rows = 8
    var k = 32
    var n = rows * k

    var host_f32 = ctx.enqueue_create_host_buffer[DType.float32](n)
    var rng = MT19937(UInt32(99))
    _pseudo_gaussian_fill(rng, host_f32.unsafe_ptr(), n, Float32(1.0))
    ctx.synchronize()
    var host_bf16 = ctx.enqueue_create_host_buffer[IN_DT](n)
    for i in range(n):
        host_bf16.unsafe_ptr()[i] = host_f32.unsafe_ptr()[i].cast[IN_DT]()
    var x_dev = ctx.enqueue_create_buffer[IN_DT](n)
    x_dev.enqueue_copy_from(host_bf16)
    ctx.synchronize()

    var q_size = nvfp4_packed_size(rows, k)
    var scale_size = nvfp4_scale_buffer_size(rows, k, 1)

    @parameter
    def _run(mut qs: List[UInt8], mut ss: List[UInt8]) raises -> None:
        var q_dev = ctx.enqueue_create_buffer[DType.uint8](q_size)
        var scale_dev = ctx.enqueue_create_buffer[DType.uint8](scale_size)
        var tscale_dev = ctx.enqueue_create_buffer[DType.float32](1)
        ctx.synchronize()
        nvfp4_quantize[IN_DT, "gpu", 1, ROUND_MODE_STOCHASTIC](
            rebind[MutKernelPtr[DType.uint8]](
                q_dev.unsafe_ptr().as_unsafe_any_origin()
            ),
            rebind[MutKernelPtr[DType.uint8]](
                scale_dev.unsafe_ptr().as_unsafe_any_origin()
            ),
            rebind[MutKernelPtr[DType.float32]](
                tscale_dev.unsafe_ptr().as_unsafe_any_origin()
            ),
            rebind[ImmutKernelPtr[IN_DT]](
                x_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin()
            ),
            rows,
            k,
            ctx,
        )
        ctx.synchronize()
        var host_q = ctx.enqueue_create_host_buffer[DType.uint8](q_size)
        var host_scale = ctx.enqueue_create_host_buffer[DType.uint8](scale_size)
        q_dev.enqueue_copy_to(host_q)
        scale_dev.enqueue_copy_to(host_scale)
        ctx.synchronize()
        for i in range(q_size):
            qs.append(host_q.unsafe_ptr()[i])
        for i in range(scale_size):
            ss.append(host_scale.unsafe_ptr()[i])

    var q1 = List[UInt8](capacity=q_size)
    var s1 = List[UInt8](capacity=scale_size)
    _run(q1, s1)
    var q2 = List[UInt8](capacity=q_size)
    var s2 = List[UInt8](capacity=scale_size)
    _run(q2, s2)

    assert_equal(len(q1), len(q2))
    assert_equal(len(s1), len(s2))
    var mismatches = 0
    for i in range(len(q1)):
        if q1[i] != q2[i]:
            mismatches += 1
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            mismatches += 1
    assert_true(
        mismatches == 0,
        "nvfp4_quantize with ROUND_MODE_STOCHASTIC and a fixed"
        " (seed, stream, step) was not deterministic: "
        + String(mismatches)
        + " byte mismatches across two identical runs",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
