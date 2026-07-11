# ===----------------------------------------------------------------------=== #
# Shared helpers for the low-precision test gates (tests/test_lowp_gemm.mojo,
# tests/test_lowp_bwd.mojo, tests/test_amax.mojo, tests/test_lowp_gemm_fp4.mojo,
# tests/test_matmul_bwd_fp4.mojo, tests/test_matmul_fwd_lowp.mojo,
# tests/test_matmul_fwd_fp4.mojo, tests/test_nvfp4_quant.mojo).
#
# Split into its own module because `TestSuite.discover_tests[
# __functions_in_module()]()` reflects over *every* function defined in the
# importing module to find its `test_`-prefixed ones, and constructing that
# reflection tuple forces every listed function's signature to be resolved.
# Plain host-side helpers like these don't hit the GPU-kernel-on-host-target
# failure tests/_rng_sr_gpu_kernels.mojo documents, but keeping shared,
# non-`test_`-prefixed helpers out of each test module's own
# `__functions_in_module()` tuple avoids them being (redundantly) treated as
# discoverable test candidates and keeps one canonical body instead of one
# per file.
# ===----------------------------------------------------------------------=== #

from std.memory import UnsafePointer
from std.math import sqrt
from std.random import random_float64
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true

from llmm.rand import MT19937


# ===----------------------------------------------------------------------=== #
# Host fp32 GEMM reference.
#
# Matches lowp_gemm_devscale's operand-orientation convention (matmul.mojo's
# `_matmul_cublaslt_fp8` module comment): `d[j*m+i] = sum_p a_role(i,p) *
# b_role(p,j)`, where a_role/b_role read `a_host`/`b_host` according to
# `transpose_a`/`transpose_b` exactly as `lowp_gemm_devscale` does
# (transpose_a=False:
# a_host is `[m,k]` row-major; True: `[k,m]` row-major. transpose_b=False:
# b_host is `[n,k]` row-major; True: `[k,n]` row-major).
# ===----------------------------------------------------------------------=== #


def _host_gemm_ref[
    transpose_a: Bool, transpose_b: Bool
](
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
                var av: Float32
                comptime if transpose_a:
                    av = a_host[p * m + i]
                else:
                    av = a_host[i * k + p]
                var bv: Float32
                comptime if transpose_b:
                    bv = b_host[p * n + j]
                else:
                    bv = b_host[j * k + p]
                acc += av * bv
            out_host[j * m + i] = acc


# ===----------------------------------------------------------------------=== #
# cosine_and_rel_l2 — two overloads, both computing aggregate metrics
# (relative L2 norm + cosine similarity) rather than a naive per-element
# `|got-want| / |want|`: GEMM outputs cross zero (sums of k signed
# products), and near those zero-crossings a tiny, entirely
# quantization-proportionate absolute error produces an arbitrarily large
# *relative* error against a near-zero `want` -- not a real correctness
# signal.
#
# The DeviceBuffer overload (from tests/test_lowp_bwd.mojo's
# `_cosine_and_rel_l2`) does its own GPU readback and asserts against fixed
# internal thresholds (rel_l2 < 0.1, cosine > 0.99) -- used by
# tests/test_lowp_bwd.mojo, tests/test_amax.mojo.
#
# The host-pointer overload below is dtype-generic over both operands (bf16
# or fp32 host arrays, matching whichever dtype a caller already has after
# its own readback/reference computation) and reports the metrics via
# out-params instead of asserting internally, so each call site keeps its
# own calibrated thresholds -- used by tests/test_lowp_gemm.mojo,
# tests/test_lowp_gemm_fp4.mojo, tests/test_matmul_bwd_fp4.mojo,
# tests/test_matmul_fwd_lowp.mojo, tests/test_matmul_fwd_fp4.mojo.
# ===----------------------------------------------------------------------=== #


def cosine_and_rel_l2(
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
    var rel_l2_val = sqrt(l2_err / (norm_want + Float32(1e-12)))
    var cosine = dot / (sqrt(norm_got) * sqrt(norm_want) + Float32(1e-12))
    assert_true(
        rel_l2_val < Float32(0.1),
        label + ": relative L2 " + String(rel_l2_val) + " >= 0.1",
    )
    assert_true(
        cosine > Float32(0.99),
        label + ": cosine similarity " + String(cosine) + " <= 0.99",
    )


def cosine_and_rel_l2[
    GotDT: DType = DType.float32,
    WantDT: DType = DType.float32,
](
    got: UnsafePointer[Scalar[GotDT], MutUntrackedOrigin],
    want: UnsafePointer[Scalar[WantDT], MutUntrackedOrigin],
    n: Int,
    label: String,
    mut rel_l2_out: Float32,
    mut cosine_out: Float32,
) raises -> None:
    """Host-pointer overload (see the module comment above). Asserts NaN/Inf
    on `got`; reports `rel_l2_out`/`cosine_out` for the CALLER to assert
    against its own thresholds.
    """
    var l2_err = Float32(0.0)
    var dot = Float32(0.0)
    var norm_got = Float32(0.0)
    var norm_want = Float32(0.0)
    for i in range(n):
        var g = got[i].cast[DType.float32]()
        var w = want[i].cast[DType.float32]()
        assert_true(g == g, label + ": NaN at " + String(i))
        assert_true(
            g > Float32(-1e30) and g < Float32(1e30),
            label + ": Inf/overflow at " + String(i),
        )
        var e = g - w
        l2_err += e * e
        dot += g * w
        norm_got += g * g
        norm_want += w * w
    rel_l2_out = sqrt(l2_err / (norm_want + Float32(1e-12)))
    cosine_out = dot / (sqrt(norm_got) * sqrt(norm_want) + Float32(1e-12))


# ===----------------------------------------------------------------------=== #
# pseudo_gaussian_fill — Irwin-Hall approximate-normal fill: sum of 12
# uniform(0,1) draws has mean 6, variance 1, so `(sum - 6)*std + mean`
# approximates N(mean, std^2). Deliberately avoids `llmm.rand.normal_`'s
# Box-Muller (sin/cos/log), which pulls in a libm link dependency (`-lm`)
# that the Makefile's generic `test-mojo` loop does not pass -- this keeps
# the test file linkable via the same plain `pixi run mojo run -I .`
# invocation as every other tests/test_*.mojo file. Dtype-generic (fp32 or
# bf16 output) so it serves both direct-fp32-input and direct-bf16-input
# callers.
# ===----------------------------------------------------------------------=== #


@always_inline
def pseudo_gaussian_fill[
    DT: DType = DType.float32,
](
    mut rng: MT19937,
    data: UnsafePointer[Scalar[DT], MutUntrackedOrigin],
    numel: Int,
    std: Float32,
    mean: Float32 = 0.0,
) -> None:
    for i in range(numel):
        var s = Float32(0.0)
        for _ in range(12):
            s += rng.randfloat32()
        data[i] = ((s - 6.0) * std + mean).cast[DT]()


# ===----------------------------------------------------------------------=== #
# bf16 fill/random helpers: used by test_lowp_bwd.mojo (random_bf16/
# zeros_bf16/clone_bf16) and test_amax.mojo (make_bf16_tensor, an
# explicit-value bf16 builder). Kept as distinct functions --
# `random_bf16`/`zeros_bf16` fill by count, `make_bf16_tensor` fills from an
# explicit `List[Float32]` -- rather than folding one into the other, to
# avoid changing either call site's behavior.
# ===----------------------------------------------------------------------=== #


def random_bf16(
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


def zeros_bf16(
    ctx: DeviceContext, n: Int
) raises -> DeviceBuffer[DType.bfloat16]:
    var host = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    for i in range(n):
        host.unsafe_ptr()[i] = Float32(0.0).cast[DType.bfloat16]()
    var dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    dev.enqueue_copy_from(host)
    ctx.synchronize()
    return dev


def clone_bf16(
    ctx: DeviceContext, src: DeviceBuffer[DType.bfloat16], n: Int
) raises -> DeviceBuffer[DType.bfloat16]:
    var host = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    src.enqueue_copy_to(host)
    ctx.synchronize()
    var dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    dev.enqueue_copy_from(host)
    ctx.synchronize()
    return dev


def make_bf16_tensor(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.bfloat16]:
    var n = len(values)
    var host = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    for i in range(n):
        host.unsafe_ptr()[i] = values[i].cast[DType.bfloat16]()
    var dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    dev.enqueue_copy_from(host)
    ctx.synchronize()
    return dev
