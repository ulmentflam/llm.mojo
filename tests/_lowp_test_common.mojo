# ===----------------------------------------------------------------------=== #
# Shared helpers for the HEAD-side low-precision test gates
# (tests/test_lowp_gemm.mojo, tests/test_lowp_bwd.mojo, tests/test_amax.mojo).
#
# Split into its own module for the same reason as
# tests/_rng_sr_gpu_kernels.mojo: `TestSuite.discover_tests[
# __functions_in_module()]()` reflects over *every* function defined in the
# importing module to find its `test_`-prefixed ones, and constructing that
# reflection tuple forces every listed function's signature to be resolved.
# Plain host-side helpers like these don't hit the GPU-kernel-on-host-target
# failure `_rng_sr_gpu_kernels.mojo` documents, but keeping shared,
# non-`test_`-prefixed helpers out of each test module's own
# `__functions_in_module()` tuple avoids them being (redundantly) treated as
# discoverable test candidates and keeps one canonical body instead of one
# per file. See docs/ai/dry_consolidation_audit_2026-07-10.md finding F5.
#
# TODO(post fp4-merge): tests/test_lowp_gemm_fp4.mojo and
# tests/test_nvfp4_quant.mojo (currently on the in-flight fp4 branch, not
# touched by this module's introduction) carry their own copies of this
# logic -- `test_lowp_gemm_fp4.mojo`'s `_host_gemm_ref` is exactly this
# module's `_host_gemm_ref[transpose_a=False, transpose_b=True]` (its own
# comment says "ported here rather than imported since it is file-local
# there"), and its `_rel_l2` is exactly this module's `rel_l2`. Once that
# branch merges, point both at this module instead of their local copies.
# ===----------------------------------------------------------------------=== #

from std.memory import UnsafePointer
from std.math import sqrt
from std.random import random_float64
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true


# ===----------------------------------------------------------------------=== #
# Host fp32 GEMM reference (tests/test_lowp_gemm.mojo lines 63-90, verbatim).
#
# Matches lowp_gemm's operand-orientation convention (matmul.mojo's
# `_matmul_cublaslt_fp8` module comment): `d[j*m+i] = sum_p a_role(i,p) *
# b_role(p,j)`, where a_role/b_role read `a_host`/`b_host` according to
# `transpose_a`/`transpose_b` exactly as `lowp_gemm` does (transpose_a=False:
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
# rel_l2 (from tests/test_lowp_gemm_fp4.mojo's `_rel_l2`, verbatim, incl. its
# Float64 accumulation/return -- kept as-is for a drop-in future import per
# the TODO above; not currently called by the three HEAD-side test files).
# ===----------------------------------------------------------------------=== #


@always_inline
def rel_l2(
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


# ===----------------------------------------------------------------------=== #
# cosine_and_rel_l2 (from tests/test_lowp_bwd.mojo's `_cosine_and_rel_l2`,
# verbatim). Used by test_lowp_bwd.mojo.
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


# ===----------------------------------------------------------------------=== #
# bf16 fill/random helpers (from tests/test_lowp_bwd.mojo, verbatim: used by
# test_lowp_bwd.mojo) plus tests/test_amax.mojo's explicit-value bf16
# builder (verbatim, used by test_amax.mojo). Kept as distinct functions --
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
