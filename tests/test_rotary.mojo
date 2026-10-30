# ===----------------------------------------------------------------------=== #
# Rotary q/k kernel (llmm/rotary.mojo), W8. Run with:  make test-mojo
#
# These checks are written to fail on the specific mistakes a rotary port makes
# (wrong pair layout, sign of the sine, rotating v, rotating the identity half,
# forward/inverse mismatch, off-by-one in the position index), not to be
# tolerance-shaped passes:
#   * a hand-computed planted value (angle 1 rad) that fixes the sign and
#     layout independently of the kernel's own formula,
#   * an independent float64 scalar re-implementation over a random qkv,
#     requiring v, the identity half of each head and position 0 to be BITWISE
#     unchanged,
#   * the adjoint identity <R x, y> == <x, R^T y> that makes `inverse=True` the
#     correct backward of the forward map,
#   * CPU target vs GPU target on the same input.
# ===----------------------------------------------------------------------=== #

from std.math import cos, sin
from std.testing import assert_true, TestSuite

from max.gpu.host import DeviceContext

from llmm.memory import MutMemPtr, as_mut_kernel, as_immut_kernel_from_mut
from llmm.rotary import rotary_qk_rows, rotary_table_host
from _gpu_test_common import shared_gpu_ctx

comptime B = 2
comptime T = 5
comptime TMAX = 8
comptime H = 3
comptime HD = 8  # n = 2 rotated pairs per head, pairs (i, i + 4)
comptime C = H * HD
comptime ROWS = B * T
comptime NQKV = ROWS * 3 * C


def _fill(x: List[Float32], seed: Int) -> List[Float32]:
    var out = x.copy()
    var s = UInt64(seed) * 2654435761 + 12345
    for i in range(len(out)):
        s = s * 6364136223846793005 + 1442695040888963407
        out[i] = Float32(Int((s >> 33) % 2001) - 1000) / 500.0
    return out^


def _zeros(n: Int) -> List[Float32]:
    return List[Float32](length=n, fill=Float32(0.0))


def _run_cpu[
    inverse: Bool
](qkv: List[Float32], tbl: List[Float32]) raises -> List[Float32]:
    var ctx = DeviceContext(api="cpu")
    var buf = ctx.enqueue_create_host_buffer[DType.float32](NQKV)
    var tb = ctx.enqueue_create_host_buffer[DType.float32](len(tbl))
    ctx.synchronize()
    for i in range(NQKV):
        buf[i] = qkv[i]
    for i in range(len(tbl)):
        tb[i] = tbl[i]
    rotary_qk_rows[DType.float32, "cpu", inverse=inverse](
        as_mut_kernel[DType.float32](
            MutMemPtr[DType.float32](unsafe_from_address=Int(buf.unsafe_ptr()))
        ),
        as_immut_kernel_from_mut[DType.float32](
            MutMemPtr[DType.float32](unsafe_from_address=Int(tb.unsafe_ptr()))
        ),
        B,
        T,
        TMAX,
        H,
        HD,
        ctx,
    )
    ctx.synchronize()
    var out = _zeros(NQKV)
    for i in range(NQKV):
        out[i] = buf[i]
    return out^


def _run_gpu[
    inverse: Bool
](qkv: List[Float32], tbl: List[Float32]) raises -> List[Float32]:
    var ctx = shared_gpu_ctx()
    var host = ctx.enqueue_create_host_buffer[DType.float32](NQKV)
    var htb = ctx.enqueue_create_host_buffer[DType.float32](len(tbl))
    var dev = ctx.enqueue_create_buffer[DType.float32](NQKV)
    var dtb = ctx.enqueue_create_buffer[DType.float32](len(tbl))
    ctx.synchronize()
    for i in range(NQKV):
        host[i] = qkv[i]
    for i in range(len(tbl)):
        htb[i] = tbl[i]
    ctx.enqueue_copy(dev, host)
    ctx.enqueue_copy(dtb, htb)
    ctx.synchronize()
    rotary_qk_rows[DType.float32, "gpu", inverse=inverse](
        as_mut_kernel[DType.float32](
            MutMemPtr[DType.float32](unsafe_from_address=Int(dev.unsafe_ptr()))
        ),
        as_immut_kernel_from_mut[DType.float32](
            MutMemPtr[DType.float32](unsafe_from_address=Int(dtb.unsafe_ptr()))
        ),
        B,
        T,
        TMAX,
        H,
        HD,
        ctx,
    )
    ctx.enqueue_copy(host, dev)
    ctx.synchronize()
    var out = _zeros(NQKV)
    for i in range(NQKV):
        out[i] = host[i]
    return out^


def _reference(x: List[Float32]) -> List[Float32]:
    """Independent float64 scalar rotary (no table, no shared code)."""
    var out = x.copy()
    for bt in range(ROWS):
        var t = bt % T
        for qk in range(2):
            for h in range(H):
                var base = bt * 3 * C + qk * C + h * HD
                for i in range(HD // 4):
                    var freq = Float64(1.0) / (
                        Float64(1024.0) ** (Float64(i) / Float64(HD // 4 - 1))
                    )
                    var ang = Float64(t) * freq
                    var x1 = Float64(x[base + i])
                    var x2 = Float64(x[base + i + HD // 2])
                    out[base + i] = Float32(x1 * cos(ang) + x2 * sin(ang))
                    out[base + i + HD // 2] = Float32(
                        x2 * cos(ang) - x1 * sin(ang)
                    )
    return out^


def test_planted_value_fixes_sign_and_layout() raises:
    # t = 1, head 0, pair i = 0 has frequency 1 (angle exactly 1 rad). Put
    # x1 = 1 at dim 0 and x2 = 0 at dim 4: the result must be
    # (cos 1, -sin 1) at (dim 0, dim 4), the speedrun's
    # y1 = x1 cos + x2 sin, y2 = x2 cos - x1 sin.
    var tbl = rotary_table_host(TMAX, HD)
    var x = _zeros(NQKV)
    var base = 1 * 3 * C  # bt = 1 -> t = 1, q third, head 0
    x[base + 0] = 1.0
    var y = _run_cpu[False](x, tbl)
    assert_true(abs(y[base + 0] - 0.5403023) < 1e-6, "y1 != cos(1)")
    assert_true(abs(y[base + 4] + 0.8414710) < 1e-6, "y2 != -sin(1)")
    # the same input in the K third rotates too, in the V third it must not.
    var xv = _zeros(NQKV)
    xv[base + 2 * C + 0] = 1.0
    var yv = _run_cpu[False](xv, tbl)
    assert_true(yv[base + 2 * C + 0] == 1.0, "v third was rotated")
    assert_true(yv[base + 2 * C + 4] == 0.0, "v third was rotated")


def test_matches_float64_reference_and_untouched_dims() raises:
    var tbl = rotary_table_host(TMAX, HD)
    var x = _fill(_zeros(NQKV), 3)
    var y = _run_cpu[False](x, tbl)
    var expect = _reference(x)
    var worst = Float32(0.0)
    for i in range(NQKV):
        var d = abs(y[i] - expect[i])
        if d > worst:
            worst = d
    assert_true(worst < 2e-6, "kernel differs from float64 reference")
    # Bitwise-untouched: v third, identity dims (>= n of each half), t == 0.
    for bt in range(ROWS):
        var t = bt % T
        for sec in range(3):
            for h in range(H):
                var base = bt * 3 * C + sec * C + h * HD
                for d in range(HD):
                    var pair = d % (HD // 2)
                    var rotated = sec < 2 and t > 0 and pair < HD // 4
                    if not rotated:
                        assert_true(
                            y[base + d] == x[base + d],
                            "an element that must be identity changed",
                        )


def test_inverse_undoes_forward_and_is_the_adjoint() raises:
    var tbl = rotary_table_host(TMAX, HD)
    var x = _fill(_zeros(NQKV), 5)
    var g = _fill(_zeros(NQKV), 9)
    var rx = _run_cpu[False](x, tbl)
    var back = _run_cpu[True](rx, tbl)
    for i in range(NQKV):
        assert_true(abs(back[i] - x[i]) < 2e-6, "inverse(forward(x)) != x")
    # <R x, g> == <x, R^T g>, which is what makes the inverse rotation the
    # correct gradient of the forward map (R is applied to dq/dk in backward).
    var rtg = _run_cpu[True](g, tbl)
    var lhs = Float64(0.0)
    var rhs = Float64(0.0)
    for i in range(NQKV):
        lhs += Float64(rx[i]) * Float64(g[i])
        rhs += Float64(x[i]) * Float64(rtg[i])
    assert_true(abs(lhs - rhs) < 1e-3, "inverse is not the adjoint")


def test_gpu_matches_cpu() raises:
    var tbl = rotary_table_host(TMAX, HD)
    var x = _fill(_zeros(NQKV), 11)
    var fc = _run_cpu[False](x, tbl)
    var fg = _run_gpu[False](x, tbl)
    var ic = _run_cpu[True](x, tbl)
    var ig = _run_gpu[True](x, tbl)
    for i in range(NQKV):
        assert_true(abs(fc[i] - fg[i]) < 1e-6, "gpu forward != cpu forward")
        assert_true(abs(ic[i] - ig[i]) < 1e-6, "gpu inverse != cpu inverse")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
