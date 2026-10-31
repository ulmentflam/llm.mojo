# ===----------------------------------------------------------------------=== #
# Fused q/k row kernels (llmm/qk_norm.mojo, do_rot=True), W10. Run: make test-mojo
#
# The fused forward does rmsnorm (x q_mul) THEN the half-truncated rotation of
# each head row in one pass over the head-layout [B, NH, T, hd] buffers; the
# fused backward does the inverse rotation THEN the norm backward against the
# raw (pre-norm, pre-rotation) row. Before W10 these were two kernels in the
# opposite order (rotary on the token-layout qkv, then the norm). Checks, each
# written to fail on a specific mistake:
#   * fused forward / backward vs an independent float64 scalar model
#     (catches a rotation applied with the wrong sign: forward rotation in the
#     backward, or the norm backward applied against the wrong row),
#   * fused vs the old two-kernel composition (rotary_qk_rows on qkv, split,
#     qk_rows_fwd; backward qk_rows_bwd against the ROTATED qkv then inverse
#     rotary): proves the reordering is a pure refactor up to fp32 rounding,
#   * CPU target vs GPU target, do_norm in {True, False}, q_mul != 1 (a wrong
#     q_mul fold shows as a q-vs-k asymmetry),
#   * v and the identity half of each head are never touched by the rotation.
# ===----------------------------------------------------------------------=== #

from std.math import cos, sin, sqrt
from std.testing import assert_true, TestSuite

from max.gpu.host import DeviceContext

from llmm.memory import MutMemPtr, as_mut_kernel, as_immut_kernel_from_mut
from llmm.qk_norm import qk_rows_fwd, qk_rows_bwd
from llmm.rotary import rotary_qk_rows, rotary_table_host
from _gpu_test_common import shared_gpu_ctx

comptime B = 2
comptime T = 5
comptime TMAX = 8
comptime H = 3
comptime HD = 64  # n = 16 rotated pairs, pairs (i, i + 32): one lane-pair each
comptime C = H * HD
comptime NQKV = B * T * 3 * C
comptime NROW = B * H * T
comptime NH_ELEMS = NROW * HD  # one head-layout buffer (q or k)
comptime QMUL = Float32(0.7)


def _fill(n: Int, seed: Int) -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0.0))
    var s = UInt64(seed) * 2654435761 + 12345
    for i in range(n):
        s = s * 6364136223846793005 + 1442695040888963407
        out[i] = Float32(Int((s >> 33) % 2001) - 1000) / 400.0
    return out^


def _split(qkv: List[Float32], sec: Int) -> List[Float32]:
    """Head-layout [B, NH, T, hd] copy of the q (sec 0) or k (sec 1) third."""
    var out = List[Float32](length=NH_ELEMS, fill=Float32(0.0))
    for b in range(B):
        for h in range(H):
            for t in range(T):
                for d in range(HD):
                    out[((b * H + h) * T + t) * HD + d] = qkv[
                        (b * T + t) * 3 * C + sec * C + h * HD + d
                    ]
    return out^


def _merge(dq: List[Float32], dk: List[Float32]) -> List[Float32]:
    """Token-layout d_qkv with v zero from head-layout dq, dk."""
    var out = List[Float32](length=NQKV, fill=Float32(0.0))
    for b in range(B):
        for h in range(H):
            for t in range(T):
                for d in range(HD):
                    var r = ((b * H + h) * T + t) * HD + d
                    out[(b * T + t) * 3 * C + h * HD + d] = dq[r]
                    out[(b * T + t) * 3 * C + C + h * HD + d] = dk[r]
    return out^


# --- runners: every one takes/returns plain lists, target picks the context --- #


def _with_ctx[target: StaticString]() raises -> DeviceContext:
    comptime if target == "cpu":
        return DeviceContext(api="cpu")
    else:
        return shared_gpu_ctx()


def _fwd[
    target: StaticString, do_norm: Bool, do_rot: Bool
](q: List[Float32], k: List[Float32], tbl: List[Float32]) raises -> Tuple[
    List[Float32], List[Float32]
]:
    var ctx = _with_ctx[target]()
    var hq = ctx.enqueue_create_host_buffer[DType.float32](NH_ELEMS)
    var hk = ctx.enqueue_create_host_buffer[DType.float32](NH_ELEMS)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](len(tbl))
    var dq = ctx.enqueue_create_buffer[DType.float32](NH_ELEMS)
    var dk = ctx.enqueue_create_buffer[DType.float32](NH_ELEMS)
    var dt = ctx.enqueue_create_buffer[DType.float32](len(tbl))
    ctx.synchronize()
    for i in range(NH_ELEMS):
        hq[i] = q[i]
        hk[i] = k[i]
    for i in range(len(tbl)):
        ht[i] = tbl[i]
    ctx.enqueue_copy(dq, hq)
    ctx.enqueue_copy(dk, hk)
    ctx.enqueue_copy(dt, ht)
    ctx.synchronize()
    qk_rows_fwd[DType.float32, target, do_norm, do_rot](
        as_mut_kernel[DType.float32](
            MutMemPtr[DType.float32](unsafe_from_address=Int(dq.unsafe_ptr()))
        ),
        as_mut_kernel[DType.float32](
            MutMemPtr[DType.float32](unsafe_from_address=Int(dk.unsafe_ptr()))
        ),
        B,
        H,
        T,
        HD,
        QMUL,
        ctx,
        as_immut_kernel_from_mut[DType.float32](
            MutMemPtr[DType.float32](unsafe_from_address=Int(dt.unsafe_ptr()))
        ),
        TMAX,
    )
    ctx.enqueue_copy(hq, dq)
    ctx.enqueue_copy(hk, dk)
    ctx.synchronize()
    var oq = List[Float32](length=NH_ELEMS, fill=Float32(0.0))
    var ok = List[Float32](length=NH_ELEMS, fill=Float32(0.0))
    for i in range(NH_ELEMS):
        oq[i] = hq[i]
        ok[i] = hk[i]
    return (oq^, ok^)


def _bwd[
    target: StaticString, do_norm: Bool, do_rot: Bool
](
    gq: List[Float32],
    gk: List[Float32],
    qkv: List[Float32],
    tbl: List[Float32],
) raises -> Tuple[List[Float32], List[Float32]]:
    var ctx = _with_ctx[target]()
    var hq = ctx.enqueue_create_host_buffer[DType.float32](NH_ELEMS)
    var hk = ctx.enqueue_create_host_buffer[DType.float32](NH_ELEMS)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](NQKV)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](len(tbl))
    var dq = ctx.enqueue_create_buffer[DType.float32](NH_ELEMS)
    var dk = ctx.enqueue_create_buffer[DType.float32](NH_ELEMS)
    var dx = ctx.enqueue_create_buffer[DType.float32](NQKV)
    var dt = ctx.enqueue_create_buffer[DType.float32](len(tbl))
    ctx.synchronize()
    for i in range(NH_ELEMS):
        hq[i] = gq[i]
        hk[i] = gk[i]
    for i in range(NQKV):
        hx[i] = qkv[i]
    for i in range(len(tbl)):
        ht[i] = tbl[i]
    ctx.enqueue_copy(dq, hq)
    ctx.enqueue_copy(dk, hk)
    ctx.enqueue_copy(dx, hx)
    ctx.enqueue_copy(dt, ht)
    ctx.synchronize()
    qk_rows_bwd[DType.float32, target, do_norm, do_rot](
        as_mut_kernel[DType.float32](
            MutMemPtr[DType.float32](unsafe_from_address=Int(dq.unsafe_ptr()))
        ),
        as_mut_kernel[DType.float32](
            MutMemPtr[DType.float32](unsafe_from_address=Int(dk.unsafe_ptr()))
        ),
        as_immut_kernel_from_mut[DType.float32](
            MutMemPtr[DType.float32](unsafe_from_address=Int(dx.unsafe_ptr()))
        ),
        B,
        H,
        T,
        HD,
        QMUL,
        ctx,
        as_immut_kernel_from_mut[DType.float32](
            MutMemPtr[DType.float32](unsafe_from_address=Int(dt.unsafe_ptr()))
        ),
        TMAX,
    )
    ctx.enqueue_copy(hq, dq)
    ctx.enqueue_copy(hk, dk)
    ctx.synchronize()
    var oq = List[Float32](length=NH_ELEMS, fill=Float32(0.0))
    var ok = List[Float32](length=NH_ELEMS, fill=Float32(0.0))
    for i in range(NH_ELEMS):
        oq[i] = hq[i]
        ok[i] = hk[i]
    return (oq^, ok^)


def _rotary_tokens[
    inverse: Bool
](qkv: List[Float32], tbl: List[Float32]) raises -> List[Float32]:
    """The pre-W10 token-layout rotary kernel (CPU), for the composition."""
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
    var out = List[Float32](length=NQKV, fill=Float32(0.0))
    for i in range(NQKV):
        out[i] = buf[i]
    return out^


# --- independent float64 model ---------------------------------------------- #


def _angle(t: Int, i: Int) -> Float64:
    var n = HD // 4
    var freq = Float64(1.0) / (Float64(1024.0) ** (Float64(i) / Float64(n - 1)))
    return Float64(t) * freq


def _ref_fwd(x: List[Float32], is_q: Bool, norm: Bool) -> List[Float32]:
    var out = x.copy()
    for r in range(NROW):
        var t = r % T
        var ss = Float64(0)
        for d in range(HD):
            ss += Float64(x[r * HD + d]) * Float64(x[r * HD + d])
        var mul = Float64(QMUL) if is_q else Float64(1)
        if norm:
            mul = mul / sqrt(ss / Float64(HD) + 1e-6)
        for i in range(HD // 2):
            var x1 = Float64(x[r * HD + i]) * mul
            var x2 = Float64(x[r * HD + i + HD // 2]) * mul
            if i < HD // 4:
                var a = _angle(t, i)
                out[r * HD + i] = Float32(x1 * cos(a) + x2 * sin(a))
                out[r * HD + i + HD // 2] = Float32(x2 * cos(a) - x1 * sin(a))
            else:
                out[r * HD + i] = Float32(x1)
                out[r * HD + i + HD // 2] = Float32(x2)
    return out^


def _ref_bwd(
    g: List[Float32], x: List[Float32], is_q: Bool, norm: Bool
) -> List[Float32]:
    """x = RAW head-layout row (pre-norm, pre-rotation); g = d/d(rotated out).
    """
    var out = g.copy()
    for r in range(NROW):
        var t = r % T
        var mul = Float64(QMUL) if is_q else Float64(1)
        # h = R^T g
        var h = List[Float64](length=HD, fill=Float64(0))
        for i in range(HD // 2):
            var g1 = Float64(g[r * HD + i])
            var g2 = Float64(g[r * HD + i + HD // 2])
            if i < HD // 4:
                var a = _angle(t, i)
                h[i] = g1 * cos(a) - g2 * sin(a)
                h[i + HD // 2] = g1 * sin(a) + g2 * cos(a)
            else:
                h[i] = g1
                h[i + HD // 2] = g2
        if norm:
            var ss = Float64(0)
            for d in range(HD):
                ss += Float64(x[r * HD + d]) * Float64(x[r * HD + d])
            var rstd = 1.0 / sqrt(ss / Float64(HD) + 1e-6)
            var dot = Float64(0)
            for d in range(HD):
                dot += h[d] * mul * Float64(x[r * HD + d]) * rstd
            dot = dot / Float64(HD)
            for d in range(HD):
                var u = Float64(x[r * HD + d]) * rstd
                out[r * HD + d] = Float32(rstd * (h[d] * mul - u * dot))
        else:
            for d in range(HD):
                out[r * HD + d] = Float32(h[d] * mul)
    return out^


def _maxdiff(a: List[Float32], b: List[Float32]) -> Float32:
    var worst = Float32(0)
    for i in range(len(a)):
        var d = abs(a[i] - b[i])
        if d > worst:
            worst = d
    return worst


def _check_fwd_vs_ref[target: StaticString, do_norm: Bool]() raises:
    var tbl = rotary_table_host(TMAX, HD)
    var qkv = _fill(NQKV, 21)
    var q = _split(qkv, 0)
    var k = _split(qkv, 1)
    var o = _fwd[target, do_norm, True](q, k, tbl)
    assert_true(
        _maxdiff(o[0], _ref_fwd(q, True, do_norm)) < 1e-5,
        "fused forward q != float64 model",
    )
    assert_true(
        _maxdiff(o[1], _ref_fwd(k, False, do_norm)) < 1e-5,
        "fused forward k != float64 model",
    )


def _check_bwd_vs_ref[target: StaticString, do_norm: Bool]() raises:
    var tbl = rotary_table_host(TMAX, HD)
    var qkv = _fill(NQKV, 22)
    var q = _split(qkv, 0)
    var k = _split(qkv, 1)
    var gq = _fill(NH_ELEMS, 23)
    var gk = _fill(NH_ELEMS, 24)
    var o = _bwd[target, do_norm, True](gq, gk, qkv, tbl)
    assert_true(
        _maxdiff(o[0], _ref_bwd(gq, q, True, do_norm)) < 1e-5,
        "fused backward dq != float64 model",
    )
    assert_true(
        _maxdiff(o[1], _ref_bwd(gk, k, False, do_norm)) < 1e-5,
        "fused backward dk != float64 model",
    )


def test_fwd_cpu_norm_rot() raises:
    _check_fwd_vs_ref["cpu", True]()


def test_fwd_cpu_rot_only() raises:
    _check_fwd_vs_ref["cpu", False]()


def test_bwd_cpu_norm_rot() raises:
    _check_bwd_vs_ref["cpu", True]()


def test_bwd_cpu_rot_only() raises:
    _check_bwd_vs_ref["cpu", False]()


def test_fwd_gpu_norm_rot() raises:
    _check_fwd_vs_ref["gpu", True]()


def test_fwd_gpu_rot_only() raises:
    _check_fwd_vs_ref["gpu", False]()


def test_bwd_gpu_norm_rot() raises:
    _check_bwd_vs_ref["gpu", True]()


def test_bwd_gpu_rot_only() raises:
    _check_bwd_vs_ref["gpu", False]()


def test_fused_equals_old_two_kernel_composition() raises:
    # OLD: rotary on token-layout qkv, split, qk_rows_fwd(norm).
    # NEW: split the RAW qkv, fused norm-then-rotate. Rotation is orthogonal per
    # head so the order change is a refactor up to fp32 rounding.
    var tbl = rotary_table_host(TMAX, HD)
    var qkv = _fill(NQKV, 31)
    var rot = _rotary_tokens[False](qkv, tbl)
    var old = _fwd["cpu", True, False](_split(rot, 0), _split(rot, 1), tbl)
    var new = _fwd["cpu", True, True](_split(qkv, 0), _split(qkv, 1), tbl)
    assert_true(_maxdiff(old[0], new[0]) < 2e-5, "fwd q: fused != old")
    assert_true(_maxdiff(old[1], new[1]) < 2e-5, "fwd k: fused != old")
    # Backward. OLD: norm backward against the ROTATED qkv, merge, then the
    # inverse rotation on the token-layout d_qkv. NEW: fused, raw qkv.
    var gq = _fill(NH_ELEMS, 32)
    var gk = _fill(NH_ELEMS, 33)
    var ob = _bwd["cpu", True, False](gq, gk, rot, tbl)
    var d_old = _rotary_tokens[True](_merge(ob[0], ob[1]), tbl)
    var nb = _bwd["cpu", True, True](gq, gk, qkv, tbl)
    var d_new = _merge(nb[0], nb[1])
    assert_true(_maxdiff(d_old, d_new) < 2e-5, "bwd: fused != old")


def test_gpu_matches_cpu_all_combos() raises:
    var tbl = rotary_table_host(TMAX, HD)
    var qkv = _fill(NQKV, 41)
    var q = _split(qkv, 0)
    var k = _split(qkv, 1)
    var gq = _fill(NH_ELEMS, 42)
    var gk = _fill(NH_ELEMS, 43)
    var fc = _fwd["cpu", True, True](q, k, tbl)
    var fg = _fwd["gpu", True, True](q, k, tbl)
    assert_true(_maxdiff(fc[0], fg[0]) < 1e-5, "gpu fwd q != cpu")
    assert_true(_maxdiff(fc[1], fg[1]) < 1e-5, "gpu fwd k != cpu")
    var bc = _bwd["cpu", True, True](gq, gk, qkv, tbl)
    var bg = _bwd["gpu", True, True](gq, gk, qkv, tbl)
    assert_true(_maxdiff(bc[0], bg[0]) < 1e-5, "gpu bwd dq != cpu")
    assert_true(_maxdiff(bc[1], bg[1]) < 1e-5, "gpu bwd dk != cpu")


def test_position_zero_and_identity_half_are_norm_only() raises:
    # t == 0 rows have angle 0, and dims >= hd/4 of each half are the identity
    # half: with do_norm=False the fused forward must reproduce q_mul * x (q) or
    # x (k) BITWISE there.
    var tbl = rotary_table_host(TMAX, HD)
    var qkv = _fill(NQKV, 51)
    var q = _split(qkv, 0)
    var k = _split(qkv, 1)
    var o = _fwd["cpu", False, True](q, k, tbl)
    for r in range(NROW):
        for d in range(HD):
            var pair = d % (HD // 2)
            if r % T == 0 or pair >= HD // 4:
                assert_true(
                    o[0][r * HD + d] == q[r * HD + d] * QMUL,
                    "identity element of q changed beyond q_mul",
                )
                assert_true(
                    o[1][r * HD + d] == k[r * HD + d],
                    "identity element of k changed",
                )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
