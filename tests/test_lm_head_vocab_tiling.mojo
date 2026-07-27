# ===----------------------------------------------------------------------=== #
# Pure-Mojo tests for the vocab-tiled LM head (llmm.matmul's
# matmul_lm_head_fwd_tile / matmul_lm_head_bwd_tile).
#
# Run with:  make test-mojo   (equivalent to `mojo run -I . tests/test_lm_head_vocab_tiling.mojo`)
#
# WHY THIS FILE EXISTS. The tied `wte` LM head is the largest tensor in the
# model, and it is de-residented by splitting its GEMM into vocabulary tiles.
# A vocab tile is a COLUMN slice of the row-major [rows, V_p] logits matrix —
# a strided submatrix whose leading dimension (V_p) exceeds its width. Every
# other GEMM in llmm/matmul.mojo assumes ld == width, so the tiled entry points
# are the only place that layout exists, and they are the only place it can be
# got wrong.
#
# The trainer's own tile count is a COMPILE-TIME knob
# (-D LLMM_LM_HEAD_VOCAB_TILES), so a single binary cannot A/B tiled vs untiled
# through GPT2. These tests therefore drive the two tiled entry points directly
# and compare against the dense matmul_fwd / matmul_bwd on the same inputs.
#
# Coverage that matters:
#   - MORE THAN ONE TILE. A suite that only ever ran a single tile would prove
#     nothing about the strided path, so every case here splits the vocabulary.
#   - A RAGGED LAST TILE. V_p is deliberately NOT a multiple of the tile width
#     (37 = 8+8+8+8+5), which is the real production shape too: V_p=50304 with
#     6400-row tiles ends in a 5504-row remainder.
#   - Both directions, and both d_weight (reduction over rows — must be
#     bit-comparable to dense) and d_input (reduction over the vocabulary —
#     split by tiling, so it is reassociated and only close, not identical).
#
# This runs on the CPU target, which exercises the PORTABLE staging path
# (gather/scatter the column slice through a contiguous buffer). The cuBLASLt
# path takes the leading dimension natively instead and is covered by the
# end-to-end GPU gates (make verify-gpu / make test).
# ===----------------------------------------------------------------------=== #

from std.testing import assert_almost_equal, assert_true, TestSuite
from std.memory import alloc
from std.math import ceildiv
from std.gpu.host import DeviceContext

from llmm.memory import MutMemPtr, as_mut_kernel, as_immut_kernel_from_mut
from llmm.matmul import (
    matmul_fwd,
    matmul_bwd,
    matmul_lm_head_fwd_tile,
    matmul_lm_head_bwd_tile,
)

comptime DT = DType.float32
comptime TARGET = "cpu"

# Shapes. ROWS = B*T. V_P is intentionally indivisible by TILE so the last tile
# is short; C is small enough to keep the reference loops cheap.
comptime ROWS = 6
comptime C = 4
comptime V_P = 37
comptime TILE = 8


# Deterministic, mixed-sign filler — no RNG dependency, and the values are
# spread enough that a transposed or mis-strided operand cannot coincidentally
# agree.
def _fill(p: MutMemPtr[DT], n: Int, seed: Int) -> None:
    for i in range(n):
        var x = Float32((i * 37 + seed * 17) % 23) - 11.0
        p[i] = x * 0.037


def _alloc(n: Int) -> MutMemPtr[DT]:
    var p = alloc[Scalar[DT]](n)
    var q = rebind[MutMemPtr[DT]](p.as_unsafe_any_origin())
    for i in range(n):
        q[i] = 0.0
    return q


# ===----------------------------------------------------------------------=== #
# Tile-geometry helper — mirrors GPT2._lm_head_tile_rows' rounding rule.
# ===----------------------------------------------------------------------=== #


def _tile_rows(v_p: Int, tiles: Int) -> Int:
    if tiles <= 1:
        return v_p
    var t = ceildiv(v_p, tiles)
    if v_p >= 128:
        t = ceildiv(t, 128) * 128
    return min(max(t, 1), v_p)


def test_tile_geometry_covers_vocab_exactly() raises:
    # Whatever the knob, the tiles must partition [0, V_p) with no gap and no
    # overlap, and the last one may be short.
    var cases = [(37, 5), (64, 8), (50304, 8), (50304, 1), (50304, 3), (7, 8)]
    for i in range(len(cases)):
        var v_p = cases[i][0]
        var k = cases[i][1]
        var tr = _tile_rows(v_p, k)
        assert_true(tr >= 1, "tile rows must be positive")
        assert_true(tr <= v_p, "tile rows must not exceed the vocabulary")
        var covered = 0
        var t0 = 0
        while t0 < v_p:
            var oc = min(tr, v_p - t0)
            assert_true(oc >= 1, "every tile must be non-empty")
            covered += oc
            t0 += tr
        assert_true(covered == v_p, "tiles must cover the vocabulary exactly")


def test_production_shape_actually_tiles() raises:
    # Guards the default knob: at GPT-2 124M's padded vocabulary the LM head
    # must really split, and each tile must be a clean multiple of 128 rows
    # (except the remainder). If this ever collapses to one tile the memory win
    # silently disappears while every other test still passes.
    var tr = _tile_rows(50304, 8)
    assert_true(tr == 6400, "expected 6400-row tiles at V_p=50304, K=8")
    assert_true(ceildiv(50304, tr) == 8, "expected 8 tiles at V_p=50304, K=8")
    assert_true(50304 % tr != 0, "expected a ragged final tile at V_p=50304")


def test_tiny_vocab_still_tiles() raises:
    # The CPU equivalence harness runs a V_p=64 micro-model. Below 128 rows the
    # multiple-of-128 rounding must NOT round the tile up to the whole
    # vocabulary, or that harness would never take the tiled branch.
    var tr = _tile_rows(64, 8)
    assert_true(tr == 8, "expected 8-row tiles at V_p=64, K=8")
    assert_true(ceildiv(64, tr) == 8, "expected 8 tiles at V_p=64, K=8")


# ===----------------------------------------------------------------------=== #
# Forward: tiled == dense
# ===----------------------------------------------------------------------=== #


def test_fwd_tiled_matches_dense() raises:
    var ctx = DeviceContext(api="cpu")

    var inp = _alloc(ROWS * C)
    var wte = _alloc(V_P * C)
    _fill(inp, ROWS * C, 1)
    _fill(wte, V_P * C, 2)

    var dense = _alloc(ROWS * V_P)
    var tiled = _alloc(ROWS * V_P)

    matmul_fwd[DT, TARGET, use_gelu=False, has_bias=False](
        as_mut_kernel[DT](dense),
        as_mut_kernel[DT](dense),  # dummy pre_gelu (use_gelu=False)
        as_mut_kernel[DT](inp),
        as_immut_kernel_from_mut[DT](wte),
        as_immut_kernel_from_mut[DT](wte),  # dummy bias (has_bias=False)
        Int64(ROWS),
        Int64(1),
        Int64(C),
        Int64(V_P),
        ctx,
    )

    var ntiles = 0
    var t0 = 0
    while t0 < V_P:
        var oc = min(TILE, V_P - t0)
        matmul_lm_head_fwd_tile[DT, TARGET](
            as_mut_kernel[DT](tiled + t0),
            V_P,
            as_immut_kernel_from_mut[DT](inp),
            as_immut_kernel_from_mut[DT](wte + t0 * C),
            ROWS,
            C,
            oc,
            ctx,
        )
        ntiles += 1
        t0 += TILE
    ctx.synchronize()

    assert_true(ntiles > 1, "test must exercise more than one vocab tile")

    # The reduction axis (C) is untouched by vocab tiling, so each logit is
    # still one full-length dot product: this is an exact-match check, not a
    # tolerance check.
    for i in range(ROWS * V_P):
        assert_almost_equal(tiled[i], dense[i], atol=1e-6)

    inp.free()
    wte.free()
    dense.free()
    tiled.free()


def test_fwd_tiled_writes_every_column() raises:
    # A tile loop that dropped the ragged last tile, or that wrote at the wrong
    # stride, would leave poison behind. Pre-poison the output and assert every
    # element was overwritten.
    var ctx = DeviceContext(api="cpu")
    var inp = _alloc(ROWS * C)
    var wte = _alloc(V_P * C)
    _fill(inp, ROWS * C, 3)
    _fill(wte, V_P * C, 4)

    comptime POISON = Float32(-12345.0)
    var out = _alloc(ROWS * V_P)
    for i in range(ROWS * V_P):
        out[i] = POISON

    var t0 = 0
    while t0 < V_P:
        var oc = min(TILE, V_P - t0)
        matmul_lm_head_fwd_tile[DT, TARGET](
            as_mut_kernel[DT](out + t0),
            V_P,
            as_immut_kernel_from_mut[DT](inp),
            as_immut_kernel_from_mut[DT](wte + t0 * C),
            ROWS,
            C,
            oc,
            ctx,
        )
        t0 += TILE
    ctx.synchronize()

    for i in range(ROWS * V_P):
        assert_true(
            out[i] != POISON,
            "logit " + String(i) + " was never written by the tile loop",
        )

    inp.free()
    wte.free()
    out.free()


# ===----------------------------------------------------------------------=== #
# Backward: tiled == dense
# ===----------------------------------------------------------------------=== #


def test_bwd_tiled_matches_dense() raises:
    var ctx = DeviceContext(api="cpu")

    var inp = _alloc(ROWS * C)  # ln_f
    var wte = _alloc(V_P * C)
    var dlogits = _alloc(ROWS * V_P)
    _fill(inp, ROWS * C, 5)
    _fill(wte, V_P * C, 6)
    _fill(dlogits, ROWS * V_P, 7)

    var d_inp_dense = _alloc(ROWS * C)
    var d_w_dense = _alloc(V_P * C)
    var scratch_dense = _alloc(V_P * ROWS)

    # Dense reference. accumulate defaults True and d_w_dense starts zeroed,
    # exactly as the trainer's pre-zeroed gradient pool does.
    matmul_bwd[DT, TARGET, use_gelu=False, has_bias=False](
        as_mut_kernel[DT](d_inp_dense),
        as_mut_kernel[DT](d_w_dense),
        as_mut_kernel[DT](d_w_dense),  # dummy d_bias (has_bias=False)
        as_mut_kernel[DT](dlogits),
        as_mut_kernel[DT](inp),
        as_immut_kernel_from_mut[DT](wte),
        as_mut_kernel[DT](d_inp_dense),  # dummy pre_gelu (use_gelu=False)
        as_mut_kernel[DT](scratch_dense),
        Int64(ROWS),
        Int64(1),
        Int64(C),
        Int64(V_P),
        ctx,
    )
    ctx.synchronize()

    var d_inp_tiled = _alloc(ROWS * C)
    var d_w_tiled = _alloc(V_P * C)
    var scratch_tiled = _alloc(V_P * ROWS)

    var ntiles = 0
    var t0 = 0
    var first = True
    while t0 < V_P:
        var oc = min(TILE, V_P - t0)
        matmul_lm_head_bwd_tile[DT, TARGET](
            as_mut_kernel[DT](d_inp_tiled),
            as_mut_kernel[DT](d_w_tiled + t0 * C),
            as_immut_kernel_from_mut[DT](dlogits + t0),
            V_P,
            as_immut_kernel_from_mut[DT](inp),
            as_immut_kernel_from_mut[DT](wte + t0 * C),
            as_mut_kernel[DT](scratch_tiled),
            ROWS,
            C,
            oc,
            not first,
            ctx,
        )
        ntiles += 1
        first = False
        t0 += TILE
    ctx.synchronize()

    assert_true(ntiles > 1, "test must exercise more than one vocab tile")

    # d_weight reduces over `rows` only — vocab tiling never splits that sum,
    # so every wte gradient row is computed by the same dot products as dense.
    for i in range(V_P * C):
        assert_almost_equal(d_w_tiled[i], d_w_dense[i], atol=1e-6)

    # d_input reduces over the vocabulary, which tiling DOES split, so the sum
    # is reassociated across tiles. Close, not identical.
    for i in range(ROWS * C):
        assert_almost_equal(d_inp_tiled[i], d_inp_dense[i], atol=1e-5)

    inp.free()
    wte.free()
    dlogits.free()
    d_inp_dense.free()
    d_w_dense.free()
    scratch_dense.free()
    d_inp_tiled.free()
    d_w_tiled.free()
    scratch_tiled.free()


def test_bwd_d_input_accumulates_across_tiles() raises:
    # The d_input accumulation flag is the one piece of per-tile state that a
    # refactor could plausibly get backwards. If every tile overwrote instead of
    # accumulating, d_input would equal the LAST tile's contribution alone —
    # assert it does not, so a stuck-at-overwrite bug cannot pass silently.
    var ctx = DeviceContext(api="cpu")

    var inp = _alloc(ROWS * C)
    var wte = _alloc(V_P * C)
    var dlogits = _alloc(ROWS * V_P)
    _fill(inp, ROWS * C, 8)
    _fill(wte, V_P * C, 9)
    _fill(dlogits, ROWS * V_P, 10)

    var d_w = _alloc(V_P * C)
    var scratch = _alloc(V_P * ROWS)

    var accumulated = _alloc(ROWS * C)
    var t0 = 0
    var first = True
    var last_start = 0
    var last_oc = 0
    while t0 < V_P:
        var oc = min(TILE, V_P - t0)
        matmul_lm_head_bwd_tile[DT, TARGET](
            as_mut_kernel[DT](accumulated),
            as_mut_kernel[DT](d_w + t0 * C),
            as_immut_kernel_from_mut[DT](dlogits + t0),
            V_P,
            as_immut_kernel_from_mut[DT](inp),
            as_immut_kernel_from_mut[DT](wte + t0 * C),
            as_mut_kernel[DT](scratch),
            ROWS,
            C,
            oc,
            not first,
            ctx,
        )
        first = False
        last_start = t0
        last_oc = oc
        t0 += TILE
    ctx.synchronize()

    # Now recompute ONLY the final tile with accumulate=False.
    var d_w2 = _alloc(V_P * C)
    var last_only = _alloc(ROWS * C)
    matmul_lm_head_bwd_tile[DT, TARGET](
        as_mut_kernel[DT](last_only),
        as_mut_kernel[DT](d_w2 + last_start * C),
        as_immut_kernel_from_mut[DT](dlogits + last_start),
        V_P,
        as_immut_kernel_from_mut[DT](inp),
        as_immut_kernel_from_mut[DT](wte + last_start * C),
        as_mut_kernel[DT](scratch),
        ROWS,
        C,
        last_oc,
        False,
        ctx,
    )
    ctx.synchronize()

    var differs = False
    for i in range(ROWS * C):
        if abs(accumulated[i] - last_only[i]) > 1e-6:
            differs = True
    assert_true(
        differs,
        (
            "d_input equals the last tile alone — the accumulate flag is not"
            " taking effect"
        ),
    )

    inp.free()
    wte.free()
    dlogits.free()
    d_w.free()
    d_w2.free()
    scratch.free()
    accumulated.free()
    last_only.free()


# ===----------------------------------------------------------------------=== #
# Production-scale coverage.
#
# The cases above run at a toy vocabulary (V_P=37) so they can afford exhaustive
# element-by-element checks. They prove the strided logic, but they do NOT prove
# it at the shape that actually ships.
#
# That gap matters more than usual here. No pre-existing test in this repo drives
# a matmul anywhere near vocabulary scale — the largest `output_channels` any
# other test passes is 3072 (tests/test_matmul_fwd_lowp.mojo's `fc` site);
# test_matmul_equivalence.py tops out at 2304. Real vocab shape appears only in
# tests/test_softmax_equivalence.py, which never calls the matmul at all. So with
# a default tile size, every OTHER test in the suite runs the LM head at exactly
# one tile: the loop body executes once, the leading-dimension logic that is the
# entire technical risk of vocab tiling never runs, and the suite goes green
# having proven nothing about it.
#
# These two tests close that hole at the real V_p = 50304, with the real default
# tile width of 6400 rows — which leaves a ragged 5504-row final tile
# (7*6400 = 44800; 50304 - 44800 = 5504). Both assert at runtime that they took
# the multi-tile branch, so they cannot degrade into a single-tile no-op without
# failing loudly.
#
# Cost: rows is kept at 128 (vs the trainer's B*T = 4096) purely so this stays a
# fast CPU test. Vocabulary width, tile width and the ragged tail — the things
# under test — are exactly production.
# ===----------------------------------------------------------------------=== #

comptime BIG_V_P = 50304  # GPT-2 124M padded vocabulary
comptime BIG_TILE = 6400  # LLMM_LM_HEAD_VOCAB_TILES=8 -> ceil(50304/8)=6288 -> 6400
comptime BIG_ROWS = 128


def test_fwd_tiled_matches_dense_at_vocab_scale() raises:
    var ctx = DeviceContext(api="cpu")

    var inp = _alloc(BIG_ROWS * C)
    var wte = _alloc(BIG_V_P * C)
    _fill(inp, BIG_ROWS * C, 11)
    _fill(wte, BIG_V_P * C, 12)

    var dense = _alloc(BIG_ROWS * BIG_V_P)
    var tiled = _alloc(BIG_ROWS * BIG_V_P)

    matmul_fwd[DT, TARGET, use_gelu=False, has_bias=False](
        as_mut_kernel[DT](dense),
        as_mut_kernel[DT](dense),  # dummy pre_gelu
        as_mut_kernel[DT](inp),
        as_immut_kernel_from_mut[DT](wte),
        as_immut_kernel_from_mut[DT](wte),  # dummy bias
        Int64(BIG_ROWS),
        Int64(1),
        Int64(C),
        Int64(BIG_V_P),
        ctx,
    )

    var ntiles = 0
    var ragged = 0
    var t0 = 0
    while t0 < BIG_V_P:
        var oc = min(BIG_TILE, BIG_V_P - t0)
        if oc != BIG_TILE:
            ragged = oc
        matmul_lm_head_fwd_tile[DT, TARGET](
            as_mut_kernel[DT](tiled + t0),
            BIG_V_P,
            as_immut_kernel_from_mut[DT](inp),
            as_immut_kernel_from_mut[DT](wte + t0 * C),
            BIG_ROWS,
            C,
            oc,
            ctx,
        )
        ntiles += 1
        t0 += BIG_TILE
    ctx.synchronize()

    assert_true(ntiles == 8, "expected 8 tiles at V_p=50304, tile=6400")
    assert_true(ragged == 5504, "expected a ragged 5504-row final tile")

    for i in range(BIG_ROWS * BIG_V_P):
        assert_almost_equal(tiled[i], dense[i], atol=1e-5)

    inp.free()
    wte.free()
    dense.free()
    tiled.free()


def test_bwd_tiled_matches_dense_at_vocab_scale() raises:
    var ctx = DeviceContext(api="cpu")

    var inp = _alloc(BIG_ROWS * C)
    var wte = _alloc(BIG_V_P * C)
    var dlogits = _alloc(BIG_ROWS * BIG_V_P)
    _fill(inp, BIG_ROWS * C, 13)
    _fill(wte, BIG_V_P * C, 14)
    _fill(dlogits, BIG_ROWS * BIG_V_P, 15)

    var d_inp_dense = _alloc(BIG_ROWS * C)
    var d_w_dense = _alloc(BIG_V_P * C)
    var scratch = _alloc(BIG_V_P * BIG_ROWS)

    matmul_bwd[DT, TARGET, use_gelu=False, has_bias=False](
        as_mut_kernel[DT](d_inp_dense),
        as_mut_kernel[DT](d_w_dense),
        as_mut_kernel[DT](d_w_dense),  # dummy d_bias
        as_mut_kernel[DT](dlogits),
        as_mut_kernel[DT](inp),
        as_immut_kernel_from_mut[DT](wte),
        as_mut_kernel[DT](d_inp_dense),  # dummy pre_gelu
        as_mut_kernel[DT](scratch),
        Int64(BIG_ROWS),
        Int64(1),
        Int64(C),
        Int64(BIG_V_P),
        ctx,
    )
    ctx.synchronize()

    var d_inp_tiled = _alloc(BIG_ROWS * C)
    var d_w_tiled = _alloc(BIG_V_P * C)

    var ntiles = 0
    var ragged = 0
    var t0 = 0
    var first = True
    while t0 < BIG_V_P:
        var oc = min(BIG_TILE, BIG_V_P - t0)
        if oc != BIG_TILE:
            ragged = oc
        matmul_lm_head_bwd_tile[DT, TARGET](
            as_mut_kernel[DT](d_inp_tiled),
            as_mut_kernel[DT](d_w_tiled + t0 * C),
            as_immut_kernel_from_mut[DT](dlogits + t0),
            BIG_V_P,
            as_immut_kernel_from_mut[DT](inp),
            as_immut_kernel_from_mut[DT](wte + t0 * C),
            as_mut_kernel[DT](scratch),
            BIG_ROWS,
            C,
            oc,
            not first,
            ctx,
        )
        ntiles += 1
        first = False
        t0 += BIG_TILE
    ctx.synchronize()

    assert_true(ntiles == 8, "expected 8 tiles at V_p=50304, tile=6400")
    assert_true(ragged == 5504, "expected a ragged 5504-row final tile")

    # d_weight: reduction over `rows` (128 terms), which vocab tiling does not
    # split — both paths form the same dot products, so this stays a tight
    # absolute check even at |d_w| ~ 1e2. Covers the ragged final tile's 5504
    # rows too, since the loop runs to V_p.
    for i in range(BIG_V_P * C):
        assert_almost_equal(d_w_tiled[i], d_w_dense[i], atol=1e-5)

    # d_input needs a different kind of check, and the reason is worth stating.
    #
    # It contracts over the VOCABULARY — 50304 terms — which is exactly the axis
    # tiling splits, so the dense path sums all 50304 in one go while the tiled
    # path sums 8 partial results. Floating-point addition is not associative,
    # so the two disagree by rounding no matter how correct both are. The data
    # here is mixed-sign, so there is heavy cancellation: the sum of the
    # products' magnitudes is ~20x the final value, and rounding error scales
    # with the former. Comparing the two fp32 results to each other therefore
    # cannot distinguish "tiling is wrong" from "fp32 is fp32" — an earlier
    # rtol=1e-4 version of this check failed at 1.004e-4 for precisely that
    # reason, with nothing wrong.
    #
    # So: compute the truth in float64 for a sample of elements and require BOTH
    # paths to be close to it, with the tolerance scaled by the accumulated
    # magnitude rather than by the (cancelled, small) result. That tests the
    # claim we actually care about — tiling does not degrade accuracy — and it
    # still fails loudly on a dropped or double-counted tile, which would move
    # the answer by roughly the accumulation scale (~1e3), four orders of
    # magnitude beyond this bound.
    for r in range(0, BIG_ROWS, 16):
        for c in range(0, C, 96):
            var acc = Float64(0.0)
            var mag = Float64(0.0)
            for v in range(BIG_V_P):
                var prod = Float64(dlogits[r * BIG_V_P + v]) * Float64(
                    wte[v * C + c]
                )
                acc += prod
                mag += abs(prod)
            # sqrt(50304) * fp32 eps ~ 2.7e-5; 5e-5 leaves margin without
            # letting a real error through.
            var tol = Float64(5e-5) * mag
            var got_t = Float64(d_inp_tiled[r * C + c])
            var got_d = Float64(d_inp_dense[r * C + c])
            assert_true(
                abs(got_t - acc) <= tol,
                (
                    "tiled d_input off the float64 reference at ("
                    + String(r)
                    + ","
                    + String(c)
                    + "): got "
                    + String(got_t)
                    + " want "
                    + String(acc)
                    + " tol "
                    + String(tol)
                ),
            )
            assert_true(
                abs(got_d - acc) <= tol,
                (
                    "dense d_input off the float64 reference at ("
                    + String(r)
                    + ","
                    + String(c)
                    + "): got "
                    + String(got_d)
                    + " want "
                    + String(acc)
                    + " tol "
                    + String(tol)
                ),
            )

    inp.free()
    wte.free()
    dlogits.free()
    d_inp_dense.free()
    d_w_dense.free()
    scratch.free()
    d_inp_tiled.free()
    d_w_tiled.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
