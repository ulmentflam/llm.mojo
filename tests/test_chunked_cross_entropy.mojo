# ===----------------------------------------------------------------------=== #
# Pure-Mojo tests for the chunked (two-pass) cross-entropy in
# llmm/fused_classifier.mojo — `chunked_ce_pass1`, `chunked_ce_loss`,
# `chunked_ce_pass2`.
#
# Run with:  make test-mojo   (equivalent to `mojo run -I . tests/test_chunked_cross_entropy.mojo`)
#
# WHY THIS FILE EXISTS. The stock `fused_classifier` reads a whole `(B*T, V_p)`
# row to get that row's max and sum-exp, then overwrites the row in place with
# dlogits. At production shape that buffer is 6.1 GiB. The chunked path computes
# the identical numbers while holding one vocabulary tile at a time, by carrying
# the running (max, sum-exp) pair across tiles — the online-softmax recurrence.
#
# Three things can go wrong and essentially only three:
#   1. THE RESCALE. When a later tile raises the running max, the sum-exp
#      accumulated against the old max has to be rescaled. Get that wrong and
#      the loss is off by a factor of exp(max_gap), not by rounding.
#      `test_late_max_forces_the_rescale` puts the row maximum in the LAST tile.
#   2. THE PADDING. Columns [V, V_p) are garbage from the GEMM. They must not
#      enter the softmax reduction, and dlogits must be EXACTLY zero there so
#      the backward GEMMs read zeros. V_P > V here, deliberately, and the filler
#      writes a huge value into the padding so a leak is loud.
#   3. THE ONE-HOT. The target's indicator fires in exactly one tile. Targets
#      below sit in the first tile, middle tiles, on a tile boundary, and in the
#      ragged last tile.
#
# EVERY reference here is computed in float64 from the same fp32 inputs, never
# by comparing two fp32 implementations. Two fp32 paths agreeing proves less
# than either agreeing with an exact reference, and a tolerance chosen against
# another fp32 result is a tolerance chosen against that result's rounding.
#
# (`fused_classifier_cpu` would be the natural second reference and is
# deliberately NOT used: its row reduction issues SIMD-width-ALIGNED loads at
# `row * V_p`, which is only a legal address when V_p is a multiple of the SIMD
# width — 50304 is, the small V_p these tests need is not. Calling it here
# segfaults. That constraint is pre-existing and unreachable in the trainer, but
# it does mean it cannot serve as a small-shape oracle.)
#
# This runs on the CPU target. The GPU kernels are the same algorithm with a
# block reduction in place of the scalar one, and are covered end to end by the
# GPU gates.
# ===----------------------------------------------------------------------=== #

from std.testing import assert_almost_equal, assert_true, TestSuite

from std.math import ceildiv, exp, log
from max.gpu.host import DeviceContext

from llmm.memory import (
    MutMemPtr,
    as_mut_kernel,
    as_immut_kernel_from_mut,
    heap_alloc,
)
from llmm.fused_classifier import (
    chunked_ce_pass1,
    chunked_ce_loss,
    chunked_ce_pass2,
)

comptime DT = DType.float32
comptime TARGET = "cpu"

# Shapes. V_P is NOT a multiple of TILE, so the last tile is ragged; V < V_P, so
# there is real padding to force to zero; and the padding boundary falls INSIDE
# a tile, which is the production situation (V=50257 inside the tile that runs
# to V_p=50304).
comptime ROWS = 6
comptime V = 29
comptime V_P = 37
comptime TILE = 8


def _alloc(n: Int) -> MutMemPtr[DT]:
    var p = heap_alloc[Scalar[DT]](n)
    var q = rebind[MutMemPtr[DT]](p.as_unsafe_any_origin())
    for i in range(n):
        q[unsafe_offset=i] = 0.0
    return q


def _alloc_i32(n: Int) -> MutMemPtr[DType.int32]:
    var p = heap_alloc[Scalar[DType.int32]](n)
    var q = rebind[MutMemPtr[DType.int32]](p.as_unsafe_any_origin())
    for i in range(n):
        q[unsafe_offset=i] = 0
    return q


def _fill_logits(p: MutMemPtr[DT], seed: Int) -> None:
    """Deterministic, mixed-sign, spread over several units so the softmax is
    not degenerate. Padding columns get a LARGE value on purpose: if the
    reduction ever reads past V the loss changes by a lot, not a little."""
    for r in range(ROWS):
        for c in range(V_P):
            if c >= V:
                p[unsafe_offset=r * V_P + c] = 1000.0
            else:
                p[unsafe_offset=r * V_P + c] = (
                    Float32((r * 13 + c * 7 + seed * 5) % 19) - 9.0
                ) * 0.41


# ===----------------------------------------------------------------------=== #
# Reference: float64 log-sum-exp over the live columns of one row
# ===----------------------------------------------------------------------=== #


def _ref_max_sum_f64(
    logits: MutMemPtr[DT], row: Int
) -> Tuple[Float64, Float64]:
    """The row's (max, sum-exp) over its LIVE columns, in float64, single pass,
    max subtracted first. Nothing here is reassociated, so it is the yardstick
    the chunked fp32 path is measured against."""
    var m = Float64(-1.0e300)
    for c in range(V):
        var x = Float64(logits[unsafe_offset=row * V_P + c])
        if x > m:
            m = x
    var s = Float64(0.0)
    for c in range(V):
        s += exp(Float64(logits[unsafe_offset=row * V_P + c]) - m)
    return (m, s)


def _ref_loss_f64(logits: MutMemPtr[DT], row: Int, target: Int) -> Float64:
    var ms = _ref_max_sum_f64(logits, row)
    return (
        log(ms[1]) + ms[0] - Float64(logits[unsafe_offset=row * V_P + target])
    )


def _ref_dlogit_f64(
    logits: MutMemPtr[DT], row: Int, col: Int, target: Int, d_loss: Float64
) -> Float64:
    """`(softmax - onehot) * dloss` for one element, in float64. Padding columns
    are zero by definition, not by computation."""
    if col >= V:
        return Float64(0.0)
    var ms = _ref_max_sum_f64(logits, row)
    var p = exp(Float64(logits[unsafe_offset=row * V_P + col]) - ms[0]) / ms[1]
    var ind = Float64(1.0) if col == target else Float64(0.0)
    return (p - ind) * d_loss


# ===----------------------------------------------------------------------=== #
# The chunked driver: exactly what train_gpt2.mojo's two call sites do, minus
# the GEMMs (which tests/test_lm_head_vocab_tiling.mojo covers). Returns the
# number of tiles it actually took, so every caller can assert it really
# chunked.
# ===----------------------------------------------------------------------=== #


def _run_chunked(
    ctx: DeviceContext,
    logits: MutMemPtr[DT],
    targets: MutMemPtr[DType.int32],
    d_losses: MutMemPtr[DT],
    tile: Int,
    losses_out: MutMemPtr[DT],
    dlogits_out: MutMemPtr[DT],
) raises -> Int:
    var m = _alloc(ROWS)
    var s = _alloc(ROWS)
    var xt = _alloc(ROWS)
    # The ONE tile of logits the whole scheme is allowed to hold. Its width is
    # the tile width, never V_P — that is the entire point.
    var tile_buf = _alloc(ROWS * tile)
    var taken = 0

    # Pass 1: fold each tile into the running (max, sum-exp).
    var t0 = 0
    var first = True
    while t0 < V_P:
        var oc = min(tile, V_P - t0)
        for r in range(ROWS):
            for c in range(oc):
                tile_buf[unsafe_offset=r * oc + c] = logits[
                    unsafe_offset=r * V_P + t0 + c
                ]
        chunked_ce_pass1[DT, TARGET](
            as_immut_kernel_from_mut[DT](tile_buf),
            as_mut_kernel[DT](m),
            as_mut_kernel[DT](s),
            as_mut_kernel[DT](xt),
            as_immut_kernel_from_mut[DType.int32](targets),
            ROWS,
            oc,
            t0,
            max(min(oc, V - t0), 0),
            first,
            ctx,
        )
        first = False
        taken += 1
        t0 += tile

    chunked_ce_loss[TARGET](
        as_mut_kernel[DT](losses_out),
        as_immut_kernel_from_mut[DT](m),
        as_immut_kernel_from_mut[DT](s),
        as_immut_kernel_from_mut[DT](xt),
        ROWS,
        ctx,
    )

    # Pass 2: recompute each tile (here: re-copy it, since there is no GEMM in
    # this harness) and convert it to dlogits in place.
    t0 = 0
    while t0 < V_P:
        var oc = min(tile, V_P - t0)
        for r in range(ROWS):
            for c in range(oc):
                tile_buf[unsafe_offset=r * oc + c] = logits[
                    unsafe_offset=r * V_P + t0 + c
                ]
        chunked_ce_pass2[DT, TARGET](
            as_mut_kernel[DT](tile_buf),
            as_immut_kernel_from_mut[DT](m),
            as_immut_kernel_from_mut[DT](s),
            as_immut_kernel_from_mut[DT](d_losses),
            as_immut_kernel_from_mut[DType.int32](targets),
            ROWS,
            oc,
            t0,
            max(min(oc, V - t0), 0),
            ctx,
        )
        for r in range(ROWS):
            for c in range(oc):
                dlogits_out[unsafe_offset=r * V_P + t0 + c] = tile_buf[
                    unsafe_offset=r * oc + c
                ]
        t0 += tile

    m.unsafe_free()
    s.unsafe_free()
    xt.unsafe_free()
    tile_buf.unsafe_free()
    return taken


# ===----------------------------------------------------------------------=== #
# Geometry — the tests below must actually chunk
# ===----------------------------------------------------------------------=== #


def test_test_shapes_really_chunk() raises:
    # Gotcha guard: a suite that ran one tile would exercise none of the
    # recurrence. Assert the split at the shapes these tests use.
    assert_true(ceildiv(V_P, TILE) == 5, "expected 5 tiles at V_P=37, TILE=8")
    assert_true(V_P % TILE != 0, "expected a ragged final tile")
    assert_true(V < V_P, "expected real padding columns")
    # The live/padding boundary must land INSIDE a tile, not on a tile edge —
    # that is the production situation and the case a per-tile clamp gets wrong.
    assert_true(V % TILE != 0, "expected the V boundary inside a tile")
    assert_true(V // TILE == 3, "expected the V boundary in the 4th tile")


def test_production_geometry_chunks_and_pads() raises:
    # The shape the flag exists for: GPT-2 124M, V=50257 inside V_p=50304, at
    # the default K=8 (6400-row tiles). If this ever collapses to one tile the
    # chunking is a no-op while every other test still passes.
    var tile = 6400
    var v_p = 50304
    var v = 50257
    assert_true(ceildiv(v_p, tile) == 8, "expected 8 tiles at V_p=50304")
    # The last tile starts at 44800 and runs to 50304; V=50257 lands inside it,
    # so exactly one tile has a partial live width.
    var last_t0 = 7 * tile
    assert_true(last_t0 < v and v < v_p, "expected V inside the last tile")
    assert_true(
        v - last_t0 < v_p - last_t0,
        "expected the last tile to be partially padding",
    )


# ===----------------------------------------------------------------------=== #
# Loss
# ===----------------------------------------------------------------------=== #


def test_loss_matches_float64_log_sum_exp() raises:
    var ctx = DeviceContext(api="cpu")

    var logits = _alloc(ROWS * V_P)
    _fill_logits(logits, 3)
    var targets = _alloc_i32(ROWS)
    # Targets spread across tiles: first, on a tile boundary, middle, last live
    # column, and two arbitrary.
    var picks = [0, 8, 15, 17, 28, 22]
    for r in range(ROWS):
        targets[unsafe_offset=r] = Int32(picks[r])
    var d_losses = _alloc(ROWS)
    for r in range(ROWS):
        d_losses[unsafe_offset=r] = 0.125 + 0.03 * Float32(r)

    var losses = _alloc(ROWS)
    var dlogits = _alloc(ROWS * V_P)
    var taken = _run_chunked(
        ctx, logits, targets, d_losses, TILE, losses, dlogits
    )
    assert_true(taken == 5, "the chunked driver must take 5 tiles")

    for r in range(ROWS):
        # Loss here is ~3.4 and the terms are O(4), so there is no catastrophic
        # cancellation and a direct fp32-vs-float64 comparison is meaningful.
        assert_almost_equal(
            Float64(losses[unsafe_offset=r]),
            _ref_loss_f64(logits, r, Int(targets[unsafe_offset=r])),
            atol=1e-5,
        )

    logits.unsafe_free()
    targets.unsafe_free()
    d_losses.unsafe_free()
    losses.unsafe_free()
    dlogits.unsafe_free()


def test_late_max_forces_the_rescale() raises:
    # The rescale branch (`s * exp(m - m')`) only runs when a LATER tile raises
    # the running max. Plant the row maximum in the last live tile so the
    # earlier tiles accumulate against a stale, much smaller max and must be
    # corrected. A missing rescale inflates the loss by exp(spike - early_max),
    # i.e. by orders of magnitude, not by rounding.
    var ctx = DeviceContext(api="cpu")

    var logits = _alloc(ROWS * V_P)
    _fill_logits(logits, 11)
    var spike_col = 27  # last live tile is [24, 29)
    for r in range(ROWS):
        logits[unsafe_offset=r * V_P + spike_col] = 14.0 + Float32(r)
    var targets = _alloc_i32(ROWS)
    for r in range(ROWS):
        targets[unsafe_offset=r] = Int32(r)  # every target in the FIRST tile
    var d_losses = _alloc(ROWS)
    for r in range(ROWS):
        d_losses[unsafe_offset=r] = 0.25

    var losses = _alloc(ROWS)
    var dlogits = _alloc(ROWS * V_P)
    var taken = _run_chunked(
        ctx, logits, targets, d_losses, TILE, losses, dlogits
    )
    assert_true(taken > 1, "test must exercise more than one tile")
    for r in range(ROWS):
        assert_almost_equal(
            Float64(losses[unsafe_offset=r]),
            _ref_loss_f64(logits, r, Int(targets[unsafe_offset=r])),
            atol=1e-4,
        )

    logits.unsafe_free()
    targets.unsafe_free()
    d_losses.unsafe_free()
    losses.unsafe_free()
    dlogits.unsafe_free()


def test_early_max_needs_no_rescale() raises:
    # The mirror image: max in the FIRST tile, so every later merge takes the
    # `m_tile < m` side of the branch and must leave the running sum alone apart
    # from adding its own down-scaled contribution.
    var ctx = DeviceContext(api="cpu")

    var logits = _alloc(ROWS * V_P)
    _fill_logits(logits, 5)
    for r in range(ROWS):
        logits[unsafe_offset=r * V_P + 2] = 12.0 + 0.5 * Float32(r)
    var targets = _alloc_i32(ROWS)
    for r in range(ROWS):
        targets[unsafe_offset=r] = Int32(
            24 + r % 5
        )  # every target in the LAST live tile
    var d_losses = _alloc(ROWS)
    for r in range(ROWS):
        d_losses[unsafe_offset=r] = 1.0 / Float32(ROWS)

    var losses = _alloc(ROWS)
    var dlogits = _alloc(ROWS * V_P)
    var taken = _run_chunked(
        ctx, logits, targets, d_losses, TILE, losses, dlogits
    )
    assert_true(taken > 1, "test must exercise more than one tile")
    for r in range(ROWS):
        assert_almost_equal(
            Float64(losses[unsafe_offset=r]),
            _ref_loss_f64(logits, r, Int(targets[unsafe_offset=r])),
            atol=1e-5,
        )

    logits.unsafe_free()
    targets.unsafe_free()
    d_losses.unsafe_free()
    losses.unsafe_free()
    dlogits.unsafe_free()


# ===----------------------------------------------------------------------=== #
# dlogits
# ===----------------------------------------------------------------------=== #


def test_dlogits_match_float64_reference_everywhere() raises:
    var ctx = DeviceContext(api="cpu")

    var logits = _alloc(ROWS * V_P)
    _fill_logits(logits, 7)
    var targets = _alloc_i32(ROWS)
    var picks = [0, 8, 15, 24, 28, 9]
    for r in range(ROWS):
        targets[unsafe_offset=r] = Int32(picks[r])
    var d_losses = _alloc(ROWS)
    for r in range(ROWS):
        d_losses[unsafe_offset=r] = 0.07 * Float32(r + 1)

    var losses = _alloc(ROWS)
    var dlogits = _alloc(ROWS * V_P)
    var taken = _run_chunked(
        ctx, logits, targets, d_losses, TILE, losses, dlogits
    )
    assert_true(taken > 1, "test must exercise more than one tile")

    # dlogits are probabilities times a scale, so every element is O(d_loss) or
    # smaller and nothing cancels except at the target (p_t - 1). An absolute
    # tolerance against the float64 value is the right instrument.
    for r in range(ROWS):
        for c in range(V_P):
            assert_almost_equal(
                Float64(dlogits[unsafe_offset=r * V_P + c]),
                _ref_dlogit_f64(
                    logits,
                    r,
                    c,
                    Int(targets[unsafe_offset=r]),
                    Float64(d_losses[unsafe_offset=r]),
                ),
                atol=1e-6,
            )

    logits.unsafe_free()
    targets.unsafe_free()
    d_losses.unsafe_free()
    losses.unsafe_free()
    dlogits.unsafe_free()


def test_dlogits_padding_is_exactly_zero() raises:
    # Not "close to zero" — exactly zero. The backward GEMMs read the [V, V_p)
    # columns as real data.
    var ctx = DeviceContext(api="cpu")

    var logits = _alloc(ROWS * V_P)
    _fill_logits(logits, 2)
    var targets = _alloc_i32(ROWS)
    for r in range(ROWS):
        targets[unsafe_offset=r] = Int32(r * 4)
    var d_losses = _alloc(ROWS)
    for r in range(ROWS):
        d_losses[unsafe_offset=r] = 0.5

    var losses = _alloc(ROWS)
    var dlogits = _alloc(ROWS * V_P)
    var taken = _run_chunked(
        ctx, logits, targets, d_losses, TILE, losses, dlogits
    )
    assert_true(taken > 1, "test must exercise more than one tile")
    for r in range(ROWS):
        for c in range(V, V_P):
            assert_true(
                dlogits[unsafe_offset=r * V_P + c] == Float32(0.0),
                "dlogits must be exactly zero on the padding columns",
            )

    logits.unsafe_free()
    targets.unsafe_free()
    d_losses.unsafe_free()
    losses.unsafe_free()
    dlogits.unsafe_free()


def test_dlogits_sum_to_zero_per_row() raises:
    # Structural invariant, independent of any reference implementation:
    # sum_c (p_c - onehot_c) * dloss = (1 - 1) * dloss = 0. It catches a
    # mis-normalized sum-exp — the thing chunking is most likely to break —
    # even if both fp32 paths were wrong the same way.
    var ctx = DeviceContext(api="cpu")

    var logits = _alloc(ROWS * V_P)
    _fill_logits(logits, 13)
    var targets = _alloc_i32(ROWS)
    var picks = [3, 8, 16, 23, 24, 28]
    for r in range(ROWS):
        targets[unsafe_offset=r] = Int32(picks[r])
    var d_losses = _alloc(ROWS)
    for r in range(ROWS):
        d_losses[unsafe_offset=r] = 0.31

    var losses = _alloc(ROWS)
    var dlogits = _alloc(ROWS * V_P)
    var taken = _run_chunked(
        ctx, logits, targets, d_losses, TILE, losses, dlogits
    )
    assert_true(taken > 1, "test must exercise more than one tile")
    for r in range(ROWS):
        var acc = Float64(0.0)
        for c in range(V_P):
            acc += Float64(dlogits[unsafe_offset=r * V_P + c])
        assert_almost_equal(acc, Float64(0.0), atol=1e-6)

    logits.unsafe_free()
    targets.unsafe_free()
    d_losses.unsafe_free()
    losses.unsafe_free()
    dlogits.unsafe_free()


# ===----------------------------------------------------------------------=== #
# Chunk-geometry stress
# ===----------------------------------------------------------------------=== #


def test_tile_width_one_still_works() raises:
    # Degenerate chunking: 37 tiles of one column each. Every merge sees a
    # single-element tile, which is the harshest test of the recurrence's
    # seeding and of the padding handling.
    var ctx = DeviceContext(api="cpu")

    var logits = _alloc(ROWS * V_P)
    _fill_logits(logits, 17)
    var targets = _alloc_i32(ROWS)
    for r in range(ROWS):
        targets[unsafe_offset=r] = Int32((r * 5 + 1) % V)
    var d_losses = _alloc(ROWS)
    for r in range(ROWS):
        d_losses[unsafe_offset=r] = 0.19

    var losses = _alloc(ROWS)
    var dlogits = _alloc(ROWS * V_P)
    var taken = _run_chunked(ctx, logits, targets, d_losses, 1, losses, dlogits)
    assert_true(taken == V_P, "expected one tile per column")
    for r in range(ROWS):
        assert_almost_equal(
            Float64(losses[unsafe_offset=r]),
            _ref_loss_f64(logits, r, Int(targets[unsafe_offset=r])),
            atol=1e-5,
        )
        for c in range(V, V_P):
            assert_true(
                dlogits[unsafe_offset=r * V_P + c] == Float32(0.0),
                "padding must stay zero at tile width 1",
            )

    logits.unsafe_free()
    targets.unsafe_free()
    d_losses.unsafe_free()
    losses.unsafe_free()
    dlogits.unsafe_free()


def test_all_padding_tile_is_absorbed() raises:
    # With tile width 10 the tiles start at 0, 10, 20, 30 — and V = 29, so the
    # tile at 30 is ENTIRELY padding and its `valid_cols` is 0. The merge must
    # absorb it as a no-op rather than poisoning the running max with the
    # MIN_FINITE sentinel an empty reduction produces.
    var ctx = DeviceContext(api="cpu")

    var logits = _alloc(ROWS * V_P)
    _fill_logits(logits, 23)
    var targets = _alloc_i32(ROWS)
    for r in range(ROWS):
        targets[unsafe_offset=r] = Int32(r)
    var d_losses = _alloc(ROWS)
    for r in range(ROWS):
        d_losses[unsafe_offset=r] = 0.4

    var losses = _alloc(ROWS)
    var dlogits = _alloc(ROWS * V_P)
    var taken = _run_chunked(
        ctx, logits, targets, d_losses, 10, losses, dlogits
    )
    assert_true(taken == 4, "expected 4 tiles at width 10")
    assert_true(30 >= V, "the last tile must start past V for this test")
    for r in range(ROWS):
        assert_almost_equal(
            Float64(losses[unsafe_offset=r]),
            _ref_loss_f64(logits, r, Int(targets[unsafe_offset=r])),
            atol=1e-5,
        )
        for c in range(V, V_P):
            assert_true(
                dlogits[unsafe_offset=r * V_P + c] == Float32(0.0),
                "padding must stay zero across an all-padding tile",
            )

    logits.unsafe_free()
    targets.unsafe_free()
    d_losses.unsafe_free()
    losses.unsafe_free()
    dlogits.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
