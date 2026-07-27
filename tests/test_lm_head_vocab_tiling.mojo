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
from std.python import Python

from llmm.memory import MutMemPtr, as_mut_kernel, as_immut_kernel_from_mut
from llmm.matmul import (
    matmul_fwd,
    matmul_bwd,
    matmul_lm_head_fwd_tile,
    matmul_lm_head_bwd_tile,
)
from llmm.encoder import ENC_ROW_CHUNK_ROWS
from train_gpt2 import GPT2, GPT2_DTYPE, Parameters, LM_HEAD_VOCAB_TILES

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
# Real GPT2 pool-sizing methods.
#
# Every test above checks `_tile_rows`, THIS FILE's own reimplementation of
# `GPT2._lm_head_tile_rows`' rounding rule — it never calls the real method.
# Grep across tests/ for `_grad_pool_elems`, `_enc_grad_pool_elems`,
# `_lm_head_tile_rows`, `_enc_chunk_rows` finds exactly one hit outside
# train_gpt2.mojo, and it is a COMMENT (the one on `_tile_rows` above) saying
# a test "mirrors" the real rounding rule — not a call to it. So a change to
# the real formula (wrong alignment constant, a dropped `+ wpe`, ...) moves
# nothing in this file's existing checks; they would stay green while
# production silently under-sized a gradient pool. That is the same shape of
# gap that cost six published checkpoints when `_encoder_backward_row_sparse`
# wrote past an under-sized pool (dropped `+ wpe` in `_enc_grad_pool_elems`).
#
# So these tests build an actual `GPT2` instance and call the real methods
# directly. Channels is kept tiny (16) so the instance is cheap to construct
# on the CPU target (num_parameters ~808K, dominated by `wte` at
# padded_vocab_size * channels) — but `padded_vocab_size` is left at the real
# GPT-2 124M value (50304), so `_lm_head_tile_rows`'s 128-alignment rounding
# runs on the exact shape production hits, not some toy vocabulary that might
# accidentally sidestep the `v_p >= 128` branch.
# ===----------------------------------------------------------------------=== #

comptime POOL_GEOM_CKPT = "gpt2_tiny_pool_geometry.bin"

comptime POOL_M = 16  # max_seq_len
comptime POOL_VOCAB = 50257
comptime POOL_L = 1  # num_layer
comptime POOL_NH = 4  # num_heads
comptime POOL_C = 16  # channels — tiny, keeps every tensor cheap
comptime POOL_VP = 50304  # padded_vocab_size — the real GPT-2 124M value


def _write_pool_geometry_checkpoint() raises:
    """Write a from-scratch-shaped `.bin` checkpoint (magic/version/header,
    then flat weights) at `POOL_GEOM_CKPT`, small enough to write and load in
    a fraction of a second, but with the real production `padded_vocab_size`.
    Mirrors tests/test_zero_equivalence.mojo's `setup_test_files`.
    """
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # GPT2_MAGIC
    # llm.c's version convention: 3 => fp32 params, 5 => bf16 params. Mirrors
    # the exact check GPT2.__init__ makes against GPT2_DTYPE.
    var version = 5 if GPT2_DTYPE == DType.bfloat16 else 3
    header[1] = version
    header[2] = POOL_M
    header[3] = POOL_VOCAB
    header[4] = POOL_L
    header[5] = POOL_NH
    header[6] = POOL_C
    header[7] = POOL_VP

    # Element count per tensor mirrors GPT2._compute_param_sizes exactly, so
    # the file is exactly as long as GPT2.__init__ expects to read. The
    # values themselves don't matter to any test here (nothing reads weights,
    # only config/param_sizes/pool-sizing methods), so an all-zero buffer is
    # fine.
    var num_params = (
        POOL_VP * POOL_C  # wte
        + POOL_M * POOL_C  # wpe
        + POOL_L * POOL_C  # ln_1_gamma
        + POOL_L * POOL_C  # ln_1_beta
        + POOL_L * (3 * POOL_C) * POOL_C  # qkv_weight
        + POOL_L * (3 * POOL_C)  # qkv_bias
        + POOL_L * POOL_C * POOL_C  # attn_proj_weight
        + POOL_L * POOL_C  # attn_proj_bias
        + POOL_L * POOL_C  # ln_2_gamma
        + POOL_L * POOL_C  # ln_2_beta
        + POOL_L * (4 * POOL_C) * POOL_C  # fc_weight
        + POOL_L * (4 * POOL_C)  # fc_bias
        + POOL_L * POOL_C * (4 * POOL_C)  # proj_weight
        + POOL_L * POOL_C  # proj_bias
        + POOL_C  # ln_f_gamma
        + POOL_C  # ln_f_beta
    )

    var weights = np.zeros(num_params, dtype=np.float32)

    var f = builtins.open(POOL_GEOM_CKPT, "wb")
    _ = f.write(header.tobytes())
    _ = f.write(weights.tobytes())
    f.close()


def _cleanup_pool_geometry_checkpoint() raises:
    var os = Python.import_module("os")
    try:
        _ = os.remove(POOL_GEOM_CKPT)
    except:
        pass


def test_lm_head_tile_rows_matches_real_gpt2_method() raises:
    # Builds an actual GPT2 at the real padded_vocab_size and asserts
    # `GPT2._lm_head_tile_rows` (the production method) agrees with
    # `_tile_rows` (this file's independent reimplementation of the rounding
    # rule). A change to the real method's 128-alignment constant moves the
    # left side only, so the two would disagree and this fails; the geometry
    # tests earlier in this file compare `_tile_rows` to itself and could
    # never catch that.
    var ctx = DeviceContext(api="cpu")
    var model = GPT2["cpu", 1](POOL_GEOM_CKPT, rank=0, zero_stage=0, ctx=ctx)

    assert_true(
        model.config.padded_vocab_size == POOL_VP,
        "checkpoint round-trip lost padded_vocab_size",
    )

    var real_tr = model._lm_head_tile_rows()
    var local_tr = _tile_rows(POOL_VP, LM_HEAD_VOCAB_TILES)
    assert_true(
        real_tr == local_tr,
        (
            "GPT2._lm_head_tile_rows disagrees with the local rounding rule:"
            " real="
            + String(real_tr)
            + " local="
            + String(local_tr)
        ),
    )

    # Absolute anchor at the shipped default (K=8), so a simultaneous drift of
    # both `_tile_rows` and the real method (e.g. copy-pasting the same wrong
    # constant into both) can't pass the comparison above unnoticed.
    comptime if LM_HEAD_VOCAB_TILES == 8:
        assert_true(
            real_tr == 6400,
            "expected 6400-row tiles at V_p=50304, K=8 from the real method",
        )


def test_enc_grad_pool_elems_matches_real_gpt2_method() raises:
    # No test anywhere calls `GPT2._enc_grad_pool_elems` or
    # `GPT2._enc_chunk_rows` (grep confirms the only previous reference is a
    # comment). This builds a real GPT2 and drives BOTH branches of
    # `_enc_grad_pool_elems` against an expectation built from the real
    # `_enc_chunk_rows()` plus a raw data lookup (`param_sizes[wpe]`) — never
    # from a reimplementation of the `+ wpe` pool-size formula itself, so a
    # dropped `+ wpe` in the real method is the only way to make this fail.
    var ctx = DeviceContext(api="cpu")
    var model = GPT2["cpu", 1](POOL_GEOM_CKPT, rank=0, zero_stage=0, ctx=ctx)

    var wpe = model.param_sizes[Parameters.wpe]
    assert_true(wpe > 0, "wpe param size must not be trivially zero")

    # ---- branch 1: row-sparsity off. This is this instance's natural state
    # (WORLD_SIZE=1 so `_use_bucketing()` -> False -> `enc_row_sparse` =
    # False), the dense wte+wpe fallback.
    assert_true(
        not model.enc_row_sparse,
        "expected enc_row_sparse=False at WORLD_SIZE=1",
    )
    var expected_dense = model.param_sizes[Parameters.wte] + wpe
    assert_true(
        model._enc_grad_pool_elems() == expected_dense,
        (
            "dense (non-row-sparse) branch of _enc_grad_pool_elems dropped wte"
            " or wpe"
        ),
    )

    # ---- branch 2: row-sparsity on — the branch the real production bug
    # lived in (`_encoder_backward_row_sparse` writing past an under-sized
    # pool). `enc_row_sparse` is plain wiring set by `_use_bucketing()` in
    # `__init__` (WORLD_SIZE > 1 and zero_stage >= 2); the pool-size formula
    # itself doesn't care how it got set, so flipping it directly on the
    # already-built instance reaches the exact same `_enc_grad_pool_elems`
    # code path a real multi-rank ZeRO-2/3 run would take, without standing
    # up a CpuCoordinator simulation across ranks.
    model.enc_row_sparse = True
    var chunk = model._enc_chunk_rows()
    assert_true(chunk > 0, "_enc_chunk_rows must not be trivially zero")
    assert_true(
        chunk <= ENC_ROW_CHUNK_ROWS,
        "_enc_chunk_rows exceeded its own cap",
    )
    var expected_sparse = chunk * model.config.channels + wpe
    assert_true(
        model._enc_grad_pool_elems() == expected_sparse,
        (
            "row-sparse branch of _enc_grad_pool_elems dropped wpe — the bug"
            " that under-sized the encoder gradient pool and let"
            " _encoder_backward_row_sparse write past it on the first chunk"
        ),
    )


def test_grad_pool_elems_composes_real_submethods() raises:
    # `_grad_pool_elems` is the max of three real methods; compare it against
    # those same three methods called directly (never a reimplementation of
    # any of their formulas), so a change to how it composes them — not just
    # to one of the three formulas, which the tests above already cover — is
    # caught too.
    var ctx = DeviceContext(api="cpu")
    var model = GPT2["cpu", 1](POOL_GEOM_CKPT, rank=0, zero_stage=0, ctx=ctx)

    var lm_head = model._lm_head_tile_rows() * model.config.channels
    var per_layer = model._per_layer_pool_elems()
    var enc = model._enc_grad_pool_elems()
    var expected = max(lm_head, max(enc, per_layer))
    assert_true(
        model._grad_pool_elems() == expected,
        "_grad_pool_elems is no longer the max of its three real components",
    )


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

    # The check just above only proves `d_w_tiled` and `d_w_dense` agree with
    # EACH OTHER. Both arms compute d_weight through the exact same
    # subroutine (`matmul_bwd` and `matmul_lm_head_bwd_tile` both call
    # `matmul_d_weight_bwd`), and `_alloc` pre-zeroes both buffers — so a
    # `matmul_d_weight_bwd` that silently no-ops without writing would leave
    # both sides at 0.0 and the loop above would pass having proven nothing
    # about d_weight at all.
    #
    # Anchor a deterministic subset against a HOST reduction, independent of
    # any GPU kernel: the first and last vocab row of every one of the 8
    # tiles (0/6399, 6400/12799, ..., 44800/50303 — the ragged final tile),
    # plus one interior row of the ragged tail (47500), each against all 4
    # channels. Anchoring BOTH ends of every tile is what proves each tile's
    # write offset (`t0 * C`) is right, not just that some one row somewhere
    # is right — a bug that mis-offset only one tile's write would still pass
    # a check that sampled just the middle of that tile by luck, but not one
    # that always includes both of its edges.
    #
    # d_weight's reduction axis (rows, 128 terms) is NOT split by tiling, so
    # unlike d_input below this does not need a loose reassociation-aware
    # tolerance — but a tight absolute tolerance still leaves headroom for
    # the two paths visiting the 128 terms in a different order.
    var d_w_sample_rows = [
        0,
        6399,
        6400,
        12799,
        12800,
        19199,
        19200,
        25599,
        25600,
        31999,
        32000,
        38399,
        38400,
        44799,
        44800,
        47500,
        50303,
    ]
    var d_w_nonzero_samples = 0
    var d_w_total_samples = 0
    for si in range(len(d_w_sample_rows)):
        var v = d_w_sample_rows[si]
        for c in range(C):
            var acc = Float64(0.0)
            var mag = Float64(0.0)
            for r in range(BIG_ROWS):
                var prod = Float64(dlogits[r * BIG_V_P + v]) * Float64(
                    inp[r * C + c]
                )
                acc += prod
                mag += abs(prod)
            d_w_total_samples += 1
            if abs(acc) > 1e-6:
                d_w_nonzero_samples += 1
            # A tiny absolute floor (1e-6) on top of the magnitude-scaled term
            # covers samples where `mag` itself happens to be small; 5e-5 * mag
            # mirrors the d_input bound below (sqrt(128) * fp32 eps is smaller
            # still, so this leaves ample margin without hiding a real error).
            var tol = Float64(5e-5) * mag + Float64(1e-6)
            var got_t = Float64(d_w_tiled[v * C + c])
            var got_d = Float64(d_w_dense[v * C + c])
            assert_true(
                abs(got_t - acc) <= tol,
                (
                    "tiled d_weight off the float64 reference at (v="
                    + String(v)
                    + ", c="
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
                    "dense d_weight off the float64 reference at (v="
                    + String(v)
                    + ", c="
                    + String(c)
                    + "): got "
                    + String(got_d)
                    + " want "
                    + String(acc)
                    + " tol "
                    + String(tol)
                ),
            )
    # If this were near-universally zero the checks above would be almost as
    # tautological as comparing two zeroed buffers — require most sampled
    # (row, channel) pairs to be meaningfully non-zero.
    assert_true(
        d_w_nonzero_samples * 2 > d_w_total_samples,
        (
            "host d_weight reference is trivially zero across the sampled"
            " rows ("
            + String(d_w_nonzero_samples)
            + "/"
            + String(d_w_total_samples)
            + " non-zero) — this check cannot distinguish a correct kernel"
            " from a dead one"
        ),
    )

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


# ===----------------------------------------------------------------------=== #
# Multiple tile counts at vocabulary scale.
#
# The two tests above pin the DEFAULT tile width (6400 rows, K=8). That is what
# ships, but it validates exactly one point. Anyone characterising cost as a
# function of tile count — the memory saving rises with K while GEMM efficiency
# falls — would otherwise be reading a curve whose every other point has no
# correctness evidence behind it at all.
#
# So this sweeps the tile widths that -D LLMM_LM_HEAD_VOCAB_TILES=2,4,8,16
# actually produce at V_p=50304, and checks each against the same dense
# reference. All four are non-divisible, so every one exercises a ragged final
# tile of a different size:
#
#   K=2  -> tile 25216, 2 tiles,  tail 25088
#   K=4  -> tile 12672, 4 tiles,  tail 12288
#   K=8  -> tile  6400, 8 tiles,  tail  5504   (the default)
#   K=16 -> tile  3200, 16 tiles, tail  2304
#
# The dense reference and the float64 d_input reference are computed ONCE and
# reused across all four, which is what keeps this affordable.
# ===----------------------------------------------------------------------=== #


def test_multiple_tile_counts_at_vocab_scale() raises:
    var ctx = DeviceContext(api="cpu")

    var inp = _alloc(BIG_ROWS * C)
    var wte = _alloc(BIG_V_P * C)
    var dlogits = _alloc(BIG_ROWS * BIG_V_P)
    _fill(inp, BIG_ROWS * C, 21)
    _fill(wte, BIG_V_P * C, 22)
    _fill(dlogits, BIG_ROWS * BIG_V_P, 23)

    # ---- dense references, computed once ----
    var logits_dense = _alloc(BIG_ROWS * BIG_V_P)
    matmul_fwd[DT, TARGET, use_gelu=False, has_bias=False](
        as_mut_kernel[DT](logits_dense),
        as_mut_kernel[DT](logits_dense),
        as_mut_kernel[DT](inp),
        as_immut_kernel_from_mut[DT](wte),
        as_immut_kernel_from_mut[DT](wte),
        Int64(BIG_ROWS),
        Int64(1),
        Int64(C),
        Int64(BIG_V_P),
        ctx,
    )

    var d_inp_dense = _alloc(BIG_ROWS * C)
    var d_w_dense = _alloc(BIG_V_P * C)
    var scratch = _alloc(BIG_V_P * BIG_ROWS)
    matmul_bwd[DT, TARGET, use_gelu=False, has_bias=False](
        as_mut_kernel[DT](d_inp_dense),
        as_mut_kernel[DT](d_w_dense),
        as_mut_kernel[DT](d_w_dense),
        as_mut_kernel[DT](dlogits),
        as_mut_kernel[DT](inp),
        as_immut_kernel_from_mut[DT](wte),
        as_mut_kernel[DT](d_inp_dense),
        as_mut_kernel[DT](scratch),
        Int64(BIG_ROWS),
        Int64(1),
        Int64(C),
        Int64(BIG_V_P),
        ctx,
    )
    ctx.synchronize()

    var tile_widths = [25216, 12672, 6400, 3200]
    var want_tiles = [2, 4, 8, 16]
    var want_tail = [25088, 12288, 5504, 2304]

    var logits_tiled = _alloc(BIG_ROWS * BIG_V_P)
    var d_inp_tiled = _alloc(BIG_ROWS * C)
    var d_w_tiled = _alloc(BIG_V_P * C)

    for w in range(len(tile_widths)):
        var tw = tile_widths[w]
        var label = "tile width " + String(tw)

        # ---------- forward ----------
        for i in range(BIG_ROWS * BIG_V_P):
            logits_tiled[i] = 0.0
        var ntiles = 0
        var tail = 0
        var t0 = 0
        while t0 < BIG_V_P:
            var oc = min(tw, BIG_V_P - t0)
            if oc != tw:
                tail = oc
            matmul_lm_head_fwd_tile[DT, TARGET](
                as_mut_kernel[DT](logits_tiled + t0),
                BIG_V_P,
                as_immut_kernel_from_mut[DT](inp),
                as_immut_kernel_from_mut[DT](wte + t0 * C),
                BIG_ROWS,
                C,
                oc,
                ctx,
            )
            ntiles += 1
            t0 += tw
        ctx.synchronize()

        assert_true(ntiles == want_tiles[w], label + ": wrong tile count")
        assert_true(tail == want_tail[w], label + ": wrong ragged tail")

        for i in range(BIG_ROWS * BIG_V_P):
            assert_almost_equal(logits_tiled[i], logits_dense[i], atol=1e-5)

        # ---------- backward ----------
        for i in range(BIG_V_P * C):
            d_w_tiled[i] = 0.0
        for i in range(BIG_ROWS * C):
            d_inp_tiled[i] = 0.0
        t0 = 0
        var first = True
        while t0 < BIG_V_P:
            var oc = min(tw, BIG_V_P - t0)
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
            first = False
            t0 += tw
        ctx.synchronize()

        # d_weight does not reassociate at any tile width.
        for i in range(BIG_V_P * C):
            assert_almost_equal(d_w_tiled[i], d_w_dense[i], atol=1e-5)

        # As in test_bwd_tiled_matches_dense_at_vocab_scale, the check just
        # above proves only that d_w_tiled and d_w_dense agree with EACH
        # OTHER, which both being zeroed-out by a dead matmul_d_weight_bwd
        # would also satisfy. Anchor a small width-generic subset — row 0,
        # the last row of the first tile, the first row of the second tile
        # (both tile-boundary offsets, generic across `tw`), and the last
        # vocab row overall — against a host reduction, and require it to be
        # non-trivially non-zero, at every tile width in this sweep.
        var d_w_rows_w = [0, tw - 1, min(tw, BIG_V_P - 1), BIG_V_P - 1]
        var d_w_nonzero_w = 0
        var d_w_total_w = 0
        for si in range(len(d_w_rows_w)):
            var v = d_w_rows_w[si]
            for c in range(C):
                var acc = Float64(0.0)
                var mag = Float64(0.0)
                for r in range(BIG_ROWS):
                    var prod = Float64(dlogits[r * BIG_V_P + v]) * Float64(
                        inp[r * C + c]
                    )
                    acc += prod
                    mag += abs(prod)
                d_w_total_w += 1
                if abs(acc) > 1e-6:
                    d_w_nonzero_w += 1
                var tol = Float64(5e-5) * mag + Float64(1e-6)
                var got = Float64(d_w_tiled[v * C + c])
                assert_true(
                    abs(got - acc) <= tol,
                    label
                    + ": tiled d_weight off the float64 reference at (v="
                    + String(v)
                    + ", c="
                    + String(c)
                    + "): got "
                    + String(got)
                    + " want "
                    + String(acc)
                    + " tol "
                    + String(tol),
                )
        assert_true(
            d_w_nonzero_w * 2 > d_w_total_w,
            label
            + ": host d_weight reference is trivially zero across the"
            " sampled rows",
        )

        # d_input does reassociate, and MORE so at larger K (more partial sums),
        # so it is checked against float64 truth rather than against the dense
        # fp32 result. Same bound at every width: if accuracy degraded with tile
        # count this is where it would show.
        for r in range(0, BIG_ROWS, 32):
            for c in range(0, C, 192):
                var acc = Float64(0.0)
                var mag = Float64(0.0)
                for v in range(BIG_V_P):
                    var prod = Float64(dlogits[r * BIG_V_P + v]) * Float64(
                        wte[v * C + c]
                    )
                    acc += prod
                    mag += abs(prod)
                var tol = Float64(5e-5) * mag
                var got = Float64(d_inp_tiled[r * C + c])
                assert_true(
                    abs(got - acc) <= tol,
                    label
                    + ": tiled d_input off the float64 reference at ("
                    + String(r)
                    + ","
                    + String(c)
                    + "): got "
                    + String(got)
                    + " want "
                    + String(acc)
                    + " tol "
                    + String(tol),
                )

    inp.free()
    wte.free()
    dlogits.free()
    logits_dense.free()
    logits_tiled.free()
    d_inp_dense.free()
    d_w_dense.free()
    d_inp_tiled.free()
    d_w_tiled.free()
    scratch.free()


def main() raises:
    _write_pool_geometry_checkpoint()
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        _cleanup_pool_geometry_checkpoint()
        raise e^
    _cleanup_pool_geometry_checkpoint()
