"""NVFP4 quantize kernels — e2m1 data, e4m3 block scales, fp32 per-tensor
scale, matching the cuBLASLt/CUTLASS conventions verified in
`tests/probe_fp4/probe_fp4.cu` (see `tests/probe_fp4/RESULTS.md` and
`docs/ai/fp4_modular_support_research.md` §6).

## Format (NVFP4, per `docs/ai/fp4_training_recipes_research.md` §1)

- **Data**: e2m1 (1 sign / 2 exponent / 1 mantissa), OCP MX 8-level
  magnitude table `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. Two values packed per byte:
  even (lower) index -> low nibble, odd index -> high nibble — the exact
  convention cross-checked against PyTorch's `pack_uint4`
  (`torch/testing/_internal/common_quantized.py`) in the probe.
- **Block scale**: one e4m3 (`ue4m3`, byte-identical to standard fp8 e4m3 —
  `CUDA_R_8F_UE4M3 == CUDA_R_8F_E4M3` per `library_types.h`) scale per
  16-element block along K. Two granularities:
    - **1D 1x16** (`BLOCK_ROWS=1`): one block per 16 contiguous elements
      along K, independently for each row — for activations & gradients.
    - **2D 16x16** (`BLOCK_ROWS=16`): one block per 16 rows x 16 K-elements
      (256 elements share a scale) — for weights, so the same weight block
      quantizes identically read row-major (Fprop) or column-major (Dgrad).
- **Per-tensor fp32 scale** (second level): `tensor_scale = tensor_amax /
  (448 * 6)` (448 = e4m3 max magnitude, 6 = e2m1 max magnitude), so block
  scales are encoded as `block_amax / 6 / tensor_scale` before being
  narrowed to e4m3 — the block with the largest amax uses e4m3's full
  dynamic range. Falls back to `1.0` for an all-zero tensor.
- **Scale swizzle**: cuBLAS's 128-row x 4-col tile / 32x4x4-internal
  block-scale-factor layout (cuBLAS docs §3.1.4.3.2), ported verbatim from
  `tests/probe_fp4/probe_fp4.cu::swizzle_scales` (itself cross-checked there
  against PyTorch's `to_blocked()`/`from_blocked()`). `rows` in the swizzle
  functions below is always the *logical scale-tensor* row count (`M` for
  1D, `M/16` for 2D) — see `nvfp4_scale_swizzle_offset`.

## CRITICAL TOOLCHAIN FACT — no fp8/fp4 SIMD dtypes, ever

Mojo's `SIMD.cast` to `float8_e4m3fn`/`float8_e5m2` is broken on GPU targets,
and there is no `cast` to `float4_e2m1fn` at all. Every function in this
file that touches e2m1/e4m3 data operates on **plain `UInt8` byte buffers**
and does the bit manipulation by hand (fp32 arithmetic + integer bit ops on
`UInt32`/`UInt8`); no `Scalar[DType.float8_e4m3fn]` or
`Scalar[DType.float4_e2m1fn]` value is ever constructed. fp32/bf16
arithmetic and `bitcast` (a same-bit-width *reinterpret*, not a narrowing
`SIMD.cast`) are unaffected and used freely.

## Rounding

`encode_e2m1`/`encode_e4m3` implement round-to-nearest (ties-away-from-zero,
matching `roundf()`/the probe's nearest-candidate search — see their
docstrings for the exact tie-breaking rule). This is the `ROUND_MODE_RNE`
path.

`ROUND_MODE_STOCHASTIC` (`encode_e2m1` only — `encode_e4m3` block scales stay
RNE always, per the recipe) is wired on top of `llmm/rng_device.mojo`'s
Squares device RNG: given an already-drawn uniform `rand in [0,1)`,
`encode_e2m1` finds the two e2m1 grid points bracketing `|x|` and rounds up
to the higher one with probability exactly `(|x| - lo) / (hi - lo)` — the
standard stochastic-rounding-on-a-nonlinear-ladder construction, unbiased in
expectation (`E[decode(encode(x))] == x` exactly, by construction of the
interpolation) unlike `sr_round_bits`' bit-dither trick (which only works for
formats whose values are monotonic in raw bit pattern for a fixed exponent
range — not applicable to e2m1's 3-bit magnitude-index encoding). Exact grid
points (`p_up` is exactly 0 or the bracket is degenerate) round deterministically
to themselves regardless of the drawn `rand`, matching RNE's behavior there.

`_nvfp4_quantize_gpu`/`nvfp4_quantize` thread a `(seed, stream, step)` triple
through to `rng_uniform01` per the same convention `llmm/adamw.mojo`'s
`LLMM_SR_MASTER` seam uses: `counter = (step << 32) | flat_element_index`
(unique per (step, element), so repeated runs with the same seed are
bit-identical) and a `stream` id reserved for this module
(`NVFP4_SR_STREAM`, distinct from `llmm/adamw.mojo`'s `SR_MASTER_STREAM=1`)
so its random substream never collides with another SR call site sharing the
same seed. `round_mode` defaults to `ROUND_MODE_RNE` everywhere (existing
callers are unaffected); a caller opts into SR by passing
`round_mode=ROUND_MODE_STOCHASTIC` as `nvfp4_quantize`'s 4th template
parameter — selecting *which* operands (gradients only, per the recipe) is
a call-site decision, not something this module enforces.

## Stream registry

`llmm/matmul.mojo`'s `lowp_gemm_fp4` (forward) reserves
`NVFP4_SR_STREAM`/`NVFP4_SR_STREAM + 1` for its A/B operands. The backward
call sites (`matmul_d_input_bwd_fp4`/`matmul_d_weight_bwd_fp4`) each
quantize a `d_output` operand with SR (the recipe's "SR on the gradient
operand"; the paired weight/input operand stays RNE, which never touches the
RNG at all) — two MORE distinct streams so dgrad's and wgrad's `d_output`
dither never collide with each other or with the forward reservation, even
though both draw from the same tensor's values (different layouts: dgrad
quantizes `d_output` in its NATIVE orientation, wgrad quantizes it
TRANSPOSED — see `nvfp4_quantize_transpose` below):
`NVFP4_SR_STREAM_DGRAD_DOUTPUT = NVFP4_SR_STREAM + 2`,
`NVFP4_SR_STREAM_WGRAD_DOUTPUT = NVFP4_SR_STREAM + 3`.

## Transposed quantize

cuBLASLt's NVFP4 GEMM is TN-only (`_matmul_cublaslt_fp4`'s module comment,
mirroring fp8's `_matmul_cublaslt_fp8`), so Dgrad's `weight` operand and
Wgrad's `input`/`d_output` operands need a physically transposed NVFP4 copy
(their contraction dimension is not the trailing axis of their natural
row-major storage — see `llmm/matmul.mojo`'s dgrad/wgrad orientation
derivation). `nvfp4_quantize_transpose` below is the FP4 analogue of
`llmm/lowp.mojo`'s `quantize_transpose_devscale` (fp8): a fused pass reading
the SOURCE tensor transposed and writes a normally-laid-out (packed e2m1 +
swizzled e4m3 + fp32 tensor scale) NVFP4 quantization of that transpose — no
separate bf16 transpose scratch buffer/pass. It shares `_nvfp4_quantize_gpu`'s
kernel body via a comptime `TRANSPOSE` flag that only changes the SOURCE READ
address formula (`x_ptr[kidx * rows + r]` instead of `x_ptr[r * k + kidx]` —
i.e. read column `r` of the `[src_rows, src_k]` source instead of row `r` of
a `[rows, k]` source); every other step (block-amax reduction, e4m3 encode,
e2m1 encode/pack, swizzled scale write) is identical, since those operate on
the LOGICAL (already-transposed) element identity, not the source's physical
layout. The two-pass tensor-scale computation (`nvfp4_compute_tensor_scale`)
needs NO transpose-aware variant: amax over all elements is transpose-
invariant (it is a plain linear reduction over the flat buffer, independent
of which axis is "rows" vs "k").

## Scope

This module owns the quantize/dequantize primitives only; `llmm/matmul.mojo`
imports `nvfp4_quantize`/`nvfp4_quantize_transpose` directly at its FP4 GEMM
call sites (`lowp_gemm_fp4` and friends) rather than going through
`llmm/lowp.mojo`'s `PrecisionSpec`/`ScalingKind.Block2D` seam — see
`llmm/lowp.mojo`'s `FP4_SPEC` comment for why the spec-constant fields stay
descriptive-only.
"""

from std.collections import InlineArray
from std.math import ceildiv
from std.sys import get_defined_int
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_gpu
from std.gpu.primitives import block
from std.gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from std.gpu.memory import AddressSpace
from layout import Layout
from layout.layout_tensor import LayoutTensor

from llmm.memory import ImmutKernelPtr, MutKernelPtr, ImmutMemPtr, MutMemPtr
from llmm.lowp import (
    E4M3_MAX,
    NanPolicy,
    TieMode,
    _fp8_decode,
    _fp8_encode,
)
from llmm.rng_device import rng_uniform01


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime NVFP4_BLOCK = 16  # elements per scale, along K (both 1D and 2D)
comptime E2M1_MAX = Float32(6.0)  # largest representable e2m1 magnitude
# E4M3_MAX (448.0, largest representable e4m3 magnitude) is imported from
# llmm/lowp.mojo — one definition since the DRY pass F1.


# Rounding-mode seam for `encode_e2m1`. Both modes are implemented (see module
# docstring). Plain comptime Int constants (not a struct namespace) to avoid
# any question of whether a field-less struct is instantiable — matches this
# file's/the codebase's existing `comptime FOO = <literal>` convention (e.g.
# llmm/gelu.mojo's UNROLL/GELU_CONSTANT).
comptime ROUND_MODE_RNE = 0
comptime ROUND_MODE_STOCHASTIC = 1

# SR seed/stream defaults for `nvfp4_quantize`'s `ROUND_MODE_STOCHASTIC` path
# (llmm/rng_device.mojo's `(seed, counter, stream)` contract). Same
# `-D LLMM_SR_SEED=<int>` override and default literal as
# `llmm/adamw.mojo`'s `SR_MASTER_SEED` (one build-wide SR seed, deterministic
# by default) — the two call sites stay decorrelated via `stream`, not seed.
comptime NVFP4_SR_SEED = UInt64(get_defined_int["LLMM_SR_SEED", 1746221221]())
# `llmm/adamw.mojo` reserves stream=1 (`SR_MASTER_STREAM`). This module's
# base stream is 2; `llmm/matmul.mojo`'s `lowp_gemm_fp4` (which quantizes two
# operands, A and B, per GEMM call) uses `NVFP4_SR_STREAM` and
# `NVFP4_SR_STREAM + 1` respectively so the two operands' dither never draws
# from the same substream.
comptime NVFP4_SR_STREAM = UInt64(2)

# Backward stream reservations — see the module docstring's "Stream
# registry" section. `llmm/matmul.mojo`'s `matmul_d_input_bwd_fp4`
# (dgrad) and `matmul_d_weight_bwd_fp4` (wgrad) each quantize a `d_output`
# operand with SR; these two streams keep their dither substreams disjoint
# from the forward reservation (2, 3) and from each other.
comptime NVFP4_SR_STREAM_DGRAD_DOUTPUT = UInt64(4)
comptime NVFP4_SR_STREAM_WGRAD_DOUTPUT = UInt64(5)


# ===----------------------------------------------------------------------=== #
# e2m1 (NVFP4 data) encode / decode / pack
#
# 8-level magnitude ladder {0, 0.5, 1, 1.5, 2, 3, 4, 6} x sign.
# `encode_e2m1`'s tie-breaking (ax exactly on a midpoint) rounds toward the
# LOWER magnitude, matching tests/probe_fp4/probe_fp4.cu::encode_e2m1 (which
# does a nearest-candidate linear search that only updates its running best
# on a *strict* `<`, so an exact tie keeps the earlier/lower-index
# candidate). The `<=` ladder below reproduces that exactly: at each
# midpoint boundary the lower magnitude wins.
# ===----------------------------------------------------------------------=== #


@always_inline
def encode_e2m1[
    round_mode: Int = ROUND_MODE_RNE
](x: Float32, rand: Float32 = 0.0) -> UInt8:
    """Encode a fp32 value to a 4-bit e2m1 code (0..15: bit3=sign,
    bits2:0=magnitude index). Values are implicitly saturated to +/-6.0.

    `round_mode == ROUND_MODE_RNE` (default): round-to-nearest,
    ties-away-from-zero (`rand` unused). `round_mode ==
    ROUND_MODE_STOCHASTIC`: `rand` must be an already-drawn uniform value in
    `[0, 1)` (e.g. from `llmm.rng_device.rng_uniform01`) — the value rounds
    UP to the higher of the two bracketing grid points with probability
    exactly proportional to how close it sits to that neighbor, and DOWN
    otherwise; unbiased in expectation by construction (see module
    docstring's Rounding section). This function is pure/RNG-agnostic (mirrors
    `llmm/rng_device.mojo`'s `sr_round_bits` pattern) — callers own drawing
    `rand`.
    """
    var sign: UInt8 = UInt8(8) if x < 0.0 else UInt8(0)
    var ax = x if x >= 0.0 else -x
    var idx: UInt8
    comptime if round_mode == ROUND_MODE_RNE:
        if ax <= 0.25:
            idx = 0
        elif ax <= 0.75:
            idx = 1
        elif ax <= 1.25:
            idx = 2
        elif ax <= 1.75:
            idx = 3
        elif ax <= 2.5:
            idx = 4
        elif ax <= 3.5:
            idx = 5
        elif ax <= 5.0:
            idx = 6
        else:
            idx = 7
    elif round_mode == ROUND_MODE_STOCHASTIC:
        if ax >= 6.0:
            idx = 7
        else:
            var lo_idx: UInt8
            var lo: Float32
            var hi: Float32
            if ax < 0.5:
                lo_idx = 0
                lo = 0.0
                hi = 0.5
            elif ax < 1.0:
                lo_idx = 1
                lo = 0.5
                hi = 1.0
            elif ax < 1.5:
                lo_idx = 2
                lo = 1.0
                hi = 1.5
            elif ax < 2.0:
                lo_idx = 3
                lo = 1.5
                hi = 2.0
            elif ax < 3.0:
                lo_idx = 4
                lo = 2.0
                hi = 3.0
            elif ax < 4.0:
                lo_idx = 5
                lo = 3.0
                hi = 4.0
            else:
                lo_idx = 6
                lo = 4.0
                hi = 6.0
            # p_up = (ax - lo) / (hi - lo); round up with that probability.
            # At an exact grid point (ax == lo), p_up == 0.0 and
            # `rand < 0.0` is never true (rand in [0,1)), so exact grid
            # points are deterministic under SR too, same as RNE.
            var p_up = (ax - lo) / (hi - lo)
            idx = (lo_idx + 1) if rand < p_up else lo_idx
    else:
        comptime assert False, "encode_e2m1: unknown round_mode"
    return sign | idx


@always_inline
def decode_e2m1(code: UInt8) -> Float32:
    """Decode a 4-bit e2m1 code back to fp32."""
    var sign = (code >> 3) & UInt8(1)
    var idx = code & UInt8(7)
    var mag: Float32
    if idx == 0:
        mag = 0.0
    elif idx == 1:
        mag = 0.5
    elif idx == 2:
        mag = 1.0
    elif idx == 3:
        mag = 1.5
    elif idx == 4:
        mag = 2.0
    elif idx == 5:
        mag = 3.0
    elif idx == 6:
        mag = 4.0
    else:
        mag = 6.0
    return -mag if sign == UInt8(1) else mag


@always_inline
def pack_e2m1x2(lo: UInt8, hi: UInt8) -> UInt8:
    """Pack two 4-bit e2m1 codes into one byte: `lo` (the even/lower-K-index
    element) -> low nibble, `hi` (odd/higher-K-index) -> high nibble. Matches
    PyTorch's `pack_uint4` convention (see module docstring).
    """
    return (hi << 4) | (lo & UInt8(0xF))


@always_inline
def unpack_e2m1x2_lo(byte: UInt8) -> UInt8:
    return byte & UInt8(0xF)


@always_inline
def unpack_e2m1x2_hi(byte: UInt8) -> UInt8:
    return (byte >> 4) & UInt8(0xF)


# ===----------------------------------------------------------------------=== #
# e4m3 (block scale) encode / decode
#
# Standard fp8 e4m3 (1 sign / 4 exponent / 3 mantissa, bias 7, no infinities,
# 0x7F/0xFF reserved for NaN). These are thin wrappers over the shared codec
# core in `llmm/lowp.mojo`:
#
# - decode is bit-for-bit the same math both modules independently carried
#   (subnormal `m/512`, normal `(1 + m/8)*2^(e-7)` by direct exponent-field
#   construction) — unified outright into `_fp8_decode[4, 3, 7]`.
# - encode DELIBERATELY diverges from `llmm/lowp.mojo`'s own `encode_e4m3`:
#   this one is round-to-nearest ties-AWAY-from-zero with NaN -> 0x7F
#   (matching tests/probe_fp4/probe_fp4.cu's `roundf` reference and the
#   cuBLAS/PyTorch NVFP4 block-scale convention), while lowp's is
#   ties-to-EVEN with NaN saturating (matching Mojo's host fp8 cast). The
#   full divergence contract — including the probe-lineage -0.0 and
#   subnormal behaviors — is documented at the `TieMode` definition in
#   llmm/lowp.mojo; do NOT collapse the two rule sets.
#
# Block scales in this module are always >= 0, but sign is handled for
# generality. The `e_biased >= 15 -> m3 = 7` saturation bug a naive
# post-round exponent clamp is prone to (decodes to 240, not 448) is
# structurally avoided by the shared core: it clamps the input magnitude to
# max_normal (448, exactly representable) BEFORE rounding, so no post-round
# exponent-clamp branch exists to get wrong — see `_fp8_encode`'s carry-out
# comment and tests/test_nvfp4_quant.mojo::test_e4m3_saturation.
# ===----------------------------------------------------------------------=== #


# NOTE: deliberately different from llmm/lowp.mojo's encode_e4m3
# (ties-to-even, NaN-saturating) — this one is ties-away / NaN-emitting to
# match the cuBLAS/PyTorch NVFP4 block-scale convention. See lowp.mojo's
# TieMode block.
@always_inline
def encode_e4m3(x: Float32) -> UInt8:
    """Encode a fp32 value to an e4m3 byte, round-to-nearest
    (ties-away-from-zero), saturating to +/-448, NaN -> 0x7F — the
    probe/cuBLAS NVFP4 block-scale convention (see the module comment above
    and the TieMode contract in llmm/lowp.mojo).
    """
    return _fp8_encode[
        4, 3, 7, E4M3_MAX, tie_mode=TieMode.AWAY, nan_policy=NanPolicy.EMIT
    ](x)


@always_inline
def decode_e4m3(code: UInt8) -> Float32:
    """Decode an e4m3 byte back to fp32 (shared core — llmm/lowp.mojo)."""
    return _fp8_decode[4, 3, 7](code)


# ===----------------------------------------------------------------------=== #
# Block-scale swizzle (cuBLAS 128x4-tile / 32x4x4-internal layout)
#
# Ported line-for-line from tests/probe_fp4/probe_fp4.cu::swizzle_scales.
# Pure Int arithmetic — no GPU-target hazard, safe to call from host or
# device code, CPU or GPU compilation target.
# ===----------------------------------------------------------------------=== #


@always_inline
def nvfp4_swizzled_scale_buffer_size(rows: Int, cols: Int) -> Int:
    """Byte size of the swizzled scale buffer for a logical `rows x cols`
    (unswizzled) scale-factor matrix.
    """
    var n_row_tiles = ceildiv(rows, 128)
    var n_col_tiles = ceildiv(cols, 4)
    return 32 * n_row_tiles * 16 * n_col_tiles


@always_inline
def nvfp4_scale_swizzle_offset(row: Int, col: Int, n_col_tiles: Int) -> Int:
    """Byte offset in the swizzled scale buffer for logical scale-tensor
    position `(row, col)`. `n_col_tiles = ceildiv(cols, 4)` must be
    precomputed by the caller (shared across all calls for one tensor).
    """
    var scale_tile_h = row // 128
    var scale_tile_w = col // 4
    var tile_offset = 512 * (scale_tile_h * n_col_tiles + scale_tile_w)
    var outer = row % 128
    var inner = col % 4
    return tile_offset + (outer % 32) * 16 + (outer // 32) * 4 + inner


# ===----------------------------------------------------------------------=== #
# Shape helpers (host-side, plain Int math)
# ===----------------------------------------------------------------------=== #


@always_inline
def nvfp4_packed_size(rows: Int, k: Int) -> Int:
    """Byte size of the packed e2m1 data buffer for a `[rows, k]` tensor
    (2 elements/byte).
    """
    return (rows * k) // 2


@always_inline
def nvfp4_scale_rows(rows: Int, BLOCK_ROWS: Int) -> Int:
    """Number of `BLOCK_ROWS`-tall row-TILES spanning `rows` physical rows
    (`ceildiv(rows, BLOCK_ROWS)`) — i.e. how many distinct scale VALUES get
    computed, not the physical scale-BUFFER row count (see
    `nvfp4_scale_buffer_size`'s docstring: the physical buffer always has
    one entry per row, `BLOCK_ROWS > 1` only controls how many consecutive
    physical rows share the same computed value). Used to size the
    amax-computation thread grid in `_nvfp4_quantize_gpu`/`nvfp4_quantize`
    and to iterate tiles in `nvfp4_dequant_reference`.
    """
    return ceildiv(rows, BLOCK_ROWS)


@always_inline
def nvfp4_scale_buffer_size(rows: Int, k: Int, BLOCK_ROWS: Int) -> Int:
    """Byte size of the swizzled scale buffer for a `[rows, k]` tensor.

    Always sized for `rows` physical scale-buffer entries along the row
    axis, REGARDLESS of `BLOCK_ROWS` (kept as a parameter for call-site
    symmetry with `nvfp4_quantize`/`nvfp4_dequant_reference`, not because it
    changes this formula). cuBLASLt's `CUBLASLT_MATMUL_MATRIX_SCALE_
    VEC16_UE4M3` scale mode is inherently a 1-row-per-16-K-element-block
    physical layout ("the scaling factor is stored for each 16-element
    block in the innermost dimension of the corresponding data tensor" —
    `cublasLtMatmulMatrixScale_t` docstring, `_cublas/cublaslt.mojo`); there
    is no native cuBLASLt notion of a physically-compressed `rows/16`-row
    scale tensor. `BLOCK_ROWS=16` ("2D 16x16", weights) is achieved by
    computing ONE shared e4m3 scale value per 16-row-by-16-column tile and
    REPLICATING that value across all 16 physical scale-buffer rows the
    tile covers (`_nvfp4_quantize_gpu`), not by shrinking the buffer.

    An earlier version of this function returned a `rows/BLOCK_ROWS`-sized
    buffer for the 2D case, which is wrong for cuBLASLt consumption (though
    self-consistent with — and therefore undetected by — this file's own
    `nvfp4_dequant_reference`, which merely mirrors whatever layout the
    quantize kernel wrote): caught by
    `tests/test_lowp_gemm_fp4.mojo`'s real-cuBLASLt MLP-shape GEMM tests
    (`b_block_rows=16`), which produced NaN outputs — a pure-software
    roundtrip test using the SAME (buggy) convention on both sides could
    never have caught this; only feeding the buffer to the actual vendor
    GEMM call could.
    """
    var k_blocks = k // NVFP4_BLOCK
    return nvfp4_swizzled_scale_buffer_size(rows, k_blocks)


# ===----------------------------------------------------------------------=== #
# GPU: per-tensor fp32 scale (two-pass amax reduction)
# ===----------------------------------------------------------------------=== #


@always_inline
def _nvfp4_amax_partial_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
](
    out_ptr: MutKernelPtr[DType.float32],
    x_ptr: ImmutKernelPtr[dtype],
    n: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)
    var local_max = Float32(0.0)
    var i = idx
    while i < n:
        var v = x_ptr[i].cast[DType.float32]()
        var av = v if v >= 0.0 else -v
        if av > local_max:
            local_max = av
        i += grid_stride
    var block_max = block.max[block_size=BLOCK_SIZE](local_max)
    if Int(thread_idx.x) == 0:
        out_ptr[Int(block_idx.x)] = block_max


@always_inline
def _nvfp4_amax_aggregate_gpu[
    BLOCK_SIZE: Int,
](
    scale_out_ptr: MutKernelPtr[DType.float32],
    partial_ptr: ImmutKernelPtr[DType.float32],
    grid_size: Int,
) -> None:
    var tid = Int(thread_idx.x)
    var local_max = Float32(0.0)
    var idx = tid
    while idx < grid_size:
        var v = partial_ptr[idx]
        if v > local_max:
            local_max = v
        idx += BLOCK_SIZE
    var total_max = block.max[block_size=BLOCK_SIZE](local_max)
    if tid == 0:
        if total_max > 0.0:
            scale_out_ptr[0] = total_max / (E4M3_MAX * E2M1_MAX)
        else:
            scale_out_ptr[0] = 1.0


def nvfp4_compute_tensor_scale[
    dtype: DType,
    target: StaticString,
](
    scale_out_ptr: MutKernelPtr[DType.float32],  # 1 element
    x_ptr: ImmutKernelPtr[dtype],
    n: Int,
    ctx: DeviceContext,
) raises -> None:
    """Computes the NVFP4 per-tensor fp32 scale
    `tensor_amax / (448 * 6)` (or `1.0` for an all-zero tensor) into
    `scale_out_ptr[0]`, fully device-resident (no host round-trip).
    """
    comptime if is_gpu[target]():
        var device_ctx = ctx
        comptime BLOCK_SIZE = 256
        var num_blocks = ceildiv(n, BLOCK_SIZE) if n > 0 else 1
        var partial = device_ctx.enqueue_create_buffer[DType.float32](
            num_blocks
        )

        comptime partial_kernel = _nvfp4_amax_partial_gpu[dtype, BLOCK_SIZE]
        var compiled_partial = device_ctx.compile_function[partial_kernel]()
        device_ctx.enqueue_function(
            compiled_partial,
            partial.unsafe_ptr(),
            x_ptr,
            n,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )

        comptime agg_kernel = _nvfp4_amax_aggregate_gpu[BLOCK_SIZE]
        var compiled_agg = device_ctx.compile_function[agg_kernel]()
        device_ctx.enqueue_function(
            compiled_agg,
            scale_out_ptr,
            partial.unsafe_ptr(),
            num_blocks,
            grid_dim=(1,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("nvfp4_compute_tensor_scale is GPU-only")


# ===----------------------------------------------------------------------=== #
# GPU: quantize kernel (1D 1x16 when BLOCK_ROWS=1, 2D 16x16 when
# BLOCK_ROWS=16)
# ===----------------------------------------------------------------------=== #


@always_inline
def _nvfp4_src_addr[
    TRANSPOSE: Bool
](r: Int, kidx: Int, rows: Int, k: Int) -> Int:
    """Source-buffer flat address for logical element `(r, kidx)` of a
    `[rows, k]` tensor. `TRANSPOSE=False`: the source IS that `[rows, k]`
    row-major tensor (`r * k + kidx`, the ordinary case). `TRANSPOSE=True`:
    the source is instead a `[k, rows]` row-major tensor (physically `x_ptr`
    has `rows` as its TRAILING/contiguous axis) and `(r, kidx)` addresses its
    LOGICAL TRANSPOSE — i.e. `x_ptr[kidx, r]` in the source's own coordinates,
    `kidx * rows + r` flat (see `nvfp4_quantize_transpose`'s docstring for the
    full derivation). Every other part of `_nvfp4_quantize_gpu` (block-amax
    reduction, e4m3/e2m1 encode, swizzled scale write, packed-byte write) is
    unaffected by `TRANSPOSE` — only this address formula changes, since the
    rest operates on the LOGICAL `(r, kidx)` element identity, not the
    source's physical layout.
    """
    comptime if TRANSPOSE:
        return kidx * rows + r
    else:
        return r * k + kidx


@always_inline
def _nvfp4_quantize_gpu[
    dtype: DType,
    BLOCK_ROWS: Int,
    round_mode: Int,
    TRANSPOSE: Bool = False,
](
    q_ptr: MutKernelPtr[DType.uint8],
    scale_ptr: MutKernelPtr[DType.uint8],
    x_ptr: ImmutKernelPtr[dtype],
    tensor_scale_ptr: MutKernelPtr[DType.float32],
    rows: Int,
    k: Int,
    tile_rows: Int,  # ceildiv(rows, BLOCK_ROWS) -- row-TILE count, not the
    # physical scale-buffer row count (see nvfp4_scale_buffer_size).
    k_blocks: Int,
    n_col_tiles: Int,
    sr_seed: UInt64,
    sr_stream: UInt64,
    sr_step: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = tile_rows * k_blocks
    if idx >= total:
        return
    var br = idx // k_blocks
    var kb = idx % k_blocks
    var tensor_scale = tensor_scale_ptr[0]

    # 1. block amax over the BLOCK_ROWS x 16 tile.
    var amax = Float32(0.0)
    for rr in range(BLOCK_ROWS):
        var r = br * BLOCK_ROWS + rr
        if r >= rows:
            continue
        for kk in range(NVFP4_BLOCK):
            var kidx = kb * NVFP4_BLOCK + kk
            if kidx >= k:
                continue
            var v = x_ptr[_nvfp4_src_addr[TRANSPOSE](r, kidx, rows, k)].cast[
                DType.float32
            ]()
            var av = v if v >= 0.0 else -v
            if av > amax:
                amax = av

    # 2. block scale: encode (amax/6)/tensor_scale to e4m3; the *decoded*
    # value (block_e4m3 * tensor_scale) is what actually scales the data,
    # matching the two-level NVFP4 scheme (module docstring).
    var block_scale_raw: Float32
    if amax <= 0.0 or tensor_scale <= 0.0:
        block_scale_raw = 1.0
    else:
        block_scale_raw = (amax / E2M1_MAX) / tensor_scale
    var sc_code = encode_e4m3(block_scale_raw)
    var sc_val = decode_e4m3(sc_code) * tensor_scale
    if sc_val <= 0.0:
        sc_val = 1.0

    # Write the SAME e4m3 code to every physical row this tile covers.
    # cuBLASLt's VEC16_UE4M3 scale mode is inherently a 1-scale-per-physical
    # -row layout (nvfp4_scale_buffer_size's docstring); "2D 16x16"
    # (BLOCK_ROWS=16) means BLOCK_ROWS consecutive physical rows share one
    # computed value, not that the physical buffer shrinks to rows/16
    # entries. For BLOCK_ROWS=1 this loop runs once (rr=0, r=br) and is a
    # no-op change from the single-offset write it replaces.
    for rr in range(BLOCK_ROWS):
        var r = br * BLOCK_ROWS + rr
        if r >= rows:
            continue
        var soff = nvfp4_scale_swizzle_offset(r, kb, n_col_tiles)
        scale_ptr[soff] = sc_code

    # 3. quantize + pack elements. k is always a multiple of NVFP4_BLOCK
    # (enforced by the host wrapper), hence even, so r*k is always even and
    # (r*k+k0) parity == k0 parity == 0 for every even kk -- the pack byte
    # index below is always exact (no odd-row/odd-k misalignment).
    for rr in range(BLOCK_ROWS):
        var r = br * BLOCK_ROWS + rr
        if r >= rows:
            continue
        var kk = 0
        while kk < NVFP4_BLOCK:
            var k0 = kb * NVFP4_BLOCK + kk
            if k0 >= k:
                break
            var v0 = (
                x_ptr[_nvfp4_src_addr[TRANSPOSE](r, k0, rows, k)].cast[
                    DType.float32
                ]()
                / sc_val
            )
            var c0: UInt8
            comptime if round_mode == ROUND_MODE_STOCHASTIC:
                var counter0 = (UInt64(sr_step) << 32) | UInt64(r * k + k0)
                var rand0 = rng_uniform01(sr_seed, counter0, sr_stream)
                c0 = encode_e2m1[round_mode](v0, rand0)
            else:
                c0 = encode_e2m1[round_mode](v0)
            var c1 = UInt8(0)
            var k1 = k0 + 1
            if k1 < k and kk + 1 < NVFP4_BLOCK:
                var v1 = (
                    x_ptr[_nvfp4_src_addr[TRANSPOSE](r, k1, rows, k)].cast[
                        DType.float32
                    ]()
                    / sc_val
                )
                comptime if round_mode == ROUND_MODE_STOCHASTIC:
                    var counter1 = (UInt64(sr_step) << 32) | UInt64(r * k + k1)
                    var rand1 = rng_uniform01(sr_seed, counter1, sr_stream)
                    c1 = encode_e2m1[round_mode](v1, rand1)
                else:
                    c1 = encode_e2m1[round_mode](v1)
            q_ptr[(r * k + k0) // 2] = pack_e2m1x2(c0, c1)
            kk += 2


@always_inline
def _nvfp4_quantize_transpose_coalesced_gpu[
    dtype: DType,
    BLOCK_ROWS: Int,
    round_mode: Int,
    BLOCK_SIZE: Int,
](
    q_ptr: MutKernelPtr[DType.uint8],
    scale_ptr: MutKernelPtr[DType.uint8],
    x_ptr: ImmutKernelPtr[dtype],  # physical [k, rows] row-major SOURCE
    tensor_scale_ptr: MutKernelPtr[DType.float32],
    rows: Int,  # LOGICAL output rows == source's trailing/contiguous axis
    k: Int,  # LOGICAL output k == source's leading axis
    n_col_tiles: Int,
    sr_seed: UInt64,
    sr_stream: UInt64,
    sr_step: Int,
) -> None:
    """Coalesced-read counterpart of `_nvfp4_quantize_gpu[TRANSPOSE=True]`,
    whose naive read pattern is ~3x slower than the natural
    `TRANSPOSE=False` kernel: `_nvfp4_src_addr[TRANSPOSE=True]`'s read
    address `kidx * rows + r` has `kidx` varying fastest within a thread —
    strided by `rows` elements per step — while the natural path's
    `r * k + kidx` walks contiguous memory.

    Thread assignment: one thread per LOGICAL row `r` within a
    `BLOCK_SIZE`-row slab, one k-block `kb` per `block_idx.x`
    (`grid = (k_blocks, ceildiv(rows, BLOCK_SIZE))`). Each thread loads its
    16 elements `x_ptr[kidx * rows + r]`, `kidx = kb*16 .. kb*16+15`, into
    registers; at each fixed `kidx` step, consecutive threads (consecutive
    `r`) read CONSECUTIVE source addresses — the transposed read becomes a
    sequence of 16 fully-coalesced warp transactions instead of 16 fully-
    strided ones. All later steps read from those registers:

    - `BLOCK_ROWS == 1`: the thread's own 16 registers ARE its scale block —
      purely thread-local amax, no cross-thread communication at all.
    - `BLOCK_ROWS == 16` (2D weights): the group amax spans 16 consecutive
      rows = 16 consecutive threads; exchanged through a small per-thread-
      amax shared-memory array (one barrier), then every thread of the group
      reduces the same 16 values in the same (ascending-row) order.

    Bit-identity with `_nvfp4_quantize_gpu[TRANSPOSE=True]`: max is an
    order-invariant reduction (no floating-point-associativity hazard,
    unlike a sum), the encode math (`encode_e4m3`/`encode_e2m1`) and the SR
    counter formula (`(sr_step << 32) | (r*k + kidx)`, LOGICAL indices) are
    byte-for-byte the same, and the packed-nibble/swizzled-scale writes are
    the unchanged per-row formulas (`(r*k + k0)//2`,
    `nvfp4_scale_swizzle_offset(r, kb, ...)` — one scale write per physical
    row, which for BLOCK_ROWS=16 reproduces the replicated-value-per-row
    layout the non-tiled kernel produces with its per-tile `rr` loop; see
    `nvfp4_scale_buffer_size`'s docstring). The write pattern is identical
    to the natural kernel's; only the READ pattern changes — verified
    byte-exact by `tests/test_nvfp4_quant.mojo::
    test_quantize_transpose_matches_materialized_transpose_gpu`.

    A 32x32 SMEM-tile variant is slower at BLOCK_ROWS=16 (only 4/256 threads
    survive to quantize); this register-per-row design keeps all threads
    active.
    """
    var kb = Int(block_idx.x)
    var r = Int(block_idx.y) * BLOCK_SIZE + Int(thread_idx.x)
    var in_bounds = r < rows
    var k0 = kb * NVFP4_BLOCK

    # Per-thread-amax exchange buffer for the BLOCK_ROWS=16 group reduction.
    var amax_sh = LayoutTensor[
        DType.float32,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # 1. Coalesced register load + thread-local amax. `k` is always a
    # multiple of 16 (host wrapper enforces it), so a full k-block is
    # guaranteed in-bounds along kidx; only `r` needs guarding. Out-of-
    # bounds threads load zeros (amax 0.0) and must NOT return before the
    # barrier below.
    var vals = InlineArray[Float32, NVFP4_BLOCK](uninitialized=True)
    var local_amax = Float32(0.0)
    for kk in range(NVFP4_BLOCK):
        var v = Float32(0.0)
        if in_bounds:
            v = x_ptr[(k0 + kk) * rows + r].cast[DType.float32]()
        vals[kk] = v
        var av = v if v >= 0.0 else -v
        if av > local_amax:
            local_amax = av

    # 2. Group amax. BLOCK_ROWS=1: the thread-local value already IS the
    # block amax. BLOCK_ROWS=16: exchange through shared memory (BLOCK_SIZE
    # is a multiple of 16, so a 16-row group never straddles a slab
    # boundary; out-of-bounds rows contribute 0.0, matching the non-tiled
    # kernel's `r >= rows: continue` skip).
    var amax: Float32
    comptime if BLOCK_ROWS == 1:
        amax = local_amax
    else:
        amax_sh.ptr[Int(thread_idx.x)] = local_amax
        barrier()
        var g0 = (Int(thread_idx.x) // BLOCK_ROWS) * BLOCK_ROWS
        var m = Float32(0.0)
        for i in range(BLOCK_ROWS):
            var v = amax_sh.ptr[g0 + i]
            if v > m:
                m = v
        amax = m

    if not in_bounds:
        return

    var tensor_scale = tensor_scale_ptr[0]

    # 3. block scale (identical formula to `_nvfp4_quantize_gpu`).
    var block_scale_raw: Float32
    if amax <= 0.0 or tensor_scale <= 0.0:
        block_scale_raw = 1.0
    else:
        block_scale_raw = (amax / E2M1_MAX) / tensor_scale
    var sc_code = encode_e4m3(block_scale_raw)
    var sc_val = decode_e4m3(sc_code) * tensor_scale
    if sc_val <= 0.0:
        sc_val = 1.0
    scale_ptr[nvfp4_scale_swizzle_offset(r, kb, n_col_tiles)] = sc_code

    # 4. quantize + pack from registers (same per-element math and packed-
    # byte addresses as `_nvfp4_quantize_gpu`).
    var kk = 0
    while kk < NVFP4_BLOCK:
        var k0_idx = k0 + kk
        var v0 = vals[kk] / sc_val
        var c0: UInt8
        comptime if round_mode == ROUND_MODE_STOCHASTIC:
            var counter0 = (UInt64(sr_step) << 32) | UInt64(r * k + k0_idx)
            var rand0 = rng_uniform01(sr_seed, counter0, sr_stream)
            c0 = encode_e2m1[round_mode](v0, rand0)
        else:
            c0 = encode_e2m1[round_mode](v0)
        var v1 = vals[kk + 1] / sc_val
        var c1: UInt8
        comptime if round_mode == ROUND_MODE_STOCHASTIC:
            var counter1 = (UInt64(sr_step) << 32) | UInt64(r * k + k0_idx + 1)
            var rand1 = rng_uniform01(sr_seed, counter1, sr_stream)
            c1 = encode_e2m1[round_mode](v1, rand1)
        else:
            c1 = encode_e2m1[round_mode](v1)
        q_ptr[(r * k + k0_idx) // 2] = pack_e2m1x2(c0, c1)
        kk += 2


def _nvfp4_quantize_impl[
    dtype: DType,
    target: StaticString,
    BLOCK_ROWS: Int,
    round_mode: Int,
    TRANSPOSE: Bool,
](
    q_ptr: MutKernelPtr[DType.uint8],
    scale_ptr: MutKernelPtr[DType.uint8],
    tensor_scale_ptr: MutKernelPtr[DType.float32],
    x_ptr: ImmutKernelPtr[dtype],
    rows: Int,  # LOGICAL output rows (== src_k when TRANSPOSE)
    k: Int,  # LOGICAL output k (== src_rows when TRANSPOSE)
    ctx: DeviceContext,
    sr_seed: UInt64,
    sr_stream: UInt64,
    sr_step: Int,
) raises -> None:
    """Shared body of `nvfp4_quantize`/`nvfp4_quantize_transpose` — see their
    docstrings. `rows`/`k` are always the LOGICAL output tensor's shape (the
    transpose, when `TRANSPOSE=True`); only `_nvfp4_quantize_gpu`'s SOURCE
    READ address formula (`_nvfp4_src_addr`) differs between the two modes,
    everything else (buffer sizing, grid, tensor-scale pass) is identical.
    """
    comptime assert (
        BLOCK_ROWS == 1 or BLOCK_ROWS == NVFP4_BLOCK
    ), "BLOCK_ROWS must be 1 (1D acts/grads) or 16 (2D weights)"
    if k % NVFP4_BLOCK != 0:
        raise Error(
            "nvfp4_quantize: k must be a multiple of " + String(NVFP4_BLOCK)
        )
    comptime if is_gpu[target]():
        var device_ctx = ctx
        nvfp4_compute_tensor_scale[dtype, target](
            tensor_scale_ptr, x_ptr, rows * k, device_ctx
        )

        var k_blocks = k // NVFP4_BLOCK
        var tile_rows = nvfp4_scale_rows(rows, BLOCK_ROWS)
        var n_col_tiles = ceildiv(k_blocks, 4)

        # `TRANSPOSE=True` dispatches to the coalesced-read kernel: a strided
        # transpose read is ~3x slower, so `TRANSPOSE=True` uses
        # `_nvfp4_quantize_transpose_coalesced_gpu` instead (see its
        # docstring). `TRANSPOSE=False` (plain `nvfp4_quantize`) is
        # untouched — its reads are already coalesced by construction
        # (consecutive threads' `kidx` steps by 1, matching the source's
        # contiguous `k` axis), so a rewrite would add overhead for no
        # access-pattern win.
        comptime if TRANSPOSE:
            comptime BLOCK_SIZE = 256
            comptime coalesced_kernel = _nvfp4_quantize_transpose_coalesced_gpu[
                dtype, BLOCK_ROWS, round_mode, BLOCK_SIZE
            ]
            var compiled_t = device_ctx.compile_function[coalesced_kernel]()
            device_ctx.enqueue_function(
                compiled_t,
                q_ptr,
                scale_ptr,
                x_ptr,
                tensor_scale_ptr,
                rows,
                k,
                n_col_tiles,
                sr_seed,
                sr_stream,
                sr_step,
                grid_dim=(k_blocks, ceildiv(rows, BLOCK_SIZE)),
                block_dim=(BLOCK_SIZE,),
            )
        else:
            var total = tile_rows * k_blocks
            comptime BLOCK_SIZE = 256
            var num_blocks = ceildiv(total, BLOCK_SIZE) if total > 0 else 1

            comptime quant_kernel = _nvfp4_quantize_gpu[
                dtype, BLOCK_ROWS, round_mode, TRANSPOSE
            ]
            var compiled = device_ctx.compile_function[quant_kernel]()
            device_ctx.enqueue_function(
                compiled,
                q_ptr,
                scale_ptr,
                x_ptr,
                tensor_scale_ptr,
                rows,
                k,
                tile_rows,
                k_blocks,
                n_col_tiles,
                sr_seed,
                sr_stream,
                sr_step,
                grid_dim=(num_blocks,),
                block_dim=(BLOCK_SIZE,),
            )
    else:
        raise Error("nvfp4_quantize is GPU-only")


def nvfp4_quantize[
    dtype: DType,
    target: StaticString,
    BLOCK_ROWS: Int,
    round_mode: Int = ROUND_MODE_RNE,
](
    q_ptr: MutKernelPtr[DType.uint8],
    scale_ptr: MutKernelPtr[DType.uint8],
    tensor_scale_ptr: MutKernelPtr[DType.float32],
    x_ptr: ImmutKernelPtr[dtype],
    rows: Int,
    k: Int,
    ctx: DeviceContext,
    sr_seed: UInt64 = NVFP4_SR_SEED,
    sr_stream: UInt64 = NVFP4_SR_STREAM,
    sr_step: Int = 0,
) raises -> None:
    """Quantizes a `[rows, k]` bf16 (or fp32) tensor to NVFP4: packed e2m1
    `q_ptr` (size `nvfp4_packed_size(rows, k)`), swizzled e4m3 block scales
    `scale_ptr` (size `nvfp4_scale_buffer_size(rows, k, BLOCK_ROWS)`), and a
    freshly-computed fp32 per-tensor scale written into `tensor_scale_ptr[0]`.

    `BLOCK_ROWS=1` -> 1D 1x16 blocks (activations & gradients per the
    recipe); `BLOCK_ROWS=16` -> 2D 16x16 blocks (weights). `k` must be a
    multiple of 16.

    `sr_seed`/`sr_stream`/`sr_step` only matter when `round_mode ==
    ROUND_MODE_STOCHASTIC` (module docstring's Rounding section); they are
    plain no-op parameters at their default values otherwise. `sr_step`
    should be the caller's training-step counter (or any per-call-site
    monotonic counter) so repeated quantize calls on the same tensor draw
    fresh dither each time instead of reusing the same bit pattern every
    call under a fixed seed — pass a distinct `sr_stream` per logical
    call-site (e.g. `llmm/matmul.mojo`'s `lowp_gemm_fp4` uses
    `NVFP4_SR_STREAM`/`NVFP4_SR_STREAM + 1` for its A/B operands) so
    unrelated tensors quantized under the same seed never share a dither
    substream.
    """
    _nvfp4_quantize_impl[dtype, target, BLOCK_ROWS, round_mode, False](
        q_ptr,
        scale_ptr,
        tensor_scale_ptr,
        x_ptr,
        rows,
        k,
        ctx,
        sr_seed,
        sr_stream,
        sr_step,
    )


def nvfp4_quantize_transpose[
    dtype: DType,
    target: StaticString,
    BLOCK_ROWS: Int,
    round_mode: Int = ROUND_MODE_RNE,
](
    q_ptr: MutKernelPtr[DType.uint8],
    scale_ptr: MutKernelPtr[DType.uint8],
    tensor_scale_ptr: MutKernelPtr[DType.float32],
    x_ptr: ImmutKernelPtr[dtype],  # [src_rows, src_k] row-major SOURCE layout
    src_rows: Int,
    src_k: Int,
    ctx: DeviceContext,
    sr_seed: UInt64 = NVFP4_SR_SEED,
    sr_stream: UInt64 = NVFP4_SR_STREAM,
    sr_step: Int = 0,
) raises -> None:
    """Fused transpose+quantize: quantizes the LOGICAL TRANSPOSE of a
    `[src_rows, src_k]` row-major bf16/fp32 SOURCE tensor `x_ptr` to NVFP4 —
    i.e. produces byte-identical output to calling `nvfp4_quantize` on a
    materialized `[src_k, src_rows]` tensor `T` where `T[i, j] = x_ptr[j,
    i]`, computed in one pass with no separate bf16 transpose scratch buffer
    (the FP4 analogue of `llmm/lowp.mojo`'s `quantize_transpose_devscale` —
    see the module docstring's "Transposed quantize" section for the full
    derivation, and `llmm/matmul.mojo`'s dgrad/wgrad orientation comment for
    which operands need this: Dgrad's `weight`, Wgrad's `input` and
    `d_output`).

    Output buffer shapes follow the LOGICAL (transposed) tensor `[src_k,
    src_rows]`: `q_ptr` >= `nvfp4_packed_size(src_k, src_rows)`, `scale_ptr`
    >= `nvfp4_scale_buffer_size(src_k, src_rows, BLOCK_ROWS)`. The LOGICAL
    output's trailing/block-16 axis is `src_rows` (the SOURCE's row count),
    NOT `src_k` — so `src_rows` must be a multiple of 16 here (the mirror
    image of `nvfp4_quantize`'s "`k` must be a multiple of 16", since
    transposition swaps which physical axis plays the block-16 role).

    `sr_seed`/`sr_stream`/`sr_step`: same contract as `nvfp4_quantize`; pass
    a stream distinct from any other call site sharing `sr_seed` (see the
    module docstring's "Stream registry" section for the reserved fp4
    backward stream ids).
    """
    _nvfp4_quantize_impl[dtype, target, BLOCK_ROWS, round_mode, True](
        q_ptr,
        scale_ptr,
        tensor_scale_ptr,
        x_ptr,
        src_k,  # logical output rows
        src_rows,  # logical output k -- must be a multiple of 16
        ctx,
        sr_seed,
        sr_stream,
        sr_step,
    )


# ===----------------------------------------------------------------------=== #
# Host-side pure-Mojo dequant reference (for tests)
#
# Uses the SAME encode_e2m1/decode_e2m1/encode_e4m3/decode_e4m3/swizzle
# functions as the GPU kernel above (they are plain fp32/UInt8 scalar math,
# no GPU-specific builtins) -- so this reference is guaranteed to use
# identical numerics to the device path, not a re-derivation of it. Callable
# from ordinary host Mojo code (no DeviceContext needed).
# ===----------------------------------------------------------------------=== #


def nvfp4_dequant_reference[
    BLOCK_ROWS: Int,
](
    out_ptr: MutMemPtr[DType.float32],
    q_ptr: ImmutMemPtr[DType.uint8],
    scale_ptr: ImmutMemPtr[DType.uint8],
    tensor_scale: Float32,
    rows: Int,
    k: Int,
) raises -> None:
    """Reconstructs a `[rows, k]` fp32 tensor from NVFP4-quantized
    `q_ptr`/`scale_ptr` (same layout `nvfp4_quantize` produces) +
    `tensor_scale`. Matches `tests/probe_fp4/probe_fp4.cu`'s software
    dequant reference.
    """
    if k % NVFP4_BLOCK != 0:
        raise Error(
            "nvfp4_dequant_reference: k must be a multiple of "
            + String(NVFP4_BLOCK)
        )
    var k_blocks = k // NVFP4_BLOCK
    var tile_rows = nvfp4_scale_rows(rows, BLOCK_ROWS)
    var n_col_tiles = ceildiv(k_blocks, 4)

    for br in range(tile_rows):
        for kb in range(k_blocks):
            # Every physical row in this tile carries the same replicated
            # code (nvfp4_scale_buffer_size's docstring) -- read from the
            # tile's first physical row, `br * BLOCK_ROWS`.
            var soff = nvfp4_scale_swizzle_offset(
                br * BLOCK_ROWS, kb, n_col_tiles
            )
            var sc_val = decode_e4m3(scale_ptr[soff]) * tensor_scale
            for rr in range(BLOCK_ROWS):
                var r = br * BLOCK_ROWS + rr
                if r >= rows:
                    continue
                for kk in range(NVFP4_BLOCK):
                    var kidx = kb * NVFP4_BLOCK + kk
                    if kidx >= k:
                        continue
                    var byte = q_ptr[(r * k + kidx) // 2]
                    var nib: UInt8
                    if kidx % 2 == 0:
                        nib = unpack_e2m1x2_lo(byte)
                    else:
                        nib = unpack_e2m1x2_hi(byte)
                    out_ptr[r * k + kidx] = decode_e2m1(nib) * sc_val
