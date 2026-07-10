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
path. `ROUND_MODE_STOCHASTIC` is a **declared seam, not implemented**: the
recipe (`fp4_training_recipes_research.md` §1) requires stochastic rounding
on *gradient* operands, which needs a counter-based device RNG that does not
exist yet in this tree (`llmm/rand.mojo` is host-only MT19937). That RNG is
explicitly the FP8 team's Chunk G
(`../llmm-goal2-fp8/docs/ai/fp8_training_design.md` §6) — wiring stochastic
rounding here is intentionally left as a `comptime assert`-guarded TODO
rather than building a parallel/conflicting RNG.

## Scope

Standalone utility kernels only — no `llmm/matmul.mojo` / `train_gpt2.mojo`
integration (that is the later merge's job, and `llmm/lowp.mojo`'s
`PrecisionSpec`/`ScalingKind.Block2D` seam is where it will plug in per the
FP8 team's design doc §3).
"""

from std.collections import InlineArray
from std.math import ceildiv
from std.memory import bitcast
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from std.gpu.primitives import block
from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from llmm.memory import ImmutKernelPtr, MutKernelPtr, ImmutMemPtr, MutMemPtr


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime NVFP4_BLOCK = 16  # elements per scale, along K (both 1D and 2D)
comptime E2M1_MAX = Float32(6.0)  # largest representable e2m1 magnitude
comptime E4M3_MAX = Float32(448.0)  # largest representable e4m3 magnitude


# Rounding-mode seam for `encode_e2m1`. RNE is implemented; STOCHASTIC is
# reserved for the FP8 team's device-RNG chunk (see module docstring). Plain
# comptime Int constants (not a struct namespace) to avoid any question of
# whether a field-less struct is instantiable — matches this file's/the
# codebase's existing `comptime FOO = <literal>` convention (e.g.
# llmm/gelu.mojo's UNROLL/GELU_CONSTANT).
comptime ROUND_MODE_RNE = 0
comptime ROUND_MODE_STOCHASTIC = 1


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
def encode_e2m1[round_mode: Int = ROUND_MODE_RNE](x: Float32) -> UInt8:
    """Encode a fp32 value to a 4-bit e2m1 code (0..15: bit3=sign,
    bits2:0=magnitude index). Values are implicitly saturated to +/-6.0.
    """
    comptime assert round_mode == ROUND_MODE_RNE, (
        "ROUND_MODE_STOCHASTIC is a declared seam, not implemented here —"
        " needs the FP8 team's counter-based device RNG"
        " (see llmm/nvfp4_quant.mojo module docstring)."
    )
    var sign: UInt8 = UInt8(8) if x < 0.0 else UInt8(0)
    var ax = x if x >= 0.0 else -x
    var idx: UInt8
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
# 0x7F/0xFF reserved for NaN) — manual bit manipulation, ported from
# tests/probe_fp4/probe_fp4.cu::encode_e4m3/decode_e4m3 (which uses
# frexpf/powf; this version decomposes the fp32 bit pattern directly via
# `bitcast`, which is mathematically identical — same e_unbiased / mantissa
# decomposition — and avoids any pop.cast to an actual float8 dtype).
# Block scales in this module are always >= 0, but sign is handled for
# generality.
# ===----------------------------------------------------------------------=== #


@always_inline
def encode_e4m3(x: Float32) -> UInt8:
    """Encode a fp32 value to an e4m3 byte, round-to-nearest
    (ties-away-from-zero), saturating to +/-448.
    """
    if x != x:  # NaN
        return UInt8(0x7F)
    var sign: UInt8 = UInt8(0x80) if x < 0.0 else UInt8(0x00)
    var ax = x if x >= 0.0 else -x
    if ax == 0.0:
        return sign
    if ax > 448.0:
        ax = 448.0
    var bits = bitcast[DType.uint32, 1](ax)
    var exp32 = Int((bits >> UInt32(23)) & UInt32(0xFF))
    if exp32 == 0:
        # fp32-subnormal input (< ~1.18e-38) is far below e4m3's smallest
        # subnormal (2^-9 ~= 0.00195) -- flush to (signed) zero.
        return sign
    var mant32 = Int(bits & UInt32(0x7FFFFF))
    var mant2 = Float32(1.0) + Float32(mant32) / Float32(1 << 23)  # [1, 2)
    var e_unbiased = exp32 - 127
    if e_unbiased < -6:
        # subnormal e4m3: value = m * 2^-9, m in [0, 7]
        var m = Int(ax * 512.0 + 0.5)
        if m > 7:
            m = 7
        return sign | UInt8(m)
    var e_biased = e_unbiased + 7
    var m3 = Int((mant2 - 1.0) * 8.0 + 0.5)
    if m3 == 8:
        m3 = 0
        e_biased += 1
    # DEVIATION FROM THE PROBE: tests/probe_fp4/probe_fp4.cu::encode_e4m3
    # has `if (e_biased >= 15) { e_biased = 14; m3 = 7; }` here, commented
    # "saturate to 448" -- but (e_biased=14, m3=7) decodes to 240, not 448
    # (verified: e4m3's true max finite is e_biased=15, m3=6 == 1.75*2^8 ==
    # 448; e_biased=15 with m3=7 is the reserved NaN pattern). Since `ax` is
    # already clamped to <=448.0 above, e_biased organically never exceeds
    # 15 and m3 never reaches 7 at e_biased=15 (mant2 tops out at 1.75) --
    # so the probe's check is both unreachable-as-a-no-op in the common
    # case *and* actively wrong (silently corrupts any value that lands
    # exactly on e_biased==15, e.g. ax in roughly [256, 448]) if it ever
    # does fire. Caught by this file's own
    # test_e4m3_saturation/test_e4m3_roundtrip_relative_error_bounded
    # (447.0 -> the buggy branch produced 240.0). Fixed here to saturate to
    # the true max finite (448) only when the encoding would otherwise
    # genuinely overflow 4 exponent bits or collide with the NaN pattern.
    if e_biased > 15:
        e_biased = 15
        m3 = 6
    elif e_biased == 15 and m3 == 7:
        m3 = 6
    return sign | (UInt8(e_biased) << 3) | UInt8(m3)


@always_inline
def decode_e4m3(code: UInt8) -> Float32:
    """Decode an e4m3 byte back to fp32."""
    var sign = (code >> 7) & UInt8(1)
    var e = Int((code >> 3) & UInt8(0xF))
    var m = Int(code & UInt8(0x7))
    var mag: Float32
    if e == 0:
        mag = Float32(m) / 512.0
    else:
        var mant2 = Float32(1.0) + Float32(m) / 8.0
        # Construct 2^(e-7) by writing the fp32 exponent field directly
        # (bias 127) instead of calling a transcendental pow()/exp2().
        var scale_bits = UInt32((e - 7 + 127) << 23)
        var pow2 = bitcast[DType.float32, 1](scale_bits)
        mag = mant2 * pow2
    return -mag if sign == UInt8(1) else mag


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
    return ceildiv(rows, BLOCK_ROWS)


@always_inline
def nvfp4_scale_buffer_size(rows: Int, k: Int, BLOCK_ROWS: Int) -> Int:
    """Byte size of the swizzled scale buffer for a `[rows, k]` tensor
    quantized with `BLOCK_ROWS` (1 for 1D 1x16, 16 for 2D 16x16).
    """
    var scale_rows = nvfp4_scale_rows(rows, BLOCK_ROWS)
    var k_blocks = k // NVFP4_BLOCK
    return nvfp4_swizzled_scale_buffer_size(scale_rows, k_blocks)


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
def _nvfp4_quantize_gpu[
    dtype: DType,
    BLOCK_ROWS: Int,
    round_mode: Int,
](
    q_ptr: MutKernelPtr[DType.uint8],
    scale_ptr: MutKernelPtr[DType.uint8],
    x_ptr: ImmutKernelPtr[dtype],
    tensor_scale_ptr: MutKernelPtr[DType.float32],
    rows: Int,
    k: Int,
    scale_rows: Int,
    k_blocks: Int,
    n_col_tiles: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = scale_rows * k_blocks
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
            var v = x_ptr[r * k + kidx].cast[DType.float32]()
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

    var soff = nvfp4_scale_swizzle_offset(br, kb, n_col_tiles)
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
            var c0 = encode_e2m1[round_mode](
                x_ptr[r * k + k0].cast[DType.float32]() / sc_val
            )
            var c1 = UInt8(0)
            var k1 = k0 + 1
            if k1 < k and kk + 1 < NVFP4_BLOCK:
                c1 = encode_e2m1[round_mode](
                    x_ptr[r * k + k1].cast[DType.float32]() / sc_val
                )
            q_ptr[(r * k + k0) // 2] = pack_e2m1x2(c0, c1)
            kk += 2


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
) raises -> None:
    """Quantizes a `[rows, k]` bf16 (or fp32) tensor to NVFP4: packed e2m1
    `q_ptr` (size `nvfp4_packed_size(rows, k)`), swizzled e4m3 block scales
    `scale_ptr` (size `nvfp4_scale_buffer_size(rows, k, BLOCK_ROWS)`), and a
    freshly-computed fp32 per-tensor scale written into `tensor_scale_ptr[0]`.

    `BLOCK_ROWS=1` -> 1D 1x16 blocks (activations & gradients per the
    recipe); `BLOCK_ROWS=16` -> 2D 16x16 blocks (weights). `k` must be a
    multiple of 16.
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
        var scale_rows = nvfp4_scale_rows(rows, BLOCK_ROWS)
        var n_col_tiles = ceildiv(k_blocks, 4)
        var total = scale_rows * k_blocks
        comptime BLOCK_SIZE = 256
        var num_blocks = ceildiv(total, BLOCK_SIZE) if total > 0 else 1

        comptime quant_kernel = _nvfp4_quantize_gpu[
            dtype, BLOCK_ROWS, round_mode
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
            scale_rows,
            k_blocks,
            n_col_tiles,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("nvfp4_quantize is GPU-only")


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
    var scale_rows = nvfp4_scale_rows(rows, BLOCK_ROWS)
    var n_col_tiles = ceildiv(k_blocks, 4)

    for br in range(scale_rows):
        for kb in range(k_blocks):
            var soff = nvfp4_scale_swizzle_offset(br, kb, n_col_tiles)
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
