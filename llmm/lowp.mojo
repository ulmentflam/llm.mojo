# ===----------------------------------------------------------------------=== #
# lowp.mojo — dtype-generic low-precision (fp8/fp4) scaling layer.
#
# See docs/ai/fp8_training_design.md §3 for the full design. This file is the
# single place the fp8 (and future fp4/NVFP4) scaling machinery lives, so a new
# precision is a new `PrecisionSpec` constant rather than new call sites.
#
# Chunk A (this file's initial state) only stubs the comptime shape: the
# `PrecisionSpec` struct, the `ScalingKind` comptime enum, and `precision_spec`
# resolving a precision name to its spec. No quantize/GEMM/scaling kernels
# exist yet — those are Chunks B/C. Every kernel added later here must stay
# `comptime if is_gpu[target]()`-guarded (never instantiated for the "cpu"
# target) per the AArch64 codegen landmine documented in
# docs/ai/fp8_training_design.md §2 and train_gpt2.mojo's `_dispatch_cpu`.
# ===----------------------------------------------------------------------=== #

from std.memory import UnsafePointer
from std.math import ceildiv
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from std.gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from std.gpu.memory import AddressSpace
from std.sys import simd_width_of, size_of
from layout import Layout, LayoutTensor

from llmm.memory import MutKernelPtr, ImmutKernelPtr


# ===----------------------------------------------------------------------=== #
# ScalingKind — comptime enum selecting the scale-tensor granularity.
# ===----------------------------------------------------------------------=== #


struct ScalingKind:
    """Comptime enum of scaling granularities.

    `PerTensor` is FP8's single scalar-per-operand scheme (§1.3). `Block1D` is
    reserved. `Block2D` is NVFP4's 16-element-block, 2D-tiled scheme (§3 seam,
    not implemented in Chunk A).
    """

    comptime PerTensor = 0
    comptime Block1D = 1
    comptime Block2D = 2


# ===----------------------------------------------------------------------=== #
# PrecisionSpec — the comptime-parameterized description of a precision.
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct PrecisionSpec(Copyable, ImplicitlyCopyable, Movable):
    """Dtype-generic description of a low-precision training regime.

    A GEMM/quantize/scaling call site is parameterized by one `PrecisionSpec`
    value (comptime), never by a hardcoded dtype — so FP4 is a new spec
    constant, not new code (docs/ai/fp8_training_design.md §3).
    """

    var fwd_dtype: DType
    """Forward linear GEMM operand dtype (E4M3 for fp8; packed E2M1 for fp4)."""
    var bwd_dtype: DType
    """Backward gradient operand dtype (E5M2 for fp8; packed E2M1 for fp4)."""
    var scale_dtype: DType
    """Dtype of the scale factor(s) (fp32 per-tensor for fp8; e4m3fn block
    scale for NVFP4)."""
    var scaling: Int
    """One of `ScalingKind`'s values."""
    var block: Int
    """Block size for block-scaled formats; 0 for per-tensor (fp8)."""
    var amax_history_len: Int
    """Length of the delayed-scaling amax history ring buffer."""
    var margin: Int
    """Exponent margin subtracted before computing `scale` (TE-style)."""
    var stochastic_rounding: Bool
    """Whether `quantize`'s narrowing cast uses stochastic rounding (Chunk G)
    instead of plain round-to-nearest-even."""
    var hadamard: Bool
    """Whether `quantize` applies a Hadamard transform before narrowing
    (outlier-spreading seam for fp4)."""


# ===----------------------------------------------------------------------=== #
# Precision spec constants + resolver.
# ===----------------------------------------------------------------------=== #

# FP8 (Transformer-Engine HYBRID): E4M3 forward operands, E5M2 backward-gradient
# operand, fp32 per-tensor scale, 16-step amax history, no margin, RNE narrowing
# (no stochastic rounding / Hadamard — those are fp4-only seams, §3).
comptime FP8_SPEC = PrecisionSpec(
    DType.float8_e4m3fn,
    DType.float8_e5m2,
    DType.float32,
    ScalingKind.PerTensor,
    0,
    16,
    0,
    False,
    False,
)

# NVFP4 (docs/ai/fp4_training_recipes_research.md §1): e2m1 data for BOTH
# forward and backward-gradient operands (unlike fp8's asymmetric e4m3/e5m2
# split — NVFP4 uses one element format everywhere), e4m3 per-block scale
# (`scale_dtype`), `Block1D` (1x16) as the *default* granularity — activations
# and gradients use 1x16 blocks; the weight path additionally uses `Block2D`
# (16x16, `ScalingKind.Block2D`) at specific call sites, selected by the
# caller (`llmm/nvfp4_quant.mojo`'s `nvfp4_quantize[..., BLOCK_ROWS=16]`), not
# by a second spec constant — `block=16` here describes the along-K block
# extent, which is identical (16) for both granularities; only the row extent
# (`BLOCK_ROWS`, a `nvfp4_quantize` template parameter, not a `PrecisionSpec`
# field) differs.
#
# `amax_history_len=0`, `margin=0`: deliberately inert, NOT a stub-pending
# value. NVFP4 quantization computes its two-level scale (fp32 per-tensor +
# e4m3 per-block) FRESH every `nvfp4_quantize` call, entirely device-resident
# (`nvfp4_compute_tensor_scale` inside `llmm/nvfp4_quant.mojo`) — there is no
# delayed/history-based scaling step for FP4 to configure (unlike fp8's
# `PerTensor` scaling, which amortizes an amax reduction across
# `amax_history_len` steps precisely because its single per-tensor scale is
# coarse and benefits from temporal smoothing). NVFP4's 16-element block
# scale already adapts *within* a tensor, so the coarser tensor-level scale
# does not need delayed smoothing to stay accurate — see the `AmaxState`
# Block1D/Block2D seam note in `llmm/amax.mojo` for the full reconciliation
# (decision: FP4 does not use `AmaxState` at all; both scale levels are
# computed in-kernel by `nvfp4_quantize`).
#
# `stochastic_rounding=True`/`hadamard=True`: precision-level markers ("this
# regime uses these techniques"), not a blanket per-operand switch the way
# fp8's `quantize_devscale`/`quantize_transpose_devscale` read
# `spec.stochastic_rounding`
# uniformly for every operand. The recipe requires SR only on *gradient*
# operands (weights/activations stay RNE) and RHT only on *Wgrad* operands
# (`llmm/hadamard.mojo`) — that per-operand-role selection is a call-site
# decision for the training-integration chunk that wires
# `llmm/nvfp4_quant.mojo`'s `round_mode`/`ROUND_MODE_STOCHASTIC` template
# parameter and `llmm/hadamard.mojo`'s `hadamard16_fwd_gpu` per GEMM operand,
# not something `FP4_SPEC` itself can express with a single struct-wide bool
# (nor should it: `PrecisionSpec` is one value per precision, not per
# operand-role).
comptime FP4_SPEC = PrecisionSpec(
    DType.float4_e2m1fn,
    DType.float4_e2m1fn,
    DType.float8_e4m3fn,
    ScalingKind.Block1D,
    16,
    0,
    0,
    True,
    True,
)

# fp32/bf16 have no low-precision GEMM transient, so their "spec" is an inert
# placeholder: never read because call sites gate on `LOWP_ENABLED` (only True
# for fp8/fp4) before consulting `SPEC`. Kept dtype-valid (not zero-initialized
# garbage) so `precision_spec` is a total function over the whole `PRECISION`
# axis and every build configuration gets a well-formed `SPEC` value.
comptime _INERT_SPEC = PrecisionSpec(
    DType.bfloat16,
    DType.bfloat16,
    DType.float32,
    ScalingKind.PerTensor,
    0,
    0,
    0,
    False,
    False,
)


# ===----------------------------------------------------------------------=== #
# Manual fp8 encode/decode (Chunk B).
#
# Mojo GPU codegen has no lowering for `pop.cast f8eXmY -> {f32,bf16}` once the
# upcast value feeds arithmetic (confirmed empirically: probe2/probe2c in
# tests/probe_fp8/RESULTS.md — bare passthrough cast compiles, but
# `x.cast[float32]() + 1.0` does not, on this toolchain / sm_121). Host-target
# fp8 casts DO work (probe1) and are used as the correctness oracle below and
# in tests/test_lowp_gemm.mojo, but any *GPU* quantize kernel must never call
# `.cast[float8_e4m3fn]()` / `.cast[float8_e5m2]()` (or the reverse) — it must
# build/read the byte pattern by hand via integer ops on a `UInt8`-viewed
# buffer, which is what these functions do. They are plain host-or-device
# integer/fp32 arithmetic (no fp8-typed value ever exists here), so they are
# safe to call from GPU kernels.
# ===----------------------------------------------------------------------=== #


struct RoundMode:
    """Comptime enum selecting the narrowing-cast rounding rule."""

    comptime RNE = 0
    """Round-to-nearest-even (implemented here)."""
    comptime SR = 1
    """Stochastic rounding — Chunk G seam (see `encode_e4m3`/`encode_e5m2`)."""


# ===----------------------------------------------------------------------=== #
# TieMode / NanPolicy — the two-oracle divergence (DRY pass F1).
#
# This repo carries TWO e4m3 encoders under one shared core (`_fp8_encode`
# below), because they answer to two DIFFERENT external oracles and the
# difference is LOAD-BEARING — do NOT force them to one rule (that is a
# correctness regression against one of the two oracles; see
# docs/ai/dry_consolidation_audit_2026-07-10.md finding F1):
#
# - `encode_e4m3` (this file, TieMode.EVEN + NanPolicy.SATURATE): round-to-
#   nearest, ties-to-EVEN, NaN saturates to ±448 — chosen to match Mojo's
#   *host* `.cast[float8_e4m3fn]()` bit-for-bit, so the GPU fp8 quantizer
#   agrees with the host-cast oracle in tests/test_lowp_gemm.mojo.
# - `llmm/nvfp4_quant.mojo`'s `encode_e4m3` (TieMode.AWAY + NanPolicy.EMIT):
#   round-to-nearest, ties-AWAY-from-zero, NaN emits the 0x7F NaN byte —
#   chosen to match tests/probe_fp4/probe_fp4.cu's `roundf`-based reference
#   and the cuBLAS/PyTorch NVFP4 block-scale convention, gated by
#   tests/test_nvfp4_quant.mojo.
#
# TieMode.AWAY additionally preserves two more probe-lineage behaviors the
# audit's tie/NaN framing did not capture (both verified bit-level against
# the pre-consolidation nvfp4 encoder, and both kept for bit-compatibility
# with the shipped FP4 training trajectories):
# - sign comes from a `x < 0` compare (so -0.0 collapses to +0x00), not from
#   fp32 bit 31 (EVEN keeps bit 31: host cast encodes -0.0 as 0x80);
# - the subnormal branch reproduces the probe's float arithmetic
#   (`Int(ax * 2^(bias-1+mant_bits) + 0.5)`, clamped at the largest
#   subnormal rather than promoted to the smallest normal on overflow) —
#   including its `+ 0.5` fp32-addition rounding, which on inputs within
#   ~2^-25 of a subnormal tie differs from exact integer ties-away.
# ===----------------------------------------------------------------------=== #


struct TieMode:
    """Comptime enum selecting the tie-breaking rule of `_fp8_encode`."""

    comptime EVEN = 0
    """Ties-to-even — the Mojo host-cast oracle (fp8 training path)."""
    comptime AWAY = 1
    """Ties-away-from-zero — the probe/cuBLAS NVFP4 block-scale oracle."""


struct NanPolicy:
    """Comptime enum selecting `_fp8_encode`'s NaN handling."""

    comptime SATURATE = 0
    """NaN clamps to ±max_normal (matches Mojo's host fp8 cast)."""
    comptime EMIT = 1
    """NaN encodes as the format's NaN byte (0x7F for e4m3fn)."""


# Saturation ceilings (Mojo's *host*-target `.cast[float8_*]()` saturates to
# these — confirmed empirically, not IEEE-754 default (e5m2 has an encodable
# infinity but the cast never produces it): e.g. `1.0e5 -> e5m2 -> f32 ==
# 57344.0`, `1.0e5 -> e4m3fn -> f32 == 448.0`. The manual encoder below
# reproduces this saturating (never-inf/nan-producing) behavior bit-for-bit so
# GPU and host quantization agree.
comptime E4M3_MAX = Float32(448.0)
comptime E5M2_MAX = Float32(57344.0)


@always_inline
def _f32_bits(x: Float32) -> UInt32:
    var v = x
    return UnsafePointer(to=v).bitcast[UInt32]()[]


@always_inline
def _bits_f32(b: UInt32) -> Float32:
    var v = b
    return UnsafePointer(to=v).bitcast[Float32]()[]


@always_inline
def _pow2_f32(e: Int) -> Float32:
    # 2**e via direct exponent-field construction (e in the tiny range fp8
    # formats need, [-9, 8] — nowhere near fp32's [-126, 127] limits).
    return _bits_f32(UInt32(e + 127) << 23)


@always_inline
def _fp8_encode[
    exp_bits: Int,
    mant_bits: Int,
    bias: Int,
    max_normal: Float32,
    tie_mode: Int = TieMode.EVEN,
    nan_policy: Int = NanPolicy.SATURATE,
](x: Float32) -> UInt8:
    """Round-to-nearest encode of `x` into an `(exp_bits, mant_bits, bias)`
    fp8 format's bit pattern, saturating (never inf) at `max_normal`.
    Generic core shared by `encode_e4m3`/`encode_e5m2` (TieMode.EVEN — the
    host-cast oracle) and `llmm/nvfp4_quant.mojo`'s `encode_e4m3`
    (TieMode.AWAY + NanPolicy.EMIT — the probe/cuBLAS NVFP4 oracle). The two
    rule sets deliberately diverge and must both be kept — see the TieMode
    module comment above for the full divergence contract, and the module
    docstring for why this must stay pure integer/fp32 arithmetic (no
    fp8-typed intermediate) to be GPU-safe.
    """
    comptime if nan_policy == NanPolicy.EMIT:
        if x != x:
            # The format's (sign-0) NaN byte: exp field all-ones + mantissa
            # all-ones — 0x7F for e4m3fn (e5m2 never uses EMIT here).
            comptime nan_byte = (1 << (exp_bits + mant_bits)) - 1
            return UInt8(nan_byte)

    var xbits = _f32_bits(x)
    var sign_bit: UInt8
    comptime if tie_mode == TieMode.AWAY:
        # Probe-lineage sign: a `< 0` compare, so -0.0 collapses to +0x00
        # (the host-cast/EVEN rule below keeps bit 31: -0.0 -> 0x80).
        sign_bit = UInt8(0x80) if x < Float32(0.0) else UInt8(0x00)
    else:
        sign_bit = UInt8((xbits >> 31) & 1) << 7
    var ax = abs(x)
    # `not (ax <= max_normal)` catches ax > max_normal AND NaN (NaN compares
    # false against everything under IEEE rules) — both saturate rather than
    # propagate, matching the native host cast's observed behavior. (Under
    # NanPolicy.EMIT, NaN already returned above, so this is a plain clamp.)
    if not (ax <= max_normal):
        ax = max_normal
    if ax == Float32(0.0):
        return sign_bit

    var bits = _f32_bits(ax)
    var f32_exp = Int((bits >> 23) & 0xFF) - 127
    comptime min_normal_exp = 1 - bias

    comptime if tie_mode == TieMode.AWAY:
        # Probe-lineage subnormal branch, kept bit-for-bit (see the TieMode
        # module comment): fp32-subnormal input flushes to (signed) zero;
        # an fp8-subnormal result comes from the probe's float arithmetic —
        # `roundf`-style `+ 0.5` (whose fp32-addition rounding is part of
        # the preserved behavior) and a clamp at the largest subnormal
        # (`m > max` -> max, NOT a promote to the smallest normal).
        if f32_exp == -127:
            return sign_bit
        if f32_exp < min_normal_exp:
            # 1/min_subnormal = 2^(bias - 1 + mant_bits) (512 for e4m3).
            var m = Int(ax * _pow2_f32(bias - 1 + mant_bits) + Float32(0.5))
            comptime max_sub = (1 << mant_bits) - 1
            if m > max_sub:
                m = max_sub
            return sign_bit | UInt8(m)

    var full_mant: UInt32 = (bits & 0x7FFFFF) | (
        UInt32(1) << 23
    )  # 24-bit, implicit 1

    var shift: Int
    var exp_field: Int
    if f32_exp >= min_normal_exp:
        shift = 23 - mant_bits
        exp_field = f32_exp + bias
    else:
        var extra = min_normal_exp - f32_exp
        shift = 23 - mant_bits + extra
        exp_field = 0
        if shift > 24:
            # Below half a ULP of the smallest subnormal — rounds to zero.
            # (Guards the UInt32 shift from overflowing for tiny inputs.)
            return sign_bit

    var ushift = UInt32(shift)
    var half: UInt32 = UInt32(1) << (ushift - 1)
    var mask: UInt32 = (UInt32(1) << ushift) - 1
    var rem = full_mant & mask
    var mant_shifted = full_mant >> ushift
    comptime if tie_mode == TieMode.AWAY:
        # Ties-away-from-zero. On the normal path this is provably identical
        # to the probe's `Int((mant2 - 1) * 8 + 0.5)`: `(mant2-1)*8` is a
        # multiple of 2^-20 below 8, so `+ 0.5` needs at most 24 significand
        # bits and is exact in fp32 — no double rounding, unlike the
        # subnormal branch above.
        if rem >= half:
            mant_shifted += 1
    else:
        if rem > half or (rem == half and (mant_shifted & 1) == 1):
            mant_shifted += 1

    comptime umant_bits = UInt32(mant_bits)
    if exp_field == 0:
        # Subnormal rounding-up to the smallest normal.
        if mant_shifted >= (UInt32(1) << umant_bits):
            exp_field = 1
            mant_shifted = 0
    else:
        # Normal mantissa carry-out bumps the exponent (input is pre-clamped
        # to max_normal, which is exactly representable, so this can only
        # promote within the format's valid exponent range, never overflow
        # it — see module docstring / lowp_gemm_devscale design notes).
        #
        # Why this can't produce the "rounds up into the NaN/inf pattern"
        # bug: `ax` was already clamped to `max_normal` (exactly
        # representable: e4m3fn 448.0 = exp_field 15, mantissa 0b110 — NOT
        # the reserved mantissa=0b111 NaN pattern; e5m2 57344.0 = exp_field
        # 30, mantissa 0b11 — one below the reserved exp_field=31 inf/nan
        # block) *before* any rounding happens, so a value that would
        # otherwise round up past `max_normal` (e.g. e4m3fn 447.0, which is
        # closer to 448.0 than to the next-lower representable 416.0) is
        # rounded from the clamped value itself and lands exactly on
        # `max_normal`'s own bit pattern — never past it. A different,
        # structurally unsafe approach (round first, then clamp the
        # *exponent field* post-hoc if it overflows) can decode to garbage:
        # a sibling fp4-probe's e4m3 C++ reference did exactly that
        # (`if (e_biased >= 15) { e_biased = 14; m3 = 7; }`, intending
        # "saturate to 448" but actually producing exp_field=14/mantissa=7,
        # which decodes to 240.0, not 448.0) — see
        # tests/test_lowp_gemm.mojo's `test_encode_near_max_no_nan_pattern`
        # for the regression coverage (dense sweep across both formats' top
        # octave/exponent-band plus the literal 447.0/57343.0 cases).
        if mant_shifted >= (UInt32(1) << (umant_bits + 1)):
            mant_shifted >>= 1
            exp_field += 1
        mant_shifted &= (UInt32(1) << umant_bits) - 1  # drop the implicit 1

    var packed = (UInt32(exp_field) << umant_bits) | mant_shifted
    return sign_bit | UInt8(packed)


@always_inline
def _fp8_decode[exp_bits: Int, mant_bits: Int, bias: Int](b: UInt8) -> Float32:
    """Decode an `(exp_bits, mant_bits, bias)` fp8 bit pattern to fp32."""
    comptime umant_bits = UInt8(mant_bits)
    comptime uexp_bits = UInt8(exp_bits)
    var sign = (b >> 7) & 1
    var exp_field = Int((b >> umant_bits) & ((UInt8(1) << uexp_bits) - 1))
    var mant_field = Int(b & ((UInt8(1) << umant_bits) - 1))
    var mag: Float32
    if exp_field == 0:
        if mant_field == 0:
            mag = Float32(0.0)
        else:
            comptime min_normal_exp = 1 - bias
            mag = (Float32(mant_field) / Float32(1 << mant_bits)) * _pow2_f32(
                min_normal_exp
            )
    else:
        var e = exp_field - bias
        mag = (
            Float32(1.0) + Float32(mant_field) / Float32(1 << mant_bits)
        ) * _pow2_f32(e)
    return -mag if sign == 1 else mag


comptime _SR_SEAM_MSG = (
    "stochastic-rounding fp8 encode is a documented extension seam"
    " (docs/ai/fp8_training_design.md §3, Chunk G): it consumes a per-element"
    " counter-based `rand: UInt32` from llmm/rng_device.mojo's device RNG,"
    " which does not exist yet in this worktree (Chunk G is a parallel,"
    " independent chunk). Do not implement an RNG here — use"
    " RoundMode.RNE until Chunk G lands and wires the SR body."
)


@always_inline
def encode_e4m3[
    mode: Int = RoundMode.RNE
](x: Float32, rand: UInt32 = 0) -> UInt8:
    """Encode `x` (fp32) into a `float8_e4m3fn` bit pattern (as a `UInt8`),
    manually (no fp8-typed cast — see module docstring). `mode` selects RNE
    (implemented) or SR (Chunk G seam, comptime error until wired)."""
    comptime if mode == RoundMode.RNE:
        return _fp8_encode[4, 3, 7, E4M3_MAX](x)
    elif mode == RoundMode.SR:
        comptime assert False, _SR_SEAM_MSG
    else:
        comptime assert False, "unknown RoundMode"


@always_inline
def encode_e5m2[
    mode: Int = RoundMode.RNE
](x: Float32, rand: UInt32 = 0) -> UInt8:
    """Encode `x` (fp32) into a `float8_e5m2` bit pattern (as a `UInt8`),
    manually (no fp8-typed cast — see module docstring). `mode` selects RNE
    (implemented) or SR (Chunk G seam, comptime error until wired)."""
    comptime if mode == RoundMode.RNE:
        return _fp8_encode[5, 2, 15, E5M2_MAX](x)
    elif mode == RoundMode.SR:
        comptime assert False, _SR_SEAM_MSG
    else:
        comptime assert False, "unknown RoundMode"


@always_inline
def decode_e4m3(b: UInt8) -> Float32:
    """Decode a `float8_e4m3fn` bit pattern (`UInt8`) to fp32, manually."""
    return _fp8_decode[4, 3, 7](b)


@always_inline
def decode_e5m2(b: UInt8) -> Float32:
    """Decode a `float8_e5m2` bit pattern (`UInt8`) to fp32, manually."""
    return _fp8_decode[5, 2, 15](b)


@always_inline
def encode_fp8[
    out_dtype: DType, mode: Int = RoundMode.RNE
](x: Float32, rand: UInt32 = 0) -> UInt8:
    """Dtype-generic wrapper over `encode_e4m3`/`encode_e5m2` — the form
    the quantize kernels (below) use so a call site is parameterized by
    `out_dtype` (`spec.fwd_dtype` or `spec.bwd_dtype`), not a hardcoded
    format."""
    comptime if out_dtype == DType.float8_e4m3fn:
        return encode_e4m3[mode](x, rand)
    elif out_dtype == DType.float8_e5m2:
        return encode_e5m2[mode](x, rand)
    else:
        comptime assert (
            False
        ), "encode_fp8: out_dtype must be float8_e4m3fn or float8_e5m2"


@always_inline
def decode_fp8[out_dtype: DType](b: UInt8) -> Float32:
    """Dtype-generic wrapper over `decode_e4m3`/`decode_e5m2`."""
    comptime if out_dtype == DType.float8_e4m3fn:
        return decode_e4m3(b)
    elif out_dtype == DType.float8_e5m2:
        return decode_e5m2(b)
    else:
        comptime assert (
            False
        ), "decode_fp8: out_dtype must be float8_e4m3fn or float8_e5m2"


# ===----------------------------------------------------------------------=== #
# GPU quantize kernels (Chunk B).
#
# `out_ptr` is always `MutKernelPtr[DType.uint8]` — the fp8 buffer's *byte*
# view, never a `Scalar[float8_e4m3fn/e5m2]` pointer, because storing through
# a genuinely fp8-typed pointer risks routing back through the same broken
# `pop.cast` lowering path documented above (the working pattern, confirmed by
# probe2a, is a fp8 value that is only ever a passthrough store — never worth
# risking when `UInt8` is unambiguously safe and `encode_fp8` already returns
# the exact bit pattern). Callers `bitcast[whatever]()` the uint8 buffer where
# a byte-identical fp8 pointer is actually required (e.g. handing it to
# cuBLASLt, which only cares about the raw bytes).
# ===----------------------------------------------------------------------=== #


# ===----------------------------------------------------------------------=== #
# Device-pointer-scale quantize kernels (Chunk D).
#
# Chunk C's `AmaxState.scale`/`scale_inv` (llmm/amax.mojo) are
# DEVICE-resident fp32 scalars, deliberately never read back to host on
# the training step path (design §4's landmine-2 audit: "Host readback? no").
# A GPU kernel launch's scalar arguments are copied host->device at launch
# time regardless of whether the source is a host variable or a device
# pointer being dereferenced — so a `scale: Float32` parameter here would
# force a host readback of `AmaxState.scale` right before every single
# quantize call, once per fp8 GEMM operand per step. These `_devscale`
# kernels instead take `scale_ptr: ImmutKernelPtr[DType.float32]` and
# dereference it *inside* the kernel (`scale_ptr[0]`), so the scale value
# never leaves the device — the call site passes `AmaxState.scale`'s own
# device pointer straight through.
#
# History (DRY pass F3, docs/ai/dry_consolidation_audit_2026-07-10.md):
# Chunk B originally also shipped host-`Float32`-scale twins
# (`quantize`/`quantize_transpose`), which existed only for its unit-test
# gate — every production call site used the `_devscale` forms. The twins
# were deleted and `tests/test_lowp_gemm.mojo` now uploads its host-computed
# scale into a 1-element device buffer and calls the `_devscale` forms.
# ===----------------------------------------------------------------------=== #


def _quantize_kernel_devscale[
    in_dtype: DType, out_dtype: DType, mode: Int
](
    out_ptr: MutKernelPtr[DType.uint8],
    in_ptr: ImmutKernelPtr[in_dtype],
    scale_ptr: ImmutKernelPtr[DType.float32],
    n: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < n:
        var v = in_ptr[idx].cast[DType.float32]() * scale_ptr[0]
        out_ptr[idx] = encode_fp8[out_dtype, mode](v)


def quantize_devscale[
    spec: PrecisionSpec,
    out_dtype: DType,
    in_dtype: DType,
    target: StaticString,
](
    out_ptr: MutKernelPtr[DType.uint8],
    in_ptr: ImmutKernelPtr[in_dtype],
    scale_ptr: ImmutKernelPtr[DType.float32],
    n: Int,
    ctx: DeviceContext,
) raises -> None:
    """Bf16/fp32 -> fp8 quantize: `out[i] = encode(in[i] * scale_ptr[0])`.

    `scale_ptr` is a device fp32 scalar (e.g. `AmaxState.scale`'s
    `unsafe_ptr()`); the kernel reads it directly, no host readback — see
    the module comment above.

    GPU-only (comptime-asserted): fp8 is a device-only GEMM transient
    (docs/ai/fp8_training_design.md §1.1), and the manual `encode_fp8` above
    is deliberately written to be GPU-safe (integer/fp32 arithmetic only, no
    fp8-typed intermediate — see the module docstring; plain `pop.cast
    f8eXmY -> {f32,bf16}` is broken on this toolchain's GPU target once the
    upcast feeds arithmetic, per `tests/probe_fp8/RESULTS.md` probes 2/2c/3).

    The scale is a value the *caller* supplies — this function does not
    compute amax or a delayed-scaling schedule; that is Chunk C's
    `AmaxState`/`compute_amax`/`update_scale` (§3).
    """
    comptime assert is_gpu[target](), (
        "quantize_devscale is GPU-only per"
        " docs/ai/fp8_training_design.md landmine #1 (low-precision kernels"
        " must never be instantiated for the cpu target)"
    )
    comptime BLOCK_SIZE = 256
    var num_blocks = ceildiv(n, BLOCK_SIZE)
    comptime kmode = RoundMode.SR if spec.stochastic_rounding else RoundMode.RNE
    comptime kernel = _quantize_kernel_devscale[in_dtype, out_dtype, kmode]
    var compiled = ctx.compile_function[kernel]()
    ctx.enqueue_function(
        compiled,
        out_ptr,
        in_ptr,
        scale_ptr,
        n,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )


# ===----------------------------------------------------------------------=== #
# quantize_transpose tiling constants (Optimization A, docs/ai/
# ai_assisted_optimizations_and_benchmarks.md 2026-07-10 fp8-quant-opt entry).
#
# The original `_quantize_transpose_kernel` was one-thread-one-element:
# `out_ptr[c * rows + r] = encode(...)` with `c = idx % cols` varying fastest
# across a warp. Consecutive threads (consecutive `idx`, hence mostly
# consecutive `c`) write to addresses `rows` elements apart — a
# severely uncoalesced strided store (confirmed empirically: this kernel ran
# ~4.3-4.9x slower than the byte-equivalent non-transposed `quantize` kernel
# on the SAME tensors, an artifact fully attributable to the write pattern
# since the per-element compute — `encode_fp8` — is identical). This tile
# body fixes it with the same 32x32 shared-memory-tile transpose pattern
# already proven in this file's `_gpu_transpose_add_into_kernel` (32x33
# padded stride to dodge shared-memory bank conflicts on the transposed
# read/write, coalesced global read AND write). See the diagnosis entry for
# the measured before/after.
# ===----------------------------------------------------------------------=== #

comptime _QT_TILE = 32
comptime _QT_STRIDE = _QT_TILE + 1  # +1 padding avoids shared-mem bank conflicts
comptime _QT_BLOCK = 256  # ROW_STEP = _QT_BLOCK // _QT_TILE = 8


def _quantize_transpose_kernel_devscale[
    in_dtype: DType, out_dtype: DType, mode: Int
](
    out_ptr: MutKernelPtr[DType.uint8],  # [cols, rows] row-major (transposed)
    in_ptr: ImmutKernelPtr[in_dtype],  # [rows, cols] row-major
    scale_ptr: ImmutKernelPtr[DType.float32],
    rows: Int,
    cols: Int,
) -> None:
    # 32x32 shared-memory tile transpose (Optimization A — see the module
    # comment above). The scale is a device pointer, dereferenced once per
    # thread here rather than passed as a launch scalar; a broadcast read of
    # one device fp32, negligible next to the tile's own global traffic.
    var scale = scale_ptr[0]
    var tile = LayoutTensor[
        in_dtype,
        Layout.row_major(_QT_TILE, _QT_STRIDE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var tiles_rows = ceildiv(rows, _QT_TILE)
    var tiles_cols = ceildiv(cols, _QT_TILE)
    var total_tiles = tiles_rows * tiles_cols

    var tx = Int(thread_idx.x) % _QT_TILE
    var ty = Int(thread_idx.x) // _QT_TILE
    comptime ROW_STEP = _QT_BLOCK // _QT_TILE

    var bt = Int(block_idx.x)
    while bt < total_tiles:
        var tile_r = (bt // tiles_cols) * _QT_TILE
        var tile_c = (bt % tiles_cols) * _QT_TILE

        var r = ty
        while r < _QT_TILE:
            var gr = tile_r + r
            var gc = tile_c + tx
            if gr < rows and gc < cols:
                tile.ptr[r * _QT_STRIDE + tx] = in_ptr[gr * cols + gc]
            r += ROW_STEP
        barrier()

        r = ty
        while r < _QT_TILE:
            var goc = tile_c + r
            var gor = tile_r + tx
            if goc < cols and gor < rows:
                var v = (
                    tile.ptr[tx * _QT_STRIDE + r].cast[DType.float32]() * scale
                )
                out_ptr[goc * rows + gor] = encode_fp8[out_dtype, mode](v)
            r += ROW_STEP
        barrier()

        bt += Int(grid_dim.x)


def quantize_transpose_devscale[
    spec: PrecisionSpec,
    out_dtype: DType,
    in_dtype: DType,
    target: StaticString,
](
    out_ptr: MutKernelPtr[DType.uint8],
    in_ptr: ImmutKernelPtr[in_dtype],
    scale_ptr: ImmutKernelPtr[DType.float32],
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises -> None:
    """Fused transpose+quantize: `out[c,r] = encode(in[r,c] * scale_ptr[0])`
    — i.e. `out` ([cols,rows] row-major) is the fp8-quantized TRANSPOSE of
    `in` ([rows,cols] row-major), computed in one pass (no separate transpose
    kernel, no extra O(rows*cols) round trip through a bf16 scratch buffer).
    `scale_ptr` is a device fp32 scalar, read with no host readback — see
    the module comment above `quantize_devscale`.

    Needed because cuBLASLt's fp8 GEMM is TN-only (`transA=True,
    transB=False`) — see `lowp_gemm_devscale`'s docstring in
    `llmm/matmul.mojo` for the full derivation, summarized here: TN's "free"
    (zero-copy, pure relabeling) duality requires the GEMM's contraction
    dimension to be the *trailing* axis of both operands' natural row-major
    storage. That holds for the forward GEMM's `input`/`weight` (both
    trailing-dim = channels), but not for dgrad's `weight` operand (trailing
    dim is `C`, contraction is `OC`) or for wgrad's `input`/`d_output`
    operands (trailing dims are `C`/`OC`, contraction is `rows`) — those
    (Chunk E's orientations) need a physically transposed fp8 copy, which
    this fused kernel produces directly from the bf16 source (skipping a
    separate bf16-scratch transpose pass). The forward GEMM itself is
    TN-native and never calls this.

    Uses a 32x32 shared-memory tile transpose (Optimization A) so both the
    global read and the global write are coalesced — see the module comment
    above.

    GPU-only, same reasoning as `quantize_devscale`.
    """
    comptime assert is_gpu[target](), "quantize_transpose_devscale is GPU-only"
    var tiles_rows = ceildiv(rows, _QT_TILE)
    var tiles_cols = ceildiv(cols, _QT_TILE)
    var num_blocks = tiles_rows * tiles_cols
    comptime kmode = RoundMode.SR if spec.stochastic_rounding else RoundMode.RNE
    comptime kernel = _quantize_transpose_kernel_devscale[
        in_dtype, out_dtype, kmode
    ]
    var compiled = ctx.compile_function[kernel]()
    ctx.enqueue_function(
        compiled,
        out_ptr,
        in_ptr,
        scale_ptr,
        rows,
        cols,
        grid_dim=(num_blocks,),
        block_dim=(_QT_BLOCK,),
    )


# ===----------------------------------------------------------------------=== #
# quantize_dual_devscale — Optimization B (docs/ai/
# ai_assisted_optimizations_and_benchmarks.md 2026-07-10 fp8-quant-opt entry).
#
# The Optimization-A diagnosis's redundancy map found three (tensor, scale)
# pairs each read from their bf16 source TWICE per step at the SAME scale —
# once by `quantize` (natural layout), once by `quantize_transpose`
# (transposed layout): weight (fwd natural + dgrad transposed), input (fwd
# natural + wgrad transposed), d_output (dgrad natural + wgrad transposed).
# `d_output`'s pair is the only one entirely local to a single call
# (`matmul_bwd_lowp` in llmm/matmul.mojo calls both `matmul_d_input_bwd_lowp`
# and `matmul_d_weight_bwd_lowp`, which are where its natural/transposed
# copies were each separately re-quantized) — weight/input's redundant
# pair crosses the forward/backward call boundary and would need a
# persistent per-layer fp8 cache to fuse (bigger surgery, not attempted in
# this pass; see the doc entry's Optimization D discussion).
#
# This kernel reads its bf16/fp32 source ONCE per tile and emits BOTH the
# natural-layout AND the transposed-layout fp8 copy from that single read:
# the natural output is written during phase 1 (same address it was just
# read from — trivially coalesced, no shared-memory round trip needed for
# it), and the transposed output is written during phase 2 exactly as
# `_quantize_transpose_kernel_devscale` does (Optimization A's coalesced
# tile transpose, reading back from the same shared-memory tile). Halves
# the redundant global-memory traffic for the d_output pair versus calling
# `quantize_devscale` + `quantize_transpose_devscale` separately.
# ===----------------------------------------------------------------------=== #


def _quantize_dual_kernel_devscale[
    in_dtype: DType, out_dtype: DType, mode: Int
](
    nat_out_ptr: MutKernelPtr[DType.uint8],  # [rows, cols] row-major (natural)
    trans_out_ptr: MutKernelPtr[DType.uint8],  # [cols, rows] row-major (T)
    in_ptr: ImmutKernelPtr[in_dtype],  # [rows, cols] row-major
    scale_ptr: ImmutKernelPtr[DType.float32],
    rows: Int,
    cols: Int,
) -> None:
    var scale = scale_ptr[0]
    var tile = LayoutTensor[
        in_dtype,
        Layout.row_major(_QT_TILE, _QT_STRIDE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var tiles_rows = ceildiv(rows, _QT_TILE)
    var tiles_cols = ceildiv(cols, _QT_TILE)
    var total_tiles = tiles_rows * tiles_cols

    var tx = Int(thread_idx.x) % _QT_TILE
    var ty = Int(thread_idx.x) // _QT_TILE
    comptime ROW_STEP = _QT_BLOCK // _QT_TILE

    var bt = Int(block_idx.x)
    while bt < total_tiles:
        var tile_r = (bt // tiles_cols) * _QT_TILE
        var tile_c = (bt % tiles_cols) * _QT_TILE

        # Phase 1 — coalesced load from in_ptr into shared memory, AND
        # (fused, free) a coalesced write of the NATURAL-layout output at
        # the exact same address just read — no separate pass needed for
        # the natural copy.
        var r = ty
        while r < _QT_TILE:
            var gr = tile_r + r
            var gc = tile_c + tx
            if gr < rows and gc < cols:
                var raw = in_ptr[gr * cols + gc]
                tile.ptr[r * _QT_STRIDE + tx] = raw
                var v = raw.cast[DType.float32]() * scale
                nat_out_ptr[gr * cols + gc] = encode_fp8[out_dtype, mode](v)
            r += ROW_STEP
        barrier()

        # Phase 2 — transposed, coalesced write to trans_out_ptr
        # [cols,rows], identical to `_quantize_transpose_kernel_devscale`.
        r = ty
        while r < _QT_TILE:
            var goc = tile_c + r
            var gor = tile_r + tx
            if goc < cols and gor < rows:
                var v = (
                    tile.ptr[tx * _QT_STRIDE + r].cast[DType.float32]() * scale
                )
                trans_out_ptr[goc * rows + gor] = encode_fp8[out_dtype, mode](v)
            r += ROW_STEP
        barrier()

        bt += Int(grid_dim.x)


def quantize_dual_devscale[
    spec: PrecisionSpec,
    out_dtype: DType,
    in_dtype: DType,
    target: StaticString,
](
    nat_out_ptr: MutKernelPtr[DType.uint8],
    trans_out_ptr: MutKernelPtr[DType.uint8],
    in_ptr: ImmutKernelPtr[in_dtype],
    scale_ptr: ImmutKernelPtr[DType.float32],
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises -> None:
    """Dual-output quantize (Optimization B): reads `in_ptr` ([rows,cols]
    row-major) exactly ONCE and writes BOTH `nat_out_ptr` ([rows,cols],
    `out[i] = encode(in[i] * scale)`) and `trans_out_ptr` ([cols,rows],
    the fp8-quantized transpose) — see the module comment above for why
    this is safe/profitable (same tensor, same scale, two orientations
    needed by two different GEMM call sites in the same training step).

    Bit-identical to calling `quantize_devscale` then
    `quantize_transpose_devscale` separately on the same inputs (same
    `encode_fp8` call per output element, same rounding) — purely a memory-
    traffic optimization, not a numerics change.

    GPU-only, same reasoning as `quantize_devscale`.
    """
    comptime assert is_gpu[target](), "quantize_dual_devscale is GPU-only"
    var tiles_rows = ceildiv(rows, _QT_TILE)
    var tiles_cols = ceildiv(cols, _QT_TILE)
    var num_blocks = tiles_rows * tiles_cols
    comptime kmode = RoundMode.SR if spec.stochastic_rounding else RoundMode.RNE
    comptime kernel = _quantize_dual_kernel_devscale[in_dtype, out_dtype, kmode]
    var compiled = ctx.compile_function[kernel]()
    ctx.enqueue_function(
        compiled,
        nat_out_ptr,
        trans_out_ptr,
        in_ptr,
        scale_ptr,
        rows,
        cols,
        grid_dim=(num_blocks,),
        block_dim=(_QT_BLOCK,),
    )


def precision_spec[name: StaticString]() -> PrecisionSpec:
    """Resolve a `LLMM_PRECISION` name ("fp32" | "bf16" | "fp8" | "fp4") to its
    `PrecisionSpec`. "fp8" -> `FP8_SPEC`; "fp4" -> `FP4_SPEC`; "fp32"/"bf16" ->
    an inert placeholder (never consulted — see `_INERT_SPEC`).

    "fp4" resolving to a real spec (rather than a comptime error) means
    `-D LLMM_PRECISION=fp4` now compiles past `train_gpt2.mojo`'s unconditional
    `comptime SPEC = precision_spec[PRECISION]()` — but that is *only* this
    one seam closing, not a functional fp4 trainer: no GEMM call site in
    `train_gpt2.mojo` reads `SPEC`/dispatches to `lowp_gemm_fp4`
    (`llmm/matmul.mojo`) yet, and `LLMM_PRECISION=fp4` is not one of this
    repo's gated build targets (`build`/`build-bf16`/`build-fp8`). Wiring the
    actual training-loop GEMM/quantize call sites to `FP4_SPEC` is the next
    chunk's job (same shape as fp8's Chunk A -> Chunks D/E progression: the
    flag existing and resolving to a real spec precedes the GEMM wiring, it
    does not imply it).
    """

    comptime if name == "fp8":
        return FP8_SPEC
    elif name == "fp4":
        return FP4_SPEC
    elif name == "fp32" or name == "bf16":
        return _INERT_SPEC
    else:
        comptime assert (
            False
        ), "unknown LLMM_PRECISION value (expected fp32 | bf16 | fp8 | fp4)"
