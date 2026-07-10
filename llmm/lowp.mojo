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
from std.gpu import block_dim, block_idx, thread_idx
from std.sys import simd_width_of, size_of

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
# fp8's `quantize`/`quantize_transpose` read `spec.stochastic_rounding`
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
def _fp8_encode_rne[
    exp_bits: Int, mant_bits: Int, bias: Int, max_normal: Float32
](x: Float32) -> UInt8:
    """Round-to-nearest-even encode of `x` into an `(exp_bits, mant_bits,
    bias)` fp8 format's bit pattern, saturating (never inf/nan) at
    `max_normal`. Generic core shared by `encode_e4m3`/`encode_e5m2` — see
    module docstring for why this must stay pure integer/fp32 arithmetic
    (no fp8-typed intermediate) to be GPU-safe.
    """
    var xbits = _f32_bits(x)
    var sign_bit = UInt8((xbits >> 31) & 1) << 7
    var ax = abs(x)
    # `not (ax <= max_normal)` catches ax > max_normal AND NaN (NaN compares
    # false against everything under IEEE rules) — both saturate rather than
    # propagate, matching the native host cast's observed behavior.
    if not (ax <= max_normal):
        ax = max_normal
    if ax == Float32(0.0):
        return sign_bit

    var bits = _f32_bits(ax)
    var f32_exp = Int((bits >> 23) & 0xFF) - 127
    var full_mant: UInt32 = (bits & 0x7FFFFF) | (
        UInt32(1) << 23
    )  # 24-bit, implicit 1

    comptime min_normal_exp = 1 - bias
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
        # it — see module docstring / lowp_gemm design notes).
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
        return _fp8_encode_rne[4, 3, 7, E4M3_MAX](x)
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
        return _fp8_encode_rne[5, 2, 15, E5M2_MAX](x)
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
    `quantize` (below) uses so a call site is parameterized by `out_dtype`
    (`spec.fwd_dtype` or `spec.bwd_dtype`), not a hardcoded format."""
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


def _quantize_kernel[
    in_dtype: DType, out_dtype: DType, mode: Int
](
    out_ptr: MutKernelPtr[DType.uint8],
    in_ptr: ImmutKernelPtr[in_dtype],
    scale: Float32,
    n: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < n:
        var v = in_ptr[idx].cast[DType.float32]() * scale
        out_ptr[idx] = encode_fp8[out_dtype, mode](v)


def quantize[
    spec: PrecisionSpec,
    out_dtype: DType,
    in_dtype: DType,
    target: StaticString,
](
    out_ptr: MutKernelPtr[DType.uint8],
    in_ptr: ImmutKernelPtr[in_dtype],
    scale: Float32,
    n: Int,
    ctx: DeviceContext,
) raises -> None:
    """Bf16/fp32 -> fp8 quantize: `out[i] = encode(in[i] * scale)`.

    GPU-only (comptime-asserted): fp8 is a device-only GEMM transient
    (docs/ai/fp8_training_design.md §1.1), and the manual `encode_fp8` above
    is deliberately written to be GPU-safe (integer/fp32 arithmetic only, no
    fp8-typed intermediate — see the module docstring; plain `pop.cast
    f8eXmY -> {f32,bf16}` is broken on this toolchain's GPU target once the
    upcast feeds arithmetic, per `tests/probe_fp8/RESULTS.md` probes 2/2c/3).

    `scale` is a single fp32 value the *caller* supplies — this function does
    not compute amax or a delayed-scaling schedule; that is Chunk C's
    `AmaxState`/`compute_amax`/`update_scale` (§3), deliberately kept out of
    this file's Chunk-B surface so the two chunks don't collide on the same
    functions while landing in parallel. A caller wires
    `quantize[...](..., scale=a_state.scale, ...)` once Chunk C lands.
    """
    comptime assert is_gpu[target](), (
        "quantize is GPU-only per docs/ai/fp8_training_design.md landmine #1"
        " (low-precision kernels must never be instantiated for the cpu"
        " target)"
    )
    comptime BLOCK_SIZE = 256
    var num_blocks = ceildiv(n, BLOCK_SIZE)
    comptime kmode = RoundMode.SR if spec.stochastic_rounding else RoundMode.RNE
    comptime kernel = _quantize_kernel[in_dtype, out_dtype, kmode]
    var compiled = ctx.compile_function[kernel]()
    ctx.enqueue_function(
        compiled,
        out_ptr,
        in_ptr,
        scale,
        n,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )


# ===----------------------------------------------------------------------=== #
# Device-pointer-scale variants (Chunk D).
#
# `quantize`/`quantize_transpose` above take `scale: Float32` — a HOST value.
# That is exactly right for Chunk B's own gate (a host-computed amax/scale in
# a unit test), but Chunk C's `AmaxState.scale`/`scale_inv` (llmm/amax.mojo)
# are DEVICE-resident fp32 scalars, deliberately never read back to host on
# the training step path (design §4's landmine-2 audit: "Host readback? no").
# A GPU kernel launch's scalar arguments are copied host->device at launch
# time regardless of whether the source is a host variable or (as here) a
# device pointer being dereferenced — so passing `scale: Float32` to
# `quantize` would force a host readback of `AmaxState.scale` right before
# every single quantize call, once per fp8 GEMM operand per step. These
# `_devscale` twins take `scale_ptr: ImmutKernelPtr[DType.float32]` instead
# and dereference it *inside* the kernel (`scale_ptr[0]`), so the scale
# value never leaves the device — the call site passes
# `AmaxState.scale`'s own device pointer straight through. Otherwise
# identical to `quantize`/`quantize_transpose` (same encode path, same
# stochastic-rounding seam, same GPU-only guard).
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
    """Device-pointer-scale twin of `quantize` — see the module comment
    above. `scale_ptr` is a device fp32 scalar (e.g. `AmaxState.scale`'s
    `unsafe_ptr()`); the kernel reads it directly, no host readback.
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


def _quantize_transpose_kernel[
    in_dtype: DType, out_dtype: DType, mode: Int
](
    out_ptr: MutKernelPtr[DType.uint8],  # [cols, rows] row-major (transposed)
    in_ptr: ImmutKernelPtr[in_dtype],  # [rows, cols] row-major
    scale: Float32,
    rows: Int,
    cols: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < rows * cols:
        var r = idx // cols
        var c = idx % cols
        var v = in_ptr[idx].cast[DType.float32]() * scale
        out_ptr[c * rows + r] = encode_fp8[out_dtype, mode](v)


def quantize_transpose[
    spec: PrecisionSpec,
    out_dtype: DType,
    in_dtype: DType,
    target: StaticString,
](
    out_ptr: MutKernelPtr[DType.uint8],
    in_ptr: ImmutKernelPtr[in_dtype],
    scale: Float32,
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises -> None:
    """Fused transpose+quantize: `out[c,r] = encode(in[r,c] * scale)` — i.e.
    `out` ([cols,rows] row-major) is the fp8-quantized TRANSPOSE of `in`
    ([rows,cols] row-major), computed in one pass (no separate transpose
    kernel, no extra O(rows*cols) round trip through a bf16 scratch buffer).

    Needed because cuBLASLt's fp8 GEMM is TN-only (`transA=True,
    transB=False`) — see `lowp_gemm`'s docstring in `llmm/matmul.mojo` for the
    full derivation, summarized here: TN's "free" (zero-copy, pure
    relabeling) duality requires the GEMM's contraction dimension to be the
    *trailing* axis of both operands' natural row-major storage. That holds
    for the forward GEMM's `input`/`weight` (both trailing-dim = channels),
    but not for dgrad's `weight` operand (trailing dim is `C`, contraction is
    `OC`) or for wgrad's `input`/`d_output` operands (trailing dims are
    `C`/`OC`, contraction is `rows`) — those need a physically transposed
    fp8 copy, which this fused kernel produces directly from the bf16 source
    (skipping a separate bf16-scratch transpose pass).

    GPU-only, same reasoning as `quantize`.
    """
    comptime assert is_gpu[target](), "quantize_transpose is GPU-only"
    comptime BLOCK_SIZE = 256
    var total = rows * cols
    var num_blocks = ceildiv(total, BLOCK_SIZE)
    comptime kmode = RoundMode.SR if spec.stochastic_rounding else RoundMode.RNE
    comptime kernel = _quantize_transpose_kernel[in_dtype, out_dtype, kmode]
    var compiled = ctx.compile_function[kernel]()
    ctx.enqueue_function(
        compiled,
        out_ptr,
        in_ptr,
        scale,
        rows,
        cols,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )


def _quantize_transpose_kernel_devscale[
    in_dtype: DType, out_dtype: DType, mode: Int
](
    out_ptr: MutKernelPtr[DType.uint8],  # [cols, rows] row-major (transposed)
    in_ptr: ImmutKernelPtr[in_dtype],  # [rows, cols] row-major
    scale_ptr: ImmutKernelPtr[DType.float32],
    rows: Int,
    cols: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < rows * cols:
        var r = idx // cols
        var c = idx % cols
        var v = in_ptr[idx].cast[DType.float32]() * scale_ptr[0]
        out_ptr[c * rows + r] = encode_fp8[out_dtype, mode](v)


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
    """Device-pointer-scale twin of `quantize_transpose` — see the module
    comment above `quantize_devscale`. Not needed by Chunk D (the forward
    GEMM is TN-native — `transpose_a=False, transpose_b=False`, see
    `lowp_gemm`'s docstring in llmm/matmul.mojo), but Chunk E's dgrad/wgrad
    orientations need a transposed fp8 copy AND a device-resident scale, so
    this twin is provided here alongside `quantize_devscale` rather than
    left for Chunk E to duplicate.
    """
    comptime assert is_gpu[target](), "quantize_transpose_devscale is GPU-only"
    comptime BLOCK_SIZE = 256
    var total = rows * cols
    var num_blocks = ceildiv(total, BLOCK_SIZE)
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
        block_dim=(BLOCK_SIZE,),
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
