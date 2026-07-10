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


def precision_spec[name: StaticString]() -> PrecisionSpec:
    """Resolve a `LLMM_PRECISION` name ("fp32" | "bf16" | "fp8" | "fp4") to its
    `PrecisionSpec`. "fp8" -> `FP8_SPEC`; "fp32"/"bf16" -> an inert placeholder
    (never consulted — see `_INERT_SPEC`). "fp4" is a documented extension seam
    (docs/ai/fp8_training_design.md §3) not yet implemented; using it is a clear
    comptime error rather than a silently-wrong spec.
    """

    comptime if name == "fp8":
        return FP8_SPEC
    elif name == "fp32" or name == "bf16":
        return _INERT_SPEC
    elif name == "fp4":
        comptime assert False, (
            "LLMM_PRECISION=fp4 is a documented extension seam"
            " (docs/ai/fp8_training_design.md §3) not yet implemented"
        )
    else:
        comptime assert (
            False
        ), "unknown LLMM_PRECISION value (expected fp32 | bf16 | fp8 | fp4)"
