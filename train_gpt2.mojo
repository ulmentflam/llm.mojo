from std.math import sqrt, ceildiv, isnan, nan, exp, log
from std.os import getenv
from std.python import Python
from std.sys import (
    argv,
    has_accelerator,
    has_apple_gpu_accelerator,
    simd_width_of,
)
from std.time import global_perf_counter_ns
from std.algorithm import sync_parallelize
from std.gpu.host.info import is_cpu, is_gpu
from std.sys import get_defined_int, get_defined_string, is_defined
from std.gpu.host import (
    DeviceContext,
    HostBuffer,
    DeviceBuffer,
    DeviceAttribute,
)
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import alloc, UnsafePointer, memcpy, memset_zero

from llmm.io import read_and_copy
from llmm.lowp import (
    FP8_SPEC,
    FP8_STATIC_SCALES,
    fp8_static_scale,
)
from llmm.amax import AmaxState
from llmm.dataloader import DataLoader
from llmm.safetensors import SafetensorsFile, read_hf_gpt2_config
from llmm.checkpointing import (
    CheckpointConfig,
    TrainingState,
    write_model_checkpoint,
    read_model_checkpoint,
    write_state_checkpoint,
    read_state_checkpoint,
    make_training_state,
    restore_dataloader_state,
)
from llmm.encoder import encoder_fwd, encoder_bwd, build_wte_buckets
from llmm.layernorm import (
    layernorm_fwd,
    layernorm_bwd,
    layernorm_fused_residual_fwd,
    layernorm_fused_residual_bwd,
    residual_grad_broadcast,
)
from llmm.attention import attention_fwd, attention_bwd, KVCache, KVCachePtr
from llmm.matmul import (
    matmul_fwd,
    matmul_bwd,
    matmul_fwd_lowp,
    matmul_bwd_lowp,
    matmul_fwd_fp4,
    matmul_bwd_fp4,
)
from llmm.softmax import softmax_fwd, softmax_bwd
from llmm.crossentropy import crossentropy_ohe_fwd, crossentropy_ohe_bwd
from llmm.global_norm import (
    global_norm_squared,
    global_norm_squared_cpu,
    global_norm_squared_gpu,
    global_norm_aggregate_gpu,
)
from llmm.adamw import adamw_update, AdamWConfig
from llmm.fused_classifier import fused_classifier
from llmm.scheduler import LearningRateScheduler
from llmm.sampler import random_f32, sample_softmax
from llmm.rand import MT19937, normal_
from llmm.mfu import estimate_mfu
from llmm.tokenizer import Tokenizer, safe_print
from llmm.memory import (
    ImmutKernelPtr,
    ImmutMemPtr,
    MutMemPtr,
    MutKernelPtr,
    as_immut_kernel,
    as_immut_kernel_from_mut,
    as_mut_kernel,
    rebind_mut_mem,
)
from llmm.zero import (
    ZeroContext,
    ShardedParameter,
    CpuCoordinator,
)
from llmm.vendor import HAS_METAL


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #


comptime WORLD_SIZE_DEF = get_defined_int["WORLD_SIZE", 1]()

comptime NUM_PARAMETER_TENSORS = 16
comptime NUM_ACTIVATION_TENSORS = 26


# Parameter/activation/gradient precision. Defaults to fp32; build with
# -D LLMM_PRECISION=bf16|fp8|fp4, or the -D LLMM_BF16=1 back-compat alias for
# bf16, for mixed-precision training. The layernorm/attention statistics stay
# fp32 (StatsDType) and the optimizer keeps fp32 moments + an fp32 master copy
# of the weights (MASTER_DTYPE) — matching llm.c — so the low-precision params
# are only ever a rounded view of the fp32 math.
#
# fp8/fp4 are transient dtypes used only inside the GEMM, never the
# parameter/activation/gradient storage dtype — so STORAGE_DTYPE collapses
# fp8/fp4 onto bfloat16, and USE_BF16 (fp32-master + bf16 storage) is true
# for fp8/fp4 too.


# ===----------------------------------------------------------------------=== #
# Build-flag registry (comptime -D unless marked env). Single source of truth;
# defining files hold the mechanics.
#
# Precision axis
#   LLMM_PRECISION=fp32|bf16|fp8|fp4   default fp32. Master axis (below).
#   LLMM_BF16=1                        alias for LLMM_PRECISION=bf16;
#                                      comptime-error if both set inconsistently.
# FP8 (all inert unless LLMM_PRECISION=fp8)
#   LLMM_FP8_FWD_ONLY=1        keep fp8 forward, force all 4 backward sites bf16.
#   LLMM_FP8_SITE_QKV=0        per-site fp8 off-switch (default 1=on); a site
#   LLMM_FP8_SITE_ATTN_PROJ=0  disabled here is bf16 in BOTH fwd and bwd (the
#   LLMM_FP8_SITE_FC=0         transpose cache + AmaxState scale are only valid
#   LLMM_FP8_SITE_PROJ=0       if that site's forward ran this step).
#   LLMM_FP8_STATIC_SCALES=1   calibrated constant scales; skips (never
#                              instantiates) the amax/update_scale kernels.
#                              Defined in llmm/lowp.mojo.
#   LLMM_FP8_STATIC_D36=1      select the d36 constant table (default d12).
#                              Only meaningful with LLMM_FP8_STATIC_SCALES.
#   LLMM_FP8_FAST_ACCUM=1      cuBLASLt fast accumulation, FORWARD GEMM only;
#                              dgrad/wgrad always precise. llmm/matmul.mojo.
# FP4 (all inert unless LLMM_PRECISION=fp4)
#   LLMM_FP4_FIRST=<int>       first N blocks stay bf16 (default 2).
#   LLMM_FP4_LAST=<int>        fp4 range end; default -1 = num_layers-2,
#                              resolved at runtime (_layer_in_fp4_range).
#   LLMM_FP4_NO_RHT=1          ablation: disable the Wgrad random Hadamard
#                              transform. llmm/matmul.mojo.
# Stochastic rounding
#   LLMM_SR_MASTER=1           SR on the master->bf16 param store (llmm/adamw.mojo).
#   LLMM_SR_SEED=<int>         shared SR seed, default 1746221221 (adamw +
#                              nvfp4; decorrelated by stream id, see
#                              llmm/rng_device.mojo's stream registry).
# Numerics / dispatch
#   LLMM_NO_TF32=1             true IEEE fp32 GEMMs (llmm/vendor.mojo).
#   LLMM_FORCE_PORTABLE_GPU=1  vendor-neutral GPU path (llmm/vendor.mojo).
#   LLMM_DISABLE_METAL=1       CPU fallback on Apple GPU (llmm/vendor.mojo).
#   WORLD_SIZE=<int>           comptime monomorphization value, default 1;
#                              runtime env WORLD_SIZE must match.
# Runtime env vars (read at startup, not -D)
#   LLMM_USE_CPU=1             force CPU dispatch (raises under bf16-storage
#                              builds, which includes fp8/fp4).
#   LLMM_OUTPUT_DIR=<path>     override output_log_dir.
#   LLMM_SAVE_EVERY=<int>      override checkpoint_every.
#   LLMM_RECOMPUTE=1           test_gpt2 only: activation-recompute build in
#                              run_test. Not read by train_gpt2 (see docs).
# Profiling (out of precision scope): LLMM_ATTN_PROFILE, LLMM_PROFILE_*,
#   LLMM_THREAD_TRACE, LLMM_TRACE.
# ===----------------------------------------------------------------------=== #
def _resolve_precision() -> StaticString:
    """Resolve the `LLMM_PRECISION` axis, honoring the `LLMM_BF16=1` back-
    compat alias (== `LLMM_PRECISION=bf16`). Comptime-errors if both are set
    inconsistently, or if `LLMM_PRECISION` is set to something other than
    fp32 | bf16 | fp8 | fp4.
    """
    comptime bf16_alias = is_defined["LLMM_BF16"]()
    comptime has_precision = is_defined["LLMM_PRECISION"]()
    comptime precision_str = get_defined_string["LLMM_PRECISION", "fp32"]()

    comptime if bf16_alias and has_precision:
        comptime assert precision_str == "bf16", (
            "LLMM_BF16=1 and LLMM_PRECISION set to something other than bf16"
            " are inconsistent — set only one"
        )
        return "bf16"
    elif bf16_alias:
        return "bf16"
    else:
        comptime assert (
            precision_str == "fp32"
            or precision_str == "bf16"
            or precision_str == "fp8"
            or precision_str == "fp4"
        ), "unknown LLMM_PRECISION value (expected fp32 | bf16 | fp8 | fp4)"
        return precision_str


comptime PRECISION = _resolve_precision()
comptime LOWP_ENABLED = PRECISION == "fp8" or PRECISION == "fp4"
# -D LLMM_FP8_FWD_ONLY=1 keeps the fp8 forward linears but forces the four
# per-block backward sites onto their bf16 matmul_bwd branch. Default unset is
# full fp8 backward.

# fp4's backward (matmul_bwd_fp4) is a separate branch at the two
# FP4-eligible MLP sites: it has a different signature (no AmaxStates; an
# sr_step counter) than fp8's matmul_bwd_lowp, so it is not folded into
# FP8_BWD_ENABLED.
comptime FP8_BWD_ENABLED = (PRECISION == "fp8") and not is_defined[
    "LLMM_FP8_FWD_ONLY"
]()

# Per-site fp8 gates (default all on). Setting -D LLMM_FP8_SITE_<SITE>=0
# routes that single site to the bf16 matmul_fwd/matmul_bwd branch in BOTH
# passes — a site disabled in forward MUST be disabled in backward, because
# matmul_fwd_lowp's transpose cache and the site's AmaxState scale are only
# valid if that site's forward ran this step.
comptime FP8_SITE_QKV = get_defined_int["LLMM_FP8_SITE_QKV", 1]() != 0
comptime FP8_SITE_ATTN_PROJ = (
    get_defined_int["LLMM_FP8_SITE_ATTN_PROJ", 1]() != 0
)
comptime FP8_SITE_FC = get_defined_int["LLMM_FP8_SITE_FC", 1]() != 0
comptime FP8_SITE_PROJ = get_defined_int["LLMM_FP8_SITE_PROJ", 1]() != 0
comptime STORAGE_DTYPE = (
    DType.float32 if PRECISION == "fp32" else DType.bfloat16
)
comptime GPT2_DTYPE = STORAGE_DTYPE  # keep the existing name/usage
comptime MASTER_DTYPE = DType.float32
# `USE_BF16` keeps its exact current meaning ("keep an fp32 master + bf16
# storage"); fp8/fp4 storage is also bf16 (see above), so they inherit the
# same fp32-master path — llmm/adamw.mojo, llmm/zero.mojo, and the
# `_dispatch_cpu` GPU-only guard (which raises whenever `USE_BF16`) need no
# change and automatically also gate fp8/fp4 off the CPU target (landmine #1).
comptime USE_BF16 = STORAGE_DTYPE == DType.bfloat16

# FP4 layer-range policy: NVFP4 applies ONLY to the MLP linears (fc/fc_proj)
# of MIDDLE transformer blocks; qkv/attn_proj/attention/LN/embeddings/LM-head
# stay bf16 EVERYWHERE regardless of PRECISION (see the four call sites below
# — only two of the four Matmul sites even consult these bounds).
# `LLMM_FP4_FIRST` defaults to 2 (first 2 blocks stay bf16). `LLMM_FP4_LAST`
# defaults to `num_layer - 2` (final 2 blocks stay bf16) — resolved at
# runtime (`-1` sentinel below) since `num_layer` is a runtime value (the
# model descriptor, e.g. `-e d12`, is not known at comptime), not a comptime
# constant; override either bound with
# `-D LLMM_FP4_FIRST=<int>`/`-D LLMM_FP4_LAST=<int>` (e.g. for the 12-layer
# d12 default this is layers [2, 10), i.e. layers 2..9 in fp4). Only
# elaborated/consulted under `PRECISION == "fp4"` — inert (unread) for every
# other PRECISION value.
comptime FP4_FIRST = get_defined_int["LLMM_FP4_FIRST", 2]()
comptime FP4_LAST_RAW = get_defined_int["LLMM_FP4_LAST", -1]()
comptime GPT2_MAGIC = 20240520
# llm.c's model-file magic (the HF starter-pack gpt2_124M*.bin that `make data`
# downloads); same header layout and version convention as ours.
comptime GPT2_MAGIC_LEGACY = 20240326
comptime EPSILON = 1e-5
comptime UNROLL = 4

# From-scratch init draws the same random numbers as llm.c (and PyTorch) so the
# initial weights are bit-identical: seed 42, weights ~N(0, 0.02), the residual
# projections additionally scaled by 1/sqrt(2*L). See gpt_build_from_descriptor.
comptime INIT_RNG_SEED = 42
comptime INIT_WEIGHT_STD = Float32(0.02)
# GPT-2 / GPT-3 share the 50257-token tokenizer, padded to a multiple of 128.
comptime SCRATCH_VOCAB_SIZE = 50257
comptime SCRATCH_PADDED_VOCAB_SIZE = 50304


@always_inline
def _layer_in_fp4_range(layer: Int, num_layers: Int) -> Bool:
    """True if `layer` falls in the FP4-eligible middle-block range (the
    `FP4_FIRST`/`FP4_LAST_RAW` comptime constants above). `LLMM_FP4_LAST`'s
    `-1` sentinel resolves to `num_layers - 2` here (runtime, since
    `num_layers` is not a comptime value). Only meaningful under
    `PRECISION == "fp4"`; call sites comptime-gate on that before consulting
    this (see the FC/Proj matmul sites in the forward pass).
    """
    var fp4_last = FP4_LAST_RAW if FP4_LAST_RAW >= 0 else num_layers - 2
    return layer >= FP4_FIRST and layer < fp4_last


# ===----------------------------------------------------------------------=== #
# printf0 — rank-0-only print, mirroring llm.c's printf0 macro
# ===----------------------------------------------------------------------=== #


@always_inline
def printf0(rank: Int, msg: String):
    """Print `msg` only on rank 0 (the master process), like llm.c's printf0."""
    if rank == 0:
        print(msg)


comptime OUTLIER_DETECTOR_WINDOW_SIZE = 128


struct OutlierDetector(Copyable, Movable):
    """Sliding-window z-score detector for the loss and gradient norm, ported
    from llm.c's outlier_detector.h. update() returns the z-score of the new
    value against the window, or NaN until the window (128 samples) fills.
    """

    var buffer: List[Float64]
    var count: Int
    var index: Int
    var sum: Float64
    var sum_sq: Float64

    def __init__(out self):
        self.buffer = List[Float64]()
        for _ in range(OUTLIER_DETECTOR_WINDOW_SIZE):
            self.buffer.append(0.0)
        self.count = 0
        self.index = 0
        self.sum = 0.0
        self.sum_sq = 0.0

    def update(mut self, new_value: Float64) -> Float64:
        if self.count < OUTLIER_DETECTOR_WINDOW_SIZE:
            # Still building up the window: record and report "not enough data".
            self.buffer[self.count] = new_value
            self.sum += new_value
            self.sum_sq += new_value * new_value
            self.count += 1
            return nan[DType.float64]()

        # Window full: pop the oldest value, push the new one, return z-score.
        var old_value = self.buffer[self.index]
        self.sum -= old_value
        self.sum_sq -= old_value * old_value
        self.buffer[self.index] = new_value
        self.sum += new_value
        self.sum_sq += new_value * new_value
        self.index = (self.index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE
        var mean = self.sum / Float64(OUTLIER_DETECTOR_WINDOW_SIZE)
        var variance = (
            self.sum_sq / Float64(OUTLIER_DETECTOR_WINDOW_SIZE)
        ) - mean * mean
        var std_dev = sqrt(variance)
        if std_dev == 0.0:
            return 0.0
        return (new_value - mean) / std_dev


def _ffmt(x: Float64, decimals: Int) -> String:
    """Format `x` with a fixed number of decimal places (no printf in Mojo)."""
    if isnan(x):
        return "nan"
    var neg = x < 0.0
    var v = -x if neg else x
    var scale = 1
    for _ in range(decimals):
        scale *= 10
    var scaled = Int(v * Float64(scale) + 0.5)  # round half up
    var int_part = scaled // scale
    var frac_part = scaled % scale
    var out = String(int_part)
    if decimals > 0:
        var frac = String(frac_part)
        while frac.byte_length() < decimals:
            frac = "0" + frac
        out += "." + frac
    return ("-" + out) if neg else out


def _zfmt(z: Float64) -> String:
    """Format a z-score with an explicit sign, like llm.c's `%+.2f`."""
    if isnan(z):
        return "nan"
    var body = _ffmt(z, 2)
    return body if z < 0.0 else "+" + body


# ===----------------------------------------------------------------------=== #
# Parameter Tensors
# ===----------------------------------------------------------------------=== #


struct Parameters:
    comptime wte = 0
    comptime wpe = 1
    comptime ln_1_gamma = 2
    comptime ln_1_beta = 3
    comptime qkv_weight = 4
    comptime qkv_bias = 5
    comptime attn_proj_weight = 6
    comptime attn_proj_bias = 7
    comptime ln_2_gamma = 8
    comptime ln_2_beta = 9
    comptime fc_weight = 10
    comptime fc_bias = 11
    comptime proj_weight = 12
    comptime proj_bias = 13
    comptime ln_f_gamma = 14
    comptime ln_f_beta = 15


struct ParameterTensors[
    dtype: DType = DType.float32,
]:
    var params_memory: MutMemPtr[Self.dtype]

    # Encoder
    var wte: MutMemPtr[Self.dtype]  # (V, C)
    var wpe: MutMemPtr[Self.dtype]  # (max(T), C)

    # Layer Norm 1
    var ln_1_gamma: MutMemPtr[Self.dtype]  # (L, C)
    var ln_1_beta: MutMemPtr[Self.dtype]  # (L, C)

    # Attention
    var qkv_weight: MutMemPtr[Self.dtype]  # (L, 3*C, C)
    var qkv_bias: MutMemPtr[Self.dtype]  # (L, 3*C)
    var attn_proj_weight: MutMemPtr[Self.dtype]  # (L, C, C)
    var attn_proj_bias: MutMemPtr[Self.dtype]  # (L, C)

    # Layer Norm 2
    var ln_2_gamma: MutMemPtr[Self.dtype]  # (L, C)
    var ln_2_beta: MutMemPtr[Self.dtype]  # (L, C)

    # MLP
    var fc_weight: MutMemPtr[Self.dtype]  # (L, 4*C, C)
    var fc_bias: MutMemPtr[Self.dtype]  # (L, 4*C)
    var proj_weight: MutMemPtr[Self.dtype]  # (L, C, 4*C)
    var proj_bias: MutMemPtr[Self.dtype]  # (L, C)

    # Layer Norm Final
    var ln_f_gamma: MutMemPtr[Self.dtype]  # (C, )
    var ln_f_beta: MutMemPtr[Self.dtype]  # (C, )

    def __init__(out self):
        var zero = 0
        self.params_memory = MutMemPtr[Self.dtype](unsafe_from_address=zero)

        self.wte = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.wpe = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_1_gamma = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_1_beta = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.qkv_weight = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.qkv_bias = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.attn_proj_weight = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.attn_proj_bias = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_2_gamma = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_2_beta = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.fc_weight = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.fc_bias = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.proj_weight = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.proj_bias = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_f_gamma = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        self.ln_f_beta = MutMemPtr[Self.dtype](unsafe_from_address=zero)

    def point_parameters(
        mut self,
        param_sizes: List[Int],
        params_memory: MutMemPtr[Self.dtype],
    ) -> None:
        self.params_memory = params_memory

        comptime ParamPtr = MutMemPtr[Self.dtype]
        comptime ParamPtrPtr = UnsafePointer[ParamPtr, MutAnyOrigin]

        var ptrs = List[ParamPtrPtr]()
        ptrs.append(ParamPtrPtr(to=self.wte))
        ptrs.append(ParamPtrPtr(to=self.wpe))
        ptrs.append(ParamPtrPtr(to=self.ln_1_gamma))
        ptrs.append(ParamPtrPtr(to=self.ln_1_beta))
        ptrs.append(ParamPtrPtr(to=self.qkv_weight))
        ptrs.append(ParamPtrPtr(to=self.qkv_bias))
        ptrs.append(ParamPtrPtr(to=self.attn_proj_weight))
        ptrs.append(ParamPtrPtr(to=self.attn_proj_bias))
        ptrs.append(ParamPtrPtr(to=self.ln_2_gamma))
        ptrs.append(ParamPtrPtr(to=self.ln_2_beta))
        ptrs.append(ParamPtrPtr(to=self.fc_weight))
        ptrs.append(ParamPtrPtr(to=self.fc_bias))
        ptrs.append(ParamPtrPtr(to=self.proj_weight))
        ptrs.append(ParamPtrPtr(to=self.proj_bias))
        ptrs.append(ParamPtrPtr(to=self.ln_f_gamma))
        ptrs.append(ParamPtrPtr(to=self.ln_f_beta))

        var params_memory_iterator = self.params_memory

        for i in range(len(param_sizes)):
            ptrs[i][] = params_memory_iterator
            params_memory_iterator += param_sizes[i]


# ===----------------------------------------------------------------------=== #
# Activation Tensors
# ===----------------------------------------------------------------------=== #


struct Activations:
    comptime encoded = 0
    comptime ln_1 = 1
    comptime ln_1_mean = 2
    comptime ln_1_rstd = 3
    comptime qkv = 4
    comptime lse = 5
    comptime attn = 6
    comptime attn_proj = 7
    comptime residual_2 = 8
    comptime ln_2 = 9
    comptime ln_2_mean = 10
    comptime ln_2_rstd = 11
    comptime fch = 12
    comptime fch_gelu = 13
    comptime fc_proj = 14
    comptime residual_3 = 15
    comptime ln_f = 16
    comptime ln_f_mean = 17
    comptime ln_f_rstd = 18
    comptime logits = 19
    comptime losses = 20
    comptime q = 21
    comptime k = 22
    comptime v = 23
    comptime attn_merged = 24
    # Stored softmax probabilities [L, B, NH, T, T] (bf16). Persisting them lets
    # the backward READ P instead of recomputing QKᵀ (llm.c's approach), dropping
    # one big batched matmul per layer.
    comptime att_probs = 25


# The layernorm/attention statistics (mean, rstd, lse) and the per-token losses
# are always kept in fp32 (StatsDType), independent of the activation precision —
# matching llm.c. They live in a separate fp32 buffer so the main activations can
# be a lower precision (bf16/fp8) while these stay numerically safe.
comptime StatsDType = DType.float32


struct ActivationTensors[
    dtype: DType = DType.float32,
]:
    # Encoder
    var encoded: MutMemPtr[Self.dtype]  # (B, T, C)

    # Layer Norm 1
    var ln_1: MutMemPtr[Self.dtype]  # (L, B, T, C)
    var ln_1_mean: MutMemPtr[StatsDType]  # (L, B, T)  fp32
    var ln_1_rstd: MutMemPtr[StatsDType]  # (L, B, T)  fp32

    # Attention
    var qkv: MutMemPtr[Self.dtype]  # (L, B, T, 3*C)
    var lse: MutMemPtr[StatsDType]  # (L, B, NH, T)  fp32
    var attn: MutMemPtr[Self.dtype]  # (L, B, T, C)
    var attn_proj: MutMemPtr[Self.dtype]  # (L, B, T, C)
    var residual_2: MutMemPtr[Self.dtype]  # (L, B, T, C)

    # Layer Norm 2
    var ln_2: MutMemPtr[Self.dtype]  # (L, B, T, C)
    var ln_2_mean: MutMemPtr[StatsDType]  # (L, B, T)  fp32
    var ln_2_rstd: MutMemPtr[StatsDType]  # (L, B, T)  fp32

    # MLP
    var fch: MutMemPtr[Self.dtype]  # (L, B, T, 4*C)
    var fch_gelu: MutMemPtr[Self.dtype]  # (L, B, T, 4*C)
    var fc_proj: MutMemPtr[Self.dtype]  # (L, B, T, C)
    var residual_3: MutMemPtr[Self.dtype]  # (L, B, T, C)

    # Layer Norm Final
    var ln_f: MutMemPtr[Self.dtype]  # (B, T, C)
    var ln_f_mean: MutMemPtr[StatsDType]  # (B, T)  fp32
    var ln_f_rstd: MutMemPtr[StatsDType]  # (B, T)  fp32
    var logits: MutMemPtr[Self.dtype]  # (B, T, V)
    var losses: MutMemPtr[StatsDType]  # (B, T)  fp32

    # Scratch / Split-attention
    var q: MutMemPtr[Self.dtype]  # (L, B, NH, T, HS)
    var k: MutMemPtr[Self.dtype]  # (L, B, NH, T, HS)
    var v: MutMemPtr[Self.dtype]  # (L, B, NH, T, HS)
    var attn_merged: MutMemPtr[Self.dtype]  # (L, B, T, C)
    var att_probs: MutMemPtr[
        Self.dtype
    ]  # (L, B, NH, T, T) stored softmax probs

    def __init__(out self):
        var zero = 0
        var null_a = MutMemPtr[Self.dtype](unsafe_from_address=zero)
        var null_s = MutMemPtr[StatsDType](unsafe_from_address=zero)
        self.encoded = null_a
        self.ln_1 = null_a
        self.ln_1_mean = null_s
        self.ln_1_rstd = null_s
        self.qkv = null_a
        self.lse = null_s
        self.attn = null_a
        self.attn_proj = null_a
        self.residual_2 = null_a
        self.ln_2 = null_a
        self.ln_2_mean = null_s
        self.ln_2_rstd = null_s
        self.fch = null_a
        self.fch_gelu = null_a
        self.fc_proj = null_a
        self.residual_3 = null_a
        self.ln_f = null_a
        self.ln_f_mean = null_s
        self.ln_f_rstd = null_s
        self.logits = null_a
        self.losses = null_s
        self.q = null_a
        self.k = null_a
        self.v = null_a
        self.attn_merged = null_a
        self.att_probs = null_a

    def point_activations(
        mut self,
        sizes: List[Int],
        main_memory: MutMemPtr[Self.dtype],
        stats_memory: MutMemPtr[StatsDType],
    ) -> None:
        # Same iterative layout as the parameters, but split across two blocks:
        # the main (param-precision) activations live in `main_memory`, the fp32
        # statistics/losses in `stats_memory`. Each list is (field-ptr, size) in
        # buffer order; a running cursor walks each block independently.
        comptime MainPtrPtr = UnsafePointer[MutMemPtr[Self.dtype], MutAnyOrigin]
        comptime StatPtrPtr = UnsafePointer[MutMemPtr[StatsDType], MutAnyOrigin]

        # Main (param-precision) activations, in buffer order, with their sizes.
        var mains = List[MainPtrPtr]()
        mains.append(MainPtrPtr(to=self.encoded))
        mains.append(MainPtrPtr(to=self.ln_1))
        mains.append(MainPtrPtr(to=self.qkv))
        mains.append(MainPtrPtr(to=self.attn))
        mains.append(MainPtrPtr(to=self.attn_proj))
        mains.append(MainPtrPtr(to=self.residual_2))
        mains.append(MainPtrPtr(to=self.ln_2))
        mains.append(MainPtrPtr(to=self.fch))
        mains.append(MainPtrPtr(to=self.fch_gelu))
        mains.append(MainPtrPtr(to=self.fc_proj))
        mains.append(MainPtrPtr(to=self.residual_3))
        mains.append(MainPtrPtr(to=self.ln_f))
        mains.append(MainPtrPtr(to=self.logits))
        mains.append(MainPtrPtr(to=self.q))
        mains.append(MainPtrPtr(to=self.k))
        mains.append(MainPtrPtr(to=self.v))
        mains.append(MainPtrPtr(to=self.attn_merged))
        mains.append(MainPtrPtr(to=self.att_probs))
        var main_idx = [
            Activations.encoded,
            Activations.ln_1,
            Activations.qkv,
            Activations.attn,
            Activations.attn_proj,
            Activations.residual_2,
            Activations.ln_2,
            Activations.fch,
            Activations.fch_gelu,
            Activations.fc_proj,
            Activations.residual_3,
            Activations.ln_f,
            Activations.logits,
            Activations.q,
            Activations.k,
            Activations.v,
            Activations.attn_merged,
            Activations.att_probs,
        ]

        # fp32 statistics/losses, in buffer order, with their sizes.
        var stats = List[StatPtrPtr]()
        stats.append(StatPtrPtr(to=self.ln_1_mean))
        stats.append(StatPtrPtr(to=self.ln_1_rstd))
        stats.append(StatPtrPtr(to=self.lse))
        stats.append(StatPtrPtr(to=self.ln_2_mean))
        stats.append(StatPtrPtr(to=self.ln_2_rstd))
        stats.append(StatPtrPtr(to=self.ln_f_mean))
        stats.append(StatPtrPtr(to=self.ln_f_rstd))
        stats.append(StatPtrPtr(to=self.losses))
        var stat_idx = [
            Activations.ln_1_mean,
            Activations.ln_1_rstd,
            Activations.lse,
            Activations.ln_2_mean,
            Activations.ln_2_rstd,
            Activations.ln_f_mean,
            Activations.ln_f_rstd,
            Activations.losses,
        ]

        var a = main_memory
        for i in range(len(mains)):
            mains[i][] = a
            a += sizes[main_idx[i]]

        var s = stats_memory
        for i in range(len(stats)):
            stats[i][] = s
            s += sizes[stat_idx[i]]


# ===----------------------------------------------------------------------=== #
# GPT-2 Model
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct GPT2Config:
    var max_seq_len: Int  # Max sequence length (e.g. 1024).
    var vocab_size: Int  # Vocab size (e.g. 50257).
    var num_layer: Int  # Number of layers (e.g. 12).
    var num_heads: Int  # Number of heads (e.g. 12).
    var channels: Int  # Number of channels (e.g. 768).
    var padded_vocab_size: Int  # Padded vocab size (%128 == 0, e.g. 50304).


# ===----------------------------------------------------------------------=== #
# Fp8State — shared per-tensor fp8 delayed-scaling state container.
#
# Each of the four per-block linear GEMMs (QKV projection, attention-output
# projection, MLP fc, MLP proj) needs its own delayed-scaling
# `AmaxState[FP8_SPEC]` PER TRANSFORMER LAYER — not one shared across layers —
# because delayed scaling's premise ("the scale used this step is derived
# from prior steps' amax") only holds if the amax history a site's scale is
# built from comes from repeated observations of *that same tensor*; layer
# 0's QKV weight and layer 11's QKV weight have unrelated magnitude
# statistics, so collapsing them onto one shared `AmaxState` would let one
# layer's outlier amax silently mis-scale every other layer's GEMM.
#
# Single source of truth for every fp8 GEMM operand's scaling state. Three
# role-groups per site, each a `List[AmaxState[FP8_SPEC]]` of length
# `num_layer`:
#   - `*_input`  — the site's forward input activation (E4M3).
#   - `*_weight` — the site's weight (E4M3, requantized every step from its
#     bf16 storage — weights change every step post-optimizer).
#   - `*_doutput` — the site's backward `d_output` gradient (E5M2).
#
# Forward's weight/input E4M3 `AmaxState`s are also what backward's "same
# tensors as forward" E4M3 operand reuses — backward reads `*_input`/
# `*_weight` (not new states) for dgrad's weight operand and wgrad's input
# operand; only the gradient operand (`*_doutput`, E5M2) is new state.
#
# The field type (`List[AmaxState[FP8_SPEC]]`) is always well-formed
# (FP8_SPEC is a fixed constant regardless of PRECISION — see
# llmm/lowp.mojo), so the `GPT2` struct declares this field unconditionally
# (mirrors the existing `master_buf`/`USE_BF16` convention: "always declare
# the field; make it an inert placeholder — empty lists here, a size-1 buffer
# there — under the regimes that don't need it"). It is `__init__`'s
# population loop that is comptime-gated on `PRECISION == "fp8"` (narrower
# than `LOWP_ENABLED`, which also covers fp4 — fp4 never reads this state, so
# gating on the broader flag would allocate 12 x num_layer unread
# `AmaxState`s and their GPU init kernels for no reason): under bf16/fp32/fp4,
# that branch is never elaborated, so no `AmaxState` (and therefore no GPU
# kernel launch — `AmaxState.__init__` compiles and enqueues an init kernel)
# is ever instantiated for those builds, preserving the invariant that no
# low-precision/GPU-only code may be instantiated for the `cpu` target
# exactly the way `_dispatch_cpu`'s `comptime if USE_BF16:` already does for
# the whole GPU dispatch path.
# ===----------------------------------------------------------------------=== #


struct Fp8State(Movable):
    """Per-layer, per-site `AmaxState[FP8_SPEC]` container for every fp8 GEMM
    operand in the model (the four per-block linears' input/weight/d_output).
    See the module comment above for the shape rationale. Empty lists unless
    `PRECISION == "fp8"`.
    """

    var qkv_input: List[AmaxState[FP8_SPEC]]
    var qkv_weight: List[AmaxState[FP8_SPEC]]
    var qkv_doutput: List[
        AmaxState[FP8_SPEC]
    ]  # backward d_output operand (E5M2)

    var attn_proj_input: List[AmaxState[FP8_SPEC]]
    var attn_proj_weight: List[AmaxState[FP8_SPEC]]
    var attn_proj_doutput: List[
        AmaxState[FP8_SPEC]
    ]  # backward d_output operand (E5M2)

    var fc_input: List[AmaxState[FP8_SPEC]]
    var fc_weight: List[AmaxState[FP8_SPEC]]
    var fc_doutput: List[
        AmaxState[FP8_SPEC]
    ]  # backward d_output operand (E5M2)

    var proj_input: List[AmaxState[FP8_SPEC]]
    var proj_weight: List[AmaxState[FP8_SPEC]]
    var proj_doutput: List[
        AmaxState[FP8_SPEC]
    ]  # backward d_output operand (E5M2)

    def __init__(out self, num_layer: Int, ctx: DeviceContext) raises:
        self.qkv_input = List[AmaxState[FP8_SPEC]]()
        self.qkv_weight = List[AmaxState[FP8_SPEC]]()
        self.qkv_doutput = List[AmaxState[FP8_SPEC]]()
        self.attn_proj_input = List[AmaxState[FP8_SPEC]]()
        self.attn_proj_weight = List[AmaxState[FP8_SPEC]]()
        self.attn_proj_doutput = List[AmaxState[FP8_SPEC]]()
        self.fc_input = List[AmaxState[FP8_SPEC]]()
        self.fc_weight = List[AmaxState[FP8_SPEC]]()
        self.fc_doutput = List[AmaxState[FP8_SPEC]]()
        self.proj_input = List[AmaxState[FP8_SPEC]]()
        self.proj_weight = List[AmaxState[FP8_SPEC]]()
        self.proj_doutput = List[AmaxState[FP8_SPEC]]()

        # See the module comment above: only elaborated for PRECISION ==
        # "fp8" GPU builds — bf16/fp32/fp4 leave every list empty and launch
        # no GPU kernels here.
        #
        # `-D LLMM_FP8_STATIC_SCALES=1`: every layer of a given (site, role)
        # shares the SAME one calibrated constant (`llmm/lowp.mojo`'s
        # `fp8_static_scale`) — deliberately NOT per-layer (unlike the
        # dynamic path's per-layer `AmaxState` history, which exists
        # because different layers have unrelated magnitude statistics —
        # see the module comment above `Fp8State`). Static mode instead
        # shares one hardcoded constant per tensor ROLE globally — the
        # calibration tool already picked the safe (min-over-layers,
        # margined) constant per (site, role), so reusing it for every layer
        # is intentional, not a simplification that drops per-layer coverage.
        comptime if PRECISION == "fp8":
            comptime if FP8_STATIC_SCALES:
                comptime qkv_input_s = fp8_static_scale["qkv", "input"]()
                comptime qkv_weight_s = fp8_static_scale["qkv", "weight"]()
                comptime qkv_doutput_s = fp8_static_scale["qkv", "doutput"]()
                comptime attn_proj_input_s = fp8_static_scale[
                    "attn_proj", "input"
                ]()
                comptime attn_proj_weight_s = fp8_static_scale[
                    "attn_proj", "weight"
                ]()
                comptime attn_proj_doutput_s = fp8_static_scale[
                    "attn_proj", "doutput"
                ]()
                comptime fc_input_s = fp8_static_scale["fc", "input"]()
                comptime fc_weight_s = fp8_static_scale["fc", "weight"]()
                comptime fc_doutput_s = fp8_static_scale["fc", "doutput"]()
                comptime proj_input_s = fp8_static_scale["proj", "input"]()
                comptime proj_weight_s = fp8_static_scale["proj", "weight"]()
                comptime proj_doutput_s = fp8_static_scale["proj", "doutput"]()
                for _ in range(num_layer):
                    self.qkv_input.append(
                        AmaxState[FP8_SPEC](ctx, static_scale=qkv_input_s)
                    )
                    self.qkv_weight.append(
                        AmaxState[FP8_SPEC](ctx, static_scale=qkv_weight_s)
                    )
                    self.qkv_doutput.append(
                        AmaxState[FP8_SPEC](ctx, static_scale=qkv_doutput_s)
                    )
                    self.attn_proj_input.append(
                        AmaxState[FP8_SPEC](ctx, static_scale=attn_proj_input_s)
                    )
                    self.attn_proj_weight.append(
                        AmaxState[FP8_SPEC](
                            ctx, static_scale=attn_proj_weight_s
                        )
                    )
                    self.attn_proj_doutput.append(
                        AmaxState[FP8_SPEC](
                            ctx, static_scale=attn_proj_doutput_s
                        )
                    )
                    self.fc_input.append(
                        AmaxState[FP8_SPEC](ctx, static_scale=fc_input_s)
                    )
                    self.fc_weight.append(
                        AmaxState[FP8_SPEC](ctx, static_scale=fc_weight_s)
                    )
                    self.fc_doutput.append(
                        AmaxState[FP8_SPEC](ctx, static_scale=fc_doutput_s)
                    )
                    self.proj_input.append(
                        AmaxState[FP8_SPEC](ctx, static_scale=proj_input_s)
                    )
                    self.proj_weight.append(
                        AmaxState[FP8_SPEC](ctx, static_scale=proj_weight_s)
                    )
                    self.proj_doutput.append(
                        AmaxState[FP8_SPEC](ctx, static_scale=proj_doutput_s)
                    )
            else:
                for _ in range(num_layer):
                    self.qkv_input.append(AmaxState[FP8_SPEC](ctx))
                    self.qkv_weight.append(AmaxState[FP8_SPEC](ctx))
                    self.qkv_doutput.append(AmaxState[FP8_SPEC](ctx))
                    self.attn_proj_input.append(AmaxState[FP8_SPEC](ctx))
                    self.attn_proj_weight.append(AmaxState[FP8_SPEC](ctx))
                    self.attn_proj_doutput.append(AmaxState[FP8_SPEC](ctx))
                    self.fc_input.append(AmaxState[FP8_SPEC](ctx))
                    self.fc_weight.append(AmaxState[FP8_SPEC](ctx))
                    self.fc_doutput.append(AmaxState[FP8_SPEC](ctx))
                    self.proj_input.append(AmaxState[FP8_SPEC](ctx))
                    self.proj_weight.append(AmaxState[FP8_SPEC](ctx))
                    self.proj_doutput.append(AmaxState[FP8_SPEC](ctx))


def _gpt2_hyperparameters(depth: Int) raises -> Tuple[Int, Int]:
    """GPT-2 (channels, num_heads) for a given depth. Mirrors llm.c."""
    if depth == 6:
        return (384, 6)  # (unofficial) gpt2-tiny (30M)
    elif depth == 12:
        return (768, 12)  # gpt2 (124M)
    elif depth == 24:
        return (1024, 16)  # gpt2-medium (350M)
    elif depth == 36:
        return (1280, 20)  # gpt2-large (774M)
    elif depth == 48:
        return (1600, 25)  # gpt2-xl (1558M)
    elif depth == 60:
        return (1920, 30)  # (unofficial) 2.7B
    elif depth == 72:
        return (2880, 30)  # (unofficial) 7.3B
    elif depth == 84:
        return (3456, 36)  # (unofficial) 12.2B
    raise Error("Unsupported GPT-2 depth: " + String(depth))


def _gpt3_hyperparameters(channels: Int) raises -> Tuple[Int, Int]:
    """GPT-3 (depth, head_size) for a given channel count. Mirrors llm.c."""
    if channels == 384:
        return (6, 64)  # (unofficial) gpt3-tiny (31M)
    elif channels == 768:
        return (12, 64)  # gpt3-small (125M)
    elif channels == 1024:
        return (24, 64)  # gpt3-medium (350M)
    elif channels == 1536:
        return (24, 96)  # gpt3-large (760M)
    elif channels == 2048:
        return (24, 128)  # gpt3-xl (1.3B)
    elif channels == 2560:
        return (32, 80)  # gpt3-2.7B
    elif channels == 4096:
        return (32, 128)  # gpt3-6.7B
    elif channels == 5140:
        return (40, 128)  # gpt3-13B
    elif channels == 12288:
        return (96, 128)  # gpt3 (175B)
    raise Error("Unsupported GPT-3 channels: " + String(channels))


def parse_model_descriptor(descriptor: String) raises -> GPT2Config:
    """Build a randomly-initializable GPT2Config from a model descriptor.

    Supported descriptors (matching llm.c's gpt_build_from_descriptor):
      - "dX"       legacy GPT-2 with depth X, e.g. "d12"
      - "gpt2:dX"  explicit GPT-2 with depth X, e.g. "gpt2:d48"
      - "gpt3:cX"  GPT-3 with channel count X, e.g. "gpt3:c768"
    """
    var num_layer: Int
    var channels: Int
    var num_heads: Int
    var max_seq_len: Int

    if descriptor.startswith("gpt2:d"):
        num_layer = atol(descriptor[byte=6:])
        var ch_nh = _gpt2_hyperparameters(num_layer)
        channels = ch_nh[0]
        num_heads = ch_nh[1]
        max_seq_len = 1024
    elif descriptor.startswith("gpt3:c"):
        channels = atol(descriptor[byte=6:])
        var d_hs = _gpt3_hyperparameters(channels)
        num_layer = d_hs[0]
        var head_size = d_hs[1]
        if channels % head_size != 0:
            raise Error("GPT-3 channels not divisible by head size")
        num_heads = channels // head_size
        # NOTE: GPT-3 uses a context length of 2048 tokens (vs 1024 for GPT-2).
        max_seq_len = 2048
    elif descriptor.startswith("d"):
        num_layer = atol(descriptor[byte=1:])
        var ch_nh = _gpt2_hyperparameters(num_layer)
        channels = ch_nh[0]
        num_heads = ch_nh[1]
        max_seq_len = 1024
    else:
        raise Error("Unsupported model descriptor: " + descriptor)

    if num_layer <= 0:
        raise Error("Invalid depth in model descriptor: " + descriptor)

    return GPT2Config(
        max_seq_len=max_seq_len,
        vocab_size=SCRATCH_VOCAB_SIZE,
        num_layer=num_layer,
        num_heads=num_heads,
        channels=channels,
        padded_vocab_size=SCRATCH_PADDED_VOCAB_SIZE,
    )


# `recompute` enables activation (gradient) checkpointing: when True, the
# per-layer MLP activations fch (pre-GELU) and fch_gelu (post-GELU) are not
# persisted across the forward/backward boundary. They collapse to a single
# scratch slot that the forward fills and the backward rematerializes by
# re-running the same fused FC matmul (matmul_fwd[use_gelu=True]) from the
# still-resident ln_2. This reuses the forward's fused-GELU kernel and drops the
# two largest per-layer activation buffers (each B*T*4C), trading one FC GEMM
# per layer in backward for the memory. That activation-memory cut is what makes
# ZeRO-3 tractable (ZeRO-3 shards params/grads/optimizer but not activations).
struct GPT2[target: StaticString, WORLD_SIZE: Int = 1, recompute: Bool = False]:
    var ctx: DeviceContext
    var config: GPT2Config

    # Buffer objects managing physical memory lifetimes
    var params_buf: DeviceBuffer[GPT2_DTYPE]
    var grads_buf: DeviceBuffer[GPT2_DTYPE]
    var m_buf: DeviceBuffer[MASTER_DTYPE]
    var v_buf: DeviceBuffer[MASTER_DTYPE]
    # fp32 master copy of the weights (size #params) for bf16/fp8 training; a
    # size-1 placeholder under fp32, where the params are their own master.
    var master_buf: DeviceBuffer[MASTER_DTYPE]
    var acts_buf: DeviceBuffer[GPT2_DTYPE]
    var grad_acts_buf: DeviceBuffer[GPT2_DTYPE]
    # fp32 statistics/losses, split out from the main (param-precision) acts.
    var acts_stats_buf: DeviceBuffer[StatsDType]
    var grad_acts_stats_buf: DeviceBuffer[StatsDType]
    var inputs_buf: HostBuffer[DType.int32]
    var targets_buf: HostBuffer[DType.int32]
    # Device-resident copies of the token/target indices. On Metal a HostBuffer
    # pointer read from inside a GPU kernel silently returns zeros (the encoder
    # would then embed token 0 everywhere and the classifier would score against
    # target 0), so the GPU kernels must read these device buffers, uploaded via
    # enqueue_copy each forward. On CPU these are size-1 placeholders (the host
    # buffers are read directly).
    var inputs_dev_buf: DeviceBuffer[DType.int32]
    var targets_dev_buf: DeviceBuffer[DType.int32]
    var bucket_info_buf: HostBuffer[DType.int32]
    var workload_indices_buf: HostBuffer[DType.int32]
    # Device-resident copies of bucket_info / workload_indices. On Metal a
    # HostBuffer pointer read from inside a GPU kernel silently returns zeros,
    # so the wte-backward GPU kernel must read these device copies instead.
    var bucket_info_dev_buf: DeviceBuffer[DType.int32]
    var workload_indices_dev_buf: DeviceBuffer[DType.int32]
    var losses_host_buf: HostBuffer[StatsDType]
    var logits_host_buf: HostBuffer[GPT2_DTYPE]
    # Persistent scratch for calculate_grad_norm's GPU reduction (per-block
    # partial sums + the 1-element host readback). Sized once in
    # allocate_optimizer_moments (grid_x is fixed by num_sm/BLOCK_SIZE, not by
    # anything that changes step to step) and reused every step instead of
    # calling enqueue_create_buffer/enqueue_create_host_buffer per call.
    var grad_norm_out_buf: DeviceBuffer[DType.float32]
    var grad_norm_host_buf: HostBuffer[DType.float32]
    # grad_norm_out_buf's element count (grid_x), cached alongside it so
    # calculate_grad_norm doesn't re-query the GPU's SM count every step.
    var grad_norm_grid_x: Int

    # Model weights and their sizes
    var params: ParameterTensors[GPT2_DTYPE]
    var param_sizes: List[Int]
    var params_memory: MutMemPtr[GPT2_DTYPE]
    var num_parameters: Int

    # Gradients of the weights
    var grads: ParameterTensors[GPT2_DTYPE]
    var grads_memory: MutMemPtr[GPT2_DTYPE]

    # Buffers for AdamW (always fp32, independent of the parameter precision).
    var m_memory: MutMemPtr[MASTER_DTYPE]
    var v_memory: MutMemPtr[MASTER_DTYPE]
    var master_memory: MutMemPtr[MASTER_DTYPE]

    # Activations of the model, and their sizes
    var acts: ActivationTensors[GPT2_DTYPE]
    var act_sizes: List[Int]
    var acts_memory: MutMemPtr[GPT2_DTYPE]
    var acts_stats_memory: MutMemPtr[StatsDType]
    var num_activations: Int

    # Gradients of the activations
    var grad_acts: ActivationTensors[GPT2_DTYPE]
    # Same shape as `act_sizes`, but on GPU builds the three tensors that are
    # dead on the GPU backward path (`fch`, `logits`, `att_probs` — see the
    # per-tensor trace above `zero_gradients`) are sized to 0, shrinking
    # `grad_acts_buf` by ~1.9 GB. On CPU this stays identical to `act_sizes`
    # (the CPU `matmul_d_weight_bwd` path reads the `fch`/`logits` scratch
    # args; `att_probs` is unused on both targets but kept full-size on CPU
    # for simplicity/symmetry).
    var grad_act_sizes: List[Int]
    var grad_acts_memory: MutMemPtr[GPT2_DTYPE]
    var grad_acts_stats_memory: MutMemPtr[StatsDType]
    var num_grads: Int

    # Shared per-tensor fp8 delayed-scaling state (see `Fp8State`'s docstring
    # above). Always declared (mirrors `master_buf`'s USE_BF16 convention);
    # populated only under PRECISION == "fp8" (empty lists otherwise).
    var fp8_state: Fp8State

    # Runstate Configurations
    var batch_size: Int  # The batch size of the current forward pass (Our B).
    var seq_len: Int  # The sequence length of the current forward pass (Our T).
    var inputs: MutMemPtr[
        DType.int32
    ]  # The input tokens for the current forward pass.
    var targets: MutMemPtr[
        DType.int32
    ]  # The target tokens for the current forward pass.
    # Device-resident token/target pointers for GPU kernels (see *_dev_buf).
    var inputs_dev: MutMemPtr[DType.int32]
    var targets_dev: MutMemPtr[DType.int32]
    var bucket_info: MutMemPtr[DType.int32]
    var workload_indices: MutMemPtr[DType.int32]
    var bucket_info_dev: MutMemPtr[DType.int32]
    var workload_indices_dev: MutMemPtr[DType.int32]
    var num_wte_buckets: Int
    var wte_bucket_capacity: Int
    var mean_loss: Float32  # The mean loss for the current forward pass.
    var checkpoint_path: String  # The path to the checkpoint file.

    # Memory allocation tracking flags
    var has_allocated_params: Bool
    var has_allocated_acts: Bool
    var has_allocated_grads: Bool
    var has_allocated_optimizer_moments: Bool

    # Cache management
    var kv_cache: KVCache

    # Sharding and parallelism
    var zero_ctx: ZeroContext[Self.target, Self.WORLD_SIZE]
    var optimizer_num_parameters: Int
    var padded_num_parameters: Int
    var sharded_grads_buf: DeviceBuffer[GPT2_DTYPE]
    var sharded_grads_memory: MutMemPtr[GPT2_DTYPE]

    def __init__(
        out self,
        checkpoint_path: String,
        rank: Int,
        zero_stage: Int,
        ctx: DeviceContext,
        cpu_coordinator_ptr: Optional[
            UnsafePointer[CpuCoordinator, MutUntrackedOrigin]
        ] = None,
    ) raises:
        self.ctx = ctx
        self.checkpoint_path = checkpoint_path
        self.has_allocated_params = False
        self.has_allocated_acts = False
        self.has_allocated_grads = False
        self.has_allocated_optimizer_moments = False
        self.kv_cache = KVCache()
        self.zero_ctx = ZeroContext[Self.target, Self.WORLD_SIZE](
            rank, zero_stage, ctx, cpu_coordinator_ptr
        )
        self.optimizer_num_parameters = 0
        self.padded_num_parameters = 0

        self.config = GPT2Config(
            max_seq_len=0,
            vocab_size=0,
            num_layer=0,
            num_heads=0,
            channels=0,
            padded_vocab_size=0,
        )
        # Trivial placeholder (num_layer=0 -> no AmaxState instances, no GPU
        # kernel launches even under LOWP_ENABLED): definite-initialization
        # requires every field set before the first `self.<method>()` call
        # below (`self.allocate_parameters`/etc.), mirroring `self.config`'s
        # own placeholder-then-real-value pattern immediately above. Replaced
        # with the real, per-layer-populated container once `self.config` is
        # finalized (see below, after the checkpoint/safetensors/from-scratch
        # branches and `self.allocate_gradients()`).
        self.fp8_state = Fp8State(0, ctx)

        var zero = 0
        var NULL_DTYPE_PTR = MutMemPtr[GPT2_DTYPE](unsafe_from_address=zero)
        var NULL_MASTER_PTR = MutMemPtr[MASTER_DTYPE](unsafe_from_address=zero)
        var NULL_INT32_PTR = MutMemPtr[DType.int32](unsafe_from_address=zero)

        self.sharded_grads_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.sharded_grads_memory = NULL_DTYPE_PTR

        self.params_memory = NULL_DTYPE_PTR
        self.grads_memory = NULL_DTYPE_PTR
        self.m_memory = NULL_MASTER_PTR
        self.v_memory = NULL_MASTER_PTR
        self.master_memory = NULL_MASTER_PTR
        self.acts_memory = NULL_DTYPE_PTR
        self.grad_acts_memory = NULL_DTYPE_PTR
        self.acts_stats_memory = NULL_MASTER_PTR
        self.grad_acts_stats_memory = NULL_MASTER_PTR
        self.inputs = NULL_INT32_PTR
        self.targets = NULL_INT32_PTR
        self.inputs_dev = NULL_INT32_PTR
        self.targets_dev = NULL_INT32_PTR
        self.bucket_info = NULL_INT32_PTR
        self.workload_indices = NULL_INT32_PTR
        self.bucket_info_dev = NULL_INT32_PTR
        self.workload_indices_dev = NULL_INT32_PTR
        self.num_wte_buckets = 0
        self.wte_bucket_capacity = 0

        self.num_parameters = 0
        self.num_activations = 0
        self.num_grads = 0
        self.batch_size = 0
        self.seq_len = 0
        self.mean_loss = -1.0

        self.params = ParameterTensors[GPT2_DTYPE]()
        self.grads = ParameterTensors[GPT2_DTYPE]()
        self.acts = ActivationTensors[GPT2_DTYPE]()
        self.grad_acts = ActivationTensors[GPT2_DTYPE]()

        self.params_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.grads_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.m_buf = self.ctx.enqueue_create_buffer[MASTER_DTYPE](1)
        self.v_buf = self.ctx.enqueue_create_buffer[MASTER_DTYPE](1)
        self.master_buf = self.ctx.enqueue_create_buffer[MASTER_DTYPE](1)
        self.acts_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.grad_acts_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](1)
        self.acts_stats_buf = self.ctx.enqueue_create_buffer[StatsDType](1)
        self.grad_acts_stats_buf = self.ctx.enqueue_create_buffer[StatsDType](1)
        self.inputs_buf = self.ctx.enqueue_create_host_buffer[DType.int32](1)
        self.targets_buf = self.ctx.enqueue_create_host_buffer[DType.int32](1)
        self.inputs_dev_buf = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.targets_dev_buf = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.bucket_info_dev_buf = self.ctx.enqueue_create_buffer[DType.int32](
            1
        )
        self.workload_indices_dev_buf = self.ctx.enqueue_create_buffer[
            DType.int32
        ](1)
        self.bucket_info_buf = self.ctx.enqueue_create_host_buffer[DType.int32](
            1
        )
        self.workload_indices_buf = self.ctx.enqueue_create_host_buffer[
            DType.int32
        ](1)
        self.grad_norm_out_buf = self.ctx.enqueue_create_buffer[DType.float32](
            1
        )
        self.grad_norm_host_buf = self.ctx.enqueue_create_host_buffer[
            DType.float32
        ](1)
        self.grad_norm_grid_x = 1
        self.losses_host_buf = self.ctx.enqueue_create_host_buffer[StatsDType](
            1
        )
        self.logits_host_buf = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
            1
        )
        self.ctx.synchronize()

        self.param_sizes = List[Int]()
        for _ in range(NUM_PARAMETER_TENSORS):
            self.param_sizes.append(0)

        self.act_sizes = List[Int]()
        for _ in range(NUM_ACTIVATION_TENSORS):
            self.act_sizes.append(0)

        self.grad_act_sizes = List[Int]()
        for _ in range(NUM_ACTIVATION_TENSORS):
            self.grad_act_sizes.append(0)

        # Two ways to initialize the model, mirroring llm.c's main():
        #  - `checkpoint_path` ends with ".bin": load config + weights from a
        #     checkpoint file (e.g. gpt2_124M.bin).
        #  - otherwise it is a model descriptor (e.g. "d12", "gpt2:d48",
        #    "gpt3:c768"): derive the config and random-init the weights so the
        #    run trains from scratch with PyTorch-identical initial conditions.
        if self.checkpoint_path.endswith(".bin"):
            var model_file = open(self.checkpoint_path, "r")
            var model_header = alloc[Int32](256)
            read_and_copy[DType.int32](model_file, model_header, 256)

            var model_magic = model_header.load(0)
            if model_magic != GPT2_MAGIC and model_magic != GPT2_MAGIC_LEGACY:
                print("Bad magic number in header: " + String(model_magic))
                model_header.free()
                raise Error("GPT2 error: Invalid magic number in header")
            # llm.c's version convention: 3 => fp32 params, 5 => bf16 params.
            comptime EXPECTED_VERSION = 5 if GPT2_DTYPE == DType.bfloat16 else 3
            if Int(model_header.load(1)) != EXPECTED_VERSION:
                print("Bad version in header: " + String(model_header.load(1)))
                model_header.free()
                raise Error("GPT2 error: Invalid version in header")

            self.config = GPT2Config(
                max_seq_len=Int(model_header.load(2)),
                vocab_size=Int(model_header.load(3)),
                num_layer=Int(model_header.load(4)),
                num_heads=Int(model_header.load(5)),
                channels=Int(model_header.load(6)),
                padded_vocab_size=Int(model_header.load(7)),
            )
            model_header.free()

            self.allocate_parameters(model_file)
        elif self.checkpoint_path.endswith(".safetensors"):
            # HuggingFace-exported checkpoint (see scripts/export_to_hf.py).
            # Config lives in a sibling config.json, not in the safetensors
            # header itself.
            var safetensors_path = self.checkpoint_path
            var slash_idx = safetensors_path.rfind("/")
            var config_dir = String(
                safetensors_path[byte=0:slash_idx]
            ) if slash_idx >= 0 else String(".")
            var checkpoint_config = read_hf_gpt2_config(
                config_dir + "/config.json"
            )
            self.config = GPT2Config(
                max_seq_len=checkpoint_config.max_seq_len,
                vocab_size=checkpoint_config.vocab_size,
                num_layer=checkpoint_config.num_layer,
                num_heads=checkpoint_config.num_heads,
                channels=checkpoint_config.channels,
                padded_vocab_size=checkpoint_config.padded_vocab_size,
            )
            self.allocate_parameters_from_safetensors(safetensors_path)
        else:
            # Build config from the descriptor and random-init the weights.
            self.config = parse_model_descriptor(self.checkpoint_path)
            self.allocate_parameters_random()

        self.allocate_gradients()

        # fp8 delayed-scaling state: one AmaxState per GEMM operand site per
        # layer. Config is finalized by all three branches above, so
        # config.num_layer is valid here. See `Fp8State`'s module comment for
        # why this is unconditional (empty lists, no GPU work, under
        # bf16/fp32/fp4 — the ctor's population loop is itself `comptime if
        # PRECISION == "fp8":` gated).
        self.fp8_state = Fp8State(self.config.num_layer, ctx)

        self.allocate_optimizer_moments()

        # Init Activations and activation gradients (allocated dynamically per batch).
        self.acts = ActivationTensors[GPT2_DTYPE]()
        self.acts_memory = MutMemPtr[GPT2_DTYPE](unsafe_from_address=zero)
        self.num_activations = 0

        self.grad_acts = ActivationTensors[GPT2_DTYPE]()
        self.grad_acts_memory = MutMemPtr[GPT2_DTYPE](unsafe_from_address=zero)

        self.batch_size = 0
        self.seq_len = 0
        self.mean_loss = -1.0  # -1.0 designates no loss has been computed yet.
        self.inputs = MutMemPtr[DType.int32](unsafe_from_address=zero)
        self.targets = MutMemPtr[DType.int32](unsafe_from_address=zero)
        self.inputs_dev = MutMemPtr[DType.int32](unsafe_from_address=zero)
        self.targets_dev = MutMemPtr[DType.int32](unsafe_from_address=zero)

        if self.zero_ctx.rank == 0:
            print("Model Summary:")
            print("--------------------------------")
            print("Model Name: GPT-2")
            print("Model Magic Number: " + String(GPT2_MAGIC))
            print("Model Config:")
            print("--------------------------------")
            print("Max Sequence Length: " + String(self.config.max_seq_len))
            print("Vocab Size: " + String(self.config.vocab_size))
            print("Number of Layers: " + String(self.config.num_layer))
            print("Number of Heads: " + String(self.config.num_heads))
            print("Number of Channels: " + String(self.config.channels))
            print("Padded Vocab Size: " + String(self.config.padded_vocab_size))
            print("--------------------------------")
            print("Number of Parameters: " + String(self.num_parameters))
            print("--------------------------------")
            print("Number of Activations: " + String(self.num_activations))
            print("--------------------------------")
            print("Number of Gradients: " + String(self.num_grads))
            print("--------------------------------")

    def _compute_param_sizes(mut self) raises:
        """Fill param_sizes, num_parameters, and the optimizer sharding counts
        from the current config. Shared by the checkpoint and from-scratch
        allocation paths.
        """
        var max_T = self.config.max_seq_len
        var L = self.config.num_layer
        var C = self.config.channels
        var V_p = self.config.padded_vocab_size

        self.param_sizes[Parameters.wte] = V_p * C
        self.param_sizes[Parameters.wpe] = max_T * C
        self.param_sizes[Parameters.ln_1_gamma] = L * C
        self.param_sizes[Parameters.ln_1_beta] = L * C
        self.param_sizes[Parameters.qkv_weight] = L * (3 * C) * C
        self.param_sizes[Parameters.qkv_bias] = L * (3 * C)
        self.param_sizes[Parameters.attn_proj_weight] = L * C * C
        self.param_sizes[Parameters.attn_proj_bias] = L * C
        self.param_sizes[Parameters.ln_2_gamma] = L * C
        self.param_sizes[Parameters.ln_2_beta] = L * C
        self.param_sizes[Parameters.fc_weight] = L * (4 * C) * C
        self.param_sizes[Parameters.fc_bias] = L * (4 * C)
        self.param_sizes[Parameters.proj_weight] = L * C * (4 * C)
        self.param_sizes[Parameters.proj_bias] = L * C
        self.param_sizes[Parameters.ln_f_gamma] = C
        self.param_sizes[Parameters.ln_f_beta] = C

        var num_parameters = 0
        for i in range(NUM_PARAMETER_TENSORS):
            num_parameters += self.param_sizes[i]
        self.num_parameters = num_parameters

        # Calculate optimizer sharding parameters
        comptime if Self.WORLD_SIZE > 1:
            # All zero stages 1/2/3 shard the optimizer parameters.
            if self.zero_ctx.zero_stage >= 1:
                # Round the per-rank shard length up to a multiple of the AdamW
                # SIMD width. Each rank's optimizer step indexes params/grads at
                # `rank * optimizer_num_parameters`, and adamw_update issues
                # `alignment = align_of[SIMD[dtype, width]]` (naturally width
                # elements) aligned vector loads/stores. If the shard length is
                # not a multiple of that width, every rank>0 shard offset is
                # misaligned and the aligned CPU load faults (segfault). Padding
                # the shard keeps every offset aligned; the extra tail elements
                # live in the zero-filled padding region of the params/grads
                # buffers (both sized to padded_num_parameters).
                comptime shard_align = simd_width_of[GPT2_DTYPE]()
                var base_shard = (
                    self.num_parameters + Self.WORLD_SIZE - 1
                ) // Self.WORLD_SIZE
                self.optimizer_num_parameters = (
                    (base_shard + shard_align - 1) // shard_align
                ) * shard_align
                self.padded_num_parameters = (
                    self.optimizer_num_parameters * Self.WORLD_SIZE
                )
            else:
                self.optimizer_num_parameters = self.num_parameters
                self.padded_num_parameters = self.num_parameters
        else:
            self.optimizer_num_parameters = self.num_parameters
            self.padded_num_parameters = self.num_parameters

    def allocate_parameters(mut self, mut model_file: FileHandle) raises:
        # Alloc space for all the parameters then read them in.
        self._compute_param_sizes()

        # Allocate parameters using device context host buffer.
        self.params = ParameterTensors[GPT2_DTYPE]()
        var temp_host_buf = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
            self.num_parameters
        )
        self.ctx.synchronize()
        var temp_ptr = rebind_mut_mem[GPT2_DTYPE](
            temp_host_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        read_and_copy[GPT2_DTYPE](model_file, temp_ptr, self.num_parameters)
        model_file.close()

        self.params_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
            self.padded_num_parameters
        )

        self.params_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
        self.ctx.enqueue_copy(
            dst_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                self.params_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
            src_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], ImmutAnyOrigin]](
                temp_ptr.as_unsafe_any_origin()
            ),
            size=self.num_parameters,
        )
        self.ctx.synchronize()
        self.params_memory = rebind_mut_mem[GPT2_DTYPE](
            self.params_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.params.point_parameters(self.param_sizes, self.params_memory)

        self.has_allocated_params = True

    def allocate_parameters_random(mut self) raises:
        """Allocate parameters and fill them with a GPT-2 random init, exactly
        matching llm.c's `gpt_build_from_descriptor` (and therefore PyTorch):
        weights ~ N(0, 0.02), biases 0, layernorm gammas 1, and the residual
        projections (attn_proj_weight, proj_weight) scaled by 1/sqrt(2*L). The
        tensors are drawn in PyTorch's layer-by-layer order so the random stream
        lines up bit-for-bit.
        """
        self._compute_param_sizes()

        var L = self.config.num_layer
        var C = self.config.channels
        var V = self.config.vocab_size

        self.params = ParameterTensors[GPT2_DTYPE]()

        # Host buffer holding the final params (in GPT2_DTYPE), zero-initialized
        # so all biases / layernorm betas are left at 0.
        var host_params = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
            self.num_parameters
        )
        self.ctx.synchronize()
        var hp = rebind_mut_mem[GPT2_DTYPE](
            host_params.unsafe_ptr().as_unsafe_any_origin()
        )
        for i in range(self.num_parameters):
            hp[i] = Scalar[GPT2_DTYPE](0.0)

        # fp32 scratch for the normal_ draws, sized to the largest single draw
        # (wte is V*C; the per-layer weight draws are at most 4*C*C).
        var max_draw = V * C
        if self.config.max_seq_len * C > max_draw:
            max_draw = self.config.max_seq_len * C
        if 4 * C * C > max_draw:
            max_draw = 4 * C * C
        var fp32_scratch = alloc[Scalar[DType.float32]](max_draw)
        var fp32_ptr = rebind_mut_mem[DType.float32](
            fp32_scratch.as_unsafe_any_origin()
        )

        var rng = MT19937(UInt32(INIT_RNG_SEED))
        var residual_scale = Float32(1.0) / sqrt(Float32(2.0 * Float64(L)))

        # Mirror gpt_build_from_descriptor: outer loop over layers, inner over
        # the 16 parameter tensors, keeping a running offset into the flat param
        # buffer. `offset` restarts every layer; layered weight tensors index
        # into their own per-layer slice via `layer_offset`.
        for l in range(L):
            var offset = 0
            for i in range(NUM_PARAMETER_TENSORS):
                var n_elem = self.param_sizes[i]
                # Layernorm gammas (ln_1/ln_2/ln_f) are initialized to 1, once.
                if l == 0 and (
                    i == Parameters.ln_1_gamma
                    or i == Parameters.ln_2_gamma
                    or i == Parameters.ln_f_gamma
                ):
                    for j in range(n_elem):
                        hp[offset + j] = Scalar[GPT2_DTYPE](1.0)
                # Weight tensors: wte/wpe once at l==0, the per-layer attention
                # and MLP weights every layer.
                var is_layer_weight = (
                    i == Parameters.qkv_weight
                    or i == Parameters.attn_proj_weight
                    or i == Parameters.fc_weight
                    or i == Parameters.proj_weight
                )
                var is_embedding = l == 0 and (
                    i == Parameters.wte or i == Parameters.wpe
                )
                if is_embedding or is_layer_weight:
                    var n = n_elem
                    var layer_offset = 0
                    if i == Parameters.wte:
                        # init the V real rows, not the padded Vp rows
                        n = V * C
                    if is_layer_weight:
                        n = n // L
                        layer_offset = l * n
                    # Residual-stream projections get the 1/sqrt(2*L) scaling.
                    var scale = INIT_WEIGHT_STD
                    if (
                        i == Parameters.attn_proj_weight
                        or i == Parameters.proj_weight
                    ):
                        scale = INIT_WEIGHT_STD * residual_scale
                    normal_(rng, fp32_ptr, n, Float32(0.0), scale)
                    for j in range(n):
                        hp[offset + layer_offset + j] = fp32_ptr[j].cast[
                            GPT2_DTYPE
                        ]()
                offset += n_elem

        fp32_scratch.free()

        # Copy the host params to the device, zero-padding the sharded tail.
        self.params_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
            self.padded_num_parameters
        )
        self.params_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
        self.ctx.enqueue_copy(
            dst_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                self.params_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
            src_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], ImmutAnyOrigin]](
                hp.as_unsafe_any_origin()
            ),
            size=self.num_parameters,
        )
        self.ctx.synchronize()
        self.params_memory = rebind_mut_mem[GPT2_DTYPE](
            self.params_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.params.point_parameters(self.param_sizes, self.params_memory)

        self.has_allocated_params = True

    def allocate_parameters_from_safetensors(
        mut self, safetensors_path: String
    ) raises:
        """Allocate parameters and populate them from a HuggingFace-exported
        `.safetensors` checkpoint (see `scripts/export_to_hf.py`, which
        produces these from our own `.bin` format).

        Reuses the exact same downstream layout as `allocate_parameters`
        (flat `temp_host_buf` in canonical parameter order -> device copy ->
        `ParameterTensors.point_parameters`) — only how the host buffer gets
        populated differs. `self.config` must already be set (from the
        sibling `config.json`) before calling this.
        """
        self._compute_param_sizes()

        var L = self.config.num_layer
        var C = self.config.channels
        var V = self.config.vocab_size
        var V_p = self.config.padded_vocab_size
        var max_T = self.config.max_seq_len

        self.params = ParameterTensors[GPT2_DTYPE]()
        var temp_host_buf = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
            self.num_parameters
        )
        self.ctx.synchronize()
        var hp = rebind_mut_mem[GPT2_DTYPE](
            temp_host_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        for i in range(self.num_parameters):
            hp[i] = Scalar[GPT2_DTYPE](0.0)

        var st = SafetensorsFile(safetensors_path)

        # wte: HF export drops the (V_p - V) padding rows entirely (see
        # export_to_hf.py's `w[key][:(V-Vp), :]` slice) — read the real V
        # rows into the front of our (V_p, C)-sized block; the trailing
        # padding rows stay zero, exactly as fused_classifier already
        # ignores them (it masks dlogits/loss to the real V columns, see
        # llmm/fused_classifier.mojo).
        var base = 0
        st.read_tensor[GPT2_DTYPE]("transformer.wte.weight", hp + base)
        base += V_p * C

        st.read_tensor[GPT2_DTYPE]("transformer.wpe.weight", hp + base)
        base += max_T * C

        # ln_1_gamma / ln_1_beta: each layer's C values are concatenated
        # back-to-back across the whole L*C block (llm.c/our own layout
        # groups by tensor TYPE across all layers, not by layer).
        for l in range(L):
            st.read_tensor[GPT2_DTYPE](
                "transformer.h." + String(l) + ".ln_1.weight",
                hp + base + l * C,
            )
        base += L * C
        for l in range(L):
            st.read_tensor[GPT2_DTYPE](
                "transformer.h." + String(l) + ".ln_1.bias",
                hp + base + l * C,
            )
        base += L * C

        # qkv_weight: HF stores Conv1D-style (C, 3C) — the transpose of our
        # (3C, C) — per export_to_hf.py's `mk_tensor(..., transpose=True)`.
        for l in range(L):
            st.read_tensor[GPT2_DTYPE](
                "transformer.h." + String(l) + ".attn.c_attn.weight",
                hp + base + l * (3 * C * C),
                transpose_rows=3 * C,
                transpose_cols=C,
            )
        base += L * (3 * C) * C
        for l in range(L):
            st.read_tensor[GPT2_DTYPE](
                "transformer.h." + String(l) + ".attn.c_attn.bias",
                hp + base + l * (3 * C),
            )
        base += L * (3 * C)

        for l in range(L):
            st.read_tensor[GPT2_DTYPE](
                "transformer.h." + String(l) + ".attn.c_proj.weight",
                hp + base + l * (C * C),
                transpose_rows=C,
                transpose_cols=C,
            )
        base += L * C * C
        for l in range(L):
            st.read_tensor[GPT2_DTYPE](
                "transformer.h." + String(l) + ".attn.c_proj.bias",
                hp + base + l * C,
            )
        base += L * C

        for l in range(L):
            st.read_tensor[GPT2_DTYPE](
                "transformer.h." + String(l) + ".ln_2.weight",
                hp + base + l * C,
            )
        base += L * C
        for l in range(L):
            st.read_tensor[GPT2_DTYPE](
                "transformer.h." + String(l) + ".ln_2.bias",
                hp + base + l * C,
            )
        base += L * C

        # fc_weight (mlp.c_fc): HF stores (C, 4C), the transpose of our
        # (4C, C).
        for l in range(L):
            st.read_tensor[GPT2_DTYPE](
                "transformer.h." + String(l) + ".mlp.c_fc.weight",
                hp + base + l * (4 * C * C),
                transpose_rows=4 * C,
                transpose_cols=C,
            )
        base += L * (4 * C) * C
        for l in range(L):
            st.read_tensor[GPT2_DTYPE](
                "transformer.h." + String(l) + ".mlp.c_fc.bias",
                hp + base + l * (4 * C),
            )
        base += L * (4 * C)

        # proj_weight (mlp.c_proj): HF stores (4C, C), the transpose of our
        # (C, 4C).
        for l in range(L):
            st.read_tensor[GPT2_DTYPE](
                "transformer.h." + String(l) + ".mlp.c_proj.weight",
                hp + base + l * (C * 4 * C),
                transpose_rows=C,
                transpose_cols=4 * C,
            )
        base += L * C * (4 * C)
        for l in range(L):
            st.read_tensor[GPT2_DTYPE](
                "transformer.h." + String(l) + ".mlp.c_proj.bias",
                hp + base + l * C,
            )
        base += L * C

        st.read_tensor[GPT2_DTYPE]("transformer.ln_f.weight", hp + base)
        base += C
        st.read_tensor[GPT2_DTYPE]("transformer.ln_f.bias", hp + base)
        base += C

        self.params_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
            self.padded_num_parameters
        )
        self.params_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
        self.ctx.enqueue_copy(
            dst_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                self.params_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
            src_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], ImmutAnyOrigin]](
                hp.as_unsafe_any_origin()
            ),
            size=self.num_parameters,
        )
        self.ctx.synchronize()
        self.params_memory = rebind_mut_mem[GPT2_DTYPE](
            self.params_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.params.point_parameters(self.param_sizes, self.params_memory)

        self.has_allocated_params = True

    def allocate_gradients(mut self) raises:
        self.grads = ParameterTensors[GPT2_DTYPE]()
        self.grads_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
            self.padded_num_parameters
        )
        self.grads_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
        self.ctx.synchronize()
        self.grads_memory = rebind_mut_mem[GPT2_DTYPE](
            self.grads_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.grads.point_parameters(self.param_sizes, self.grads_memory)
        self.num_grads = self.num_parameters

        comptime if Self.WORLD_SIZE > 1:
            # ZeRO-2/3 shards the gradient communication (reduce-scatter).
            # ZeRO-1 uses allreduce so gradients stay fully replicated in grads_memory;
            # the optimizer reads grads_memory + rank*opt directly, no shard buffer needed.
            if self.zero_ctx.zero_stage >= 2:
                self.sharded_grads_buf = self.ctx.enqueue_create_buffer[
                    GPT2_DTYPE
                ](self.optimizer_num_parameters)
                self.sharded_grads_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
                self.ctx.synchronize()
                self.sharded_grads_memory = rebind_mut_mem[GPT2_DTYPE](
                    self.sharded_grads_buf.unsafe_ptr().as_unsafe_any_origin()
                )
            else:
                # ZeRO-0 (DDP) and ZeRO-1 keep gradients fully replicated;
                # no sharded_grads_buf is needed.
                self.sharded_grads_buf = self.ctx.enqueue_create_buffer[
                    GPT2_DTYPE
                ](1)
                var zero = 0
                self.sharded_grads_memory = MutMemPtr[GPT2_DTYPE](
                    unsafe_from_address=zero
                )
        else:
            self.sharded_grads_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
                1
            )
            var zero = 0
            self.sharded_grads_memory = MutMemPtr[GPT2_DTYPE](
                unsafe_from_address=zero
            )

        self.has_allocated_grads = True

    def allocate_optimizer_moments(mut self) raises:
        var n = self.optimizer_num_parameters
        # Adam moments are always fp32 (MASTER_DTYPE), independent of param dtype.
        self.m_buf = self.ctx.enqueue_create_buffer[MASTER_DTYPE](n)
        self.v_buf = self.ctx.enqueue_create_buffer[MASTER_DTYPE](n)
        self.m_buf.enqueue_fill(Scalar[MASTER_DTYPE](0.0))
        self.v_buf.enqueue_fill(Scalar[MASTER_DTYPE](0.0))
        self.ctx.synchronize()
        self.m_memory = rebind_mut_mem[MASTER_DTYPE](
            self.m_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.v_memory = rebind_mut_mem[MASTER_DTYPE](
            self.v_buf.unsafe_ptr().as_unsafe_any_origin()
        )

        comptime if USE_BF16:
            # Mixed precision: keep an fp32 master copy of the weights, seeded
            # from the loaded (low-precision) params promoted to fp32. A one-time
            # host round-trip with the cast (this runs once at startup).
            self.master_buf = self.ctx.enqueue_create_buffer[MASTER_DTYPE](n)
            self.master_memory = rebind_mut_mem[MASTER_DTYPE](
                self.master_buf.unsafe_ptr().as_unsafe_any_origin()
            )
            var host_p = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](n)
            var host_m = self.ctx.enqueue_create_host_buffer[MASTER_DTYPE](n)
            self.ctx.enqueue_copy(
                dst_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                    host_p.unsafe_ptr().as_unsafe_any_origin()
                ),
                src_ptr=rebind[
                    UnsafePointer[Scalar[GPT2_DTYPE], ImmutAnyOrigin]
                ](self.params_memory.as_unsafe_any_origin()),
                size=n,
            )
            self.ctx.synchronize()
            for i in range(n):
                host_m[i] = host_p[i].cast[MASTER_DTYPE]()
            self.ctx.enqueue_copy(
                dst_ptr=rebind[
                    UnsafePointer[Scalar[MASTER_DTYPE], MutAnyOrigin]
                ](self.master_memory.as_unsafe_any_origin()),
                src_ptr=rebind[
                    UnsafePointer[Scalar[MASTER_DTYPE], ImmutAnyOrigin]
                ](host_m.unsafe_ptr().as_unsafe_any_origin()),
                size=n,
            )
            self.ctx.synchronize()

        # Persistent scratch for calculate_grad_norm's GPU reduction (see the
        # field comment): sized once here to grid_x, the same
        # num_sm/BLOCK_SIZE/RESIDENT_THREADS formula calculate_grad_norm uses
        # to launch global_norm_squared_gpu, so it never needs to reallocate
        # per step. CPU targets don't reduce through this buffer, so a size-1
        # placeholder (matching the __init__ placeholder above) is enough.
        comptime if is_gpu[Self.target]():
            comptime BLOCK_SIZE = 512
            comptime RESIDENT_THREADS = 2048
            var num_sm = self.ctx.get_attribute(
                DeviceAttribute.MULTIPROCESSOR_COUNT
            )
            self.grad_norm_grid_x = ceildiv(
                num_sm * RESIDENT_THREADS, BLOCK_SIZE
            )
            self.grad_norm_out_buf = self.ctx.enqueue_create_buffer[
                DType.float32
            ](self.grad_norm_grid_x)
            self.grad_norm_host_buf = self.ctx.enqueue_create_host_buffer[
                DType.float32
            ](1)

        self.has_allocated_optimizer_moments = True

    def allocate_activations(mut self, batch_size: Int, seq_len: Int) raises:
        self.batch_size = batch_size
        self.seq_len = seq_len

        var B = batch_size
        var T = seq_len
        var C = self.config.channels
        var V_p = self.config.padded_vocab_size
        var L = self.config.num_layer
        var NH = self.config.num_heads

        self.act_sizes[Activations.encoded] = B * T * C
        self.act_sizes[Activations.ln_1] = L * B * T * C
        self.act_sizes[Activations.ln_1_mean] = L * B * T
        self.act_sizes[Activations.ln_1_rstd] = L * B * T
        self.act_sizes[Activations.qkv] = L * B * T * 3 * C
        self.act_sizes[Activations.lse] = L * B * NH * T
        self.act_sizes[Activations.attn] = L * B * T * C
        self.act_sizes[Activations.attn_proj] = L * B * T * C
        self.act_sizes[Activations.residual_2] = L * B * T * C
        self.act_sizes[Activations.ln_2] = L * B * T * C
        self.act_sizes[Activations.ln_2_mean] = L * B * T
        self.act_sizes[Activations.ln_2_rstd] = L * B * T
        # With activation recompute on, fch and fch_gelu are not persisted per
        # layer: they collapse to a single-layer scratch slot that forward fills
        # and backward rematerializes via the fused FC matmul (see `recompute`).
        var fch_layers = L
        comptime if Self.recompute:
            fch_layers = 1
        self.act_sizes[Activations.fch] = fch_layers * B * T * 4 * C
        self.act_sizes[Activations.fch_gelu] = fch_layers * B * T * 4 * C
        self.act_sizes[Activations.fc_proj] = L * B * T * C
        self.act_sizes[Activations.residual_3] = L * B * T * C
        self.act_sizes[Activations.ln_f] = B * T * C
        self.act_sizes[Activations.ln_f_mean] = B * T
        self.act_sizes[Activations.ln_f_rstd] = B * T
        self.act_sizes[Activations.logits] = B * T * V_p
        self.act_sizes[Activations.losses] = B * T
        self.act_sizes[Activations.q] = L * B * T * C
        self.act_sizes[Activations.k] = L * B * T * C
        self.act_sizes[Activations.v] = L * B * T * C
        self.act_sizes[Activations.attn_merged] = L * B * T * C
        self.act_sizes[Activations.att_probs] = L * B * NH * T * T

        var num_activations = 0
        for i in range(NUM_ACTIVATION_TENSORS):
            num_activations += self.act_sizes[i]
        self.num_activations = num_activations

        print("Number of Activations: " + String(self.num_activations))

        # `grad_act_sizes` mirrors `act_sizes`, except that on GPU builds
        # `fch`, `logits`, and `att_probs` are zeroed out: those three
        # `grad_acts` tensors are never meaningfully read or written on the
        # GPU backward path (`logits`/`fch` grads are passed only as the
        # unused `scratch` arg of `matmul_bwd`'s GPU d_weight path —
        # llmm/matmul.mojo `matmul_d_weight_bwd`, `is_gpu[target]()` branch
        # never touches `scratch_ptr`, cuBLAS and vendor-neutral fallback
        # alike; `att_probs`'s grad is never referenced at all — backward
        # reads the *forward*-stored probs via `acts.att_probs`/`kv_cache`,
        # and dP/dS/P live in separate persistent GEMM scratch). This drops
        # `grad_acts_buf` by ~1.9 GB. The CPU d_weight path's `else` branch
        # DOES materialize into `scratch_ptr` (host transpose), so CPU keeps
        # the full sizes; `att_probs` is unused on both targets but left
        # full-size on CPU for simplicity.
        for i in range(NUM_ACTIVATION_TENSORS):
            self.grad_act_sizes[i] = self.act_sizes[i]
        comptime if is_gpu[Self.target]():
            self.grad_act_sizes[Activations.fch] = 0
            self.grad_act_sizes[Activations.logits] = 0
            self.grad_act_sizes[Activations.att_probs] = 0

        # The fp32 statistics/losses are split into their own buffer; everything
        # else stays in the main (param-precision) buffer.
        var stats_size = (
            self.act_sizes[Activations.ln_1_mean]
            + self.act_sizes[Activations.ln_1_rstd]
            + self.act_sizes[Activations.lse]
            + self.act_sizes[Activations.ln_2_mean]
            + self.act_sizes[Activations.ln_2_rstd]
            + self.act_sizes[Activations.ln_f_mean]
            + self.act_sizes[Activations.ln_f_rstd]
            + self.act_sizes[Activations.losses]
        )
        var main_size = num_activations - stats_size

        var num_grad_activations = 0
        for i in range(NUM_ACTIVATION_TENSORS):
            num_grad_activations += self.grad_act_sizes[i]
        # `stats_size` is identical for grad_act_sizes (none of the three
        # shrunk tensors are stats entries).
        var grad_main_size = num_grad_activations - stats_size
        print("Number of Activation Gradients: " + String(num_grad_activations))

        # Re-allocate device memory and point the structures
        self.acts_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](main_size)
        # Zero once so the stored att_probs' above-diagonal half stays 0 (the
        # causal softmax writes only the lower triangle each step).
        self.ctx.enqueue_memset(self.acts_buf, Scalar[GPT2_DTYPE](0))
        self.grad_acts_buf = self.ctx.enqueue_create_buffer[GPT2_DTYPE](
            grad_main_size
        )
        # One-time full zero at allocation time. Per step, `zero_gradients`
        # (GPU path) only re-zeroes the five accumulator tensors (encoded,
        # attn_proj, residual_2, fc_proj, residual_3); every other grad_acts
        # tensor is overwritten before its first read or never touched on
        # the GPU path (see the per-tensor trace this alloc-time zero backs
        # up as defense-in-depth for step 0).
        self.ctx.enqueue_memset(self.grad_acts_buf, Scalar[GPT2_DTYPE](0))
        self.acts_stats_buf = self.ctx.enqueue_create_buffer[StatsDType](
            stats_size
        )
        self.grad_acts_stats_buf = self.ctx.enqueue_create_buffer[StatsDType](
            stats_size
        )

        self.inputs_buf = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.batch_size * self.seq_len
        )
        self.targets_buf = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.batch_size * self.seq_len
        )
        # GPU-visible device copies of the token/target indices (see field docs).
        self.inputs_dev_buf = self.ctx.enqueue_create_buffer[DType.int32](
            self.batch_size * self.seq_len
        )
        self.targets_dev_buf = self.ctx.enqueue_create_buffer[DType.int32](
            self.batch_size * self.seq_len
        )

        # Encoder backward scratch: one bucket per (token, channel_group).
        var wte_c_per_warp = 128
        var num_channel_groups = (C + wte_c_per_warp - 1) // wte_c_per_warp
        self.wte_bucket_capacity = B * T * num_channel_groups
        self.bucket_info_buf = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.wte_bucket_capacity * 4
        )
        self.workload_indices_buf = self.ctx.enqueue_create_host_buffer[
            DType.int32
        ](B * T)

        self.losses_host_buf = self.ctx.enqueue_create_host_buffer[StatsDType](
            self.batch_size * self.seq_len
        )
        self.logits_host_buf = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
            self.config.padded_vocab_size
        )

        self.ctx.synchronize()

        self.acts_memory = rebind_mut_mem[GPT2_DTYPE](
            self.acts_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.grad_acts_memory = rebind_mut_mem[GPT2_DTYPE](
            self.grad_acts_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.acts_stats_memory = rebind_mut_mem[StatsDType](
            self.acts_stats_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.grad_acts_stats_memory = rebind_mut_mem[StatsDType](
            self.grad_acts_stats_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.inputs = rebind_mut_mem[DType.int32](
            self.inputs_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.targets = rebind_mut_mem[DType.int32](
            self.targets_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.inputs_dev = rebind_mut_mem[DType.int32](
            self.inputs_dev_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.targets_dev = rebind_mut_mem[DType.int32](
            self.targets_dev_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.bucket_info = rebind_mut_mem[DType.int32](
            self.bucket_info_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.workload_indices = rebind_mut_mem[DType.int32](
            self.workload_indices_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.bucket_info_dev_buf = self.ctx.enqueue_create_buffer[DType.int32](
            self.wte_bucket_capacity * 4
        )
        self.workload_indices_dev_buf = self.ctx.enqueue_create_buffer[
            DType.int32
        ](B * T)
        self.bucket_info_dev = rebind_mut_mem[DType.int32](
            self.bucket_info_dev_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        self.workload_indices_dev = rebind_mut_mem[DType.int32](
            self.workload_indices_dev_buf.unsafe_ptr().as_unsafe_any_origin()
        )

        self.acts.point_activations(
            self.act_sizes, self.acts_memory, self.acts_stats_memory
        )
        self.grad_acts.point_activations(
            self.grad_act_sizes,
            self.grad_acts_memory,
            self.grad_acts_stats_memory,
        )
        self.has_allocated_acts = True

    def build_encoder_buckets(mut self, batch_size: Int, seq_len: Int) raises:
        self.num_wte_buckets = build_wte_buckets(
            as_immut_kernel_from_mut[DType.int32](self.inputs),
            as_mut_kernel[DType.int32](self.bucket_info),
            as_mut_kernel[DType.int32](self.workload_indices),
            batch_size,
            seq_len,
            self.config.vocab_size,
            self.config.channels,
            self.wte_bucket_capacity,
        )

    def forward(
        mut self,
        inputs: MutMemPtr[DType.int32],
        targets: MutMemPtr[DType.int32],
        batch_size: Int,
        seq_len: Int,
        grad_accum_steps: Int = 1,
    ) raises:
        var zero = 0
        var NULL_DTYPE_PTR = MutMemPtr[GPT2_DTYPE](unsafe_from_address=zero)
        var NULL_MASTER_PTR = MutMemPtr[MASTER_DTYPE](unsafe_from_address=zero)
        var NULL_INT32_PTR = MutMemPtr[DType.int32](unsafe_from_address=zero)

        if (
            not self.has_allocated_params
            or self.params_memory == NULL_DTYPE_PTR
        ):
            raise Error("GPT2 error: Parameters not allocated")
        if not self.has_allocated_grads or self.grads_memory == NULL_DTYPE_PTR:
            raise Error("GPT2 error: Gradients not allocated")
        if (
            not self.has_allocated_optimizer_moments
            or self.m_memory == NULL_MASTER_PTR
            or self.v_memory == NULL_MASTER_PTR
        ):
            raise Error("GPT2 error: Optimizer moments not allocated")

        var vocab_size = self.config.vocab_size
        var vocab_size_padded = self.config.padded_vocab_size
        var num_layers = self.config.num_layer
        var num_heads = self.config.num_heads
        var channels = self.config.channels

        # Lazily allocate activations if needed.
        if (
            not self.has_allocated_acts
            or self.acts_memory == NULL_DTYPE_PTR
            or self.grad_acts_memory == NULL_DTYPE_PTR
        ):
            self.allocate_activations(batch_size, seq_len)
        else:
            # Validate activations and gradients are not larger then the prvious allocations.
            # In the future we could resize and reallocate for now we will just raise an error.
            if seq_len > self.seq_len or batch_size > self.batch_size:
                raise Error(
                    "GPT2 error: Sequence length or batch size is larger than"
                    " the previous allocations"
                )

        memcpy(dest=self.inputs, src=inputs, count=batch_size * seq_len)

        if targets != NULL_INT32_PTR:
            memcpy(dest=self.targets, src=targets, count=batch_size * seq_len)

        # On GPU, upload the token/target indices into device-resident buffers.
        # A HostBuffer pointer read from inside a Metal kernel silently yields
        # zeros, so the encoder / fused classifier must read these device copies.
        # `enc_inputs` / `cls_targets` pick the correct pointer per target below.
        var enc_inputs = self.inputs
        var cls_targets = self.targets
        comptime if is_gpu[Self.target]():
            self.ctx.enqueue_copy(
                dst_ptr=rebind[
                    UnsafePointer[Scalar[DType.int32], MutAnyOrigin]
                ](self.inputs_dev.as_unsafe_any_origin()),
                src_ptr=rebind[
                    UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin]
                ](self.inputs.as_unsafe_any_origin()),
                size=batch_size * seq_len,
            )
            enc_inputs = self.inputs_dev
            if targets != NULL_INT32_PTR:
                self.ctx.enqueue_copy(
                    dst_ptr=rebind[
                        UnsafePointer[Scalar[DType.int32], MutAnyOrigin]
                    ](self.targets_dev.as_unsafe_any_origin()),
                    src_ptr=rebind[
                        UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin]
                    ](self.targets.as_unsafe_any_origin()),
                    size=batch_size * seq_len,
                )
                cls_targets = self.targets_dev
            # Metal: in-order queue guarantees host→device copies complete
            # before subsequent GPU kernels see inputs_dev/targets_dev; staging
            # buffer reuse is safe because forward() syncs at the mean_loss
            # read-back (line ~1925) before the next micro-step can overwrite.
            comptime if not HAS_METAL:
                self.ctx.synchronize()

        self.build_encoder_buckets(batch_size, seq_len)

        # On GPU (Metal), bucket_info/workload_indices are HostBuffers and their
        # raw pointers read as zeros inside Metal kernels. Upload the just-built
        # host data to device buffers so the wte-backward GPU kernel reads correctly.
        comptime if is_gpu[Self.target]():
            self.ctx.enqueue_copy(
                dst_ptr=rebind[
                    UnsafePointer[Scalar[DType.int32], MutAnyOrigin]
                ](self.bucket_info_dev.as_unsafe_any_origin()),
                src_ptr=rebind[
                    UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin]
                ](self.bucket_info.as_unsafe_any_origin()),
                size=self.num_wte_buckets * 4,
            )
            self.ctx.enqueue_copy(
                dst_ptr=rebind[
                    UnsafePointer[Scalar[DType.int32], MutAnyOrigin]
                ](self.workload_indices_dev.as_unsafe_any_origin()),
                src_ptr=rebind[
                    UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin]
                ](self.workload_indices.as_unsafe_any_origin()),
                size=batch_size * seq_len,
            )
            # Metal: in-order queue ensures bucket_info_dev/workload_indices_dev
            # are visible to encoder_fwd before it starts — no explicit sync needed.
            comptime if not HAS_METAL:
                self.ctx.synchronize()

        comptime if Self.WORLD_SIZE > 1:
            # ZeRO-3: Gather all parameter shards into params_buf before running forward.
            # Each rank holds its persistent shard at params_memory + rank * optimizer_num_parameters.
            # This is a coarse-grained gather: one allgather per forward pass rather than per layer.
            # It is memory-correct ZeRO-3 sharding of persistent state, but does NOT yet achieve
            # the peak-memory savings of true per-layer streaming (gather/free per tensor).
            if self.zero_ctx.zero_stage >= 3:
                self.zero_ctx.allgather(
                    rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                        self.params_memory.as_unsafe_any_origin()
                    ),
                    self.optimizer_num_parameters,
                )
                self.ctx.synchronize()

        #########################################################
        # Forward Pass
        #########################################################

        var residual: MutMemPtr[GPT2_DTYPE]
        encoder_fwd[GPT2_DTYPE, Self.target](
            as_mut_kernel[GPT2_DTYPE](self.acts.encoded),
            as_immut_kernel_from_mut[DType.int32](enc_inputs),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.params.wte),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.params.wpe),
            batch_size,
            seq_len,
            channels,
            self.ctx,
        )

        for layer in range(num_layers):
            if (
                layer == 0
            ):  # First layer, use the encoded activations as the pointer
                residual = self.acts.encoded
            else:
                residual = (
                    self.acts.residual_3
                    + (layer - 1) * batch_size * seq_len * channels
                )

            var l_ln_1_gamma = self.params.ln_1_gamma + layer * channels
            var l_ln_1_beta = self.params.ln_1_beta + layer * channels
            var l_qkv_weight = (
                self.params.qkv_weight + layer * (3 * channels) * channels
            )
            var l_qkv_bias = self.params.qkv_bias + layer * (3 * channels)
            var l_attn_proj_weight = (
                self.params.attn_proj_weight + layer * channels * channels
            )
            var l_attn_proj_bias = self.params.attn_proj_bias + layer * channels
            var l_ln_2_gamma = self.params.ln_2_gamma + layer * channels
            var l_ln_2_beta = self.params.ln_2_beta + layer * channels
            var l_fc_weight = (
                self.params.fc_weight + layer * (4 * channels) * channels
            )
            var l_fc_bias = self.params.fc_bias + layer * (4 * channels)
            var l_proj_weight = self.params.proj_weight + layer * channels * (
                4 * channels
            )
            var l_proj_bias = self.params.proj_bias + layer * channels

            var l_ln_1 = (
                self.acts.ln_1 + layer * batch_size * seq_len * channels
            )
            var l_ln_1_mean = self.acts.ln_1_mean + layer * batch_size * seq_len
            var l_ln_1_rstd = self.acts.ln_1_rstd + layer * batch_size * seq_len
            var l_qkv = self.acts.qkv + layer * batch_size * seq_len * (
                3 * channels
            )
            var l_q = self.acts.q + layer * batch_size * seq_len * channels
            var l_k = self.acts.k + layer * batch_size * seq_len * channels
            var l_v = self.acts.v + layer * batch_size * seq_len * channels
            var l_lse = self.acts.lse + layer * batch_size * num_heads * seq_len
            var l_attn = (
                self.acts.attn + layer * batch_size * seq_len * channels
            )
            var l_attn_merged = (
                self.acts.attn_merged + layer * batch_size * seq_len * channels
            )
            var l_attn_proj = (
                self.acts.attn_proj + layer * batch_size * seq_len * channels
            )
            var l_residual_2 = (
                self.acts.residual_2 + layer * batch_size * seq_len * channels
            )
            var l_ln_2 = (
                self.acts.ln_2 + layer * batch_size * seq_len * channels
            )
            var l_ln_2_mean = self.acts.ln_2_mean + layer * batch_size * seq_len
            var l_ln_2_rstd = self.acts.ln_2_rstd + layer * batch_size * seq_len
            # fch / fch_gelu live in a single-layer scratch slot when recompute
            # is on, so their per-layer offset collapses to 0.
            var fch_layer = layer
            comptime if Self.recompute:
                fch_layer = 0
            var l_fch = self.acts.fch + fch_layer * batch_size * seq_len * (
                4 * channels
            )
            var l_fch_gelu = (
                self.acts.fch_gelu
                + fch_layer * batch_size * seq_len * (4 * channels)
            )
            var l_fc_proj = (
                self.acts.fc_proj + layer * batch_size * seq_len * channels
            )
            var l_residual_3 = (
                self.acts.residual_3 + layer * batch_size * seq_len * channels
            )

            # 1. LayerNorm 1 / Residual Fused
            if layer == 0:
                layernorm_fwd[GPT2_DTYPE, Self.target](
                    as_mut_kernel[GPT2_DTYPE](l_ln_1),
                    as_mut_kernel[GPT2_DTYPE](residual),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_1_gamma),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_1_beta),
                    EPSILON,
                    as_mut_kernel[StatsDType](l_ln_1_mean),
                    as_mut_kernel[StatsDType](l_ln_1_rstd),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    self.ctx,
                )
            else:
                var prev_residual_2 = (
                    self.acts.residual_2
                    + (layer - 1) * batch_size * seq_len * channels
                )
                var prev_fc_proj = (
                    self.acts.fc_proj
                    + (layer - 1) * batch_size * seq_len * channels
                )
                layernorm_fused_residual_fwd[GPT2_DTYPE, Self.target](
                    as_mut_kernel[GPT2_DTYPE](residual),
                    as_mut_kernel[GPT2_DTYPE](l_ln_1),
                    as_mut_kernel[GPT2_DTYPE](prev_residual_2),
                    as_mut_kernel[GPT2_DTYPE](prev_fc_proj),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_1_gamma),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_1_beta),
                    EPSILON,
                    as_mut_kernel[StatsDType](l_ln_1_mean),
                    as_mut_kernel[StatsDType](l_ln_1_rstd),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    self.ctx,
                )

            # Matmul QKV. Always bf16, even under LLMM_PRECISION=fp4 (the
            # recipe keeps attention's QKV/out proj + softmax out of FP4 —
            # docs/ai/fp4_training_recipes_research.md §1 "Selective
            # high-precision layers"); only PRECISION=="fp8" takes the
            # matmul_fwd_lowp path here.
            comptime if PRECISION == "fp8" and FP8_SITE_QKV:
                matmul_fwd_lowp[GPT2_DTYPE, Self.target, use_gelu=False](
                    as_mut_kernel[GPT2_DTYPE](l_qkv),
                    as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                    as_mut_kernel[GPT2_DTYPE](l_ln_1),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_qkv_weight),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_qkv_bias),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    Int64(3 * channels),
                    self.fp8_state.qkv_input[layer],
                    self.fp8_state.qkv_weight[layer],
                    "qkv",
                    layer,
                    self.ctx,
                )
            else:
                matmul_fwd[GPT2_DTYPE, Self.target, use_gelu=False](
                    as_mut_kernel[GPT2_DTYPE](l_qkv),
                    as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                    as_mut_kernel[GPT2_DTYPE](l_ln_1),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_qkv_weight),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_qkv_bias),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    Int64(3 * channels),
                    self.ctx,
                )

            # Attention Forward (handles QKV Split/Transpose and Merge Heads internally).
            var head_dim = channels // num_heads
            # Store-P: the forward stores this layer's softmax probs P into
            # acts.att_probs[layer] so the backward reads them back instead of
            # recomputing QKᵀ. Enabled on BOTH targets.
            #
            # NOTE (Metal recompute): the QKᵀ-recompute backward is now
            # numerically correct — the root-cause bug was in attention.mojo,
            # where the Metal backward reused the single, layer-overwritten
            # gemm_att scratch (p_buf) as "the stored P". Since forward and
            # backward run as two separate whole-model loops, p_buf held only the
            # LAST forward layer's P at backward time, aliasing every backward
            # layer to it and amplifying gradients with depth. That fast-path is
            # removed; disabling the store (att_probs_addr=0) now takes the true
            # per-layer QKᵀ recompute (correct, verified 16/16). It is NOT enabled
            # because it is a net LOSS with the current tensor-core Metal GEMM
            # kernels: measured +3.5% step time (fp32 736.6→762.9 ms, bf16
            # 587.1→606.8 ms, B=4 T=1024) — the att_probs store is no longer the
            # bottleneck, so the extra backward QKᵀ GEMM costs more than the store
            # saves. Flip these two assignments to `if not HAS_METAL` /
            # att_probs_addr=0 to trade that 3.5% for ~2.4 GB (fp32) less memory
            # if T-scaling (att_probs grows as T²) ever makes the store dominate.
            self.kv_cache.att_probs_addr = Int(self.acts.att_probs)
            self.kv_cache.att_probs_layer = layer
            self.kv_cache.att_probs_stride = (
                batch_size * num_heads * seq_len * seq_len
            )
            attention_fwd[GPT2_DTYPE, Self.target, use_soft_exp=True](
                as_mut_kernel[GPT2_DTYPE](l_qkv),
                as_mut_kernel[GPT2_DTYPE](l_q),
                as_mut_kernel[GPT2_DTYPE](l_k),
                as_mut_kernel[GPT2_DTYPE](l_v),
                as_mut_kernel[GPT2_DTYPE](l_attn),
                as_mut_kernel[GPT2_DTYPE](l_attn_merged),
                as_mut_kernel[StatsDType](l_lse),
                Int64(batch_size),
                Int64(num_heads),
                Int64(seq_len),
                Int64(head_dim),
                self.ctx,
                cache=rebind[KVCachePtr](UnsafePointer(to=self.kv_cache)),
            )

            # Matmul Attn Proj. Always bf16 under fp4 (same rationale as the
            # QKV site above — attention stays out of FP4 per the recipe).
            comptime if PRECISION == "fp8" and FP8_SITE_ATTN_PROJ:
                matmul_fwd_lowp[GPT2_DTYPE, Self.target, use_gelu=False](
                    as_mut_kernel[GPT2_DTYPE](l_attn_proj),
                    as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                    as_mut_kernel[GPT2_DTYPE](l_attn_merged),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_attn_proj_weight),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_attn_proj_bias),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    Int64(channels),
                    self.fp8_state.attn_proj_input[layer],
                    self.fp8_state.attn_proj_weight[layer],
                    "attn_proj",
                    layer,
                    self.ctx,
                )
            else:
                matmul_fwd[GPT2_DTYPE, Self.target, use_gelu=False](
                    as_mut_kernel[GPT2_DTYPE](l_attn_proj),
                    as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                    as_mut_kernel[GPT2_DTYPE](l_attn_merged),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_attn_proj_weight),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_attn_proj_bias),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    Int64(channels),
                    self.ctx,
                )
            # LayerNorm 2 & Residual.
            layernorm_fused_residual_fwd[GPT2_DTYPE, Self.target](
                as_mut_kernel[GPT2_DTYPE](l_residual_2),
                as_mut_kernel[GPT2_DTYPE](l_ln_2),
                as_mut_kernel[GPT2_DTYPE](residual),
                as_mut_kernel[GPT2_DTYPE](l_attn_proj),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_2_gamma),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_2_beta),
                EPSILON,
                as_mut_kernel[StatsDType](l_ln_2_mean),
                as_mut_kernel[StatsDType](l_ln_2_rstd),
                Int64(batch_size),
                Int64(seq_len),
                Int64(channels),
                self.ctx,
            )

            # Matmul FC (fused GELU). One of the two FP4-eligible MLP
            # linears (docs/ai/fp4_training_recipes_research.md §1): under
            # PRECISION=="fp4", middle blocks (_layer_in_fp4_range) take
            # matmul_fwd_fp4; first/last blocks fall through to the same
            # bf16 matmul_fwd the fp32/bf16 builds use. fp8 is unchanged
            # (matmul_fwd_lowp on every layer, byte-identical to before).
            comptime if PRECISION == "fp8" and FP8_SITE_FC:
                matmul_fwd_lowp[GPT2_DTYPE, Self.target, use_gelu=True](
                    as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                    as_mut_kernel[GPT2_DTYPE](l_fch),
                    as_mut_kernel[GPT2_DTYPE](l_ln_2),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_weight),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_bias),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    Int64(4 * channels),
                    self.fp8_state.fc_input[layer],
                    self.fp8_state.fc_weight[layer],
                    "fc",
                    layer,
                    self.ctx,
                )
            elif PRECISION == "fp4":
                if _layer_in_fp4_range(layer, num_layers):
                    matmul_fwd_fp4[GPT2_DTYPE, Self.target, use_gelu=True](
                        as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                        as_mut_kernel[GPT2_DTYPE](l_fch),
                        as_mut_kernel[GPT2_DTYPE](l_ln_2),
                        as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_weight),
                        as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_bias),
                        Int64(batch_size),
                        Int64(seq_len),
                        Int64(channels),
                        Int64(4 * channels),
                        self.ctx,
                    )
                else:
                    matmul_fwd[GPT2_DTYPE, Self.target, use_gelu=True](
                        as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                        as_mut_kernel[GPT2_DTYPE](l_fch),
                        as_mut_kernel[GPT2_DTYPE](l_ln_2),
                        as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_weight),
                        as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_bias),
                        Int64(batch_size),
                        Int64(seq_len),
                        Int64(channels),
                        Int64(4 * channels),
                        self.ctx,
                    )
            else:
                matmul_fwd[GPT2_DTYPE, Self.target, use_gelu=True](
                    as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                    as_mut_kernel[GPT2_DTYPE](l_fch),
                    as_mut_kernel[GPT2_DTYPE](l_ln_2),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_weight),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_bias),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    Int64(4 * channels),
                    self.ctx,
                )

            # Matmul Proj. The other FP4-eligible MLP linear — same
            # three-way dispatch as the FC site above.
            comptime if PRECISION == "fp8" and FP8_SITE_PROJ:
                matmul_fwd_lowp[GPT2_DTYPE, Self.target, use_gelu=False](
                    as_mut_kernel[GPT2_DTYPE](l_fc_proj),
                    as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                    as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_weight),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_bias),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(4 * channels),
                    Int64(channels),
                    self.fp8_state.proj_input[layer],
                    self.fp8_state.proj_weight[layer],
                    "proj",
                    layer,
                    self.ctx,
                )
            elif PRECISION == "fp4":
                if _layer_in_fp4_range(layer, num_layers):
                    matmul_fwd_fp4[GPT2_DTYPE, Self.target, use_gelu=False](
                        as_mut_kernel[GPT2_DTYPE](l_fc_proj),
                        as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                        as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                        as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_weight),
                        as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_bias),
                        Int64(batch_size),
                        Int64(seq_len),
                        Int64(4 * channels),
                        Int64(channels),
                        self.ctx,
                    )
                else:
                    matmul_fwd[GPT2_DTYPE, Self.target, use_gelu=False](
                        as_mut_kernel[GPT2_DTYPE](l_fc_proj),
                        as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                        as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                        as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_weight),
                        as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_bias),
                        Int64(batch_size),
                        Int64(seq_len),
                        Int64(4 * channels),
                        Int64(channels),
                        self.ctx,
                    )
            else:
                matmul_fwd[GPT2_DTYPE, Self.target, use_gelu=False](
                    as_mut_kernel[GPT2_DTYPE](l_fc_proj),
                    as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                    as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_weight),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_bias),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(4 * channels),
                    Int64(channels),
                    self.ctx,
                )

        # Final LayerNorm
        var last_residual_3 = (
            self.acts.residual_3
            + (num_layers - 1) * batch_size * seq_len * channels
        )
        var last_residual_2 = (
            self.acts.residual_2
            + (num_layers - 1) * batch_size * seq_len * channels
        )
        var last_fc_proj = (
            self.acts.fc_proj
            + (num_layers - 1) * batch_size * seq_len * channels
        )
        layernorm_fused_residual_fwd[GPT2_DTYPE, Self.target](
            as_mut_kernel[GPT2_DTYPE](last_residual_3),
            as_mut_kernel[GPT2_DTYPE](self.acts.ln_f),
            as_mut_kernel[GPT2_DTYPE](last_residual_2),
            as_mut_kernel[GPT2_DTYPE](last_fc_proj),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.params.ln_f_gamma),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.params.ln_f_beta),
            EPSILON,
            as_mut_kernel[StatsDType](self.acts.ln_f_mean),
            as_mut_kernel[StatsDType](self.acts.ln_f_rstd),
            Int64(batch_size),
            Int64(seq_len),
            Int64(channels),
            self.ctx,
        )

        # Output Logits (wte has no bias).
        matmul_fwd[GPT2_DTYPE, Self.target, use_gelu=False, has_bias=False](
            as_mut_kernel[GPT2_DTYPE](self.acts.logits),
            as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
            as_mut_kernel[GPT2_DTYPE](self.acts.ln_f),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.params.wte),
            as_immut_kernel_from_mut[GPT2_DTYPE](NULL_DTYPE_PTR),
            Int64(batch_size),
            Int64(seq_len),
            Int64(channels),
            Int64(vocab_size_padded),
            self.ctx,
        )
        # Metal: matmul_fwd and fused_classifier are both GPU kernels; the
        # in-order queue sequences them without an explicit sync.
        comptime if not HAS_METAL:
            self.ctx.synchronize()

        # Fused classifier: cross-entropy loss AND dlogits in ONE pass (llm.c's
        # structure). Since dL/dloss = 1/(B·T·grad_accum_steps) is a constant, we
        # seed it here and write dlogits in-place into acts.logits now — the
        # backward reads them directly and skips a second 206M-element pass.
        if targets != NULL_INT32_PTR:
            var dloss_mean = Scalar[DType.float32](1.0) / Scalar[DType.float32](
                batch_size * seq_len * grad_accum_steps
            )
            for i in range(batch_size * seq_len):
                self.losses_host_buf[i] = dloss_mean
            self.ctx.enqueue_copy(
                dst_ptr=rebind[UnsafePointer[Scalar[StatsDType], MutAnyOrigin]](
                    self.grad_acts.losses.as_unsafe_any_origin()
                ),
                src_ptr=rebind[
                    UnsafePointer[Scalar[StatsDType], ImmutAnyOrigin]
                ](self.losses_host_buf.unsafe_ptr().as_unsafe_any_origin()),
                size=batch_size * seq_len,
            )
            fused_classifier[GPT2_DTYPE, Self.target, write_d_logits=True](
                as_mut_kernel[GPT2_DTYPE](self.acts.logits),
                as_mut_kernel[StatsDType](self.acts.losses),
                as_immut_kernel_from_mut[DType.float32](self.grad_acts.losses),
                as_immut_kernel_from_mut[DType.int32](cls_targets),
                Int64(batch_size),
                Int64(seq_len),
                Int64(vocab_size),
                Int64(vocab_size_padded),
                self.ctx,
            )

            # Non-Metal: ensure fused_classifier completes before the
            # device→host copy that follows (no in-order guarantee on CUDA).
            # Metal: in-order queue sequences fused_classifier → enqueue_copy
            # automatically; the actual CPU read is guarded by the sync below.
            comptime if not HAS_METAL:
                self.ctx.synchronize()

            var count = batch_size * seq_len
            self.ctx.enqueue_copy(
                dst_ptr=rebind[UnsafePointer[Scalar[StatsDType], MutAnyOrigin]](
                    self.losses_host_buf.unsafe_ptr().as_unsafe_any_origin()
                ),
                src_ptr=rebind[
                    UnsafePointer[Scalar[StatsDType], ImmutAnyOrigin]
                ](self.acts.losses.as_unsafe_any_origin()),
                size=count,
            )
            self.ctx.synchronize()

            var total_loss: Float32 = 0.0
            for i in range(count):
                total_loss += self.losses_host_buf[i].cast[DType.float32]()
            self.mean_loss = total_loss / Float32(count)

    def zero_gradients(mut self, zero_param_grads: Bool = True) raises:
        # Activation gradients are scratch and are always re-zeroed. Parameter
        # gradients accumulate across micro-steps, so they are only zeroed on the
        # first micro-step (zero_param_grads=False on later micro-steps lets the
        # backward kernels keep `+=`-accumulating into them).
        if zero_param_grads and self.has_allocated_grads:
            self.grads_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
            comptime if Self.WORLD_SIZE > 1:
                if self.zero_ctx.zero_stage >= 2:
                    self.sharded_grads_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))

        comptime if is_cpu[Self.target]():
            # CPU backward kernels were not audited for overwrite-vs-accumulate
            # semantics (that trace only covers the GPU dispatch), so keep the
            # original full-buffer zero here (guarded exactly as before this
            # change). This is not the perf target.
            if self.has_allocated_acts:
                self.grad_acts_buf.enqueue_fill(Scalar[GPT2_DTYPE](0.0))
        else:
            if self.has_allocated_acts:
                # Of the 18 tensors in grad_acts_buf, only 5 are read-modify-
                # write accumulators: `encoded`, `attn_proj`, `residual_2`,
                # `fc_proj`, and `residual_3` are the `+=` targets of the
                # fused-residual-backward broadcast (layernorm.mojo
                # ~1653-1663). Every other tensor is overwritten before its
                # first read (e.g. cuBLASLt d_input GEMMs, beta=0) or never
                # touched on the GPU path (`fch`, `logits`, `att_probs` are
                # passed only as an unused GPU-path scratch arg). Zeroing just
                # these accumulators instead of the full 3.29 GB buffer drops
                # the fill from ~17.5 ms to ~1.6 ms/step.
                #
                # `attn_proj`+`residual_2` and `fc_proj`+`residual_3` are
                # adjacent in `ActivationTensors.point_activations`'s buffer
                # layout (no stats tensor in between — those live in a
                # separate buffer), so each pair is one contiguous fill.
                # Offsets/counts are derived from `self.grad_act_sizes` (not
                # `self.act_sizes`, and not hardcoded): on GPU builds `fch`,
                # `logits`, `att_probs` are sized 0 in `grad_act_sizes` (see
                # `allocate_activations`), which shifts the fc_proj/
                # residual_3 offset down by `fch`'s size — using `act_sizes`
                # here (the *forward* acts layout, still full-size) would
                # compute an offset into memory `grad_acts_buf` no longer
                # has. This also still tracks other config changes
                # (`Self.recompute` shrinking `fch`/`fch_gelu`).
                var off_encoded = 0
                var cnt_encoded = self.grad_act_sizes[Activations.encoded]

                var off_attn_proj = (
                    self.grad_act_sizes[Activations.encoded]
                    + self.grad_act_sizes[Activations.ln_1]
                    + self.grad_act_sizes[Activations.qkv]
                    + self.grad_act_sizes[Activations.attn]
                )
                var cnt_attn_proj_residual_2 = (
                    self.grad_act_sizes[Activations.attn_proj]
                    + self.grad_act_sizes[Activations.residual_2]
                )

                var off_fc_proj = (
                    off_attn_proj
                    + cnt_attn_proj_residual_2
                    + self.grad_act_sizes[Activations.ln_2]
                    + self.grad_act_sizes[Activations.fch]
                    + self.grad_act_sizes[Activations.fch_gelu]
                )
                var cnt_fc_proj_residual_3 = (
                    self.grad_act_sizes[Activations.fc_proj]
                    + self.grad_act_sizes[Activations.residual_3]
                )

                self.ctx.enqueue_memset(
                    self.grad_acts_buf.create_sub_buffer[GPT2_DTYPE](
                        off_encoded, cnt_encoded
                    ),
                    Scalar[GPT2_DTYPE](0),
                )
                self.ctx.enqueue_memset(
                    self.grad_acts_buf.create_sub_buffer[GPT2_DTYPE](
                        off_attn_proj, cnt_attn_proj_residual_2
                    ),
                    Scalar[GPT2_DTYPE](0),
                )
                self.ctx.enqueue_memset(
                    self.grad_acts_buf.create_sub_buffer[GPT2_DTYPE](
                        off_fc_proj, cnt_fc_proj_residual_3
                    ),
                    Scalar[GPT2_DTYPE](0),
                )
        # Metal: enqueue_memset/fill ops and subsequent backward kernels are
        # all on the same in-order queue; no sync needed for GPU→GPU ordering.
        comptime if not HAS_METAL:
            self.ctx.synchronize()

    def backward(
        mut self,
        grad_accum_steps: Int = 1,
        micro_step: Int = 0,
        step: Int = 0,
    ) raises:
        if self.mean_loss == -1.0:
            raise Error("GPT2 error: must call forward pass first")
        if not self.has_allocated_acts:
            raise Error("GPT2 error: activations not allocated")

        # Zero the parameter gradients only at the first micro-step so they
        # accumulate over the gradient-accumulation loop.
        self.zero_gradients(zero_param_grads=(micro_step == 0))

        var zero = 0
        var NULL_DTYPE_PTR = MutMemPtr[GPT2_DTYPE](unsafe_from_address=zero)

        # fp4 SR step counter: unique per (training step, micro-step) so
        # grad-accum micro-steps draw distinct dither; only consulted under
        # fp4.
        var fp4_sr_step = step * grad_accum_steps + micro_step

        var batch_size = self.batch_size
        var seq_len = self.seq_len
        var channels = self.config.channels
        var num_layers = self.config.num_layer
        var num_heads = self.config.num_heads
        var vocab_size = self.config.vocab_size
        var vocab_size_padded = self.config.padded_vocab_size
        var head_dim = channels // num_heads
        var layer_stride = batch_size * seq_len * channels
        var qkv_layer_stride = layer_stride * 3
        var fch_layer_stride = layer_stride * 4

        ##############################################################
        # Backward Pass
        ##############################################################

        # NOTE: The fused classifier (cross-entropy loss + dlogits) now runs ONCE,
        # in forward(), which already wrote dlogits in-place into acts.logits with
        # the constant dL/dloss = 1/(B·T·grad_accum_steps). The backward reads them
        # directly here, eliminating a redundant 206M-element pass (matches llm.c).

        # LM head matmul backward (wte has no bias).
        matmul_bwd[GPT2_DTYPE, Self.target, use_gelu=False, has_bias=False](
            as_mut_kernel[GPT2_DTYPE](self.grad_acts.ln_f),
            as_mut_kernel[GPT2_DTYPE](self.grads.wte),
            as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
            as_mut_kernel[GPT2_DTYPE](self.acts.logits),
            as_mut_kernel[GPT2_DTYPE](self.acts.ln_f),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.params.wte),
            as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
            as_mut_kernel[GPT2_DTYPE](self.grad_acts.logits),
            Int64(batch_size),
            Int64(seq_len),
            Int64(channels),
            Int64(vocab_size_padded),
            self.ctx,
        )

        # Final fused LayerNorm backward.
        var last_layer = num_layers - 1
        var last_residual_3 = self.acts.residual_3 + last_layer * layer_stride
        var last_residual_2 = self.acts.residual_2 + last_layer * layer_stride
        var last_fc_proj = self.acts.fc_proj + last_layer * layer_stride
        var d_last_residual_2 = (
            self.grad_acts.residual_2 + last_layer * layer_stride
        )
        var d_last_fc_proj = self.grad_acts.fc_proj + last_layer * layer_stride

        layernorm_fused_residual_bwd[GPT2_DTYPE, Self.target](
            as_mut_kernel[GPT2_DTYPE](d_last_residual_2),
            as_mut_kernel[GPT2_DTYPE](d_last_fc_proj),
            as_mut_kernel[GPT2_DTYPE](self.grad_acts.ln_f),
            as_mut_kernel[GPT2_DTYPE](last_residual_3),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.params.ln_f_gamma),
            as_mut_kernel[StatsDType](self.acts.ln_f_mean),
            as_mut_kernel[StatsDType](self.acts.ln_f_rstd),
            as_mut_kernel[GPT2_DTYPE](self.grads.ln_f_gamma),
            as_mut_kernel[GPT2_DTYPE](self.grads.ln_f_beta),
            as_mut_kernel[GPT2_DTYPE](
                self.grad_acts.residual_3 + last_layer * layer_stride
            ),
            # No incoming residual gradient to seed here (ln_f receives only
            # the LM-head matmul backward's d_output); HAS_RESID_IN defaults
            # False, so this placeholder is never read.
            as_immut_kernel_from_mut[GPT2_DTYPE](NULL_DTYPE_PTR),
            Int64(batch_size),
            Int64(seq_len),
            Int64(channels),
            self.ctx,
        )

        # Backward through the transformer blocks.
        for layer in range(num_layers - 1, -1, -1):
            var layer_offset = layer * layer_stride

            var l_ln_1_gamma = self.params.ln_1_gamma + layer * channels
            var l_ln_1_beta = self.params.ln_1_beta + layer * channels
            var l_qkv_weight = (
                self.params.qkv_weight + layer * (3 * channels) * channels
            )
            var l_qkv_bias = self.params.qkv_bias + layer * (3 * channels)
            var l_attn_proj_weight = (
                self.params.attn_proj_weight + layer * channels * channels
            )
            var l_attn_proj_bias = self.params.attn_proj_bias + layer * channels
            var l_ln_2_gamma = self.params.ln_2_gamma + layer * channels
            var l_ln_2_beta = self.params.ln_2_beta + layer * channels
            var l_fc_weight = (
                self.params.fc_weight + layer * (4 * channels) * channels
            )
            var l_fc_bias = self.params.fc_bias + layer * (4 * channels)
            var l_proj_weight = self.params.proj_weight + layer * channels * (
                4 * channels
            )
            var l_proj_bias = self.params.proj_bias + layer * channels

            var d_l_ln_1_gamma = self.grads.ln_1_gamma + layer * channels
            var d_l_ln_1_beta = self.grads.ln_1_beta + layer * channels
            var d_l_qkv_weight = (
                self.grads.qkv_weight + layer * (3 * channels) * channels
            )
            var d_l_qkv_bias = self.grads.qkv_bias + layer * (3 * channels)
            var d_l_attn_proj_weight = (
                self.grads.attn_proj_weight + layer * channels * channels
            )
            var d_l_attn_proj_bias = (
                self.grads.attn_proj_bias + layer * channels
            )
            var d_l_ln_2_gamma = self.grads.ln_2_gamma + layer * channels
            var d_l_ln_2_beta = self.grads.ln_2_beta + layer * channels
            var d_l_fc_weight = (
                self.grads.fc_weight + layer * (4 * channels) * channels
            )
            var d_l_fc_bias = self.grads.fc_bias + layer * (4 * channels)
            var d_l_proj_weight = self.grads.proj_weight + layer * channels * (
                4 * channels
            )
            var d_l_proj_bias = self.grads.proj_bias + layer * channels

            var l_ln_1 = self.acts.ln_1 + layer_offset
            var l_ln_1_mean = self.acts.ln_1_mean + layer * batch_size * seq_len
            var l_ln_1_rstd = self.acts.ln_1_rstd + layer * batch_size * seq_len
            var l_qkv = self.acts.qkv + layer * qkv_layer_stride
            var l_q = self.acts.q + layer_offset
            var l_k = self.acts.k + layer_offset
            var l_v = self.acts.v + layer_offset
            var l_lse = self.acts.lse + layer * batch_size * num_heads * seq_len
            var l_attn = self.acts.attn + layer_offset
            var l_attn_merged = self.acts.attn_merged + layer_offset
            var l_attn_proj = self.acts.attn_proj + layer_offset
            var l_residual_2 = self.acts.residual_2 + layer_offset
            var l_ln_2 = self.acts.ln_2 + layer_offset
            var l_ln_2_mean = self.acts.ln_2_mean + layer * batch_size * seq_len
            var l_ln_2_rstd = self.acts.ln_2_rstd + layer * batch_size * seq_len
            # With recompute on, fch / fch_gelu (and their grads) share a single
            # scratch slot, so the per-layer offset collapses to 0; backward
            # rematerializes them below before the MLP backward kernels run.
            var fch_layer = layer
            comptime if Self.recompute:
                fch_layer = 0
            var l_fch = self.acts.fch + fch_layer * fch_layer_stride
            var l_fch_gelu = self.acts.fch_gelu + fch_layer * fch_layer_stride
            var l_fc_proj = self.acts.fc_proj + layer_offset

            var d_l_ln_1 = self.grad_acts.ln_1 + layer_offset
            var d_l_qkv = self.grad_acts.qkv + layer * qkv_layer_stride
            var d_l_q = self.grad_acts.q + layer_offset
            var d_l_k = self.grad_acts.k + layer_offset
            var d_l_v = self.grad_acts.v + layer_offset
            var d_l_attn = self.grad_acts.attn + layer_offset
            var d_l_attn_merged = self.grad_acts.attn_merged + layer_offset
            var d_l_attn_proj = self.grad_acts.attn_proj + layer_offset
            var d_l_ln_2 = self.grad_acts.ln_2 + layer_offset
            var d_l_fch = self.grad_acts.fch + fch_layer * fch_layer_stride
            var d_l_fch_gelu = (
                self.grad_acts.fch_gelu + fch_layer * fch_layer_stride
            )
            var d_l_fc_proj = self.grad_acts.fc_proj + layer_offset

            var d_block_input: MutMemPtr[GPT2_DTYPE]
            if layer == 0:
                d_block_input = self.grad_acts.encoded
            else:
                d_block_input = (
                    self.grad_acts.residual_3 + (layer - 1) * layer_stride
                )

            # Recompute checkpointing: rematerialize fch (pre-GELU) and fch_gelu
            # (post-GELU) into the scratch slot by re-running the same fused FC
            # matmul the forward used, reading the still-resident ln_2. Both MLP
            # backward kernels below then read them exactly as in the no-recompute
            # path. Bit-identical to having stored them, since the GEMM + GELU are
            # deterministic in the same inputs.
            comptime if Self.recompute:
                matmul_fwd[GPT2_DTYPE, Self.target, use_gelu=True](
                    as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                    as_mut_kernel[GPT2_DTYPE](l_fch),
                    as_mut_kernel[GPT2_DTYPE](l_ln_2),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_weight),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_bias),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    Int64(4 * channels),
                    self.ctx,
                )

            # GOTCHA (gradient flow): GELU lives at the 4C boundary between FC
            # and PROJ. Its gradient must fuse into THIS (PROJ) backward —
            # use_gelu=True here, use_gelu=False in the FC backward below. The
            # reverse assignment was wrong (C2): gelu'() ran at C-wide stride
            # instead of 4C-wide, corrupting d_ln_2. (llm.c: gelu fuses into the
            # fcproj backward, not the fc one.) See metal_port_gotchas_and_optimizations.md C2.
            #
            # MLP projection backward (fused GELU). The GELU nonlinearity sits
            # between the FC matmul (produces fch, 4C-wide pre-activation) and
            # this projection matmul (consumes fch_gelu, 4C-wide). Its gradient
            # must therefore be applied at the 4C boundary — i.e. fused into
            # THIS backward, which crosses that boundary going from d_fc_proj to
            # d(fch_gelu) and on to d(fch). With use_gelu=True and pre_gelu=l_fch,
            # matmul_d_input_bwd computes d_l_fch_gelu = gelu'(l_fch) ⊙
            # (d_fc_proj @ proj_weight) = d_fch (post-GELU-grad, still 4C-wide).
            var loop_t0 = global_perf_counter_ns()
            # fp8 site: C=4*channels, OC=channels;
            # input=l_fch_gelu, weight=l_proj_weight, d_output=d_l_fc_proj.
            comptime if FP8_BWD_ENABLED and FP8_SITE_PROJ:
                matmul_bwd_lowp[GPT2_DTYPE, Self.target, use_gelu=True](
                    as_mut_kernel[GPT2_DTYPE](d_l_fch_gelu),
                    as_mut_kernel[GPT2_DTYPE](d_l_proj_weight),
                    as_mut_kernel[GPT2_DTYPE](d_l_proj_bias),
                    as_mut_kernel[GPT2_DTYPE](d_l_fc_proj),
                    as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_weight),
                    as_mut_kernel[GPT2_DTYPE](l_fch),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(4 * channels),
                    Int64(channels),
                    self.fp8_state.proj_input[layer],
                    self.fp8_state.proj_weight[layer],
                    self.fp8_state.proj_doutput[layer],
                    "proj",
                    layer,
                    self.ctx,
                )
            elif PRECISION == "fp4":
                # fp4 site (MLP-eligible middle blocks only): same
                # C=4*channels, OC=channels, input=l_fch_gelu,
                # weight=l_proj_weight, d_output=d_l_fc_proj mapping as the
                # fp8 branch above; outside `_layer_in_fp4_range` falls
                # through to the same bf16 `matmul_bwd` the fp32/bf16/first-
                # last-block-under-fp4 builds use (matches the forward pass's
                # FC/Proj three-way dispatch).
                if _layer_in_fp4_range(layer, num_layers):
                    matmul_bwd_fp4[GPT2_DTYPE, Self.target, use_gelu=True](
                        as_mut_kernel[GPT2_DTYPE](d_l_fch_gelu),
                        as_mut_kernel[GPT2_DTYPE](d_l_proj_weight),
                        as_mut_kernel[GPT2_DTYPE](d_l_proj_bias),
                        as_mut_kernel[GPT2_DTYPE](d_l_fc_proj),
                        as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                        as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_weight),
                        as_mut_kernel[GPT2_DTYPE](l_fch),
                        Int64(batch_size),
                        Int64(seq_len),
                        Int64(4 * channels),
                        Int64(channels),
                        self.ctx,
                        sr_step=fp4_sr_step,
                    )
                else:
                    matmul_bwd[GPT2_DTYPE, Self.target, use_gelu=True](
                        as_mut_kernel[GPT2_DTYPE](d_l_fch_gelu),
                        as_mut_kernel[GPT2_DTYPE](d_l_proj_weight),
                        as_mut_kernel[GPT2_DTYPE](d_l_proj_bias),
                        as_mut_kernel[GPT2_DTYPE](d_l_fc_proj),
                        as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                        as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_weight),
                        as_mut_kernel[GPT2_DTYPE](l_fch),
                        as_mut_kernel[GPT2_DTYPE](d_l_attn),
                        Int64(batch_size),
                        Int64(seq_len),
                        Int64(4 * channels),
                        Int64(channels),
                        self.ctx,
                    )
            else:
                matmul_bwd[GPT2_DTYPE, Self.target, use_gelu=True](
                    as_mut_kernel[GPT2_DTYPE](d_l_fch_gelu),
                    as_mut_kernel[GPT2_DTYPE](d_l_proj_weight),
                    as_mut_kernel[GPT2_DTYPE](d_l_proj_bias),
                    as_mut_kernel[GPT2_DTYPE](d_l_fc_proj),
                    as_mut_kernel[GPT2_DTYPE](l_fch_gelu),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_proj_weight),
                    as_mut_kernel[GPT2_DTYPE](l_fch),
                    as_mut_kernel[GPT2_DTYPE](d_l_attn),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(4 * channels),
                    Int64(channels),
                    self.ctx,
                )

            # MLP FC backward (no GELU — d_l_fch_gelu already carries d_fch, the
            # post-GELU-grad, from the fused projection backward above). This
            # matmul lives entirely below the GELU, so it just backprops the
            # linear FC: d_l_ln_2 = d_fch @ fc_weight.
            # fp8 site: C=channels, OC=4*channels; input=l_ln_2,
            # weight=l_fc_weight, d_output=d_l_fch_gelu.
            comptime if FP8_BWD_ENABLED and FP8_SITE_FC:
                matmul_bwd_lowp[GPT2_DTYPE, Self.target, use_gelu=False](
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_2),
                    as_mut_kernel[GPT2_DTYPE](d_l_fc_weight),
                    as_mut_kernel[GPT2_DTYPE](d_l_fc_bias),
                    as_mut_kernel[GPT2_DTYPE](d_l_fch_gelu),
                    as_mut_kernel[GPT2_DTYPE](l_ln_2),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_weight),
                    as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    Int64(4 * channels),
                    self.fp8_state.fc_input[layer],
                    self.fp8_state.fc_weight[layer],
                    self.fp8_state.fc_doutput[layer],
                    "fc",
                    layer,
                    self.ctx,
                )
            elif PRECISION == "fp4":
                # fp4 site (MLP-eligible middle blocks only): same
                # C=channels, OC=4*channels, input=l_ln_2, weight=l_fc_weight,
                # d_output=d_l_fch_gelu mapping as the fp8 branch above.
                if _layer_in_fp4_range(layer, num_layers):
                    matmul_bwd_fp4[GPT2_DTYPE, Self.target, use_gelu=False](
                        as_mut_kernel[GPT2_DTYPE](d_l_ln_2),
                        as_mut_kernel[GPT2_DTYPE](d_l_fc_weight),
                        as_mut_kernel[GPT2_DTYPE](d_l_fc_bias),
                        as_mut_kernel[GPT2_DTYPE](d_l_fch_gelu),
                        as_mut_kernel[GPT2_DTYPE](l_ln_2),
                        as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_weight),
                        as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                        Int64(batch_size),
                        Int64(seq_len),
                        Int64(channels),
                        Int64(4 * channels),
                        self.ctx,
                        sr_step=fp4_sr_step,
                    )
                else:
                    matmul_bwd[GPT2_DTYPE, Self.target, use_gelu=False](
                        as_mut_kernel[GPT2_DTYPE](d_l_ln_2),
                        as_mut_kernel[GPT2_DTYPE](d_l_fc_weight),
                        as_mut_kernel[GPT2_DTYPE](d_l_fc_bias),
                        as_mut_kernel[GPT2_DTYPE](d_l_fch_gelu),
                        as_mut_kernel[GPT2_DTYPE](l_ln_2),
                        as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_weight),
                        as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                        as_mut_kernel[GPT2_DTYPE](d_l_fch),
                        Int64(batch_size),
                        Int64(seq_len),
                        Int64(channels),
                        Int64(4 * channels),
                        self.ctx,
                    )
            else:
                matmul_bwd[GPT2_DTYPE, Self.target, use_gelu=False](
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_2),
                    as_mut_kernel[GPT2_DTYPE](d_l_fc_weight),
                    as_mut_kernel[GPT2_DTYPE](d_l_fc_bias),
                    as_mut_kernel[GPT2_DTYPE](d_l_fch_gelu),
                    as_mut_kernel[GPT2_DTYPE](l_ln_2),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_fc_weight),
                    as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                    as_mut_kernel[GPT2_DTYPE](d_l_fch),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    Int64(4 * channels),
                    self.ctx,
                )

            # GOTCHA (gradient flow): the fused LN backward only adds LN_dinp to
            # its targets (`d_inp1 += LN_dinp; d_inp2 += LN_dinp`). The incoming
            # residual gradient dR must be seeded into both targets BEFORE the fused
            # kernel runs; omitting this drops the identity-skip and causes block
            # gradients to decay geometrically with depth (C1, July 2026).
            # See docs/ai/metal_port_gotchas_and_optimizations.md C1.
            #
            # HAS_RESID_IN=True fuses that seed into the SAME kernel pass as
            # the `+= LN2_dinp` accumulate (GPU: one launch instead of a
            # separate `residual_grad_broadcast` kernel before this call;
            # CPU: unchanged — see layernorm_fused_residual_bwd's CPU
            # branch). The forward op is residual_2 = block_input + attn_proj
            # (then ln_2 = LN(residual_2)); its backward must produce
            # d_block_input = d_attn_proj = LN2_dinp + d_residual_2, where the
            # incoming d_residual_2 = dR (grad flowing back through the residual
            # stream). That incoming gradient is exactly what the MLP-proj
            # backward just consumed as d_output (d_l_fc_proj), since the next
            # residual add is residual_3 = residual_2 + fc_proj. Without this
            # seed the residual identity skip is dropped and block gradients
            # decay geometrically with depth.
            layernorm_fused_residual_bwd[
                GPT2_DTYPE, Self.target, HAS_RESID_IN=True
            ](
                as_mut_kernel[GPT2_DTYPE](d_block_input),
                as_mut_kernel[GPT2_DTYPE](d_l_attn_proj),
                as_mut_kernel[GPT2_DTYPE](d_l_ln_2),
                as_mut_kernel[GPT2_DTYPE](l_residual_2),
                as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_2_gamma),
                as_mut_kernel[StatsDType](l_ln_2_mean),
                as_mut_kernel[StatsDType](l_ln_2_rstd),
                as_mut_kernel[GPT2_DTYPE](d_l_ln_2_gamma),
                as_mut_kernel[GPT2_DTYPE](d_l_ln_2_beta),
                as_mut_kernel[GPT2_DTYPE](
                    self.grad_acts.residual_2 + layer_offset
                ),
                as_immut_kernel_from_mut[GPT2_DTYPE](d_l_fc_proj),
                Int64(batch_size),
                Int64(seq_len),
                Int64(channels),
                self.ctx,
            )

            # Attention projection backward.
            # fp8 site: C=channels, OC=channels;
            # input=l_attn_merged, weight=l_attn_proj_weight,
            # d_output=d_l_attn_proj.
            comptime if FP8_BWD_ENABLED and FP8_SITE_ATTN_PROJ:
                matmul_bwd_lowp[GPT2_DTYPE, Self.target, use_gelu=False](
                    as_mut_kernel[GPT2_DTYPE](d_l_attn_merged),
                    as_mut_kernel[GPT2_DTYPE](d_l_attn_proj_weight),
                    as_mut_kernel[GPT2_DTYPE](d_l_attn_proj_bias),
                    as_mut_kernel[GPT2_DTYPE](d_l_attn_proj),
                    as_mut_kernel[GPT2_DTYPE](l_attn_merged),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_attn_proj_weight),
                    as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    Int64(channels),
                    self.fp8_state.attn_proj_input[layer],
                    self.fp8_state.attn_proj_weight[layer],
                    self.fp8_state.attn_proj_doutput[layer],
                    "attn_proj",
                    layer,
                    self.ctx,
                )
            else:
                matmul_bwd[GPT2_DTYPE, Self.target, use_gelu=False](
                    as_mut_kernel[GPT2_DTYPE](d_l_attn_merged),
                    as_mut_kernel[GPT2_DTYPE](d_l_attn_proj_weight),
                    as_mut_kernel[GPT2_DTYPE](d_l_attn_proj_bias),
                    as_mut_kernel[GPT2_DTYPE](d_l_attn_proj),
                    as_mut_kernel[GPT2_DTYPE](l_attn_merged),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_attn_proj_weight),
                    as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                    as_mut_kernel[GPT2_DTYPE](d_l_qkv),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    Int64(channels),
                    self.ctx,
                )

            # Attention backward — read the forward-stored probs P from
            # acts.att_probs[layer] (see the forward store note; recompute is
            # correct but a net perf loss with the current Metal GEMM kernels).
            self.kv_cache.att_probs_addr = Int(self.acts.att_probs)
            self.kv_cache.att_probs_layer = layer
            self.kv_cache.att_probs_stride = (
                batch_size * num_heads * seq_len * seq_len
            )
            attention_bwd[GPT2_DTYPE, Self.target, use_soft_exp=True](
                as_mut_kernel[GPT2_DTYPE](d_l_qkv),
                as_mut_kernel[GPT2_DTYPE](d_l_q),
                as_mut_kernel[GPT2_DTYPE](d_l_k),
                as_mut_kernel[GPT2_DTYPE](d_l_v),
                as_mut_kernel[GPT2_DTYPE](d_l_attn),
                as_mut_kernel[GPT2_DTYPE](d_l_attn_merged),
                as_mut_kernel[GPT2_DTYPE](l_q),
                as_mut_kernel[GPT2_DTYPE](l_k),
                as_mut_kernel[GPT2_DTYPE](l_v),
                as_mut_kernel[GPT2_DTYPE](l_attn),
                as_mut_kernel[StatsDType](l_lse),
                Int64(batch_size),
                Int64(num_heads),
                Int64(seq_len),
                Int64(head_dim),
                self.ctx,
                cache=rebind[KVCachePtr](UnsafePointer(to=self.kv_cache)),
            )

            # QKV matmul backward.
            # fp8 site: C=channels, OC=3*channels; input=l_ln_1,
            # weight=l_qkv_weight, d_output=d_l_qkv.
            comptime if FP8_BWD_ENABLED and FP8_SITE_QKV:
                matmul_bwd_lowp[GPT2_DTYPE, Self.target, use_gelu=False](
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_1),
                    as_mut_kernel[GPT2_DTYPE](d_l_qkv_weight),
                    as_mut_kernel[GPT2_DTYPE](d_l_qkv_bias),
                    as_mut_kernel[GPT2_DTYPE](d_l_qkv),
                    as_mut_kernel[GPT2_DTYPE](l_ln_1),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_qkv_weight),
                    as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    Int64(3 * channels),
                    self.fp8_state.qkv_input[layer],
                    self.fp8_state.qkv_weight[layer],
                    self.fp8_state.qkv_doutput[layer],
                    "qkv",
                    layer,
                    self.ctx,
                )
            else:
                matmul_bwd[GPT2_DTYPE, Self.target, use_gelu=False](
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_1),
                    as_mut_kernel[GPT2_DTYPE](d_l_qkv_weight),
                    as_mut_kernel[GPT2_DTYPE](d_l_qkv_bias),
                    as_mut_kernel[GPT2_DTYPE](d_l_qkv),
                    as_mut_kernel[GPT2_DTYPE](l_ln_1),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_qkv_weight),
                    as_mut_kernel[GPT2_DTYPE](NULL_DTYPE_PTR),
                    as_mut_kernel[GPT2_DTYPE](d_l_fch_gelu),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    Int64(3 * channels),
                    self.ctx,
                )

            # LayerNorm 1 backward.
            if layer == 0:
                # d_block_input (encoded) already holds the accumulated residual
                # gradient dR_0 (seed + ln_2 backward above). The plain
                # layernorm_bwd overwrites its d_x output, so route the LN1 input
                # gradient into a dead scratch (grad_acts.residual_3[0] — layer
                # 1's backward already consumed it) and then add it into encoded,
                # preserving the residual identity skip into the encoder.
                var ln1_scratch = self.grad_acts.residual_3
                # GOTCHA (gradient flow): layernorm_bwd reconstructs
                # xhat = (input - mean)/rstd from the PRE-NORM INPUT, not the
                # normed output. For layer 0, that input is acts.encoded (the
                # encoder output), not l_ln_1 (the LN output). Passing l_ln_1
                # corrupts xhat and hence dgamma; dbeta is unaffected (it only
                # depends on d_out, not xhat). Bug class: C3 (July 2026).
                # See docs/ai/metal_port_gotchas_and_optimizations.md C3.
                # LN1's forward INPUT for layer 0 is the encoder output
                # (acts.encoded); layernorm_bwd needs that pre-norm input to
                # form xhat = (input - mean)*rstd, NOT l_ln_1 (the normed
                # OUTPUT). Passing the output corrupts xhat and hence the
                # gamma gradient (dbeta, which ignores xhat, stays correct).
                layernorm_bwd[GPT2_DTYPE, Self.target](
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_1),
                    as_mut_kernel[GPT2_DTYPE](self.acts.encoded),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_1_gamma),
                    as_mut_kernel[StatsDType](l_ln_1_mean),
                    as_mut_kernel[StatsDType](l_ln_1_rstd),
                    as_mut_kernel[GPT2_DTYPE](ln1_scratch),
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_1_gamma),
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_1_beta),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    self.ctx,
                )
                # encoded += LN1_dinp (the second broadcast target, attn_proj[0],
                # is already consumed and dead, so the extra add is harmless).
                residual_grad_broadcast[GPT2_DTYPE, Self.target](
                    as_mut_kernel[GPT2_DTYPE](d_block_input),
                    as_mut_kernel[GPT2_DTYPE](d_l_attn_proj),
                    as_immut_kernel_from_mut[GPT2_DTYPE](ln1_scratch),
                    batch_size * seq_len * channels,
                    self.ctx,
                )
            else:
                var prev_layer_offset = (layer - 1) * layer_stride
                # Seed the incoming residual gradient d_R_L (accumulated into
                # d_block_input by the ln_2 residual backward above) into the
                # ln_1 residual-backward targets residual_2[L-1] / fc_proj[L-1]
                # — the two forward inputs whose sum is this layer's block input
                # (residual_3[L-1] = residual_2[L-1] + fc_proj[L-1]). The fused
                # `+= LN1_dinp` then yields the full d_R_L split that becomes
                # layer L-1's incoming residual gradient. HAS_RESID_IN=True
                # fuses this seed into the same kernel pass as the LN1
                # input-gradient accumulate (see the ln_2 call site above for
                # the full rationale).
                layernorm_fused_residual_bwd[
                    GPT2_DTYPE, Self.target, HAS_RESID_IN=True
                ](
                    as_mut_kernel[GPT2_DTYPE](
                        self.grad_acts.residual_2 + prev_layer_offset
                    ),
                    as_mut_kernel[GPT2_DTYPE](
                        self.grad_acts.fc_proj + prev_layer_offset
                    ),
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_1),
                    as_mut_kernel[GPT2_DTYPE](
                        self.acts.residual_3 + prev_layer_offset
                    ),
                    as_immut_kernel_from_mut[GPT2_DTYPE](l_ln_1_gamma),
                    as_mut_kernel[StatsDType](l_ln_1_mean),
                    as_mut_kernel[StatsDType](l_ln_1_rstd),
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_1_gamma),
                    as_mut_kernel[GPT2_DTYPE](d_l_ln_1_beta),
                    as_mut_kernel[GPT2_DTYPE](
                        self.grad_acts.residual_3 + prev_layer_offset
                    ),
                    as_immut_kernel_from_mut[GPT2_DTYPE](d_block_input),
                    Int64(batch_size),
                    Int64(seq_len),
                    Int64(channels),
                    self.ctx,
                )

        # Encoder backward: scatter token grads into wte, sum position grads into wpe.
        # On GPU (Metal), bucket_info/workload_indices are HostBuffers whose raw
        # pointers read as zeros inside Metal kernels — use device copies uploaded
        # during forward's build_encoder_buckets phase.
        var enc_bucket_info = self.bucket_info
        var enc_workload_indices = self.workload_indices
        comptime if is_gpu[Self.target]():
            enc_bucket_info = self.bucket_info_dev
            enc_workload_indices = self.workload_indices_dev
        encoder_bwd[GPT2_DTYPE, Self.target](
            as_mut_kernel[GPT2_DTYPE](self.grads.wte),
            as_mut_kernel[GPT2_DTYPE](self.grads.wpe),
            as_immut_kernel_from_mut[DType.int32](enc_bucket_info),
            as_immut_kernel_from_mut[DType.int32](enc_workload_indices),
            as_immut_kernel_from_mut[GPT2_DTYPE](self.grad_acts.encoded),
            self.num_wte_buckets,
            batch_size,
            seq_len,
            channels,
            self.ctx,
        )

        # Wait for all GPU backward kernels to finish. Metal's enqueue_copy
        # (device→host) requires the source buffer to be idle; without this
        # synchronize the runtime raises "Invalid Metal buffer pointer" when
        # callers try to read gradient values immediately after backward().
        comptime if is_gpu[Self.target]():
            self.ctx.synchronize()

        comptime if Self.WORLD_SIZE > 1:
            # ZeRO-0 (DDP) and ZeRO-1 use allreduce: full gradient sum is replicated
            # to all ranks (2N communication). ZeRO-1 then reads grads_memory + rank*opt
            # in the optimizer step.
            # ZeRO-2/3 use reduce-scatter: each rank receives only its gradient shard
            # (N communication), which is stored in sharded_grads_memory.
            if self.zero_ctx.zero_stage <= 1:
                self.zero_ctx.allreduce(
                    rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                        self.grads_memory.as_unsafe_any_origin()
                    ),
                    self.num_parameters,
                )
            else:
                self.zero_ctx.reducescatter(
                    rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                        self.grads_memory.as_unsafe_any_origin()
                    ),
                    rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                        self.sharded_grads_memory.as_unsafe_any_origin()
                    ),
                    self.optimizer_num_parameters,
                )
            self.ctx.synchronize()

    def update(
        mut self,
        t: UInt32,
        learning_rate: Scalar[DType.float32],
        beta1: Scalar[DType.float32] = Scalar[DType.float32](0.9),
        beta2: Scalar[DType.float32] = Scalar[DType.float32](0.999),
        eps: Scalar[DType.float32] = Scalar[DType.float32](1e-8),
        weight_decay: Scalar[DType.float32] = Scalar[DType.float32](0.0),
        grad_scale: Scalar[DType.float32] = Scalar[DType.float32](1.0),
    ) raises:
        var scaled_grad_scale = grad_scale
        comptime if Self.WORLD_SIZE > 1:
            scaled_grad_scale = grad_scale / Scalar[DType.float32](
                Self.WORLD_SIZE
            )

        var config = AdamWConfig(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            grad_scale=scaled_grad_scale,
        )

        # The optimizer reads/writes the fp32 master copy when USE_BF16; in pure
        # fp32 `master_memory` is null and `has_master=USE_BF16` is False, so the
        # params are their own master (see llmm.adamw).

        comptime if is_cpu[Self.target]():
            var p_ptr: MutMemPtr[GPT2_DTYPE] = self.params_memory
            var g_ptr: MutMemPtr[GPT2_DTYPE] = self.grads_memory
            var update_num_params = self.num_parameters

            comptime if Self.WORLD_SIZE > 1:
                if self.zero_ctx.zero_stage >= 1:
                    var offset = (
                        self.zero_ctx.rank * self.optimizer_num_parameters
                    )
                    var local_num_params = min(
                        self.num_parameters - offset,
                        self.optimizer_num_parameters,
                    )
                    p_ptr = self.params_memory + offset
                    # ZeRO-1: grads were allreduced — each rank reads its shard directly
                    # from the replicated grads_memory (no sharded_grads_buf allocated).
                    # ZeRO-2/3: grads were reduce-scattered into sharded_grads_memory.
                    if self.zero_ctx.zero_stage == 1:
                        g_ptr = self.grads_memory + offset
                    else:
                        g_ptr = self.sharded_grads_memory
                    update_num_params = local_num_params

            if update_num_params > 0:
                comptime simd_w = simd_width_of[GPT2_DTYPE]()
                adamw_update[GPT2_DTYPE, Self.target, width=simd_w](
                    update_num_params,
                    p_ptr.as_unsafe_any_origin(),
                    g_ptr.as_unsafe_any_origin(),
                    self.master_memory.as_unsafe_any_origin(),
                    USE_BF16,
                    self.m_memory.as_unsafe_any_origin(),
                    self.v_memory.as_unsafe_any_origin(),
                    t,
                    config,
                    self.ctx,
                )

            self.ctx.synchronize()

            comptime if Self.WORLD_SIZE > 1:
                # ZeRO-3 does NOT allgather after the optimizer step — the next
                # forward() call gathers param shards just-in-time.
                # ZeRO-1 and ZeRO-2 allgather here so params_memory is consistent
                # before the next forward (they do not pre-gather in forward).
                if (
                    self.zero_ctx.zero_stage >= 1
                    and self.zero_ctx.zero_stage < 3
                ):
                    self.zero_ctx.allgather(
                        rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                            self.params_memory.as_unsafe_any_origin()
                        ),
                        self.optimizer_num_parameters,
                    )
                    self.ctx.synchronize()
        else:
            comptime if Self.WORLD_SIZE > 1:
                if self.zero_ctx.zero_stage == 0:
                    # ZeRO-0 (DDP): every rank updates all params using full grads.
                    adamw_update[GPT2_DTYPE, Self.target](
                        self.num_parameters,
                        as_mut_kernel[GPT2_DTYPE](self.params_memory),
                        as_mut_kernel[GPT2_DTYPE](self.grads_memory),
                        as_mut_kernel[MASTER_DTYPE](self.master_memory),
                        USE_BF16,
                        as_mut_kernel[MASTER_DTYPE](self.m_memory),
                        as_mut_kernel[MASTER_DTYPE](self.v_memory),
                        t,
                        config,
                        self.ctx,
                    )
                elif self.zero_ctx.zero_stage == 1:
                    # ZeRO-1: grads were allreduced — read shard from grads_memory + offset.
                    var offset = (
                        self.zero_ctx.rank * self.optimizer_num_parameters
                    )
                    var local_num_params = min(
                        self.num_parameters - offset,
                        self.optimizer_num_parameters,
                    )
                    if local_num_params > 0:
                        adamw_update[GPT2_DTYPE, Self.target](
                            local_num_params,
                            as_mut_kernel[GPT2_DTYPE](
                                self.params_memory + offset
                            ),
                            as_mut_kernel[GPT2_DTYPE](
                                self.grads_memory + offset
                            ),
                            as_mut_kernel[MASTER_DTYPE](self.master_memory),
                            USE_BF16,
                            as_mut_kernel[MASTER_DTYPE](self.m_memory),
                            as_mut_kernel[MASTER_DTYPE](self.v_memory),
                            t,
                            config,
                            self.ctx,
                        )
                    self.ctx.synchronize()
                    # Allgather so params_memory is consistent before the next forward.
                    self.zero_ctx.allgather(
                        rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                            self.params_memory.as_unsafe_any_origin()
                        ),
                        self.optimizer_num_parameters,
                    )
                    self.ctx.synchronize()
                else:
                    # ZeRO 2/3: grads were reduce-scattered into sharded_grads_memory.
                    var offset = (
                        self.zero_ctx.rank * self.optimizer_num_parameters
                    )
                    var local_num_params = min(
                        self.num_parameters - offset,
                        self.optimizer_num_parameters,
                    )
                    if local_num_params > 0:
                        adamw_update[GPT2_DTYPE, Self.target](
                            local_num_params,
                            as_mut_kernel[GPT2_DTYPE](
                                self.params_memory + offset
                            ),
                            as_mut_kernel[GPT2_DTYPE](
                                self.sharded_grads_memory
                            ),
                            as_mut_kernel[MASTER_DTYPE](self.master_memory),
                            USE_BF16,
                            as_mut_kernel[MASTER_DTYPE](self.m_memory),
                            as_mut_kernel[MASTER_DTYPE](self.v_memory),
                            t,
                            config,
                            self.ctx,
                        )
                    self.ctx.synchronize()
                    # ZeRO-2 allgathers here so params_memory is immediately consistent.
                    if self.zero_ctx.zero_stage == 2:
                        self.zero_ctx.allgather(
                            rebind[
                                UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]
                            ](self.params_memory.as_unsafe_any_origin()),
                            self.optimizer_num_parameters,
                        )
                        self.ctx.synchronize()
            else:
                adamw_update[GPT2_DTYPE, Self.target](
                    self.num_parameters,
                    as_mut_kernel[GPT2_DTYPE](self.params_memory),
                    as_mut_kernel[GPT2_DTYPE](self.grads_memory),
                    as_mut_kernel[MASTER_DTYPE](self.master_memory),
                    USE_BF16,
                    as_mut_kernel[MASTER_DTYPE](self.m_memory),
                    as_mut_kernel[MASTER_DTYPE](self.v_memory),
                    t,
                    config,
                    self.ctx,
                )

    def calculate_grad_norm(mut self) raises -> Float32:
        """Global L2 norm of the parameter gradients (sqrt of the sum of squares
        over all `num_parameters` grads), used for gradient clipping and the
        grad-norm z-score. Reuses the llmm.global_norm kernels.

        NOTE: this reduces over the full replicated grads_memory; it does not yet
        all-reduce the norm across ranks in the multi-GPU sharded case.
        """
        var n = self.num_parameters
        comptime width = simd_width_of[GPT2_DTYPE]()

        comptime if is_cpu[Self.target]():
            var host_out = alloc[Scalar[DType.float32]](1)
            host_out[0] = Scalar[DType.float32](0.0)
            global_norm_squared_cpu[GPT2_DTYPE, width](
                rebind[MutKernelPtr[DType.float32]](
                    host_out.as_unsafe_any_origin()
                ),
                as_immut_kernel_from_mut[GPT2_DTYPE](self.grads_memory),
                n,
            )
            var sumsq = host_out[0]
            host_out.free()
            return sqrt(sumsq)
        else:
            comptime BLOCK_SIZE = 512
            var grid_x = self.grad_norm_grid_x
            # Per-block partial sums land in out_buf[0:grid_x]; the aggregate
            # kernel reduces them into out_buf[0]. out_buf/host_out are
            # persistent (grad_norm_out_buf/grad_norm_host_buf, sized once in
            # allocate_optimizer_moments) rather than reallocated every call —
            # this runs once per training step, so a fresh
            # enqueue_create_buffer/enqueue_create_host_buffer here would
            # otherwise reallocate device/host memory on every step.
            self.grad_norm_out_buf.enqueue_fill(Scalar[DType.float32](0.0))
            # Metal: unsafe_ptr() is a CPU-side query (pointer value, not
            # content); the fill and subsequent norm kernel are sequenced by
            # the in-order queue — no sync needed here.
            comptime if not HAS_METAL:
                self.ctx.synchronize()
            var out_ptr = rebind_mut_mem[DType.float32](
                self.grad_norm_out_buf.unsafe_ptr().as_unsafe_any_origin()
            )

            comptime norm_kernel = global_norm_squared_gpu[
                GPT2_DTYPE, BLOCK_SIZE, width
            ]
            var compiled_norm = self.ctx.compile_function[norm_kernel]()
            self.ctx.enqueue_function(
                compiled_norm,
                as_mut_kernel[DType.float32](out_ptr),
                as_immut_kernel_from_mut[GPT2_DTYPE](self.grads_memory),
                n,  # count (single slice)
                n,  # stride (unused with one slice)
                grid_dim=(grid_x, 1),
                block_dim=(BLOCK_SIZE,),
            )

            comptime agg_kernel = global_norm_aggregate_gpu[BLOCK_SIZE]
            var compiled_agg = self.ctx.compile_function[agg_kernel]()
            self.ctx.enqueue_function(
                compiled_agg,
                as_mut_kernel[DType.float32](out_ptr),
                grid_x,
                grid_dim=(1,),
                block_dim=(BLOCK_SIZE,),
            )
            # Metal: aggregate kernel → enqueue_copy (device→host) sequenced
            # by in-order queue; no sync needed before creating the host buffer.
            comptime if not HAS_METAL:
                self.ctx.synchronize()

            comptime if not HAS_METAL:
                self.ctx.synchronize()
            self.ctx.enqueue_copy(
                dst_ptr=rebind[
                    UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
                ](self.grad_norm_host_buf.unsafe_ptr().as_unsafe_any_origin()),
                src_ptr=rebind[
                    UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]
                ](out_ptr.as_unsafe_any_origin()),
                size=1,
            )
            self.ctx.synchronize()
            return sqrt(self.grad_norm_host_buf[0])

    def _checkpoint_config(self) -> CheckpointConfig:
        return CheckpointConfig(
            max_seq_len=self.config.max_seq_len,
            vocab_size=self.config.vocab_size,
            num_layer=self.config.num_layer,
            num_heads=self.config.num_heads,
            channels=self.config.channels,
            padded_vocab_size=self.config.padded_vocab_size,
        )

    def _copy_device_to_host[
        dtype: DType
    ](mut self, host: HostBuffer[dtype], src: MutMemPtr[dtype], n: Int,) raises:
        self.ctx.enqueue_copy(
            dst_ptr=rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                host.unsafe_ptr().as_unsafe_any_origin()
            ),
            src_ptr=rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
                src.as_unsafe_any_origin()
            ),
            size=n,
        )

    def _copy_host_to_device[
        dtype: DType
    ](mut self, dst: MutMemPtr[dtype], host: HostBuffer[dtype], n: Int,) raises:
        self.ctx.enqueue_copy(
            dst_ptr=rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                dst.as_unsafe_any_origin()
            ),
            src_ptr=rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
                host.unsafe_ptr().as_unsafe_any_origin()
            ),
            size=n,
        )

    def write_checkpoint(
        mut self,
        model_path: String,
        state_path: String,
        step: Int,
        loader: DataLoader,
        sampler_rng_state: UInt64,
    ) raises:
        """Write a model + optimizer-state checkpoint to disk.

        Rank 0 writes the full model file; every rank writes its own optimizer
        shard (m, v) to its `state_path`. Mirrors llm.c's split of model_*.bin
        (shared) and state_*_rank.bin (per-process).
        """
        # ZeRO-3 keeps only this rank's parameter shard resident in
        # params_memory after update() (the next forward re-gathers it). Gather
        # the full parameter set before snapshotting the model.
        comptime if Self.WORLD_SIZE > 1:
            if self.zero_ctx.zero_stage >= 3:
                self.zero_ctx.allgather(
                    rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                        self.params_memory.as_unsafe_any_origin()
                    ),
                    self.optimizer_num_parameters,
                )
                self.ctx.synchronize()

        if self.zero_ctx.rank == 0:
            var host_params = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
                self.num_parameters
            )
            self.ctx.synchronize()
            self._copy_device_to_host[GPT2_DTYPE](
                host_params, self.params_memory, self.num_parameters
            )
            self.ctx.synchronize()
            write_model_checkpoint(
                model_path,
                self._checkpoint_config(),
                rebind_mut_mem[GPT2_DTYPE](
                    host_params.unsafe_ptr().as_unsafe_any_origin()
                ),
                self.num_parameters,
            )

        # AdamW moments are fp32 (MASTER_DTYPE), independent of the parameter
        # precision, so the state file stores them in fp32 (matching llm.c).
        var n_opt = self.optimizer_num_parameters
        var host_m = self.ctx.enqueue_create_host_buffer[MASTER_DTYPE](n_opt)
        var host_v = self.ctx.enqueue_create_host_buffer[MASTER_DTYPE](n_opt)
        self.ctx.synchronize()
        self._copy_device_to_host[MASTER_DTYPE](host_m, self.m_memory, n_opt)
        self._copy_device_to_host[MASTER_DTYPE](host_v, self.v_memory, n_opt)
        self.ctx.synchronize()
        write_state_checkpoint[MASTER_DTYPE](
            state_path,
            make_training_state(loader, step, sampler_rng_state),
            rebind_mut_mem[MASTER_DTYPE](
                host_m.unsafe_ptr().as_unsafe_any_origin()
            ),
            rebind_mut_mem[MASTER_DTYPE](
                host_v.unsafe_ptr().as_unsafe_any_origin()
            ),
            n_opt,
        )

    def load_checkpoint(
        mut self, model_path: String, state_path: String
    ) raises -> TrainingState:
        """Restore params and optimizer moments from a checkpoint into device
        memory; return the TrainingState (step, rng, dataloader position).

        The model must already be allocated with the same config that wrote the
        checkpoint (build it from the same base checkpoint first). Each rank
        reads the shared model file plus its own optimizer-state shard.
        """
        var host_params = self.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
            self.num_parameters
        )
        self.ctx.synchronize()
        _ = read_model_checkpoint(
            model_path,
            rebind_mut_mem[GPT2_DTYPE](
                host_params.unsafe_ptr().as_unsafe_any_origin()
            ),
            self.num_parameters,
        )
        self._copy_host_to_device[GPT2_DTYPE](
            self.params_memory, host_params, self.num_parameters
        )
        self.ctx.synchronize()

        # Mixed precision: the fp32 master copy was seeded from the *initial*
        # params in allocate_optimizer_moments, which runs before this resume
        # load. Re-seed it from the just-loaded params — otherwise the first
        # post-resume update() writes bf16(stale_initial_master - delta) back
        # into params, discarding all trained progress (loss returns to ~11.0
        # for a from-scratch d12 run). The state file has no master copy, so
        # promoting the loaded low-precision params is the best restoration
        # (matches llm.c's fallback when master weights are absent).
        comptime if USE_BF16:
            var host_master = self.ctx.enqueue_create_host_buffer[MASTER_DTYPE](
                self.num_parameters
            )
            self.ctx.synchronize()
            for i in range(self.num_parameters):
                host_master[i] = host_params[i].cast[MASTER_DTYPE]()
            self._copy_host_to_device[MASTER_DTYPE](
                self.master_memory, host_master, self.num_parameters
            )
            self.ctx.synchronize()

        # AdamW moments are fp32 (MASTER_DTYPE); read them back in fp32.
        var n_opt = self.optimizer_num_parameters
        var host_m = self.ctx.enqueue_create_host_buffer[MASTER_DTYPE](n_opt)
        var host_v = self.ctx.enqueue_create_host_buffer[MASTER_DTYPE](n_opt)
        self.ctx.synchronize()
        var state = read_state_checkpoint[MASTER_DTYPE](
            state_path,
            rebind_mut_mem[MASTER_DTYPE](
                host_m.unsafe_ptr().as_unsafe_any_origin()
            ),
            rebind_mut_mem[MASTER_DTYPE](
                host_v.unsafe_ptr().as_unsafe_any_origin()
            ),
            n_opt,
        )
        self._copy_host_to_device[MASTER_DTYPE](self.m_memory, host_m, n_opt)
        self._copy_host_to_device[MASTER_DTYPE](self.v_memory, host_v, n_opt)
        self.ctx.synchronize()
        return state^


# ===----------------------------------------------------------------------=== #
# The Main Training Loop!
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct TrainArgs(Copyable, Movable):
    """Parsed command-line configuration, mirroring train_gpt2.cu's main()."""

    var train_data_pattern: String  # -i  train tokens (glob ok)
    var val_data_pattern: String  # -j  val tokens (glob ok)
    var load_filename: String  # -e  .bin checkpoint OR model descriptor
    var output_log_dir: String  # -o  checkpoint dir ("" = no logging)
    var checkpoint_every: Int  # -n  write a checkpoint every N steps
    var resume: Int  # -y  resume from latest checkpoint in -o
    var batch_size: Int  # -b  (micro) batch size B
    var seq_len: Int  # -t  sequence length T
    var total_batch_size: Int  # -d  desired total batch (-1 => B*T*procs)
    var learning_rate: Float32  # -l
    var lr_scheduler_type: String  # -k  cosine|linear|constant|wsd
    var warmup_iterations: Int  # -u
    var final_learning_rate_frac: Float32  # -q
    var weight_decay: Float32  # -c
    var val_loss_every: Int  # -v
    var val_max_steps: Int  # -m
    var sample_every: Int  # -s
    var gen_tokens: Int  # -g  genT
    var overfit_single_batch: Int  # -a
    var max_steps: Int  # -x  (-1 => run one epoch)
    var zero_stage: Int  # -z


def default_train_args() -> TrainArgs:
    """Default arguments, matching llm.c. The only deviation is `load_filename`,
    which defaults to the fp32 checkpoint `gpt2_124M.bin` (llm.c defaults to the
    bf16 file); build with -D LLMM_BF16=1 and pass `-e gpt2_124M_bf16.bin` for
    bf16.
    """
    return TrainArgs(
        train_data_pattern="./data/.tinyshakespeare/tiny_shakespeare_train.bin",
        val_data_pattern="./data/.tinyshakespeare/tiny_shakespeare_val.bin",
        load_filename="gpt2_124M.bin",
        output_log_dir="",
        checkpoint_every=0,
        resume=0,
        batch_size=4,
        seq_len=1024,
        total_batch_size=-1,
        learning_rate=Float32(3e-4),
        lr_scheduler_type="cosine",
        warmup_iterations=0,
        final_learning_rate_frac=Float32(1.0),
        weight_decay=Float32(0.0),
        val_loss_every=20,
        val_max_steps=20,
        sample_every=20,
        gen_tokens=64,
        overfit_single_batch=0,
        max_steps=-1,
        zero_stage=0,
    )


def _build_lr_scheduler(
    args: TrainArgs, train_num_batches: Int
) raises -> LearningRateScheduler:
    """Construct the LR scheduler. scheduler_type is a StaticString, so we
    dispatch the runtime string to the matching compile-time literal."""
    var k = args.lr_scheduler_type
    if k == "cosine":
        return LearningRateScheduler(
            "cosine",
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_iterations,
            train_num_batches=train_num_batches,
            final_learning_rate_fraction=args.final_learning_rate_frac,
        )
    elif k == "linear":
        return LearningRateScheduler(
            "linear",
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_iterations,
            train_num_batches=train_num_batches,
            final_learning_rate_fraction=args.final_learning_rate_frac,
        )
    elif k == "constant":
        return LearningRateScheduler(
            "constant",
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_iterations,
            train_num_batches=train_num_batches,
            final_learning_rate_fraction=args.final_learning_rate_frac,
        )
    elif k == "wsd":
        return LearningRateScheduler(
            "wsd",
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_iterations,
            train_num_batches=train_num_batches,
            final_learning_rate_fraction=args.final_learning_rate_frac,
        )
    raise Error("Invalid scheduler type: " + k)


def _find_max_step(output_log_dir: String) raises -> Int:
    """Return the highest step N for which `model_N.bin` exists in the dir, or
    -1 if none is found. Used by `-y 1` to resume the latest checkpoint."""
    var os = Python.import_module("os")
    if not Bool(os.path.isdir(output_log_dir)):
        return -1
    var max_step = -1
    var entries = os.listdir(output_log_dir)
    for entry in entries:
        var name = String(entry)
        if name.startswith("model_") and name.endswith(".bin"):
            var step = atol(name[byte = 6 : name.byte_length() - 4])
            if step > max_step:
                max_step = step
    return max_step


def _table_row(label: String, value: String) -> String:
    # "| <label padded to 21> | <value padded to 50> |", like llm.c's printf0.
    var lhs = label
    while lhs.byte_length() < 21:
        lhs += " "
    var rhs = value
    while rhs.byte_length() < 50:
        rhs += " "
    return "| " + lhs + " | " + rhs + " |"


def train[
    target: StaticString,
    WORLD_SIZE: Int = 1,
](
    args: TrainArgs,
    rank: Int = 0,
    cpu_coord: Optional[
        UnsafePointer[CpuCoordinator, MutUntrackedOrigin]
    ] = Optional[UnsafePointer[CpuCoordinator, MutUntrackedOrigin]](),
) raises:
    var ctx: DeviceContext
    comptime if is_cpu[target]():
        ctx = DeviceContext(api="cpu")
    else:
        ctx = DeviceContext()

    var zero_stage = args.zero_stage

    # If we are only overfitting a single batch for debugging, overfit the first
    # batch of val (it is smaller and faster), exactly like llm.c / train_gpt2.py.
    var train_data_pattern = args.train_data_pattern
    if args.overfit_single_batch == 1:
        train_data_pattern = args.val_data_pattern

    var B = args.batch_size
    var T = args.seq_len
    var tokens_per_fwdbwd = B * T * WORLD_SIZE
    var total_batch_size = args.total_batch_size
    if total_batch_size == -1:
        total_batch_size = tokens_per_fwdbwd
    if total_batch_size % tokens_per_fwdbwd != 0:
        raise Error(
            "total batch size ("
            + String(total_batch_size)
            + ") must be divisible by B*T*num_processes ("
            + String(tokens_per_fwdbwd)
            + ")"
        )
    var grad_accum_steps = total_batch_size // tokens_per_fwdbwd

    var bar = "+-----------------------+----------------------------------------------------+"
    printf0(rank, bar)
    printf0(rank, _table_row("Parameter", "Value"))
    printf0(rank, bar)
    printf0(rank, _table_row("train data pattern", train_data_pattern))
    printf0(rank, _table_row("val data pattern", args.val_data_pattern))
    printf0(
        rank,
        _table_row(
            "output log dir",
            "NULL" if args.output_log_dir == "" else args.output_log_dir,
        ),
    )
    printf0(rank, _table_row("checkpoint_every", String(args.checkpoint_every)))
    printf0(rank, _table_row("resume", String(args.resume)))
    printf0(rank, _table_row("micro batch size B", String(B)))
    printf0(rank, _table_row("sequence length T", String(T)))
    printf0(rank, _table_row("total batch size", String(total_batch_size)))
    printf0(rank, _table_row("LR scheduler", args.lr_scheduler_type))
    printf0(rank, _table_row("learning rate (LR)", String(args.learning_rate)))
    printf0(
        rank, _table_row("warmup iterations", String(args.warmup_iterations))
    )
    printf0(
        rank,
        _table_row("final LR fraction", String(args.final_learning_rate_frac)),
    )
    printf0(rank, _table_row("weight decay", String(args.weight_decay)))
    printf0(rank, _table_row("max_steps", String(args.max_steps)))
    printf0(rank, _table_row("val_loss_every", String(args.val_loss_every)))
    printf0(rank, _table_row("val_max_steps", String(args.val_max_steps)))
    printf0(rank, _table_row("sample_every", String(args.sample_every)))
    printf0(rank, _table_row("genT", String(args.gen_tokens)))
    printf0(
        rank,
        _table_row("overfit_single_batch", String(args.overfit_single_batch)),
    )
    printf0(rank, bar)

    # Build the GPT-2 model from a checkpoint (.bin) or a descriptor (from
    # scratch). See GPT2.__init__ / parse_model_descriptor.
    var model = GPT2[target, WORLD_SIZE](
        args.load_filename,
        rank,
        zero_stage,
        ctx,
        cpu_coordinator_ptr=cpu_coord,
    )

    printf0(rank, bar)
    printf0(rank, _table_row("weight init method", args.load_filename))
    printf0(
        rank,
        _table_row("max_sequence_length T", String(model.config.max_seq_len)),
    )
    printf0(rank, _table_row("vocab_size V", String(model.config.vocab_size)))
    printf0(
        rank,
        _table_row(
            "padded_vocab_size Vp", String(model.config.padded_vocab_size)
        ),
    )
    printf0(rank, _table_row("num_layers L", String(model.config.num_layer)))
    printf0(rank, _table_row("num_heads NH", String(model.config.num_heads)))
    printf0(rank, _table_row("channels C", String(model.config.channels)))
    printf0(rank, _table_row("num_parameters", String(model.num_parameters)))
    printf0(rank, bar)

    # Disk checkpointing: -o output_log_dir, -n checkpoint_every, -y resume.
    var output_dir = args.output_log_dir
    var save_every = args.checkpoint_every
    if save_every > 0 and output_dir != "" and rank == 0:
        var os = Python.import_module("os")
        _ = os.makedirs(output_dir, exist_ok=True)

    # Build the dataloaders. The data patterns may be a single file or a glob.
    var train_tokens = train_data_pattern
    var val_tokens = args.val_data_pattern

    var train_loader = DataLoader(train_tokens, B, T, rank, WORLD_SIZE)
    printf0(rank, "Loaded train tokens from " + train_tokens)
    printf0(rank, "Number of tokens: " + String(train_loader.num_tokens))
    var val_loader = DataLoader(val_tokens, B, T, rank, WORLD_SIZE)
    printf0(rank, "Loaded val tokens from " + val_tokens)
    printf0(rank, "Number of tokens: " + String(val_loader.num_tokens))

    # Build the tokenizer.
    var tokenizer = Tokenizer("gpt2_tokenizer.bin")
    printf0(rank, "Loaded tokenizer from gpt2_tokenizer.bin")

    # Number of training/validation batches. With -x set we run exactly that
    # many steps; otherwise we run a single epoch over the training tokens.
    # One step consumes total_batch_size tokens (grad accumulation included),
    # matching llm.c — dividing by tokens_per_fwdbwd would count micro-batches
    # and inflate the epoch (and the cosine-decay horizon) by grad_accum_steps.
    var train_num_batches = args.max_steps
    if train_num_batches <= 0:
        train_num_batches = train_loader.num_tokens // total_batch_size
    var val_num_batches = args.val_max_steps
    if val_loader.num_tokens // tokens_per_fwdbwd < val_num_batches:
        val_num_batches = val_loader.num_tokens // tokens_per_fwdbwd

    printf0(
        rank,
        "batch_size B="
        + String(B)
        + " * seq_len T="
        + String(T)
        + " * num_processes="
        + String(WORLD_SIZE)
        + " and total_batch_size="
        + String(total_batch_size),
    )
    printf0(
        rank,
        "=> setting grad_accum_steps=" + String(grad_accum_steps),
    )
    printf0(rank, bar)
    printf0(rank, _table_row("train_num_batches", String(train_num_batches)))
    printf0(rank, _table_row("val_num_batches", String(val_num_batches)))
    printf0(rank, _table_row("num_processes", String(WORLD_SIZE)))
    printf0(rank, _table_row("zero_stage", String(zero_stage)))
    printf0(rank, bar)

    # Build the learning rate scheduler.
    var learning_rate_scheduler = _build_lr_scheduler(args, train_num_batches)

    # Initialize some memory for generating samples from the model.
    var rng_state = UInt64(1337)
    var gen_max_length = args.gen_tokens
    var gen_tokens = alloc[Scalar[DType.int32]](gen_max_length)
    var zero = 0
    var null_int32_ptr = MutMemPtr[DType.int32](unsafe_from_address=zero)

    # Optionally resume params, optimizer moments, RNG and dataloader position
    # from the latest checkpoint in the output dir (mirrors llm.c's -y 1).
    var start_step = 0
    if args.resume == 1 and output_dir != "":
        var resume_from = _find_max_step(output_dir)
        if resume_from >= 0:
            var model_path = (
                output_dir + "/model_" + String(resume_from) + ".bin"
            )
            var state_path = (
                output_dir
                + "/state_"
                + String(resume_from)
                + "_"
                + String(rank)
                + ".bin"
            )
            var resumed = model.load_checkpoint(model_path, state_path)
            restore_dataloader_state(train_loader, resumed)
            rng_state = resumed.sampler_rng_state
            start_step = resumed.step
            printf0(
                rank, "Resumed from checkpoint at step " + String(start_step)
            )

    #####################################################################################
    # Training Loop
    #####################################################################################

    var tokens_per_step = WORLD_SIZE * B * T * grad_accum_steps
    var total_sum_iteration_time_s = 0.0
    var ema_tokens_per_second = 0.0
    # Sliding-window z-score detectors for the loss and gradient norm.
    var loss_detector = OutlierDetector()
    var grad_norm_detector = OutlierDetector()
    # Device name drives the MFU peak-FLOPs lookup; CPU has no entry => "n/a".
    var device_name = String("cpu")
    comptime if is_gpu[target]():
        device_name = ctx.name()

    for step in range(start_step, train_num_batches + 1):
        var last_step = step == train_num_batches

        # Once in a while estimate the validation loss.
        if args.val_loss_every > 0 and (
            step % args.val_loss_every == 0 or last_step
        ):
            var val_loss = Float32(0.0)
            val_loader.reset()
            for _ in range(val_num_batches):
                val_loader.next_batch()
                model.forward(val_loader.inputs, val_loader.targets, B, T)
                val_loss += model.mean_loss
            if val_num_batches > 0:
                val_loss /= Float32(val_num_batches)
            printf0(rank, "val loss " + String(val_loss))

        # Once in a while do model inference to print generated text (rank 0).
        if (
            rank == 0
            and args.sample_every > 0
            and (step > 0 and step % args.sample_every == 0 or last_step)
        ):
            gen_tokens[0] = Scalar[DType.int32](
                tokenizer.eot_token
            )  # The GPT-2 EOT token kicks off generation.

            print("generating:\n---")
            for t in range(1, gen_max_length):
                # NOTE: Inference is wasteful here because for each t we recompute all activations between 0 and t.
                # In a real inference setting we would use a separate code for this anyway.
                # Inference is here only for sanity checking.
                model.forward(gen_tokens, null_int32_ptr, 1, t)
                var dev_logits_ptr = (
                    model.acts.logits + (t - 1) * model.config.padded_vocab_size
                )
                model.ctx.enqueue_copy(
                    dst_ptr=rebind[
                        UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]
                    ](
                        model.logits_host_buf.unsafe_ptr().as_unsafe_any_origin()
                    ),
                    src_ptr=rebind[
                        UnsafePointer[Scalar[GPT2_DTYPE], ImmutAnyOrigin]
                    ](dev_logits_ptr.as_unsafe_any_origin()),
                    size=model.config.vocab_size,
                )
                model.ctx.synchronize()

                var logits = rebind[ImmutMemPtr[DType.float32]](
                    model.logits_host_buf.unsafe_ptr().as_unsafe_any_origin()
                )
                var coin = random_f32(rng_state)
                var next_token = sample_softmax(
                    logits, model.config.vocab_size, coin
                )
                gen_tokens[t] = Scalar[DType.int32](next_token)
                var token_str = tokenizer.decode(next_token)
                safe_print(token_str)
            print("\n---")

        # Once in a while checkpoint the optimization state (all ranks).
        if (
            save_every > 0
            and output_dir != ""
            and ((step > 0 and step % save_every == 0) or last_step)
        ):
            var model_path = output_dir + "/model_" + String(step) + ".bin"
            var state_path = (
                output_dir
                + "/state_"
                + String(step)
                + "_"
                + String(rank)
                + ".bin"
            )
            model.write_checkpoint(
                model_path, state_path, step, train_loader, rng_state
            )
            printf0(rank, "Writing checkpoint at step " + String(step))

        if last_step:
            break

        # --------------- TRAINING SECTION -------------------
        # If overfitting a single batch for debugging, reset the loader so every
        # step re-reads the same batch (matches llm.c / train_gpt2.py).
        if args.overfit_single_batch == 1:
            train_loader.reset()

        var time_start = global_perf_counter_ns()

        # Gradient/loss accumulation over micro-batches. Gradients accumulate
        # inside backward (zeroed only on micro_step 0); the loss is averaged.
        var accumulated_loss = Float32(0.0)
        for micro_step in range(grad_accum_steps):
            train_loader.next_batch()
            model.forward(
                train_loader.inputs,
                train_loader.targets,
                B,
                T,
                grad_accum_steps,
            )
            accumulated_loss += model.mean_loss
            model.backward(grad_accum_steps, micro_step, step)
        model.ctx.synchronize()
        accumulated_loss /= Float32(grad_accum_steps)

        var zloss = loss_detector.update(Float64(accumulated_loss))
        var step_learning_rate = learning_rate_scheduler.get_learning_rate(step)

        # Gradient norm, its z-score, and clipping to a max norm of 1.0.
        var grad_norm = model.calculate_grad_norm()
        var zgrad = grad_norm_detector.update(Float64(grad_norm))
        var grad_clip = Float32(1.0)
        var grad_scale = (
            grad_clip / grad_norm if grad_norm > grad_clip else Float32(1.0)
        )

        model.update(
            UInt32(step + 1),
            step_learning_rate,
            weight_decay=args.weight_decay,
            grad_scale=grad_scale,
        )
        model.ctx.synchronize()
        var time_end = global_perf_counter_ns()
        # --------------- TRAINING SECTION END -------------------

        var time_elapsed_ms = Float64(time_end - time_start) / 1e6
        var tokens_per_second = (
            Float64(tokens_per_step) / time_elapsed_ms * 1000.0
        )
        # Smooth tok/s with a bias-corrected EMA, treating step 0 as warmup.
        var bias_corrected_ema = tokens_per_second
        if step > 0:
            total_sum_iteration_time_s += time_elapsed_ms / 1000.0
            ema_tokens_per_second = (
                0.95 * ema_tokens_per_second + 0.05 * tokens_per_second
            )
            var bias_correction = 1.0 - exp(Float64(step) * log(0.95))
            bias_corrected_ema = ema_tokens_per_second / bias_correction

        var mfu = estimate_mfu(
            model.num_parameters,
            model.config.num_layer,
            model.config.channels,
            T,
            B * T * grad_accum_steps,
            time_elapsed_ms / 1000.0,
            device_name,
            USE_BF16,
        )
        var prec_label = String(PRECISION)
        var mfu_str = (
            String("n/a") if mfu < 0.0 else _ffmt(mfu * 100.0, 1) + "%"
        )

        printf0(
            rank,
            "step "
            + String(step + 1)
            + "/"
            + String(train_num_batches)
            + " | loss "
            + _ffmt(Float64(accumulated_loss), 6)
            + " ("
            + _zfmt(zloss)
            + "z)| norm "
            + _ffmt(Float64(grad_norm), 4)
            + " ("
            + _zfmt(zgrad)
            + "z)| lr "
            + String(step_learning_rate)
            + " | "
            + _ffmt(time_elapsed_ms, 2)
            + " ms | "
            + mfu_str
            + " "
            + prec_label
            + " MFU | "
            + _ffmt(bias_corrected_ema, 0)
            + " tok/s",
        )

    train_loader.close()
    val_loader.close()


def _dispatch_world_size[
    target: StaticString
](
    args: TrainArgs,
    rank: Int,
    world_size: Int,
    cpu_coord: Optional[UnsafePointer[CpuCoordinator, MutUntrackedOrigin]],
) raises:
    # train() is monomorphized on WORLD_SIZE (a comptime parameter), so a runtime
    # world_size can only dispatch to an instantiation compiled into this binary:
    # the common sizes below, plus whatever this binary was built for
    # (-D WORLD_SIZE=..., exposed as WORLD_SIZE_DEF). Listing WORLD_SIZE_DEF as
    # one branch is what makes larger sizes (16/32/64/...) reachable, without
    # compiling an instantiation for every possible size.
    if world_size == 1:
        train[target, 1](args, rank, cpu_coord)
    elif world_size == 2:
        train[target, 2](args, rank, cpu_coord)
    elif world_size == 4:
        train[target, 4](args, rank, cpu_coord)
    elif world_size == 8:
        train[target, 8](args, rank, cpu_coord)
    elif world_size == WORLD_SIZE_DEF:
        train[target, WORLD_SIZE_DEF](args, rank, cpu_coord)
    else:
        raise Error(
            "Unsupported world_size: "
            + String(world_size)
            + ". Supported values: 1, 2, 4, 8, or the compile-time WORLD_SIZE: "
            + String(WORLD_SIZE_DEF)
        )


def _print_usage():
    print("Usage: ./train_gpt2 [options]")
    print("Options (mirroring Karpathy's train_gpt2.cu):")
    print("  -h, --help  print this help message and exit")
    print("  -i <string> train data filename pattern (glob ok)")
    print("  -j <string> val data filename pattern (glob ok)")
    print(
        "  -e <string> input .bin filename OR model descriptor (dX / gpt2:dX /"
        " gpt3:cX) to train from scratch"
    )
    print("  -o <string> output log/checkpoint dir (default = none)")
    print("  -n <int>    write a checkpoint every N steps (default 0)")
    print("  -y <int>    resume from latest checkpoint in -o? (0/1)")
    print("  -b <int>    (micro) batch size B (default = 4)")
    print("  -t <int>    sequence length T (default = 1024)")
    print(
        "  -d <int>    total desired batch size (default = B*T*num_processes)"
    )
    print("  -l <float>  learning rate (default = 3e-4)")
    print("  -k <string> learning rate scheduler (default = cosine)")
    print("  -u <int>    learning rate warmup iterations (default = 0)")
    print("  -q <float>  final learning rate fraction (default = 1.0)")
    print("  -c <float>  weight decay (default = 0.0)")
    print("  -v <int>    val_loss_every (default = 20)")
    print("  -m <int>    val_max_steps (default = 20)")
    print("  -s <int>    sample_every (default = 20)")
    print("  -g <int>    genT, tokens of inference to generate (default = 64)")
    print("  -a <int>    overfit a single batch? 0/1 (default = 0)")
    print("  -x <int>    max_steps (-1 = one epoch, default)")
    print(
        "  -z <int>    zero_stage, ZeRO optimization stage 0/1/2/3 (default 0)"
    )
    print("  -pn <int>   num_processes / world_size (default = 1)")
    print("  -pr <int>   process_rank (default = 0)")


def _error_usage() raises:
    _print_usage()
    raise Error("invalid command-line usage")


def _parse_train_args(
    mut args: TrainArgs, mut rank: Int, mut world_size: Int
) raises -> Bool:
    # Seeds TrainArgs from env vars, then parses llm.c-style flags over them.
    # Returns False when -h/--help was handled and the caller should exit.
    var cli = argv()

    # Env variables seed the defaults; command-line flags override them below.
    var env_rank = getenv("RANK")
    if env_rank != "":
        rank = atol(env_rank)
    var env_world_size = getenv("WORLD_SIZE")
    if env_world_size != "":
        world_size = atol(env_world_size)
    var env_zero_stage = getenv("ZERO_STAGE")
    if env_zero_stage != "":
        args.zero_stage = atol(env_zero_stage)
    if getenv("LLMM_OUTPUT_DIR") != "":
        args.output_log_dir = getenv("LLMM_OUTPUT_DIR")
    if getenv("LLMM_SAVE_EVERY") != "":
        args.checkpoint_every = atol(getenv("LLMM_SAVE_EVERY"))

    # Parse the flags. Each is "-x" or "-xy" followed by a value, like llm.c.
    var i = 1
    while i < len(cli):
        var flag = String(cli[i])
        # Help takes no value, so handle it before the "must have a value" check.
        if flag == "-h" or flag == "--help":
            _print_usage()
            return False
        if (
            i + 1 >= len(cli)
            or not flag.startswith("-")
            or flag.byte_length() < 2
        ):
            _error_usage()
        var val = String(cli[i + 1])
        if flag == "-i":
            args.train_data_pattern = val
        elif flag == "-j":
            args.val_data_pattern = val
        elif flag == "-e":
            args.load_filename = val
        elif flag == "-o":
            args.output_log_dir = val
        elif flag == "-n":
            args.checkpoint_every = atol(val)
        elif flag == "-y":
            args.resume = atol(val)
        elif flag == "-b":
            args.batch_size = atol(val)
        elif flag == "-t":
            args.seq_len = atol(val)
        elif flag == "-d":
            args.total_batch_size = atol(val)
        elif flag == "-l":
            args.learning_rate = Float32(atof(val))
        elif flag == "-k":
            args.lr_scheduler_type = val
        elif flag == "-u":
            args.warmup_iterations = atol(val)
        elif flag == "-q":
            args.final_learning_rate_frac = Float32(atof(val))
        elif flag == "-c":
            args.weight_decay = Float32(atof(val))
        elif flag == "-v":
            args.val_loss_every = atol(val)
        elif flag == "-m":
            args.val_max_steps = atol(val)
        elif flag == "-s":
            args.sample_every = atol(val)
        elif flag == "-g":
            args.gen_tokens = atol(val)
        elif flag == "-a":
            args.overfit_single_batch = atol(val)
        elif flag == "-x":
            args.max_steps = atol(val)
        elif flag == "-z":
            args.zero_stage = atol(val)
        elif flag == "-pn":
            world_size = atol(val)
        elif flag == "-pr":
            rank = atol(val)
        else:
            print("Unknown flag: " + flag)
            _error_usage()
        i += 2
    return True


def _dispatch_cpu(args: TrainArgs, world_size: Int) raises:
    # CPU training is fp32 by policy (matching profile_gpt2.mojo): a bf16 (or
    # fp8/fp4, which also store bf16 — see USE_BF16 above) build must not
    # instantiate the "cpu" dispatch — the CPU low-precision GEMM packing path
    # crashes AArch64 instruction selection at compile time.
    comptime if USE_BF16:
        raise Error(
            "'"
            + String(PRECISION)
            + "' builds support only the GPU target (CPU stays fp32)."
            " Rebuild with -D LLMM_PRECISION=fp32 (or without -D LLMM_BF16=1)"
            " to train on CPU."
        )
    else:
        print("Training on CPU.")
        if world_size == 1:
            _dispatch_world_size["cpu"](
                args,
                0,
                world_size,
                Optional[UnsafePointer[CpuCoordinator, MutUntrackedOrigin]](),
            )
        else:
            var cpu_coord_ptr = alloc[CpuCoordinator](1)
            cpu_coord_ptr[] = CpuCoordinator(world_size)

            @parameter
            def _run_rank(rank: Int):
                try:
                    _dispatch_world_size["cpu"](
                        args, rank, world_size, cpu_coord_ptr
                    )
                except e:
                    print("Rank", rank, "failed:", e)

            sync_parallelize[_run_rank](world_size)
            cpu_coord_ptr[].free()
            cpu_coord_ptr.free()


def _try_gpu(args: TrainArgs, rank: Int, world_size: Int) raises -> Bool:
    """Dispatch GPU training when this build and host support it; else False."""
    # The "gpu" target is comptime-guarded: stdlib GPU-arch lookup fails the
    # build on hosts with no accelerator (e.g. CPU-only Linux).
    comptime if has_accelerator():
        # Metal disabled at compile time (-D LLMM_DISABLE_METAL=1).
        comptime if has_apple_gpu_accelerator() and not HAS_METAL:
            return False

        if has_apple_gpu_accelerator():
            print(
                "Note: Metal (Apple GPU) training is experimental. Set"
                " LLMM_USE_CPU=1 or build with -D LLMM_DISABLE_METAL=1 to fall"
                " back to CPU."
            )
            if world_size > 1:
                raise Error(
                    "multi-rank training (world_size="
                    + String(world_size)
                    + ") is not supported on Apple GPU (collectives require"
                    " NVIDIA). Re-run with WORLD_SIZE=1, set LLMM_USE_CPU=1,"
                    " or use an NVIDIA backend."
                )

        print("GPU detected — training on GPU.")
        _dispatch_world_size["gpu"](
            args,
            rank,
            world_size,
            Optional[UnsafePointer[CpuCoordinator, MutUntrackedOrigin]](),
        )
        return True
    else:
        return False


def main() raises:
    var args = default_train_args()
    var rank = 0
    var world_size = 1
    if not _parse_train_args(args, rank, world_size):
        return

    print(
        "Rank:",
        rank,
        "World size:",
        world_size,
        "ZeRO stage:",
        args.zero_stage,
    )

    if getenv("LLMM_USE_CPU") != "" or not _try_gpu(args, rank, world_size):
        _dispatch_cpu(args, world_size)
