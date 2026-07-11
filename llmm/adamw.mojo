import compiler
from std.algorithm import vectorize
from extensibility import InputTensor
from std.gpu.host import DeviceContext
from std.math import fma, sqrt, ceildiv
from std.sys import simd_width_of, align_of, is_defined, get_defined_int
from std.gpu.host.info import is_cpu, is_gpu
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.gpu import block_dim, block_idx, thread_idx

from llmm.profiler import traced_parallelize
from llmm.memory import ImmutKernelPtr, MutKernelPtr
from llmm.rng_device import sr_cast_bf16


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime CHUNK_SIZE = 4096
comptime UNROLL = 4

# Opt-in stochastic rounding for the master->low-precision param store,
# enabled with `-D LLMM_SR_MASTER=1`. Default OFF: the store falls through
# to the plain `param.cast[dtype]()` RNE path below. Only valid for a bf16
# param store (asserted in `_adamw_update`); fp8/fp4 encoders are not wired
# through here.
comptime SR_MASTER_ENABLED = is_defined["LLMM_SR_MASTER"]()

# Fixed by default (not wall-clock-derived) so `-D LLMM_SR_MASTER=1` runs are
# reproducible out of the box; override with `-D LLMM_SR_SEED=<int>` for a
# different (still deterministic) random stream. This is a 64-bit RNG seed,
# not a training hyperparameter — it has no relationship to `INIT_RNG_SEED`
# (train_gpt2.mojo's from-scratch weight-init seed).
comptime SR_MASTER_SEED = UInt64(get_defined_int["LLMM_SR_SEED", 1746221221]())

# rng_device stream id reserved for this call site so its substream stays
# independent of any other SR call site sharing LLMM_SR_SEED. See
# llmm/rng_device.mojo's stream registry.
comptime SR_MASTER_STREAM = UInt64(1)


# ===----------------------------------------------------------------------=== #
# Utility Functions
# ===----------------------------------------------------------------------=== #


@always_inline
def lerp[
    start_dtype: DType,
    end_dtype: DType,
    weight_dtype: DType,
    width: Int,
](
    start: SIMD[start_dtype, width],
    end: SIMD[end_dtype, width],
    weight: SIMD[weight_dtype, 1],
) -> SIMD[end_dtype, width]:
    """Linear interpolation: (1 - weight) * start + weight * end.

    Inputs are cast to `end_dtype` (the accumulator) and `weight` is broadcast
    to full width, so callers can pass mixed-precision values
    (e.g., bf16 grad + fp32 moment + bf16 beta) without having to cast.
    """
    var s = start.cast[end_dtype]()
    var w = SIMD[end_dtype, width](weight.cast[end_dtype]())
    return fma(w, end, fma(-w, s, s))


# ===----------------------------------------------------------------------=== #
# AdamW Optimizer
# ===----------------------------------------------------------------------=== #


# All AdamW hyperparameters are Float32, independent of the parameter precision
# (matching llm.c, where they are plain `float`). Keeping them in fp32 means the
# whole update runs in fp32 — in particular `eps` doesn't underflow to 0 when the
# parameters are bf16/fp8 — and the low-precision weights are only ever a rounded
# view of the fp32 math.
@fieldwise_init
struct AdamWConfig:
    var learning_rate: Scalar[DType.float32]
    var beta1: Scalar[DType.float32]
    var beta2: Scalar[DType.float32]
    var eps: Scalar[DType.float32]
    var weight_decay: Scalar[DType.float32]
    var grad_scale: Scalar[DType.float32]


@always_inline
def _adamw_update[
    dtype: DType,
    width: Int,
](
    idx: Int,
    params_ptr: MutKernelPtr[dtype],
    grads_ptr: ImmutKernelPtr[dtype],
    # fp32 master copy of the weights, used only when `has_master` is set (mixed-
    # precision training — bf16/fp8/fp4 params — where the master is the source of
    # truth and the low-precision `params` is a rounded view of it). For pure-fp32
    # training `has_master` is False and `params` is itself the master; the
    # pointer is then a harmless unused stand-in. MAX forbids null UnsafePointers,
    # so we model "no master" with the flag rather than a null pointer.
    master_ptr: MutKernelPtr[DType.float32],
    has_master: Bool,
    m_ptr: MutKernelPtr[DType.float32],
    v_ptr: MutKernelPtr[DType.float32],
    config: AdamWConfig,
    beta1_correction: Scalar[DType.float32],
    beta2_correction: Scalar[DType.float32],
    # AdamW step, only consumed when `SR_MASTER_ENABLED` (folded into the SR
    # counter below so repeated master->bf16 stores don't reuse the same
    # random bit for a given element across steps). Otherwise unused — kept
    # as a plain arg (not Optional) to match the rest of this kernel's
    # "no null pointers / no Optionals in the hot path" style.
    t: UInt32,
) -> None:
    # This mirrors llm.c's `adamw_update`: read the grad as fp32, keep both Adam
    # moments in fp32, and do the weight update in fp32 against the master copy.
    # Explicit alignment (idx = global_tid*width is naturally aligned but the
    # compiler can't prove it) so the wide 128-bit loads/stores are emitted —
    # without it Mojo issues narrow transactions and this bandwidth-bound kernel
    # runs at ~half speed (same fix as the fused classifier).
    comptime align_d = align_of[SIMD[dtype, width]]()
    comptime align_f = align_of[SIMD[DType.float32, width]]()
    var grad = (
        config.grad_scale
        * (grads_ptr + idx)
        .load[width=width, alignment=align_d]()
        .cast[DType.float32]()
    )
    var m = (m_ptr + idx).load[width=width, alignment=align_f]()
    var v = (v_ptr + idx).load[width=width, alignment=align_f]()

    # First moment (momentum), then bias-corrected to m_hat. The correction is a
    # width-1 scalar; broadcast it so the in-place op matches widths on CPU.
    m = lerp(grad, m, config.beta1)
    (m_ptr + idx).store[width=width, alignment=align_f](m)
    m /= SIMD[DType.float32, width](beta1_correction)

    # Second moment (RMSProp), then bias-corrected to v_hat.
    v = lerp(grad * grad, v, config.beta2)
    (v_ptr + idx).store[width=width, alignment=align_f](v)
    v /= SIMD[DType.float32, width](beta2_correction)

    # The current weight comes from the master copy when we keep one, otherwise
    # the params are already fp32-equivalent and serve as their own master.
    var old_param: SIMD[DType.float32, width]
    if has_master:
        old_param = (master_ptr + idx).load[width=width, alignment=align_f]()
    else:
        old_param = (
            (params_ptr + idx)
            .load[width=width, alignment=align_d]()
            .cast[DType.float32]()
        )

    # Decoupled weight decay update (this is what distinguishes AdamW from Adam).
    var param = old_param - config.learning_rate * (
        m / (sqrt(v) + config.eps) + config.weight_decay * old_param
    )

    # Round the fp32 result down into the low-precision params, and keep the fp32
    # master in sync when we have one.
    #
    # Karpathy's llm.c leaves a plain round-to-nearest-even (RNE) cast here
    # ("TODO: stochastic rounding for low-precision params"); RNE
    # systematically truncates small updates toward zero over many steps.
    # `-D LLMM_SR_MASTER=1` swaps in stochastic rounding (llmm/rng_device.mojo)
    # instead: each element gets a fresh, deterministic pseudo-random draw
    # keyed by (SR_MASTER_SEED, element index, step `t`), and rounds to the
    # nearer of the two representable bf16 neighbors with probability
    # proportional to proximity — unbiased in expectation instead of
    # systematically-truncating. Default OFF; the `else` branch below is the
    # original, untouched `param.cast[dtype]()` RNE line.
    comptime if SR_MASTER_ENABLED:
        comptime assert dtype == DType.bfloat16, (
            "LLMM_SR_MASTER=1 requires a bf16 low-precision param store"
            " (dtype == DType.bfloat16); fp8/fp4 stochastic rounding is not"
            " wired into this kernel"
        )
        var sr_param = SIMD[DType.bfloat16, width]()
        comptime for lane in range(width):
            # Counter = (t << 32) | element_index: unique per (step, element)
            # pair under a fixed seed, so repeated runs with the same seed
            # are bit-identical (same t, same idx -> same counter -> same
            # draw) while different steps/elements never collide.
            var counter = (UInt64(t) << 32) | UInt64(idx + lane)
            sr_param[lane] = sr_cast_bf16(
                param[lane], SR_MASTER_SEED, counter, SR_MASTER_STREAM
            )
        (params_ptr + idx).store[width=width, alignment=align_d](
            sr_param.cast[dtype]()
        )
    else:
        (params_ptr + idx).store[width=width, alignment=align_d](
            param.cast[dtype]()
        )
    if has_master:
        (master_ptr + idx).store[width=width, alignment=align_f](param)


def adamw_update_cpu[
    dtype: DType,
    width: Int,
](
    num_params: Int,
    params_ptr: MutKernelPtr[dtype],
    grads_ptr: ImmutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    has_master: Bool,
    m_ptr: MutKernelPtr[DType.float32],
    v_ptr: MutKernelPtr[DType.float32],
    config: AdamWConfig,
    beta1_correction: Scalar[DType.float32],
    beta2_correction: Scalar[DType.float32],
    t: UInt32,
) raises -> None:
    var num_chunks = (num_params + CHUNK_SIZE - 1) // CHUNK_SIZE

    @parameter
    def _chunk(c: Int):
        var base = c * CHUNK_SIZE
        var count = min(CHUNK_SIZE, num_params - base)

        @always_inline
        def _simd[
            w: Int
        ](local: Int) {
            params_ptr,
            grads_ptr,
            master_ptr,
            has_master,
            m_ptr,
            v_ptr,
            config,
            beta1_correction,
            beta2_correction,
            base,
            t,
        }:
            var idx = base + local
            _adamw_update[dtype, w](
                idx,
                params_ptr,
                grads_ptr,
                master_ptr,
                has_master,
                m_ptr,
                v_ptr,
                config,
                beta1_correction,
                beta2_correction,
                t,
            )

        vectorize[width, unroll_factor=UNROLL](count, _simd)

    traced_parallelize["adamw_update", _chunk](num_chunks)


def adamw_update_cpu_seq[
    dtype: DType,
    width: Int,
](
    num_params: Int,
    params_ptr: MutKernelPtr[dtype],
    grads_ptr: ImmutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    has_master: Bool,
    m_ptr: MutKernelPtr[DType.float32],
    v_ptr: MutKernelPtr[DType.float32],
    config: AdamWConfig,
    beta1_correction: Scalar[DType.float32],
    beta2_correction: Scalar[DType.float32],
    t: UInt32,
) -> None:
    var i = 0
    while i + width <= num_params:
        _adamw_update[dtype, width](
            i,
            params_ptr,
            grads_ptr,
            master_ptr,
            has_master,
            m_ptr,
            v_ptr,
            config,
            beta1_correction,
            beta2_correction,
            t,
        )
        i += width
    while i < num_params:
        _adamw_update[dtype, 1](
            i,
            params_ptr,
            grads_ptr,
            master_ptr,
            has_master,
            m_ptr,
            v_ptr,
            config,
            beta1_correction,
            beta2_correction,
            t,
        )
        i += 1


def adamw_update_gpu[
    dtype: DType,
    width: Int = 4,
](
    num_params: Int,
    params_ptr: MutKernelPtr[dtype],
    grads_ptr: ImmutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    # A 0/1 flag passed as a single byte (not Bool): enqueue_function only
    # marshals scalar/pointer kernel arguments, and one byte is all it needs.
    has_master_flag: UInt8,
    m_ptr: MutKernelPtr[DType.float32],
    v_ptr: MutKernelPtr[DType.float32],
    learning_rate: Scalar[DType.float32],
    beta1: Scalar[DType.float32],
    beta2: Scalar[DType.float32],
    eps: Scalar[DType.float32],
    weight_decay: Scalar[DType.float32],
    grad_scale: Scalar[DType.float32],
    beta1_correction: Scalar[DType.float32],
    beta2_correction: Scalar[DType.float32],
    t: UInt32,
) -> None:
    var config = AdamWConfig(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        grad_scale=grad_scale,
    )
    var has_master = has_master_flag != 0
    var idx = Int((block_idx.x * block_dim.x + thread_idx.x) * width)
    if idx + width <= num_params:
        _adamw_update[dtype, width](
            idx,
            params_ptr,
            grads_ptr,
            master_ptr,
            has_master,
            m_ptr,
            v_ptr,
            config,
            beta1_correction,
            beta2_correction,
            t,
        )
    elif idx < num_params:
        # Last vector straddles num_params so handle the remainder one element at a time.
        for i in range(idx, num_params):
            _adamw_update[dtype, 1](
                i,
                params_ptr,
                grads_ptr,
                master_ptr,
                has_master,
                m_ptr,
                v_ptr,
                config,
                beta1_correction,
                beta2_correction,
                t,
            )


def adamw_update[
    dtype: DType,
    target: StaticString,
    width: Int = 8,  # Pinning to 8 — optimal for 32-byte coalesced GPU loads.
    parallel: Bool = True,
](
    num_params: Int,
    params_ptr: MutKernelPtr[dtype],
    grads_ptr: ImmutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    has_master: Bool,
    m_ptr: MutKernelPtr[DType.float32],
    v_ptr: MutKernelPtr[DType.float32],
    t: UInt32,
    config: AdamWConfig,
    ctx: DeviceContext,
) capturing raises:
    # Bias-correction denominators (1 - beta**t), matching llm.c; the kernel
    # divides the moments by these to form m_hat / v_hat. `Float32(t)` is a
    # constructor promotion (not a numeric cast), the natural exponent type for
    # Float32 `**`.
    var t_fp32 = Float32(t)
    var beta1_correction = Scalar[DType.float32](1) - config.beta1**t_fp32
    var beta2_correction = Scalar[DType.float32](1) - config.beta2**t_fp32

    comptime if is_cpu[target]():
        comptime simd_width = simd_width_of[dtype]()
        comptime if parallel:
            adamw_update_cpu[dtype, simd_width](
                num_params,
                params_ptr,
                grads_ptr,
                master_ptr,
                has_master,
                m_ptr,
                v_ptr,
                config,
                beta1_correction,
                beta2_correction,
                t,
            )
        else:
            adamw_update_cpu_seq[dtype, simd_width](
                num_params,
                params_ptr,
                grads_ptr,
                master_ptr,
                has_master,
                m_ptr,
                v_ptr,
                config,
                beta1_correction,
                beta2_correction,
                t,
            )
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 256
        var dev_ctx = ctx
        # Each thread handles `width` elements, so the total thread count is
        # ceil(n / width) and the grid is ceil(num_threads / BLOCK_SIZE).
        var num_threads = (num_params + width - 1) // width
        var num_blocks = ceildiv(num_threads, BLOCK_SIZE)

        comptime gpu_kernel = adamw_update_gpu[dtype, width]
        var compiled = dev_ctx.compile_function[gpu_kernel]()
        dev_ctx.enqueue_function(
            compiled,
            num_params,
            params_ptr,
            grads_ptr,
            master_ptr,
            UInt8(1) if has_master else UInt8(0),
            m_ptr,
            v_ptr,
            config.learning_rate,
            config.beta1,
            config.beta2,
            config.eps,
            config.weight_decay,
            config.grad_scale,
            beta1_correction,
            beta2_correction,
            t,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


# ===----------------------------------------------------------------------=== #
# AdamW Compiler Registration
# ===----------------------------------------------------------------------=== #


@compiler.register("adamw_update")
struct AdamWUpdate:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        width: Int = 8,
    ](
        params: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        m_memory: MutableInputTensor[
            dtype=DType.float32, rank=1, static_spec=...
        ],
        v_memory: MutableInputTensor[
            dtype=DType.float32, rank=1, static_spec=...
        ],
        t: UInt32,
        grads: InputTensor[dtype=dtype, rank=1, static_spec=...],
        learning_rate: Scalar[dtype],
        beta1: Scalar[dtype],
        beta2: Scalar[dtype],
        eps: Scalar[dtype],
        weight_decay: Scalar[dtype],
        grad_scale: Scalar[dtype],
        ctx: DeviceContext,
    ) capturing raises:
        if params.size() != grads.size():
            raise Error("params and grads must have the same length")
        if m_memory.size() != params.size():
            raise Error("m_memory and params must have the same length")
        if v_memory.size() != params.size():
            raise Error("v_memory and params must have the same length")

        var config = AdamWConfig(
            learning_rate=Scalar[DType.float32](learning_rate),
            beta1=Scalar[DType.float32](beta1),
            beta2=Scalar[DType.float32](beta2),
            eps=Scalar[DType.float32](eps),
            weight_decay=Scalar[DType.float32](weight_decay),
            grad_scale=Scalar[DType.float32](grad_scale),
        )

        # This op exercises the no-master path (the params are their own master).
        # `has_master=False` makes the master pointer unused, so we pass any valid
        # fp32 buffer (m_memory) as a harmless stand-in rather than a null pointer
        # (which MAX forbids). Mixed-precision training passes a real master to
        # `adamw_update` directly; see GPT2.update.
        # Lower to kernel pointers at the dispatch boundary so CPU + GPU
        # share one code path (`_adamw_update` works on pointers in both cases).
        adamw_update[dtype, target, width](
            params.size(),
            params.unsafe_ptr(),
            grads.unsafe_ptr(),
            m_memory.unsafe_ptr(),
            False,
            m_memory.unsafe_ptr(),
            v_memory.unsafe_ptr(),
            t,
            config,
            ctx,
        )
