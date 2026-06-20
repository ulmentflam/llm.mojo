import compiler
from std.sys import simd_width_of
from extensibility import InputTensor
from std.gpu.host import DeviceContext
from std.math import fma, sqrt, ceildiv
from std.gpu.host.info import is_cpu, is_gpu
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.gpu import block_dim, block_idx, thread_idx
from std.algorithm import vectorize, sync_parallelize
from llmm.memory import ImmutKernelPtr, MutKernelPtr


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime CHUNK_SIZE = 4096
comptime UNROLL = 4


# ===----------------------------------------------------------------------=== #
# Utility Functions
# ===----------------------------------------------------------------------=== #


# NOTE: If we reuse this function, we can move it to a shared file.
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


@fieldwise_init
struct AdamWConfig[dtype: DType]:
    var learning_rate: Scalar[Self.dtype]
    var beta1: Scalar[Self.dtype]
    var beta2: Scalar[Self.dtype]
    var eps: Scalar[Self.dtype]
    var weight_decay: Scalar[Self.dtype]
    var grad_scale: Scalar[Self.dtype]


@always_inline
def _adamw_update[
    dtype: DType,
    width: Int,
](
    idx: Int,
    params_ptr: MutKernelPtr[dtype],
    grads_ptr: ImmutKernelPtr[dtype],
    m_ptr: MutKernelPtr[DType.float32],
    v_ptr: MutKernelPtr[DType.float32],
    config: AdamWConfig[dtype],
    beta1_correction: Scalar[DType.float32],
    beta2_correction: Scalar[DType.float32],
) -> None:
    var param = (params_ptr + idx).load[width=width]()
    var grad = config.grad_scale * (grads_ptr + idx).load[width=width]()
    var m = (m_ptr + idx).load[width=width]()
    var v = (v_ptr + idx).load[width=width]()

    # First moment (momentum). Kept in Float32 to avoid precision loss.
    # `grad` is cast once to fp32 so squaring (for v) doesn't over/underflow
    # in low-precision dtypes; lerp handles the beta cast internally.
    var grad_fp32 = grad.cast[DType.float32]()
    m = lerp(grad_fp32, m, config.beta1)
    (m_ptr + idx).store[width=width](m)
    m *= beta1_correction

    # Second moment (RMSProp). Also kept in Float32 to avoid precision loss.
    v = lerp(grad_fp32 * grad_fp32, v, config.beta2)
    (v_ptr + idx).store[width=width](v)
    v *= beta2_correction

    # Decoupled weight decay update (this is what distinguishes AdamW from Adam).
    var step_fp32 = m / (sqrt(v) + config.eps.cast[DType.float32]())
    var step = step_fp32.cast[dtype]()
    param -= config.learning_rate * (step + config.weight_decay * param)
    # TODO: Karpathy adds a stochastic rounding function here for low-precision params.
    (params_ptr + idx).store[width=width](param)


def adamw_update_cpu[
    dtype: DType,
    width: Int,
](
    num_params: Int,
    params_ptr: MutKernelPtr[dtype],
    grads_ptr: ImmutKernelPtr[dtype],
    m_ptr: MutKernelPtr[DType.float32],
    v_ptr: MutKernelPtr[DType.float32],
    config: AdamWConfig[dtype],
    beta1_correction: Scalar[DType.float32],
    beta2_correction: Scalar[DType.float32],
) -> None:
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
            m_ptr,
            v_ptr,
            config,
            beta1_correction,
            beta2_correction,
            base,
        }:
            var idx = base + local
            _adamw_update[dtype, w](
                idx,
                params_ptr,
                grads_ptr,
                m_ptr,
                v_ptr,
                config,
                beta1_correction,
                beta2_correction,
            )

        vectorize[width, unroll_factor=UNROLL](count, _simd)

    sync_parallelize[_chunk](num_chunks)


def adamw_update_gpu[
    dtype: DType,
    width: Int = 4,
](
    num_params: Int,
    params_ptr: MutKernelPtr[dtype],
    grads_ptr: ImmutKernelPtr[dtype],
    m_ptr: MutKernelPtr[DType.float32],
    v_ptr: MutKernelPtr[DType.float32],
    learning_rate: Scalar[dtype],
    beta1: Scalar[dtype],
    beta2: Scalar[dtype],
    eps: Scalar[dtype],
    weight_decay: Scalar[dtype],
    grad_scale: Scalar[dtype],
    beta1_correction: Scalar[DType.float32],
    beta2_correction: Scalar[DType.float32],
) -> None:
    var config = AdamWConfig[dtype](
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        grad_scale=grad_scale,
    )
    var idx = Int((block_idx.x * block_dim.x + thread_idx.x) * width)
    if idx + width <= num_params:
        _adamw_update[dtype, width](
            idx,
            params_ptr,
            grads_ptr,
            m_ptr,
            v_ptr,
            config,
            beta1_correction,
            beta2_correction,
        )
    elif idx < num_params:
        # Last vector straddles num_params so handle the remainder one element at a time.
        for i in range(idx, num_params):
            _adamw_update[dtype, 1](
                i,
                params_ptr,
                grads_ptr,
                m_ptr,
                v_ptr,
                config,
                beta1_correction,
                beta2_correction,
            )


def adamw_update[
    dtype: DType,
    target: StaticString,
    width: Int = 4,  # Pinning to 4 — optimal for the Float32 moment loads.
](
    num_params: Int,
    params_ptr: MutKernelPtr[dtype],
    grads_ptr: ImmutKernelPtr[dtype],
    m_ptr: MutKernelPtr[DType.float32],
    v_ptr: MutKernelPtr[DType.float32],
    t: UInt32,
    config: AdamWConfig[dtype],
    ctx: DeviceContext,
) capturing raises:
    # Bias correction math runs in Float32 to match the moment storage.
    # `Float32(t)` is a constructor promotion (not a numeric cast), the
    # natural exponent type for Float32 `**`.
    var beta1_fp32 = Scalar[DType.float32](config.beta1)
    var beta2_fp32 = Scalar[DType.float32](config.beta2)
    var t_fp32 = Float32(t)
    var beta1_correction = Scalar[DType.float32](1) / (
        Scalar[DType.float32](1) - beta1_fp32**t_fp32
    )
    var beta2_correction = Scalar[DType.float32](1) / (
        Scalar[DType.float32](1) - beta2_fp32**t_fp32
    )

    comptime if is_cpu[target]():
        comptime simd_width = simd_width_of[dtype]()
        adamw_update_cpu[dtype, simd_width](
            num_params,
            params_ptr,
            grads_ptr,
            m_ptr,
            v_ptr,
            config,
            beta1_correction,
            beta2_correction,
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
        width: Int = 4,
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

        var config = AdamWConfig[dtype](
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            grad_scale=grad_scale,
        )

        # Lower to kernel pointers at the dispatch boundary so CPU + GPU
        # share one code path (`_adamw_step` works on pointers in both cases).
        adamw_update[dtype, target, width](
            params.size(),
            params.unsafe_ptr(),
            grads.unsafe_ptr(),
            m_memory.unsafe_ptr(),
            v_memory.unsafe_ptr(),
            t,
            config,
            ctx,
        )
