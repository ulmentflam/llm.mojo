import std.math
from std.sys import simd_width_of
from std.memory import UnsafePointer
from std.gpu.host import DeviceContext
from layout import TensorLayout, TileTensor
from std.gpu.host.info import is_cpu, is_gpu
from std.algorithm import sync_parallelize, vectorize
from std.gpu import (
    block_dim,
    block_idx,
    thread_idx,
)


# ===----------------------------------------------------------------------=== #
# Constants and Type Aliases
# ===----------------------------------------------------------------------=== #

alias Gradient = TileTensor
alias CHUNK_SIZE = 4096
alias UNROLL = 4


# ===----------------------------------------------------------------------=== #
# Utility Functions
# ===----------------------------------------------------------------------=== #


def lerp[
    dtype: DType, out_dtype: DType
](
    start: Scalar[dtype], end: Scalar[out_dtype], weight: Scalar[dtype]
) -> Scalar[out_dtype]:
    return math.fma(weight, end, math.fma(-weight, start, start))


# ===----------------------------------------------------------------------=== #
# AdamW Optimizer
# ===----------------------------------------------------------------------=== #


@value
struct AdamWConfig[dtype: DType]:
    var learning_rate: Scalar[dtype]
    var beta1: Scalar[dtype]
    var beta2: Scalar[dtype]
    var eps: Scalar[dtype]
    var weight_decay: Scalar[dtype]
    var grad_scale: Scalar[dtype]


def _adamw_update[
    dtype: DType,
    width: UInt32,
](
    inoutparams: TileTensor[dtype],
    grads: Gradient[dtype],
    m_memory: UnsafePointer[Float32],
    v_memory: UnsafePointer[Float32],
    config: AdamWConfig[dtype],
    beta1_correction: Scalar[dtype],
    beta2_correction: Scalar[dtype],
    idx: UInt32,
) -> None:
    var param = params.data.load[width=width](idx)
    var grad = config.grad_scale * grads.data.load[width=width](idx)
    var m = m_memory.load[width=width](idx)
    var v = v_memory.load[width=width](idx)

    # First Moment (momentum)
    m = lerp[dtype, Float32](
        grad, m, config.beta1
    )  # The first moment will stay in Float32 to avoid percision loss.
    m_memory.store[width=width](idx, m)
    m *= beta1_correction

    # Second Moment (RMSProp)
    v = lerp[dtype, Float32](
        grad**2, v, config.beta2
    )  # The second moment will stay in Float32 to avoid percision loss.
    v_memory.store[width=width](idx, v)
    v *= beta2_correction

    # Update
    # First Update
    param -= config.learning_rate * m / (math.sqrt(v) + config.eps)
    # Decoupled Weight Decay, this is what distinguishes AdamW from Adam.
    param -= config.learning_rate * config.weight_decay * param
    # TODO: Karpathy adds a stocastic rounding function here, this updates low percision values in the params.
    params.data.store[width=width](idx, param)


def adamw_update_cpu[
    dtype: DType,
    width: UInt32,
](
    inoutparams: TileTensor[dtype],
    grads: Gradient[dtype],
    m_memory: UnsafePointer[Float32],
    v_memory: UnsafePointer[Float32],
    config: AdamWConfig[dtype],
    beta1_correction: Scalar[dtype],
    beta2_correction: Scalar[dtype],
) -> None:
    assert (
        params.shape == grads.shape
    ), "params and grads must have the same shape"

    var n = params.size()
    var num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

    @parameter
    def _chunk(c: Int32) -> None:
        var base = c * CHUNK_SIZE
        var count = min(CHUNK_SIZE, n - base)

        @parameter
        def _simd[width: Int32](local: Int32) -> None:
            var idx = base + local
            _adamw_update[dtype, width](
                params,
                grads,
                m_memory,
                v_memory,
                config,
                beta1_correction,
                beta2_correction,
                idx,
            )

        vectorize[_simd, width, unroll_factor=UNROLL](count)

    sync_parallelize[_chunk](num_chunks)


def adamw_update_gpu[
    dtype: DType,
    width: UInt32 = 4,  # Pinning to 4 as this should be optimial for m_memory and v_memory in float32. If we go higher, we will start to lose the benifits.
](
    inoutparams: TileTensor[dtype],
    grads: Gradient[dtype],
    m_memory: UnsafePointer[Float32],
    v_memory: UnsafePointer[Float32],
    config: AdamWConfig[dtype],
    beta1_correction: Scalar[dtype],
    beta2_correction: Scalar[dtype],
) -> None:
    var num_params = params.size()
    var idx = (block_idx.x * block_dim.x + thread_idx.x) * width

    if idx + width <= num_params:
        _adamw_update[dtype, width](
            params,
            grads,
            m_memory,
            v_memory,
            config,
            beta1_correction,
            beta2_correction,
            idx,
        )
    elif idx < num_params:
        # If the last vector straddles num_params, we need to handle the remaining elements one by one.
        for i in range(idx, num_params):
            _adamw_update[dtype, 1](
                params, grads, m_memory, v_memory, config, i
            )


def adamw_update[
    dtype: DType,
    target: StaticString,
    width: UInt32 = 4,  # Pinning to 4 as this should be optimial for m_memory and v_memory in float32, same reason as the pinned width in adamw_update_gpu.
](
    inoutparams: TileTensor[dtype],
    grads: Gradient[dtype],
    m_memory: UnsafePointer[Float32],
    v_memory: UnsafePointer[Float32],
    t: UInt32,
    config: AdamWConfig[dtype],
    ctx: DeviceContext,
) capturing raises:
    var beta1_correction = 1 / (1 - config.beta1**t)
    var beta2_correction = 1 / (1 - config.beta2**t)

    @parameter
    if is_cpu[target]():
        var simd_width = simd_width_of[
            dtype
        ]()  # This is the width of the SIMD register.
        adamw_update_cpu[dtype, simd_width](
            params,
            grads,
            m_memory,
            v_memory,
            config,
            beta1_correction,
            beta2_correction,
        )
    elif is_gpu[target]():
        alias BLOCK_SIZE = 256
        var num_params = params.size()
        var num_threads = (
            num_params + width - 1
        ) // width  # This is the number of threads needed to process all the parameters.
        var num_blocks = ceildiv(num_threads, BLOCK_SIZE)

        var compiled = ctx.compile_function[adamw_update_gpu[dtype, width]]()

        ctx.enqueue_function(
            compiled,
            params,
            grads,
            m_memory,
            v_memory,
            config,
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
        width: UInt32 = 4,
    ](
        params: OutputTensor[dtype, rank=1, ...],
        m_memory: OutputTensor[Float32, rank=1, ...],
        v_memory: OutputTensor[Float32, rank=1, ...],
        t: UInt32,
        grads: InputTensor[dtype, rank=1, ...],
        config: AdamWConfig[dtype],
        ctx: DeviceContext,
    ) capturing raises:
        if params.shape() != grads.shape():
            raise Error("params and grads must have the same shape")
        if m_memory.shape() != params.shape():
            raise Error("m_memory and params must have the same shape")
        if v_memory.shape() != params.shape():
            raise Error("v_memory and params must have the same shape")

        var params_tt = params.to_tile_tensor[Int32]()
        var grads_tt = grads.to_tile_tensor[Int32]()
        var m_tt = m_memory.to_tile_tensor[Int32]()
        var v_tt = v_memory.to_tile_tensor[Int32]()

        adamw_update[dtype, target, width](
            params_tt,
            grads_tt,
            m_tt,
            v_tt,
            t,
            config,
            ctx,
        )
