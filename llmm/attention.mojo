import compiler
from layout import Layout, TileTensor
from layout.tile_layout import row_major
from std.memory import alloc
from std.sys import simd_width_of, align_of, is_defined
from std.time import global_perf_counter_ns
from std.utils.index import IndexList, Index
from linalg.matmul import matmul
from linalg.matmul.vendor import blas
from _cublas.cublas import (
    cublasGemmStridedBatchedEx,
    ComputeType,
    Algorithm,
    check_cublas_error,
    _convert_to_cublas_transpose,
)
from _cublas.dtype import DataType
from linalg.matmul.vendor.blas import _get_global_handle
from layout.tensor_core import TensorCore
from std.gpu.primitives import warp, block
from extensibility import InputTensor
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceAttribute
from layout.layout_tensor import LayoutTensor
from std.gpu.host.info import is_cpu, is_gpu
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.runtime.asyncrt import parallelism_level
from std.algorithm import vectorize, sync_parallelize
from std.math import fma, sqrt, ceildiv, exp, log, ldexp, floor, exp2
from std.gpu import (
    barrier,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
    WARP_SIZE,
)

from llmm.split import split_fwd, split_bwd
from llmm.merge import merge_fwd, merge_bwd
from llmm.matmul import _matmul_cublaslt
from llmm.profiler import traced_parallelize
from llmm.memory import ImmutKernelPtr, MutKernelPtr
from llmm.vendor import HAS_CUBLAS, HAS_METAL, USE_TF32


# ===----------------------------------------------------------------------=== #
# KV Cache
# ===----------------------------------------------------------------------=== #


struct KVCache:
    var fwd_addr: Int
    var bwd_dq_addr: Int
    var bwd_dkv_addr: Int
    # Persistent GEMM-attention scratch (heap-held DeviceBuffer addresses). The
    # [B·nh, T, T] scores/probability planes are reused across every layer and
    # step, so they are allocated once and kept alive here. Allocating fresh
    # DeviceBuffers per layer races: their frees are not ordered against the
    # async GEMM/softmax kernels that still read them.
    var gemm_scores_addr: Int
    var gemm_att_addr: Int
    # Persistent GEMM-attention backward scratch (heap-held DeviceBuffer
    # addresses), allocated once like the forward scratch above.
    var gemm_ds_addr: Int
    var gemm_dst_addr: Int
    var gemm_pt_addr: Int
    var gemm_d_addr: Int
    var gemm_dp_addr: Int
    # Stored softmax probs (⑳-storage): base pointer of the train's per-layer
    # [L, B·nh, T, T] att_probs buffer, the current layer, and the per-layer
    # stride (B·nh·T·T). When att_probs_addr != 0 the forward writes P here (per
    # layer) and the backward READS it instead of recomputing QKᵀ.
    var att_probs_addr: Int
    var att_probs_layer: Int
    var att_probs_stride: Int

    def __init__(out self):
        self.fwd_addr = 0
        self.bwd_dq_addr = 0
        self.bwd_dkv_addr = 0
        self.gemm_scores_addr = 0
        self.gemm_att_addr = 0
        self.gemm_ds_addr = 0
        self.gemm_dst_addr = 0
        self.gemm_pt_addr = 0
        self.gemm_d_addr = 0
        self.gemm_dp_addr = 0
        self.att_probs_addr = 0
        self.att_probs_layer = 0
        self.att_probs_stride = 0


comptime KVCachePtr = UnsafePointer[
    mut=True, type=KVCache, origin=MutUntrackedOrigin
]


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime UNROLL = 4
comptime USE_LT_ATTN = False  # cuBLASLt-batched attention: verified NEUTRAL — it
# selects the identical cutlass wmma/tensorop kernels as cublasGemmStridedBatchedEx
# (hd=64 is at the hardware floor for both APIs, same as llm.c). Path kept, off.
comptime MAX_HEAD_DIM = 128

# Metal batched-scoreout gate (_attention_bmm_scoreout). The single-launch batched
# QKᵀ kernel is comptime-pinned to HD=64 and needs T a multiple of its tile
# (SCOREOUT_HEAD_DIM / SCOREOUT_T_MULTIPLE). It only wins while QKᵀ is dispatch-
# bound (short T); past SCOREOUT_MAX_T the per-head linalg.matmul reaches far more
# of peak, so long sequences take that fallback instead.
comptime SCOREOUT_HEAD_DIM = 64
comptime SCOREOUT_T_MULTIPLE = 32
comptime SCOREOUT_MAX_T = 256
comptime TAU = Scalar[DType.float32](
    8.0
)  # FlashAttention 4 deferred-max threshold
comptime LN_2 = Scalar[DType.float32](0.69314718056)  # ln(2)
comptime LOG2_EXP_MIN = Scalar[DType.float32](
    -126.0
)  # FlashAttention 4: ldexp/ex2 emulation clamps here
comptime C4 = Scalar[DType.float32](
    0.009618129
)  # FlashAttention 4 polynomial coefficient
comptime C3 = Scalar[DType.float32](0.055504108)
comptime C2 = Scalar[DType.float32](0.240179544)
comptime C1 = Scalar[DType.float32](0.69314718)
comptime C0 = Scalar[DType.float32](1.0)


# ===----------------------------------------------------------------------=== #
# Utilities and Helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def _exp2_int(val: Int32) -> Float32:
    return exp2(Float32(val))


@always_inline
def software_emulated_exp[
    use_soft_exp: Bool = True
](x: Scalar[DType.float32],) -> Scalar[DType.float32]:
    # From the FlashAttention 4 implementation.
    comptime if not use_soft_exp:
        return exp(x)
    else:
        if x <= Scalar[DType.float32](-103.0):
            return Scalar[DType.float32](0.0)
        var log2_input = x / LN_2
        # FA4 ex2_emulation clamps the log2 exponent before range reduction.
        # Mojo ldexp returns garbage for exponents below -128.
        log2_input = max(log2_input, LOG2_EXP_MIN)
        var integer_part = Int(log2_input)
        if log2_input < 0.0 and Float32(integer_part) != log2_input:
            integer_part -= 1
        var fractional_part = log2_input - Float32(integer_part)
        var polynomial = fma(
            fma(
                fma(
                    fma(C4, fractional_part, C3),
                    fractional_part,
                    C2,
                ),
                fractional_part,
                C1,
            ),
            fractional_part,
            C0,
        )
        return polynomial * _exp2_int(Int32(integer_part))


struct OnlineSoftmaxState[
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
]:
    var softmax_max_deferred: Scalar[
        DType.float32
    ]  # FlashAttention 4: deferred max, updated only when jump > TAU
    var softmax_max_true: Scalar[
        DType.float32
    ]  # FlashAttention 2: true running max m
    var softmax_denominator_true: Scalar[
        DType.float32
    ]  # FlashAttention 2: running denominator l
    var output_rescale_factor: Scalar[
        DType.float32
    ]  # FlashAttention 4: accumulated deferred rescale; 1.0 when use_conditional_rescale=False

    @always_inline
    def __init__(out self):
        self.softmax_max_deferred = Scalar[DType.float32].MIN_FINITE
        self.softmax_max_true = Scalar[DType.float32].MIN_FINITE
        self.softmax_denominator_true = Scalar[DType.float32](0.0)
        self.output_rescale_factor = Scalar[DType.float32](1.0)

    @always_inline
    def log_sum_exp(self) -> Scalar[DType.float32]:
        return self.softmax_max_true + log(self.softmax_denominator_true)

    @always_inline
    def epilogue_output_scale(self) -> Scalar[DType.float32]:
        comptime if Self.use_conditional_rescale:
            return (
                software_emulated_exp[Self.use_soft_exp](
                    self.softmax_max_deferred - self.softmax_max_true
                )
                * self.output_rescale_factor
            )
        else:
            return Scalar[DType.float32](1.0)


# ===----------------------------------------------------------------------=== #
# Attention Forward
# ===----------------------------------------------------------------------=== #


@always_inline
def _attention_query_key_dot_product[
    dtype: DType,
    width: Int,
](
    query_row: ImmutKernelPtr[dtype],
    key_row: ImmutKernelPtr[dtype],
    head_dim: Int,
    attention_scale: Scalar[DType.float32],
) -> Scalar[DType.float32]:
    var accumulator = SIMD[DType.float32, width](0.0)
    var dimension = 0
    while dimension + width <= head_dim:
        accumulator = fma(
            (query_row + dimension).load[width=width]().cast[DType.float32](),
            (key_row + dimension).load[width=width]().cast[DType.float32](),
            accumulator,
        )
        dimension += width
    var score = accumulator.reduce_add()
    for tail_dimension in range(dimension, head_dim):
        score += (
            query_row[tail_dimension].cast[DType.float32]()
            * key_row[tail_dimension].cast[DType.float32]()
        )
    return score * attention_scale


@always_inline
def _attention_rescale_and_add_value_to_output[
    dtype: DType,
    width: Int,
](
    output_row: MutKernelPtr[dtype],
    value_row: ImmutKernelPtr[dtype],
    rescale_factor: Scalar[DType.float32],
    attention_weight: Scalar[DType.float32],
    head_dim: Int,
) -> None:
    @always_inline
    def _vectorize_head_dim[
        w: Int
    ](local: Int) {
        output_row,
        value_row,
        rescale_factor,
        attention_weight,
        head_dim,
    }:
        var output_vector = (
            (output_row + local).load[width=w]().cast[DType.float32]()
        )
        var value_vector = (
            (value_row + local).load[width=w]().cast[DType.float32]()
        )
        (output_row + local).store[width=w](
            fma(
                output_vector,
                SIMD[DType.float32, w](rescale_factor),
                value_vector * SIMD[DType.float32, w](attention_weight),
            ).cast[dtype]()
        )

    vectorize[width, unroll_factor=UNROLL](head_dim, _vectorize_head_dim)


@always_inline
def _attention_add_weighted_value_to_output[
    dtype: DType,
    width: Int,
](
    output_row: MutKernelPtr[dtype],
    value_row: ImmutKernelPtr[dtype],
    attention_weight: Scalar[DType.float32],
    head_dim: Int,
) -> None:
    @always_inline
    def _vectorize_head_dim[
        w: Int
    ](local: Int) {output_row, value_row, attention_weight, head_dim,}:
        var output_vector = (
            (output_row + local).load[width=w]().cast[DType.float32]()
        )
        var value_vector = (
            (value_row + local).load[width=w]().cast[DType.float32]()
        )
        (output_row + local).store[width=w](
            (
                output_vector
                + value_vector * SIMD[DType.float32, w](attention_weight)
            ).cast[dtype]()
        )

    vectorize[width, unroll_factor=UNROLL](head_dim, _vectorize_head_dim)


@always_inline
def _attention_scale_output_row[
    dtype: DType,
    width: Int,
](
    output_row: MutKernelPtr[dtype],
    scale_factor: Scalar[DType.float32],
    head_dim: Int,
) -> None:
    @always_inline
    def _vectorize_head_dim[
        w: Int
    ](local: Int) {output_row, scale_factor, head_dim}:
        var output_vector = (
            (output_row + local).load[width=w]().cast[DType.float32]()
        )
        (output_row + local).store[width=w](
            (output_vector * SIMD[DType.float32, w](scale_factor)).cast[dtype]()
        )

    vectorize[width, unroll_factor=UNROLL](head_dim, _vectorize_head_dim)


@always_inline
def _attention_update_query_row_for_key[
    dtype: DType,
    width: Int,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    mut state: OnlineSoftmaxState[use_soft_exp, use_conditional_rescale],
    query_row: ImmutKernelPtr[dtype],
    key_row: ImmutKernelPtr[dtype],
    value_row: ImmutKernelPtr[dtype],
    output_row: MutKernelPtr[dtype],
    head_dim: Int,
    attention_scale: Scalar[DType.float32],
) -> None:
    var attention_score = _attention_query_key_dot_product[dtype, width](
        query_row, key_row, head_dim, attention_scale
    )
    _attention_online_softmax_update_row[
        dtype, width, use_soft_exp, use_conditional_rescale
    ](state, attention_score, output_row, value_row, head_dim)


@always_inline
def _attention_online_softmax_update_row[
    dtype: DType,
    width: Int,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    mut state: OnlineSoftmaxState[use_soft_exp, use_conditional_rescale],
    attention_score: Scalar[DType.float32],
    output_row: MutKernelPtr[dtype],
    value_row: ImmutKernelPtr[dtype],
    head_dim: Int,
) -> None:
    # From the FlashAttention 4 implementation.
    var previous_true_max = state.softmax_max_true
    state.softmax_max_true = max(state.softmax_max_true, attention_score)
    var true_rescale = software_emulated_exp[use_soft_exp](
        previous_true_max - state.softmax_max_true
    )
    var true_weight = software_emulated_exp[use_soft_exp](
        attention_score - state.softmax_max_true
    )
    state.softmax_denominator_true = fma(
        true_rescale, state.softmax_denominator_true, true_weight
    )

    comptime if use_conditional_rescale:
        var candidate_max = max(state.softmax_max_deferred, attention_score)
        var max_delta = candidate_max - state.softmax_max_deferred
        if max_delta > TAU:
            var rescale_factor = software_emulated_exp[use_soft_exp](
                state.softmax_max_deferred - candidate_max
            )
            var attention_weight = software_emulated_exp[use_soft_exp](
                attention_score - candidate_max
            )
            _attention_rescale_and_add_value_to_output[dtype, width](
                output_row,
                value_row,
                rescale_factor,
                attention_weight,
                head_dim,
            )
            state.softmax_max_deferred = candidate_max
        else:
            var attention_weight = software_emulated_exp[use_soft_exp](
                attention_score - state.softmax_max_deferred
            )
            _attention_add_weighted_value_to_output[dtype, width](
                output_row, value_row, attention_weight, head_dim
            )
    else:
        var previous_deferred_max = state.softmax_max_deferred
        state.softmax_max_deferred = max(
            state.softmax_max_deferred, attention_score
        )
        var rescale_factor = software_emulated_exp[use_soft_exp](
            previous_deferred_max - state.softmax_max_deferred
        )
        var attention_weight = software_emulated_exp[use_soft_exp](
            attention_score - state.softmax_max_deferred
        )
        _attention_rescale_and_add_value_to_output[dtype, width](
            output_row,
            value_row,
            rescale_factor,
            attention_weight,
            head_dim,
        )


@always_inline
def _attention_finalize_output_row[
    dtype: DType,
    width: Int,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    state: OnlineSoftmaxState[use_soft_exp, use_conditional_rescale],
    output_row: MutKernelPtr[dtype],
    log_sum_exp_out: MutKernelPtr[DType.float32],
    head_dim: Int,
) -> None:
    var inverse_denominator = (
        Scalar[DType.float32](1.0) / state.softmax_denominator_true
    )
    _attention_scale_output_row[dtype, width](
        output_row,
        state.epilogue_output_scale() * inverse_denominator,
        head_dim,
    )
    log_sum_exp_out.store(state.log_sum_exp())


@always_inline
def _attention_forward_query_row[
    dtype: DType,
    width: Int,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    output_ptr: MutKernelPtr[dtype],
    query_ptr: ImmutKernelPtr[dtype],
    key_ptr: ImmutKernelPtr[dtype],
    value_ptr: ImmutKernelPtr[dtype],
    log_sum_exp_ptr: MutKernelPtr[DType.float32],
    head_offset: Int,
    query_index: Int,
    seq_len: Int,
    head_dim: Int,
    attention_scale: Scalar[DType.float32],
) -> None:
    var query_row = query_ptr + head_offset + query_index * head_dim
    var output_row = output_ptr + head_offset + query_index * head_dim
    var log_sum_exp_out = (
        log_sum_exp_ptr
        + (head_offset // (seq_len * head_dim)) * seq_len
        + query_index
    )

    var state = OnlineSoftmaxState[use_soft_exp, use_conditional_rescale]()
    for dimension in range(head_dim):
        output_row[dimension] = Scalar[dtype](0.0)

    for key_index in range(query_index + 1):
        _attention_update_query_row_for_key[
            dtype, width, use_soft_exp, use_conditional_rescale
        ](
            state,
            query_row,
            key_ptr + head_offset + key_index * head_dim,
            value_ptr + head_offset + key_index * head_dim,
            output_row,
            head_dim,
            attention_scale,
        )

    _attention_finalize_output_row[
        dtype, width, use_soft_exp, use_conditional_rescale
    ](state, output_row, log_sum_exp_out, head_dim)


# NOTE: The former `_attention_shared_memory_row_pointer` helper was removed:
# it cast SHARED (threadgroup) pointers to AddressSpace.GENERIC, which Metal
# AIR silently mis-compiles (loads return 0, stores go to device memory).
# GPU kernels now index shared LayoutTensors via SHARED-typed pointers.
# See docs/ai/metal_port_gotchas_and_optimizations.md G3.


@always_inline
def _attention_copy_rows_dram_to_shared[
    dtype: DType,
    BLOCK_SIZE: Int,
](
    mut shared_tensor: LayoutTensor,
    dram_ptr: ImmutKernelPtr[dtype],
    dram_row_start: Int,
    row_count: Int,
    head_dim: Int,
) -> None:
    # Block-strided copy: each thread covers a stride of elements so the CTA
    # collectively fills the shared tile. Uses synchronous stores because the
    # a copy_dram_to_sram_async thread_layout API is not available;
    # callers issue a barrier() after this returns to ensure visibility.
    # Metal fix: rebind to MutAnyOrigin+SHARED so writes go to threadgroup memory.
    # address_space_cast[GENERIC] corrupts threadgroup pointers on Metal AIR.
    var shared_ptr = rebind[
        UnsafePointer[
            Scalar[dtype], MutAnyOrigin, address_space=AddressSpace.SHARED
        ]
    ](shared_tensor.ptr)
    var thread_id = Int(thread_idx.x)
    var element_count = row_count * head_dim
    var element_index = thread_id
    while element_index < element_count:
        var local_row = element_index // head_dim
        var column = element_index % head_dim
        shared_ptr[local_row * MAX_HEAD_DIM + column] = dram_ptr[
            (dram_row_start + local_row) * head_dim + column
        ]
        element_index += BLOCK_SIZE


@always_inline
def _attention_zero_output_tile_rows[
    dtype: DType,
    BLOCK_SIZE: Int,
](
    output_ptr: MutKernelPtr[dtype],
    head_offset: Int,
    query_tile_start: Int,
    query_tile_rows: Int,
    seq_len: Int,
    head_dim: Int,
) -> None:
    var thread_id = Int(thread_idx.x)
    var element_count = query_tile_rows * head_dim
    var element_index = thread_id
    while element_index < element_count:
        var local_row = element_index // head_dim
        var dimension = element_index % head_dim
        if query_tile_start + local_row < seq_len:
            output_ptr[
                head_offset
                + (query_tile_start + local_row) * head_dim
                + dimension
            ] = Scalar[dtype](0.0)
        element_index += BLOCK_SIZE


@always_inline
def _attention_load_online_softmax_state_from_shared[
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    softmax_max_deferred_shared: LayoutTensor,
    softmax_max_true_shared: LayoutTensor,
    softmax_denominator_true_shared: LayoutTensor,
    output_rescale_factor_shared: LayoutTensor,
    local_query_row: Int,
    mut state: OnlineSoftmaxState[use_soft_exp, use_conditional_rescale],
) -> None:
    state.softmax_max_deferred = softmax_max_deferred_shared.ptr[
        local_query_row
    ].cast[DType.float32]()
    state.softmax_max_true = softmax_max_true_shared.ptr[local_query_row].cast[
        DType.float32
    ]()
    state.softmax_denominator_true = softmax_denominator_true_shared.ptr[
        local_query_row
    ].cast[DType.float32]()
    state.output_rescale_factor = output_rescale_factor_shared.ptr[
        local_query_row
    ].cast[DType.float32]()


@always_inline
def _attention_store_online_softmax_state_to_shared[
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    mut softmax_max_deferred_shared: LayoutTensor,
    mut softmax_max_true_shared: LayoutTensor,
    mut softmax_denominator_true_shared: LayoutTensor,
    mut output_rescale_factor_shared: LayoutTensor,
    local_query_row: Int,
    state: OnlineSoftmaxState[use_soft_exp, use_conditional_rescale],
) -> None:
    # Rebind to MutAnyOrigin+SHARED so writes go to threadgroup memory on Metal.
    comptime _F32SharedPtr = UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin, address_space=AddressSpace.SHARED
    ]
    var p_max_def = rebind[_F32SharedPtr](softmax_max_deferred_shared.ptr)
    var p_max_true = rebind[_F32SharedPtr](softmax_max_true_shared.ptr)
    var p_denom = rebind[_F32SharedPtr](softmax_denominator_true_shared.ptr)
    var p_rescale = rebind[_F32SharedPtr](output_rescale_factor_shared.ptr)
    p_max_def[local_query_row] = state.softmax_max_deferred
    p_max_true[local_query_row] = state.softmax_max_true
    p_denom[local_query_row] = state.softmax_denominator_true
    p_rescale[local_query_row] = state.output_rescale_factor


@always_inline
def _attention_init_online_softmax_state_shared[
    Br: Int,
    BLOCK_SIZE: Int,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    mut softmax_max_deferred_shared: LayoutTensor,
    mut softmax_max_true_shared: LayoutTensor,
    mut softmax_denominator_true_shared: LayoutTensor,
    mut output_rescale_factor_shared: LayoutTensor,
    query_tile_start: Int,
    query_tile_rows: Int,
    seq_len: Int,
) -> None:
    # Rebind to MutAnyOrigin+SHARED so writes go to threadgroup memory on Metal.
    comptime _F32SharedPtr = UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin, address_space=AddressSpace.SHARED
    ]
    var p_max_def = rebind[_F32SharedPtr](softmax_max_deferred_shared.ptr)
    var p_max_true = rebind[_F32SharedPtr](softmax_max_true_shared.ptr)
    var p_denom = rebind[_F32SharedPtr](softmax_denominator_true_shared.ptr)
    var p_rescale = rebind[_F32SharedPtr](output_rescale_factor_shared.ptr)
    var thread_id = Int(thread_idx.x)
    var row_index = thread_id
    var initial_state = OnlineSoftmaxState[
        use_soft_exp, use_conditional_rescale
    ]()
    while row_index < query_tile_rows:
        if query_tile_start + row_index < seq_len:
            p_max_def[row_index] = initial_state.softmax_max_deferred
            p_max_true[row_index] = initial_state.softmax_max_true
            p_denom[row_index] = initial_state.softmax_denominator_true
            p_rescale[row_index] = initial_state.output_rescale_factor
        row_index += BLOCK_SIZE


# ===----------------------------------------------------------------------=== #
# Attention Forward — CPU
# ===----------------------------------------------------------------------=== #


@always_inline
def _attention_fwd_cpu[
    dtype: DType,
    width: Int,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    head_index: Int,
    output_ptr: MutKernelPtr[dtype],
    query_ptr: ImmutKernelPtr[dtype],
    key_ptr: ImmutKernelPtr[dtype],
    value_ptr: ImmutKernelPtr[dtype],
    log_sum_exp_ptr: MutKernelPtr[DType.float32],
    seq_len: Int,
    head_dim: Int,
    attention_scale: Scalar[DType.float32],
) -> None:
    var head_offset = head_index * seq_len * head_dim
    for query_index in range(seq_len):
        _attention_forward_query_row[
            dtype, width, use_soft_exp, use_conditional_rescale
        ](
            output_ptr,
            query_ptr,
            key_ptr,
            value_ptr,
            log_sum_exp_ptr,
            head_offset,
            query_index,
            seq_len,
            head_dim,
            attention_scale,
        )


def attention_fwd_cpu[
    dtype: DType,
    width: Int,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    output_ptr: MutKernelPtr[dtype],
    query_ptr: ImmutKernelPtr[dtype],
    key_ptr: ImmutKernelPtr[dtype],
    value_ptr: ImmutKernelPtr[dtype],
    log_sum_exp_ptr: MutKernelPtr[DType.float32],
    batch_size: Int,
    num_heads: Int,
    seq_len: Int,
    head_dim: Int,
) raises -> None:
    var attention_scale = Scalar[DType.float32](1) / sqrt(
        Scalar[DType.float32](head_dim)
    )
    var total_heads = Int(batch_size * num_heads)
    var max_workers = parallelism_level()
    var heads_per_worker = ceildiv(total_heads, max_workers)
    var num_workers = ceildiv(total_heads, heads_per_worker)

    @parameter
    def _worker(worker_index: Int):
        var base = worker_index * heads_per_worker
        var count = min(heads_per_worker, total_heads - base)
        for local in range(count):
            _attention_fwd_cpu[
                dtype, width, use_soft_exp, use_conditional_rescale
            ](
                base + local,
                output_ptr,
                query_ptr,
                key_ptr,
                value_ptr,
                log_sum_exp_ptr,
                seq_len,
                head_dim,
                attention_scale,
            )

    traced_parallelize["attention_fwd", _worker](num_workers)


# ===----------------------------------------------------------------------=== #
# Attention Forward — GPU
# ===----------------------------------------------------------------------=== #


@always_inline
def _attention_gpu_process_key_value_tile_warp[
    dtype: DType,
    width: Int,
    Br: Int,
    Bc: Int,
    BLOCK_SIZE: Int,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    warp_index: Int,
    query_tile_start: Int,
    key_tile_start: Int,
    seq_len: Int,
    head_dim: Int,
    head_offset: Int,
    attention_scale: Scalar[DType.float32],
    output_ptr: MutKernelPtr[dtype],
    query_shared: LayoutTensor,
    key_shared: LayoutTensor,
    value_shared: LayoutTensor,
    mut softmax_max_deferred_shared: LayoutTensor,
    mut softmax_max_true_shared: LayoutTensor,
    mut softmax_denominator_true_shared: LayoutTensor,
    mut output_rescale_factor_shared: LayoutTensor,
    reduction_shared: LayoutTensor,
) -> None:
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE
    comptime ROWS_PER_WARP = Br // NUM_WARPS

    var lane_id = Int(thread_idx.x) % WARP_SIZE
    var warp_offset = warp_index * WARP_SIZE

    comptime ELEMENTS_PER_THREAD = MAX_HEAD_DIM // WARP_SIZE
    var d_start = lane_id * ELEMENTS_PER_THREAD

    comptime for local_row in range(ROWS_PER_WARP):
        var query_index = (
            query_tile_start + warp_index * ROWS_PER_WARP + local_row
        )
        var local_query_row = warp_index * ROWS_PER_WARP + local_row
        var output_row = output_ptr + head_offset + query_index * head_dim
        var state = OnlineSoftmaxState[use_soft_exp, use_conditional_rescale]()
        _attention_load_online_softmax_state_from_shared[
            use_soft_exp, use_conditional_rescale
        ](
            softmax_max_deferred_shared,
            softmax_max_true_shared,
            softmax_denominator_true_shared,
            output_rescale_factor_shared,
            local_query_row,
            state,
        )

        # NOTE: This is a runtime loop (not comptime) on purpose. The Metal GPU
        # compiler (MetalAIRPass) crashes when the inner KV loop is fully
        # unrolled at compile-time inside the already-unrolled outer
        # `comptime for local_row` loop: the resulting function body is too
        # large for the Metal LLVM/AIR backend and causes an ICE. On NVIDIA
        # (HAS_CUBLAS=True) the flash-forward kernel is not called at all
        # (USE_GEMM_ATTENTION=True takes precedence), so removing the unroll
        # has zero effect on NVIDIA forward performance. The backward kernels
        # never used comptime-unrolled KV loops and compile fine on both
        # targets. See docs/ai/metal_port_gotchas_and_optimizations.md G8.
        for key_column in range(Bc):
            var key_index = key_tile_start + key_column
            var is_active_and_valid = (
                (query_index < seq_len)
                and (key_index < seq_len)
                and (key_index <= query_index)
            )

            # Metal fix: access shared Q/K/V rows directly via .ptr to preserve
            # AddressSpace.SHARED — the old _attention_shared_memory_row_pointer
            # path casts to GENERIC which silently returns zeros on Metal AIR.
            var q_base = local_query_row * MAX_HEAD_DIM
            var k_base = key_column * MAX_HEAD_DIM
            var local_sum = Scalar[DType.float32](0.0)
            if is_active_and_valid:
                for d in range(ELEMENTS_PER_THREAD):
                    var idx = d_start + d
                    if idx < head_dim:
                        local_sum += (
                            query_shared.ptr[q_base + idx].cast[DType.float32]()
                            * key_shared.ptr[k_base + idx].cast[DType.float32]()
                        )

            var S_ij = warp.sum(local_sum) * attention_scale

            if is_active_and_valid:
                var previous_true_max = state.softmax_max_true
                state.softmax_max_true = max(state.softmax_max_true, S_ij)
                var true_rescale = software_emulated_exp[use_soft_exp](
                    previous_true_max - state.softmax_max_true
                )
                var true_weight = software_emulated_exp[use_soft_exp](
                    S_ij - state.softmax_max_true
                )
                state.softmax_denominator_true = fma(
                    true_rescale, state.softmax_denominator_true, true_weight
                )

                var v_base = key_column * MAX_HEAD_DIM

                comptime if use_conditional_rescale:
                    var candidate_max = max(state.softmax_max_deferred, S_ij)
                    var max_delta = candidate_max - state.softmax_max_deferred
                    if max_delta > TAU:
                        var rescale_factor = software_emulated_exp[
                            use_soft_exp
                        ](state.softmax_max_deferred - candidate_max)
                        var attention_weight = software_emulated_exp[
                            use_soft_exp
                        ](S_ij - candidate_max)
                        for d in range(ELEMENTS_PER_THREAD):
                            var idx = d_start + d
                            if idx < head_dim:
                                var v_val = value_shared.ptr[v_base + idx]
                                var out_val = output_row[idx]
                                output_row[idx] = (
                                    rescale_factor
                                    * out_val.cast[DType.float32]()
                                    + attention_weight
                                    * v_val.cast[DType.float32]()
                                ).cast[dtype]()
                        state.softmax_max_deferred = candidate_max
                    else:
                        var attention_weight = software_emulated_exp[
                            use_soft_exp
                        ](S_ij - state.softmax_max_deferred)
                        for d in range(ELEMENTS_PER_THREAD):
                            var idx = d_start + d
                            if idx < head_dim:
                                var v_val = value_shared.ptr[v_base + idx]
                                var out_val = output_row[idx]
                                output_row[idx] = (
                                    out_val.cast[DType.float32]()
                                    + attention_weight
                                    * v_val.cast[DType.float32]()
                                ).cast[dtype]()
                else:
                    var candidate_max = max(state.softmax_max_deferred, S_ij)
                    var rescale_factor = software_emulated_exp[use_soft_exp](
                        state.softmax_max_deferred - candidate_max
                    )
                    var attention_weight = software_emulated_exp[use_soft_exp](
                        S_ij - candidate_max
                    )
                    for d in range(ELEMENTS_PER_THREAD):
                        var idx = d_start + d
                        if idx < head_dim:
                            var v_val = value_shared.ptr[v_base + idx]
                            var out_val = output_row[idx]
                            output_row[idx] = (
                                rescale_factor * out_val.cast[DType.float32]()
                                + attention_weight * v_val.cast[DType.float32]()
                            ).cast[dtype]()
                    state.softmax_max_deferred = candidate_max

        # NOTE: No intra-tile barrier here. Each warp owns a disjoint set of
        # query rows, reads only the (read-only) shared Q/K/V tiles, and writes
        # only its own DRAM output rows and its own shared softmax-state slots.
        # There is no inter-warp dependency inside this routine; KV-tile reload
        # safety is enforced by the caller's barriers around the shared K/V copy.
        if lane_id == 0:
            _attention_store_online_softmax_state_to_shared[
                use_soft_exp, use_conditional_rescale
            ](
                softmax_max_deferred_shared,
                softmax_max_true_shared,
                softmax_denominator_true_shared,
                output_rescale_factor_shared,
                local_query_row,
                state,
            )


@always_inline
def _attention_gpu_finalize_query_rows_warp[
    dtype: DType,
    width: Int,
    Br: Int,
    BLOCK_SIZE: Int,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    warp_index: Int,
    query_tile_start: Int,
    seq_len: Int,
    head_dim: Int,
    head_offset: Int,
    head_index: Int,
    output_ptr: MutKernelPtr[dtype],
    log_sum_exp_ptr: MutKernelPtr[DType.float32],
    softmax_max_deferred_shared: LayoutTensor,
    softmax_max_true_shared: LayoutTensor,
    softmax_denominator_true_shared: LayoutTensor,
    output_rescale_factor_shared: LayoutTensor,
) -> None:
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE
    comptime ROWS_PER_WARP = Br // NUM_WARPS

    var lane_id = Int(thread_idx.x) % WARP_SIZE
    comptime ELEMENTS_PER_THREAD = MAX_HEAD_DIM // WARP_SIZE
    var d_start = lane_id * ELEMENTS_PER_THREAD

    comptime for local_row in range(ROWS_PER_WARP):
        var query_index = (
            query_tile_start + warp_index * ROWS_PER_WARP + local_row
        )
        var local_query_row = warp_index * ROWS_PER_WARP + local_row
        var state = OnlineSoftmaxState[use_soft_exp, use_conditional_rescale]()
        _attention_load_online_softmax_state_from_shared[
            use_soft_exp, use_conditional_rescale
        ](
            softmax_max_deferred_shared,
            softmax_max_true_shared,
            softmax_denominator_true_shared,
            output_rescale_factor_shared,
            local_query_row,
            state,
        )
        var output_row = output_ptr + head_offset + query_index * head_dim
        var log_sum_exp_out = (
            log_sum_exp_ptr + head_index * seq_len + query_index
        )

        var scale = (
            state.epilogue_output_scale() / state.softmax_denominator_true
        )
        if query_index < seq_len:
            for d in range(ELEMENTS_PER_THREAD):
                var idx = d_start + d
                if idx < head_dim:
                    var out_val = output_row[idx]
                    output_row[idx] = (
                        out_val.cast[DType.float32]() * scale
                    ).cast[dtype]()

            if lane_id == 0:
                log_sum_exp_out.store(state.log_sum_exp())
        barrier()


@always_inline
def _attention_gpu_forward_query_tile_block[
    dtype: DType,
    width: Int,
    Br: Int,
    Bc: Int,
    BLOCK_SIZE: Int,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    tile_index: Int,
    output_ptr: MutKernelPtr[dtype],
    query_ptr: ImmutKernelPtr[dtype],
    key_ptr: ImmutKernelPtr[dtype],
    value_ptr: ImmutKernelPtr[dtype],
    log_sum_exp_ptr: MutKernelPtr[DType.float32],
    seq_len: Int,
    head_dim: Int,
    query_tiles: Int,
) -> None:
    var head_index = tile_index // query_tiles
    var query_tile_index = tile_index % query_tiles
    var query_tile_start = query_tile_index * Br
    var query_tile_rows = min(Br, seq_len - query_tile_start)
    var head_offset = head_index * seq_len * head_dim
    var attention_scale = Scalar[DType.float32](1) / sqrt(
        Scalar[DType.float32](head_dim)
    )

    var query_head = query_ptr + head_offset
    var key_head = key_ptr + head_offset
    var value_head = value_ptr + head_offset

    var query_shared = LayoutTensor[
        dtype,
        Layout.row_major(Br, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var key_shared = LayoutTensor[
        dtype,
        Layout.row_major(Bc, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var value_shared = LayoutTensor[
        dtype,
        Layout.row_major(Bc, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var softmax_max_deferred_shared = LayoutTensor[
        DType.float32,
        Layout.row_major(Br),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var softmax_max_true_shared = LayoutTensor[
        DType.float32,
        Layout.row_major(Br),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var softmax_denominator_true_shared = LayoutTensor[
        DType.float32,
        Layout.row_major(Br),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var output_rescale_factor_shared = LayoutTensor[
        DType.float32,
        Layout.row_major(Br),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var reduction_shared = LayoutTensor[
        DType.float32,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # Metal fix: pass the LayoutTensors directly to helper functions instead of
    # casting to GENERIC pointers. The old pattern corrupted threadgroup pointers
    # on Metal AIR; using .ptr[i] on the LayoutTensor preserves AddressSpace.SHARED.

    # Load this CTA's query tile once; it is reused across every KV tile.
    _attention_copy_rows_dram_to_shared[dtype, BLOCK_SIZE](
        query_shared, query_head, query_tile_start, query_tile_rows, head_dim
    )
    barrier()
    _attention_zero_output_tile_rows[dtype, BLOCK_SIZE](
        output_ptr,
        head_offset,
        query_tile_start,
        query_tile_rows,
        seq_len,
        head_dim,
    )
    _attention_init_online_softmax_state_shared[
        Br, BLOCK_SIZE, use_soft_exp, use_conditional_rescale
    ](
        softmax_max_deferred_shared,
        softmax_max_true_shared,
        softmax_denominator_true_shared,
        output_rescale_factor_shared,
        query_tile_start,
        query_tile_rows,
        seq_len,
    )
    barrier()

    for key_tile_start in range(0, seq_len, Bc):
        var key_tile_rows = min(Bc, seq_len - key_tile_start)
        _attention_copy_rows_dram_to_shared[dtype, BLOCK_SIZE](
            key_shared, key_head, key_tile_start, key_tile_rows, head_dim
        )
        _attention_copy_rows_dram_to_shared[dtype, BLOCK_SIZE](
            value_shared, value_head, key_tile_start, key_tile_rows, head_dim
        )
        barrier()

        _attention_gpu_process_key_value_tile_warp[
            dtype,
            width,
            Br,
            Bc,
            BLOCK_SIZE,
            use_soft_exp,
            use_conditional_rescale,
        ](
            Int(thread_idx.x) // WARP_SIZE,
            query_tile_start,
            key_tile_start,
            seq_len,
            head_dim,
            head_offset,
            attention_scale,
            output_ptr,
            query_shared,
            key_shared,
            value_shared,
            softmax_max_deferred_shared,
            softmax_max_true_shared,
            softmax_denominator_true_shared,
            output_rescale_factor_shared,
            reduction_shared,
        )
        barrier()

    _attention_gpu_finalize_query_rows_warp[
        dtype, width, Br, BLOCK_SIZE, use_soft_exp, use_conditional_rescale
    ](
        Int(thread_idx.x) // WARP_SIZE,
        query_tile_start,
        seq_len,
        head_dim,
        head_offset,
        head_index,
        output_ptr,
        log_sum_exp_ptr,
        softmax_max_deferred_shared,
        softmax_max_true_shared,
        softmax_denominator_true_shared,
        output_rescale_factor_shared,
    )


def attention_fwd_gpu[
    dtype: DType,
    width: Int,
    Br: Int,
    Bc: Int,
    BLOCK_SIZE: Int,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    num_tiles: Int,
    query_tiles: Int,
    output_ptr: MutKernelPtr[dtype],
    query_ptr: ImmutKernelPtr[dtype],
    key_ptr: ImmutKernelPtr[dtype],
    value_ptr: ImmutKernelPtr[dtype],
    log_sum_exp_ptr: MutKernelPtr[DType.float32],
    seq_len: Int64,
    head_dim: Int64,
) -> None:
    var grid_stride = Int(grid_dim.x)
    var block_tile = Int(block_idx.x)

    for tile_index in range(block_tile, num_tiles, grid_stride):
        _attention_gpu_forward_query_tile_block[
            dtype,
            width,
            Br,
            Bc,
            BLOCK_SIZE,
            use_soft_exp,
            use_conditional_rescale,
        ](
            tile_index,
            output_ptr,
            query_ptr,
            key_ptr,
            value_ptr,
            log_sum_exp_ptr,
            Int(seq_len),
            Int(head_dim),
            query_tiles,
        )


# ===----------------------------------------------------------------------=== #
# Attention Forward — GPU, tensor-core GEMM path
# ===----------------------------------------------------------------------=== #
#
# The flash kernels above run the O(T²·d) QKᵀ and A·V math on CUDA cores at 0%
# tensor-core utilization, which dominates the step at long T (see
# docs/benchmarks.md). This path instead decomposes attention the way llm.c does:
#   1. QKᵀ as a per-head tensor-core GEMM (bf16 in, fp32 scores out, scaled).
#   2. a dedicated causal softmax that also emits log-sum-exp (for the existing
#      flash backward, which is left untouched).
#   3. A·V as a per-head tensor-core GEMM (bf16 probabilities × bf16 V).
# It materializes the [B·nh, T, T] scores/probabilities, trading memory for
# tensor-core throughput. Toggle with USE_GEMM_ATTENTION. This path is built on
# cuBLAS(Lt) batched GEMMs (see _attn_gemm_batched/attention_bwd_gemm below), so
# it is enabled on NVIDIA (HAS_CUBLAS) and Apple Silicon (HAS_METAL): on Metal,
# scalar dot-product loops use zero GPU matrix-unit throughput while linalg.matmul
# dispatches to Metal's matrix shaders — 8–10× faster (see P11). Other GPU targets
# (AMD, portable) fall back to the flash kernels (vendor-neutral, no FFI).
# Independently overridable on NVIDIA with LLMM_FORCE_PORTABLE_GPU=1 (forces both
# off; USE_GEMM_ATTENTION derives from HAS_CUBLAS when Metal is absent).
# See docs/ai/metal_port_gotchas_and_optimizations.md P11.

comptime USE_GEMM_ATTENTION = HAS_CUBLAS or HAS_METAL

# Softmax-forward vectorized-load A/B (item 3): width=1 is the original
# scalar-per-thread-per-iteration load (bit-identical baseline); True switches
# to simd_width_of[dtype]()-wide chunked loads (llm.c kernel5's 4-wide
# `regarray` pattern). See docs/benchmarks.md for the measured verdict.
comptime USE_SOFTMAX_VEC_LOADS = True


@always_inline
def _attention_softmax_causal_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 1,
](
    num_rows: Int,
    seq_len: Int,
    attention_scale: Scalar[DType.float32],
    scores_ptr: ImmutKernelPtr[dtype],
    att_ptr: MutKernelPtr[dtype],
    lse_ptr: MutKernelPtr[DType.float32],
) -> None:
    # One block per scores row (grid-strided). Row r belongs to head r // T and
    # query position i = r % T, so the causal prefix is columns [0, i]. The raw
    # QKᵀ dot products are scaled here (kept out of the GEMM epilogue so Q/K stay
    # unscaled for the backward pass). lse = m + log(sum exp(s - m)) matches the
    # flash kernel's definition exactly.
    #
    # Both passes read the row in `width`-wide chunks per thread (the tile
    # pattern used in layernorm's fused-residual kernel:
    # `lane_base = tile_base + tid*width`, scalar tail for the remainder) —
    # llm.c's softmax_forward_kernel5 similarly reads via a `regarray[4]`
    # vector load. The causal prefix length (`valid = query_index + 1`) isn't
    # width-aligned in general, so each tile falls back to a per-element
    # scalar loop once `lane_base + width > valid`.
    comptime BLOCK_SPAN = BLOCK_SIZE * width
    comptime align = align_of[SIMD[dtype, width]]()
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x)
    var block_row = Int(block_idx.x)
    for row in range(block_row, num_rows, stride):
        var query_index = row % seq_len
        var valid = query_index + 1
        var base = row * seq_len

        var m_thread = Scalar[DType.float32].MIN_FINITE
        var s_thread = Scalar[DType.float32](0.0)
        for tile_base in range(0, valid, BLOCK_SPAN):
            var lane_base = tile_base + tid * width
            if lane_base + width <= valid:
                var xv = (scores_ptr + base + lane_base).load[
                    width=width, alignment=align
                ]().cast[DType.float32]() * attention_scale

                comptime for i in range(width):
                    var x = xv[i]
                    var new_m = max(m_thread, x)
                    s_thread = s_thread * exp(m_thread - new_m) + exp(x - new_m)
                    m_thread = new_m
            elif lane_base < valid:
                for c in range(lane_base, valid):
                    var x = (
                        scores_ptr[base + c].cast[DType.float32]()
                        * attention_scale
                    )
                    var new_m = max(m_thread, x)
                    s_thread = s_thread * exp(m_thread - new_m) + exp(x - new_m)
                    m_thread = new_m

        var m_row = block.max[block_size=BLOCK_SIZE](m_thread)
        s_thread = s_thread * exp(m_thread - m_row)
        var s_row = block.sum[block_size=BLOCK_SIZE](s_thread)
        var inv_s = Scalar[DType.float32](1.0) / s_row

        if tid == 0:
            lse_ptr[row] = m_row + log(s_row)

        # Write only the causal prefix [0, valid); the above-diagonal half of
        # att_ptr is left at its persistent zero (memset once on allocation),
        # which halves this store pass.
        var m_row_vec = SIMD[DType.float32, width](m_row)
        var inv_s_vec = SIMD[DType.float32, width](inv_s)
        for tile_base in range(0, valid, BLOCK_SPAN):
            var lane_base = tile_base + tid * width
            if lane_base + width <= valid:
                var idx = base + lane_base
                var xv = (scores_ptr + idx).load[
                    width=width, alignment=align
                ]().cast[DType.float32]() * attention_scale
                var p = exp(xv - m_row_vec) * inv_s_vec
                (att_ptr + idx).store[alignment=align](p.cast[dtype]())
            elif lane_base < valid:
                for c2 in range(lane_base, valid):
                    var idx = base + c2
                    var p = (
                        exp(
                            scores_ptr[idx].cast[DType.float32]()
                            * attention_scale
                            - m_row
                        )
                        * inv_s
                    )
                    att_ptr[idx] = p.cast[dtype]()


# Integrated, validated, but DISABLED: the fused flash forward below is correct
# (bf16 loss matches the GEMM forward within noise) but, at BR=16/BC=16 with one
# warp per query-tile, it runs the forward at ~0.16 s vs the GEMM forward's
# ~0.075 s — poor MMA occupancy outweighs the saved [B·nh,T,T] bandwidth passes.
# Beating the tuned GEMM needs bigger tiles (BR/BC=64) + multi-warp + occupancy
# tuning. Kept wired behind the flag for that future work; off so the faster GEMM
# forward ships.
comptime USE_FLASH_FWD = False


@always_inline
def _flash_sh[
    dt: DType, R: Int, C: Int
]() -> LayoutTensor[
    dt, Layout.row_major(R, C), MutAnyOrigin, address_space=AddressSpace.SHARED
]:
    return LayoutTensor[
        dt,
        Layout.row_major(R, C),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()


@always_inline
def _attention_flash_fwd_gpu[
    dtype: DType,
    BR: Int,
    BC: Int,
    HEAD_DIM: Int,
](
    num_tiles: Int,
    num_query_tiles: Int,
    seq_len: Int,
    attention_scale: Scalar[DType.float32],
    q_ptr: ImmutKernelPtr[dtype],
    k_ptr: ImmutKernelPtr[dtype],
    v_ptr: ImmutKernelPtr[dtype],
    o_ptr: MutKernelPtr[dtype],
    lse_ptr: MutKernelPtr[DType.float32],
) -> None:
    # One warp per (head, query-tile of BR rows). Streams causal KV tiles of BC
    # with online-softmax rescale; QKᵀ and P·V on tensor cores (P·V over Vᵀ),
    # softmax in shared. Produces per-head attn + lse (the GEMM backward consumes
    # these unchanged). Validated bit-accurately in scratch/test_flash.mojo.
    var q_sh = _flash_sh[dtype, BR, HEAD_DIM]()
    var k_sh = _flash_sh[dtype, BC, HEAD_DIM]()
    var v_sh = _flash_sh[dtype, BC, HEAD_DIM]()
    var vt_sh = _flash_sh[dtype, HEAD_DIM, BC]()
    var s_sh = _flash_sh[DType.float32, BR, BC]()
    var p_sh = _flash_sh[dtype, BR, BC]()
    var o_sh = _flash_sh[DType.float32, BR, HEAD_DIM]()
    var pv_sh = _flash_sh[DType.float32, BR, HEAD_DIM]()
    var m_sh = _flash_sh[DType.float32, 1, BR]()
    var l_sh = _flash_sh[DType.float32, 1, BR]()

    var qk = TensorCore[
        DType.float32, dtype, Index(16, 8, 16), transpose_b=True
    ]()
    var pv = TensorCore[
        DType.float32, dtype, Index(16, 8, 16), transpose_b=True
    ]()
    comptime NWARPS = BR // 16
    comptime NTHREADS = NWARPS * WARP_SIZE
    var t = Int(thread_idx.x)
    var warp_id = t // WARP_SIZE  # each warp owns rows [warp_id*16, +16]

    for tile in range(Int(block_idx.x), num_tiles, Int(grid_dim.x)):
        var bh = tile // num_query_tiles
        var query_start = (tile % num_query_tiles) * BR
        var head_off = bh * seq_len * HEAD_DIM
        var qoff = head_off + query_start * HEAD_DIM

        var e = t
        while e < BR * HEAD_DIM:
            var r = e // HEAD_DIM
            q_sh.ptr[e] = q_ptr[
                qoff + e
            ] if query_start + r < seq_len else Scalar[dtype](0)
            o_sh.ptr[e] = 0.0
            e += NTHREADS
        if t < BR:
            m_sh.ptr[t] = Scalar[DType.float32].MIN_FINITE
            l_sh.ptr[t] = 0.0
        barrier()

        var last_key = min(query_start + BR, seq_len)
        for kt in range(0, last_key, BC):
            var e2 = t
            while e2 < BC * HEAD_DIM:
                var kr = e2 // HEAD_DIM
                var ok = kt + kr < seq_len
                k_sh.ptr[e2] = k_ptr[
                    head_off + kt * HEAD_DIM + e2
                ] if ok else Scalar[dtype](0)
                v_sh.ptr[e2] = v_ptr[
                    head_off + kt * HEAD_DIM + e2
                ] if ok else Scalar[dtype](0)
                e2 += NTHREADS
            barrier()
            e2 = t
            while e2 < BC * HEAD_DIM:
                vt_sh.ptr[(e2 % HEAD_DIM) * BC + (e2 // HEAD_DIM)] = v_sh.ptr[
                    e2
                ]
                e2 += NTHREADS
            barrier()

            comptime for n in range(BC // 8):
                var c0 = qk.c_reg_tile_type.stack_allocation().fill(0.0)
                comptime for kk in range(HEAD_DIM // 16):
                    var qa = qk.load_a(q_sh.tile[16, 16](warp_id, kk))
                    var kb = qk.load_b(k_sh.tile[8, 16](n, kk))
                    c0 = qk.mma_op(qa, kb, c0)
                qk.store_d(s_sh.tile[16, 8](warp_id, n), c0)
            barrier()

            if t < BR:
                var qi = query_start + t
                var srow = s_sh.ptr + t * BC
                var prow = p_sh.ptr + t * BC
                var tile_max = Scalar[DType.float32].MIN_FINITE
                for j in range(BC):
                    var kj = kt + j
                    if kj > qi or kj >= seq_len:
                        srow[j] = Scalar[DType.float32].MIN_FINITE
                    else:
                        srow[j] = srow[j] * attention_scale
                    tile_max = max(tile_max, srow[j])
                var m_old = m_sh.ptr[t]
                var m_new = max(m_old, tile_max)
                var rescale = software_emulated_exp[True](m_old - m_new)
                var tile_sum = Scalar[DType.float32](0.0)
                for j in range(BC):
                    var p = software_emulated_exp[True](srow[j] - m_new)
                    tile_sum += p
                    prow[j] = p.cast[dtype]()
                l_sh.ptr[t] = l_sh.ptr[t] * rescale + tile_sum
                m_sh.ptr[t] = m_new
                var orow = o_sh.ptr + t * HEAD_DIM
                for d in range(HEAD_DIM):
                    orow[d] = orow[d] * rescale
            barrier()

            # O += P·V over Vᵀ. P is [BR,BC], so loop the BC dimension in 16-wide
            # k-tiles (BC//16 dense MMAs per n) — this is what makes the big-tile
            # version keep the tensor cores busy vs one MMA per n at BC=16.
            comptime for n in range(HEAD_DIM // 8):
                var oc = pv.c_reg_tile_type.stack_allocation().fill(0.0)
                comptime for kk in range(BC // 16):
                    var pa = pv.load_a(p_sh.tile[16, 16](warp_id, kk))
                    var vb = pv.load_b(vt_sh.tile[8, 16](n, kk))
                    oc = pv.mma_op(pa, vb, oc)
                pv.store_d(pv_sh.tile[16, 8](warp_id, n), oc)
            barrier()
            if t < BR:
                var orow = o_sh.ptr + t * HEAD_DIM
                var pvrow = pv_sh.ptr + t * HEAD_DIM
                for d in range(HEAD_DIM):
                    orow[d] += pvrow[d]
            barrier()

        if t < BR and query_start + t < seq_len:
            var inv = Scalar[DType.float32](1.0) / l_sh.ptr[t]
            var orow = o_sh.ptr + t * HEAD_DIM
            for d in range(HEAD_DIM):
                o_ptr[qoff + t * HEAD_DIM + d] = (orow[d] * inv).cast[dtype]()
            lse_ptr[bh * seq_len + query_start + t] = m_sh.ptr[t] + log(
                l_sh.ptr[t]
            )
        barrier()


def attention_fwd_gemm[
    dtype: DType,
    target: StaticString,
](
    output_ptr: MutKernelPtr[dtype],
    query_ptr: ImmutKernelPtr[dtype],
    key_ptr: ImmutKernelPtr[dtype],
    value_ptr: ImmutKernelPtr[dtype],
    log_sum_exp_ptr: MutKernelPtr[DType.float32],
    batch_size: Int64,
    num_heads: Int64,
    seq_len: Int64,
    head_dim: Int64,
    ctx: DeviceContext,
    cache: Optional[KVCachePtr] = None,
) capturing raises:
    var B = Int(batch_size)
    var NH = Int(num_heads)
    var T = Int(seq_len)
    var hd = Int(head_dim)
    var BH = B * NH
    var attention_scale = Scalar[DType.float32](1.0) / sqrt(
        Scalar[DType.float32](hd)
    )
    var device_ctx = ctx

    # Fused flash forward (bf16, head_dim=64): one warp per (head, query-tile),
    # streams causal KV tiles with online softmax — no [B·nh,T,T] materialization.
    # Produces the same per-head attn + lse the GEMM backward already consumes.
    comptime if USE_FLASH_FWD and dtype == DType.bfloat16:
        if hd == 64:
            comptime BR = 64
            comptime BC = 64
            comptime SM_OVERPROVISION_F = 32
            var num_query_tiles = ceildiv(T, BR)
            var num_tiles_f = BH * num_query_tiles
            var num_sm_f = device_ctx.get_attribute(
                DeviceAttribute.MULTIPROCESSOR_COUNT
            )
            var num_blocks_f = max(
                min(num_tiles_f, SM_OVERPROVISION_F * num_sm_f), 1
            )
            comptime flash_kernel = _attention_flash_fwd_gpu[dtype, BR, BC, 64]
            var flash_compiled = device_ctx.compile_function[flash_kernel]()
            device_ctx.enqueue_function(
                flash_compiled,
                num_tiles_f,
                num_query_tiles,
                T,
                attention_scale,
                query_ptr,
                key_ptr,
                value_ptr,
                output_ptr,
                log_sum_exp_ptr,
                grid_dim=(num_blocks_f,),
                block_dim=(BR // 16 * 32,),
            )
            device_ctx.synchronize()
            return

    # Scratch: bf16 scores and bf16 probabilities, one [T, T] plane per head.
    # scores feed only the softmax's max/sum and exp, so bf16 halves that pass
    # and matches llm.c's bf16 attention-matrix precision. Allocated once and
    # kept alive across layers/steps via the KVCache; one shared pair is safe
    # because each layer fully consumes the scratch in stream order.
    comptime BufType = type_of(device_ctx.enqueue_create_buffer[dtype](1))

    var scores_addr = 0
    var att_addr = 0
    if cache:
        scores_addr = cache.value()[].gemm_scores_addr
        att_addr = cache.value()[].gemm_att_addr

    if scores_addr == 0:
        var scores_buf = device_ctx.enqueue_create_buffer[dtype](BH * T * T)
        var att_buf = device_ctx.enqueue_create_buffer[dtype](BH * T * T)
        # Zero the probability buffer's above-diagonal half once: the causal
        # softmax then writes only the lower triangle each step (the upper stays
        # structurally zero), and A·V / the backward read it as zero. Halves the
        # softmax's store traffic.
        device_ctx.enqueue_memset(att_buf, Scalar[dtype](0))
        var sptr = alloc[BufType](1)
        sptr.unsafe_write(scores_buf^)
        var aptr = alloc[BufType](1)
        aptr.unsafe_write(att_buf^)
        scores_addr = Int(sptr)
        att_addr = Int(aptr)
        if cache:
            cache.value()[].gemm_scores_addr = scores_addr
            cache.value()[].gemm_att_addr = att_addr

    var scores_buf_ptr = UnsafePointer[
        mut=True, type=BufType, origin=MutUntrackedOrigin
    ](unsafe_from_address=scores_addr)
    var att_buf_ptr = UnsafePointer[
        mut=True, type=BufType, origin=MutUntrackedOrigin
    ](unsafe_from_address=att_addr)
    var scores_base = rebind[MutKernelPtr[dtype]](scores_buf_ptr[].unsafe_ptr())
    var scores_immut = rebind[ImmutKernelPtr[dtype]](
        scores_buf_ptr[].unsafe_ptr()
    )
    var att_base = rebind[MutKernelPtr[dtype]](att_buf_ptr[].unsafe_ptr())
    # ⑳-storage: if a per-layer att_probs buffer is provided, write the softmax
    # probs into this layer's slice so the backward can read them (no recompute).
    if cache and cache.value()[].att_probs_addr != 0:
        var c = cache.value()
        var ap = UnsafePointer[
            mut=True, type=Scalar[dtype], origin=MutUntrackedOrigin
        ](unsafe_from_address=c[].att_probs_addr)
        att_base = rebind[MutKernelPtr[dtype]](
            ap + c[].att_probs_layer * c[].att_probs_stride
        )

    # Step 1: QKᵀ for all heads in one cublasGemmStridedBatchedEx call. bf16 Q/K
    # in, bf16 scores out; the scale is applied in the softmax so Q/K stay
    # unscaled for backward. Batching all heads lifts the tensor-core utilization
    # the per-head (head_dim=64) launches left on the table.
    _attention_bmm_scoreout[dtype, dtype, target](
        query_ptr, key_ptr, scores_base, BH, T, hd, device_ctx
    )
    # No fence needed: cuBLAS (via the vendor handle bound to ctx's stream) and
    # the softmax kernel run on the *same* stream, so the softmax's read of the
    # scores is already ordered after QKᵀ.

    # Step 2: causal softmax over scores rows + log-sum-exp. (256 threads/block
    # measured best — 512 lost occupancy.)
    comptime BLOCK_SIZE = 256
    comptime SM_OVERPROVISION = 32
    var num_rows = BH * T
    var num_sm = device_ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var num_blocks = max(min(num_rows, SM_OVERPROVISION * num_sm), 1)
    comptime softmax_width = simd_width_of[
        dtype
    ]() if USE_SOFTMAX_VEC_LOADS else 1
    # The vectorized kernel's per-row base offset is `row * T` elements; its
    # `width`-wide loads/stores are only guaranteed aligned when that offset is
    # always a multiple of `width`, i.e. when T itself is a multiple of
    # `width`. Training always calls with T=max_seq_len (1024, a multiple of
    # 8) so this held for the whole 19,552-step run — but single-token-at-a-
    # time generation walks T through every value 1..genT-1, and the first
    # non-multiple-of-8 T (e.g. T=9) drifts `row * T` out of the required
    # 16-byte alignment for most rows, faulting the hardware vector load with
    # CUDA_ERROR_MISALIGNED_ADDRESS. Compile both kernel variants and pick at
    # runtime: this leaves training's fast path byte-for-byte unchanged (T is
    # always width-aligned there) while generation's non-aligned T falls back
    # to the scalar (width=1) kernel, which has no such alignment assumption.
    if T % softmax_width == 0:
        comptime softmax_kernel = _attention_softmax_causal_gpu[
            dtype, BLOCK_SIZE, softmax_width
        ]
        var compiled = device_ctx.compile_function[softmax_kernel]()
        device_ctx.enqueue_function(
            compiled,
            num_rows,
            T,
            attention_scale,
            scores_immut,
            att_base,
            log_sum_exp_ptr,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        comptime softmax_kernel_scalar = _attention_softmax_causal_gpu[
            dtype, BLOCK_SIZE, 1
        ]
        var compiled_scalar = device_ctx.compile_function[
            softmax_kernel_scalar
        ]()
        device_ctx.enqueue_function(
            compiled_scalar,
            num_rows,
            T,
            attention_scale,
            scores_immut,
            att_base,
            log_sum_exp_ptr,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )

    # Step 3: A·V for all heads in one batched call. bf16 probabilities × bf16 V
    # -> bf16 attention output. Same-stream: no fence before it (A·V's read of the
    # probabilities is ordered after the softmax's write).
    var att_immut = rebind[ImmutKernelPtr[dtype]](att_base)
    _attention_bmm_headout[dtype, target](
        att_immut, value_ptr, output_ptr, BH, T, hd, device_ctx
    )


def attention_fwd[
    dtype: DType,
    target: StaticString,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
    use_kv_cache: Bool = True,
](
    qkv_ptr: ImmutKernelPtr[dtype],
    q_ptr: MutKernelPtr[dtype],
    k_ptr: MutKernelPtr[dtype],
    v_ptr: MutKernelPtr[dtype],
    attn_ptr: MutKernelPtr[dtype],
    attn_merged_ptr: MutKernelPtr[dtype],
    log_sum_exp_ptr: MutKernelPtr[DType.float32],
    batch_size: Int64,
    num_heads: Int64,
    seq_len: Int64,
    head_dim: Int64,
    ctx: DeviceContext,
    cache: Optional[KVCachePtr] = None,
) capturing raises:
    var qkv_mut = rebind[MutKernelPtr[dtype]](qkv_ptr)
    var dst_ptrs = List[MutKernelPtr[dtype]]()
    dst_ptrs.append(q_ptr)
    dst_ptrs.append(k_ptr)
    dst_ptrs.append(v_ptr)
    split_fwd[dtype, target, num_splits=3](
        qkv_mut,
        dst_ptrs,
        Int(batch_size),
        Int(seq_len),
        Int(num_heads),
        Int(head_dim),
        ctx,
    )
    attention_fwd[
        dtype, target, use_soft_exp, use_conditional_rescale, use_kv_cache
    ](
        attn_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        log_sum_exp_ptr,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        ctx,
        cache=cache,
    )
    merge_fwd[dtype, target](
        attn_ptr,
        attn_merged_ptr,
        Int(batch_size),
        Int(seq_len),
        Int(num_heads),
        Int(head_dim),
        ctx,
    )


def attention_fwd[
    dtype: DType,
    target: StaticString,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
    use_kv_cache: Bool = True,
](
    output_ptr: MutKernelPtr[dtype],
    query_ptr: ImmutKernelPtr[dtype],
    key_ptr: ImmutKernelPtr[dtype],
    value_ptr: ImmutKernelPtr[dtype],
    log_sum_exp_ptr: MutKernelPtr[DType.float32],
    batch_size: Int64,
    num_heads: Int64,
    seq_len: Int64,
    head_dim: Int64,
    ctx: DeviceContext,
    cache: Optional[KVCachePtr] = None,
) capturing raises:
    comptime if is_cpu[target]():
        comptime simd_width = simd_width_of[dtype]()
        attention_fwd_cpu[
            dtype, simd_width, use_soft_exp, use_conditional_rescale
        ](
            output_ptr,
            query_ptr,
            key_ptr,
            value_ptr,
            log_sum_exp_ptr,
            Int(batch_size),
            Int(num_heads),
            Int(seq_len),
            Int(head_dim),
        )
    elif is_gpu[target]():
        if Int(head_dim) > MAX_HEAD_DIM:
            raise Error("head_dim exceeds MAX_HEAD_DIM for GPU attention")
        comptime if USE_GEMM_ATTENTION:
            attention_fwd_gemm[dtype, target](
                output_ptr,
                query_ptr,
                key_ptr,
                value_ptr,
                log_sum_exp_ptr,
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                ctx,
                cache=cache,
            )
            return
        comptime simd_width = simd_width_of[dtype]()
        comptime is_float32 = dtype == DType.float32
        comptime Br = 16 if is_float32 else 32
        comptime Bc = 16 if is_float32 else 32
        comptime BLOCK_SIZE = 256
        comptime SM_OVERPROVISION = 32
        var device_ctx = ctx
        var query_tiles = ceildiv(Int(seq_len), Br)
        var num_tiles = Int(batch_size * num_heads * Int64(query_tiles))
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var num_blocks = max(min(num_tiles, SM_OVERPROVISION * num_sm), 1)

        comptime gpu_kernel = attention_fwd_gpu[
            dtype,
            simd_width,
            Br,
            Bc,
            BLOCK_SIZE,
            use_soft_exp=use_soft_exp,
            use_conditional_rescale=use_conditional_rescale,
        ]

        comptime if use_kv_cache:
            # PERFORMANCE NOTE:
            # Previously, JIT compilation (device_ctx.compile_function) was run dynamically on every
            # layer call and training step, adding massive host JIT compiler overhead. To avoid this,
            # we now query the cache pointer passed from the GPT2 model.
            #
            # In the backward pass, threads were also doing heap alloc() and free() inside the thread-block
            # inner loops, which serialized execution on the GPU. This is now resolved by using
            # stack-allocated LayoutTensors instead of heap memory.
            var addr_fwd = 0
            if cache:
                addr_fwd = cache.value()[].fwd_addr

            comptime CompiledType = type_of(
                device_ctx.compile_function[gpu_kernel]()
            )

            # Cache miss, compile the kernel and store the address.
            if addr_fwd == 0:
                var compiled = device_ctx.compile_function[gpu_kernel]()
                var ptr = alloc[CompiledType](1)
                ptr.unsafe_write(compiled^)
                addr_fwd = Int(ptr)
                if cache:
                    cache.value()[].fwd_addr = addr_fwd

            var casted_ptr = UnsafePointer[
                mut=True, type=CompiledType, origin=MutUntrackedOrigin
            ](unsafe_from_address=addr_fwd)
            var retrieved = casted_ptr[]

            device_ctx.enqueue_function(
                retrieved,
                num_tiles,
                query_tiles,
                output_ptr,
                query_ptr,
                key_ptr,
                value_ptr,
                log_sum_exp_ptr,
                seq_len,
                head_dim,
                grid_dim=(num_blocks,),
                block_dim=(BLOCK_SIZE,),
            )
        else:
            var compiled = device_ctx.compile_function[gpu_kernel]()
            device_ctx.enqueue_function(
                compiled,
                num_tiles,
                query_tiles,
                output_ptr,
                query_ptr,
                key_ptr,
                value_ptr,
                log_sum_exp_ptr,
                seq_len,
                head_dim,
                grid_dim=(num_blocks,),
                block_dim=(BLOCK_SIZE,),
            )
    else:
        raise Error("Invalid target")


@compiler.register("attention_fwd")
struct AttentionFwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        use_soft_exp: Bool = True,
        use_conditional_rescale: Bool = True,
    ](
        output: MutableInputTensor[dtype=dtype, rank=4, static_spec=...],
        q: InputTensor[dtype=dtype, rank=4, static_spec=...],
        k: InputTensor[dtype=dtype, rank=4, static_spec=...],
        v: InputTensor[dtype=dtype, rank=4, static_spec=...],
        l_vec: MutableInputTensor[dtype=DType.float32, rank=3, static_spec=...],
        batch_size: Int64,
        num_heads: Int64,
        seq_len: Int64,
        head_dim: Int64,
        ctx: DeviceContext,
    ) capturing raises:
        if output.size() != Int(batch_size * num_heads * seq_len * head_dim):
            raise Error("output size mismatch")
        if q.size() != Int(batch_size * num_heads * seq_len * head_dim):
            raise Error("q size mismatch")
        if k.size() != Int(batch_size * num_heads * seq_len * head_dim):
            raise Error("k size mismatch")
        if v.size() != Int(batch_size * num_heads * seq_len * head_dim):
            raise Error("v size mismatch")
        if l_vec.size() != Int(batch_size * num_heads * seq_len):
            raise Error("l_vec size mismatch")

        attention_fwd[dtype, target, use_soft_exp, use_conditional_rescale](
            output.unsafe_ptr(),
            q.unsafe_ptr(),
            k.unsafe_ptr(),
            v.unsafe_ptr(),
            l_vec.unsafe_ptr(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            ctx,
        )


# ===----------------------------------------------------------------------=== #
# Attention Backward
# ===----------------------------------------------------------------------=== #


# GPU backward follows FlashAttention-2 Algorithm 4 (Dao et al., 2023). The
# forward pass saved log-sum-exp L per query row so we can recompute softmax
# weights P_ij = exp(S_ij - L_i) on the fly without materializing the full
# N×N attention matrix. We also precompute D_i = sum_k(dO_i,k * O_i,k), the
# softmax "correction" term from the chain rule.
#
# FA2 splits the backward into two tiled passes with different outer loops:
#   Pass 1 (dQ): fix a Br×d query tile, stream KV tiles of size Bc×d.
#   Pass 2 (dK/dV): fix a Bc×d KV tile, stream query tiles of size Br×d.
# This mirrors the forward's Br×Bc blocking but swaps which dimension is
# resident in SRAM. Splitting avoids atomic adds into dK/dV/dQ in global
# memory. Each CTA owns one output tile and writes it once at the end.
# NOTE: With better support for atomic operations in Mojo, we could merge the
# two passes into a single kernel.


@always_inline
def _attention_bwd_update_step[
    dtype: DType,
    width: Int,
    use_soft_exp: Bool = True,
](
    q_row: ImmutKernelPtr[dtype],
    k_row: ImmutKernelPtr[dtype],
    v_row: ImmutKernelPtr[dtype],
    do_row: ImmutKernelPtr[dtype],
    dq_row: MutKernelPtr[dtype],
    dk_row: MutKernelPtr[dtype],
    dv_row: MutKernelPtr[dtype],
    L_i: Scalar[DType.float32],
    D_i: Scalar[DType.float32],
    head_dim: Int,
    attention_scale: Scalar[DType.float32],
) -> None:
    # S_ij = scale * Q_i * K_j
    var S_ij = _attention_query_key_dot_product[dtype, width](
        q_row, k_row, head_dim, attention_scale
    )

    # P_ji = exp(S_ji - L_i)
    var P_ij = software_emulated_exp[use_soft_exp](S_ij - L_i)

    # dP_ji = dO_i * V_j
    var dP_ij = _attention_query_key_dot_product[dtype, width](
        do_row, v_row, head_dim, Scalar[DType.float32](1.0)
    )

    # dS_ij = P_ij *(dP_ij - D_i)
    var dS_ij = P_ij * (dP_ij - D_i)

    # dq_row += (scale * dS_ij) * K_j
    _attention_add_weighted_value_to_output[dtype, width](
        dq_row, k_row, attention_scale * dS_ij, head_dim
    )

    # dk_row += (scale * dS_ij) * Q_i
    _attention_add_weighted_value_to_output[dtype, width](
        dk_row, q_row, attention_scale * dS_ij, head_dim
    )

    # dv_row += P_ij * dO_i
    _attention_add_weighted_value_to_output[dtype, width](
        dv_row, do_row, P_ij, head_dim
    )


@always_inline
def _attention_bwd_cpu[
    dtype: DType,
    width: Int,
    use_soft_exp: Bool = True,
](
    head_index: Int,
    d_query_ptr: MutKernelPtr[dtype],
    d_key_ptr: MutKernelPtr[dtype],
    d_value_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    query_ptr: ImmutKernelPtr[dtype],
    key_ptr: ImmutKernelPtr[dtype],
    value_ptr: ImmutKernelPtr[dtype],
    output_ptr: ImmutKernelPtr[dtype],
    log_sum_exp_ptr: ImmutKernelPtr[DType.float32],
    seq_len: Int,
    head_dim: Int,
    attention_scale: Scalar[DType.float32],
) -> None:
    var head_offset = head_index * seq_len * head_dim
    var lse_head = log_sum_exp_ptr + head_index * seq_len

    var q_head = query_ptr + head_offset
    var k_head = key_ptr + head_offset
    var v_head = value_ptr + head_offset
    var o_head = output_ptr + head_offset
    var do_head = d_output_ptr + head_offset

    var dq_head = d_query_ptr + head_offset
    var dk_head = d_key_ptr + head_offset
    var dv_head = d_value_ptr + head_offset

    # Zero out dk and dv for this head since we accumulate into them.
    for i in range(seq_len * head_dim):
        dk_head[i] = Scalar[dtype](0.0)
        dv_head[i] = Scalar[dtype](0.0)

    # Precompute row dot products D_i = sum_k (dO_i,k * O_i,k).
    var D = alloc[Scalar[DType.float32]](seq_len)
    for i in range(seq_len):
        D[i] = _attention_query_key_dot_product[dtype, width](
            do_head + i * head_dim,
            o_head + i * head_dim,
            head_dim,
            Scalar[DType.float32](1.0),
        )

    for i in range(seq_len):
        var q_row = q_head + i * head_dim
        var do_row = do_head + i * head_dim
        var dq_row = dq_head + i * head_dim
        var L_i = lse_head[i]
        var D_i = D[i]

        for d in range(head_dim):
            dq_row[d] = Scalar[dtype](0.0)

        for j in range(i + 1):  # Causal mask constraint, j <= i.
            var k_row = k_head + j * head_dim
            var v_row = v_head + j * head_dim
            var dk_row = dk_head + j * head_dim
            var dv_row = dv_head + j * head_dim

            _attention_bwd_update_step[dtype, width, use_soft_exp](
                q_row,
                k_row,
                v_row,
                do_row,
                dq_row,
                dk_row,
                dv_row,
                L_i,
                D_i,
                head_dim,
                attention_scale,
            )

    D.free()


def attention_bwd_cpu[
    dtype: DType,
    width: Int,
    use_soft_exp: Bool = True,
](
    d_query_ptr: MutKernelPtr[dtype],
    d_key_ptr: MutKernelPtr[dtype],
    d_value_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    query_ptr: ImmutKernelPtr[dtype],
    key_ptr: ImmutKernelPtr[dtype],
    value_ptr: ImmutKernelPtr[dtype],
    output_ptr: ImmutKernelPtr[dtype],
    log_sum_exp_ptr: ImmutKernelPtr[DType.float32],
    batch_size: Int,
    num_heads: Int,
    seq_len: Int,
    head_dim: Int,
) raises -> None:
    var attention_scale = Scalar[DType.float32](1) / sqrt(
        Scalar[DType.float32](head_dim)
    )
    var total_heads = Int(batch_size * num_heads)
    var max_workers = parallelism_level()
    var heads_per_worker = ceildiv(total_heads, max_workers)
    var num_workers = ceildiv(total_heads, heads_per_worker)

    @parameter
    def _worker(worker_index: Int):
        var base = worker_index * heads_per_worker
        var count = min(heads_per_worker, total_heads - base)
        for local in range(count):
            var head_index = base + local
            _attention_bwd_cpu[dtype, width, use_soft_exp](
                head_index,
                d_query_ptr,
                d_key_ptr,
                d_value_ptr,
                d_output_ptr,
                query_ptr,
                key_ptr,
                value_ptr,
                output_ptr,
                log_sum_exp_ptr,
                seq_len,
                head_dim,
                attention_scale,
            )

    traced_parallelize["attention_bwd", _worker](num_workers)


@always_inline
def _attention_copy_tile[
    dtype: DType,
    BLOCK_SIZE: Int,
    R: Int,
    C: Int,
    is_dram_to_shared: Bool,
](
    dest: LayoutTensor,
    src: LayoutTensor,
    row_count: Int,
    head_dim: Int,
) -> None:
    # Metal fix: rebind dest.ptr to the correct MutAnyOrigin+address_space so
    # writes go to the right memory space. address_space_cast[GENERIC] corrupted
    # threadgroup writes on Metal AIR. src is read-only; no rebind needed.
    var thread_id = Int(thread_idx.x)
    var element_count = row_count * head_dim
    var element_index = thread_id
    while element_index < element_count:
        var local_row = element_index // head_dim
        var column = element_index % head_dim
        comptime if is_dram_to_shared:
            var dest_ptr = rebind[
                UnsafePointer[
                    Scalar[dtype],
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                ]
            ](dest.ptr)
            dest_ptr[local_row * MAX_HEAD_DIM + column] = src.ptr[
                local_row * head_dim + column
            ].cast[dtype]()
        else:
            var dest_ptr = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                dest.ptr
            )
            dest_ptr[local_row * head_dim + column] = src.ptr[
                local_row * MAX_HEAD_DIM + column
            ].cast[dtype]()
        element_index += BLOCK_SIZE


# FA2 backward pass 1:
#    One CTA per (head, query_tile).
#    Outer tile is Q/dO/O.
# (Br rows): The inner loop streams K/V tiles
# (Bc cols): with j <= i causal mask.
@always_inline
def _attention_gpu_bwd_dq_tile_block[
    dtype: DType,
    width: Int,
    Br: Int,
    Bc: Int,
    BLOCK_SIZE: Int,
    use_soft_exp: Bool = True,
](
    query_tile_index: Int,
    query_tile_start: Int,
    q_tile_dram: LayoutTensor[
        dtype, Layout.row_major(Br, MAX_HEAD_DIM), ImmutAnyOrigin
    ],
    do_tile_dram: LayoutTensor[
        dtype, Layout.row_major(Br, MAX_HEAD_DIM), ImmutAnyOrigin
    ],
    o_tile_dram: LayoutTensor[
        dtype, Layout.row_major(Br, MAX_HEAD_DIM), ImmutAnyOrigin
    ],
    dq_tile_dram: LayoutTensor[
        dtype, Layout.row_major(Br, MAX_HEAD_DIM), MutAnyOrigin
    ],
    k_head: ImmutKernelPtr[dtype],
    v_head: ImmutKernelPtr[dtype],
    lse_head: ImmutKernelPtr[DType.float32],
    seq_len: Int,
    head_dim: Int,
) -> None:
    var query_tile_rows = min(Br, seq_len - query_tile_start)
    var attention_scale = Scalar[DType.float32](1) / sqrt(
        Scalar[DType.float32](head_dim)
    )

    # Shared tiles via .stack_allocation() (Mojo's __shared__). Rows padded to
    # MAX_HEAD_DIM so one binary serves any head_dim <= 128.
    var q_shared = LayoutTensor[
        dtype,
        Layout.row_major(Br, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var do_shared = LayoutTensor[
        dtype,
        Layout.row_major(Br, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var o_shared = LayoutTensor[
        dtype,
        Layout.row_major(Br, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var dq_shared = LayoutTensor[
        dtype,
        Layout.row_major(Br, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var key_shared = LayoutTensor[
        dtype,
        Layout.row_major(Bc, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var val_shared = LayoutTensor[
        dtype,
        Layout.row_major(Bc, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var lse_shared = LayoutTensor[
        DType.float32,
        Layout.row_major(Br),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var D_shared = LayoutTensor[
        DType.float32,
        Layout.row_major(Br),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # Metal fix: use lse_shared.ptr / D_shared.ptr directly (no GENERIC cast).

    # Load Q, dO, O tiles into SRAM once and reuse across every KV inner tile.
    _attention_copy_tile[dtype, BLOCK_SIZE, Br, MAX_HEAD_DIM, True](
        q_shared, q_tile_dram, query_tile_rows, head_dim
    )
    _attention_copy_tile[dtype, BLOCK_SIZE, Br, MAX_HEAD_DIM, True](
        do_shared, do_tile_dram, query_tile_rows, head_dim
    )
    _attention_copy_tile[dtype, BLOCK_SIZE, Br, MAX_HEAD_DIM, True](
        o_shared, o_tile_dram, query_tile_rows, head_dim
    )

    var thread_id = Int(thread_idx.x)
    if thread_id < query_tile_rows:
        lse_shared.ptr[thread_id] = lse_head[query_tile_start + thread_id]
        for d in range(head_dim):
            dq_shared[thread_id, d] = Scalar[dtype](0.0)
    barrier()

    # Compute row dot products D_i = dO_i · O_i (inline — Metal fix: use
    # shared .ptr directly to preserve AddressSpace.SHARED on Metal AIR).
    if thread_id < query_tile_rows:
        var d_val: Scalar[DType.float32] = 0.0
        for d in range(head_dim):
            d_val += (
                do_shared.ptr[thread_id * MAX_HEAD_DIM + d].cast[
                    DType.float32
                ]()
                * o_shared.ptr[thread_id * MAX_HEAD_DIM + d].cast[
                    DType.float32
                ]()
            )
        D_shared.ptr[thread_id] = d_val
    barrier()

    var local_row = thread_id % Br
    var thread_row_index = thread_id // Br
    # Scratch for cross-thread reduction:
    # 4 threads per query row (Br rows × 4 lanes)
    # merge their private dQ partials before writing dq_shared.
    var reduction_shared = LayoutTensor[
        DType.float32,
        Layout.row_major(Br, 4, 8),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # FA2 inner loop (dQ path): stream KV tiles J over the Br×Bc attention
    # sub-block. Causal masking stops at j <= i. K/V tiles are reloaded each
    # iteration. Q/dO/O stay resident from the load above.
    for key_tile_start in range(0, query_tile_start + Br, Bc):
        var key_tile_rows = min(Bc, seq_len - key_tile_start)

        var k_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Bc, MAX_HEAD_DIM), ImmutAnyOrigin
        ]((k_head + key_tile_start * head_dim))
        var v_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Bc, MAX_HEAD_DIM), ImmutAnyOrigin
        ]((v_head + key_tile_start * head_dim))

        _attention_copy_tile[dtype, BLOCK_SIZE, Bc, MAX_HEAD_DIM, True](
            key_shared, k_tile_dram, key_tile_rows, head_dim
        )
        _attention_copy_tile[dtype, BLOCK_SIZE, Bc, MAX_HEAD_DIM, True](
            val_shared, v_tile_dram, key_tile_rows, head_dim
        )
        barrier()

        # Per-thread dQ partials live in local memory (registers/spill), not
        # shared SRAM, only this thread reads/writes them before the reduction.
        var private_dq_tensor = LayoutTensor[
            DType.float32,
            Layout.row_major(MAX_HEAD_DIM),
            MutAnyOrigin,
            address_space=AddressSpace.GENERIC,
        ].stack_allocation()
        var private_dq = private_dq_tensor.ptr
        for col in range(head_dim):
            private_dq[col] = 0.0

        if local_row < query_tile_rows:
            var global_i = query_tile_start + local_row
            var q_base = local_row * MAX_HEAD_DIM
            var L_i = lse_shared.ptr[local_row]
            var D_i = D_shared.ptr[local_row]

            for j in range(thread_row_index, key_tile_rows, 4):
                var global_j = key_tile_start + j
                if global_j > global_i:
                    continue

                var k_base = j * MAX_HEAD_DIM

                # Metal fix: inline dot products via shared .ptr to preserve
                # AddressSpace.SHARED — _attention_query_key_dot_product takes
                # ImmutKernelPtr (GENERIC) which reads device memory on Metal.
                var S_ij: Scalar[DType.float32] = 0.0
                for d in range(head_dim):
                    S_ij += (
                        q_shared.ptr[q_base + d].cast[DType.float32]()
                        * key_shared.ptr[k_base + d].cast[DType.float32]()
                    )
                S_ij *= attention_scale
                var P_ij = software_emulated_exp[use_soft_exp](S_ij - L_i)
                var dP_ij: Scalar[DType.float32] = 0.0
                for d in range(head_dim):
                    dP_ij += (
                        do_shared.ptr[q_base + d].cast[DType.float32]()
                        * val_shared.ptr[k_base + d].cast[DType.float32]()
                    )
                var dS_ij = P_ij * (dP_ij - D_i)

                var factor = attention_scale * dS_ij
                var d = 0
                while d + width <= head_dim:
                    var k_vec = (
                        (key_shared.ptr + k_base + d)
                        .load[width=width]()
                        .cast[DType.float32]()
                    )
                    var dq_vec = (private_dq + d).load[width=width]()
                    (private_dq + d).store[width=width](
                        fma(SIMD[DType.float32, width](factor), k_vec, dq_vec)
                    )
                    d += width
                for tail in range(d, head_dim):
                    private_dq[tail] += (
                        factor
                        * key_shared.ptr[k_base + tail].cast[DType.float32]()
                    )

        for col_base in range(0, head_dim, 8):
            for c in range(8):
                var col = col_base + c
                if col < head_dim:
                    reduction_shared[
                        local_row, thread_row_index, c
                    ] = private_dq[col]
            barrier()
            for step in range(2):
                var c = thread_row_index + step * 4
                var col = col_base + c
                if col < head_dim:
                    var sum_col = (
                        reduction_shared[local_row, 0, c]
                        + reduction_shared[local_row, 1, c]
                        + reduction_shared[local_row, 2, c]
                        + reduction_shared[local_row, 3, c]
                    )
                    var existing = dq_shared[local_row, col].cast[
                        DType.float32
                    ]()
                    dq_shared[local_row, col] = (existing + sum_col).cast[
                        dtype
                    ]()
            barrier()

    _attention_copy_tile[dtype, BLOCK_SIZE, Br, MAX_HEAD_DIM, False](
        dq_tile_dram, dq_shared, query_tile_rows, head_dim
    )


def attention_bwd_dq_gpu[
    dtype: DType,
    width: Int,
    Br: Int,
    Bc: Int,
    BLOCK_SIZE: Int,
    use_soft_exp: Bool = True,
](
    num_tiles: Int,
    query_tiles: Int,
    d_query_ptr: MutKernelPtr[dtype],
    query_ptr: ImmutKernelPtr[dtype],
    key_ptr: ImmutKernelPtr[dtype],
    value_ptr: ImmutKernelPtr[dtype],
    output_ptr: ImmutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    log_sum_exp_ptr: ImmutKernelPtr[DType.float32],
    seq_len: Int64,
    head_dim: Int64,
) -> None:
    var grid_stride = Int(grid_dim.x)
    var block_tile = Int(block_idx.x)

    for tile_index in range(block_tile, num_tiles, grid_stride):
        var head_index = tile_index // query_tiles
        var query_tile_index = tile_index % query_tiles
        var query_tile_start = query_tile_index * Br
        var head_offset = head_index * Int(seq_len * head_dim)

        var q_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Br, MAX_HEAD_DIM), ImmutAnyOrigin
        ]((query_ptr + head_offset + query_tile_start * Int(head_dim)))
        var do_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Br, MAX_HEAD_DIM), ImmutAnyOrigin
        ]((d_output_ptr + head_offset + query_tile_start * Int(head_dim)))
        var o_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Br, MAX_HEAD_DIM), ImmutAnyOrigin
        ]((output_ptr + head_offset + query_tile_start * Int(head_dim)))
        var dq_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Br, MAX_HEAD_DIM), MutAnyOrigin
        ]((d_query_ptr + head_offset + query_tile_start * Int(head_dim)))

        var k_head = key_ptr + head_offset
        var v_head = value_ptr + head_offset
        var lse_head = log_sum_exp_ptr + head_index * Int(seq_len)

        _attention_gpu_bwd_dq_tile_block[
            dtype, width, Br, Bc, BLOCK_SIZE, use_soft_exp
        ](
            query_tile_index,
            query_tile_start,
            q_tile_dram,
            do_tile_dram,
            o_tile_dram,
            dq_tile_dram,
            k_head,
            v_head,
            lse_head,
            Int(seq_len),
            Int(head_dim),
        )


# FA2 backward pass 2: one CTA per (head, kv_tile). Outer tile is K/V
# (Bc rows); inner loop streams Q/dO/O tiles (Br cols) with j <= i.
@always_inline
def _attention_gpu_bwd_dkv_tile_block[
    dtype: DType,
    width: Int,
    Br: Int,
    Bc: Int,
    BLOCK_SIZE: Int,
    use_soft_exp: Bool = True,
](
    key_tile_index: Int,
    key_tile_start: Int,
    dk_tile_dram: LayoutTensor[
        dtype, Layout.row_major(Bc, MAX_HEAD_DIM), MutAnyOrigin
    ],
    dv_tile_dram: LayoutTensor[
        dtype, Layout.row_major(Bc, MAX_HEAD_DIM), MutAnyOrigin
    ],
    k_tile_dram: LayoutTensor[
        dtype, Layout.row_major(Bc, MAX_HEAD_DIM), ImmutAnyOrigin
    ],
    v_tile_dram: LayoutTensor[
        dtype, Layout.row_major(Bc, MAX_HEAD_DIM), ImmutAnyOrigin
    ],
    q_head: ImmutKernelPtr[dtype],
    do_head: ImmutKernelPtr[dtype],
    o_head: ImmutKernelPtr[dtype],
    lse_head: ImmutKernelPtr[DType.float32],
    seq_len: Int,
    head_dim: Int,
    query_tiles: Int,
) -> None:
    var key_tile_rows = min(Bc, seq_len - key_tile_start)
    var attention_scale = Scalar[DType.float32](1) / sqrt(
        Scalar[DType.float32](head_dim)
    )

    # Shared layout mirrors pass 1: K/V/dK/dV are the outer (Bc-row) tiles;
    # Q/dO/O stream inner.
    var key_shared = LayoutTensor[
        dtype,
        Layout.row_major(Bc, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var val_shared = LayoutTensor[
        dtype,
        Layout.row_major(Bc, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var dk_shared = LayoutTensor[
        dtype,
        Layout.row_major(Bc, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var dv_shared = LayoutTensor[
        dtype,
        Layout.row_major(Bc, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var q_shared = LayoutTensor[
        dtype,
        Layout.row_major(Br, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var do_shared = LayoutTensor[
        dtype,
        Layout.row_major(Br, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var o_shared = LayoutTensor[
        dtype,
        Layout.row_major(Br, MAX_HEAD_DIM),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var lse_shared = LayoutTensor[
        DType.float32,
        Layout.row_major(Br),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var D_shared = LayoutTensor[
        DType.float32,
        Layout.row_major(Br),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # Metal fix: use lse_shared.ptr / D_shared.ptr directly (no GENERIC cast).

    # Load K, V tiles into SRAM once; reused across every query inner tile.
    _attention_copy_tile[dtype, BLOCK_SIZE, Bc, MAX_HEAD_DIM, True](
        key_shared, k_tile_dram, key_tile_rows, head_dim
    )
    _attention_copy_tile[dtype, BLOCK_SIZE, Bc, MAX_HEAD_DIM, True](
        val_shared, v_tile_dram, key_tile_rows, head_dim
    )

    var thread_id = Int(thread_idx.x)
    if thread_id < key_tile_rows:
        for d in range(head_dim):
            dk_shared[thread_id, d] = Scalar[dtype](0.0)
            dv_shared[thread_id, d] = Scalar[dtype](0.0)
    barrier()

    var local_col = thread_id % Bc
    var thread_col_index = thread_id // Bc

    # 4 threads per KV column merge private dK/dV partials before dk/dv_shared.
    var reduction_dk = LayoutTensor[
        DType.float32,
        Layout.row_major(Bc, 4, 8),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var reduction_dv = LayoutTensor[
        DType.float32,
        Layout.row_major(Bc, 4, 8),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    for query_tile_idx in range((key_tile_index * Bc) // Br, query_tiles):
        var query_tile_start = query_tile_idx * Br
        var query_tile_rows = min(Br, seq_len - query_tile_start)

        var q_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Br, MAX_HEAD_DIM), ImmutAnyOrigin
        ]((q_head + query_tile_start * head_dim))
        var do_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Br, MAX_HEAD_DIM), ImmutAnyOrigin
        ]((do_head + query_tile_start * head_dim))
        var o_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Br, MAX_HEAD_DIM), ImmutAnyOrigin
        ]((o_head + query_tile_start * head_dim))

        _attention_copy_tile[dtype, BLOCK_SIZE, Br, MAX_HEAD_DIM, True](
            q_shared, q_tile_dram, query_tile_rows, head_dim
        )
        _attention_copy_tile[dtype, BLOCK_SIZE, Br, MAX_HEAD_DIM, True](
            do_shared, do_tile_dram, query_tile_rows, head_dim
        )
        _attention_copy_tile[dtype, BLOCK_SIZE, Br, MAX_HEAD_DIM, True](
            o_shared, o_tile_dram, query_tile_rows, head_dim
        )

        if thread_id < query_tile_rows:
            lse_shared.ptr[thread_id] = lse_head[query_tile_start + thread_id]
        barrier()

        # Compute D_i = dO_i · O_i (inline — Metal fix: use shared .ptr directly).
        if thread_id < query_tile_rows:
            var d_val: Scalar[DType.float32] = 0.0
            for d in range(head_dim):
                d_val += (
                    do_shared.ptr[thread_id * MAX_HEAD_DIM + d].cast[
                        DType.float32
                    ]()
                    * o_shared.ptr[thread_id * MAX_HEAD_DIM + d].cast[
                        DType.float32
                    ]()
                )
            D_shared.ptr[thread_id] = d_val
        barrier()

        # Per-thread dK/dV partials in local memory, same rationale as dQ pass.
        var private_dk_tensor = LayoutTensor[
            DType.float32,
            Layout.row_major(MAX_HEAD_DIM),
            MutAnyOrigin,
            address_space=AddressSpace.GENERIC,
        ].stack_allocation()
        var private_dk = private_dk_tensor.ptr
        var private_dv_tensor = LayoutTensor[
            DType.float32,
            Layout.row_major(MAX_HEAD_DIM),
            MutAnyOrigin,
            address_space=AddressSpace.GENERIC,
        ].stack_allocation()
        var private_dv = private_dv_tensor.ptr
        for col in range(head_dim):
            private_dk[col] = 0.0
            private_dv[col] = 0.0

        if local_col < key_tile_rows:
            var global_j = key_tile_start + local_col
            var k_base = local_col * MAX_HEAD_DIM

            for i in range(thread_col_index, query_tile_rows, 4):
                var global_i = query_tile_start + i
                if global_j > global_i:
                    continue

                var q_base = i * MAX_HEAD_DIM
                var L_i = lse_shared.ptr[i]
                var D_i = D_shared.ptr[i]

                # Metal fix: inline dot products via shared .ptr to preserve
                # AddressSpace.SHARED — _attention_query_key_dot_product takes
                # ImmutKernelPtr (GENERIC) which reads device memory on Metal.
                var S_ij: Scalar[DType.float32] = 0.0
                for d in range(head_dim):
                    S_ij += (
                        q_shared.ptr[q_base + d].cast[DType.float32]()
                        * key_shared.ptr[k_base + d].cast[DType.float32]()
                    )
                S_ij *= attention_scale
                var P_ij = software_emulated_exp[use_soft_exp](S_ij - L_i)
                var dP_ij: Scalar[DType.float32] = 0.0
                for d in range(head_dim):
                    dP_ij += (
                        do_shared.ptr[q_base + d].cast[DType.float32]()
                        * val_shared.ptr[k_base + d].cast[DType.float32]()
                    )
                var dS_ij = P_ij * (dP_ij - D_i)

                var dk_factor = attention_scale * dS_ij
                var dv_factor = P_ij
                var d = 0
                while d + width <= head_dim:
                    var q_vec = (
                        (q_shared.ptr + q_base + d)
                        .load[width=width]()
                        .cast[DType.float32]()
                    )
                    var dk_vec = (private_dk + d).load[width=width]()
                    (private_dk + d).store[width=width](
                        fma(
                            SIMD[DType.float32, width](dk_factor), q_vec, dk_vec
                        )
                    )
                    var do_vec = (
                        (do_shared.ptr + q_base + d)
                        .load[width=width]()
                        .cast[DType.float32]()
                    )
                    var dv_vec = (private_dv + d).load[width=width]()
                    (private_dv + d).store[width=width](
                        fma(
                            SIMD[DType.float32, width](dv_factor),
                            do_vec,
                            dv_vec,
                        )
                    )
                    d += width
                for tail in range(d, head_dim):
                    private_dk[tail] += (
                        dk_factor
                        * q_shared.ptr[q_base + tail].cast[DType.float32]()
                    )
                    private_dv[tail] += (
                        dv_factor
                        * do_shared.ptr[q_base + tail].cast[DType.float32]()
                    )

        for col_base in range(0, head_dim, 8):
            for c in range(8):
                var col = col_base + c
                if col < head_dim:
                    reduction_dk[local_col, thread_col_index, c] = private_dk[
                        col
                    ]
                    reduction_dv[local_col, thread_col_index, c] = private_dv[
                        col
                    ]
            barrier()
            for step in range(2):
                var c = thread_col_index + step * 4
                var col = col_base + c
                if col < head_dim:
                    var sum_dk = (
                        reduction_dk[local_col, 0, c]
                        + reduction_dk[local_col, 1, c]
                        + reduction_dk[local_col, 2, c]
                        + reduction_dk[local_col, 3, c]
                    )
                    var sum_dv = (
                        reduction_dv[local_col, 0, c]
                        + reduction_dv[local_col, 1, c]
                        + reduction_dv[local_col, 2, c]
                        + reduction_dv[local_col, 3, c]
                    )
                    var existing_dk = dk_shared[local_col, col].cast[
                        DType.float32
                    ]()
                    var existing_dv = dv_shared[local_col, col].cast[
                        DType.float32
                    ]()
                    dk_shared[local_col, col] = (existing_dk + sum_dk).cast[
                        dtype
                    ]()
                    dv_shared[local_col, col] = (existing_dv + sum_dv).cast[
                        dtype
                    ]()
            barrier()

    # Write final dk_shared and dv_shared blocks out to DRAM (fully contention-free)
    _attention_copy_tile[dtype, BLOCK_SIZE, Bc, MAX_HEAD_DIM, False](
        dk_tile_dram, dk_shared, key_tile_rows, head_dim
    )
    _attention_copy_tile[dtype, BLOCK_SIZE, Bc, MAX_HEAD_DIM, False](
        dv_tile_dram, dv_shared, key_tile_rows, head_dim
    )


def attention_bwd_dkv_gpu[
    dtype: DType,
    width: Int,
    Br: Int,
    Bc: Int,
    BLOCK_SIZE: Int,
    use_soft_exp: Bool = True,
](
    num_tiles: Int,
    kv_tiles: Int,
    query_tiles: Int,
    d_key_ptr: MutKernelPtr[dtype],
    d_value_ptr: MutKernelPtr[dtype],
    query_ptr: ImmutKernelPtr[dtype],
    key_ptr: ImmutKernelPtr[dtype],
    value_ptr: ImmutKernelPtr[dtype],
    output_ptr: ImmutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    log_sum_exp_ptr: ImmutKernelPtr[DType.float32],
    seq_len: Int64,
    head_dim: Int64,
) -> None:
    var grid_stride = Int(grid_dim.x)
    var block_tile = Int(block_idx.x)

    for tile_index in range(block_tile, num_tiles, grid_stride):
        var head_index = tile_index // kv_tiles
        var key_tile_index = tile_index % kv_tiles
        var key_tile_start = key_tile_index * Bc
        var head_offset = head_index * Int(seq_len * head_dim)

        var dk_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Bc, MAX_HEAD_DIM), MutAnyOrigin
        ]((d_key_ptr + head_offset + key_tile_start * Int(head_dim)))
        var dv_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Bc, MAX_HEAD_DIM), MutAnyOrigin
        ]((d_value_ptr + head_offset + key_tile_start * Int(head_dim)))
        var k_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Bc, MAX_HEAD_DIM), ImmutAnyOrigin
        ]((key_ptr + head_offset + key_tile_start * Int(head_dim)))
        var v_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Bc, MAX_HEAD_DIM), ImmutAnyOrigin
        ]((value_ptr + head_offset + key_tile_start * Int(head_dim)))

        var q_head = query_ptr + head_offset
        var do_head = d_output_ptr + head_offset
        var o_head = output_ptr + head_offset
        var lse_head = log_sum_exp_ptr + head_index * Int(seq_len)

        _attention_gpu_bwd_dkv_tile_block[
            dtype, width, Br, Bc, BLOCK_SIZE, use_soft_exp
        ](
            key_tile_index,
            key_tile_start,
            dk_tile_dram,
            dv_tile_dram,
            k_tile_dram,
            v_tile_dram,
            q_head,
            do_head,
            o_head,
            lse_head,
            Int(seq_len),
            Int(head_dim),
            query_tiles,
        )


def attention_bwd[
    dtype: DType,
    target: StaticString,
    use_soft_exp: Bool = True,
    use_kv_cache: Bool = True,
](
    d_qkv_ptr: MutKernelPtr[dtype],
    d_q_ptr: MutKernelPtr[dtype],
    d_k_ptr: MutKernelPtr[dtype],
    d_v_ptr: MutKernelPtr[dtype],
    d_attn_ptr: MutKernelPtr[dtype],
    d_attn_merged_ptr: ImmutKernelPtr[dtype],
    q_ptr: ImmutKernelPtr[dtype],
    k_ptr: ImmutKernelPtr[dtype],
    v_ptr: ImmutKernelPtr[dtype],
    attn_ptr: ImmutKernelPtr[dtype],
    log_sum_exp_ptr: ImmutKernelPtr[DType.float32],
    batch_size: Int64,
    num_heads: Int64,
    seq_len: Int64,
    head_dim: Int64,
    ctx: DeviceContext,
    cache: Optional[KVCachePtr] = None,
) capturing raises:
    # 1. Split heads backward: split d_attn_merged into d_attn
    merge_bwd[dtype, target](
        d_attn_merged_ptr,
        d_attn_ptr,
        Int(batch_size),
        Int(seq_len),
        Int(num_heads),
        Int(head_dim),
        ctx,
    )

    # 2. Core attention backward
    attention_bwd[dtype, target, use_soft_exp, use_kv_cache](
        d_q_ptr,
        d_k_ptr,
        d_v_ptr,
        d_attn_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        attn_ptr,
        log_sum_exp_ptr,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        ctx,
        cache=cache,
    )

    # 3. Merge QKV backward: merge d_q, d_k, d_v back into d_qkv
    var d_dst_ptrs = List[ImmutKernelPtr[dtype]]()
    d_dst_ptrs.append(d_q_ptr)
    d_dst_ptrs.append(d_k_ptr)
    d_dst_ptrs.append(d_v_ptr)
    split_bwd[dtype, target, num_splits=3](
        d_qkv_ptr,
        d_dst_ptrs,
        Int(batch_size),
        Int(seq_len),
        Int(num_heads),
        Int(head_dim),
        ctx,
    )


# ===----------------------------------------------------------------------=== #
# Attention Backward — GPU, tensor-core GEMM path
# ===----------------------------------------------------------------------=== #
#
# FlashAttention-2 backward, but with every O(T²·d) product as a tensor-core
# GEMM. The softmax probabilities P are recomputed from the saved log-sum-exp
# (no online accumulation, no max/sum reduction). linalg.matmul has no
# transpose_a, so the two gradients that need Aᵀ·B (dV = Pᵀ·dO, dK = dSᵀ·Q) are
# fed pre-transposed Pᵀ/dSᵀ, which the dS elementwise kernel emits for free.


@always_inline
def _gemm_scratch_buffer[
    bdt: DType
](
    addr: Int, count: Int, ctx: DeviceContext, zero_on_alloc: Bool = False
) raises -> Tuple[Int, MutKernelPtr[bdt]]:
    # Allocate a persistent DeviceBuffer on first use (heap-held, never freed) or
    # reconstruct the data pointer from a cached heap address. Returns the
    # (possibly new) heap address and the device data pointer. When
    # `zero_on_alloc` is set, the freshly allocated buffer is memset to 0 once —
    # used for the causal P/dS buffers whose above-diagonal half must stay zero
    # for the dense gradient GEMMs while the P+dS kernel writes only the lower
    # triangle each step.
    comptime BufType = type_of(ctx.enqueue_create_buffer[bdt](1))
    var out_addr = addr
    if out_addr == 0:
        var buf = ctx.enqueue_create_buffer[bdt](count)
        if zero_on_alloc:
            ctx.enqueue_memset(buf, Scalar[bdt](0))
        var p = alloc[BufType](1)
        p.unsafe_write(buf^)
        out_addr = Int(p)
    var bp = UnsafePointer[mut=True, type=BufType, origin=MutUntrackedOrigin](
        unsafe_from_address=out_addr
    )
    return (out_addr, rebind[MutKernelPtr[bdt]](bp[].unsafe_ptr()))


@always_inline
def _attention_bwd_recompute_p_gpu[
    dtype: DType,
](
    num_elems: Int,
    seq_len: Int,
    attention_scale: Scalar[DType.float32],
    scores_ptr: ImmutKernelPtr[DType.float32],
    lse_ptr: ImmutKernelPtr[DType.float32],
    p_ptr: MutKernelPtr[dtype],
) -> None:
    # P_ij = exp(scale·S_ij − L_i) for j ≤ i, else 0. Grid-strided over the flat
    # [B·nh, T, T] scores plane.
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var grid_stride = Int(grid_dim.x) * Int(block_dim.x)
    var plane = seq_len * seq_len
    var e = tid
    while e < num_elems:
        var rem = e % plane
        var i = rem // seq_len
        var j = rem % seq_len
        var bh = e // plane
        var p = Scalar[DType.float32](0.0)
        if j <= i:
            p = exp(scores_ptr[e] * attention_scale - lse_ptr[bh * seq_len + i])
        p_ptr[e] = p.cast[dtype]()
        e += grid_stride


@always_inline
def _attention_bwd_rowdot_gpu[
    dtype: DType,
](
    num_rows: Int,
    head_dim: Int,
    do_ptr: ImmutKernelPtr[dtype],
    o_ptr: ImmutKernelPtr[dtype],
    d_ptr: MutKernelPtr[DType.float32],
) -> None:
    # D_i = Σ_d dO_i,d · O_i,d, one thread per [B·nh, T] row.
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var grid_stride = Int(grid_dim.x) * Int(block_dim.x)
    var row = tid
    while row < num_rows:
        var base = row * head_dim
        var acc = Scalar[DType.float32](0.0)
        for d in range(head_dim):
            acc += (
                do_ptr[base + d].cast[DType.float32]()
                * o_ptr[base + d].cast[DType.float32]()
            )
        d_ptr[row] = acc
        row += grid_stride


@always_inline
def _attention_bwd_p_and_ds_gpu[
    dtype: DType,
    width: Int,
    stored_p: Bool = False,
    aligned: Bool = True,
](
    num_blocks: Int,
    seq_len: Int,
    attention_scale: Scalar[DType.float32],
    scores_ptr: ImmutKernelPtr[dtype],
    lse_ptr: ImmutKernelPtr[DType.float32],
    dp_ptr: ImmutKernelPtr[dtype],
    d_ptr: ImmutKernelPtr[DType.float32],
    p_ptr: MutKernelPtr[dtype],
    ds_ptr: MutKernelPtr[dtype],
) -> None:
    # Fused P recompute + dS, one pass over the [B·nh,T,T] plane:
    #   P_ij = exp(scale·S_ij − L_i)        (causal; j > i → 0)
    #   dS_ij = scale·P_ij·(dP_ij − D_i)
    # Writes P (consumed by the Pᵀ transpose → dV) and dS (→ dQ, and the dSᵀ
    # transpose → dK). This is the backward's largest kernel and is bandwidth
    # bound, so loads/stores are vectorized `width` elements at a time. Each
    # thread owns a `width`-wide block; because seq_len is a multiple of `width`,
    # a block never straddles a row, so `i`/`L_i`/`D_i` are constant across it.
    #
    # `aligned` (comptime): True is the production path above — requires
    # seq_len % width == 0 (checked on the host before dispatch). False (the
    # equivalence suite's odd seq_len, e.g. 7) falls back to a scalar,
    # one-element-per-iteration grid-stride sweep over the *whole* [B·nh,T,T]
    # plane below: with seq_len not a multiple of width, a width-wide block
    # can straddle a row boundary, so `jbase + w` would silently address the
    # wrong row's i/L_i/D_i for the wrapped lanes — this is a correctness bug,
    # not just an alignment one, so the fallback recomputes bh/i/j per element
    # instead of trying to patch up the vectorized indexing.
    var lane = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var grid_stride = Int(grid_dim.x) * Int(block_dim.x)
    var plane = seq_len * seq_len

    # scores_ptr/dp_ptr/p_ptr/ds_ptr all share `dtype`, so one alignment
    # covers every vectorized load/store below. e0 = b*width for grid-strided
    # integer b, so it is provably a multiple of width regardless of the
    # runtime plane/seq_len values — same proof shape as adamw's
    # idx = global_tid*width.
    comptime align = align_of[SIMD[dtype, width]]()
    comptime if aligned:
        var b = lane
        while b < num_blocks:
            var e0 = b * width
            var bh = e0 // plane
            var rem = e0 % plane
            var i = rem // seq_len
            var jbase = rem % seq_len
            # Causal: a width-block lies entirely above the diagonal iff
            # jbase > i (width divides seq_len, so the block never straddles
            # a row). Its P/dS are structurally zero.
            #
            # We MUST explicitly zero these blocks every step. The earlier
            # design skipped them (`continue`) and relied on the persistent
            # scratch staying zero from its one-time zero-on-alloc memset,
            # since nothing "should" ever write the upper triangle. On the
            # Blackwell (sm_120) build that invariant did not hold: the
            # untouched upper-triangle of the dS buffer came back holding
            # NaN/Inf bit patterns (238 of them in layer 11 head 0 alone on
            # the very first backward call, buffer verified all-zero the
            # instant before this kernel launched), which the dense
            # dQ = dS·K / dK = dSᵀ·Q GEMMs then summed into every query/key
            # gradient — turning the whole backward pass NaN from step 1.
            # The forward was unaffected (finite loss) and fp32 was clean,
            # which is why it looked LR-independent and precision-specific.
            # An explicit vectorized zero-store is cheap (a streaming write,
            # no load/compute) and makes correctness independent of whatever
            # stale contents the scratch carries.
            # See docs/ai/bf16_backward_nan_upper_triangle_bug.md.
            if jbase > i:
                var zero_vec = SIMD[dtype, width](0)
                (ds_ptr + e0).store[width=width, alignment=align](zero_vec)
                comptime if not stored_p:
                    (p_ptr + e0).store[width=width, alignment=align](zero_vec)
                b += grid_stride
                continue
            var d_i = d_ptr[bh * seq_len + i]
            # `stored_p`: scores_ptr already holds P (the forward's stored
            # softmax probs), so read it directly — no QKᵀ recompute, no
            # exp/lse, no P store.
            var raw = (
                (scores_ptr + e0)
                .load[width=width, alignment=align]()
                .cast[DType.float32]()
            )
            var dpv = (
                (dp_ptr + e0)
                .load[width=width, alignment=align]()
                .cast[DType.float32]()
            )

            var pv = SIMD[DType.float32, width](0.0)
            var dsv = SIMD[DType.float32, width](0.0)

            comptime if stored_p:
                comptime for w in range(width):
                    if jbase + w <= i:
                        var pw = raw[w]
                        dsv[w] = attention_scale * pw * (dpv[w] - d_i)
                (ds_ptr + e0).store[width=width, alignment=align](
                    dsv.cast[dtype]()
                )
            else:
                var lse_i = lse_ptr[bh * seq_len + i]
                comptime for w in range(width):
                    if jbase + w <= i:
                        var pw = exp(raw[w] * attention_scale - lse_i)
                        pv[w] = pw
                        dsv[w] = attention_scale * pw * (dpv[w] - d_i)
                (p_ptr + e0).store[width=width, alignment=align](
                    pv.cast[dtype]()
                )
                (ds_ptr + e0).store[width=width, alignment=align](
                    dsv.cast[dtype]()
                )
            b += grid_stride
    else:
        # num_blocks here is the TOTAL element count (B·nh·T·T), not a
        # width-block count — the host passes `plane` directly in this mode.
        var idx = lane
        while idx < num_blocks:
            var bh = idx // plane
            var rem = idx % plane
            var i = rem // seq_len
            var j = rem % seq_len
            if j <= i:
                var d_i = d_ptr[bh * seq_len + i]
                var raw = scores_ptr[idx].cast[DType.float32]()
                var dpv = dp_ptr[idx].cast[DType.float32]()
                comptime if stored_p:
                    var pw = raw
                    var dsv = attention_scale * pw * (dpv - d_i)
                    ds_ptr[idx] = dsv.cast[dtype]()
                else:
                    var lse_i = lse_ptr[bh * seq_len + i]
                    var pw = exp(raw * attention_scale - lse_i)
                    var dsv = attention_scale * pw * (dpv - d_i)
                    p_ptr[idx] = pw.cast[dtype]()
                    ds_ptr[idx] = dsv.cast[dtype]()
            idx += grid_stride


comptime TRANSPOSE_TILE = 32


@always_inline
def _attention_transpose_planes_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
](
    num_planes: Int,
    seq_len: Int,
    src_ptr: ImmutKernelPtr[dtype],
    dst_ptr: MutKernelPtr[dtype],
) -> None:
    # Coalesced per-plane transpose of a [num_planes, T, T] buffer using 32×32
    # shared-memory tiles (padded to avoid bank conflicts). Reads a tile
    # coalesced, writes the transposed tile coalesced — replacing the strided
    # scatter stores that previously dominated the backward.
    comptime TILE = TRANSPOSE_TILE
    comptime STRIDE = TILE + 1  # pad to avoid shared-memory bank conflicts
    # Metal fix: use tile.ptr[i] directly to preserve AddressSpace.SHARED.
    var tile = LayoutTensor[
        dtype,
        Layout.row_major(TILE, STRIDE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var plane = seq_len * seq_len
    var tiles = ceildiv(seq_len, TILE)
    var tiles_per_plane = tiles * tiles
    var total_tiles = num_planes * tiles_per_plane

    var tx = Int(thread_idx.x) % TILE
    var ty = Int(thread_idx.x) // TILE
    comptime ROW_STEP = BLOCK_SIZE // TILE

    var bt = Int(block_idx.x)
    while bt < total_tiles:
        var bh = bt // tiles_per_plane
        var rem = bt % tiles_per_plane
        var tile_row = (rem // tiles) * TILE
        var tile_col = (rem % tiles) * TILE
        var base = bh * plane

        var r = ty
        while r < TILE:
            var gr = tile_row + r
            var gc = tile_col + tx
            if gr < seq_len and gc < seq_len:
                tile.ptr[r * STRIDE + tx] = src_ptr[base + gr * seq_len + gc]
            r += ROW_STEP
        barrier()

        r = ty
        while r < TILE:
            var gr = tile_col + r
            var gc = tile_row + tx
            if gr < seq_len and gc < seq_len:
                dst_ptr[base + gr * seq_len + gc] = tile.ptr[tx * STRIDE + r]
            r += ROW_STEP
        barrier()

        bt += Int(grid_dim.x)


@always_inline
def _attention_transpose_rect_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
](
    num_planes: Int,
    rows: Int,
    cols: Int,
    src_ptr: ImmutKernelPtr[dtype],
    dst_ptr: MutKernelPtr[dtype],
) -> None:
    # Coalesced transpose of each [rows, cols] plane → [cols, rows], 32×32 tiles.
    # Used for the small [T,hd]↔[hd,T] transposes that let dK/dV avoid
    # transposing the big [T,T] dS/P matrices.
    comptime TILE = TRANSPOSE_TILE
    comptime STRIDE = TILE + 1
    # Metal fix: use tile.ptr[i] directly to preserve AddressSpace.SHARED.
    var tile = LayoutTensor[
        dtype,
        Layout.row_major(TILE, STRIDE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var src_plane = rows * cols
    var dst_plane = cols * rows
    var tiles_r = ceildiv(rows, TILE)
    var tiles_c = ceildiv(cols, TILE)
    var tiles_per_plane = tiles_r * tiles_c
    var total_tiles = num_planes * tiles_per_plane

    var tx = Int(thread_idx.x) % TILE
    var ty = Int(thread_idx.x) // TILE
    comptime ROW_STEP = BLOCK_SIZE // TILE

    var bt = Int(block_idx.x)
    while bt < total_tiles:
        var bh = bt // tiles_per_plane
        var rem = bt % tiles_per_plane
        var tile_row = (rem // tiles_c) * TILE  # offset along rows
        var tile_col = (rem % tiles_c) * TILE  # offset along cols
        var src_base = bh * src_plane
        var dst_base = bh * dst_plane

        var r = ty
        while r < TILE:
            var gr = tile_row + r
            var gc = tile_col + tx
            if gr < rows and gc < cols:
                tile.ptr[r * STRIDE + tx] = src_ptr[src_base + gr * cols + gc]
            r += ROW_STEP
        barrier()

        # dst is [cols, rows]: element (tile_col+r, tile_row+tx).
        r = ty
        while r < TILE:
            var gr = tile_col + r
            var gc = tile_row + tx
            if gr < cols and gc < rows:
                dst_ptr[dst_base + gr * rows + gc] = tile.ptr[tx * STRIDE + r]
            r += ROW_STEP
        barrier()

        bt += Int(grid_dim.x)


@always_inline
def _attn_batched_gemm_gpu[
    dt: DType,
    BM: Int,  # output rows per threadgroup  (= BK)
    BN: Int,  # output cols per threadgroup
    BK: Int,  # K-tile step  (must equal BM so A-tile = B-tile element count)
](
    bh: Int,  # total batch×head count (BH = B*NH)
    M: Int,  # output rows (T or HD)
    N: Int,  # output cols (HD or T)
    K: Int,  # inner dimension (T)
    a_ptr: ImmutKernelPtr[dt],  # [bh, M, K]
    b_ptr: ImmutKernelPtr[dt],  # [bh, K, N]
    c_ptr: MutKernelPtr[dt],  # [bh, M, N]
    a_stride: Int,  # M*K
    b_stride: Int,  # K*N
    c_stride: Int,  # M*N
    tiles_m: Int,  # ceildiv(M, BM)
    tiles_n: Int,  # ceildiv(N, BN)
) -> None:
    # Generic tiled batched GEMM: C[bh, M, N] = A[bh, M, K] · B[bh, K, N].
    # One threadgroup per (head, m-tile, n-tile); flat grid encodes the triple.
    # BM = BK is required so A-tile [BM,BK] and B-tile [BK,BN] can both be loaded
    # in one barrier using the same (ty,tx) thread mapping for B and the first
    # BM*BK threads for A.
    #
    # Shared-memory layout (Metal: access via .ptr directly, no GENERIC casts):
    #   a_sh[BM, BK+1]  — +1 column padding avoids bank conflicts
    #   b_sh[BK, BN+1]
    #
    # Thread mapping: ty = tid // BN  (output row in tile),  tx = tid % BN  (col).
    # All BM*BN threads load one B element each (ty ∈ [0,BK), tx ∈ [0,BN)).
    # First BM*BK threads load one A element each.
    # NOTE: BN must equal N when used for headout (N=HD=64, BN=64) so the full N
    # dimension is covered in one tile (no n-tile loop needed, tiles_n=1).
    var a_sh = LayoutTensor[
        dt,
        Layout.row_major(BM, BK + 1),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var b_sh = LayoutTensor[
        dt,
        Layout.row_major(BK, BN + 1),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var tid = Int(thread_idx.x)  # [0, BM*BN)
    var ty = tid // BN  # output-tile row   [0, BM)
    var tx = tid % BN  # output-tile col   [0, BN)

    # Decode flat block index → (head, m-tile, n-tile).
    var bid = Int(block_idx.x)
    var bh_idx = bid // (tiles_m * tiles_n)
    var rem = bid % (tiles_m * tiles_n)
    var tile_m = rem // tiles_n
    var tile_n = rem % tiles_n
    var m_off = tile_m * BM
    var n_off = tile_n * BN

    var a_base = bh_idx * a_stride + m_off * K
    var b_base = bh_idx * b_stride + n_off
    var c_base = bh_idx * c_stride + m_off * N + n_off

    var acc = Scalar[dt](0.0)

    for k in range(0, K, BK):
        # Load A tile [BM, BK]: first BM*BK threads, each loads one element.
        # (BM=BK, so BM*BK = BK*BK ≤ BM*BN always when BK ≤ BN.)
        if tid < BM * BK:
            var ra = tid // BK  # row in A tile  [0, BM)
            var ca = tid % BK  # col in A tile  [0, BK)
            a_sh.ptr[ra * (BK + 1) + ca] = a_ptr[a_base + ra * K + k + ca]
        # Load B tile [BK, BN]: all threads, one element each.
        # ty ∈ [0, BM=BK) serves as the k-row index; tx as the n-col index.
        b_sh.ptr[ty * (BN + 1) + tx] = b_ptr[b_base + (k + ty) * N + tx]
        barrier()

        # Accumulate: acc += sum_{kk} A_sh[ty, kk] * B_sh[kk, tx]
        for kk in range(BK):
            acc = fma(
                a_sh.ptr[ty * (BK + 1) + kk],
                b_sh.ptr[kk * (BN + 1) + tx],
                acc,
            )
        barrier()

    # Boundary guard: write if within M×N (needed when tiles don't divide evenly).
    if m_off + ty < M and n_off + tx < N:
        c_ptr[c_base + ty * N + tx] = acc


def _launch_batched_headout[
    dtype: DType,
](
    a_ptr: ImmutKernelPtr[dtype],  # [BH, T, T]   — dS or P (big matrix)
    b_ptr: ImmutKernelPtr[dtype],  # [BH, T, HD]  — K or dO (small matrix)
    c_ptr: MutKernelPtr[dtype],  # [BH, T, HD]  — output
    BH: Int,
    T: Int,
    HD: Int,
    ctx: DeviceContext,
) raises:
    # Batched headout GEMM: C[BH,T,HD] = A[BH,T,T] · B[BH,T,HD].
    # Used on Metal for dQ = dS·K and (forward) A·V.
    # Tile: BM=BK=16 rows, BN=HD=64 cols (full N in one tile), 1024 threads/group.
    # Grid: (BH × T/BM) flat blocks.
    comptime BM = 16
    comptime BK = 16  # must equal BM
    comptime BN = 64  # = HD; covers all 64 head-dim cols in one tile
    comptime THREADS = BM * BN  # = 1024
    var tiles_m = ceildiv(T, BM)
    var tiles_n = ceildiv(HD, BN)  # = 1 when HD=64=BN
    var num_blocks = BH * tiles_m * tiles_n
    comptime k = _attn_batched_gemm_gpu[dtype, BM, BN, BK]
    var compiled = ctx.compile_function[k]()
    ctx.enqueue_function(
        compiled,
        BH,
        T,
        HD,
        T,
        a_ptr,
        b_ptr,
        c_ptr,
        T * T,
        T * HD,
        T * HD,
        tiles_m,
        tiles_n,
        grid_dim=(num_blocks,),
        block_dim=(THREADS,),
    )


def _launch_batched_kvgrad[
    dtype: DType,
](
    a_ptr: ImmutKernelPtr[dtype],  # [BH, HD, T]  — Qᵀ or dOᵀ (small matrix)
    b_ptr: ImmutKernelPtr[dtype],  # [BH, T,  T]  — dS or P   (big matrix)
    c_ptr: MutKernelPtr[dtype],  # [BH, HD, T]  — output dKᵀ or dVᵀ
    BH: Int,
    T: Int,
    HD: Int,
    ctx: DeviceContext,
) raises:
    # Batched kvgrad GEMM: C[BH,HD,T] = A[BH,HD,T] · B[BH,T,T].
    # Used on Metal for dKᵀ = Qᵀ·dS and dVᵀ = dOᵀ·P (after rect-transposing Q/dO).
    # Tile: BM=BK=16 rows (of HD=64), BN=32 cols (of T=1024), 512 threads/group.
    # Grid: (BH × HD/BM × T/BN) flat blocks.
    comptime BM = 16
    comptime BK = 16  # must equal BM
    comptime BN = 32  # T=1024 needs tiles_n = T/BN steps
    comptime THREADS = BM * BN  # = 512
    var tiles_m = ceildiv(HD, BM)  # = 4 when HD=64
    var tiles_n = ceildiv(T, BN)  # = 32 when T=1024
    var num_blocks = BH * tiles_m * tiles_n
    comptime k = _attn_batched_gemm_gpu[dtype, BM, BN, BK]
    var compiled = ctx.compile_function[k]()
    ctx.enqueue_function(
        compiled,
        BH,
        HD,
        T,
        T,
        a_ptr,
        b_ptr,
        c_ptr,
        HD * T,
        T * T,
        HD * T,
        tiles_m,
        tiles_n,
        grid_dim=(num_blocks,),
        block_dim=(THREADS,),
    )


# ---------------------------------------------------------------------------
# Metal-only batched headout kernel (4-rows-per-thread):
# C[BH,M,N] = A[BH,M,K] · B[BH,K,N] where M=K=T, N=HD.
#
# Each thread computes 4 consecutive C rows (register tiling). With BM=64,
# BN=HD=64, BK=16, THREADS=1024: BM×BK = BK×BN = THREADS → 100% A and B
# loading efficiency (every thread loads exactly 1 element per tile per pass,
# no idle threads). 4 FMAs per B-read → 4× better register reuse vs 1-per-thread.
# Grid: (tiles_n, tiles_m, BH) 3D — eliminates integer divisions per block.
# Shared mem: A_sh[64,17] + B_sh[16,65] = 8.3 KB per block.
# ---------------------------------------------------------------------------
@always_inline
def _attn_headout4_gpu[
    dt: DType,
    BM: Int,  # output rows per block; BM × BK must equal THREADS
    BN: Int,  # output cols per block (= HD = 64)
    BK: Int,  # K-tile step; BK × BN must equal THREADS; ROWS_PER_THREAD = BM//(THREADS//BN)
](
    bh: Int,
    M: Int,  # T
    N: Int,  # HD
    K: Int,  # T (the common K-dimension)
    a_ptr: ImmutKernelPtr[dt],  # [BH, M, K]
    b_ptr: ImmutKernelPtr[dt],  # [BH, K, N]
    c_ptr: MutKernelPtr[dt],  # [BH, M, N]
    a_stride: Int,  # M*K
    b_stride: Int,  # K*N
    c_stride: Int,  # M*N
) -> None:
    # With BM=64, BN=64, BK=16, THREADS=1024:
    #   ROWS_PER_THREAD = BM / (THREADS / BN) = 64 / (1024 / 64) = 64 / 16 = 4
    # ty = tid // BN: which "4-row group" this thread belongs to [0, THREADS/BN)
    # tx = tid %  BN: which output column                         [0, BN)
    comptime ROWS_PER_THREAD = BM * BN // (BM * BK)  # = BN // BK = 64 // 16 = 4
    var a_sh = LayoutTensor[
        dt,
        Layout.row_major(BM, BK + 1),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var b_sh = LayoutTensor[
        dt,
        Layout.row_major(BK, BN + 1),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var tid = Int(thread_idx.x)
    var bh_idx = Int(block_idx.z)  # head index (no integer division!)
    var tile_m = Int(block_idx.y)  # M-tile index
    var tile_n = Int(block_idx.x)  # N-tile index
    var m_off = tile_m * BM
    var n_off = tile_n * BN

    var ty = tid // BN  # output row-group [0, THREADS/BN = BM/ROWS)
    var tx = tid % BN  # output col       [0, BN)

    var a_base = bh_idx * a_stride + m_off * K
    var b_base = bh_idx * b_stride + n_off
    var c_base = bh_idx * c_stride + m_off * N + n_off

    # Four register accumulators (one per output row per thread).
    var acc0 = Scalar[dt](0.0)
    var acc1 = Scalar[dt](0.0)
    var acc2 = Scalar[dt](0.0)
    var acc3 = Scalar[dt](0.0)

    for k in range(0, K, BK):
        # Load A tile [BM, BK]: tid → row = tid//BK, col = tid%BK.
        # BM×BK = THREADS → every thread loads exactly 1 element — 0% idle.
        # SIMD coalescing: 32 consecutive tids → 2 rows × 16 cols each (2 cache
        # lines, both fully utilised because rows are K=1024 apart in memory).
        # Zero-pad the K-remainder. K (= T for A·V and dQ) need not be a multiple
        # of BK, but the inner kk loop is a `comptime for` fixed at BK steps. Metal
        # returns OOB reads as garbage instead of faulting (CUDA never runs this;
        # cuBLAS does A·V there), so an unaligned T (generation/tests at 9, 17)
        # would spill garbage into every accumulator. Always true for aligned
        # production shapes (T=1024, HD=64), so no fast-path change.
        var ra = tid // BK
        var ca = tid % BK
        a_sh.ptr[ra * (BK + 1) + ca] = a_ptr[a_base + ra * K + k + ca] if (
            m_off + ra < M and k + ca < K
        ) else Scalar[dt](0)

        # Load B tile [BK, BN]: tid → row = tid//BN, col = tid%BN.
        # BK×BN = THREADS → every thread loads exactly 1 element.
        # SIMD coalescing: 32 consecutive tids → same BK row, 32 consecutive
        # BN cols → 1 cache line per SIMD pair → perfect coalescing.
        var rb = tid // BN
        var cb = tid % BN
        b_sh.ptr[rb * (BN + 1) + cb] = b_ptr[b_base + (k + rb) * N + cb] if (
            k + rb < K and n_off + cb < N
        ) else Scalar[dt](0)

        barrier()

        # Inner K-tile loop: 4 FMAs per B-read (register tile over 4 output rows).
        # A reads: ty is uniform within each SIMD group → all 4 A-reads are BROADCASTs.
        # B reads: tx varies 0..31 within SIMD → bank = (kk+tx)%32 cycles all banks.
        # comptime for forces compile-time unroll: BK=16 iterations, each yields a
        # B-broadcast and 4 FMAs → the Metal shader sees 16 inlined instruction groups.
        comptime for kk in range(BK):
            var bv = b_sh.ptr[kk * (BN + 1) + tx]
            acc0 = fma(
                a_sh.ptr[(ty * ROWS_PER_THREAD + 0) * (BK + 1) + kk], bv, acc0
            )
            acc1 = fma(
                a_sh.ptr[(ty * ROWS_PER_THREAD + 1) * (BK + 1) + kk], bv, acc1
            )
            acc2 = fma(
                a_sh.ptr[(ty * ROWS_PER_THREAD + 2) * (BK + 1) + kk], bv, acc2
            )
            acc3 = fma(
                a_sh.ptr[(ty * ROWS_PER_THREAD + 3) * (BK + 1) + kk], bv, acc3
            )

        barrier()

    # Write outputs (boundary guard in case M is not a multiple of BM).
    var row_base = m_off + ty * ROWS_PER_THREAD
    var c_off = c_base + ty * ROWS_PER_THREAD * N + tx
    if row_base + 0 < M and n_off + tx < N:
        c_ptr[c_off + 0 * N] = acc0
    if row_base + 1 < M and n_off + tx < N:
        c_ptr[c_off + 1 * N] = acc1
    if row_base + 2 < M and n_off + tx < N:
        c_ptr[c_off + 2 * N] = acc2
    if row_base + 3 < M and n_off + tx < N:
        c_ptr[c_off + 3 * N] = acc3


def _launch_headout4[
    dtype: DType
](
    a_ptr: ImmutKernelPtr[dtype],  # [BH, T, T]
    b_ptr: ImmutKernelPtr[dtype],  # [BH, T, HD]
    c_ptr: MutKernelPtr[dtype],  # [BH, T, HD]
    BH: Int,
    T: Int,
    HD: Int,
    ctx: DeviceContext,
) raises:
    # C[BH,T,HD] = A[BH,T,T] · B[BH,T,HD].
    # BM=64 rows per block; THREADS=1024; BK=16; BN=HD=64; ROWS_PER_THREAD=4.
    # 3D grid: (tiles_n=1, tiles_m, BH) → block_idx gives indices without division.
    comptime BM = 64
    comptime BN = 64  # must equal HD for GPT-2 124M
    comptime BK = 16  # BM*BK = BN*BK = 1024 = THREADS
    comptime THREADS = BM * BK  # = 1024
    var tiles_m = ceildiv(T, BM)  # = T/64 = 16
    var tiles_n = ceildiv(HD, BN)  # = HD/64 = 1 (when HD=BN=64)
    comptime k = _attn_headout4_gpu[dtype, BM, BN, BK]
    var compiled = ctx.compile_function[k]()
    ctx.enqueue_function(
        compiled,
        BH,
        T,
        HD,
        T,
        a_ptr,
        b_ptr,
        c_ptr,
        T * T,
        T * HD,
        T * HD,
        grid_dim=(tiles_n, tiles_m, BH),
        block_dim=(THREADS,),
    )


# ---------------------------------------------------------------------------
# Metal-only transposed-A batched kernel (4-rows-per-thread):
# C[BH,M,N] = A[BH,K,M]ᵀ · B[BH,K,N]   (i.e. C[m,n] = Σ_k A[k,m]·B[k,n])
#
# This is the "transpose_a" GEMM that dK = dSᵀ·Q and dV = Pᵀ·dO need. Instead of
# the old three-pass path (rect-transpose the [T,HD] operand → generic 1-elem/
# thread kvgrad GEMM producing dKᵀ/dVᵀ → rect-transpose the result back), it
# folds the A-transpose into the shared-memory load and reuses headout4's fast
# register-tiled inner loop, so dK/dV each cost one launch with zero transpose
# traffic and the same 4-FMA-per-B-read efficiency as dQ.
#
# A stored [BH,K,M] = the [BH,T,T] dS/P plane (K=query axis, M=key axis).
# B stored [BH,K,N] = the [BH,T,HD] Q/dO plane (K=query axis, N=head dim).
# The A-load is coalesced (consecutive threads → consecutive M, which is the
# contiguous inner stride of the [K,M] plane) and staged into shared transposed,
# so the inner loop reads A_sh[kk, m] as a per-SIMD broadcast exactly like
# headout4. Grid: (tiles_n, tiles_m, BH) 3D — no per-block integer division.
# ---------------------------------------------------------------------------
@always_inline
def _attn_headout4_transA_gpu[
    dt: DType,
    BM: Int,  # output rows per block (over M = T); BM × BK == THREADS
    BN: Int,  # output cols per block (= HD = 64); BK × BN == THREADS
    BK: Int,  # K-tile step over the summed query axis
](
    bh: Int,
    M: Int,  # T (key axis → dK/dV rows)
    N: Int,  # HD
    K: Int,  # T (query axis, summed)
    a_ptr: ImmutKernelPtr[dt],  # [BH, K, M]  — dS or P
    b_ptr: ImmutKernelPtr[dt],  # [BH, K, N]  — Q or dO
    c_ptr: MutKernelPtr[dt],  # [BH, M, N]  — dK or dV
    a_stride: Int,  # K*M
    b_stride: Int,  # K*N
    c_stride: Int,  # M*N
) -> None:
    comptime ROWS_PER_THREAD = BM * BN // (BM * BK)  # = BN // BK
    var a_sh = LayoutTensor[
        dt,
        Layout.row_major(BK, BM + 1),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var b_sh = LayoutTensor[
        dt,
        Layout.row_major(BK, BN + 1),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var tid = Int(thread_idx.x)
    var bh_idx = Int(block_idx.z)
    var tile_m = Int(block_idx.y)
    var tile_n = Int(block_idx.x)
    var m_off = tile_m * BM
    var n_off = tile_n * BN

    var ty = tid // BN  # output row-group [0, BM/ROWS)
    var tx = tid % BN  # output col       [0, BN)

    var a_base = bh_idx * a_stride
    var b_base = bh_idx * b_stride
    var c_base = bh_idx * c_stride + m_off * N + n_off

    var acc0 = Scalar[dt](0.0)
    var acc1 = Scalar[dt](0.0)
    var acc2 = Scalar[dt](0.0)
    var acc3 = Scalar[dt](0.0)

    for k in range(0, K, BK):
        # Load A tile [BK, BM] transposed into a_sh[k_local, m_local]:
        # A[k0+k_local, m_off+m_local]. Row-major [K,M] → m is contiguous, so
        # consecutive tids (→ consecutive m_local) coalesce.
        # Same K-remainder zero-pad as _attn_headout4_gpu (K = T here): the
        # comptime-unrolled kk loop always runs BK steps, so an unaligned T would
        # otherwise spill garbage into the dK/dV accumulators on Metal.
        var ra = tid // BM  # k-row in tile  [0, BK)
        var ca = tid % BM  # m-col in tile  [0, BM)
        a_sh.ptr[ra * (BM + 1) + ca] = a_ptr[
            a_base + (k + ra) * M + m_off + ca
        ] if (k + ra < K and m_off + ca < M) else Scalar[dt](0)

        # Load B tile [BK, BN] into b_sh[k_local, n_local]: B[k0+k_local, n_off+n].
        var rb = tid // BN
        var cb = tid % BN
        b_sh.ptr[rb * (BN + 1) + cb] = b_ptr[
            b_base + (k + rb) * N + n_off + cb
        ] if (k + rb < K and n_off + cb < N) else Scalar[dt](0)

        barrier()

        # 4 FMAs per B-read: A_sh[kk, m] is a per-SIMD broadcast (ty uniform in a
        # SIMD group); b_sh[kk, tx] cycles all banks over tx.
        comptime for kk in range(BK):
            var bv = b_sh.ptr[kk * (BN + 1) + tx]
            acc0 = fma(
                a_sh.ptr[kk * (BM + 1) + ty * ROWS_PER_THREAD + 0], bv, acc0
            )
            acc1 = fma(
                a_sh.ptr[kk * (BM + 1) + ty * ROWS_PER_THREAD + 1], bv, acc1
            )
            acc2 = fma(
                a_sh.ptr[kk * (BM + 1) + ty * ROWS_PER_THREAD + 2], bv, acc2
            )
            acc3 = fma(
                a_sh.ptr[kk * (BM + 1) + ty * ROWS_PER_THREAD + 3], bv, acc3
            )

        barrier()

    var row_base = m_off + ty * ROWS_PER_THREAD
    var c_off = c_base + ty * ROWS_PER_THREAD * N + tx
    if row_base + 0 < M and n_off + tx < N:
        c_ptr[c_off + 0 * N] = acc0
    if row_base + 1 < M and n_off + tx < N:
        c_ptr[c_off + 1 * N] = acc1
    if row_base + 2 < M and n_off + tx < N:
        c_ptr[c_off + 2 * N] = acc2
    if row_base + 3 < M and n_off + tx < N:
        c_ptr[c_off + 3 * N] = acc3


def _launch_headout4_transA[
    dtype: DType
](
    a_ptr: ImmutKernelPtr[dtype],  # [BH, T, T]   — dS or P
    b_ptr: ImmutKernelPtr[dtype],  # [BH, T, HD]  — Q or dO
    c_ptr: MutKernelPtr[dtype],  # [BH, T, HD]  — dK or dV
    BH: Int,
    T: Int,
    HD: Int,
    ctx: DeviceContext,
) raises:
    # C[BH,T,HD] = A[BH,T,T]ᵀ · B[BH,T,HD]  (dK = dSᵀ·Q, dV = Pᵀ·dO).
    # Mirrors _launch_headout4's geometry: BM=64 rows, BN=HD=64, BK=16,
    # THREADS=1024, ROWS_PER_THREAD=4. K (the summed query axis) = T.
    comptime BM = 64
    comptime BN = 64  # must equal HD for GPT-2 124M
    comptime BK = 16  # BM*BK = BN*BK = 1024 = THREADS
    comptime THREADS = BM * BK  # = 1024
    var tiles_m = ceildiv(T, BM)
    var tiles_n = ceildiv(HD, BN)  # = 1 when HD=BN=64
    comptime k = _attn_headout4_transA_gpu[dtype, BM, BN, BK]
    var compiled = ctx.compile_function[k]()
    ctx.enqueue_function(
        compiled,
        BH,
        T,
        HD,
        T,
        a_ptr,
        b_ptr,
        c_ptr,
        T * T,
        T * HD,
        T * HD,
        grid_dim=(tiles_n, tiles_m, BH),
        block_dim=(THREADS,),
    )


# ---------------------------------------------------------------------------
# Metal-only batched scoreout kernel: C[BH,T,T] = A[BH,T,HD] · B[BH,T,HD]ᵀ
#
# K = HD = 64 is small enough to load the ENTIRE K-dimension into shared memory
# in one step — no outer K-tile loop, one barrier, 64 FMAs per thread.
# Grid: BH × (T/BM) × (T/BN) = 48×32×32 = 49152 blocks at 1024 threads each.
# Shared mem: A_sh[32,65] + B_sh[32,65] = 16640 B per block.
# ---------------------------------------------------------------------------
@always_inline
def _attn_batched_scoreout_gpu[
    dt: DType,
    BM: Int,  # output rows per block (= N tile)  — both BM and BN must equal T//tiles
    BN: Int,  # output cols per block
    BK: Int,  # = HD = 64; entire K loaded in one tile (BM//2 * BK must equal THREADS)
](
    bh: Int,
    M: Int,  # T
    N: Int,  # T
    K: Int,  # HD
    a_ptr: ImmutKernelPtr[dt],  # [BH, T, HD] — Q or dO
    b_ptr: ImmutKernelPtr[
        dt
    ],  # [BH, T, HD] — K or V (accessed as row-major, col is K dim)
    c_ptr: MutKernelPtr[dt],  # [BH, T, T]  — output scores (written in fp32)
    tiles_m: Int,
    tiles_n: Int,
) -> None:
    # Shared mem: A_sh[BM, BK+1] and B_sh[BN, BK+1] — +1 padding avoids bank conflicts.
    var a_sh = LayoutTensor[
        dt,
        Layout.row_major(BM, BK + 1),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var b_sh = LayoutTensor[
        dt,
        Layout.row_major(BN, BK + 1),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var tid = Int(thread_idx.x)  # [0, BM*BN = THREADS)
    var ty = tid // BN  # output row in C-tile: [0, BM)
    var tx = tid % BN  # output col in C-tile: [0, BN)

    var bid = Int(block_idx.x)
    var bh_idx = bid // (tiles_m * tiles_n)
    var rem = bid % (tiles_m * tiles_n)
    var tile_m = rem // tiles_n
    var tile_n = rem % tiles_n
    var m_off = tile_m * BM
    var n_off = tile_n * BN

    # Base pointers into A[bh,m_off,0] and B[bh,n_off,0] and C[bh,m_off,n_off]
    var a_base = bh_idx * M * K + m_off * K
    var b_base = bh_idx * N * K + n_off * K
    var c_base = bh_idx * M * N + m_off * N + n_off

    # Load A tile [BM, BK] = 2*THREADS elements using THREADS threads (2 each).
    # THREADS = BM*BN but the A tile has BM*BK elements. We require BM*BK = 2*THREADS
    # i.e. BK = 2*BN, which holds for BM=BN=32 and BK=64.
    # Pass 1: tid in [0, THREADS) → flat element index ia = tid
    #         ia // BK = row ∈ [0, BM//2), ia % BK = col ∈ [0, BK)
    # Pass 2: ia = tid + BM//2 * BK (= tid + THREADS) → row ∈ [BM//2, BM)
    var ia = tid
    a_sh.ptr[ia // BK * (BK + 1) + ia % BK] = a_ptr[
        a_base + ia // BK * K + ia % BK
    ]
    ia = tid + BM // 2 * BK
    a_sh.ptr[ia // BK * (BK + 1) + ia % BK] = a_ptr[
        a_base + ia // BK * K + ia % BK
    ]

    # Load B tile [BN, BK] = 2*THREADS elements; same pattern as A.
    var ib = tid
    b_sh.ptr[ib // BK * (BK + 1) + ib % BK] = b_ptr[
        b_base + ib // BK * K + ib % BK
    ]
    ib = tid + BN // 2 * BK
    b_sh.ptr[ib // BK * (BK + 1) + ib % BK] = b_ptr[
        b_base + ib // BK * K + ib % BK
    ]

    barrier()

    # Compute: C[ty, tx] = Σ_{kk=0}^{K-1} A_sh[ty, kk] · B_sh[tx, kk]
    # A_sh read: a_sh.ptr[ty * (BK+1) + kk] — ty is same for all threads in a SIMD group
    #            → BROADCAST (32 threads same location) ✓
    # B_sh read: b_sh.ptr[tx * (BK+1) + kk] — tx cycles 0..31 within SIMD
    #            bank = (tx * 65 + kk) % 32 = (tx + kk) % 32 → all 32 banks ✓
    var acc = Scalar[dt](0.0)
    for kk in range(K):
        acc = fma(
            a_sh.ptr[ty * (BK + 1) + kk], b_sh.ptr[tx * (BK + 1) + kk], acc
        )

    if m_off + ty < M and n_off + tx < N:
        c_ptr[c_base + ty * N + tx] = acc


def _launch_batched_scoreout[
    dtype: DType
](
    a_ptr: ImmutKernelPtr[dtype],  # [BH, T, HD] — Q or dO
    b_ptr: ImmutKernelPtr[dtype],  # [BH, T, HD] — K or V
    c_ptr: MutKernelPtr[dtype],  # [BH, T, T]  — output
    BH: Int,
    T: Int,
    HD: Int,
    ctx: DeviceContext,
) raises:
    # C[BH,T,T] = A[BH,T,HD] · B[BH,T,HD]ᵀ in one launch covering all BH heads.
    # Tile sizes: BM=BN=32, BK=HD=64 (full K in shared mem; BK = 2*BN required).
    comptime BM = 32
    comptime BN = 32
    comptime BK = 64  # must equal HD for GPT-2 124M
    comptime THREADS = BM * BN  # = 1024
    var tiles_m = ceildiv(T, BM)
    var tiles_n = ceildiv(T, BN)
    var num_blocks = BH * tiles_m * tiles_n  # 48 × 32 × 32 = 49152 at T=1024
    comptime k = _attn_batched_scoreout_gpu[dtype, BM, BN, BK]
    var compiled = ctx.compile_function[k]()
    ctx.enqueue_function(
        compiled,
        BH,
        T,
        T,
        HD,
        a_ptr,
        b_ptr,
        c_ptr,
        tiles_m,
        tiles_n,
        grid_dim=(num_blocks,),
        block_dim=(THREADS,),
    )


@always_inline
def _cublas_dt[dt: DType]() -> DataType:
    comptime if dt == DType.bfloat16:
        return DataType.R_16BF
    elif dt == DType.float32:
        return DataType.R_32F
    else:
        return DataType.R_16F


def _launch_transpose_planes[
    dtype: DType,
](
    src_ptr: ImmutKernelPtr[dtype],
    dst_ptr: MutKernelPtr[dtype],
    num_planes: Int,
    seq_len: Int,
    ctx: DeviceContext,
) raises:
    # Metal helper: per-plane transpose of a [num_planes, T, T] buffer via the
    # coalesced 32×32-tile kernel. Used so the dK/dV gradient GEMMs can avoid the
    # transpose_a linalg.matmul cannot express (#6626) — we materialise dSᵀ / Pᵀ
    # and run plain (transpose_a-free) per-head GEMMs.
    comptime BLOCK_SIZE = 256
    comptime SM_OVERPROVISION = 32
    var tiles = ceildiv(seq_len, TRANSPOSE_TILE)
    var total_tiles = num_planes * tiles * tiles
    var num_sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var num_blocks = max(min(total_tiles, SM_OVERPROVISION * num_sm), 1)
    comptime k = _attention_transpose_planes_gpu[dtype, BLOCK_SIZE]
    var compiled = ctx.compile_function[k]()
    ctx.enqueue_function(
        compiled,
        num_planes,
        seq_len,
        src_ptr,
        dst_ptr,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )


def _launch_rect_transpose[
    dtype: DType,
](
    src_ptr: ImmutKernelPtr[dtype],
    dst_ptr: MutKernelPtr[dtype],
    num_planes: Int,
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    # NOTE: No longer on the Metal backward path — the four Phase-C rect-transposes
    # (Q→Qᵀ, dKᵀ→dK, dO→dOᵀ, dVᵀ→dV) were eliminated by folding the A-transpose
    # into `_launch_headout4_transA`'s shared-memory load. Retained as a general
    # coalesced rect-transpose helper.
    # Metal helper: per-plane transpose of a [num_planes, rows, cols] buffer →
    # [num_planes, cols, rows] via the coalesced 32×32-tile rect-transpose kernel.
    # Used for the small [T,HD]↔[HD,T] transposes in Phase C of the backward pass
    # (transposing Q and dO before their GEMMs, and transposing dKᵀ/dVᵀ after).
    # Each [T,HD] plane is 256 KB (16× smaller than a [T,T] plane at T=1024,HD=64),
    # making this ~16× cheaper than _launch_transpose_planes on the same BH planes.
    comptime BLOCK_SIZE = 256
    comptime SM_OVERPROVISION = 32
    var tiles_r = ceildiv(rows, TRANSPOSE_TILE)
    var tiles_c = ceildiv(cols, TRANSPOSE_TILE)
    var total_tiles = num_planes * tiles_r * tiles_c
    var num_sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var num_blocks = max(min(total_tiles, SM_OVERPROVISION * num_sm), 1)
    comptime k = _attention_transpose_rect_gpu[dtype, BLOCK_SIZE]
    var compiled = ctx.compile_function[k]()
    ctx.enqueue_function(
        compiled,
        num_planes,
        rows,
        cols,
        src_ptr,
        dst_ptr,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )


@always_inline
def _attn_gemm_batched_metal[
    dt: DType,
](
    a_ptr: ImmutKernelPtr[dt],
    b_ptr: ImmutKernelPtr[dt],
    c_ptr: MutKernelPtr[dt],
    M: Int,
    N: Int,
    K: Int,
    a_stride: Int,
    b_stride: Int,
    c_stride: Int,
    transpose_b: Bool,
    BH: Int,
    ctx: DeviceContext,
) raises:
    # Metal (vendor-neutral) batched GEMM: one `linalg.matmul` per head over the
    # BH contiguous [.,.] planes. Row-major C[M,N] = A[M,K] · op_b(B), where
    # op_b(B) is B[K,N] (transpose_b=False) or B[N,K]ᵀ (transpose_b=True). Matches
    # cuBLAS's per-head semantics for the transpose_a=False calls (QKᵀ, A·V, dQ,
    # dO·Vᵀ); the two transpose_a=True gradients (dK, dV) are handled by the
    # caller pre-transposing the square [T,T] plane (see attention_bwd_gemm).
    #
    # No synchronize() brackets: linalg.matmul enqueues on ctx's stream, so the
    # per-head GEMMs are ordered against the surrounding softmax / P+dS / transpose
    # kernels (also on ctx's stream) without host fences. The forward/backward
    # callers place the one required drain (attention_fwd_gemm's trailing fence /
    # attention_bwd_gemm's final synchronize). Dropping the per-call brackets
    # removed ~11 full-GPU drains per backward layer.
    if transpose_b:
        for bh in range(BH):
            var a = TileTensor(
                Span[Scalar[dt], ImmutAnyOrigin](
                    ptr=a_ptr + bh * a_stride, length=M * K
                ),
                row_major(M, K),
            )
            var b = TileTensor(
                Span[Scalar[dt], ImmutAnyOrigin](
                    ptr=b_ptr + bh * b_stride, length=N * K
                ),
                row_major(N, K),
            )
            var c = TileTensor(
                Span[Scalar[dt], MutAnyOrigin](
                    ptr=c_ptr + bh * c_stride, length=M * N
                ),
                row_major(M, N),
            )
            matmul[transpose_b=True, target="gpu"](c, a, b, ctx=ctx)
    else:
        for bh in range(BH):
            var a = TileTensor(
                Span[Scalar[dt], ImmutAnyOrigin](
                    ptr=a_ptr + bh * a_stride, length=M * K
                ),
                row_major(M, K),
            )
            var b = TileTensor(
                Span[Scalar[dt], ImmutAnyOrigin](
                    ptr=b_ptr + bh * b_stride, length=K * N
                ),
                row_major(K, N),
            )
            var c = TileTensor(
                Span[Scalar[dt], MutAnyOrigin](
                    ptr=c_ptr + bh * c_stride, length=M * N
                ),
                row_major(M, N),
            )
            matmul[transpose_b=False, target="gpu"](c, a, b, ctx=ctx)


@always_inline
def _attn_gemm_batched[
    a_dt: DType,
    b_dt: DType,
    c_dt: DType,
](
    a_ptr: ImmutKernelPtr[a_dt],
    b_ptr: ImmutKernelPtr[b_dt],
    c_ptr: MutKernelPtr[c_dt],
    M: Int,
    N: Int,
    K: Int,
    a_stride: Int,
    b_stride: Int,
    c_stride: Int,
    transpose_b: Bool,
    BH: Int,
    ctx: DeviceContext,
    transpose_a: Bool = False,
) raises:
    comptime if HAS_METAL:
        # All three operands share a dtype on the Metal attention path (fp32/bf16),
        # so the rebinds are identity. transpose_a is never True here — the caller
        # pre-transposes for dK/dV (linalg.matmul has no transpose_a).
        _attn_gemm_batched_metal[a_dt](
            a_ptr,
            rebind[ImmutKernelPtr[a_dt]](b_ptr),
            rebind[MutKernelPtr[a_dt]](c_ptr),
            M,
            N,
            K,
            a_stride,
            b_stride,
            c_stride,
            transpose_b,
            BH,
            ctx,
        )
        return
    # All BH heads of a per-head GEMM C[M,N] = op(A) · op(B) in ONE
    # cublasGemmStridedBatchedEx call. This is
    # what llm.c does (cublasGemmStridedBatched): packing all heads into a single
    # kernel fills the GPU and lifts the ~16 % tensor-core utilization the small
    # per-head (head_dim=64) launches suffered. Uses the same row-major swap trick
    # as the vendor _cublas_matmul: to get row-major C = A·op(B), call cublas
    # (col-major) with the A/B operands swapped, computing the N×M col-major
    # result that equals the M×N row-major one.
    #
    # When all three operands share a dtype, route through cuBLASLt's strided-
    # batched path (llm.c's actual attention API) instead of the legacy
    # cublasGemmStridedBatchedEx — its heuristic can pick a faster kernel for the
    # head_dim=64 shapes. The col-major swap maps to _matmul_cublaslt directly:
    # A_lt=b, B_lt=a, m=N, n=M, k=K, transA=transpose_b, transB=transpose_a.
    comptime if USE_LT_ATTN and a_dt == b_dt and a_dt == c_dt:
        var d = rebind[MutKernelPtr[a_dt]](c_ptr)
        var bb = rebind[ImmutKernelPtr[a_dt]](b_ptr)
        var aa = rebind[ImmutKernelPtr[a_dt]](a_ptr)
        # bias/aux are unused with epilogue=1; pass valid dummies (never read).
        var nb = aa
        var na = d
        if transpose_b:
            if transpose_a:
                _matmul_cublaslt[a_dt, True, True](
                    d,
                    bb,
                    aa,
                    nb,
                    na,
                    N,
                    M,
                    K,
                    Int32(1),
                    False,
                    ctx,
                    BH,
                    b_stride,
                    a_stride,
                    c_stride,
                )
            else:
                _matmul_cublaslt[a_dt, True, False](
                    d,
                    bb,
                    aa,
                    nb,
                    na,
                    N,
                    M,
                    K,
                    Int32(1),
                    False,
                    ctx,
                    BH,
                    b_stride,
                    a_stride,
                    c_stride,
                )
        else:
            if transpose_a:
                _matmul_cublaslt[a_dt, False, True](
                    d,
                    bb,
                    aa,
                    nb,
                    na,
                    N,
                    M,
                    K,
                    Int32(1),
                    False,
                    ctx,
                    BH,
                    b_stride,
                    a_stride,
                    c_stride,
                )
            else:
                _matmul_cublaslt[a_dt, False, False](
                    d,
                    bb,
                    aa,
                    nb,
                    na,
                    N,
                    M,
                    K,
                    Int32(1),
                    False,
                    ctx,
                    BH,
                    b_stride,
                    a_stride,
                    c_stride,
                )
        return

    # cuBLAS-symbol-using tail: comptime-excluded on Metal so its FFI symbols are
    # never referenced (they don't link on a Metal build). The Metal branch above
    # returns before here.
    comptime if not HAS_METAL:
        # Same TF32 gate as _matmul_cublaslt (llmm/matmul.mojo): fp32 attention
        # GEMMs (QKᵀ/A·V fwd, dQ/dK/dV bwd) route through TF32 tensor cores by
        # default, matching llm.c's fp32 arm (cublasSgemmStridedBatched honors
        # the handle's CUBLAS_TF32_TENSOR_OP_MATH math mode on cc>=8.0). bf16
        # is unaffected. Disable with -D LLMM_NO_TF32=1.
        comptime compute_type = (
            ComputeType.COMPUTE_32F_FAST_TF32 if (
                a_dt == DType.float32 and USE_TF32
            ) else ComputeType.COMPUTE_32F
        )
        var handle = _get_global_handle[a_dt](ctx)
        var alpha = Float32(1.0)
        var beta = Float32(0.0)
        var op_b = _convert_to_cublas_transpose(transpose_b)
        var op_a = _convert_to_cublas_transpose(transpose_a)
        var lda = K if transpose_b else N
        var ldb = M if transpose_a else K
        check_cublas_error(
            cublasGemmStridedBatchedEx(
                handle._get_cublas(),
                op_b,
                op_a,
                Int64(N),
                Int64(M),
                Int64(K),
                UnsafePointer(to=alpha)
                .bitcast[NoneType]()
                .as_immutable()
                .as_unsafe_any_origin(),
                b_ptr.bitcast[NoneType]().as_immutable().as_unsafe_any_origin(),
                _cublas_dt[b_dt](),
                Int64(lda),
                Int64(b_stride),
                a_ptr.bitcast[NoneType]().as_immutable().as_unsafe_any_origin(),
                _cublas_dt[a_dt](),
                Int64(ldb),
                Int64(a_stride),
                UnsafePointer(to=beta)
                .bitcast[NoneType]()
                .as_immutable()
                .as_unsafe_any_origin(),
                c_ptr.bitcast[NoneType]().as_unsafe_any_origin(),
                _cublas_dt[c_dt](),
                Int64(N),
                Int64(c_stride),
                Int64(BH),
                compute_type,
                Algorithm.DEFAULT,
            )
        )


@always_inline
def _attention_bmm_kvgrad[
    dtype: DType,
    target: StaticString,
](
    at_ptr: ImmutKernelPtr[dtype],
    big_ptr: ImmutKernelPtr[dtype],
    out_ptr: MutKernelPtr[dtype],
    BH: Int,
    T: Int,
    hd: Int,
    ctx: DeviceContext,
) capturing raises:
    # NOTE: No longer on the Metal backward path. dK/dV now use the fused
    # transpose_a kernel `_launch_headout4_transA` (dK = dSᵀ·Q, dV = Pᵀ·dO in one
    # launch each, zero transpose passes), which is ~2.4× faster than this generic
    # 1-elem/thread kvgrad GEMM and eliminates the four rect-transposes that used
    # to bracket it. Retained for the cuBLAS/reference path and as a fallback.
    # C[hd,T] = A[hd,T] · B[T,T], all heads batched. With A = Qᵀ, B = dS this is
    # dKᵀ; with A = dOᵀ, B = P this is dVᵀ — letting dK/dV skip transposing [T,T].
    # Metal: dispatch all BH heads in a single custom tiled kernel instead of
    # 48 serial linalg.matmul calls, saturating the GPU with enough tiles.
    comptime if HAS_METAL:
        _launch_batched_kvgrad[dtype](at_ptr, big_ptr, out_ptr, BH, T, hd, ctx)
        return
    _attn_gemm_batched[dtype, dtype, dtype](
        at_ptr,
        big_ptr,
        out_ptr,
        hd,
        T,
        T,
        hd * T,
        T * T,
        hd * T,
        False,
        BH,
        ctx,
    )


@always_inline
def _attention_bmm_scoreout[
    dtype: DType,
    out_dtype: DType,
    target: StaticString,
](
    a_ptr: ImmutKernelPtr[dtype],
    b_ptr: ImmutKernelPtr[dtype],
    out_ptr: MutKernelPtr[out_dtype],
    BH: Int,
    T: Int,
    hd: Int,
    ctx: DeviceContext,
) capturing raises:
    # C[T,T] = A[T,hd] · B[T,hd]ᵀ, all heads batched. out_dtype selects the
    # written-out precision: bf16 for QKᵀ (feeds only exp in the P recompute) and
    # (optionally) fp32 for dO·Vᵀ. cuBLAS writes the fp32 accumulator straight
    # into c; the scale is deferred to the softmax/dS.
    #
    # Metal: dispatch all BH heads in one tiled kernel (_launch_batched_scoreout)
    # instead of BH serial linalg.matmul calls. It only wins while QKᵀ is
    # dispatch-bound at short T (T=64: 164 vs 261 ms/step); once T is long enough
    # that QKᵀ is compute-bound the per-head matmul reaches more of peak and wins
    # (T=1024: 718 batched vs 502 fallback, M4 Max B=4 bf16), so gate the
    # batched path to T <= SCOREOUT_MAX_T. Aligned-T-only contract (no load-side
    # bounds check); odd T falls through to the per-head path below.
    comptime if HAS_METAL and dtype == out_dtype:
        if (
            hd == SCOREOUT_HEAD_DIM
            and T % SCOREOUT_T_MULTIPLE == 0
            and T <= SCOREOUT_MAX_T
        ):
            _launch_batched_scoreout[dtype](
                a_ptr,
                b_ptr,
                rebind[MutKernelPtr[dtype]](out_ptr),
                BH,
                T,
                hd,
                ctx,
            )
            return
    _attn_gemm_batched[dtype, dtype, out_dtype](
        a_ptr,
        b_ptr,
        out_ptr,
        T,
        T,
        hd,
        T * hd,
        T * hd,
        T * T,
        True,
        BH,
        ctx,
    )


@always_inline
def _attention_bmm_headout[
    dtype: DType,
    target: StaticString,
](
    a_ptr: ImmutKernelPtr[dtype],
    b_ptr: ImmutKernelPtr[dtype],
    out_ptr: MutKernelPtr[dtype],
    BH: Int,
    T: Int,
    hd: Int,
    ctx: DeviceContext,
) capturing raises:
    # C[T,hd] = A[T,T] · B[T,hd], all heads batched. dQ = dS·K (the 1/√d scale is
    # already folded into dS); also the forward A·V. No epilogue.
    # Metal: use the 4-rows-per-thread batched kernel. BM=64 rows per block with
    # BM×BK=BK×BN=THREADS=1024 gives 100% A/B load efficiency. 3D grid eliminates
    # integer divisions per block. All 48 heads in one launch vs 48 serial matmuls.
    comptime if HAS_METAL:
        _launch_headout4[dtype](a_ptr, b_ptr, out_ptr, BH, T, hd, ctx)
        return
    _attn_gemm_batched[dtype, dtype, dtype](
        a_ptr,
        b_ptr,
        out_ptr,
        T,
        hd,
        T,
        T * T,
        T * hd,
        T * hd,
        False,
        BH,
        ctx,
    )


def attention_bwd_gemm[
    dtype: DType,
    target: StaticString,
](
    d_query_ptr: MutKernelPtr[dtype],
    d_key_ptr: MutKernelPtr[dtype],
    d_value_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    query_ptr: ImmutKernelPtr[dtype],
    key_ptr: ImmutKernelPtr[dtype],
    value_ptr: ImmutKernelPtr[dtype],
    output_ptr: ImmutKernelPtr[dtype],
    log_sum_exp_ptr: ImmutKernelPtr[DType.float32],
    batch_size: Int64,
    num_heads: Int64,
    seq_len: Int64,
    head_dim: Int64,
    ctx: DeviceContext,
    cache: Optional[KVCachePtr] = None,
) capturing raises:
    var B = Int(batch_size)
    var NH = Int(num_heads)
    var T = Int(seq_len)
    var hd = Int(head_dim)
    var BH = B * NH
    var plane = BH * T * T
    var attention_scale = Scalar[DType.float32](1.0) / sqrt(
        Scalar[DType.float32](hd)
    )
    var device_ctx = ctx

    # Persistent scratch (allocated once, kept alive in the KVCache). The f32
    # score buffer holds QKᵀ then is read by the P-recompute kernel; dP has its
    # own fp32 buffer; P reuses the forward probability buffer.
    var sc_addr = 0
    var p_addr = 0
    var ds_addr = 0
    var dst_addr = 0
    var pt_addr = 0
    var d_addr = 0
    var dp_addr = 0
    if cache:
        sc_addr = cache.value()[].gemm_scores_addr
        p_addr = cache.value()[].gemm_att_addr
        ds_addr = cache.value()[].gemm_ds_addr
        dst_addr = cache.value()[].gemm_dst_addr
        pt_addr = cache.value()[].gemm_pt_addr
        d_addr = cache.value()[].gemm_d_addr
        dp_addr = cache.value()[].gemm_dp_addr

    # gemm_scores_addr is bf16 (the forward allocates it bf16); in the backward it
    # is only Phase-C dKᵀ scratch, so the types must agree.
    var sc = _gemm_scratch_buffer[dtype](sc_addr, plane, device_ctx)
    var pp = _gemm_scratch_buffer[dtype](
        p_addr, plane, device_ctx, zero_on_alloc=True
    )
    var dss = _gemm_scratch_buffer[dtype](
        ds_addr, plane, device_ctx, zero_on_alloc=True
    )
    var dstt = _gemm_scratch_buffer[dtype](dst_addr, plane, device_ctx)
    var ptt = _gemm_scratch_buffer[dtype](pt_addr, plane, device_ctx)
    var dd = _gemm_scratch_buffer[DType.float32](d_addr, BH * T, device_ctx)
    var dpp = _gemm_scratch_buffer[DType.float32](dp_addr, plane, device_ctx)
    if cache:
        cache.value()[].gemm_scores_addr = sc[0]
        cache.value()[].gemm_att_addr = pp[0]
        cache.value()[].gemm_ds_addr = dss[0]
        cache.value()[].gemm_dst_addr = dstt[0]
        cache.value()[].gemm_pt_addr = ptt[0]
        cache.value()[].gemm_d_addr = dd[0]
        cache.value()[].gemm_dp_addr = dpp[0]

    var score_buf = sc[1]
    var p_buf = pp[1]
    var ds_buf = dss[1]
    var dst_buf = dstt[1]
    var pt_buf = ptt[1]
    var d_buf = dd[1]
    var dp_buf = dpp[1]
    # Backward scores are bf16 (they only feed exp): QKᵀ writes them into pt_buf,
    # which is free until Phase C reuses it for Qᵀ. Halves the P+dS read pass.
    var scores_immut = rebind[ImmutKernelPtr[dtype]](pt_buf)
    var dp_immut = rebind[ImmutKernelPtr[dtype]](dst_buf)
    var p_immut = rebind[ImmutKernelPtr[dtype]](p_buf)
    var d_immut = rebind[ImmutKernelPtr[DType.float32]](d_buf)
    # storage: read the forward's stored probs (this layer's att_probs slice)
    # instead of recomputing QKᵀ → P. `scores_immut` then points at the stored P.
    var use_stored = cache and cache.value()[].att_probs_addr != 0
    if use_stored:
        var c = cache.value()
        var ap = UnsafePointer[
            mut=True, type=Scalar[dtype], origin=MutUntrackedOrigin
        ](unsafe_from_address=c[].att_probs_addr)
        var att_slice = ap + c[].att_probs_layer * c[].att_probs_stride
        scores_immut = rebind[ImmutKernelPtr[dtype]](att_slice)
        p_immut = rebind[ImmutKernelPtr[dtype]](att_slice)
    # NOTE (Metal): do NOT reuse the forward's gemm_att scratch (p_buf) as the
    # stored P here. The forward writes P into that single reused buffer per
    # layer, and forward/backward run as two separate whole-model loops, so by
    # the time the backward for layer L runs, p_buf holds the LAST forward
    # layer's P — not layer L's. Reusing it aliased every backward layer to the
    # final layer's probabilities, producing per-layer-amplified gradients that
    # compounded with depth. Instead, when the store is disabled (att_probs_addr
    # == 0) we fall through to the true recompute path below: QKᵀ is recomputed
    # from this layer's Q/K into pt_buf, and pds_recompute re-derives P from the
    # per-layer saved log-sum-exp — both correct and self-contained per layer.
    # score_buf / dp_buf (fp32) are retained only as Phase-C dKᵀ/dVᵀ scratch.
    # See docs/ai/metal_port_gotchas_and_optimizations.md P12 for the full story.
    _ = score_buf

    comptime BLOCK_SIZE = 256
    comptime SM_OVERPROVISION = 32
    var num_sm = device_ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    @parameter
    def _grid(work: Int) -> Int:
        return max(min(ceildiv(work, BLOCK_SIZE), SM_OVERPROVISION * num_sm), 1)

    # Env-gated per-phase timing (LLMM_ATTN_PROFILE build): inserts synchronize
    # fences and prints per-phase ms. Compiled out of the production path — the
    # timer variable itself only exists in the profiling build.
    comptime _PROF = is_defined["LLMM_ATTN_PROFILE"]()
    var _tp = UInt64(0)
    comptime if _PROF:
        _tp = global_perf_counter_ns()

    # Phase A: dP = dO·Vᵀ. QKᵀ (scores) is only recomputed when the forward's
    # probs were NOT stored — otherwise we read them directly (skip a big matmul).
    if not use_stored:
        _attention_bmm_scoreout[dtype, dtype, target](
            query_ptr, key_ptr, pt_buf, BH, T, hd, device_ctx
        )
    _attention_bmm_scoreout[dtype, dtype, target](
        d_output_ptr, value_ptr, dst_buf, BH, T, hd, device_ctx
    )
    comptime if _PROF:
        device_ctx.synchronize()
        var _n = global_perf_counter_ns()
        print("  A scoreout(dP)", Float64(_n - _tp) / 1e6, "ms")
        _tp = _n
    # No fence: Phase B runs on the same stream and is ordered after Phase A's
    # cuBLAS writes to pt_buf/dst_buf.

    # Phase B (default-stream kernels, ordered with no inter-fences): D_i, then
    # the fused P-recompute + dS over the scores plane.
    comptime d_kernel = _attention_bwd_rowdot_gpu[dtype]
    var d_compiled = device_ctx.compile_function[d_kernel]()
    device_ctx.enqueue_function(
        d_compiled,
        BH * T,
        hd,
        d_output_ptr,
        output_ptr,
        d_buf,
        grid_dim=(_grid(BH * T),),
        block_dim=(BLOCK_SIZE,),
    )
    comptime if _PROF:
        device_ctx.synchronize()
        var _n = global_perf_counter_ns()
        print("  B rowdot(D)", Float64(_n - _tp) / 1e6, "ms")
        _tp = _n
    # Vectorized: each thread handles `PDS_WIDTH` contiguous elements. T is a
    # power of two ≥ 32, so PDS_WIDTH=8 divides plane and never straddles a row
    # for production shapes. The equivalence suite's odd seq_len (e.g. 7)
    # violates that, so dispatch a scalar per-element fallback in that case —
    # see _attention_bwd_p_and_ds_gpu's docstring.
    comptime PDS_WIDTH = 8
    var pds_aligned = T % PDS_WIDTH == 0
    var pds_blocks = plane // PDS_WIDTH if pds_aligned else plane
    if use_stored:
        if pds_aligned:
            comptime pds_stored = _attention_bwd_p_and_ds_gpu[
                dtype, PDS_WIDTH, stored_p=True, aligned=True
            ]
            var pds_c = device_ctx.compile_function[pds_stored]()
            device_ctx.enqueue_function(
                pds_c,
                pds_blocks,
                T,
                attention_scale,
                scores_immut,
                log_sum_exp_ptr,
                dp_immut,
                d_immut,
                p_buf,
                ds_buf,
                grid_dim=(_grid(pds_blocks),),
                block_dim=(BLOCK_SIZE,),
            )
        else:
            comptime pds_stored_u = _attention_bwd_p_and_ds_gpu[
                dtype, PDS_WIDTH, stored_p=True, aligned=False
            ]
            var pds_c = device_ctx.compile_function[pds_stored_u]()
            device_ctx.enqueue_function(
                pds_c,
                pds_blocks,
                T,
                attention_scale,
                scores_immut,
                log_sum_exp_ptr,
                dp_immut,
                d_immut,
                p_buf,
                ds_buf,
                grid_dim=(_grid(pds_blocks),),
                block_dim=(BLOCK_SIZE,),
            )
    else:
        if pds_aligned:
            comptime pds_recompute = _attention_bwd_p_and_ds_gpu[
                dtype, PDS_WIDTH, stored_p=False, aligned=True
            ]
            var pds_c = device_ctx.compile_function[pds_recompute]()
            device_ctx.enqueue_function(
                pds_c,
                pds_blocks,
                T,
                attention_scale,
                scores_immut,
                log_sum_exp_ptr,
                dp_immut,
                d_immut,
                p_buf,
                ds_buf,
                grid_dim=(_grid(pds_blocks),),
                block_dim=(BLOCK_SIZE,),
            )
        else:
            comptime pds_recompute_u = _attention_bwd_p_and_ds_gpu[
                dtype, PDS_WIDTH, stored_p=False, aligned=False
            ]
            var pds_c = device_ctx.compile_function[pds_recompute_u]()
            device_ctx.enqueue_function(
                pds_c,
                pds_blocks,
                T,
                attention_scale,
                scores_immut,
                log_sum_exp_ptr,
                dp_immut,
                d_immut,
                p_buf,
                ds_buf,
                grid_dim=(_grid(pds_blocks),),
                block_dim=(BLOCK_SIZE,),
            )

    comptime if _PROF:
        device_ctx.synchronize()
        var _n = global_perf_counter_ns()
        print("  B pds(P+dS)", Float64(_n - _tp) / 1e6, "ms")
        _tp = _n

    # No fence: the gradient matmuls run on the same stream, ordered after P+dS.

    # dQ = dS·K, dK = dSᵀ·Q, dV = Pᵀ·dO — all per-head GEMMs.
    # cuBLAS does transpose_a natively in one strided-batched call.
    # Metal (linalg.matmul has no transpose_a): instead of materialising the large
    # [T,T] dSᵀ / Pᵀ planes (4 MB × 48 = 192 MB each, ~754 MB transpose traffic),
    # we transpose the small [T,HD] operands Q and dO (256 KB × 48 = 12 MB each):
    #   dKᵀ[HD,T] = Qᵀ[HD,T] · dS[T,T]  then  dK[T,HD] = (dKᵀ)ᵀ
    #   dVᵀ[HD,T] = dOᵀ[HD,T] · P[T,T]  then  dV[T,HD] = (dVᵀ)ᵀ
    # Total rect-transpose traffic: 4 × 12 MB ≈ 50 MB (~15× less than the old path).
    # Scratch reuse (same stream, no aliasing): pt_buf → Qᵀ → (reused) dVᵀ;
    # score_buf → dKᵀ; dst_buf → dOᵀ. dS already carries the 1/√d scale.
    var ds_immut = rebind[ImmutKernelPtr[dtype]](ds_buf)
    _ = dp_buf
    _attention_bmm_headout[dtype, target](
        ds_immut, key_ptr, d_query_ptr, BH, T, hd, device_ctx
    )
    comptime if _PROF:
        device_ctx.synchronize()
        var _n = global_perf_counter_ns()
        print("  C dQ=dS.K headout", Float64(_n - _tp) / 1e6, "ms")
        _tp = _n
    comptime if HAS_METAL:
        # Phase C-Metal: fused transpose_a GEMMs — dK/dV are computed directly
        # from the [T,T] plane with ZERO transpose passes. The transposed-A
        # register-tiled kernel reads dS/P as the [K=query, M=key] operand and
        # Q/dO as [K=query, N=head], emitting dK[T,HD]/dV[T,HD] in one launch
        # each (headout4's 4-FMA-per-B-read efficiency). This replaces the old
        # 6-op path (4 rect-transposes of the small [T,HD] operands + 2 generic
        # 1-elem/thread kvgrad GEMMs producing dKᵀ/dVᵀ): the generic kvgrad was
        # ~2.4× slower than headout4 for identical FLOPs, and the 4 transposes
        # added launch+bandwidth overhead. dS already carries the 1/√d scale.
        #   dK[T,HD] = dSᵀ · Q   (A=dS[BH,T,T], B=Q[BH,T,HD])
        #   dV[T,HD] = Pᵀ · dO    (A=P [BH,T,T], B=dO[BH,T,HD])
        _launch_headout4_transA[dtype](
            ds_immut, query_ptr, d_key_ptr, BH, T, hd, device_ctx
        )
        _launch_headout4_transA[dtype](
            p_immut, d_output_ptr, d_value_ptr, BH, T, hd, device_ctx
        )
        comptime if _PROF:
            device_ctx.synchronize()
            var _n = global_perf_counter_ns()
            print("  C dK/dV (2 transA GEMM)", Float64(_n - _tp) / 1e6, "ms")
            _tp = _n
    else:
        _attn_gemm_batched[dtype, dtype, dtype](
            ds_immut,
            query_ptr,
            d_key_ptr,
            T,
            hd,
            T,
            T * T,
            T * hd,
            T * hd,
            False,
            BH,
            device_ctx,
            transpose_a=True,
        )
        _attn_gemm_batched[dtype, dtype, dtype](
            p_immut,
            d_output_ptr,
            d_value_ptr,
            T,
            hd,
            T,
            T * T,
            T * hd,
            T * hd,
            False,
            BH,
            device_ctx,
            transpose_a=True,
        )
    # NOTE: No explicit fence needed here — all kernels are on the same stream
    # (same DeviceContext command queue), so GPU ordering is guaranteed without
    # a CPU-GPU synchronize. The caller's ctx.synchronize() or any downstream
    # same-stream GPU op (split_bwd, optimizer, etc.) will correctly wait for
    # these gradients to be written.


def attention_bwd[
    dtype: DType,
    target: StaticString,
    use_soft_exp: Bool = True,
    use_kv_cache: Bool = True,
](
    d_query_ptr: MutKernelPtr[dtype],
    d_key_ptr: MutKernelPtr[dtype],
    d_value_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    query_ptr: ImmutKernelPtr[dtype],
    key_ptr: ImmutKernelPtr[dtype],
    value_ptr: ImmutKernelPtr[dtype],
    output_ptr: ImmutKernelPtr[dtype],
    log_sum_exp_ptr: ImmutKernelPtr[DType.float32],
    batch_size: Int64,
    num_heads: Int64,
    seq_len: Int64,
    head_dim: Int64,
    ctx: DeviceContext,
    cache: Optional[KVCachePtr] = None,
) capturing raises:
    comptime if is_cpu[target]():
        comptime simd_width = simd_width_of[dtype]()
        attention_bwd_cpu[dtype, simd_width, use_soft_exp](
            d_query_ptr,
            d_key_ptr,
            d_value_ptr,
            d_output_ptr,
            query_ptr,
            key_ptr,
            value_ptr,
            output_ptr,
            log_sum_exp_ptr,
            Int(batch_size),
            Int(num_heads),
            Int(seq_len),
            Int(head_dim),
        )
    elif is_gpu[target]():
        if Int(head_dim) > MAX_HEAD_DIM:
            raise Error("head_dim exceeds MAX_HEAD_DIM for GPU attention")
        comptime if USE_GEMM_ATTENTION:
            attention_bwd_gemm[dtype, target](
                d_query_ptr,
                d_key_ptr,
                d_value_ptr,
                d_output_ptr,
                query_ptr,
                key_ptr,
                value_ptr,
                output_ptr,
                log_sum_exp_ptr,
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                ctx,
                cache=cache,
            )
            return
        comptime simd_width = simd_width_of[dtype]()
        comptime is_float32 = dtype == DType.float32
        # Apple Metal shared-memory hard limit: 32768 bytes/threadgroup (G9).
        # This portable (non-GEMM) backward path only runs when
        # USE_GEMM_ATTENTION = HAS_CUBLAS = False — i.e. on Metal/non-Nvidia
        # hardware. The tile-size choices below are therefore provably dead on
        # Nvidia, where the early `return` above (USE_GEMM_ATTENTION path) is
        # taken instead.
        #
        # Shared-memory budgets (sizeof = bytes per element):
        #   dQ kernel:  (4·Br + 2·Bc)·128·sizeof + 2·Br·4 + Br·128·4  bytes
        #   dKV kernel: (4·Bc + 3·Br)·128·sizeof + 2·Br·4 + 2·Bc·128·4 bytes
        #
        # With the values below:
        #   float32  (sizeof=4): Br=8,  Bc=8  → dQ=25 664,  dKV=30 784 bytes ✓
        #   bfloat16 (sizeof=2): Br=8,  Bc=8  → dQ=13 376,  dKV=13 376 bytes ✓
        #
        # bf16 previously used Br=16 (BLOCK_SIZE_DQ=64, two Metal simdgroups);
        # that configuration produced intermittently corrupted dQ/dK/dV on
        # Apple M4 Max (grad norms 1e6–1e14 in training) while the fp32 Br=8
        # single-simdgroup geometry is fully validated — so bf16 pins to the
        # same Br=8. See docs/ai/metal_port_gotchas_and_optimizations.md G9, P19.
        comptime Br = 8
        comptime Bc = 8
        comptime BLOCK_SIZE_DQ = Br * 4
        comptime BLOCK_SIZE_DKV = Bc * 4
        comptime SM_OVERPROVISION = 32

        var device_ctx = ctx
        var query_tiles = ceildiv(Int(seq_len), Br)
        var kv_tiles = ceildiv(Int(seq_len), Bc)
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )

        # Pass 1: dQ - FA2 backward kernel with Q tile as outer loop (Br×Bc tiling).
        var num_tiles_dq = Int(batch_size * num_heads * Int64(query_tiles))
        var num_blocks_dq = max(min(num_tiles_dq, SM_OVERPROVISION * num_sm), 1)

        comptime gpu_kernel_dq = attention_bwd_dq_gpu[
            dtype,
            simd_width,
            Br,
            Bc,
            BLOCK_SIZE_DQ,
            use_soft_exp=use_soft_exp,
        ]

        comptime if use_kv_cache:
            var addr_dq = 0
            if cache:
                addr_dq = cache.value()[].bwd_dq_addr

            comptime CompiledTypeDQ = type_of(
                device_ctx.compile_function[gpu_kernel_dq]()
            )

            if addr_dq == 0:
                var compiled_dq = device_ctx.compile_function[gpu_kernel_dq]()
                var ptr_dq = alloc[CompiledTypeDQ](1)
                ptr_dq.unsafe_write(compiled_dq^)
                addr_dq = Int(ptr_dq)
                if cache:
                    cache.value()[].bwd_dq_addr = addr_dq

            var casted_ptr_dq = UnsafePointer[
                mut=True, type=CompiledTypeDQ, origin=MutUntrackedOrigin
            ](unsafe_from_address=addr_dq)
            var retrieved_dq = casted_ptr_dq[]

            device_ctx.enqueue_function(
                retrieved_dq,
                num_tiles_dq,
                query_tiles,
                d_query_ptr,
                query_ptr,
                key_ptr,
                value_ptr,
                output_ptr,
                d_output_ptr,
                log_sum_exp_ptr,
                seq_len,
                head_dim,
                grid_dim=(num_blocks_dq,),
                block_dim=(BLOCK_SIZE_DQ,),
            )
        else:
            var compiled_dq = device_ctx.compile_function[gpu_kernel_dq]()
            device_ctx.enqueue_function(
                compiled_dq,
                num_tiles_dq,
                query_tiles,
                d_query_ptr,
                query_ptr,
                key_ptr,
                value_ptr,
                output_ptr,
                d_output_ptr,
                log_sum_exp_ptr,
                seq_len,
                head_dim,
                grid_dim=(num_blocks_dq,),
                block_dim=(BLOCK_SIZE_DQ,),
            )

        # Pass 2: dK/dV - FA2 backward kernel with KV tile as outer loop.
        var num_tiles_dkv = Int(batch_size * num_heads * Int64(kv_tiles))
        var num_blocks_dkv = max(
            min(num_tiles_dkv, SM_OVERPROVISION * num_sm), 1
        )

        # Cache retrieve or compile dkv
        comptime gpu_kernel_dkv = attention_bwd_dkv_gpu[
            dtype,
            simd_width,
            Br,
            Bc,
            BLOCK_SIZE_DKV,
            use_soft_exp=use_soft_exp,
        ]

        comptime if use_kv_cache:
            var addr_dkv = 0
            if cache:
                addr_dkv = cache.value()[].bwd_dkv_addr

            comptime CompiledTypeDKV = type_of(
                device_ctx.compile_function[gpu_kernel_dkv]()
            )
            # Cache miss, compile the kernel and store the address.
            if addr_dkv == 0:
                var compiled_dkv = device_ctx.compile_function[gpu_kernel_dkv]()
                var ptr_dkv = alloc[CompiledTypeDKV](1)
                ptr_dkv.unsafe_write(compiled_dkv^)
                addr_dkv = Int(ptr_dkv)
                if cache:
                    cache.value()[].bwd_dkv_addr = addr_dkv

            var casted_ptr_dkv = UnsafePointer[
                mut=True, type=CompiledTypeDKV, origin=MutUntrackedOrigin
            ](unsafe_from_address=addr_dkv)
            var retrieved_dkv = casted_ptr_dkv[]

            device_ctx.enqueue_function(
                retrieved_dkv,
                num_tiles_dkv,
                kv_tiles,
                query_tiles,
                d_key_ptr,
                d_value_ptr,
                query_ptr,
                key_ptr,
                value_ptr,
                output_ptr,
                d_output_ptr,
                log_sum_exp_ptr,
                seq_len,
                head_dim,
                grid_dim=(num_blocks_dkv,),
                block_dim=(BLOCK_SIZE_DKV,),
            )
        else:
            var compiled_dkv = device_ctx.compile_function[gpu_kernel_dkv]()
            device_ctx.enqueue_function(
                compiled_dkv,
                num_tiles_dkv,
                kv_tiles,
                query_tiles,
                d_key_ptr,
                d_value_ptr,
                query_ptr,
                key_ptr,
                value_ptr,
                output_ptr,
                d_output_ptr,
                log_sum_exp_ptr,
                seq_len,
                head_dim,
                grid_dim=(num_blocks_dkv,),
                block_dim=(BLOCK_SIZE_DKV,),
            )
    else:
        raise Error("Invalid target")


@compiler.register("attention_bwd")
struct AttentionBwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        use_soft_exp: Bool = True,
    ](
        d_q: MutableInputTensor[dtype=dtype, rank=4, static_spec=...],
        d_k: MutableInputTensor[dtype=dtype, rank=4, static_spec=...],
        d_v: MutableInputTensor[dtype=dtype, rank=4, static_spec=...],
        d_output: InputTensor[dtype=dtype, rank=4, static_spec=...],
        q: InputTensor[dtype=dtype, rank=4, static_spec=...],
        k: InputTensor[dtype=dtype, rank=4, static_spec=...],
        v: InputTensor[dtype=dtype, rank=4, static_spec=...],
        output: InputTensor[dtype=dtype, rank=4, static_spec=...],
        l_vec: InputTensor[dtype=DType.float32, rank=3, static_spec=...],
        batch_size: Int64,
        num_heads: Int64,
        seq_len: Int64,
        head_dim: Int64,
        ctx: DeviceContext,
    ) capturing raises:
        if d_q.size() != Int(batch_size * num_heads * seq_len * head_dim):
            raise Error("d_q size mismatch")
        if d_k.size() != Int(batch_size * num_heads * seq_len * head_dim):
            raise Error("d_k size mismatch")
        if d_v.size() != Int(batch_size * num_heads * seq_len * head_dim):
            raise Error("d_v size mismatch")
        if d_output.size() != Int(batch_size * num_heads * seq_len * head_dim):
            raise Error("d_output size mismatch")
        if q.size() != Int(batch_size * num_heads * seq_len * head_dim):
            raise Error("q size mismatch")
        if k.size() != Int(batch_size * num_heads * seq_len * head_dim):
            raise Error("k size mismatch")
        if v.size() != Int(batch_size * num_heads * seq_len * head_dim):
            raise Error("v size mismatch")
        if output.size() != Int(batch_size * num_heads * seq_len * head_dim):
            raise Error("output size mismatch")
        if l_vec.size() != Int(batch_size * num_heads * seq_len):
            raise Error("l_vec size mismatch")

        attention_bwd[dtype, target, use_soft_exp](
            d_q.unsafe_ptr(),
            d_k.unsafe_ptr(),
            d_v.unsafe_ptr(),
            d_output.unsafe_ptr(),
            q.unsafe_ptr(),
            k.unsafe_ptr(),
            v.unsafe_ptr(),
            output.unsafe_ptr(),
            l_vec.unsafe_ptr(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            ctx,
        )
