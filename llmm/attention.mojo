import compiler
from layout import Layout
from std.sys import simd_width_of
from std.memory import alloc, UnsafePointer
from extensibility import InputTensor
from std.gpu.host import DeviceContext
from std.gpu.host import DeviceAttribute
from std.gpu.memory import AddressSpace
from layout.layout_tensor import LayoutTensor
from std.gpu.host.info import is_cpu, is_gpu
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.runtime.asyncrt import parallelism_level
from std.algorithm import vectorize, sync_parallelize
from std.math import fma, sqrt, ceildiv, exp, log, ldexp, floor
from std.gpu import barrier, block_idx, grid_dim, thread_idx, WARP_SIZE


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime UNROLL = 4
comptime MAX_HEAD_DIM = 128
comptime TAU = Scalar[DType.float32](
    8.0
)  # FlashAttention 4 deferred-max threshold
comptime LN_2 = Scalar[DType.float32](0.69314718056)  # ln(2)
comptime C3 = Scalar[DType.float32](
    0.009618129
)  # FlashAttention 4 polynomial coefficient
comptime C2 = Scalar[DType.float32](
    0.055504108
)  # FlashAttention 4 polynomial coefficient
comptime C1 = Scalar[DType.float32](
    0.240179544
)  # FlashAttention 4 polynomial coefficient
comptime C0 = Scalar[DType.float32](
    1.0
)  # FlashAttention 4 polynomial coefficient


# ===----------------------------------------------------------------------=== #
# Utilities and Helpers
# ===----------------------------------------------------------------------=== #


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
        var integer_part = Int(floor(log2_input))
        var fractional_part = log2_input - Float32(integer_part)
        var polynomial = fma(
            fma(
                fma(C3, fractional_part, C2),
                fractional_part,
                C1,
            ),
            fractional_part,
            C0,
        )
        return ldexp(polynomial, Int32(integer_part))


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
    query_row: UnsafePointer[
        Scalar[dtype], ImmutAnyOrigin
    ],  # Pointer to the query row in the query tensor
    key_row: UnsafePointer[
        Scalar[dtype], ImmutAnyOrigin
    ],  # Pointer to the key row in the key tensor
    head_dim: Int,  # The dimension of the head
    attention_scale: Scalar[DType.float32],  # The attention scale
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
    output_row: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ],  # Pointer to the output row in the output tensor
    value_row: UnsafePointer[
        Scalar[dtype], ImmutAnyOrigin
    ],  # Pointer to the value row in the value tensor
    rescale_factor: Scalar[DType.float32],  # The rescale factor
    attention_weight: Scalar[DType.float32],  # The attention weight
    head_dim: Int,  # The dimension of the head
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
    output_row: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ],  # Pointer to the output row in the output tensor
    value_row: UnsafePointer[
        Scalar[dtype], ImmutAnyOrigin
    ],  # Pointer to the value row in the value tensor
    attention_weight: Scalar[DType.float32],  # The attention weight
    head_dim: Int,  # The dimension of the head
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
    output_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
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
    query_row: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key_row: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value_row: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    output_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
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
    output_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    value_row: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
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
    output_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    log_sum_exp_out: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
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
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    query_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    log_sum_exp_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
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


@always_inline
def _attention_shared_memory_row_pointer[
    dtype: DType,
](
    shared_tensor: LayoutTensor,
    row_index: Int,
) -> UnsafePointer[
    Scalar[dtype], ImmutAnyOrigin
]:
    return rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
        shared_tensor.ptr + row_index * MAX_HEAD_DIM
    )


@always_inline
def _attention_copy_rows_dram_to_shared[
    dtype: DType,
    BLOCK_SIZE: Int,
](
    shared_tensor: LayoutTensor,
    dram_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dram_row_start: Int,
    row_count: Int,
    head_dim: Int,
) -> None:
    # Block-strided copy: each thread covers a stride of elements so the CTA
    # collectively fills the shared tile. Uses synchronous stores because the
    # a copy_dram_to_sram_async thread_layout API is not available;
    # callers issue a barrier() after this returns to ensure visibility.
    var shared_ptr = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
        shared_tensor.ptr
    )
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
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
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
    softmax_max_deferred_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
    softmax_max_true_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    softmax_denominator_true_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
    output_rescale_factor_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
    local_query_row: Int,
    mut state: OnlineSoftmaxState[use_soft_exp, use_conditional_rescale],
) -> None:
    state.softmax_max_deferred = softmax_max_deferred_ptr[local_query_row]
    state.softmax_max_true = softmax_max_true_ptr[local_query_row]
    state.softmax_denominator_true = softmax_denominator_true_ptr[
        local_query_row
    ]
    state.output_rescale_factor = output_rescale_factor_ptr[local_query_row]


@always_inline
def _attention_store_online_softmax_state_to_shared[
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    softmax_max_deferred_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
    softmax_max_true_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    softmax_denominator_true_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
    output_rescale_factor_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
    local_query_row: Int,
    state: OnlineSoftmaxState[use_soft_exp, use_conditional_rescale],
) -> None:
    softmax_max_deferred_ptr[local_query_row] = state.softmax_max_deferred
    softmax_max_true_ptr[local_query_row] = state.softmax_max_true
    softmax_denominator_true_ptr[
        local_query_row
    ] = state.softmax_denominator_true
    output_rescale_factor_ptr[local_query_row] = state.output_rescale_factor


@always_inline
def _attention_init_online_softmax_state_shared[
    Br: Int,
    BLOCK_SIZE: Int,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    softmax_max_deferred_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
    softmax_max_true_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    softmax_denominator_true_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
    output_rescale_factor_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
    query_tile_start: Int,
    query_tile_rows: Int,
    seq_len: Int,
) -> None:
    var thread_id = Int(thread_idx.x)
    var row_index = thread_id
    var initial_state = OnlineSoftmaxState[
        use_soft_exp, use_conditional_rescale
    ]()
    while row_index < query_tile_rows:
        if query_tile_start + row_index < seq_len:
            softmax_max_deferred_ptr[
                row_index
            ] = initial_state.softmax_max_deferred
            softmax_max_true_ptr[row_index] = initial_state.softmax_max_true
            softmax_denominator_true_ptr[
                row_index
            ] = initial_state.softmax_denominator_true
            output_rescale_factor_ptr[
                row_index
            ] = initial_state.output_rescale_factor
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
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    query_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    log_sum_exp_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
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
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    query_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    log_sum_exp_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    batch_size: Int,
    num_heads: Int,
    seq_len: Int,
    head_dim: Int,
) -> None:
    var attention_scale = Scalar[DType.float32](1) / sqrt(
        Scalar[DType.float32](head_dim)
    )
    var total_heads = Int(batch_size * num_heads)
    var num_workers = min(total_heads, parallelism_level())
    var heads_per_worker = ceildiv(total_heads, num_workers)

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

    sync_parallelize[_worker](num_workers)


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
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    query_shared: LayoutTensor,
    key_shared: LayoutTensor,
    value_shared: LayoutTensor,
    softmax_max_deferred_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
    softmax_max_true_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    softmax_denominator_true_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
    output_rescale_factor_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
) -> None:
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE
    comptime ROWS_PER_WARP = Br // NUM_WARPS

    comptime for local_row in range(ROWS_PER_WARP):
        var query_index = (
            query_tile_start + warp_index * ROWS_PER_WARP + local_row
        )
        if query_index >= seq_len:
            continue

        var local_query_row = warp_index * ROWS_PER_WARP + local_row
        var output_row = output_ptr + head_offset + query_index * head_dim
        var state = OnlineSoftmaxState[use_soft_exp, use_conditional_rescale]()
        _attention_load_online_softmax_state_from_shared[
            use_soft_exp, use_conditional_rescale
        ](
            softmax_max_deferred_ptr,
            softmax_max_true_ptr,
            softmax_denominator_true_ptr,
            output_rescale_factor_ptr,
            local_query_row,
            state,
        )

        comptime for key_column in range(Bc):
            var key_index = key_tile_start + key_column
            if key_index >= seq_len:
                continue
            if key_index > query_index:
                continue

            _attention_update_query_row_for_key[
                dtype, width, use_soft_exp, use_conditional_rescale
            ](
                state,
                _attention_shared_memory_row_pointer[dtype](
                    query_shared, local_query_row
                ),
                _attention_shared_memory_row_pointer[dtype](
                    key_shared, key_column
                ),
                _attention_shared_memory_row_pointer[dtype](
                    value_shared, key_column
                ),
                output_row,
                head_dim,
                attention_scale,
            )

        _attention_store_online_softmax_state_to_shared[
            use_soft_exp, use_conditional_rescale
        ](
            softmax_max_deferred_ptr,
            softmax_max_true_ptr,
            softmax_denominator_true_ptr,
            output_rescale_factor_ptr,
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
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    log_sum_exp_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    softmax_max_deferred_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
    softmax_max_true_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    softmax_denominator_true_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
    output_rescale_factor_ptr: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin
    ],
) -> None:
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE
    comptime ROWS_PER_WARP = Br // NUM_WARPS

    comptime for local_row in range(ROWS_PER_WARP):
        var query_index = (
            query_tile_start + warp_index * ROWS_PER_WARP + local_row
        )
        if query_index >= seq_len:
            continue

        var local_query_row = warp_index * ROWS_PER_WARP + local_row
        var state = OnlineSoftmaxState[use_soft_exp, use_conditional_rescale]()
        _attention_load_online_softmax_state_from_shared[
            use_soft_exp, use_conditional_rescale
        ](
            softmax_max_deferred_ptr,
            softmax_max_true_ptr,
            softmax_denominator_true_ptr,
            output_rescale_factor_ptr,
            local_query_row,
            state,
        )
        var output_row = output_ptr + head_offset + query_index * head_dim
        var log_sum_exp_out = (
            log_sum_exp_ptr + head_index * seq_len + query_index
        )
        _attention_finalize_output_row[
            dtype, width, use_soft_exp, use_conditional_rescale
        ](state, output_row, log_sum_exp_out, head_dim)


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
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    query_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    log_sum_exp_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
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
    var softmax_max_deferred_ptr = rebind[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ](softmax_max_deferred_shared.ptr)
    var softmax_max_true_ptr = rebind[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ](softmax_max_true_shared.ptr)
    var softmax_denominator_true_ptr = rebind[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ](softmax_denominator_true_shared.ptr)
    var output_rescale_factor_ptr = rebind[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ](output_rescale_factor_shared.ptr)

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
        softmax_max_deferred_ptr,
        softmax_max_true_ptr,
        softmax_denominator_true_ptr,
        output_rescale_factor_ptr,
        query_tile_start,
        query_tile_rows,
        seq_len,
    )
    barrier()

    # Stream over KV tiles (FA2 loop_over_kvcache). Each iteration loads one K
    # and V tile synchronously then folds them into the online softmax state.
    # The barrier between copy and compute stands in for async_copy_wait_all()
    # that the reference uses to overlap the load of tile i+1 with compute on
    # tile i; here the copy and compute are serialized because a
    # copy_dram_to_sram_async thread_layout API is not yet available.
    for key_tile_start in range(0, seq_len, Bc):
        var key_tile_rows = min(Bc, seq_len - key_tile_start)
        _attention_copy_rows_dram_to_shared[dtype, BLOCK_SIZE](
            key_shared, key_head, key_tile_start, key_tile_rows, head_dim
        )
        _attention_copy_rows_dram_to_shared[dtype, BLOCK_SIZE](
            value_shared, value_head, key_tile_start, key_tile_rows, head_dim
        )
        barrier()

        if Int(thread_idx.x) % WARP_SIZE == 0:
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
                softmax_max_deferred_ptr,
                softmax_max_true_ptr,
                softmax_denominator_true_ptr,
                output_rescale_factor_ptr,
            )
        barrier()

    if Int(thread_idx.x) % WARP_SIZE == 0:
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
            softmax_max_deferred_ptr,
            softmax_max_true_ptr,
            softmax_denominator_true_ptr,
            output_rescale_factor_ptr,
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
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    query_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    log_sum_exp_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
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


def attention_fwd[
    dtype: DType,
    target: StaticString,
    use_soft_exp: Bool = True,
    use_conditional_rescale: Bool = True,
](
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    query_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    log_sum_exp_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    batch_size: Int64,
    num_heads: Int64,
    seq_len: Int64,
    head_dim: Int64,
    ctx: DeviceContext,
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
        comptime simd_width = simd_width_of[dtype]()
        comptime Br = 64
        comptime Bc = 64
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
    q_row: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k_row: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v_row: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    do_row: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dq_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dk_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dv_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
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

    # Accumulate gradients
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
    d_query_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_key_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_value_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    query_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    log_sum_exp_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    seq_len: Int,
    head_dim: Int,
    attention_scale: Scalar[DType.float32],
) -> None:
    var head_offset = head_index * seq_len * head_dim
    var lse_head = log_sum_exp_ptr + head_index * seq_len

    # Pointers for this head
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

    # Main backward pass.
    for i in range(seq_len):
        var q_row = q_head + i * head_dim
        var do_row = do_head + i * head_dim
        var dq_row = dq_head + i * head_dim
        var L_i = lse_head[i]
        var D_i = D[i]

        # Initialize dq_row to 0.
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
    d_query_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_key_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_value_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    query_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    log_sum_exp_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    batch_size: Int,
    num_heads: Int,
    seq_len: Int,
    head_dim: Int,
) -> None:
    var attention_scale = Scalar[DType.float32](1) / sqrt(
        Scalar[DType.float32](head_dim)
    )
    var total_heads = Int(batch_size * num_heads)
    var num_workers = min(total_heads, parallelism_level())
    var heads_per_worker = ceildiv(total_heads, num_workers)

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

    sync_parallelize[_worker](num_workers)


@always_inline
def _attention_copy_tile[
    dtype: DType,
    BLOCK_SIZE: Int,
    R: Int,
    C: Int,
](
    dest: LayoutTensor,
    src: LayoutTensor,
    row_count: Int,
    head_dim: Int,
) -> None:
    var dest_ptr = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](dest.ptr)
    var src_ptr = rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](src.ptr)
    var thread_id = Int(thread_idx.x)
    var element_count = row_count * head_dim
    var element_index = thread_id
    while element_index < element_count:
        var local_row = element_index // head_dim
        var column = element_index % head_dim
        dest_ptr[local_row * MAX_HEAD_DIM + column] = src_ptr[
            local_row * MAX_HEAD_DIM + column
        ]
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
    k_head: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v_head: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse_head: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    seq_len: Int,
    head_dim: Int,
) -> None:
    var query_tile_rows = min(Br, seq_len - query_tile_start)
    var attention_scale = Scalar[DType.float32](1) / sqrt(
        Scalar[DType.float32](head_dim)
    )

    # Shared memory (SRAM) via .stack_allocation().
    #
    # GPU shared memory is a small, fast scratchpad. It cannot be heap-allocated
    # at runtime from device code. The compiler must know each buffer's size before
    # launch so it can reserve space and reject kernels that exceed the hardware limit.
    #
    # .stack_allocation() is Mojo's equivalent of CUDA's `__shared__ T buf[N]`:
    # each call reserves one compile-time-sized slab in the block's shared
    # memory. We declare separate LayoutTensors (not one raw byte array) so
    # indexing stays typed and row-major strides are explicit.
    #
    # Br/Bc/MAX_HEAD_DIM are comptime constants so tile geometry is fixed at
    # compile time. Rows are padded to MAX_HEAD_DIM columns even when
    # head_dim < 128, letting one kernel binary serve any head_dim ≤ 128
    # without recompilation (same trick as the forward kernel).
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
    var lse_shared_ptr = rebind[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ](lse_shared.ptr)
    var D_shared_ptr = rebind[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ](D_shared.ptr)

    # Load Q, dO, O tiles into SRAM once and reuse across every KV inner tile.
    _attention_copy_tile[dtype, BLOCK_SIZE, Br, MAX_HEAD_DIM](
        q_shared, q_tile_dram, query_tile_rows, head_dim
    )
    _attention_copy_tile[dtype, BLOCK_SIZE, Br, MAX_HEAD_DIM](
        do_shared, do_tile_dram, query_tile_rows, head_dim
    )
    _attention_copy_tile[dtype, BLOCK_SIZE, Br, MAX_HEAD_DIM](
        o_shared, o_tile_dram, query_tile_rows, head_dim
    )

    var thread_id = Int(thread_idx.x)
    if thread_id < query_tile_rows:
        lse_shared_ptr[thread_id] = lse_head[query_tile_start + thread_id]
        for d in range(head_dim):
            dq_shared[thread_id, d] = Scalar[dtype](0.0)
    barrier()

    # Compute row dot products D_i
    if thread_id < query_tile_rows:
        D_shared_ptr[thread_id] = _attention_query_key_dot_product[
            dtype, width
        ](
            _attention_shared_memory_row_pointer[dtype](do_shared, thread_id),
            _attention_shared_memory_row_pointer[dtype](o_shared, thread_id),
            head_dim,
            Scalar[DType.float32](1.0),
        )
    barrier()

    var local_row = thread_id % Br
    var thread_row_index = thread_id // Br
    # Scratch for cross-thread reduction:
    # 4 threads per query row (Br rows × 4 lanes)
    # merge their private dQ partials before writing dq_shared.
    var reduction_shared = LayoutTensor[
        DType.float32,
        Layout.row_major(Br, 4),
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
        ]((k_head + key_tile_start * head_dim).as_unsafe_any_origin())
        var v_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Bc, MAX_HEAD_DIM), ImmutAnyOrigin
        ]((v_head + key_tile_start * head_dim).as_unsafe_any_origin())

        _attention_copy_tile[dtype, BLOCK_SIZE, Bc, MAX_HEAD_DIM](
            key_shared, k_tile_dram, key_tile_rows, head_dim
        )
        _attention_copy_tile[dtype, BLOCK_SIZE, Bc, MAX_HEAD_DIM](
            val_shared, v_tile_dram, key_tile_rows, head_dim
        )
        barrier()

        # Per-thread dQ partials live in local memory (registers/spill), not
        # shared SRAM, only this thread reads/writes them before the reduction.
        var private_dq = alloc[Scalar[DType.float32]](MAX_HEAD_DIM)
        for col in range(head_dim):
            private_dq[col] = 0.0

        if local_row < query_tile_rows:
            var global_i = query_tile_start + local_row
            var q_row = _attention_shared_memory_row_pointer[dtype](
                q_shared, local_row
            )
            var do_row = _attention_shared_memory_row_pointer[dtype](
                do_shared, local_row
            )
            var L_i = lse_shared_ptr[local_row]
            var D_i = D_shared_ptr[local_row]

            for j in range(thread_row_index, key_tile_rows, 4):
                var global_j = key_tile_start + j
                if global_j > global_i:
                    continue

                var k_row = _attention_shared_memory_row_pointer[dtype](
                    key_shared, j
                )
                var v_row = _attention_shared_memory_row_pointer[dtype](
                    val_shared, j
                )

                var S_ij = _attention_query_key_dot_product[dtype, width](
                    q_row, k_row, head_dim, attention_scale
                )
                var P_ij = software_emulated_exp[use_soft_exp](S_ij - L_i)
                var dP_ij = _attention_query_key_dot_product[dtype, width](
                    do_row, v_row, head_dim, Scalar[DType.float32](1.0)
                )
                var dS_ij = P_ij * (dP_ij - D_i)

                var factor = attention_scale * dS_ij
                var d = 0
                while d + width <= head_dim:
                    var k_vec = (
                        (k_row + d).load[width=width]().cast[DType.float32]()
                    )
                    var dq_vec = (private_dq + d).load[width=width]()
                    (private_dq + d).store[width=width](
                        fma(SIMD[DType.float32, width](factor), k_vec, dq_vec)
                    )
                    d += width
                for tail in range(d, head_dim):
                    private_dq[tail] += (
                        factor * k_row[tail].cast[DType.float32]()
                    )

        for col in range(head_dim):
            reduction_shared[local_row, thread_row_index] = private_dq[col]
            barrier()
            if thread_row_index == 0:
                var sum_col = (
                    reduction_shared[local_row, 0]
                    + reduction_shared[local_row, 1]
                    + reduction_shared[local_row, 2]
                    + reduction_shared[local_row, 3]
                )
                var existing = dq_shared[local_row, col].cast[DType.float32]()
                dq_shared[local_row, col] = (existing + sum_col).cast[dtype]()
            barrier()

        private_dq.free()

    # Write final dq_shared block out to DRAM.
    _attention_copy_tile[dtype, BLOCK_SIZE, Br, MAX_HEAD_DIM](
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
    d_query_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    query_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d_output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    log_sum_exp_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
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
        ](
            (
                query_ptr + head_offset + query_tile_start * Int(head_dim)
            ).as_unsafe_any_origin()
        )
        var do_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Br, MAX_HEAD_DIM), ImmutAnyOrigin
        ](
            (
                d_output_ptr + head_offset + query_tile_start * Int(head_dim)
            ).as_unsafe_any_origin()
        )
        var o_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Br, MAX_HEAD_DIM), ImmutAnyOrigin
        ](
            (
                output_ptr + head_offset + query_tile_start * Int(head_dim)
            ).as_unsafe_any_origin()
        )
        var dq_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Br, MAX_HEAD_DIM), MutAnyOrigin
        ](
            (
                d_query_ptr + head_offset + query_tile_start * Int(head_dim)
            ).as_unsafe_any_origin()
        )

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
    q_head: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    do_head: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o_head: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse_head: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    seq_len: Int,
    head_dim: Int,
    query_tiles: Int,
) -> None:
    var key_tile_rows = min(Bc, seq_len - key_tile_start)
    var attention_scale = Scalar[DType.float32](1) / sqrt(
        Scalar[DType.float32](head_dim)
    )

    # Shared memory layout mirrors pass 1 but with K/V/dK/dV as the outer
    # (Bc-row) tiles and Q/dO/O streamed on the inner loop. Same as FlashAttention-2 Algorithm 4.
    # Same .stack_allocation() is required as in the previous kernel.
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
    var lse_shared_ptr = rebind[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ](lse_shared.ptr)
    var D_shared_ptr = rebind[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ](D_shared.ptr)

    # Load K, V tiles into SRAM once; reused across every query inner tile.
    _attention_copy_tile[dtype, BLOCK_SIZE, Bc, MAX_HEAD_DIM](
        key_shared, k_tile_dram, key_tile_rows, head_dim
    )
    _attention_copy_tile[dtype, BLOCK_SIZE, Bc, MAX_HEAD_DIM](
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
        Layout.row_major(Bc, 4),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var reduction_dv = LayoutTensor[
        DType.float32,
        Layout.row_major(Bc, 4),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # FA2 inner loop (dK/dV path): stream query tiles I. Start at
    # key_tile_index because causal pairs require i >= j.
    # Only later query rows can attend to keys in this tile).
    for query_tile_idx in range(key_tile_index, query_tiles):
        var query_tile_start = query_tile_idx * Br
        var query_tile_rows = min(Br, seq_len - query_tile_start)

        var q_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Br, MAX_HEAD_DIM), ImmutAnyOrigin
        ]((q_head + query_tile_start * head_dim).as_unsafe_any_origin())
        var do_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Br, MAX_HEAD_DIM), ImmutAnyOrigin
        ]((do_head + query_tile_start * head_dim).as_unsafe_any_origin())
        var o_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Br, MAX_HEAD_DIM), ImmutAnyOrigin
        ]((o_head + query_tile_start * head_dim).as_unsafe_any_origin())

        _attention_copy_tile[dtype, BLOCK_SIZE, Br, MAX_HEAD_DIM](
            q_shared, q_tile_dram, query_tile_rows, head_dim
        )
        _attention_copy_tile[dtype, BLOCK_SIZE, Br, MAX_HEAD_DIM](
            do_shared, do_tile_dram, query_tile_rows, head_dim
        )
        _attention_copy_tile[dtype, BLOCK_SIZE, Br, MAX_HEAD_DIM](
            o_shared, o_tile_dram, query_tile_rows, head_dim
        )

        if thread_id < query_tile_rows:
            lse_shared_ptr[thread_id] = lse_head[query_tile_start + thread_id]
        barrier()

        # Compute row dot products D_i
        if thread_id < query_tile_rows:
            D_shared_ptr[thread_id] = _attention_query_key_dot_product[
                dtype, width
            ](
                _attention_shared_memory_row_pointer[dtype](
                    do_shared, thread_id
                ),
                _attention_shared_memory_row_pointer[dtype](
                    o_shared, thread_id
                ),
                head_dim,
                Scalar[DType.float32](1.0),
            )
        barrier()

        # Per-thread dK/dV partials in local memory, same rationale as dQ pass.
        var private_dk = alloc[Scalar[DType.float32]](MAX_HEAD_DIM)
        var private_dv = alloc[Scalar[DType.float32]](MAX_HEAD_DIM)
        for col in range(head_dim):
            private_dk[col] = 0.0
            private_dv[col] = 0.0

        if local_col < key_tile_rows:
            var global_j = key_tile_start + local_col
            var k_row = _attention_shared_memory_row_pointer[dtype](
                key_shared, local_col
            )
            var v_row = _attention_shared_memory_row_pointer[dtype](
                val_shared, local_col
            )

            for i in range(thread_col_index, query_tile_rows, 4):
                var global_i = query_tile_start + i
                if global_j > global_i:
                    continue

                var q_row = _attention_shared_memory_row_pointer[dtype](
                    q_shared, i
                )
                var do_row = _attention_shared_memory_row_pointer[dtype](
                    do_shared, i
                )
                var L_i = lse_shared_ptr[i]
                var D_i = D_shared_ptr[i]

                var S_ij = _attention_query_key_dot_product[dtype, width](
                    q_row, k_row, head_dim, attention_scale
                )
                var P_ij = software_emulated_exp[use_soft_exp](S_ij - L_i)
                var dP_ij = _attention_query_key_dot_product[dtype, width](
                    do_row, v_row, head_dim, Scalar[DType.float32](1.0)
                )
                var dS_ij = P_ij * (dP_ij - D_i)

                var dk_factor = attention_scale * dS_ij
                var d = 0
                while d + width <= head_dim:
                    var q_vec = (
                        (q_row + d).load[width=width]().cast[DType.float32]()
                    )
                    var dk_vec = (private_dk + d).load[width=width]()
                    (private_dk + d).store[width=width](
                        fma(
                            SIMD[DType.float32, width](dk_factor), q_vec, dk_vec
                        )
                    )
                    d += width
                for tail in range(d, head_dim):
                    private_dk[tail] += (
                        dk_factor * q_row[tail].cast[DType.float32]()
                    )

                var dv_factor = P_ij
                d = 0
                while d + width <= head_dim:
                    var do_vec = (
                        (do_row + d).load[width=width]().cast[DType.float32]()
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
                    private_dv[tail] += (
                        dv_factor * do_row[tail].cast[DType.float32]()
                    )

        for col in range(head_dim):
            reduction_dk[local_col, thread_col_index] = private_dk[col]
            reduction_dv[local_col, thread_col_index] = private_dv[col]
            barrier()
            if thread_col_index == 0:
                var sum_dk = (
                    reduction_dk[local_col, 0]
                    + reduction_dk[local_col, 1]
                    + reduction_dk[local_col, 2]
                    + reduction_dk[local_col, 3]
                )
                var sum_dv = (
                    reduction_dv[local_col, 0]
                    + reduction_dv[local_col, 1]
                    + reduction_dv[local_col, 2]
                    + reduction_dv[local_col, 3]
                )

                var existing_dk = dk_shared[local_col, col].cast[
                    DType.float32
                ]()
                var existing_dv = dv_shared[local_col, col].cast[
                    DType.float32
                ]()

                dk_shared[local_col, col] = (existing_dk + sum_dk).cast[dtype]()
                dv_shared[local_col, col] = (existing_dv + sum_dv).cast[dtype]()
            barrier()

        private_dk.free()
        private_dv.free()

    # Write final dk_shared and dv_shared blocks out to DRAM (fully contention-free)
    _attention_copy_tile[dtype, BLOCK_SIZE, Bc, MAX_HEAD_DIM](
        dk_tile_dram, dk_shared, key_tile_rows, head_dim
    )
    _attention_copy_tile[dtype, BLOCK_SIZE, Bc, MAX_HEAD_DIM](
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
    d_key_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_value_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    query_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d_output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    log_sum_exp_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
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
        ](
            (
                d_key_ptr + head_offset + key_tile_start * Int(head_dim)
            ).as_unsafe_any_origin()
        )
        var dv_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Bc, MAX_HEAD_DIM), MutAnyOrigin
        ](
            (
                d_value_ptr + head_offset + key_tile_start * Int(head_dim)
            ).as_unsafe_any_origin()
        )
        var k_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Bc, MAX_HEAD_DIM), ImmutAnyOrigin
        ](
            (
                key_ptr + head_offset + key_tile_start * Int(head_dim)
            ).as_unsafe_any_origin()
        )
        var v_tile_dram = LayoutTensor[
            dtype, Layout.row_major(Bc, MAX_HEAD_DIM), ImmutAnyOrigin
        ](
            (
                value_ptr + head_offset + key_tile_start * Int(head_dim)
            ).as_unsafe_any_origin()
        )

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
](
    d_query_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_key_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_value_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    query_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    log_sum_exp_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    batch_size: Int64,
    num_heads: Int64,
    seq_len: Int64,
    head_dim: Int64,
    ctx: DeviceContext,
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
        comptime simd_width = simd_width_of[dtype]()
        comptime Br = 64
        comptime Bc = 64
        comptime BLOCK_SIZE = 256
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
            BLOCK_SIZE,
            use_soft_exp=use_soft_exp,
        ]
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
            block_dim=(BLOCK_SIZE,),
        )

        # Pass 2: dK/dV - FA2 backward kernel with KV tile as outer loop.
        var num_tiles_dkv = Int(batch_size * num_heads * Int64(kv_tiles))
        var num_blocks_dkv = max(
            min(num_tiles_dkv, SM_OVERPROVISION * num_sm), 1
        )

        comptime gpu_kernel_dkv = attention_bwd_dkv_gpu[
            dtype,
            simd_width,
            Br,
            Bc,
            BLOCK_SIZE,
            use_soft_exp=use_soft_exp,
        ]
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
            block_dim=(BLOCK_SIZE,),
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
