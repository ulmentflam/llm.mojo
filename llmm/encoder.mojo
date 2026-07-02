import compiler
from layout import Layout
from std.math import ceildiv
from extensibility import InputTensor
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceAttribute
from std.sys import simd_width_of, align_of
from std.gpu.host.info import is_cpu, is_gpu
from layout.layout_tensor import LayoutTensor
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.runtime.asyncrt import parallelism_level
from std.algorithm import vectorize, sync_parallelize
from std.gpu import (
    barrier,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
    WARP_SIZE,
)

from llmm.profiler import traced_parallelize
from llmm.memory import ImmutKernelPtr, MutKernelPtr


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #


comptime WTE_BUCKET_IDX_SIZE = 4
comptime WTE_BWD_SIMD_WIDTH = 4
comptime WTE_C_PER_WARP = 32 * WTE_BWD_SIMD_WIDTH


# ===----------------------------------------------------------------------=== #
# Encoder Forward Helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def _encoder_fwd_vector_slice[
    dtype: DType,
    width: Int,
    aligned: Bool = False,
](
    out_row_ptr: MutKernelPtr[dtype],
    wte_row_ptr: ImmutKernelPtr[dtype],
    wpe_row_ptr: ImmutKernelPtr[dtype],
    c: Int,
) -> None:
    # `aligned` is opt-in: the GPU caller's c = c_base + tid*width (c_base a
    # multiple of block_dim.x*width) plus row bases that are multiples of
    # `channels` (itself always a multiple of width) is provably
    # width-aligned; the CPU caller isn't re-proven here, so it keeps the
    # unaligned default.
    comptime if aligned:
        comptime align = align_of[SIMD[dtype, width]]()
        var wte_val = (wte_row_ptr + c).load[width=width, alignment=align]()
        var wpe_val = (wpe_row_ptr + c).load[width=width, alignment=align]()
        (out_row_ptr + c).store[width=width, alignment=align](wte_val + wpe_val)
    else:
        var wte_val = (wte_row_ptr + c).load[width=width]()
        var wpe_val = (wpe_row_ptr + c).load[width=width]()
        (out_row_ptr + c).store(wte_val + wpe_val)


def encoder_fwd_cpu[
    dtype: DType,
    width: Int,
](
    out_ptr: MutKernelPtr[dtype],
    inp_ptr: ImmutKernelPtr[DType.int32],
    wte_ptr: ImmutKernelPtr[dtype],
    wpe_ptr: ImmutKernelPtr[dtype],
    batch_size: Int,
    seq_len: Int,
    channels: Int,
) raises -> None:
    var total_rows = batch_size * seq_len
    var max_workers = parallelism_level()
    var rows_per_worker = ceildiv(total_rows, max_workers)
    var num_workers = ceildiv(total_rows, rows_per_worker)

    @parameter
    def _worker(w: Int):
        var base_row = w * rows_per_worker
        var count_row = min(rows_per_worker, total_rows - base_row)

        for local_row in range(count_row):
            var bt = base_row + local_row
            var t = bt % seq_len
            var ix = Int((inp_ptr + bt).load())

            var out_row = out_ptr + bt * channels
            var wte_row = wte_ptr + ix * channels
            var wpe_row = wpe_ptr + t * channels

            @always_inline
            def _simd[simd_w: Int](c: Int) {out_row, wte_row, wpe_row}:
                _encoder_fwd_vector_slice[dtype, simd_w](
                    out_row, wte_row, wpe_row, c
                )

            vectorize[width, unroll_factor=4](channels, _simd)

    traced_parallelize["encoder_fwd", _worker](num_workers)


@always_inline
def encoder_fwd_gpu_kernel[
    dtype: DType,
    width: Int,
    aligned: Bool = True,
](
    out_ptr: MutKernelPtr[dtype],
    inp_ptr: ImmutKernelPtr[DType.int32],
    wte_ptr: ImmutKernelPtr[dtype],
    wpe_ptr: ImmutKernelPtr[dtype],
    seq_len: Int,
    channels: Int,
) -> None:
    # `aligned` (comptime): True is the production fast path — requires
    # channels % width == 0 (checked on the host), which makes
    # bt*channels + c / ix*channels + c / t*channels + c provably
    # width-aligned. False (odd channels, e.g. the equivalence suite's 33)
    # can't make that guarantee since bt/ix/t*channels isn't a multiple of
    # width in general, so it falls back to one scalar element per thread —
    # see _encoder_fwd_vector_slice's docstring for the same proof shape.
    var bt = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var t = bt % seq_len
    var ix = Int((inp_ptr + bt).load())

    comptime if aligned:
        var c_base = Int(block_idx.y) * Int(block_dim.x) * width
        var c = c_base + tid * width

        if c >= channels:
            return

        if c + width <= channels:
            _encoder_fwd_vector_slice[dtype, width, aligned=True](
                out_ptr + bt * channels,
                wte_ptr + ix * channels,
                wpe_ptr + t * channels,
                c,
            )
        else:
            for i in range(c, channels):
                _encoder_fwd_vector_slice[dtype, 1](
                    out_ptr + bt * channels,
                    wte_ptr + ix * channels,
                    wpe_ptr + t * channels,
                    i,
                )
    else:
        var c = Int(block_idx.y) * Int(block_dim.x) + tid
        if c >= channels:
            return
        _encoder_fwd_vector_slice[dtype, 1](
            out_ptr + bt * channels,
            wte_ptr + ix * channels,
            wpe_ptr + t * channels,
            c,
        )


def encoder_fwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
    aligned: Bool = True,
](
    out_ptr: MutKernelPtr[dtype],
    inp_ptr: ImmutKernelPtr[DType.int32],
    wte_ptr: ImmutKernelPtr[dtype],
    wpe_ptr: ImmutKernelPtr[dtype],
    seq_len: Int,
    channels: Int,
) -> None:
    encoder_fwd_gpu_kernel[dtype, width, aligned=aligned](
        out_ptr, inp_ptr, wte_ptr, wpe_ptr, seq_len, channels
    )


# ===----------------------------------------------------------------------=== #
# Encoder Forward Dispatcher
# ===----------------------------------------------------------------------=== #


def encoder_fwd[
    dtype: DType,
    target: StaticString,
](
    out_ptr: MutKernelPtr[dtype],
    inp_ptr: ImmutKernelPtr[DType.int32],
    wte_ptr: ImmutKernelPtr[dtype],
    wpe_ptr: ImmutKernelPtr[dtype],
    batch_size: Int,
    seq_len: Int,
    channels: Int,
    ctx: DeviceContext,
) capturing raises:
    comptime if is_cpu[target]():
        comptime width = simd_width_of[dtype]()
        encoder_fwd_cpu[dtype, width](
            out_ptr,
            inp_ptr,
            wte_ptr,
            wpe_ptr,
            batch_size,
            seq_len,
            channels,
        )
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 256
        comptime width = 4
        var device_ctx = ctx
        var grid_x = batch_size * seq_len

        # Dispatch aligned vs. scalar-fallback kernels at the host; see
        # encoder_fwd_gpu_kernel's docstring.
        if channels % width == 0:
            var grid_y = ceildiv(channels, BLOCK_SIZE * width)
            comptime gpu_kernel = encoder_fwd_gpu[
                dtype, BLOCK_SIZE, width, aligned=True
            ]
            var compiled = device_ctx.compile_function[gpu_kernel]()
            device_ctx.enqueue_function(
                compiled,
                out_ptr,
                inp_ptr,
                wte_ptr,
                wpe_ptr,
                seq_len,
                channels,
                grid_dim=(grid_x, grid_y),
                block_dim=(BLOCK_SIZE,),
            )
        else:
            var grid_y_u = ceildiv(channels, BLOCK_SIZE)
            comptime gpu_kernel_u = encoder_fwd_gpu[
                dtype, BLOCK_SIZE, width, aligned=False
            ]
            var compiled_u = device_ctx.compile_function[gpu_kernel_u]()
            device_ctx.enqueue_function(
                compiled_u,
                out_ptr,
                inp_ptr,
                wte_ptr,
                wpe_ptr,
                seq_len,
                channels,
                grid_dim=(grid_x, grid_y_u),
                block_dim=(BLOCK_SIZE,),
            )
    else:
        raise Error("Invalid target")


@compiler.register("encoder_fwd")
struct EncoderFwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: MutableInputTensor[dtype=dtype, rank=3, static_spec=...],
        inp: InputTensor[dtype=DType.int32, rank=2, static_spec=...],
        wte: InputTensor[dtype=dtype, rank=2, static_spec=...],
        wpe: InputTensor[dtype=dtype, rank=2, static_spec=...],
        batch_size: Int64,
        seq_len: Int64,
        channels: Int64,
        ctx: DeviceContext,
    ) capturing raises:
        if output.size() != Int(batch_size * seq_len * channels):
            raise Error("output size mismatch")
        if inp.size() != Int(batch_size * seq_len):
            raise Error("inp size mismatch")
        if wpe.size() != Int(seq_len * channels):
            raise Error("wpe size mismatch")

        encoder_fwd[dtype, target](
            output.unsafe_ptr(),
            inp.unsafe_ptr(),
            wte.unsafe_ptr(),
            wpe.unsafe_ptr(),
            Int(batch_size),
            Int(seq_len),
            Int(channels),
            ctx,
        )


# ===----------------------------------------------------------------------=== #
# Encoder Backward Bucket Builder
# ===----------------------------------------------------------------------=== #


def build_wte_buckets(
    inputs_ptr: ImmutKernelPtr[DType.int32],
    bucket_info_ptr: MutKernelPtr[DType.int32],
    workload_indices_ptr: MutKernelPtr[DType.int32],
    batch_size: Int,
    seq_len: Int,
    vocab_size: Int,
    channels: Int,
    bucket_info_capacity: Int,
) raises -> Int:
    """
    Build scatter-add buckets for wte backward from the current token batch.
    """
    var total_positions = batch_size * seq_len
    var num_channel_groups = ceildiv(channels, WTE_C_PER_WARP)

    var counts = List[Int]()
    for _ in range(vocab_size):
        counts.append(0)

    for bt in range(total_positions):
        var token = Int(inputs_ptr.load(bt))
        if token < 0 or token >= vocab_size:
            raise Error("encoder bucket build: token index out of range")
        counts[token] = counts[token] + 1

    var offsets = List[Int]()
    var cursor = 0
    for token in range(vocab_size):
        offsets.append(cursor)
        cursor += counts[token]

    var cursors = List[Int]()
    for token in range(vocab_size):
        cursors.append(offsets[token])

    for bt in range(total_positions):
        var token = Int(inputs_ptr.load(bt))
        var idx = cursors[token]
        workload_indices_ptr.store(idx, Scalar[DType.int32](bt))
        cursors[token] = idx + 1

    var num_buckets = 0
    for token in range(vocab_size):
        var size = counts[token]
        if size == 0:
            continue
        var start_idx = offsets[token]
        for g in range(num_channel_groups):
            if num_buckets >= bucket_info_capacity:
                raise Error(
                    "encoder bucket build: bucket_info capacity exceeded"
                )
            var base = num_buckets * WTE_BUCKET_IDX_SIZE
            bucket_info_ptr.store(base + 0, Scalar[DType.int32](start_idx))
            bucket_info_ptr.store(base + 1, Scalar[DType.int32](size))
            bucket_info_ptr.store(base + 2, Scalar[DType.int32](token))
            bucket_info_ptr.store(base + 3, Scalar[DType.int32](g))
            num_buckets += 1

    return num_buckets


# ===----------------------------------------------------------------------=== #
# Encoder Backward Helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def _accumulate_token_gradients[
    dtype: DType,
    width: Int,
    aligned: Bool = False,
](
    dout_ptr: ImmutKernelPtr[dtype],
    workload_indices_ptr: ImmutKernelPtr[DType.int32],
    start_idx: Int,
    bucket_size: Int,
    channels: Int,
    c: Int,
    start_item: Int,
    step_size: Int,
) -> SIMD[DType.float32, width]:
    # `aligned` opt-in: the GPU caller's c = channel_group*(WARP_SIZE*width) +
    # lane_id*width is provably width-aligned (channels is also always a
    # multiple of width), so bt*channels + c is too. The CPU caller isn't
    # re-proven here, so it keeps the unaligned default.
    var accum = SIMD[DType.float32, width](0.0)
    var item = start_item
    comptime if aligned:
        comptime align = align_of[SIMD[dtype, width]]()
        while item < bucket_size:
            var bt = Int((workload_indices_ptr + start_idx + item).load())
            var dout_val = (
                (dout_ptr + bt * channels + c)
                .load[width=width, alignment=align]()
                .cast[DType.float32]()
            )
            accum += dout_val
            item += step_size
    else:
        while item < bucket_size:
            var bt = Int((workload_indices_ptr + start_idx + item).load())
            var dout_val = (
                (dout_ptr + bt * channels + c)
                .load[width=width]()
                .cast[DType.float32]()
            )
            accum += dout_val
            item += step_size
    return accum


@always_inline
def _write_dwte_accumulation[
    dtype: DType,
    width: Int,
    aligned: Bool = False,
](
    dwte_ptr: MutKernelPtr[dtype],
    dwte_offset: Int,
    accum: SIMD[DType.float32, width],
) -> None:
    # `aligned` opt-in: the GPU caller's dwte_offset = token_idx*channels + c
    # is provably width-aligned (channels and c are both multiples of
    # width — see _accumulate_token_gradients); the CPU caller keeps the
    # unaligned default.
    comptime if aligned:
        comptime align = align_of[SIMD[dtype, width]]()
        var prev_dwte = (
            (dwte_ptr + dwte_offset)
            .load[width=width, alignment=align]()
            .cast[DType.float32]()
        )
        (dwte_ptr + dwte_offset).store[width=width, alignment=align](
            (prev_dwte + accum).cast[dtype]()
        )
    else:
        var prev_dwte = (
            (dwte_ptr + dwte_offset).load[width=width]().cast[DType.float32]()
        )
        (dwte_ptr + dwte_offset).store((prev_dwte + accum).cast[dtype]())


# ===----------------------------------------------------------------------=== #
# Encoder Backward
# ===----------------------------------------------------------------------=== #


def wte_backward_cpu[
    dtype: DType,
    width: Int,
    BUCKET_IDX_SIZE: Int = 4,
](
    dwte_ptr: MutKernelPtr[dtype],
    bucket_info_ptr: ImmutKernelPtr[DType.int32],
    workload_indices_ptr: ImmutKernelPtr[DType.int32],
    dout_ptr: ImmutKernelPtr[dtype],
    num_buckets: Int,
    channels: Int,
) raises -> None:
    var max_workers = parallelism_level()
    var buckets_per_worker = ceildiv(num_buckets, max_workers)
    var num_workers = ceildiv(num_buckets, buckets_per_worker)

    @parameter
    def _worker(w: Int):
        var base = w * buckets_per_worker
        var count = min(buckets_per_worker, num_buckets - base)

        for local in range(count):
            var bucket_idx = base + local

            var info = (bucket_info_ptr + bucket_idx * BUCKET_IDX_SIZE).load[
                width=BUCKET_IDX_SIZE
            ]()
            var start_idx = Int(info[0])
            var size = Int(info[1])
            var token_idx = Int(info[2])
            var channel_group = Int(info[3])

            var c_per_warp = WARP_SIZE * width
            var c_base = channel_group * c_per_warp

            var c_end = min(c_base + c_per_warp, channels)
            var c_len = c_end - c_base

            if c_len > 0:

                @always_inline
                def _simd[
                    simd_w: Int
                ](c_offset: Int) {
                    dout_ptr,
                    workload_indices_ptr,
                    dwte_ptr,
                    start_idx,
                    size,
                    token_idx,
                    channels,
                    c_base,
                }:
                    var c = c_base + c_offset

                    var accum = _accumulate_token_gradients[dtype, simd_w](
                        dout_ptr,
                        workload_indices_ptr,
                        start_idx,
                        size,
                        channels,
                        c,
                        start_item=0,
                        step_size=1,
                    )

                    _write_dwte_accumulation[dtype, simd_w](
                        dwte_ptr,
                        token_idx * channels + c,
                        accum,
                    )

                vectorize[width, unroll_factor=4](c_len, _simd)

    traced_parallelize["wte_backward", _worker](num_workers)


def wpe_backward_cpu[
    dtype: DType,
    width: Int,
](
    dwpe_ptr: MutKernelPtr[dtype],
    dout_ptr: ImmutKernelPtr[dtype],
    batch_size: Int,
    seq_len: Int,
    channels: Int,
) raises -> None:
    var max_workers = parallelism_level()
    var t_per_worker = ceildiv(seq_len, max_workers)
    var num_workers = ceildiv(seq_len, t_per_worker)

    @parameter
    def _worker(w: Int):
        var base_t = w * t_per_worker
        var count_t = min(t_per_worker, seq_len - base_t)

        for local_t in range(count_t):
            var t = base_t + local_t
            var dwpe_row = dwpe_ptr + t * channels

            @always_inline
            def _simd[
                simd_w: Int
            ](c: Int) {dwpe_row, dout_ptr, batch_size, seq_len, channels, t}:
                var accum = SIMD[DType.float32, simd_w](0.0)
                for b in range(batch_size):
                    var dout_val = (
                        (dout_ptr + (b * seq_len + t) * channels + c)
                        .load[width=simd_w]()
                        .cast[DType.float32]()
                    )
                    accum += dout_val

                var prev_dwpe = (
                    (dwpe_row + c).load[width=simd_w]().cast[DType.float32]()
                )
                (dwpe_row + c).store((prev_dwpe + accum).cast[dtype]())

            vectorize[width, unroll_factor=4](channels, _simd)

    traced_parallelize["wpe_backward", _worker](num_workers)


@always_inline
def wte_backward_gpu_kernel[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
    aligned: Bool = True,
](
    dwte_ptr: MutKernelPtr[dtype],
    bucket_info_ptr: ImmutKernelPtr[DType.int32],
    workload_indices_ptr: ImmutKernelPtr[DType.int32],
    dout_ptr: ImmutKernelPtr[dtype],
    channels: Int,
) -> None:
    # `aligned` (comptime): True is the production fast path — requires
    # channels % width == 0 (checked on the host), which makes
    # bt*channels + c / token_idx*channels + c provably width-aligned. False
    # (odd channels, e.g. the equivalence suite's 33) forces the scalar,
    # per-element accumulate/write path for every channel group, not just
    # the true tail — see _accumulate_token_gradients's docstring for the
    # same proof shape.
    var bucket = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane_id = tid % WARP_SIZE
    var c_per_warp = WARP_SIZE * width

    var info = (bucket_info_ptr + bucket * 4).load[width=4]()
    var start_idx = Int(info[0])
    var bucket_size = Int(info[1])
    var token_idx = Int(info[2])
    var channel_group = Int(info[3])

    var c = channel_group * c_per_warp + lane_id * width
    if c >= channels:
        return

    var num_warps = BLOCK_SIZE // WARP_SIZE
    if warp_id >= bucket_size:
        return

    comptime SMEM_SIZE = BLOCK_SIZE * width
    var accum_shared = LayoutTensor[
        DType.float32,
        Layout.row_major(SMEM_SIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var accum_shared_ptr = rebind[MutKernelPtr[DType.float32]](
        accum_shared.ptr.address_space_cast[AddressSpace.GENERIC]()
    )

    comptime if aligned:
        if c + width <= channels:
            var accum = _accumulate_token_gradients[dtype, width, aligned=True](
                dout_ptr,
                workload_indices_ptr,
                start_idx,
                bucket_size,
                channels,
                c,
                start_item=warp_id,
                step_size=num_warps,
            )
            for k in range(width):
                accum_shared_ptr[tid * width + k] = accum[k]
        else:
            var accum = SIMD[DType.float32, width](0.0)
            var item = warp_id
            while item < bucket_size:
                var bt = Int((workload_indices_ptr + start_idx + item).load())
                for k in range(channels - c):
                    var val = dout_ptr[bt * channels + c + k].cast[
                        DType.float32
                    ]()
                    accum[k] += val
                item += num_warps
            for k in range(width):
                accum_shared_ptr[tid * width + k] = accum[k]
    else:
        var accum = SIMD[DType.float32, width](0.0)
        var lanes = min(width, channels - c)
        var item = warp_id
        while item < bucket_size:
            var bt = Int((workload_indices_ptr + start_idx + item).load())
            for k in range(lanes):
                var val = dout_ptr[bt * channels + c + k].cast[DType.float32]()
                accum[k] += val
            item += num_warps
        for k in range(width):
            accum_shared_ptr[tid * width + k] = accum[k]

    barrier()

    if warp_id == 0:
        var final_accum = SIMD[DType.float32, width](0.0)
        for w in range(num_warps):
            if w < bucket_size:
                var partner_tid = w * WARP_SIZE + lane_id
                var val = SIMD[DType.float32, width](0.0)
                for k in range(width):
                    val[k] = accum_shared_ptr[partner_tid * width + k]
                final_accum += val

        comptime if aligned:
            if c + width <= channels:
                _write_dwte_accumulation[dtype, width, aligned=True](
                    dwte_ptr,
                    token_idx * channels + c,
                    final_accum,
                )
            else:
                for k in range(channels - c):
                    var prev_dwte = dwte_ptr[token_idx * channels + c + k].cast[
                        DType.float32
                    ]()
                    dwte_ptr[token_idx * channels + c + k] = (
                        prev_dwte + final_accum[k]
                    ).cast[dtype]()
        else:
            for k in range(min(width, channels - c)):
                var prev_dwte = dwte_ptr[token_idx * channels + c + k].cast[
                    DType.float32
                ]()
                dwte_ptr[token_idx * channels + c + k] = (
                    prev_dwte + final_accum[k]
                ).cast[dtype]()


@always_inline
def wpe_backward_gpu_kernel[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
    aligned: Bool = True,
](
    dwpe_ptr: MutKernelPtr[dtype],
    dout_ptr: ImmutKernelPtr[dtype],
    batch_size: Int,
    seq_len: Int,
    channels: Int,
) -> None:
    # GPU-only kernel (no shared CPU helper): c = tile_base + tid*width is
    # provably width-aligned (tile_base a multiple of c_per_block=BLOCK_SIZE*
    # width). `aligned` (comptime): True additionally requires
    # channels % width == 0 (checked on the host), which makes both
    # (b*seq_len+t)*channels + c and t*channels + c provably aligned too.
    # False (odd channels, e.g. the equivalence suite's 33) falls back to a
    # fully scalar sweep — channels isn't a multiple of width there, so
    # t*channels + c isn't provably aligned even though c itself is.
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var c_per_block = BLOCK_SIZE * width
    comptime align = align_of[SIMD[dtype, width]]()

    comptime if aligned:
        for tile_base in range(0, channels, c_per_block):
            var c = tile_base + tid * width
            if c >= channels:
                break

            if c + width <= channels:
                var accum = SIMD[DType.float32, width](0.0)
                for b in range(batch_size):
                    var dout_val = (
                        (dout_ptr + (b * seq_len + t) * channels + c)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    accum += dout_val

                var offset = t * channels + c
                var prev_dwpe = (
                    (dwpe_ptr + offset)
                    .load[width=width, alignment=align]()
                    .cast[DType.float32]()
                )
                (dwpe_ptr + offset).store[width=width, alignment=align](
                    (prev_dwpe + accum).cast[dtype]()
                )
            else:
                for col in range(c, channels):
                    var accum = Scalar[DType.float32](0.0)
                    for b in range(batch_size):
                        var dout_val = dout_ptr[
                            (b * seq_len + t) * channels + col
                        ].cast[DType.float32]()
                        accum += dout_val

                    var offset = t * channels + col
                    var prev_dwpe = dwpe_ptr[offset].cast[DType.float32]()
                    dwpe_ptr[offset] = (prev_dwpe + accum).cast[dtype]()
    else:
        for tile_base in range(0, channels, BLOCK_SIZE):
            var col = tile_base + tid
            if col >= channels:
                break
            var accum = Scalar[DType.float32](0.0)
            for b in range(batch_size):
                var dout_val = dout_ptr[
                    (b * seq_len + t) * channels + col
                ].cast[DType.float32]()
                accum += dout_val

            var offset = t * channels + col
            var prev_dwpe = dwpe_ptr[offset].cast[DType.float32]()
            dwpe_ptr[offset] = (prev_dwpe + accum).cast[dtype]()


def encoder_bwd[
    dtype: DType,
    target: StaticString,
](
    dwte_ptr: MutKernelPtr[dtype],
    dwpe_ptr: MutKernelPtr[dtype],
    bucket_info_ptr: ImmutKernelPtr[DType.int32],
    workload_indices_ptr: ImmutKernelPtr[DType.int32],
    dout_ptr: ImmutKernelPtr[dtype],
    num_buckets: Int,
    batch_size: Int,
    seq_len: Int,
    channels: Int,
    ctx: DeviceContext,
) capturing raises:
    comptime if is_cpu[target]():
        comptime width = simd_width_of[dtype]()
        wte_backward_cpu[dtype, width](
            dwte_ptr,
            bucket_info_ptr,
            workload_indices_ptr,
            dout_ptr,
            num_buckets,
            channels,
        )
        wpe_backward_cpu[dtype, width](
            dwpe_ptr,
            dout_ptr,
            batch_size,
            seq_len,
            channels,
        )
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 256
        comptime width = 4
        var device_ctx = ctx
        # Dispatch aligned vs. scalar-fallback kernels at the host; see
        # wte_backward_gpu_kernel/wpe_backward_gpu_kernel's docstrings.
        var aligned = channels % width == 0

        if num_buckets > 0:
            if aligned:
                comptime wte_kernel = wte_backward_gpu_kernel[
                    dtype, BLOCK_SIZE, width, aligned=True
                ]
                var wte_compiled = device_ctx.compile_function[wte_kernel]()
                device_ctx.enqueue_function(
                    wte_compiled,
                    dwte_ptr,
                    bucket_info_ptr,
                    workload_indices_ptr,
                    dout_ptr,
                    channels,
                    grid_dim=(num_buckets,),
                    block_dim=(BLOCK_SIZE,),
                )
            else:
                comptime wte_kernel_u = wte_backward_gpu_kernel[
                    dtype, BLOCK_SIZE, width, aligned=False
                ]
                var wte_compiled_u = device_ctx.compile_function[wte_kernel_u]()
                device_ctx.enqueue_function(
                    wte_compiled_u,
                    dwte_ptr,
                    bucket_info_ptr,
                    workload_indices_ptr,
                    dout_ptr,
                    channels,
                    grid_dim=(num_buckets,),
                    block_dim=(BLOCK_SIZE,),
                )

        if seq_len > 0:
            if aligned:
                comptime wpe_kernel = wpe_backward_gpu_kernel[
                    dtype, BLOCK_SIZE, width, aligned=True
                ]
                var wpe_compiled = device_ctx.compile_function[wpe_kernel]()
                device_ctx.enqueue_function(
                    wpe_compiled,
                    dwpe_ptr,
                    dout_ptr,
                    batch_size,
                    seq_len,
                    channels,
                    grid_dim=(seq_len,),
                    block_dim=(BLOCK_SIZE,),
                )
            else:
                comptime wpe_kernel_u = wpe_backward_gpu_kernel[
                    dtype, BLOCK_SIZE, width, aligned=False
                ]
                var wpe_compiled_u = device_ctx.compile_function[wpe_kernel_u]()
                device_ctx.enqueue_function(
                    wpe_compiled_u,
                    dwpe_ptr,
                    dout_ptr,
                    batch_size,
                    seq_len,
                    channels,
                    grid_dim=(seq_len,),
                    block_dim=(BLOCK_SIZE,),
                )
    else:
        raise Error("Invalid target")


@compiler.register("encoder_bwd")
struct EncoderBwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        dwte: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        dwpe: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        bucket_info: InputTensor[dtype=DType.int32, rank=2, static_spec=...],
        workload_indices: InputTensor[
            dtype=DType.int32, rank=1, static_spec=...
        ],
        dout: InputTensor[dtype=dtype, rank=3, static_spec=...],
        num_buckets: Int64,
        batch_size: Int64,
        seq_len: Int64,
        channels: Int64,
        ctx: DeviceContext,
    ) capturing raises:
        if dwte.size() != Int(dwte.shape()[0]) * Int(channels):
            raise Error("dwte size mismatch")
        if dwpe.size() != Int(seq_len * channels):
            raise Error("dwpe size mismatch")
        if bucket_info.size() != Int(num_buckets * 4):
            raise Error("bucket_info size mismatch")
        if dout.size() != Int(batch_size * seq_len * channels):
            raise Error("dout size mismatch")

        encoder_bwd[dtype, target](
            dwte.unsafe_ptr(),
            dwpe.unsafe_ptr(),
            bucket_info.unsafe_ptr(),
            workload_indices.unsafe_ptr(),
            dout.unsafe_ptr(),
            Int(num_buckets),
            Int(batch_size),
            Int(seq_len),
            Int(channels),
            ctx,
        )
