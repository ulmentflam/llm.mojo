from std.collections import InlineArray
from std.memory import UnsafePointer, memcpy, alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.host.info import is_cpu
from std.sys.info import size_of
from std.sys import has_accelerator, has_nvidia_gpu_accelerator
from std.algorithm import sync_parallelize
from layout.tile_layout import row_major

from llmm.memory import MutMemPtr

from llmm.zero import (
    ZeroContext,
    ShardedParameter,
    CpuCoordinator,
)


from std.testing import TestSuite, assert_almost_equal


# ===----------------------------------------------------------------------=== #
# Single-rank tests (world_size=1)
# ===----------------------------------------------------------------------=== #


def test_single_cpu_allreduce() raises:
    var ctx = DeviceContext(api="cpu")
    var z_ctx = ZeroContext[target="cpu"](
        rank=0,
        zero_stage=0,
        ctx=ctx,
    )

    comptime DTYPE = DType.float32
    comptime size = 64

    var host_in = ctx.enqueue_create_host_buffer[DTYPE](size)
    var host_out = ctx.enqueue_create_host_buffer[DTYPE](size)

    var host_in_ptr = host_in.unsafe_ptr()
    var host_out_ptr = host_out.unsafe_ptr()
    for i in range(size):
        host_in_ptr[i] = 4.2
        host_out_ptr[i] = 0.0

    var buf = ctx.enqueue_create_buffer[DTYPE](size)
    buf.enqueue_copy_from(host_in)
    ctx.synchronize()

    z_ctx.allreduce[DTYPE](
        rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
            buf.unsafe_ptr().as_unsafe_any_origin()
        ),
        size,
    )
    ctx.synchronize()

    buf.enqueue_copy_to(host_out)
    ctx.synchronize()

    for i in range(size):
        assert_almost_equal[DTYPE](host_out_ptr[i], 4.2, atol=1e-6)


def test_single_cpu_reducescatter() raises:
    var ctx = DeviceContext(api="cpu")
    var z_ctx = ZeroContext[target="cpu"](
        rank=0,
        zero_stage=0,
        ctx=ctx,
    )

    comptime DTYPE = DType.float32
    comptime size = 64

    var host_in = ctx.enqueue_create_host_buffer[DTYPE](size)
    var host_out = ctx.enqueue_create_host_buffer[DTYPE](size)

    var host_in_ptr = host_in.unsafe_ptr()
    var host_out_ptr = host_out.unsafe_ptr()
    for i in range(size):
        host_in_ptr[i] = 1.5
        host_out_ptr[i] = 0.0

    var in_buf = ctx.enqueue_create_buffer[DTYPE](size)
    var out_buf = ctx.enqueue_create_buffer[DTYPE](size)

    in_buf.enqueue_copy_from(host_in)
    out_buf.enqueue_copy_from(host_out)
    ctx.synchronize()

    z_ctx.reducescatter[DTYPE](
        rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
            in_buf.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
            out_buf.unsafe_ptr().as_unsafe_any_origin()
        ),
        size,
    )
    ctx.synchronize()

    out_buf.enqueue_copy_to(host_out)
    ctx.synchronize()

    for i in range(size):
        assert_almost_equal[DTYPE](host_out_ptr[i], 1.5, atol=1e-6)


def test_single_cpu_allgather() raises:
    var ctx = DeviceContext(api="cpu")
    var z_ctx = ZeroContext[target="cpu"](
        rank=0,
        zero_stage=0,
        ctx=ctx,
    )

    comptime DTYPE = DType.float32
    comptime size = 32

    var host_in = ctx.enqueue_create_host_buffer[DTYPE](size)
    var host_out = ctx.enqueue_create_host_buffer[DTYPE](size)

    var host_in_ptr = host_in.unsafe_ptr()
    var host_out_ptr = host_out.unsafe_ptr()
    for i in range(size):
        host_in_ptr[i] = 8.8
        host_out_ptr[i] = 0.0

    var buf = ctx.enqueue_create_buffer[DTYPE](size)
    buf.enqueue_copy_from(host_in)
    ctx.synchronize()

    z_ctx.allgather[DTYPE](
        rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
            buf.unsafe_ptr().as_unsafe_any_origin()
        ),
        size,
    )
    ctx.synchronize()

    buf.enqueue_copy_to(host_out)
    ctx.synchronize()

    for i in range(size):
        assert_almost_equal[DTYPE](host_out_ptr[i], 8.8, atol=1e-6)


def test_sharded_parameter_gather_cpu() raises:
    var ctx = DeviceContext(api="cpu")
    var z_ctx = ZeroContext[target="cpu"](
        rank=0,
        zero_stage=0,
        ctx=ctx,
    )

    comptime DTYPE = DType.float32
    comptime size = 64

    var host_in = ctx.enqueue_create_host_buffer[DTYPE](size)
    var host_out = ctx.enqueue_create_host_buffer[DTYPE](size)
    var host_in_ptr = host_in.unsafe_ptr()
    var host_out_ptr = host_out.unsafe_ptr()
    for i in range(size):
        host_in_ptr[i] = 7.5
        host_out_ptr[i] = 0.0

    var param = ShardedParameter[DTYPE, 1, "cpu"](size, ctx)
    param.sharded_buffer.enqueue_copy_from(host_in)
    ctx.synchronize()

    var all_sharded = InlineArray[
        UnsafePointer[Scalar[DTYPE], MutUntrackedOrigin], 1
    ](uninitialized=True)
    all_sharded[0] = param.sharded_buffer.unsafe_ptr().unsafe_origin_cast[
        MutUntrackedOrigin
    ]()

    comptime in_layout = row_major(1, size)
    comptime out_layout = row_major(1, size)

    var full = param.gather[type_of(in_layout), type_of(out_layout)](
        z_ctx, all_sharded, in_layout, out_layout
    )

    full.enqueue_copy_to(host_out)
    ctx.synchronize()

    for i in range(size):
        assert_almost_equal[DTYPE](host_out_ptr[i], 7.5, atol=1e-6)


def test_sharded_parameter_gather_gpu() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var z_ctx = ZeroContext[target="gpu"](
        rank=0,
        zero_stage=0,
        ctx=ctx,
    )

    comptime DTYPE = DType.float32
    comptime size = 64

    var host_in = ctx.enqueue_create_host_buffer[DTYPE](size)
    var host_out = ctx.enqueue_create_host_buffer[DTYPE](size)
    var host_in_ptr = host_in.unsafe_ptr()
    var host_out_ptr = host_out.unsafe_ptr()
    for i in range(size):
        host_in_ptr[i] = 9.9
        host_out_ptr[i] = 0.0

    var param = ShardedParameter[DTYPE, 1, "gpu"](size, ctx)
    param.sharded_buffer.enqueue_copy_from(host_in)
    ctx.synchronize()

    var all_sharded = InlineArray[
        UnsafePointer[Scalar[DTYPE], MutUntrackedOrigin], 1
    ](uninitialized=True)
    all_sharded[0] = param.sharded_buffer.unsafe_ptr().unsafe_origin_cast[
        MutUntrackedOrigin
    ]()

    comptime in_layout = row_major(1, size)
    comptime out_layout = row_major(1, size)

    var full = param.gather[type_of(in_layout), type_of(out_layout)](
        z_ctx, all_sharded, in_layout, out_layout
    )

    full.enqueue_copy_to(host_out)
    ctx.synchronize()

    for i in range(size):
        assert_almost_equal[DTYPE](host_out_ptr[i], 9.9, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# Multi-rank CPU tests using sync_parallelize
# ===----------------------------------------------------------------------=== #


def test_multi_cpu_allreduce() raises:
    var ctx = DeviceContext(api="cpu")
    comptime WORLD_SIZE = 4
    comptime DTYPE = DType.float32
    comptime size = 64

    var cpu_coord_ptr = alloc[CpuCoordinator](1)
    cpu_coord_ptr[] = CpuCoordinator(WORLD_SIZE)

    var rank_inputs = alloc[Scalar[DTYPE]](WORLD_SIZE * size)
    var rank_outputs = alloc[Scalar[DTYPE]](WORLD_SIZE * size)

    for r in range(WORLD_SIZE):
        for i in range(size):
            rank_inputs[r * size + i] = Float32(r + 1)
            rank_outputs[r * size + i] = 0.0

    @parameter
    def _run_rank(rank: Int):
        try:
            var z_ctx = ZeroContext["cpu", WORLD_SIZE](
                rank=rank,
                zero_stage=0,
                ctx=ctx,
                cpu_coord=cpu_coord_ptr,
            )

            var input_ptr = rank_inputs + rank * size
            var output_ptr = rank_outputs + rank * size

            var buf = ctx.enqueue_create_buffer[DTYPE](size)
            var host_in = ctx.enqueue_create_host_buffer[DTYPE](size)
            for j in range(size):
                host_in.unsafe_ptr()[j] = input_ptr[j]
            buf.enqueue_copy_from(host_in)
            ctx.synchronize()

            z_ctx.allreduce[DTYPE](
                rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                    buf.unsafe_ptr().as_unsafe_any_origin()
                ),
                size,
            )
            ctx.synchronize()

            var host_out = ctx.enqueue_create_host_buffer[DTYPE](size)
            buf.enqueue_copy_to(host_out)
            ctx.synchronize()

            for j in range(size):
                output_ptr[j] = host_out.unsafe_ptr()[j]

        except e:
            print("allreduce rank error:", e)

    sync_parallelize[_run_rank](WORLD_SIZE)

    for r in range(WORLD_SIZE):
        for i in range(size):
            assert_almost_equal[DTYPE](
                rank_outputs[r * size + i], 10.0, atol=1e-6
            )

    rank_inputs.free()
    rank_outputs.free()
    cpu_coord_ptr[].free()
    cpu_coord_ptr.free()


def test_multi_cpu_reducescatter() raises:
    var ctx = DeviceContext(api="cpu")
    comptime WORLD_SIZE = 4
    comptime DTYPE = DType.float32
    comptime size = 64
    comptime sharded_size = size // WORLD_SIZE

    var cpu_coord_ptr = alloc[CpuCoordinator](1)
    cpu_coord_ptr[] = CpuCoordinator(WORLD_SIZE)

    var rank_inputs = alloc[Scalar[DTYPE]](WORLD_SIZE * size)
    var rank_outputs = alloc[Scalar[DTYPE]](WORLD_SIZE * sharded_size)

    for r in range(WORLD_SIZE):
        for i in range(size):
            rank_inputs[r * size + i] = Float32(r + 1)
        for i in range(sharded_size):
            rank_outputs[r * sharded_size + i] = 0.0

    @parameter
    def _run_rank(rank: Int):
        try:
            var z_ctx = ZeroContext["cpu", WORLD_SIZE](
                rank=rank,
                zero_stage=0,
                ctx=ctx,
                cpu_coord=cpu_coord_ptr,
            )

            var input_ptr = rank_inputs + rank * size
            var output_ptr = rank_outputs + rank * sharded_size

            var in_buf = ctx.enqueue_create_buffer[DTYPE](size)
            var out_buf = ctx.enqueue_create_buffer[DTYPE](sharded_size)

            var host_in = ctx.enqueue_create_host_buffer[DTYPE](size)
            for j in range(size):
                host_in.unsafe_ptr()[j] = input_ptr[j]
            in_buf.enqueue_copy_from(host_in)
            ctx.synchronize()

            z_ctx.reducescatter[DTYPE](
                rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                    in_buf.unsafe_ptr().as_unsafe_any_origin()
                ),
                rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                    out_buf.unsafe_ptr().as_unsafe_any_origin()
                ),
                sharded_size,
            )
            ctx.synchronize()

            var host_out = ctx.enqueue_create_host_buffer[DTYPE](sharded_size)
            out_buf.enqueue_copy_to(host_out)
            ctx.synchronize()

            for j in range(sharded_size):
                output_ptr[j] = host_out.unsafe_ptr()[j]

        except e:
            print("reducescatter rank error:", e)

    sync_parallelize[_run_rank](WORLD_SIZE)

    for r in range(WORLD_SIZE):
        for i in range(sharded_size):
            assert_almost_equal[DTYPE](
                rank_outputs[r * sharded_size + i], 10.0, atol=1e-6
            )

    rank_inputs.free()
    rank_outputs.free()
    cpu_coord_ptr[].free()
    cpu_coord_ptr.free()


def test_multi_cpu_reducescatter_inplace() raises:
    """In-place reduce-scatter (the ZeRO-2/3 gradient path): rank r's OWN slice
    [r*shard, (r+1)*shard) of its full buffer is overwritten with the cross-rank
    sum; every other slice keeps this rank's local value. This is what lets the
    optimizer read its reduced gradient shard from grads_memory + rank*shard
    without a separate shard buffer.
    """
    var ctx = DeviceContext(api="cpu")
    comptime WORLD_SIZE = 4
    comptime DTYPE = DType.float32
    comptime size = 64
    comptime sharded_size = size // WORLD_SIZE

    var cpu_coord_ptr = alloc[CpuCoordinator](1)
    cpu_coord_ptr[] = CpuCoordinator(WORLD_SIZE)

    # Each rank's full buffer holds the constant (r+1) everywhere.
    var rank_bufs = alloc[Scalar[DTYPE]](WORLD_SIZE * size)
    for r in range(WORLD_SIZE):
        for i in range(size):
            rank_bufs[r * size + i] = Float32(r + 1)

    @parameter
    def _run_rank(rank: Int):
        try:
            var z_ctx = ZeroContext["cpu", WORLD_SIZE](
                rank=rank,
                zero_stage=2,
                ctx=ctx,
                cpu_coord=cpu_coord_ptr,
            )

            var host_ptr = rank_bufs + rank * size
            var buf = ctx.enqueue_create_buffer[DTYPE](size)
            var host_in = ctx.enqueue_create_host_buffer[DTYPE](size)
            for j in range(size):
                host_in.unsafe_ptr()[j] = host_ptr[j]
            buf.enqueue_copy_from(host_in)
            ctx.synchronize()

            z_ctx.reducescatter_inplace[DTYPE](
                rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                    buf.unsafe_ptr().as_unsafe_any_origin()
                ),
                size,
            )
            ctx.synchronize()

            var host_out = ctx.enqueue_create_host_buffer[DTYPE](size)
            buf.enqueue_copy_to(host_out)
            ctx.synchronize()
            for j in range(size):
                host_ptr[j] = host_out.unsafe_ptr()[j]

        except e:
            print("reducescatter_inplace rank error:", e)

    sync_parallelize[_run_rank](WORLD_SIZE)

    # Sum over ranks of (r+1) == 1+2+3+4 == 10.
    for r in range(WORLD_SIZE):
        for k in range(WORLD_SIZE):
            var expected = Float32(10.0) if k == r else Float32(r + 1)
            for i in range(sharded_size):
                assert_almost_equal[DTYPE](
                    rank_bufs[r * size + k * sharded_size + i],
                    expected,
                    atol=1e-6,
                )

    rank_bufs.free()
    cpu_coord_ptr[].free()
    cpu_coord_ptr.free()


def test_multi_cpu_reducescatter_buckets() raises:
    """Bucketed reduce-scatter (the ZeRO-2/3 backward-bucketing path): buckets
    living contiguously in a per-rank pool but destined for SCATTERED global
    flat ranges are cross-rank summed and land, per element, in the owning
    rank's shard accumulator. Mirrors the tensor-major GPT-2 layout, where a
    layer's gradient tensors sit at far-apart flat offsets but are produced
    together into one pool.
    """
    var ctx = DeviceContext(api="cpu")
    comptime WORLD_SIZE = 4
    comptime DTYPE = DType.float32
    comptime opt = 16
    comptime padded = opt * WORLD_SIZE  # 64

    var cpu_coord_ptr = alloc[CpuCoordinator](1)
    cpu_coord_ptr[] = CpuCoordinator(WORLD_SIZE)

    # Two buckets: bucket 0 -> flat [0,16) (rank 0's shard), bucket 1 ->
    # flat [40,56) (spans ranks 2 and 3). The pool holds them back to back.
    # A third bucket -> flat [16,20) (start of rank 1's shard).
    comptime B0_LEN = 16
    comptime B1_LEN = 16
    comptime B2_LEN = 4
    comptime POOL = B0_LEN + B1_LEN + B2_LEN
    var dest = List[Int]()
    dest.append(0)
    dest.append(40)
    dest.append(16)
    var poff = List[Int]()
    poff.append(0)
    poff.append(B0_LEN)
    poff.append(B0_LEN + B1_LEN)
    var lens = List[Int]()
    lens.append(B0_LEN)
    lens.append(B1_LEN)
    lens.append(B2_LEN)

    var rank_shards = alloc[Scalar[DTYPE]](WORLD_SIZE * opt)
    for i in range(WORLD_SIZE * opt):
        rank_shards[i] = 0.0

    @parameter
    def _run_rank(rank: Int):
        try:
            var z_ctx = ZeroContext["cpu", WORLD_SIZE](
                rank=rank, zero_stage=2, ctx=ctx, cpu_coord=cpu_coord_ptr
            )
            var pool = ctx.enqueue_create_buffer[DTYPE](POOL)
            var host_pool = ctx.enqueue_create_host_buffer[DTYPE](POOL)
            # pool element value == (rank+1) * (global_flat_index + 1)
            for b in range(len(dest)):
                for j in range(lens[b]):
                    host_pool.unsafe_ptr()[poff[b] + j] = Float32(
                        (rank + 1) * (dest[b] + j + 1)
                    )
            pool.enqueue_copy_from(host_pool)
            var shard = ctx.enqueue_create_buffer[DTYPE](opt)
            shard.enqueue_fill(Float32(0.0))
            ctx.synchronize()

            z_ctx.reducescatter_buckets[DTYPE](
                rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                    pool.unsafe_ptr().as_unsafe_any_origin()
                ),
                dest,
                poff,
                lens,
                rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                    shard.unsafe_ptr().as_unsafe_any_origin()
                ),
                opt,
            )
            ctx.synchronize()
            var host_shard = ctx.enqueue_create_host_buffer[DTYPE](opt)
            shard.enqueue_copy_to(host_shard)
            ctx.synchronize()
            for j in range(opt):
                rank_shards[rank * opt + j] = host_shard.unsafe_ptr()[j]
        except e:
            print("reducescatter_buckets rank error:", e)

    sync_parallelize[_run_rank](WORLD_SIZE)

    # Expected: for each global flat index f covered by a bucket, the owning
    # rank's shard[f - r*opt] == sum_k (k+1)*(f+1) == 10*(f+1). Uncovered
    # indices stay 0. Covered flats: [0,16), [16,20), [40,56).
    @parameter
    def _covered(f: Int) -> Bool:
        return (f < 20) or (f >= 40 and f < 56)

    for r in range(WORLD_SIZE):
        for j in range(opt):
            var f = r * opt + j
            var expected = Float32(10 * (f + 1)) if _covered(f) else Float32(
                0.0
            )
            assert_almost_equal[DTYPE](
                rank_shards[r * opt + j], expected, atol=1e-4
            )

    rank_shards.free()
    cpu_coord_ptr[].free()
    cpu_coord_ptr.free()


def test_multi_cpu_allgather() raises:
    var ctx = DeviceContext(api="cpu")
    comptime WORLD_SIZE = 4
    comptime DTYPE = DType.float32
    comptime sharded_size = 16
    comptime size = sharded_size * WORLD_SIZE

    var cpu_coord_ptr = alloc[CpuCoordinator](1)
    cpu_coord_ptr[] = CpuCoordinator(WORLD_SIZE)

    var rank_inputs = alloc[Scalar[DTYPE]](WORLD_SIZE * sharded_size)
    var rank_outputs = alloc[Scalar[DTYPE]](WORLD_SIZE * size)

    for r in range(WORLD_SIZE):
        for i in range(sharded_size):
            rank_inputs[r * sharded_size + i] = Float32(r + 1)
        for i in range(size):
            rank_outputs[r * size + i] = 0.0

    @parameter
    def _run_rank(rank: Int):
        try:
            var z_ctx = ZeroContext["cpu", WORLD_SIZE](
                rank=rank,
                zero_stage=0,
                ctx=ctx,
                cpu_coord=cpu_coord_ptr,
            )

            var input_ptr = rank_inputs + rank * sharded_size
            var output_ptr = rank_outputs + rank * size

            var buf = ctx.enqueue_create_buffer[DTYPE](size)
            var host_in = ctx.enqueue_create_host_buffer[DTYPE](size)
            for j in range(size):
                host_in.unsafe_ptr()[j] = 0.0
            var offset = rank * sharded_size
            for j in range(sharded_size):
                host_in.unsafe_ptr()[offset + j] = input_ptr[j]

            buf.enqueue_copy_from(host_in)
            ctx.synchronize()

            z_ctx.allgather[DTYPE](
                rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                    buf.unsafe_ptr().as_unsafe_any_origin()
                ),
                sharded_size,
            )
            ctx.synchronize()

            var host_out = ctx.enqueue_create_host_buffer[DTYPE](size)
            buf.enqueue_copy_to(host_out)
            ctx.synchronize()

            for j in range(size):
                output_ptr[j] = host_out.unsafe_ptr()[j]

        except e:
            print("allgather rank error:", e)

    sync_parallelize[_run_rank](WORLD_SIZE)

    for r in range(WORLD_SIZE):
        for k in range(WORLD_SIZE):
            var offset = k * sharded_size
            for i in range(sharded_size):
                if (
                    abs(
                        Float64(rank_outputs[r * size + offset + i])
                        - Float64(k + 1)
                    )
                    > 1e-6
                ):
                    print(
                        "ALLGATHER MISMATCH at rank",
                        r,
                        "k",
                        k,
                        "index",
                        offset + i,
                        "got",
                        rank_outputs[r * size + offset + i],
                        "expected",
                        Float32(k + 1),
                    )
                assert_almost_equal[DTYPE](
                    rank_outputs[r * size + offset + i],
                    Float32(k + 1),
                    atol=1e-6,
                )

    rank_inputs.free()
    rank_outputs.free()
    cpu_coord_ptr[].free()
    cpu_coord_ptr.free()


def test_multi_sharded_parameter_gather_cpu() raises:
    var ctx = DeviceContext(api="cpu")
    comptime WORLD_SIZE = 4
    comptime DTYPE = DType.float32
    comptime size = 64
    comptime sharded_size = size // WORLD_SIZE

    var cpu_coord_ptr = alloc[CpuCoordinator](1)
    cpu_coord_ptr[] = CpuCoordinator(WORLD_SIZE)

    var rank_inputs = alloc[Scalar[DTYPE]](WORLD_SIZE * sharded_size)
    var rank_outputs = alloc[Scalar[DTYPE]](WORLD_SIZE * size)

    for r in range(WORLD_SIZE):
        for i in range(sharded_size):
            rank_inputs[r * sharded_size + i] = Float32(r + 1)
        for i in range(size):
            rank_outputs[r * size + i] = 0.0

    var shared_ptrs = alloc[UnsafePointer[Scalar[DTYPE], MutUntrackedOrigin]](
        WORLD_SIZE
    )

    @parameter
    def _run_rank(rank: Int):
        try:
            var z_ctx = ZeroContext["cpu", WORLD_SIZE](
                rank=rank,
                zero_stage=0,
                ctx=ctx,
                cpu_coord=cpu_coord_ptr,
            )

            var input_ptr = rank_inputs + rank * sharded_size
            var output_ptr = rank_outputs + rank * size

            var host_in = ctx.enqueue_create_host_buffer[DTYPE](sharded_size)
            for j in range(sharded_size):
                host_in.unsafe_ptr()[j] = input_ptr[j]

            var param = ShardedParameter[DTYPE, WORLD_SIZE, "cpu"](size, ctx)
            param.sharded_buffer.enqueue_copy_from(host_in)
            ctx.synchronize()

            shared_ptrs[
                rank
            ] = param.sharded_buffer.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ]()
            z_ctx.cpu_coordinator_ptr.value()[].barrier2[].wait()

            var all_sharded = InlineArray[
                UnsafePointer[Scalar[DTYPE], MutUntrackedOrigin], WORLD_SIZE
            ](uninitialized=True)
            for k in range(WORLD_SIZE):
                all_sharded[k] = shared_ptrs[k]

            comptime in_layout = row_major(1, sharded_size)
            comptime out_layout = row_major(1, size)

            var full = param.gather[type_of(in_layout), type_of(out_layout)](
                z_ctx, all_sharded, in_layout, out_layout
            )

            var host_out = ctx.enqueue_create_host_buffer[DTYPE](size)
            full.enqueue_copy_to(host_out)
            ctx.synchronize()

            for j in range(size):
                output_ptr[j] = host_out.unsafe_ptr()[j]

        except e:
            print("sharded parameter gather rank error:", e)

    sync_parallelize[_run_rank](WORLD_SIZE)

    for r in range(WORLD_SIZE):
        for k in range(WORLD_SIZE):
            var offset = k * sharded_size
            for i in range(sharded_size):
                assert_almost_equal[DTYPE](
                    rank_outputs[r * size + offset + i],
                    Float32(k + 1),
                    atol=1e-6,
                )

    shared_ptrs.free()
    rank_inputs.free()
    rank_outputs.free()
    cpu_coord_ptr[].free()
    cpu_coord_ptr.free()


# ===----------------------------------------------------------------------=== #
# Multi-rank GPU tests (staged-copy collectives; 2 NVIDIA GPUs required)
# ===----------------------------------------------------------------------=== #


def _gpu_multirank_available() raises -> Bool:
    if not has_nvidia_gpu_accelerator():
        return False
    if DeviceContext.number_of_devices() < 2:
        return False
    # A driver-faulted GPU ("GPU requires reset") can shrink the usable
    # ordinal range below number_of_devices(); probe both devices for real.
    try:
        var c0 = DeviceContext(device_id=0)
        var c1 = DeviceContext(device_id=1)
        var b0 = c0.enqueue_create_buffer[DType.float32](4)
        var b1 = c1.enqueue_create_buffer[DType.float32](4)
        b0.enqueue_fill(Float32(0.0))
        b1.enqueue_fill(Float32(0.0))
        c0.synchronize()
        c1.synchronize()
        return True
    except:
        return False


def test_multi_gpu_collectives() raises:
    """Drive all three staged-copy GPU collectives end to end (N=2).

    Exercises per-rank DeviceContext(device_id=rank), coordinator pointer
    exchange, cross-device staged copies, and the fp32-accumulate add
    kernel — allreduce, reducescatter, and allgather sequentially inside
    ONE rank-thread session. A single combined test (not three) on
    purpose: every extra GPU test pays a full CUDA context-init per rank,
    which on a degraded driver (e.g. a co-resident faulted GPU) runs into
    minutes and pushed this file past make test-mojo's 600 s per-file
    timeout when the ops were separate tests.
    """
    if not _gpu_multirank_available():
        return

    comptime WORLD_SIZE = 2
    comptime DTYPE = DType.float32
    comptime shard = 64
    comptime size = shard * WORLD_SIZE

    var cpu_coord_ptr = alloc[CpuCoordinator](1)
    cpu_coord_ptr[] = CpuCoordinator(WORLD_SIZE)

    # Per-op snapshots, indexed [rank*len + j]:
    #   ar_out    — full buffer after allreduce
    #   rs_shard  — shard output after reducescatter
    #   ag_out    — full buffer after allgather
    var ar_out = alloc[Scalar[DTYPE]](WORLD_SIZE * size)
    var rs_shard = alloc[Scalar[DTYPE]](WORLD_SIZE * shard)
    var rs_ip_out = alloc[Scalar[DTYPE]](WORLD_SIZE * size)
    var rsb_shard = alloc[Scalar[DTYPE]](WORLD_SIZE * shard)
    var ag_out = alloc[Scalar[DTYPE]](WORLD_SIZE * size)

    # reducescatter_buckets legs: two buckets in a contiguous pool mapping to
    # SCATTERED global flat ranges — bucket A -> flat [0,32) (rank 0's shard),
    # bucket B -> flat [48,80) (spans rank 0's tail and rank 1's head). This is
    # the ZeRO-2/3 backward-bucketing path; it exercises the staged-copy GPU
    # reduce with a partial bucket/shard overlap. Pool value == (r+1)*(f+1).
    comptime BA_LEN = 32
    comptime BB_LEN = 32
    comptime BPOOL = BA_LEN + BB_LEN
    var bdest = List[Int]()
    bdest.append(0)
    bdest.append(48)
    var bpoff = List[Int]()
    bpoff.append(0)
    bpoff.append(BA_LEN)
    var blens = List[Int]()
    blens.append(BA_LEN)
    blens.append(BB_LEN)

    @parameter
    def _run_rank(rank: Int):
        try:
            var ctx = DeviceContext(device_id=rank)
            var z_ctx = ZeroContext["gpu", WORLD_SIZE](
                rank=rank,
                zero_stage=1,
                ctx=ctx,
                cpu_coord=cpu_coord_ptr,
            )
            z_ctx.ensure_comm_setup(shard * size_of[Scalar[DTYPE]]())

            var host = ctx.enqueue_create_host_buffer[DTYPE](size)
            var host_s = ctx.enqueue_create_host_buffer[DTYPE](shard)
            var buf = ctx.enqueue_create_buffer[DTYPE](size)
            var shard_buf = ctx.enqueue_create_buffer[DTYPE](shard)
            var buf_ptr = rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                buf.unsafe_ptr().as_unsafe_any_origin()
            )
            var shard_ptr = rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                shard_buf.unsafe_ptr().as_unsafe_any_origin()
            )

            # ---- allreduce: rank r's element j = (r+1)*(j+1) ----
            for j in range(size):
                host[j] = Float32((rank + 1) * (j + 1))
            buf.enqueue_copy_from(host)
            ctx.synchronize()
            z_ctx.allreduce[DTYPE](buf_ptr, size)
            buf.enqueue_copy_to(host)
            ctx.synchronize()
            for j in range(size):
                ar_out[rank * size + j] = host[j]

            # ---- reducescatter: same inputs, shard output ----
            for j in range(size):
                host[j] = Float32((rank + 1) * (j + 1))
            buf.enqueue_copy_from(host)
            shard_buf.enqueue_fill(Float32(0.0))
            ctx.synchronize()
            z_ctx.reducescatter[DTYPE](buf_ptr, shard_ptr, shard)
            shard_buf.enqueue_copy_to(host_s)
            ctx.synchronize()
            for j in range(shard):
                rs_shard[rank * shard + j] = host_s[j]

            # ---- reducescatter_inplace: same inputs, reduced in place ----
            for j in range(size):
                host[j] = Float32((rank + 1) * (j + 1))
            buf.enqueue_copy_from(host)
            ctx.synchronize()
            z_ctx.reducescatter_inplace[DTYPE](buf_ptr, size)
            buf.enqueue_copy_to(host)
            ctx.synchronize()
            for j in range(size):
                rs_ip_out[rank * size + j] = host[j]

            # ---- reducescatter_buckets: scattered buckets in a pool -> shard --
            var bpool = ctx.enqueue_create_buffer[DTYPE](BPOOL)
            var bpool_host = ctx.enqueue_create_host_buffer[DTYPE](BPOOL)
            for b in range(len(bdest)):
                for j in range(blens[b]):
                    bpool_host[bpoff[b] + j] = Float32(
                        (rank + 1) * (bdest[b] + j + 1)
                    )
            bpool.enqueue_copy_from(bpool_host)
            var bshard = ctx.enqueue_create_buffer[DTYPE](shard)
            bshard.enqueue_fill(Float32(0.0))
            ctx.synchronize()
            z_ctx.reducescatter_buckets[DTYPE](
                rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                    bpool.unsafe_ptr().as_unsafe_any_origin()
                ),
                bdest,
                bpoff,
                blens,
                rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                    bshard.unsafe_ptr().as_unsafe_any_origin()
                ),
                shard,
            )
            bshard.enqueue_copy_to(host_s)
            ctx.synchronize()
            for j in range(shard):
                rsb_shard[rank * shard + j] = host_s[j]

            # ---- allgather: only my slice is mine; the other is zeroed so
            # stale data can't fake a pass ----
            for j in range(size):
                host[j] = Float32(0.0)
            var off = rank * shard
            for j in range(shard):
                host[off + j] = Float32((rank + 1) * 1000 + j)
            buf.enqueue_copy_from(host)
            ctx.synchronize()
            z_ctx.allgather[DTYPE](buf_ptr, shard)
            buf.enqueue_copy_to(host)
            ctx.synchronize()
            for j in range(size):
                ag_out[rank * size + j] = host[j]
        except e:
            print("multi-gpu rank", rank, "error:", e)

    sync_parallelize[_run_rank](WORLD_SIZE)

    # allreduce: every rank's full buffer[j] == 3*(j+1)  (sum of r+1).
    for r in range(WORLD_SIZE):
        for j in range(size):
            assert_almost_equal[DTYPE](
                ar_out[r * size + j], Float32(3 * (j + 1)), atol=1e-5
            )
    # reducescatter: rank r's shard[j] == 3*(r*shard + j + 1).
    for r in range(WORLD_SIZE):
        for j in range(shard):
            assert_almost_equal[DTYPE](
                rs_shard[r * shard + j],
                Float32(3 * (r * shard + j + 1)),
                atol=1e-5,
            )
    # reducescatter_inplace: rank r's OWN slice r holds the reduced sum
    # 3*(global_index+1); every other slice k keeps rank r's local input
    # (r+1)*(global_index+1).
    for r in range(WORLD_SIZE):
        for k in range(WORLD_SIZE):
            for j in range(shard):
                var gidx = k * shard + j
                var expected = Float32(3 * (gidx + 1)) if k == r else Float32(
                    (r + 1) * (gidx + 1)
                )
                assert_almost_equal[DTYPE](
                    rs_ip_out[r * size + gidx], expected, atol=1e-5
                )
    # allgather: every rank's slice k == rank k's staged shard values.
    for r in range(WORLD_SIZE):
        for k in range(WORLD_SIZE):
            for j in range(shard):
                assert_almost_equal[DTYPE](
                    ag_out[r * size + k * shard + j],
                    Float32((k + 1) * 1000 + j),
                    atol=1e-5,
                )
    # reducescatter_buckets: for each global flat f covered by a bucket, the
    # owning rank's shard[f - r*shard] == sum_k (k+1)*(f+1) == 3*(f+1);
    # uncovered indices stay 0. Covered flats: [0,32) and [48,80).
    for r in range(WORLD_SIZE):
        for j in range(shard):
            var f = r * shard + j
            var covered = (f < 32) or (f >= 48 and f < 80)
            var expected = Float32(3 * (f + 1)) if covered else Float32(0.0)
            assert_almost_equal[DTYPE](
                rsb_shard[r * shard + j], expected, atol=1e-5
            )

    ar_out.free()
    rs_shard.free()
    rs_ip_out.free()
    rsb_shard.free()
    ag_out.free()
    cpu_coord_ptr[].free()
    cpu_coord_ptr.free()


def test_multi_cpu_allgather_ranges() raises:
    """Gather arbitrary flat sub-ranges (the ZeRO-3 per-layer parameter-
    streaming primitive `allgather_ranges`) of a sharded vector from every rank's
    persistent shard into a local window. The conceptual full vector holds
    value == index; rank r owns [r*shard, (r+1)*shard). Two requested ranges
    deliberately straddle shard boundaries so the cross-rank staged pulls are
    exercised. Every rank must reconstruct the same window == the full vector's
    values at those indices.
    """
    var ctx = DeviceContext(api="cpu")
    comptime WORLD_SIZE = 4
    comptime DTYPE = DType.float32
    comptime shard = 16
    # Two ranges: [8,24) spans shards 0-1; [30,50) spans shards 1-3.
    comptime lenA = 16
    comptime lenB = 20
    comptime win = lenA + lenB

    var cpu_coord_ptr = alloc[CpuCoordinator](1)
    cpu_coord_ptr[] = CpuCoordinator(WORLD_SIZE)

    var rank_windows = alloc[Scalar[DTYPE]](WORLD_SIZE * win)
    for i in range(WORLD_SIZE * win):
        rank_windows[i] = 0.0

    @parameter
    def _run_rank(rank: Int):
        try:
            var z_ctx = ZeroContext["cpu", WORLD_SIZE](
                rank=rank,
                zero_stage=3,
                ctx=ctx,
                cpu_coord=cpu_coord_ptr,
            )

            # This rank's persistent shard: value == global index.
            var shard_buf = ctx.enqueue_create_buffer[DTYPE](shard)
            var host_shard = ctx.enqueue_create_host_buffer[DTYPE](shard)
            for j in range(shard):
                host_shard.unsafe_ptr()[j] = Float32(rank * shard + j)
            shard_buf.enqueue_copy_from(host_shard)
            ctx.synchronize()

            var win_buf = ctx.enqueue_create_buffer[DTYPE](win)
            var dst_offsets = List[Int]()
            dst_offsets.append(0)
            dst_offsets.append(lenA)
            var flat_starts = List[Int]()
            flat_starts.append(8)
            flat_starts.append(30)
            var lengths = List[Int]()
            lengths.append(lenA)
            lengths.append(lenB)

            z_ctx.allgather_ranges[DTYPE](
                rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                    shard_buf.unsafe_ptr().as_unsafe_any_origin()
                ),
                shard,
                rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                    win_buf.unsafe_ptr().as_unsafe_any_origin()
                ),
                dst_offsets,
                flat_starts,
                lengths,
            )
            ctx.synchronize()

            var host_win = ctx.enqueue_create_host_buffer[DTYPE](win)
            win_buf.enqueue_copy_to(host_win)
            ctx.synchronize()
            for j in range(win):
                rank_windows[rank * win + j] = host_win.unsafe_ptr()[j]
        except e:
            print("allgather_ranges rank error:", e)

    sync_parallelize[_run_rank](WORLD_SIZE)

    # Every rank's window == full-vector values at [8,24) then [30,50).
    for r in range(WORLD_SIZE):
        for j in range(lenA):
            assert_almost_equal[DTYPE](
                rank_windows[r * win + j], Float32(8 + j), atol=1e-6
            )
        for j in range(lenB):
            assert_almost_equal[DTYPE](
                rank_windows[r * win + lenA + j], Float32(30 + j), atol=1e-6
            )

    rank_windows.free()
    cpu_coord_ptr[].free()
    cpu_coord_ptr.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
