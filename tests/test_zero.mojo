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
    all_sharded[0] = param.sharded_buffer.unsafe_ptr()

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
    all_sharded[0] = param.sharded_buffer.unsafe_ptr()

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

            shared_ptrs[rank] = param.sharded_buffer.unsafe_ptr()
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
