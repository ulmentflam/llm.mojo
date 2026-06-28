# ===----------------------------------------------------------------------=== #
# Pure-Mojo unit + property tests for llmm.zero.
#
# Run with:  make test-mojo   (equivalent to `mojo run -I . tests/test_zero.mojo`)
# ===----------------------------------------------------------------------=== #

from std.testing import (
    assert_almost_equal,
    assert_true,
    assert_equal,
    TestSuite,
)
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.collections import InlineArray
from layout import TileTensor
from layout.tile_layout import row_major
from llmm.zero import ZeroContext, ShardedParameter, CpuCoordinator


def test_zero_context_init_cpu() raises:
    var ctx = DeviceContext(api="cpu")
    var z_ctx = ZeroContext[target="cpu"](rank=0, world_size=1, ctx=ctx)
    assert_equal(z_ctx.rank, 0)
    assert_equal(z_ctx.world_size, 1)


def test_single_cpu_allreduce() raises:
    var ctx = DeviceContext(api="cpu")
    var z_ctx = ZeroContext[target="cpu"](rank=0, world_size=1, ctx=ctx)

    comptime DTYPE = DType.float32
    comptime size = 128

    var host_in = ctx.enqueue_create_host_buffer[DTYPE](size)
    var host_out = ctx.enqueue_create_host_buffer[DTYPE](size)

    var host_in_ptr = host_in.unsafe_ptr()
    var host_out_ptr = host_out.unsafe_ptr()
    for i in range(size):
        host_in_ptr[i] = 4.2
        host_out_ptr[i] = 0.0

    var in_buf = ctx.enqueue_create_buffer[DTYPE](size)
    var out_buf = ctx.enqueue_create_buffer[DTYPE](size)

    in_buf.enqueue_copy_from(host_in)
    out_buf.enqueue_copy_from(host_out)
    ctx.synchronize()

    comptime in_layout = row_major(1, size)
    comptime out_layout = row_major(1, size)

    var in_tile = TileTensor(
        Span[Scalar[DTYPE], ImmutAnyOrigin](
            ptr=rebind[UnsafePointer[Scalar[DTYPE], ImmutAnyOrigin]](
                in_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
            length=size,
        ),
        in_layout,
    )

    var out_tile = TileTensor(
        Span[Scalar[DTYPE], MutAnyOrigin](
            ptr=rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                out_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
            length=size,
        ),
        out_layout,
    )

    comptime InTileType = type_of(in_tile)
    var input_tensors = InlineArray[InTileType, 1](uninitialized=True)
    input_tensors[0] = in_tile

    z_ctx.allreduce[
        DTYPE, 1, type_of(in_layout), ImmutAnyOrigin, type_of(out_layout)
    ](input_tensors, out_tile, size)
    ctx.synchronize()

    out_buf.enqueue_copy_to(host_out)
    ctx.synchronize()

    for i in range(size):
        assert_almost_equal(host_out_ptr[i], 4.2, atol=1e-6)


def test_single_cpu_reducescatter() raises:
    var ctx = DeviceContext(api="cpu")
    var z_ctx = ZeroContext[target="cpu"](rank=0, world_size=1, ctx=ctx)

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

    comptime in_layout = row_major(1, size)
    comptime out_layout = row_major(1, size)

    var in_tile = TileTensor(
        Span[Scalar[DTYPE], ImmutAnyOrigin](
            ptr=rebind[UnsafePointer[Scalar[DTYPE], ImmutAnyOrigin]](
                in_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
            length=size,
        ),
        in_layout,
    )

    var out_tile = TileTensor(
        Span[Scalar[DTYPE], MutAnyOrigin](
            ptr=rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                out_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
            length=size,
        ),
        out_layout,
    )

    comptime InTileType = type_of(in_tile)
    var input_tensors = InlineArray[InTileType, 1](uninitialized=True)
    input_tensors[0] = in_tile

    z_ctx.reducescatter[
        DTYPE, 1, type_of(in_layout), ImmutAnyOrigin, type_of(out_layout)
    ](input_tensors, out_tile, size)
    ctx.synchronize()

    out_buf.enqueue_copy_to(host_out)
    ctx.synchronize()

    for i in range(size):
        assert_almost_equal(host_out_ptr[i], 1.5, atol=1e-6)


def test_single_cpu_allgather() raises:
    var ctx = DeviceContext(api="cpu")
    var z_ctx = ZeroContext[target="cpu"](rank=0, world_size=1, ctx=ctx)

    comptime DTYPE = DType.float32
    comptime size = 32

    var host_in = ctx.enqueue_create_host_buffer[DTYPE](size)
    var host_out = ctx.enqueue_create_host_buffer[DTYPE](size)

    var host_in_ptr = host_in.unsafe_ptr()
    var host_out_ptr = host_out.unsafe_ptr()
    for i in range(size):
        host_in_ptr[i] = 8.8
        host_out_ptr[i] = 0.0

    var in_buf = ctx.enqueue_create_buffer[DTYPE](size)
    var out_buf = ctx.enqueue_create_buffer[DTYPE](size)

    in_buf.enqueue_copy_from(host_in)
    out_buf.enqueue_copy_from(host_out)
    ctx.synchronize()

    comptime in_layout = row_major(1, size)
    comptime out_layout = row_major(1, size)

    var in_tile = TileTensor(
        Span[Scalar[DTYPE], ImmutAnyOrigin](
            ptr=rebind[UnsafePointer[Scalar[DTYPE], ImmutAnyOrigin]](
                in_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
            length=size,
        ),
        in_layout,
    )

    var out_tile = TileTensor(
        Span[Scalar[DTYPE], MutAnyOrigin](
            ptr=rebind[UnsafePointer[Scalar[DTYPE], MutAnyOrigin]](
                out_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
            length=size,
        ),
        out_layout,
    )

    comptime InTileType = type_of(in_tile)
    comptime OutTileType = type_of(out_tile)

    var input_tensors = InlineArray[InTileType, 1](uninitialized=True)
    input_tensors[0] = in_tile

    var output_tensors = InlineArray[OutTileType, 1](uninitialized=True)
    output_tensors[0] = out_tile

    z_ctx.allgather[
        DTYPE,
        1,
        type_of(in_layout),
        ImmutAnyOrigin,
        type_of(out_layout),
        MutAnyOrigin,
    ](input_tensors, output_tensors, size)
    ctx.synchronize()

    out_buf.enqueue_copy_to(host_out)
    ctx.synchronize()

    for i in range(size):
        assert_almost_equal(host_out_ptr[i], 8.8, atol=1e-6)


def test_sharded_parameter_gather_cpu() raises:
    var ctx = DeviceContext(api="cpu")
    var z_ctx = ZeroContext[target="cpu"](rank=0, world_size=1, ctx=ctx)

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
        z_ctx, ctx, all_sharded, in_layout, out_layout
    )

    full.enqueue_copy_to(host_out)
    ctx.synchronize()

    for i in range(size):
        assert_almost_equal(host_out_ptr[i], 7.5, atol=1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
