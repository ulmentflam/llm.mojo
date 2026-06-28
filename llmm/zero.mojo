from std.sys.info import size_of
from comm import Signal, MAX_GPUS
from comm.allreduce import allreduce
from comm.allgather import allgather
from std.memory import UnsafePointer, memcpy
from std.collections import InlineArray
from layout.tile_layout import row_major
from std.algorithm import sync_parallelize
from comm.reducescatter import reducescatter
from layout import TileTensor, TensorLayout
from comm.sync import enable_p2p, init_signal_buffer
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.host.info import is_cpu, is_gpu
from std.atomic import Atomic


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #


comptime RANK_SIGNAL_SIZE = 8


# ===----------------------------------------------------------------------=== #
# CPU Collective Synchronization and Messaging
# ===----------------------------------------------------------------------=== #


struct CpuBarrier:
    var counter: Atomic[DType.int32]
    var generation: Atomic[DType.int32]
    var num_threads: Int

    def __init__(out self, num_threads: Int):
        self.counter = Atomic[DType.int32](0)
        self.generation = Atomic[DType.int32](0)
        self.num_threads = num_threads

    def wait(mut self) -> None:
        if self.num_threads <= 1:
            return
        var gen = self.generation.load()
        var val = self.counter.fetch_add(1) + 1
        if Int(val) == self.num_threads:
            self.counter.store(0)
            _ = self.generation.fetch_add(1)
        else:
            while self.generation.load() == gen:
                pass


struct CpuCoordinator:
    var barrier1: CpuBarrier
    var barrier2: CpuBarrier
    var shared_inputs: InlineArray[UnsafePointer[UInt8, MutUntrackedOrigin], 8]
    var shared_outputs: InlineArray[UnsafePointer[UInt8, MutUntrackedOrigin], 8]

    def __init__(out self, num_threads: Int):
        self.barrier1 = CpuBarrier(num_threads)
        self.barrier2 = CpuBarrier(num_threads)
        self.shared_inputs = InlineArray[
            UnsafePointer[UInt8, MutUntrackedOrigin], 8
        ](uninitialized=True)
        self.shared_outputs = InlineArray[
            UnsafePointer[UInt8, MutUntrackedOrigin], 8
        ](uninitialized=True)


# ===----------------------------------------------------------------------=== #
# Shared CPU Helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def _register_and_sync[
    dtype: DType, in_origin: Origin, out_origin: Origin
](
    rank: Int,
    cpu_coordinator_ptr: Optional[
        UnsafePointer[CpuCoordinator, MutUntrackedOrigin]
    ],
    input_ptr: UnsafePointer[Scalar[dtype], in_origin],
    output_ptr: UnsafePointer[Scalar[dtype], out_origin],
) raises -> UnsafePointer[CpuCoordinator, MutUntrackedOrigin]:
    if not cpu_coordinator_ptr:
        raise Error(
            "ZeroContext: cpu_coordinator_ptr must be provided for world_size >"
            " 1 on CPU"
        )
    var coord_ptr = cpu_coordinator_ptr.value()
    coord_ptr[].shared_inputs[rank] = rebind[
        UnsafePointer[UInt8, MutUntrackedOrigin]
    ](input_ptr)
    coord_ptr[].shared_outputs[rank] = rebind[
        UnsafePointer[UInt8, MutUntrackedOrigin]
    ](output_ptr)
    coord_ptr[].barrier1.wait()
    return coord_ptr


# ===----------------------------------------------------------------------=== #
# ZeRO Context
# ===----------------------------------------------------------------------=== #


struct ZeroContext[target: StaticString]:
    var rank: Int
    var world_size: Int
    var signal_buffer: DeviceBuffer[DType.uint8]
    var rank_sigs: InlineArray[
        UnsafePointer[Signal, MutUntrackedOrigin], RANK_SIGNAL_SIZE
    ]
    var ctx: DeviceContext
    var cpu_coordinator_ptr: Optional[
        UnsafePointer[CpuCoordinator, MutUntrackedOrigin]
    ]

    def __init__(
        out self,
        rank: Int,
        world_size: Int,
        ctx: DeviceContext,
        cpu_coord: Optional[
            UnsafePointer[CpuCoordinator, MutUntrackedOrigin]
        ] = Optional[UnsafePointer[CpuCoordinator, MutUntrackedOrigin]](),
    ) raises:
        self.rank = rank
        self.world_size = world_size
        self.ctx = ctx
        self.cpu_coordinator_ptr = cpu_coord

        comptime if not is_cpu[Self.target]():
            _ = enable_p2p()
            var signal_buffer_size = MAX_GPUS * size_of[Signal]()
            self.signal_buffer = ctx.enqueue_create_buffer[DType.uint8](
                signal_buffer_size
            )
            init_signal_buffer(self.signal_buffer, ctx)
            var signal_ptr = rebind[UnsafePointer[Signal, MutUntrackedOrigin]](
                self.signal_buffer.unsafe_ptr()
            )
            self.rank_sigs = InlineArray[
                UnsafePointer[Signal, MutUntrackedOrigin], RANK_SIGNAL_SIZE
            ](uninitialized=True)
            for i in range(RANK_SIGNAL_SIZE):
                self.rank_sigs[i] = signal_ptr + i
        else:
            self.signal_buffer = ctx.enqueue_create_buffer[DType.uint8](1)
            self.rank_sigs = InlineArray[
                UnsafePointer[Signal, MutUntrackedOrigin], RANK_SIGNAL_SIZE
            ](uninitialized=True)

    def get_rank_sigs_any(
        self,
    ) -> InlineArray[UnsafePointer[Signal, MutAnyOrigin], RANK_SIGNAL_SIZE]:
        var rank_sigs_any = InlineArray[
            UnsafePointer[Signal, MutAnyOrigin], RANK_SIGNAL_SIZE
        ](uninitialized=True)
        for i in range(RANK_SIGNAL_SIZE):
            rank_sigs_any[i] = self.rank_sigs[i].as_unsafe_any_origin()
        return rank_sigs_any

    def allreduce[
        dtype: DType,
        N_GPUS: Int,
        in_layout: TensorLayout,
        in_origin: Origin,
        out_layout: TensorLayout,
    ](
        self,
        input_tensors: InlineArray[
            TileTensor[dtype, in_layout, in_origin], N_GPUS
        ],
        output_tensor: TileTensor[dtype, out_layout, MutAnyOrigin],
        size: Int,
    ) raises:
        comptime if is_cpu[Self.target]():
            if self.world_size == 1:
                for j in range(size):
                    output_tensor.ptr[j] = input_tensors[0].ptr[j]
            else:
                var coord_ptr = _register_and_sync[
                    dtype, in_origin, MutAnyOrigin
                ](
                    self.rank,
                    self.cpu_coordinator_ptr,
                    input_tensors[0].ptr,
                    output_tensor.ptr,
                )

                var chunk = (size + self.world_size - 1) // self.world_size
                var start = self.rank * chunk
                var end = min(size, (self.rank + 1) * chunk)
                for j in range(start, end):
                    var sum_val = Scalar[dtype](0.0)
                    for k in range(self.world_size):
                        var k_input_ptr = rebind[
                            UnsafePointer[Scalar[dtype], MutUntrackedOrigin]
                        ](coord_ptr[].shared_inputs[k])
                        sum_val += k_input_ptr[j]
                    for k in range(self.world_size):
                        var k_output_ptr = rebind[
                            UnsafePointer[Scalar[dtype], MutUntrackedOrigin]
                        ](coord_ptr[].shared_outputs[k])
                        k_output_ptr[j] = sum_val
                coord_ptr[].barrier2.wait()
        else:
            var sigs = self.get_rank_sigs_any()
            allreduce[dtype, N_GPUS, in_layout, in_origin, out_layout](
                input_tensors, output_tensor, sigs, self.ctx
            )

    def reducescatter[
        dtype: DType,
        N_GPUS: Int,
        in_layout: TensorLayout,
        in_origin: Origin,
        out_layout: TensorLayout,
    ](
        self,
        input_buffers: InlineArray[
            TileTensor[dtype, in_layout, in_origin], N_GPUS
        ],
        output_buffer: TileTensor[dtype, out_layout, MutAnyOrigin],
        sharded_size: Int,
    ) raises:
        comptime if is_cpu[Self.target]():
            if self.world_size == 1:
                for j in range(sharded_size):
                    output_buffer.ptr[j] = input_buffers[0].ptr[j]
            else:
                var coord_ptr = _register_and_sync[
                    dtype, in_origin, MutAnyOrigin
                ](
                    self.rank,
                    self.cpu_coordinator_ptr,
                    input_buffers[self.rank].ptr,
                    output_buffer.ptr,
                )

                var offset = self.rank * sharded_size
                for j in range(sharded_size):
                    var sum_val = Scalar[dtype](0.0)
                    for k in range(self.world_size):
                        var k_input_ptr = rebind[
                            UnsafePointer[Scalar[dtype], MutUntrackedOrigin]
                        ](coord_ptr[].shared_inputs[k])
                        sum_val += k_input_ptr[offset + j]
                    output_buffer.ptr[j] = sum_val
                coord_ptr[].barrier2.wait()
        else:
            var sigs = self.get_rank_sigs_any()
            reducescatter[dtype, N_GPUS, in_layout, in_origin](
                input_buffers, output_buffer, sigs, self.ctx
            )

    def allgather[
        dtype: DType,
        N_GPUS: Int,
        in_layout: TensorLayout,
        in_origin: Origin,
        out_layout: TensorLayout,
        out_origin: MutOrigin,
    ](
        self,
        input_buffers: InlineArray[
            TileTensor[dtype, in_layout, in_origin], N_GPUS
        ],
        output_buffers: InlineArray[
            TileTensor[dtype, out_layout, out_origin], N_GPUS
        ],
        sharded_size: Int,
    ) raises:
        comptime if is_cpu[Self.target]():
            if self.world_size == 1:
                for j in range(sharded_size):
                    output_buffers[0].ptr[j] = input_buffers[0].ptr[j]
            else:
                var coord_ptr = _register_and_sync[
                    dtype, in_origin, out_origin
                ](
                    self.rank,
                    self.cpu_coordinator_ptr,
                    input_buffers[self.rank].ptr,
                    output_buffers[self.rank].ptr,
                )

                var offset = self.rank * sharded_size
                for k in range(self.world_size):
                    var k_output_ptr = rebind[
                        UnsafePointer[Scalar[dtype], MutUntrackedOrigin]
                    ](coord_ptr[].shared_outputs[k])
                    var my_input_ptr = rebind[
                        UnsafePointer[Scalar[dtype], MutUntrackedOrigin]
                    ](coord_ptr[].shared_inputs[self.rank])
                    for j in range(sharded_size):
                        k_output_ptr[offset + j] = my_input_ptr[j]
                coord_ptr[].barrier2.wait()
        else:
            var sigs = self.get_rank_sigs_any()
            allgather[
                dtype, N_GPUS, in_layout, in_origin, out_layout, out_origin
            ](input_buffers, output_buffers, sigs, self.ctx, my_rank=self.rank)


# ===----------------------------------------------------------------------=== #
# ShardedParameter for Zero-3 Sharding & Offload
# ===----------------------------------------------------------------------=== #


struct ShardedParameter[
    dtype: DType, N_GPUS: Int, target: StaticString, offload: Bool = False
]:
    var sharded_buffer: DeviceBuffer[Self.dtype]
    var host_sharded_buffer: HostBuffer[Self.dtype]
    var size: Int
    var sharded_size: Int

    def __init__(out self, size: Int, ctx: DeviceContext) raises:
        self.size = size
        self.sharded_size = size // Self.N_GPUS

        comptime if Self.target == "cpu":
            self.sharded_buffer = ctx.enqueue_create_buffer[Self.dtype](
                self.sharded_size
            )
            self.host_sharded_buffer = ctx.enqueue_create_host_buffer[
                Self.dtype
            ](1)
        else:
            self.sharded_buffer = ctx.enqueue_create_buffer[Self.dtype](
                self.sharded_size
            )
            comptime if Self.offload:
                self.host_sharded_buffer = ctx.enqueue_create_host_buffer[
                    Self.dtype
                ](self.sharded_size)
            else:
                self.host_sharded_buffer = ctx.enqueue_create_host_buffer[
                    Self.dtype
                ](1)

    # Load sharded parameter from CPU HostBuffer to GPU DeviceBuffer
    @always_inline
    def load_to_gpu(self, ctx: DeviceContext) raises -> None:
        comptime if Self.offload and not is_cpu[Self.target]():
            self.sharded_buffer.enqueue_copy_from(self.host_sharded_buffer)
            ctx.synchronize()

    # Offload sharded buffer from GPU DeviceBuffer to CPU HostBuffer
    @always_inline
    def offload_to_cpu(self, ctx: DeviceContext) raises -> None:
        comptime if Self.offload and not is_cpu[Self.target]():
            self.sharded_buffer.enqueue_copy_to(self.host_sharded_buffer)
            ctx.synchronize()

    def gather[
        in_layout: TensorLayout, out_layout: TensorLayout
    ](
        self,
        zero_ctx: ZeroContext[Self.target],
        ctx: DeviceContext,
        all_sharded_buffers: InlineArray[
            UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin], Self.N_GPUS
        ],
        in_tensor_layout: in_layout,
        out_tensor_layout: out_layout,
    ) raises -> DeviceBuffer[Self.dtype]:
        var full_buffer = ctx.enqueue_create_buffer[Self.dtype](self.size)

        comptime if is_cpu[Self.target]():
            if Self.N_GPUS == 1:
                var dest_ptr = full_buffer.unsafe_ptr()
                var src_ptr = all_sharded_buffers[0]
                for j in range(self.size):
                    dest_ptr[j] = src_ptr[j]
            else:
                var coord_ptr = _register_and_sync[
                    Self.dtype, MutUntrackedOrigin, MutUntrackedOrigin
                ](
                    zero_ctx.rank,
                    zero_ctx.cpu_coordinator_ptr,
                    all_sharded_buffers[zero_ctx.rank],
                    full_buffer.unsafe_ptr(),
                )

                var offset = zero_ctx.rank * self.sharded_size
                for k in range(Self.N_GPUS):
                    var k_output_ptr = rebind[
                        UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]
                    ](coord_ptr[].shared_outputs[k])
                    var my_input_ptr = rebind[
                        UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]
                    ](coord_ptr[].shared_inputs[zero_ctx.rank])
                    for j in range(self.sharded_size):
                        k_output_ptr[offset + j] = my_input_ptr[j]
                coord_ptr[].barrier2.wait()
        else:
            var input_tensors = InlineArray[
                TileTensor[Self.dtype, in_layout, ImmutAnyOrigin], Self.N_GPUS
            ](uninitialized=True)

            var out_tile = TileTensor(
                Span[Scalar[Self.dtype], MutAnyOrigin](
                    ptr=rebind[UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]](
                        full_buffer.unsafe_ptr().as_unsafe_any_origin()
                    ),
                    length=self.size,
                ),
                out_tensor_layout,
            )

            var output_tensors = InlineArray[
                TileTensor[Self.dtype, out_layout, MutAnyOrigin], Self.N_GPUS
            ](uninitialized=True)

            for i in range(Self.N_GPUS):
                input_tensors[i] = TileTensor(
                    Span[Scalar[Self.dtype], ImmutAnyOrigin](
                        ptr=rebind[
                            UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]
                        ](all_sharded_buffers[i].as_unsafe_any_origin()),
                        length=self.sharded_size,
                    ),
                    in_tensor_layout,
                )
                output_tensors[i] = out_tile

            var rank_sigs_any = InlineArray[
                UnsafePointer[Signal, MutAnyOrigin], RANK_SIGNAL_SIZE
            ](uninitialized=True)
            for i in range(RANK_SIGNAL_SIZE):
                rank_sigs_any[i] = zero_ctx.rank_sigs[i].as_unsafe_any_origin()

            allgather[
                Self.dtype,
                Self.N_GPUS,
                in_layout,
                ImmutAnyOrigin,
                out_layout,
                MutAnyOrigin,
            ](
                input_tensors,
                output_tensors,
                rank_sigs_any,
                ctx,
                my_rank=zero_ctx.rank,
            )
            ctx.synchronize()

        return full_buffer
