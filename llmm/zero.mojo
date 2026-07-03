from std.atomic import Atomic
from std.sys.info import size_of
from std.ffi import external_call
from comm import Signal, MAX_GPUS
from comm.allgather import allgather
from comm.allreduce import allreduce
from std.collections import InlineArray
from layout.tile_layout import row_major
from std.algorithm import sync_parallelize
from layout import TileTensor, TensorLayout
from std.memory import UnsafePointer, memcpy, alloc
from std.gpu.host.info import is_cpu, is_gpu
from comm.reducescatter import reducescatter
from comm.sync import enable_p2p, init_signal_buffer
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.sys import has_nvidia_gpu_accelerator


# ===----------------------------------------------------------------------=== #
# CPU Collective Synchronization and Messaging
# ===----------------------------------------------------------------------=== #


struct SpinLock:
    var locked: Atomic[DType.int32]

    def __init__(out self):
        self.locked = Atomic[DType.int32](0)

    def lock(mut self) -> None:
        while True:
            var expected = Scalar[DType.int32](0)
            if self.locked.compare_exchange(expected, 1):
                break
            _ = external_call["sched_yield", Int32]()

    def unlock(mut self) -> None:
        self.locked.store(0)


struct CpuBarrier:
    var counter: Int
    var generation: Atomic[DType.int32]
    var num_threads: Int
    var lock: SpinLock

    def __init__(out self, num_threads: Int):
        self.counter = 0
        self.generation = Atomic[DType.int32](0)
        self.num_threads = num_threads
        self.lock = SpinLock()

    def wait(mut self) -> None:
        if self.num_threads <= 1:
            return

        self.lock.lock()
        var gen = self.generation.load()
        self.counter += 1
        if self.counter == self.num_threads:
            self.counter = 0
            self.generation.store(gen + 1)
            self.lock.unlock()
        else:
            self.lock.unlock()
            while self.generation.load() == gen:
                _ = external_call["sched_yield", Int32]()


struct CpuCoordinator:
    var barrier1: UnsafePointer[CpuBarrier, MutUntrackedOrigin]
    var barrier2: UnsafePointer[CpuBarrier, MutUntrackedOrigin]
    var shared_inputs: UnsafePointer[
        UnsafePointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin
    ]
    var shared_outputs: UnsafePointer[
        UnsafePointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin
    ]

    def __init__(out self, num_threads: Int):
        self.barrier1 = alloc[CpuBarrier](1)
        self.barrier1[] = CpuBarrier(num_threads)
        self.barrier2 = alloc[CpuBarrier](1)
        self.barrier2[] = CpuBarrier(num_threads)
        self.shared_inputs = alloc[UnsafePointer[UInt8, MutUntrackedOrigin]](
            num_threads
        )
        self.shared_outputs = alloc[UnsafePointer[UInt8, MutUntrackedOrigin]](
            num_threads
        )

    def free(self) -> None:
        self.barrier1.free()
        self.barrier2.free()
        self.shared_inputs.free()
        self.shared_outputs.free()


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
    coord_ptr[].barrier1[].wait()
    return coord_ptr


def _allreduce_cpu[
    dtype: DType
](
    rank: Int,
    world_size: Int,
    size: Int,
    coord_ptr: UnsafePointer[CpuCoordinator, MutUntrackedOrigin],
):
    var sharded = size // world_size
    var output_shard = rebind[UnsafePointer[Scalar[dtype], MutUntrackedOrigin]](
        coord_ptr[].shared_outputs[rank]
    )
    var offset = rank * sharded
    _reducescatter_cpu[dtype](
        rank,
        world_size,
        sharded,
        (output_shard + offset).as_unsafe_any_origin(),
        coord_ptr,
    )

    coord_ptr[].barrier2[].wait()
    _allgather_cpu[dtype](rank, world_size, sharded, coord_ptr)
    coord_ptr[].barrier1[].wait()


@always_inline
def _reducescatter_cpu[
    dtype: DType
](
    rank: Int,
    world_size: Int,
    sharded_size: Int,
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    coord_ptr: UnsafePointer[CpuCoordinator, MutUntrackedOrigin],
):
    var offset = rank * sharded_size
    for j in range(sharded_size):
        var sum_val = Scalar[dtype](0.0)
        for k in range(world_size):
            var k_input_ptr = rebind[
                UnsafePointer[Scalar[dtype], MutUntrackedOrigin]
            ](coord_ptr[].shared_inputs[k])
            sum_val += k_input_ptr[offset + j]
        output_ptr[j] = sum_val


@always_inline
def _allgather_cpu[
    dtype: DType
](
    rank: Int,
    world_size: Int,
    sharded_size: Int,
    coord_ptr: UnsafePointer[CpuCoordinator, MutUntrackedOrigin],
):
    var offset = rank * sharded_size
    for k in range(world_size):
        var k_output_ptr = rebind[
            UnsafePointer[Scalar[dtype], MutUntrackedOrigin]
        ](coord_ptr[].shared_outputs[k])
        var my_input_ptr = rebind[
            UnsafePointer[Scalar[dtype], MutUntrackedOrigin]
        ](coord_ptr[].shared_inputs[rank])
        for j in range(sharded_size):
            k_output_ptr[offset + j] = my_input_ptr[offset + j]


# ===----------------------------------------------------------------------=== #
# ZeRO Context
# ===----------------------------------------------------------------------=== #


struct ZeroContext[target: StaticString, N: Int = 1]:
    var rank: Int
    var zero_stage: Int
    var ctx: DeviceContext
    var signal_buffer: DeviceBuffer[DType.uint8]
    var cpu_coordinator_ptr: Optional[
        UnsafePointer[CpuCoordinator, MutUntrackedOrigin]
    ]

    def __init__(
        out self,
        rank: Int,
        zero_stage: Int,
        ctx: DeviceContext,
        cpu_coord: Optional[
            UnsafePointer[CpuCoordinator, MutUntrackedOrigin]
        ] = Optional[UnsafePointer[CpuCoordinator, MutUntrackedOrigin]](),
    ) raises:
        self.rank = rank
        self.zero_stage = zero_stage
        self.ctx = ctx
        self.cpu_coordinator_ptr = cpu_coord
        # P2P signaling (enable_p2p / init_signal_buffer) uses CUDA IPC handles
        # that have no Metal equivalent. On Apple GPU or CPU the signal_buffer
        # is a 1-byte dummy that is never used; the GPU collective methods
        # (allreduce / reducescatter / allgather) raise at runtime for N>=2 on
        # non-NVIDIA targets — see the comments in those methods.
        comptime if not is_cpu[Self.target]() and has_nvidia_gpu_accelerator():
            _ = enable_p2p()
            self.signal_buffer = ctx.enqueue_create_buffer[DType.uint8](
                max(Self.N, MAX_GPUS) * size_of[Signal]()
            )
            init_signal_buffer(self.signal_buffer, ctx)
        else:
            self.signal_buffer = ctx.enqueue_create_buffer[DType.uint8](1)

    def signal_ptr(self) -> UnsafePointer[Signal, MutUntrackedOrigin]:
        return rebind[UnsafePointer[Signal, MutUntrackedOrigin]](
            self.signal_buffer.unsafe_ptr()
        )

    def get_rank_sigs_any(
        self,
    ) -> InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS]:
        var rank_sigs_any = InlineArray[
            UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
        ](uninitialized=True)
        for i in range(MAX_GPUS):
            rank_sigs_any[i] = (self.signal_ptr() + i).as_unsafe_any_origin()
        return rank_sigs_any

    def allreduce[
        dtype: DType
    ](self, ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], size: Int,) raises:
        comptime if is_cpu[Self.target]():
            if Self.N == 1:
                return
            if not self.cpu_coordinator_ptr:
                return
            var coord_ptr = _register_and_sync[
                dtype, MutAnyOrigin, MutAnyOrigin
            ](
                self.rank,
                self.cpu_coordinator_ptr,
                ptr,
                ptr,
            )
            _allreduce_cpu[dtype](self.rank, Self.N, size, coord_ptr)
            coord_ptr[].barrier2[].wait()
        else:
            # Multi-GPU collectives use the `comm` package's allreduce /
            # reducescatter / allgather, which rely on CUDA P2P (NVLink /
            # CUDA IPC) — there is no equivalent for Apple Metal. For N>=2
            # on a non-NVIDIA GPU this branch raises at runtime; single-GPU
            # (N==1) returns early above and never hits this check.
            comptime if Self.N >= 2:
                comptime if has_nvidia_gpu_accelerator():
                    var in_layout = row_major(size)
                    var out_layout = row_major(size)

                    var input_tensors = InlineArray[
                        TileTensor[dtype, type_of(in_layout), MutAnyOrigin],
                        Self.N,
                    ](uninitialized=True)
                    for i in range(Self.N):
                        input_tensors[i] = TileTensor(
                            Span[Scalar[dtype], MutAnyOrigin](
                                ptr=ptr, length=size
                            ),
                            in_layout,
                        )

                    var output_tensor = TileTensor(
                        Span[Scalar[dtype], MutAnyOrigin](ptr=ptr, length=size),
                        out_layout,
                    )

                    var rank_sigs = self.get_rank_sigs_any()

                    allreduce[
                        dtype,
                        Self.N,
                        type_of(in_layout),
                        MutAnyOrigin,
                        type_of(out_layout),
                    ](
                        input_tensors,
                        output_tensor,
                        rank_sigs,
                        self.ctx,
                    )
                    self.ctx.synchronize()
                else:
                    raise Error(
                        "Multi-GPU collectives require Nvidia GPUs; not"
                        " supported on this hardware"
                    )

    def reducescatter[
        dtype: DType
    ](
        self,
        input_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        sharded_size: Int,
    ) raises:
        comptime if is_cpu[Self.target]():
            if Self.N == 1:
                for j in range(sharded_size):
                    output_ptr[j] = input_ptr[j]
                return
            if not self.cpu_coordinator_ptr:
                return

            var coord_ptr = _register_and_sync[
                dtype, MutAnyOrigin, MutAnyOrigin
            ](
                self.rank,
                self.cpu_coordinator_ptr,
                input_ptr,
                output_ptr,
            )
            _reducescatter_cpu[dtype](
                self.rank, Self.N, sharded_size, output_ptr, coord_ptr
            )
            coord_ptr[].barrier2[].wait()
        else:
            # Same CUDA P2P requirement as allreduce — raises on non-NVIDIA for N>=2.
            comptime if Self.N >= 2:
                comptime if has_nvidia_gpu_accelerator():
                    var in_layout = row_major(sharded_size)
                    var out_layout = row_major(sharded_size)

                    var input_tensors = InlineArray[
                        TileTensor[dtype, type_of(in_layout), MutAnyOrigin],
                        Self.N,
                    ](uninitialized=True)
                    for i in range(Self.N):
                        input_tensors[i] = TileTensor(
                            Span[Scalar[dtype], MutAnyOrigin](
                                ptr=input_ptr + i * sharded_size,
                                length=sharded_size,
                            ),
                            in_layout,
                        )

                    var output_tensor = TileTensor(
                        Span[Scalar[dtype], MutAnyOrigin](
                            ptr=output_ptr, length=sharded_size
                        ),
                        out_layout,
                    )

                    var rank_sigs = self.get_rank_sigs_any()

                    reducescatter[
                        dtype,
                        Self.N,
                        type_of(in_layout),
                        MutAnyOrigin,
                    ](
                        input_tensors,
                        output_tensor,
                        rank_sigs,
                        self.ctx,
                    )
                    self.ctx.synchronize()
                else:
                    raise Error(
                        "Multi-GPU collectives require Nvidia GPUs; not"
                        " supported on this hardware"
                    )

    def allgather[
        dtype: DType
    ](
        self,
        ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        sharded_size: Int,
    ) raises:
        comptime if is_cpu[Self.target]():
            if Self.N == 1:
                return
            if not self.cpu_coordinator_ptr:
                return

            var coord_ptr = _register_and_sync[
                dtype, MutAnyOrigin, MutAnyOrigin
            ](
                self.rank,
                self.cpu_coordinator_ptr,
                ptr,
                ptr,
            )
            _allgather_cpu[dtype](self.rank, Self.N, sharded_size, coord_ptr)
            coord_ptr[].barrier2[].wait()
        else:
            # Same CUDA P2P requirement as allreduce — raises on non-NVIDIA for N>=2.
            comptime if Self.N >= 2:
                comptime if has_nvidia_gpu_accelerator():
                    var in_layout = row_major(sharded_size)
                    var out_layout = row_major(sharded_size)

                    var input_tensors = InlineArray[
                        TileTensor[dtype, type_of(in_layout), MutAnyOrigin],
                        Self.N,
                    ](uninitialized=True)
                    for i in range(Self.N):
                        input_tensors[i] = TileTensor(
                            Span[Scalar[dtype], MutAnyOrigin](
                                ptr=ptr + i * sharded_size,
                                length=sharded_size,
                            ),
                            in_layout,
                        )

                    var output_tensors = InlineArray[
                        TileTensor[dtype, type_of(out_layout), MutAnyOrigin],
                        Self.N,
                    ](uninitialized=True)
                    for i in range(Self.N):
                        output_tensors[i] = TileTensor(
                            Span[Scalar[dtype], MutAnyOrigin](
                                ptr=ptr + i * sharded_size,
                                length=sharded_size,
                            ),
                            out_layout,
                        )

                    var rank_sigs = self.get_rank_sigs_any()

                    allgather[
                        dtype,
                        Self.N,
                        type_of(in_layout),
                        MutAnyOrigin,
                        type_of(out_layout),
                        MutAnyOrigin,
                    ](
                        input_tensors,
                        output_tensors,
                        rank_sigs,
                        self.ctx,
                        my_rank=self.rank,
                    )
                    self.ctx.synchronize()
                else:
                    raise Error(
                        "Multi-GPU collectives require Nvidia GPUs; not"
                        " supported on this hardware"
                    )


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
        zero_ctx: ZeroContext[Self.target, Self.N_GPUS],
        all_sharded_buffers: InlineArray[
            UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin], Self.N_GPUS
        ],
        in_tensor_layout: in_layout,
        out_tensor_layout: out_layout,
    ) raises -> DeviceBuffer[Self.dtype]:
        var full_buffer = zero_ctx.ctx.enqueue_create_buffer[Self.dtype](
            self.size
        )

        comptime if is_cpu[Self.target]():
            if Self.N_GPUS == 1:
                var dest_ptr = full_buffer.unsafe_ptr()
                var src_ptr = all_sharded_buffers[0]
                for j in range(self.size):
                    dest_ptr[j] = src_ptr[j]
                return full_buffer
            if not zero_ctx.cpu_coordinator_ptr:
                return full_buffer
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
            coord_ptr[].barrier2[].wait()
        else:
            comptime if Self.N_GPUS == 1:
                var dest_ptr = rebind[
                    UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
                ](full_buffer.unsafe_ptr().as_unsafe_any_origin())
                var src_ptr = rebind[
                    UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]
                ](all_sharded_buffers[0].as_unsafe_any_origin())
                zero_ctx.ctx.enqueue_copy(
                    dst_ptr=dest_ptr,
                    src_ptr=src_ptr,
                    size=self.size,
                )
                zero_ctx.ctx.synchronize()
            else:
                # Same CUDA P2P requirement — raises on non-NVIDIA for N_GPUS>=2.
                comptime if has_nvidia_gpu_accelerator():
                    var input_tensors = InlineArray[
                        TileTensor[Self.dtype, in_layout, ImmutAnyOrigin],
                        Self.N_GPUS,
                    ](uninitialized=True)

                    var out_tile = TileTensor(
                        Span[Scalar[Self.dtype], MutAnyOrigin](
                            ptr=rebind[
                                UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
                            ](full_buffer.unsafe_ptr().as_unsafe_any_origin()),
                            length=self.size,
                        ),
                        out_tensor_layout,
                    )

                    var output_tensors = InlineArray[
                        TileTensor[Self.dtype, out_layout, MutAnyOrigin],
                        Self.N_GPUS,
                    ](uninitialized=True)

                    for i in range(Self.N_GPUS):
                        input_tensors[i] = TileTensor(
                            Span[Scalar[Self.dtype], ImmutAnyOrigin](
                                ptr=rebind[
                                    UnsafePointer[
                                        Scalar[Self.dtype], ImmutAnyOrigin
                                    ]
                                ](
                                    all_sharded_buffers[
                                        i
                                    ].as_unsafe_any_origin()
                                ),
                                length=self.sharded_size,
                            ),
                            in_tensor_layout,
                        )
                        output_tensors[i] = out_tile

                    var rank_sigs_any = InlineArray[
                        UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
                    ](uninitialized=True)
                    for i in range(MAX_GPUS):
                        rank_sigs_any[i] = (
                            zero_ctx.signal_ptr() + i
                        ).as_unsafe_any_origin()

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
                        zero_ctx.ctx,
                        my_rank=zero_ctx.rank,
                    )
                    zero_ctx.ctx.synchronize()
                else:
                    raise Error(
                        "Multi-GPU collectives require Nvidia GPUs; not"
                        " supported on this hardware"
                    )

        return full_buffer
