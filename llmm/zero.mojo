from std.atomic import Atomic
from std.sys.info import size_of
from std.ffi import external_call
from std.math import ceildiv
from comm import Signal, MAX_GPUS
from comm.allgather import allgather
from std.collections import InlineArray
from layout.tile_layout import row_major
from std.algorithm import sync_parallelize
from std.gpu import block_dim, block_idx, thread_idx
from layout import TileTensor, TensorLayout
from std.memory import UnsafePointer, memcpy, alloc
from std.gpu.host.info import is_cpu, is_gpu
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
    """Host-thread coordinator for N ranks living in one process.

    Despite the name it coordinates both CPU-rank and GPU-rank threads: the
    barriers and pointer-exchange slots are pure host constructs, and for the
    multi-GPU path `shared_inputs`/`shared_outputs` carry *device* pointer
    addresses (each rank registers its own device buffer, peers read the
    address and pull via cross-device staged copies).
    """

    var barrier1: UnsafePointer[CpuBarrier, MutUntrackedOrigin]
    var barrier2: UnsafePointer[CpuBarrier, MutUntrackedOrigin]
    var shared_inputs: UnsafePointer[
        UnsafePointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin
    ]
    var shared_outputs: UnsafePointer[
        UnsafePointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin
    ]
    # One Float64 slot per rank for host-side scalar reductions (loss
    # averaging, global grad-norm partial sums across shards).
    var shared_scalars: UnsafePointer[Float64, MutUntrackedOrigin]

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
        self.shared_scalars = alloc[Float64](num_threads)

    def free(self) -> None:
        self.barrier1.free()
        self.barrier2.free()
        self.shared_inputs.free()
        self.shared_outputs.free()
        self.shared_scalars.free()


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
# GPU staged-copy collective kernel
# ===----------------------------------------------------------------------=== #


def _add_inplace_gpu[
    dtype: DType
](
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
) -> None:
    """dst[i] += src[i], accumulating in fp32 (bf16-safe)."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < size:
        var s = (dst + idx).load().cast[DType.float32]() + (
            src + idx
        ).load().cast[DType.float32]()
        (dst + idx).store(s.cast[dtype]())


comptime _ADD_BLOCK = 256


# ===----------------------------------------------------------------------=== #
# ZeRO Context
# ===----------------------------------------------------------------------=== #


struct ZeroContext[target: StaticString, N: Int = 1]:
    """Per-rank collective context.

    CPU (N >= 2): ranks are host threads; collectives are memcpy loops
    synchronized by the shared CpuCoordinator barriers.

    NVIDIA GPU (N >= 2): ranks are host threads, one DeviceContext per GPU
    (device_id == rank). The GPUs on the target box expose no CUDA P2P
    mappings (DeviceContext.can_access == False even though nvidia-smi
    reports P2P "OK"), so Modular's `comm` kernels — which hard-require P2P
    — are unusable; the collectives are instead hand-rolled as
    reduce-scatter + all-gather over driver-staged cross-device copies:

      * each rank registers its device pointer in the shared coordinator,
      * peers wrap the registered addresses in non-owning DeviceBuffer views
        (with a peer DeviceContext handle) and pull slices via
        ctx.enqueue_copy(dst_buf=..., src_buf=...), which the driver routes
        without P2P mappings,
      * partial sums accumulate through a per-rank scratch shard buffer with
        a small fp32-accumulate add kernel (_add_inplace_gpu).

    Traffic per rank is 2*(N-1)/N of the buffer per allreduce — same as a
    ring — measured ~75 GB/s aggregate on the 8-GPU fp32 GPT-2 gradient
    (~92 ms), vs. N*(N-1) for comm's naive all-pull fallback (which also
    crashes on cross-device raw-pointer copies on this toolchain).

    signal_buffer remains only because the (P2P-only, unusable here)
    comm-based ShardedParameter.gather path reads it; it is a 1-byte dummy.
    """

    var rank: Int
    var zero_stage: Int
    var ctx: DeviceContext
    var signal_buffer: DeviceBuffer[DType.uint8]
    var cpu_coordinator_ptr: Optional[
        UnsafePointer[CpuCoordinator, MutUntrackedOrigin]
    ]
    # Multi-GPU staged-copy state (empty / 1-byte until ensure_comm_setup runs
    # on an NVIDIA GPU target with N >= 2).
    var peer_ctxs: List[DeviceContext]
    var comm_scratch: DeviceBuffer[DType.uint8]
    var comm_scratch_bytes: Int

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
        self.peer_ctxs = List[DeviceContext]()
        self.comm_scratch_bytes = 0
        # 1-byte placeholders. ensure_comm_setup sizes comm_scratch for real
        # multi-GPU use; signal_buffer is a dummy on every path now (see the
        # struct docstring — the P2P comm kernels are unusable on this box,
        # and every collective early-returns for N == 1 without touching it).
        self.signal_buffer = ctx.enqueue_create_buffer[DType.uint8](1)
        self.comm_scratch = ctx.enqueue_create_buffer[DType.uint8](1)

    def ensure_comm_setup(mut self, max_shard_bytes: Int) raises:
        """Size the staged-copy scratch shard and build peer DeviceContext
        handles. Must be called (with the largest per-rank shard size in
        bytes) before any GPU collective at N >= 2; a no-op elsewhere.
        """
        comptime if not is_cpu[Self.target]() and Self.N >= 2:
            comptime if has_nvidia_gpu_accelerator():
                if not self.cpu_coordinator_ptr:
                    raise Error(
                        "ZeroContext: a shared CpuCoordinator is required for"
                        " multi-GPU collectives (world_size >= 2)"
                    )
                if self.comm_scratch_bytes < max_shard_bytes:
                    self.comm_scratch = self.ctx.enqueue_create_buffer[
                        DType.uint8
                    ](max_shard_bytes)
                    self.comm_scratch_bytes = max_shard_bytes
                    self.ctx.synchronize()
                if len(self.peer_ctxs) == 0:
                    for i in range(Self.N):
                        self.peer_ctxs.append(DeviceContext(device_id=i))
            else:
                raise Error(
                    "Multi-GPU collectives require Nvidia GPUs; not supported"
                    " on this hardware"
                )

    def allreduce_scalar(self, v: Float64) raises -> Float64:
        """Sum a host-side scalar across all ranks (any target). Returns v
        unchanged for N == 1 / missing coordinator."""
        if Self.N == 1 or not self.cpu_coordinator_ptr:
            return v
        var coord_ptr = self.cpu_coordinator_ptr.value()
        coord_ptr[].shared_scalars[self.rank] = v
        coord_ptr[].barrier1[].wait()
        var total = Float64(0.0)
        for i in range(Self.N):
            total += coord_ptr[].shared_scalars[i]
        coord_ptr[].barrier2[].wait()
        return total

    @always_inline
    def _check_scratch[dtype: DType](self, shard_elems: Int) raises:
        if self.comm_scratch_bytes < shard_elems * size_of[Scalar[dtype]]():
            raise Error(
                "ZeroContext: comm scratch too small — call"
                " ensure_comm_setup(max_shard_bytes) before GPU collectives"
            )

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
            # Staged-copy allreduce = in-place reduce-scatter + all-gather.
            # See the struct docstring for why comm's P2P kernels are not used.
            # In-place is safe by slice-disjointness: in phase 1 rank r only
            # writes its own buffer's slice [r*shard, (r+1)*shard) while peers
            # only read slice p (their own index) of r's buffer; in phase 2
            # rank r writes its slices p != r while peers read slice r.
            comptime if Self.N >= 2:
                comptime if has_nvidia_gpu_accelerator():
                    var shard = size // Self.N
                    if shard * Self.N != size:
                        raise Error(
                            "ZeroContext.allreduce: size must be divisible by"
                            " N (pass the padded parameter count)"
                        )
                    self._check_scratch[dtype](shard)
                    var coord_ptr = _register_and_sync[
                        dtype, MutAnyOrigin, MutAnyOrigin
                    ](self.rank, self.cpu_coordinator_ptr, ptr, ptr)

                    # Phase 1: reduce every rank's slice[rank] into my
                    # slice[rank].
                    var scr = rebind[
                        UnsafePointer[Scalar[dtype], MutAnyOrigin]
                    ](
                        self.comm_scratch.unsafe_ptr()
                        .bitcast[Scalar[dtype]]()
                        .as_unsafe_any_origin()
                    )
                    comptime add_k = _add_inplace_gpu[dtype]
                    var compiled_add = self.ctx.compile_function[add_k]()
                    var my_slice = ptr + self.rank * shard
                    for step in range(1, Self.N):
                        var p = (self.rank + step) % Self.N
                        var peer_base = rebind[
                            UnsafePointer[Scalar[dtype], MutAnyOrigin]
                        ](
                            coord_ptr[]
                            .shared_inputs[p]
                            .bitcast[Scalar[dtype]]()
                            .as_unsafe_any_origin()
                        )
                        var src_view = DeviceBuffer[dtype](
                            self.peer_ctxs[p],
                            peer_base + self.rank * shard,
                            shard,
                            owning=False,
                        )
                        var scr_view = DeviceBuffer[dtype](
                            self.ctx, scr, shard, owning=False
                        )
                        self.ctx.enqueue_copy(
                            dst_buf=scr_view, src_buf=src_view
                        )
                        self.ctx.enqueue_function(
                            compiled_add,
                            my_slice,
                            rebind[
                                UnsafePointer[Scalar[dtype], ImmutAnyOrigin]
                            ](scr.as_unsafe_any_origin()),
                            shard,
                            grid_dim=(ceildiv(shard, _ADD_BLOCK), 1),
                            block_dim=(_ADD_BLOCK,),
                        )
                    self.ctx.synchronize()
                    coord_ptr[].barrier2[].wait()

                    # Phase 2: gather every peer's reduced slice[p].
                    for step in range(1, Self.N):
                        var p = (self.rank + step) % Self.N
                        var peer_base = rebind[
                            UnsafePointer[Scalar[dtype], MutAnyOrigin]
                        ](
                            coord_ptr[]
                            .shared_inputs[p]
                            .bitcast[Scalar[dtype]]()
                            .as_unsafe_any_origin()
                        )
                        var src_view = DeviceBuffer[dtype](
                            self.peer_ctxs[p],
                            peer_base + p * shard,
                            shard,
                            owning=False,
                        )
                        var dst_view = DeviceBuffer[dtype](
                            self.ctx, ptr + p * shard, shard, owning=False
                        )
                        self.ctx.enqueue_copy(
                            dst_buf=dst_view, src_buf=src_view
                        )
                    self.ctx.synchronize()
                    coord_ptr[].barrier1[].wait()
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
            # Staged-copy reduce-scatter: each rank pulls slice
            # [rank*shard, (rank+1)*shard) of every peer's full input buffer
            # (length N*shard — callers pass the padded parameter count) and
            # accumulates into its own separate shard output buffer. Nobody
            # writes any input buffer, so concurrent peer reads are safe.
            comptime if Self.N >= 2:
                comptime if has_nvidia_gpu_accelerator():
                    var shard = sharded_size
                    self._check_scratch[dtype](shard)
                    var coord_ptr = _register_and_sync[
                        dtype, MutAnyOrigin, MutAnyOrigin
                    ](
                        self.rank,
                        self.cpu_coordinator_ptr,
                        input_ptr,
                        output_ptr,
                    )

                    # Own slice seeds the accumulator (same-device copy).
                    self.ctx.enqueue_copy(
                        dst_ptr=output_ptr,
                        src_ptr=rebind[
                            UnsafePointer[Scalar[dtype], ImmutAnyOrigin]
                        ](
                            (
                                input_ptr + self.rank * shard
                            ).as_unsafe_any_origin()
                        ),
                        size=shard,
                    )
                    var scr = rebind[
                        UnsafePointer[Scalar[dtype], MutAnyOrigin]
                    ](
                        self.comm_scratch.unsafe_ptr()
                        .bitcast[Scalar[dtype]]()
                        .as_unsafe_any_origin()
                    )
                    comptime add_k = _add_inplace_gpu[dtype]
                    var compiled_add = self.ctx.compile_function[add_k]()
                    for step in range(1, Self.N):
                        var p = (self.rank + step) % Self.N
                        var peer_base = rebind[
                            UnsafePointer[Scalar[dtype], MutAnyOrigin]
                        ](
                            coord_ptr[]
                            .shared_inputs[p]
                            .bitcast[Scalar[dtype]]()
                            .as_unsafe_any_origin()
                        )
                        var src_view = DeviceBuffer[dtype](
                            self.peer_ctxs[p],
                            peer_base + self.rank * shard,
                            shard,
                            owning=False,
                        )
                        var scr_view = DeviceBuffer[dtype](
                            self.ctx, scr, shard, owning=False
                        )
                        self.ctx.enqueue_copy(
                            dst_buf=scr_view, src_buf=src_view
                        )
                        self.ctx.enqueue_function(
                            compiled_add,
                            output_ptr,
                            rebind[
                                UnsafePointer[Scalar[dtype], ImmutAnyOrigin]
                            ](scr.as_unsafe_any_origin()),
                            shard,
                            grid_dim=(ceildiv(shard, _ADD_BLOCK), 1),
                            block_dim=(_ADD_BLOCK,),
                        )
                    self.ctx.synchronize()
                    coord_ptr[].barrier2[].wait()
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
            # Staged-copy all-gather: rank r's shard already sits at
            # ptr + r*shard of its own full buffer; pull every peer p's slice
            # [p*shard, (p+1)*shard) from p's buffer into mine. Rank r never
            # writes its own slice r (peers read exactly that slice), so
            # concurrent pulls are safe.
            comptime if Self.N >= 2:
                comptime if has_nvidia_gpu_accelerator():
                    var shard = sharded_size
                    var coord_ptr = _register_and_sync[
                        dtype, MutAnyOrigin, MutAnyOrigin
                    ](self.rank, self.cpu_coordinator_ptr, ptr, ptr)
                    for step in range(1, Self.N):
                        var p = (self.rank + step) % Self.N
                        var peer_base = rebind[
                            UnsafePointer[Scalar[dtype], MutAnyOrigin]
                        ](
                            coord_ptr[]
                            .shared_inputs[p]
                            .bitcast[Scalar[dtype]]()
                            .as_unsafe_any_origin()
                        )
                        var src_view = DeviceBuffer[dtype](
                            self.peer_ctxs[p],
                            peer_base + p * shard,
                            shard,
                            owning=False,
                        )
                        var dst_view = DeviceBuffer[dtype](
                            self.ctx, ptr + p * shard, shard, owning=False
                        )
                        self.ctx.enqueue_copy(
                            dst_buf=dst_view, src_buf=src_view
                        )
                    self.ctx.synchronize()
                    coord_ptr[].barrier2[].wait()
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

    @always_inline
    def load_to_gpu(self, ctx: DeviceContext) raises -> None:
        comptime if Self.offload and not is_cpu[Self.target]():
            self.sharded_buffer.enqueue_copy_from(self.host_sharded_buffer)
            ctx.synchronize()

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
                full_buffer.unsafe_ptr().unsafe_origin_cast[
                    MutUntrackedOrigin
                ](),
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
