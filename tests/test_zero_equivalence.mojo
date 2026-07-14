from std.testing import assert_almost_equal, assert_true, TestSuite
from std.sys.info import size_of
from std.memory import alloc, UnsafePointer
from std.gpu.host import DeviceContext
from std.algorithm import sync_parallelize
from std.python import Python

from llmm.memory import MutMemPtr
from llmm.zero import ZeroContext, CpuCoordinator

from train_gpt2 import GPT2, GPT2_DTYPE

comptime NUM_PARAMS = 4592
comptime B = 2
comptime T = 8


def read_to_dtype_pointer[
    T: DType
](
    ptr: UnsafePointer[Scalar[T], _], mut file_handle: FileHandle, size: Int
) raises -> None:
    # Element-wise copy, mirroring llmm.io.read_and_copy. A byte-wise memcpy from
    # `bytes_data.unsafe_ptr()` is unsafe here: rebinding that pointer to an
    # untracked origin drops the List's lifetime tracking, so `bytes_data` is
    # freed at its last tracked use (the `.unsafe_ptr()` call) *before* the copy
    # runs — the allocator then clobbers the first element (magic read back as
    # garbage). Looping over the List keeps it live across the whole copy.
    var bytes_to_read = size * size_of[T]()
    var bytes_data = file_handle.read_bytes(bytes_to_read)
    if len(bytes_data) < bytes_to_read:
        raise Error("Failed to read enough bytes from file")
    var dest = rebind[UnsafePointer[Scalar[T], MutUntrackedOrigin]](ptr)
    var src_ptr = bytes_data.unsafe_ptr().bitcast[Scalar[T]]()
    for i in range(size):
        dest[i] = src_ptr[i]
    # Keep `bytes_data` live until the copy is done: `src_ptr` does not own it,
    # so without this the List can be freed mid-loop (see comment above).
    _ = bytes_data^


def setup_test_files() raises:
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    # Micro model config for fast CPU test
    var max_seq_len = 16
    var vocab_size = 64
    var num_layer = 1
    var num_heads = 1
    var channels = 16
    var padded_vocab_size = 64

    var header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 3  # version
    header[2] = max_seq_len
    header[3] = vocab_size
    header[4] = num_layer
    header[5] = num_heads
    header[6] = channels
    header[7] = padded_vocab_size

    np.random.seed(42)
    var weights = np.random.normal(0, 0.02, NUM_PARAMS).astype(np.float32)

    var f = builtins.open("gpt2_tiny.bin", "wb")
    _ = f.write(header.tobytes())
    _ = f.write(weights.tobytes())
    f.close()

    var state_header = np.zeros(256, dtype=np.int32)
    state_header[0] = 20240520
    state_header[1] = 3
    state_header[2] = B
    state_header[3] = T

    var x = np.random.randint(0, vocab_size, B * T).astype(np.int32)
    var y = np.random.randint(0, vocab_size, B * T).astype(np.int32)

    var f_state = builtins.open("gpt2_tiny_debug_state.bin", "wb")
    _ = f_state.write(state_header.tobytes())
    _ = f_state.write(x.tobytes())
    _ = f_state.write(y.tobytes())
    f_state.close()


def cleanup_test_files() raises:
    var os = Python.import_module("os")
    try:
        _ = os.remove("gpt2_tiny.bin")
    except:
        pass
    try:
        _ = os.remove("gpt2_tiny_debug_state.bin")
    except:
        pass


def run_zero_equivalence_test[N: Int](stage: Int) raises:
    """ZeRO optimizer-sharding equivalence at world size `N`.

    Simulates `N` ranks sequentially on CPU (all ranks see the same batch —
    this checks the sharded-optimizer math, not data parallelism) and asserts
    the reduce-scatter -> sharded AdamW -> all-gather round trip reconstructs
    the exact parameters a single-GPU (WORLD_SIZE=1) baseline produces.
    """
    # 1. Load inputs and targets from the debug state file
    var state_file = open("gpt2_tiny_debug_state.bin", "r")
    var state_header = alloc[Int32](256)
    read_to_dtype_pointer[DType.int32](state_header, state_file, 256)
    if state_header[0] != 20240520:
        state_file.close()
        state_header.free()
        raise Error("Bad magic model file")

    var file_B: Int = Int(state_header[2])
    var file_T: Int = Int(state_header[3])
    if file_B != B or file_T != T:
        state_file.close()
        state_header.free()
        raise Error("Dimension mismatch in state file")

    var x = alloc[SIMD[DType.int32, 1]](B * T)
    var y = alloc[SIMD[DType.int32, 1]](B * T)
    read_to_dtype_pointer[DType.int32](x, state_file, B * T)
    read_to_dtype_pointer[DType.int32](y, state_file, B * T)
    state_file.close()
    state_header.free()

    # 2. Run baseline (zero_stage = 0, WORLD_SIZE = 1)
    var ctx = DeviceContext(api="cpu")
    var baseline_params = alloc[Float32](NUM_PARAMS)

    try:
        var baseline_model = GPT2["cpu", 1](
            "gpt2_tiny.bin",
            rank=0,
            zero_stage=0,
            ctx=ctx,
        )

        baseline_model.forward(
            rebind[MutMemPtr[DType.int32]](x),
            rebind[MutMemPtr[DType.int32]](y),
            B,
            T,
        )
        baseline_model.ctx.synchronize()
        baseline_model.zero_gradients()
        baseline_model.backward()
        baseline_model.ctx.synchronize()
        baseline_model.update(
            t=1,
            learning_rate=Scalar[DType.float32](1e-4),
            beta1=Scalar[DType.float32](0.9),
            beta2=Scalar[DType.float32](0.999),
            eps=Scalar[DType.float32](1e-8),
            weight_decay=Scalar[DType.float32](0.01),
        )
        baseline_model.ctx.synchronize()

        # Match params_buf's dtype (GPT2_DTYPE folds per -D LLMM_BF16); the
        # read loop below casts each element to fp32.
        var host_out_baseline = ctx.enqueue_create_host_buffer[GPT2_DTYPE](
            NUM_PARAMS
        )
        baseline_model.params_buf.enqueue_copy_to(host_out_baseline)
        ctx.synchronize()
        for i in range(NUM_PARAMS):
            baseline_params[i] = host_out_baseline.unsafe_ptr()[i].cast[
                DType.float32
            ]()
    except e:
        x.free()
        y.free()
        baseline_params.free()
        raise Error("Baseline execution failed")

    # 3. Simulate the N ranks one at a time. GPT2 is not Movable (so it can't
    # live in a List); instead we rely on every rank being identical here — same
    # gpt2_tiny.bin init, same batch, so every rank computes the same gradient
    # `g`. The all-reduce (sum over N ranks) is therefore exactly `N * g`, and
    # each rank's owned shard [rank*shard, ...) is optimized independently of the
    # others. So we can build one rank's model, drive it, assert its shard
    # matches the baseline slice, discard it, and repeat. The full parameter
    # vector is the concatenation of the N shards, so per-shard equivalence to
    # the baseline is exactly whole-vector equivalence after an all-gather.
    for r in range(N):
        var model = GPT2["cpu", N](
            "gpt2_tiny.bin",
            rank=r,
            zero_stage=stage,
            ctx=ctx,
            cpu_coordinator_ptr=None,
        )
        model.forward(
            rebind[MutMemPtr[DType.int32]](x),
            rebind[MutMemPtr[DType.int32]](y),
            B,
            T,
        )
        model.ctx.synchronize()
        model.zero_gradients()
        model.backward()
        model.ctx.synchronize()

        # `optimizer_num_parameters` is padded up to the AdamW SIMD width, so
        # N * shard >= NUM_PARAMS; the last rank(s) may run into zero padding.
        var shard = model.optimizer_num_parameters
        var off = r * shard
        var ln = min(NUM_PARAMS - off, shard) if off < NUM_PARAMS else 0

        # All-reduce of gradients over N identical ranks == N * g. update()
        # divides grad_scale by WORLD_SIZE, so the effective gradient is g again.
        for i in range(NUM_PARAMS):
            model.grads_memory[i] = model.grads_memory[i] * Scalar[GPT2_DTYPE](
                N
            )

        # Every sharded stage now reads its gradient shard from
        # grads_memory + rank*shard: ZeRO-1 all-reduces (grads replicated),
        # ZeRO-2/3 reduce-scatter IN PLACE (only slice [off, off+ln) reduced).
        # The `* N` above already put the reduced sum there, so no separate
        # shard copy is needed for any stage.

        model.update(
            t=1,
            learning_rate=Scalar[DType.float32](1e-4),
            beta1=Scalar[DType.float32](0.9),
            beta2=Scalar[DType.float32](0.999),
            eps=Scalar[DType.float32](1e-8),
            weight_decay=Scalar[DType.float32](0.01),
        )
        model.ctx.synchronize()

        # Verify this rank's owned shard matches the baseline slice. For ZeRO-3
        # this is the pre-allgather persistent shard; for ZeRO-1/2 update() has
        # already written params_memory[off:off+ln] in place.
        try:
            for i in range(ln):
                assert_almost_equal(
                    model.params_memory[off + i].cast[DType.float32](),
                    baseline_params[off + i],
                    atol=1e-5,
                )
        except e:
            x.free()
            y.free()
            baseline_params.free()
            raise Error(
                "Equivalence check failed at rank "
                + String(r)
                + ": "
                + String(e)
            )

    x.free()
    y.free()
    baseline_params.free()


def test_zero_stage1_equivalence_w2() raises:
    run_zero_equivalence_test[2](1)


def test_zero_stage2_equivalence_w2() raises:
    run_zero_equivalence_test[2](2)


def test_zero_stage3_equivalence_w2() raises:
    # ZeRO-3 uses reduce-scatter for gradients (same as stage 2) and sharded
    # AdamW (same as stage 2), but defers the post-update param allgather to the
    # next forward() call rather than doing it immediately after update(). The
    # test verifies the reduce-scatter shards feed the optimizer correctly, the
    # per-shard params match the baseline BEFORE any allgather (step 4b), and the
    # simulated allgather reconstructs the full vector identically to baseline.
    run_zero_equivalence_test[2](3)


def test_zero_stage1_equivalence_w8() raises:
    run_zero_equivalence_test[8](1)


def test_zero_stage2_equivalence_w8() raises:
    run_zero_equivalence_test[8](2)


def test_zero_stage3_equivalence_w8() raises:
    # Same round trip as the W2 stage-3 test at the mission's target world size
    # (8). Exercises the shard-length padding (NUM_PARAMS=4592 is not divisible
    # by 8) and the last-rank shard that runs into the zero padding.
    run_zero_equivalence_test[8](3)


def main() raises:
    setup_test_files()
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        cleanup_test_files()
        raise e^
    cleanup_test_files()
