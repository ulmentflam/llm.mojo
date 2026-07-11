from std.testing import assert_almost_equal, assert_true, TestSuite
from std.sys.info import size_of
from std.memory import alloc, UnsafePointer, memcpy
from std.gpu.host import DeviceContext
from std.algorithm import sync_parallelize
from std.python import Python

from llmm.memory import MutMemPtr
from llmm.zero import ZeroContext, CpuCoordinator

from train_gpt2 import GPT2, GPT2_DTYPE

comptime NUM_PARAMS = 4592
comptime B = 2
comptime T = 8
comptime WORLD_SIZE = 2


def read_to_dtype_pointer[
    T: DType
](
    ptr: UnsafePointer[Scalar[T], _], mut file_handle: FileHandle, size: Int
) raises -> None:
    var bytes_to_read = size * size_of[T]()
    var bytes_data = file_handle.read_bytes(bytes_to_read)
    var d = rebind[UnsafePointer[UInt8, MutUntrackedOrigin]](ptr)
    var s = rebind[UnsafePointer[UInt8, MutUntrackedOrigin]](
        bytes_data.unsafe_ptr()
    )
    memcpy(dest=d, src=s, count=bytes_to_read)


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


def run_zero_equivalence_test(stage: Int) raises:
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

    # 3. Test ZeRO Stage 1 and Stage 2 with WORLD_SIZE = 2 (Sequential simulation)
    var model0 = GPT2["cpu", WORLD_SIZE](
        "gpt2_tiny.bin",
        rank=0,
        zero_stage=stage,
        ctx=ctx,
        cpu_coordinator_ptr=None,
    )
    var model1 = GPT2["cpu", WORLD_SIZE](
        "gpt2_tiny.bin",
        rank=1,
        zero_stage=stage,
        ctx=ctx,
        cpu_coordinator_ptr=None,
    )

    model0.forward(
        rebind[MutMemPtr[DType.int32]](x),
        rebind[MutMemPtr[DType.int32]](y),
        B,
        T,
    )
    model0.ctx.synchronize()

    model1.forward(
        rebind[MutMemPtr[DType.int32]](x),
        rebind[MutMemPtr[DType.int32]](y),
        B,
        T,
    )
    model1.ctx.synchronize()

    model0.zero_gradients()
    model0.backward()
    model0.ctx.synchronize()

    model1.zero_gradients()
    model1.backward()
    model1.ctx.synchronize()

    # 4. Manual Allreduce / Reducescatter of gradients
    for i in range(NUM_PARAMS):
        var sum_grad = model0.grads_memory[i] + model1.grads_memory[i]
        model0.grads_memory[i] = sum_grad
        model1.grads_memory[i] = sum_grad

    # ZeRO-1: grads are allreduced and replicated in grads_memory; update() reads
    #         grads_memory + rank*opt directly — no sharded_grads_memory needed.
    # ZeRO-2/3: fill sharded_grads_memory with the reduce-scattered shard so
    #           update() reads from sharded_grads_memory.
    if stage >= 2:
        var shard_size = model0.optimizer_num_parameters
        var local_len0 = min(NUM_PARAMS, shard_size)
        for i in range(local_len0):
            model0.sharded_grads_memory[i] = model0.grads_memory[i]

        var offset1 = shard_size
        var local_len1 = min(NUM_PARAMS - offset1, shard_size)
        for i in range(local_len1):
            model1.sharded_grads_memory[i] = model1.grads_memory[offset1 + i]

    model0.update(
        t=1,
        learning_rate=Scalar[DType.float32](1e-4),
        beta1=Scalar[DType.float32](0.9),
        beta2=Scalar[DType.float32](0.999),
        eps=Scalar[DType.float32](1e-8),
        weight_decay=Scalar[DType.float32](0.01),
    )
    model0.ctx.synchronize()

    model1.update(
        t=1,
        learning_rate=Scalar[DType.float32](1e-4),
        beta1=Scalar[DType.float32](0.9),
        beta2=Scalar[DType.float32](0.999),
        eps=Scalar[DType.float32](1e-8),
        weight_decay=Scalar[DType.float32](0.01),
    )
    model1.ctx.synchronize()

    # 4b. Stage-3 specific: verify per-rank param shards BEFORE any allgather.
    # ZeRO-3 skips the post-update allgather (stage 1/2 call it inside update());
    # instead the next forward() gathers shards on demand. Assert that each rank's
    # persistent owned shard already matches the corresponding baseline slice.
    if stage >= 3:
        try:
            var ss3 = model0.optimizer_num_parameters
            var len3_0 = min(NUM_PARAMS, ss3)
            for i in range(len3_0):
                assert_almost_equal(
                    model0.params_memory[i].cast[DType.float32](),
                    baseline_params[i],
                    atol=1e-5,
                )
            var off3_1 = ss3
            var len3_1 = min(NUM_PARAMS - off3_1, ss3)
            for i in range(len3_1):
                assert_almost_equal(
                    model1.params_memory[off3_1 + i].cast[DType.float32](),
                    baseline_params[off3_1 + i],
                    atol=1e-5,
                )
        except e:
            x.free()
            y.free()
            baseline_params.free()
            raise Error(
                "Stage-3 per-shard equivalence check failed: " + String(e)
            )

    # 5. Manual Allgather of parameters
    var shard_size = model0.optimizer_num_parameters
    var local_len0 = min(NUM_PARAMS, shard_size)
    for i in range(local_len0):
        model1.params_memory[i] = model0.params_memory[i]

    var offset1 = shard_size
    var local_len1 = min(NUM_PARAMS - offset1, shard_size)
    for i in range(local_len1):
        model0.params_memory[offset1 + i] = model1.params_memory[offset1 + i]

    # 6. Verify equivalence
    try:
        for i in range(NUM_PARAMS):
            assert_almost_equal(
                model0.params_memory[i].cast[DType.float32](),
                baseline_params[i],
                atol=1e-5,
            )
            assert_almost_equal(
                model1.params_memory[i].cast[DType.float32](),
                baseline_params[i],
                atol=1e-5,
            )
    except e:
        x.free()
        y.free()
        baseline_params.free()
        raise Error("Equivalence check failed: " + String(e))

    x.free()
    y.free()
    baseline_params.free()


def test_zero_stage1_equivalence() raises:
    run_zero_equivalence_test(1)


def test_zero_stage2_equivalence() raises:
    run_zero_equivalence_test(2)


def test_zero_stage3_equivalence() raises:
    # ZeRO-3 uses reduce-scatter for gradients (same as stage 2) and sharded
    # AdamW (same as stage 2), but defers the post-update param allgather to
    # the next forward() call rather than doing it immediately after update().
    # run_zero_equivalence_test(3) verifies:
    #   - Grad reduce-scatter shards feed the optimizer correctly (implicitly,
    #     via the final param equivalence).
    #   - Each rank's per-shard params match the baseline BEFORE any allgather
    #     (the stage-3 specific check inserted at step 4b above).
    #   - Simulating the forward()-triggered allgather reconstructs the full
    #     parameter vector identically to the stage-2 / baseline result.
    run_zero_equivalence_test(3)


def main() raises:
    setup_test_files()
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        cleanup_test_files()
        raise e^
    cleanup_test_files()
