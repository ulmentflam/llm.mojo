# ===----------------------------------------------------------------------=== #
# Regression test for CPU async-write / host-read ordering in GPT2.
#
# Run with:  make test-mojo
#
# WHY THIS FILE EXISTS. On the CPU target, `ctx.enqueue_copy` / `enqueue_fill`
# go onto the DeviceContext's queue and run later on an AsyncRT worker. The
# llmm CPU kernels do not go through that queue. They run synchronously on the
# host, under `traced_parallelize`. So every async write that a host kernel
# reads needs an explicit `ctx.synchronize()` in between. When one was missing
# (the dL/dloss seed in GPT2.forward, fixed in 25fff32), the classifier
# sometimes read the seed buffer before the copy had landed. Then every
# parameter gradient came out wrong while the loss stayed correct. That is
# where the intermittent `make verify-cpu` failures came from: a few
# gradient tensors NOT OK with our_l2/ref_l2 well below 1, and the loss
# wandering off from step 1 (docs/ai/verify_cpu_wrong_values_flake.md).
#
# On an idle box the copy nearly always wins the race, so a plain
# forward/backward cannot catch a missing barrier. 25fff32 shipped without a
# test for exactly that reason. This file makes the race deterministic
# instead. `GPT2.debug_stall_cpu_queue` puts a busy-wait in front of each
# such async write (llmm.memory.stall_cpu_queue). The queue runs in order, so
# the write cannot land before the stall ends. A missing barrier then reads
# the stale buffer every time, and a present barrier just waits.
#
# test_stall_holds_back_queued_copy checks that premise. If the stall stops
# delaying later queue work (an API change, or a no-op stall), that test
# fails, rather than the gradient test passing for no reason.
# ===----------------------------------------------------------------------=== #

from std.testing import assert_true, TestSuite
from std.python import Python

from max.gpu.host import DeviceContext

from llmm.memory import MutMemPtr, heap_alloc, stall_cpu_queue
from train_gpt2 import GPT2, GPT2_DTYPE, UNTIED_HEAD

comptime CKPT = "gpt2_tiny_queue_ordering.bin"
comptime NUM_PARAMS = 4592  # tiny config below: L=1, C=16, V=V_p=64, maxT=16
# The tied .bin written here is 16 tensors; an LLMM_UNTIED_HEAD build loads it
# by copying wte into a 17th head block (V_p * C = 64 * 16 more elements).
comptime MODEL_PARAMS = NUM_PARAMS + (64 * 16 if UNTIED_HEAD else 0)
comptime B = 2
comptime T = 8


def _write_checkpoint() raises:
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    var header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 3  # version
    header[2] = 16  # max_seq_len
    header[3] = 64  # vocab_size
    header[4] = 1  # num_layer
    header[5] = 1  # num_heads
    header[6] = 16  # channels
    header[7] = 64  # padded_vocab_size
    np.random.seed(7)
    var weights = np.random.normal(0, 0.02, NUM_PARAMS).astype(np.float32)
    var f = builtins.open(CKPT, "wb")
    _ = f.write(header.tobytes())
    _ = f.write(weights.tobytes())
    f.close()


def _remove_checkpoint() raises:
    var os = Python.import_module("os")
    try:
        _ = os.remove(CKPT)
    except:
        pass


def _fill_batch(x: MutMemPtr[DType.int32], y: MutMemPtr[DType.int32]):
    for i in range(B * T):
        x[unsafe_offset=i] = Int32((i * 13 + 5) % 64)
        y[unsafe_offset=i] = Int32((i * 29 + 11) % 64)


def _run_two_steps(
    ctx: DeviceContext,
    stall: Bool,
    out_grads: MutMemPtr[DType.float32],
) raises:
    """Run two forward/zero_gradients/backward steps on the same batch, with no
    update in between. Writes both steps' gradients to `out_grads`:
    [0, NUM_PARAMS) holds step 0 and [NUM_PARAMS, 2*NUM_PARAMS) holds step 1.

    Each step stalls the queue in front of the dL/dloss seed copy in forward.
    That is the barrier this test is known to guard. With it removed, the
    stalled run fails on 3107 of 9184 elements (dlogits all zero). Each step
    also stalls in front of zero_gradients' async fill. Removing that
    barrier alone does NOT fail here (measured). The backward's CPU GEMMs
    run through linalg.matmul(ctx=ctx), which enqueues on the same in-order
    queue and synchronizes it, and that orders the fill too. That stall covers the case where
    the backward's leading kernel stops synchronizing. It is not proof that
    zero_gradients' own barrier is needed.
    """
    var x = heap_alloc[Scalar[DType.int32]](B * T)
    var y = heap_alloc[Scalar[DType.int32]](B * T)
    _fill_batch(x, y)
    var model = GPT2["cpu", 1](CKPT, rank=0, zero_stage=0, ctx=ctx)
    assert_true(model.num_parameters == MODEL_PARAMS, "unexpected param count")
    model.debug_stall_cpu_queue = stall
    var host = ctx.enqueue_create_host_buffer[GPT2_DTYPE](MODEL_PARAMS)
    for step in range(2):
        model.forward(x, y, B, T)
        model.zero_gradients()
        model.backward()
        model.ctx.synchronize()
        model.grads_buf.enqueue_copy_to(host)
        ctx.synchronize()
        for i in range(NUM_PARAMS):
            out_grads[unsafe_offset=step * NUM_PARAMS + i] = host[i].cast[
                DType.float32
            ]()
    _ = host^
    x.unsafe_free()
    y.unsafe_free()


def test_stall_holds_back_queued_copy() raises:
    # Premise of the gradient test: a write queued behind stall_cpu_queue has
    # not landed when control returns to the host, and lands by synchronize().
    var ctx = DeviceContext(api="cpu")
    var src = ctx.enqueue_create_host_buffer[DType.float32](16)
    var dst = ctx.enqueue_create_buffer[DType.float32](16)
    for i in range(16):
        src[i] = 1.0
    dst.enqueue_fill(0.0)
    ctx.synchronize()
    stall_cpu_queue(ctx)
    ctx.enqueue_copy(dst_buf=dst, src_buf=src)
    var dst_ptr = dst.unsafe_ptr()
    var landed_early = 0
    for i in range(16):
        if dst_ptr[unsafe_offset=i] != 0.0:
            landed_early += 1
    ctx.synchronize()
    var landed_late = 0
    for i in range(16):
        if dst_ptr[unsafe_offset=i] == 1.0:
            landed_late += 1
    assert_true(
        landed_early == 0,
        "stall_cpu_queue did not hold back the queued copy ("
        + String(landed_early)
        + "/16 landed early); the gradient test below would be vacuous",
    )
    assert_true(landed_late == 16, "queued copy never landed after the stall")


def test_stalled_cpu_queue_does_not_change_gradients() raises:
    # Every async write that a host kernel reads must be barriered. If it
    # is, stalling the queue in front of the write changes nothing. If one
    # barrier is missing, the stalled run reads the stale buffer (for the
    # dL/dloss seed: all-zero dlogits, so every gradient is zero), and the
    # comparison fails on thousands of elements.
    var ctx = DeviceContext(api="cpu")
    var ref_grads = heap_alloc[Scalar[DType.float32]](2 * NUM_PARAMS)
    var got_grads = heap_alloc[Scalar[DType.float32]](2 * NUM_PARAMS)
    _run_two_steps(ctx, False, ref_grads)
    _run_two_steps(ctx, True, got_grads)

    var ref_nonzero = 0
    var mismatches = 0
    var first_bad = -1
    for i in range(2 * NUM_PARAMS):
        var r = ref_grads[unsafe_offset=i]
        var g = got_grads[unsafe_offset=i]
        if r != 0.0:
            ref_nonzero += 1
        # The CPU backward is deterministic run to run, so the two runs should
        # agree exactly. Allow fp32 reassociation noise anyway. Both failure
        # modes are off by O(|r|), far outside this band.
        if abs(g - r) > 1e-6 + 1e-4 * abs(r):
            mismatches += 1
            if first_bad < 0:
                first_bad = i
    # The reference must have real gradients, or a stale read could not be
    # told apart from a correct one.
    assert_true(
        ref_nonzero > NUM_PARAMS,
        "reference gradients are mostly zero; test cannot discriminate",
    )
    var msg = String("")
    if first_bad >= 0:
        msg = (
            String(mismatches)
            + " of "
            + String(2 * NUM_PARAMS)
            + " gradient elements changed when the CPU queue was stalled"
            " (first at "
            + String(first_bad)
            + ": ref="
            + String(ref_grads[unsafe_offset=first_bad])
            + " got="
            + String(got_grads[unsafe_offset=first_bad])
            + "). An async write in GPT2 is read by a host kernel without a"
            " ctx.synchronize() in between."
        )
    ref_grads.unsafe_free()
    got_grads.unsafe_free()
    assert_true(mismatches == 0, msg)


def main() raises:
    _write_checkpoint()
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        _remove_checkpoint()
        raise e^
    _remove_checkpoint()
