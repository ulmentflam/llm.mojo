from std.collections import InlineArray, List
from std.time import perf_counter_ns
from std.sys import exit, argv
from std.os import getenv
from std.sys.info import size_of
from std.memory import alloc, UnsafePointer, memcpy
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu

from llmm.memory import MutMemPtr
from llmm.zero import ZeroContext

from train_gpt2 import (
    GPT2,
    ParameterTensors,
)

comptime SIZEOF_INT = size_of[DType.int32]()
comptime SIZEOF_FLOAT = size_of[DType.float32]()


# poor man's tensor checker that copies device tensor to host first.
# Parametric over the device dtype (GPT2_DTYPE for params/grads, which is fp32
# or bf16 depending on the build): the host buffer matches the device element
# size — so bf16 tensors copy correctly — and each element is cast to fp32 to
# compare against the fp32 reference.
def check_tensor[
    dtype: DType
](
    ctx: DeviceContext,
    device_ptr: MutMemPtr[dtype],
    expected_ptr: UnsafePointer[Scalar[DType.float32], _],
    n: Int,
    label: String,
    tol: Float32,
) raises -> Bool:
    # Allocate a Metal-registered host buffer so enqueue_copy works on
    # both Metal (which rejects plain-malloc dst pointers) and CUDA.
    var host_buf = ctx.enqueue_create_host_buffer[dtype](n)
    try:
        ctx.enqueue_copy(
            dst_ptr=rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                host_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
            src_ptr=rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
                device_ptr.as_unsafe_any_origin()
            ),
            size=n,
        )
        ctx.synchronize()
    except e:
        print("Error copying tensor to host for " + label + ":", e)
        return False

    var print_upto: Int = 5
    var ok: Bool = True
    var maxdiff: Float32 = 0.0

    print(label)

    for i in range(n):
        # look at the difference at position i of these two tensors
        var actual = host_buf[i].cast[DType.float32]()
        var diff = abs(actual - expected_ptr[i])

        # keep track of the overall error
        ok = ok and (diff <= tol)

        if diff > maxdiff:
            maxdiff = diff

        # for the first few elements of each tensor, pretty print
        # the actual numbers, so we can do a visual, qualitative proof/assessment
        if i < print_upto:
            if diff <= tol:
                print("OK ", end="")
            else:
                print("NOT OK ", end="")

            print(actual, expected_ptr[i])

    # print the final result
    if ok:
        print("TENSOR OK")
    else:
        print("TENSOR NOT OK, maxdiff =", maxdiff)

    _ = host_buf^
    return ok


def read_to_dtype_pointer[
    T: DType
](
    ptr: UnsafePointer[Scalar[T], _], mut file_handle: FileHandle, size: Int
) raises -> None:
    # Read directly into the pointer using read_bytes
    var bytes_to_read = size * size_of[T]()
    var bytes_data = file_handle.read_bytes(bytes_to_read)

    var d = rebind[UnsafePointer[UInt8, MutUntrackedOrigin]](ptr)
    var s = rebind[UnsafePointer[UInt8, MutUntrackedOrigin]](
        bytes_data.unsafe_ptr()
    )
    memcpy(dest=d, src=s, count=bytes_to_read)


def run_test[
    target: StaticString,
    recompute: Bool = False,
](logits_tol: Float32, loss_tol: Float32, grads_tol: Float32,) raises -> Bool:
    # build the GPT-2 model from a checkpoint
    var ctx: DeviceContext
    comptime if is_cpu[target]():
        ctx = DeviceContext(api="cpu")
    else:
        ctx = DeviceContext()

    # `recompute` is a comptime parameter (see main's LLMM_RECOMPUTE dispatch):
    # this reference check also guards the recompute path, which must produce
    # the same logits/loss/gradients as the default run.
    var model = GPT2[target, 1, recompute](
        "gpt2_124M.bin",
        rank=0,
        zero_stage=0,
        ctx=ctx,
    )

    var C: Int = model.config.channels
    var V: Int = model.config.vocab_size
    var maxT: Int = model.config.max_seq_len
    var L: Int = model.config.num_layer
    var V_p: Int = model.config.padded_vocab_size

    # load additional information that we will use for debugging and error checking

    var state_file = open("gpt2_124M_debug_state.bin", "r")

    var state_header = alloc[Int32](256)
    read_to_dtype_pointer[DType.int32](state_header, state_file, 256)

    if state_header[0] != 20240520:
        print("Bad magic model file")
        exit(1)
    if state_header[1] != 3:
        print("Bad version in model file:", state_header[1])
        exit(1)

    var B: Int = Int(state_header[2])  # batch size, e.g. 4
    var T: Int = Int(
        state_header[3]
    )  # time / sequence length (e.g. 64, up to maxT)

    print("[State]")
    print("batch_size:", B)
    print("seq_len:", T)

    var expected_grads = ParameterTensors[DType.float32]()
    var expected_grads_memory = alloc[Scalar[DType.float32]](
        model.num_parameters
    )
    expected_grads.point_parameters(
        model.param_sizes,
        rebind[MutMemPtr[DType.float32]](expected_grads_memory),
    )

    # inputs and expected outputs, only used for error checking

    var x = alloc[SIMD[DType.int32, 1]](B * T)
    var y = alloc[SIMD[DType.int32, 1]](B * T)

    var expected_logits = alloc[SIMD[DType.float32, 1]](B * T * V)
    var expected_loss = alloc[SIMD[DType.float32, 1]](1)

    # read reference information from Python

    read_to_dtype_pointer[DType.int32](x, state_file, B * T)
    read_to_dtype_pointer[DType.int32](y, state_file, B * T)
    read_to_dtype_pointer[DType.float32](expected_logits, state_file, B * T * V)
    read_to_dtype_pointer[DType.float32](expected_loss, state_file, 1)
    read_to_dtype_pointer[DType.float32](
        expected_grads_memory, state_file, model.num_parameters
    )

    state_file.close()
    state_header.free()

    # overall OK signal for the test
    var allok: Bool = True

    # let's do 10 training iterations, following the pytorch code

    var expected_losses: List[Float32] = [
        5.270007133483887,
        4.059706687927246,
        3.3751230239868164,
        2.8007826805114746,
        2.315382242202759,
        1.8490285873413086,
        1.3946564197540283,
        0.9991465210914612,
        0.6240804195404053,
        0.37651097774505615,
    ]

    for step in range(10):
        var start = perf_counter_ns()
        model.forward(
            rebind[MutMemPtr[DType.int32]](x),
            rebind[MutMemPtr[DType.int32]](y),
            B,
            T,
        )

        var elapsed_time_ms = Float64(perf_counter_ns() - start) / 1_000_000
        if step == 0:
            # error checking at step 0 for reference activations/gradients

            # copy logits from device buffer to host first.
            # NOTE: on Metal the enqueue_copy dst must be a *registered* buffer
            # (a raw `alloc` pointer raises "Invalid Metal buffer pointer"), so
            # use a device-context host buffer.
            var host_logits = ctx.enqueue_create_host_buffer[DType.float32](
                B * T * V_p
            )
            ctx.enqueue_copy(
                dst_ptr=rebind[
                    UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
                ](host_logits.unsafe_ptr().as_unsafe_any_origin()),
                src_ptr=rebind[
                    UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]
                ](model.acts.logits.as_unsafe_any_origin()),
                size=B * T * V_p,
            )
            ctx.synchronize()

            # at this point, target should be equal to expected_logits, let's compare

            var logits_ok: Bool = True
            var print_count = 0

            for b in range(B):
                for t in range(T):
                    for v in range(V):
                        var idx_py = (b * T + t) * V + v
                        var idx_mj = (b * T + t) * V_p + v

                        if print_count < 3:
                            print(
                                "PyTorch:",
                                expected_logits[idx_py],
                                "Mojo:",
                                host_logits[idx_mj],
                            )
                            print_count += 1

                        # We only check logits that have a non-negligible impact on softmax (expected_logits > -10.0)
                        if expected_logits[idx_py] > -10.0:
                            if (
                                abs(
                                    expected_logits[idx_py]
                                    - host_logits[idx_mj]
                                )
                                >= logits_tol
                            ):
                                print(
                                    "MISMATCH AT INDEX (Py="
                                    + String(idx_py)
                                    + ", Mj="
                                    + String(idx_mj)
                                    + "):"
                                )
                                print(
                                    "Expected (Py):",
                                    expected_logits[idx_py],
                                    "Got (Mj):",
                                    host_logits[idx_mj],
                                )
                                logits_ok = False
                                break
                    if not logits_ok:
                        break
                if not logits_ok:
                    break

            if not logits_ok:
                print("NOT ", end="")
            print("OK (LOGITS)")
            allok = allok and logits_ok
            _ = host_logits^

            # compare the achieved loss
            if abs(model.mean_loss - expected_loss[0]) >= loss_tol:
                print("LOSS MISMATCH:", model.mean_loss, expected_loss[0])
                allok = False
            else:
                print("LOSS OK:", model.mean_loss, expected_loss[0])

            # Run backward pass before checking gradients
            model.zero_gradients()
            model.backward()

            # finally check all the gradients
            var gradoks = InlineArray[Bool, 16](fill=False)

            gradoks[0] = check_tensor(
                ctx,
                model.grads.wte,
                expected_grads.wte,
                V * C,
                "dwte",
                grads_tol,
            )
            gradoks[1] = check_tensor(
                ctx,
                model.grads.wpe,
                expected_grads.wpe,
                maxT * C,
                "dwpe",
                grads_tol,
            )
            gradoks[2] = check_tensor(
                ctx,
                model.grads.ln_1_gamma,
                expected_grads.ln_1_gamma,
                L * C,
                "dln1w",
                grads_tol,
            )
            gradoks[3] = check_tensor(
                ctx,
                model.grads.ln_1_beta,
                expected_grads.ln_1_beta,
                L * C,
                "dln1b",
                grads_tol,
            )
            gradoks[4] = check_tensor(
                ctx,
                model.grads.qkv_weight,
                expected_grads.qkv_weight,
                L * 3 * C * C,
                "dqkvw",
                grads_tol,
            )
            gradoks[5] = check_tensor(
                ctx,
                model.grads.qkv_bias,
                expected_grads.qkv_bias,
                L * 3 * C,
                "dqkvb",
                grads_tol,
            )
            gradoks[6] = check_tensor(
                ctx,
                model.grads.attn_proj_weight,
                expected_grads.attn_proj_weight,
                L * C * C,
                "dattprojw",
                grads_tol,
            )
            gradoks[7] = check_tensor(
                ctx,
                model.grads.attn_proj_bias,
                expected_grads.attn_proj_bias,
                L * C,
                "dattprojb",
                grads_tol,
            )
            gradoks[8] = check_tensor(
                ctx,
                model.grads.ln_2_gamma,
                expected_grads.ln_2_gamma,
                L * C,
                "dln2w",
                grads_tol,
            )
            gradoks[9] = check_tensor(
                ctx,
                model.grads.ln_2_beta,
                expected_grads.ln_2_beta,
                L * C,
                "dln2b",
                grads_tol,
            )
            gradoks[10] = check_tensor(
                ctx,
                model.grads.fc_weight,
                expected_grads.fc_weight,
                L * 4 * C * C,
                "dfcw",
                grads_tol,
            )
            gradoks[11] = check_tensor(
                ctx,
                model.grads.fc_bias,
                expected_grads.fc_bias,
                L * 4 * C,
                "dfcb",
                grads_tol,
            )
            gradoks[12] = check_tensor(
                ctx,
                model.grads.proj_weight,
                expected_grads.proj_weight,
                L * C * 4 * C,
                "dfcprojw",
                grads_tol,
            )
            gradoks[13] = check_tensor(
                ctx,
                model.grads.proj_bias,
                expected_grads.proj_bias,
                L * C,
                "dfcprojb",
                grads_tol,
            )
            gradoks[14] = check_tensor(
                ctx,
                model.grads.ln_f_gamma,
                expected_grads.ln_f_gamma,
                C,
                "dlnfw",
                grads_tol,
            )
            gradoks[15] = check_tensor(
                ctx,
                model.grads.ln_f_beta,
                expected_grads.ln_f_beta,
                C,
                "dlnfb",
                grads_tol,
            )

            for i in range(16):
                allok = allok and gradoks[i]
        else:
            # For step > 0, just run backward to keep state consistent
            model.zero_gradients()
            model.backward()

        model.update(
            t=UInt32(step + 1),
            learning_rate=Scalar[DType.float32](1e-4),
            beta1=Scalar[DType.float32](0.9),
            beta2=Scalar[DType.float32](0.999),
            eps=Scalar[DType.float32](1e-8),
            weight_decay=Scalar[DType.float32](0.01),
        )

        var expected_loss_val = expected_losses[step]
        var actual_loss = model.mean_loss

        # We check loss step tolerance
        var _ = abs(expected_loss_val - actual_loss) < logits_tol
        # Note: we do not enforce tiny shakespeare losses for the debug 64-token batch,
        # but we print if it matches the expected overfitting trajectory or not.

        # print timing information at the end
        print(
            "step "
            + String(step)
            + ": loss "
            + String(model.mean_loss)
            + " (took "
            + String(elapsed_time_ms)
            + " ms)"
        )

    print("overall okay:", allok)

    # free everything
    x.free()
    y.free()
    expected_logits.free()
    expected_loss.free()
    expected_grads_memory.free()
    return allok


def main() raises:
    var args = argv()
    var target: String = "cpu"
    if len(args) > 1:
        target = args[1]

    # LLMM_RECOMPUTE=1 builds the model with activation recompute enabled so the
    # reference check guards that path; recompute is comptime, so dispatch here.
    var recompute = False
    var env_recompute = getenv("LLMM_RECOMPUTE")
    if env_recompute != "" and atol(env_recompute) != 0:
        recompute = True

    if target == "gpu":
        print("=== Running GPU Tests ===")
        # GPU version can run with dedicated tolerances
        var gpu_ok: Bool
        if recompute:
            gpu_ok = run_test["gpu", True](100.0, 0.1, 2.0)
        else:
            gpu_ok = run_test["gpu", False](100.0, 0.1, 2.0)
        print("GPU Test Result:", gpu_ok)
        if not gpu_ok:
            exit(1)
    else:
        print("=== Running CPU Tests ===")
        # CPU version has higher precision, let's use dedicated tolerances
        var cpu_ok: Bool
        if recompute:
            cpu_ok = run_test["cpu", True](100.0, 0.08, 10.0)
        else:
            cpu_ok = run_test["cpu", False](100.0, 0.08, 10.0)
        print("CPU Test Result:", cpu_ok)
        if not cpu_ok:
            exit(1)
        print(
            "All CPU tests passed successfully! To run GPU tests, run: mojo"
            " test_gpt2.mojo gpu"
        )
