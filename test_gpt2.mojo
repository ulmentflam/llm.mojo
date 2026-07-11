from std.collections import InlineArray, List
from std.time import perf_counter_ns
from std.sys import exit, argv, has_accelerator
from std.os import getenv
from std.sys.info import size_of
from std.memory import alloc, UnsafePointer, memcpy
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from std.math import sqrt

from llmm.memory import MutMemPtr
from llmm.vendor import HAS_CUBLAS, USE_TF32
from llmm.zero import ZeroContext

from train_gpt2 import (
    GPT2,
    GPT2_DTYPE,
    ParameterTensors,
)

comptime SIZEOF_INT = size_of[DType.int32]()
comptime SIZEOF_FLOAT = size_of[DType.float32]()


# Tensor checker with mixed relative+absolute tolerance and L2 norm guard.
#
# Parametric over the device dtype (GPT2_DTYPE for params/grads, which is fp32
# or bf16 depending on the build): the host buffer matches the device element
# size — so bf16 tensors copy correctly — and each element is cast to fp32 to
# compare against the fp32 reference.
#
# Criteria (both must pass):
#   1. maxdiff  <=  atol + rtol * ref_maxabs       (mixed absolute+relative)
#   2. our_l2   is within l2_factor of ref_l2       (L2 norm sanity)
#
# The L2 norm check is the key guard against dead-gradient bugs: a tensor whose
# values are geometrically decaying (near-zero) while the reference is O(1) will
# fail check 2 immediately even if the element-wise maxdiff happens to be small.
def check_tensor[
    dtype: DType
](
    ctx: DeviceContext,
    device_ptr: MutMemPtr[dtype],
    expected_ptr: UnsafePointer[Scalar[DType.float32], _],
    n: Int,
    label: String,
    atol: Float32,
    rtol: Float32,
    l2_factor: Float32,
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

    # --- Pass 1: compute full-tensor statistics ---
    var maxdiff: Float32 = 0.0
    var ref_maxabs: Float32 = 0.0
    var ref_l2_sq: Float32 = 0.0
    var our_l2_sq: Float32 = 0.0

    for i in range(n):
        var actual = host_buf[i].cast[DType.float32]()
        var expected = expected_ptr[i]
        var diff = abs(actual - expected)
        if diff > maxdiff:
            maxdiff = diff
        var ref_abs = abs(expected)
        if ref_abs > ref_maxabs:
            ref_maxabs = ref_abs
        ref_l2_sq = ref_l2_sq + expected * expected
        our_l2_sq = our_l2_sq + actual * actual

    var ref_l2 = sqrt(ref_l2_sq)
    var our_l2 = sqrt(our_l2_sq)

    var threshold = atol + rtol * ref_maxabs
    var maxdiff_ok = maxdiff <= threshold

    # L2 norm sanity: our L2 must be within l2_factor of ref L2.
    # Dead-gradient signature: our_l2 << ref_l2 (fails lower bound).
    var l2_ok: Bool
    if ref_l2 > Float32(1e-6):
        l2_ok = (our_l2 <= l2_factor * ref_l2) and (
            our_l2 * l2_factor >= ref_l2
        )
    else:
        # Reference is essentially zero — accept ours only if also near-zero.
        l2_ok = our_l2 <= Float32(1e-3)

    var ok = maxdiff_ok and l2_ok

    # --- Pass 2: print label + first few elements ---
    var print_upto: Int = 5
    print(label)
    for i in range(n):
        if i >= print_upto:
            break
        var actual = host_buf[i].cast[DType.float32]()
        var diff = abs(actual - expected_ptr[i])
        if diff <= threshold:
            print("  OK  ", end="")
        else:
            print("  FAIL", end="")
        print(
            " actual="
            + String(actual)
            + " ref="
            + String(expected_ptr[i])
            + " diff="
            + String(diff)
        )

    # --- Per-tensor diagnostic summary ---
    print(
        "  maxdiff="
        + String(maxdiff)
        + "  threshold="
        + String(threshold)
        + "  (atol="
        + String(atol)
        + " rtol="
        + String(rtol)
        + " ref_maxabs="
        + String(ref_maxabs)
        + ")"
    )
    if ref_l2 > Float32(1e-6):
        print(
            "  our_l2="
            + String(our_l2)
            + "  ref_l2="
            + String(ref_l2)
            + "  ratio="
            + String(our_l2 / ref_l2)
            + "  [must be in (1/"
            + String(l2_factor)
            + ", "
            + String(l2_factor)
            + ")]"
        )
    else:
        print(
            "  our_l2="
            + String(our_l2)
            + "  ref_l2="
            + String(ref_l2)
            + "  (ref≈0; our must be <1e-3)"
        )

    # --- Verdict ---
    if ok:
        print("TENSOR OK")
    else:
        var reasons: String = ""
        if not maxdiff_ok:
            reasons = (
                reasons
                + " MAXDIFF("
                + String(maxdiff)
                + ">"
                + String(threshold)
                + ")"
            )
        if not l2_ok:
            reasons = (
                reasons
                + " L2_RATIO(ours="
                + String(our_l2)
                + " ref="
                + String(ref_l2)
                + " limit="
                + String(l2_factor)
                + "x)"
            )
        print("TENSOR NOT OK:" + reasons)

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
](logits_tol: Float32, loss_tol: Float32,) raises -> Bool:
    # Gradient check tolerances — mixed relative+absolute criterion:
    #   pass iff maxdiff <= GRAD_ATOL + GRAD_RTOL * ref_maxabs
    # This is tighter than the old flat absolute tolerance (2.0) for any tensor
    # whose ref_maxabs is in the typical range O(0.1–5).  For near-zero tensors
    # the absolute floor (GRAD_ATOL) still applies.
    #
    # The L2 norm must additionally be within GRAD_L2_FACTOR of the reference.
    # A dead-gradient (geometrically decaying to ~0) fails this check immediately
    # even if its maxdiff happens to be below the threshold.
    #
    # Calibration rationale:
    #   - fp32 dot-product accumulation error over 4096 rows ≈ O(√4096 · ε₃₂) ≈ 8e-6
    #   - GPU vs CPU reorder can add an order of magnitude: practical maxdiff O(1e-4)
    #   - GRAD_ATOL = 0.01 and GRAD_RTOL = 0.05 gives threshold O(0.01–0.26) depending
    #     on ref_maxabs; this passes the 9 healthy tensors while rejecting near-zero garbage.
    #   - GRAD_L2_FACTOR = 3.0 allows the L2 to be at most 3× off — catching dead grads
    #     whose L2 is orders of magnitude below the reference, while tolerating normal
    #     fp32 accumulation noise (which shifts L2 by < 1%).
    #
    # Loss-trajectory tolerance (LOSS_STEP_TOL):
    #   llm.c uses 1e-2 absolute for per-step loss checks against the PyTorch reference.
    #
    # TF32 exception: when the build's fp32 GEMMs run on TF32 tensor cores
    # (default on NVIDIA — llmm/vendor.mojo's USE_TF32; `make verify-gpu`
    # disables it with -D LLMM_NO_TF32=1 to keep this gate strict-IEEE), the
    # 10-bit-mantissa GEMM inputs drift the per-step loss slightly: measured
    # max |delta| over the 10-step overfit run is 0.0102 with no growth
    # trend. 0.02 = ~2x that measured drift — loose enough that healthy TF32
    # rounding passes, tight enough that a real regression (which shows up as
    # a large per-step deviation or a growing trend, e.g. the dead-gradient
    # trajectories this check exists to catch deviate by O(0.1+) within a few
    # steps) still fails. The gradient tolerances are NOT loosened: TF32
    # passes them with 30-100x margin, so they stay at full strength.
    var GRAD_ATOL: Float32 = 0.01
    var GRAD_RTOL: Float32 = 0.05
    var GRAD_L2_FACTOR: Float32 = 3.0
    var LOSS_STEP_TOL: Float32 = 0.01
    comptime tf32_active = (
        USE_TF32
        and HAS_CUBLAS
        and is_gpu[target]()
        and GPT2_DTYPE == DType.float32
    )
    comptime if tf32_active:
        LOSS_STEP_TOL = 0.02
        print(
            "TF32 fp32 GEMMs active (USE_TF32=1): LOSS_STEP_TOL=0.02"
            " (calibrated for TF32 drift; build with -D LLMM_NO_TF32=1 for"
            " the strict-IEEE 0.01 gate)"
        )
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
        if state_header[0] == 20240327:
            # llm.c's downloadable debug state (magic 20240327, version 2) has
            # no activation tensors, so this test can't consume it. The
            # reference files must be regenerated with the PyTorch script.
            print(
                "gpt2_124M_debug_state.bin is llm.c's downloaded debug state,"
                " which lacks the activation tensors this test checks."
                " Regenerate the reference files with:"
                " pixi run python train_gpt2.py"
            )
        else:
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

    var allok: Bool = True

    # 10-step overfitting trajectory regenerated by:
    #   pixi run python /Users/evanowen/Workspace/scripts/llmm-metal-probes/gen_expected_losses.py
    # Optimizer config matching model.update() below: AdamW lr=1e-4, beta1=0.9,
    # beta2=0.999, eps=1e-8, weight_decay=0.01 (uniform, all params), no grad clip.
    # Batch: x/y from gpt2_124M_debug_state.bin (B=4, T=64), repeated 10 steps.
    # dtype=float32, device=cpu. Step-0 must be ≈5.354 (debug-state reference).
    var expected_losses: List[Float32] = [
        5.354427337646484,
        3.86495041847229,
        3.102630615234375,
        2.49684739112854,
        2.052116870880127,
        1.6149173974990845,
        1.2080059051513672,
        0.8576483130455017,
        0.5846155881881714,
        0.3554312288761139,
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
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[1] = check_tensor(
                ctx,
                model.grads.wpe,
                expected_grads.wpe,
                maxT * C,
                "dwpe",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[2] = check_tensor(
                ctx,
                model.grads.ln_1_gamma,
                expected_grads.ln_1_gamma,
                L * C,
                "dln1w",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[3] = check_tensor(
                ctx,
                model.grads.ln_1_beta,
                expected_grads.ln_1_beta,
                L * C,
                "dln1b",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[4] = check_tensor(
                ctx,
                model.grads.qkv_weight,
                expected_grads.qkv_weight,
                L * 3 * C * C,
                "dqkvw",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[5] = check_tensor(
                ctx,
                model.grads.qkv_bias,
                expected_grads.qkv_bias,
                L * 3 * C,
                "dqkvb",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[6] = check_tensor(
                ctx,
                model.grads.attn_proj_weight,
                expected_grads.attn_proj_weight,
                L * C * C,
                "dattprojw",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[7] = check_tensor(
                ctx,
                model.grads.attn_proj_bias,
                expected_grads.attn_proj_bias,
                L * C,
                "dattprojb",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[8] = check_tensor(
                ctx,
                model.grads.ln_2_gamma,
                expected_grads.ln_2_gamma,
                L * C,
                "dln2w",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[9] = check_tensor(
                ctx,
                model.grads.ln_2_beta,
                expected_grads.ln_2_beta,
                L * C,
                "dln2b",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[10] = check_tensor(
                ctx,
                model.grads.fc_weight,
                expected_grads.fc_weight,
                L * 4 * C * C,
                "dfcw",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[11] = check_tensor(
                ctx,
                model.grads.fc_bias,
                expected_grads.fc_bias,
                L * 4 * C,
                "dfcb",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[12] = check_tensor(
                ctx,
                model.grads.proj_weight,
                expected_grads.proj_weight,
                L * C * 4 * C,
                "dfcprojw",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[13] = check_tensor(
                ctx,
                model.grads.proj_bias,
                expected_grads.proj_bias,
                L * C,
                "dfcprojb",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[14] = check_tensor(
                ctx,
                model.grads.ln_f_gamma,
                expected_grads.ln_f_gamma,
                C,
                "dlnfw",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
            )
            gradoks[15] = check_tensor(
                ctx,
                model.grads.ln_f_beta,
                expected_grads.ln_f_beta,
                C,
                "dlnfb",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
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

        # --- Loss-trajectory check (assertive, tight tolerance) ---
        # Compare this step's loss against the expected overfitting trajectory
        # from gpt2_124M_debug_state.bin (the same PyTorch reference used for
        # logits/grads).  We use LOSS_STEP_TOL = 1e-2, matching llm.c.
        # A dead-gradient bug makes the loss stall (or diverge) after step 0,
        # so this check is the first to catch training-dynamics regressions.
        var expected_loss_val = expected_losses[step]
        var actual_loss = model.mean_loss
        var loss_step_diff = abs(expected_loss_val - actual_loss)
        var loss_step_ok = loss_step_diff <= LOSS_STEP_TOL
        allok = allok and loss_step_ok

        if loss_step_ok:
            print(
                "step "
                + String(step)
                + ": loss "
                + String(actual_loss)
                + " expected "
                + String(expected_loss_val)
                + " LOSS OK"
                + " (took "
                + String(elapsed_time_ms)
                + " ms)"
            )
        else:
            print(
                "step "
                + String(step)
                + ": loss "
                + String(actual_loss)
                + " expected "
                + String(expected_loss_val)
                + " LOSS MISMATCH diff="
                + String(loss_step_diff)
                + " > tol="
                + String(LOSS_STEP_TOL)
                + " (took "
                + String(elapsed_time_ms)
                + " ms)"
            )

    print("overall okay:", allok)

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
        # The "gpu" instantiation is comptime-guarded: the stdlib's GPU-arch
        # lookup fails the whole build on hosts with no accelerator.
        comptime if has_accelerator():
            print("=== Running GPU Tests ===")
            # GPU version can run with dedicated tolerances
            var gpu_ok: Bool
            if recompute:
                gpu_ok = run_test["gpu", True](100.0, 0.1)
            else:
                gpu_ok = run_test["gpu", False](100.0, 0.1)
            print("GPU Test Result:", gpu_ok)
            if not gpu_ok:
                exit(1)
        else:
            print(
                "No accelerator detected on this host; GPU tests unavailable."
            )
            exit(1)
    else:
        print("=== Running CPU Tests ===")
        # CPU version has higher precision, let's use dedicated tolerances
        var cpu_ok: Bool
        if recompute:
            cpu_ok = run_test["cpu", True](100.0, 0.08)
        else:
            cpu_ok = run_test["cpu", False](100.0, 0.08)
        print("CPU Test Result:", cpu_ok)
        if not cpu_ok:
            exit(1)
        print(
            "All CPU tests passed successfully! To run GPU tests, run: mojo"
            " test_gpt2.mojo gpu"
        )
