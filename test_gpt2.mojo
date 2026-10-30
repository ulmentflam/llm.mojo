from std.collections import Array, List
from std.time import perf_counter_ns
from std.sys import exit, argv, has_accelerator
from std.os import getenv
from std.sys.info import size_of

from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_gpu
from std.math import sqrt

from llmm.memory import MutMemPtr, heap_alloc
from llmm.log import Field, Log, num, text, DEBUG, NORMAL, QUIET, VERBOSE
from llmm.term import pad_right
from llmm.vendor import HAS_CUBLAS, USE_TF32
from llmm.gelu import MLP_RELU2

from train_gpt2 import (
    GPT2,
    GPT2_DTYPE,
    ROTARY,
    ParameterTensors,
    _ffmt,
)

# -D LLMM_MLP_ACT=relu2 swaps the MLP activation, so the reference state and the
# loss trajectory come from the PyTorch reference run with GPT2_ACT=relu2
# (train_gpt2.py writes gpt2_124M_relu2_debug_state.bin; the trajectory is
# scripts/gen_expected_losses.py's output under the same env var).
# -D LLMM_ROTARY=1 reads the rotary reference (same weights, rotary q/k, no wpe
# add; written by `make rotary-reference`). Rotary and relu2 together have no
# PyTorch trajectory here, so that combination is refused at compile time.
comptime STATE_FILE = (
    "gpt2_124M_rotary_debug_state.bin" if ROTARY else (
        "gpt2_124M_relu2_debug_state.bin" if MLP_RELU2 else "gpt2_124M_debug_state.bin"
    )
)
comptime SIZEOF_INT = size_of[DType.int32]()
comptime SIZEOF_FLOAT = size_of[DType.float32]()


# Output: one event per check (llmm/log.mojo), so every LLMM_LOG format works.
# In pretty, the gradient and loss-trajectory counters redraw in place on a
# terminal. A failing check always prints its full detail; LLMM_LOG_LEVEL=
# verbose adds a line per tensor and per trajectory step, and debug adds the
# element dumps and first logits this gate used to print unconditionally.
def _row(log: Log, label: String, body: String) -> String:
    return "  " + log.style.dim(pad_right(label, 11)) + body


def _ok(log: Log) -> String:
    return log.style.good("ok")


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
    expected_ptr: Pointer[Scalar[DType.float32], _],
    n: Int,
    label: String,
    atol: Float32,
    rtol: Float32,
    l2_factor: Float32,
    log: Log,
    gate: String,
    mut checked: Int,
) raises -> Bool:
    # Allocate a Metal-registered host buffer so enqueue_copy works on
    # both Metal (which rejects plain-malloc dst pointers) and CUDA.
    var host_buf = ctx.enqueue_create_host_buffer[dtype](n)
    try:
        ctx.enqueue_copy(
            dst_ptr=rebind[Pointer[Scalar[dtype], MutAnyOrigin]](
                host_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
            src_ptr=rebind[Pointer[Scalar[dtype], ImmutAnyOrigin]](
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
        var expected = expected_ptr[unsafe_offset=i]
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
    checked += 1

    var l2_ratio = our_l2 / ref_l2 if ref_l2 > Float32(1e-6) else our_l2
    var tensor_fields: List[Field] = [
        text("gate", gate),
        text("tensor", label),
        text("status", "ok" if ok else "fail"),
        num("maxdiff", maxdiff),
        num("threshold", threshold),
        num("l2_ratio", l2_ratio),
    ]
    var tensor_body = (
        pad_right(label, 10)
        + log.style.dim("maxdiff ")
        + String(maxdiff)
        + log.style.dim(" / ")
        + String(threshold)
        + log.style.dim("  l2 ")
        + String(l2_ratio)
        + " "
    )
    var tensor_line = _row(
        log,
        "tensor",
        tensor_body + (_ok(log) if ok else log.style.bad("FAIL")),
    )
    log.emit(
        "tensor",
        VERBOSE if ok else QUIET,
        tensor_fields,
        pretty=tensor_line,
        plain=tensor_line,
    )
    if ok:
        log.redraw(
            NORMAL,
            _row(log, "gradients", String(checked) + " tensors ") + _ok(log),
        )

    # The element dump: always for a failure, for every tensor at debug.
    var dump_level = DEBUG if ok else QUIET
    if not log.at(dump_level):
        _ = host_buf^
        return ok

    # --- Pass 2: print label + first few elements ---
    var print_upto: Int = 5
    log.raw(dump_level, label)
    for i in range(n):
        if i >= print_upto:
            break
        var actual = host_buf[i].cast[DType.float32]()
        var diff = abs(actual - expected_ptr[unsafe_offset=i])
        log.raw(
            dump_level,
            ("  OK  " if diff <= threshold else "  FAIL")
            + " actual="
            + String(actual)
            + " ref="
            + String(expected_ptr[unsafe_offset=i])
            + " diff="
            + String(diff),
        )

    # --- Per-tensor diagnostic summary ---
    log.raw(
        dump_level,
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
        + ")",
    )
    if ref_l2 > Float32(1e-6):
        log.raw(
            dump_level,
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
            + ")]",
        )
    else:
        log.raw(
            dump_level,
            "  our_l2="
            + String(our_l2)
            + "  ref_l2="
            + String(ref_l2)
            + "  (ref≈0; our must be <1e-3)",
        )

    # --- Verdict ---
    if ok:
        log.raw(dump_level, "TENSOR OK")
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
        log.raw(dump_level, log.style.bad("TENSOR NOT OK:" + reasons))

    _ = host_buf^
    return ok


def read_to_dtype_pointer[
    T: DType
](
    ptr: Pointer[Scalar[T], _], mut file_handle: FileHandle, size: Int
) raises -> None:
    # Element-wise copy with an explicit keep-alive, mirroring
    # llmm.io.read_and_copy and tests/test_zero_equivalence.mojo. A rebound
    # pointer drops the List's lifetime tracking, so without the keep-alive
    # ASAP destruction frees the buffer before the copy and the allocator
    # clobbers the leading bytes (the debug-state magic read back as a heap
    # address).
    var bytes_to_read = size * size_of[T]()
    var bytes_data = file_handle.read_bytes(bytes_to_read)
    if len(bytes_data) < bytes_to_read:
        raise Error("Failed to read enough bytes from file")
    var dest = rebind[Pointer[Scalar[T], MutUntrackedOrigin]](ptr)
    var src_ptr = rebind[Pointer[UInt8, MutUntrackedOrigin]](
        bytes_data.unsafe_ptr()
    ).unsafe_bitcast[Scalar[T]]()
    for i in range(size):
        dest[unsafe_offset=i] = src_ptr[unsafe_offset=i]
    _ = bytes_data^


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
    var log = Log()
    var gate = String(target)
    var checked = 0
    comptime if tf32_active:
        LOSS_STEP_TOL = 0.02
        var tf32_note = String(
            "TF32 fp32 GEMMs active (USE_TF32=1): LOSS_STEP_TOL=0.02"
            " (calibrated for TF32 drift; build with -D LLMM_NO_TF32=1 for"
            " the strict-IEEE 0.01 gate)"
        )
        var note_fields: List[Field] = [
            text("gate", gate),
            text("text", tf32_note),
        ]
        log.emit(
            "note",
            NORMAL,
            note_fields,
            pretty=log.style.dim("  note: " + tf32_note),
            plain="  note: " + tf32_note,
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

    comptime assert not (
        ROTARY and MLP_RELU2
    ), "test_gpt2 has no reference for LLMM_ROTARY=1 with LLMM_MLP_ACT=relu2"

    # STATE_FILE selects the reference by build flag (default, relu2 or rotary),
    # all written by train_gpt2.py / `make rotary-reference`.
    var state_file = open(STATE_FILE, "r")

    var state_header = heap_alloc[Int32](256)
    read_to_dtype_pointer[DType.int32](state_header, state_file, 256)

    if state_header[unsafe_offset=0] != 20240520:
        if state_header[unsafe_offset=0] == 20240327:
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
    if state_header[unsafe_offset=1] != 3:
        print("Bad version in model file:", state_header[unsafe_offset=1])
        exit(1)

    var B: Int = Int(state_header[unsafe_offset=2])  # batch size, e.g. 4
    var T: Int = Int(
        state_header[unsafe_offset=3]
    )  # time / sequence length (e.g. 64, up to maxT)

    log.raw(DEBUG, "[State]")
    log.raw(DEBUG, "batch_size: " + String(B))
    log.raw(DEBUG, "seq_len: " + String(T))

    var expected_grads = ParameterTensors[DType.float32]()
    var expected_grads_memory = heap_alloc[Scalar[DType.float32]](
        model.num_parameters
    )
    expected_grads.point_parameters(
        model.param_sizes,
        rebind[MutMemPtr[DType.float32]](expected_grads_memory),
    )

    # inputs and expected outputs, only used for error checking

    var x = heap_alloc[SIMD[DType.int32, 1]](B * T)
    var y = heap_alloc[SIMD[DType.int32, 1]](B * T)

    var expected_logits = heap_alloc[SIMD[DType.float32, 1]](B * T * V)
    var expected_loss = heap_alloc[SIMD[DType.float32, 1]](1)

    # read reference information from Python

    read_to_dtype_pointer[DType.int32](x, state_file, B * T)
    read_to_dtype_pointer[DType.int32](y, state_file, B * T)
    read_to_dtype_pointer[DType.float32](expected_logits, state_file, B * T * V)
    read_to_dtype_pointer[DType.float32](expected_loss, state_file, 1)
    read_to_dtype_pointer[DType.float32](
        expected_grads_memory, state_file, model.num_parameters
    )

    state_file.close()
    state_header.unsafe_free()

    var allok: Bool = True

    # 10-step overfitting trajectory. Regenerate with:
    #   pixi run python scripts/gen_expected_losses.py
    # Optimizer config matching model.update() below: AdamW lr=1e-4, beta1=0.9,
    # beta2=0.999, eps=1e-8, weight_decay=0.01 (uniform, all params), no grad clip.
    # Batch: x/y from gpt2_124M_debug_state.bin (B=4, T=64), repeated 10 steps.
    # dtype=float32, TF32 off. Step 0 is a plain forward pass on the debug
    # state's own batch and weights, so it reproduces the reference loss stored
    # in that file exactly: 5.2699823 (llm.c's own reference for this batch is
    # 5.2700086). The generator asserts this and refuses to emit a list if it
    # ever stops holding.
    #
    # The previous list (step 0 = 5.2678394) came from train_gpt2.py while its
    # LayerNorm used the unbiased (N-1) variance, which puts PyTorch ~2e-3 off
    # GPT-2 (and off llmm, which uses the population variance). The 0.02-0.3
    # per-step misses `make verify` used to print were a second problem: a
    # debug state left by an older train_gpt2.py held a different first batch
    # (stored loss 5.354448) than the one the list was generated on. With the
    # LayerNorm fixed and the state regenerated, PyTorch and this trainer agree
    # to ~2e-4 per step.
    #
    # These numbers come from PyTorch and must ONLY ever come from PyTorch.
    # Regenerating them from this trainer's own output would make the check
    # compare the implementation against itself -- green forever, testing
    # nothing. The previous list was produced by a script at a /Users/... path
    # that was never in this repo; it claimed step 0 was 5.354 while the
    # reference file said 5.2678394, so every one of these ten steps reported
    # LOSS MISMATCH on every commit, and `make verify-gpu` exited 1 for
    # everyone. See scripts/gen_expected_losses.py for the full story.
    var expected_losses: List[Float32]
    comptime if ROTARY:
        # PyTorch with GPT2_ROTARY=1 (same weights, rotary q/k, no wpe add),
        # from `make rotary-reference`; step 0 is the rotary debug state's
        # stored loss. Pretrained GPT-2 weights are not trained for rotary,
        # hence the high start.
        expected_losses = [
            9.35813045501709,
            6.650364875793457,
            5.8415021896362305,
            5.305992603302002,
            4.834591388702393,
            4.441601753234863,
            4.0521063804626465,
            3.70202374458313,
            3.3840603828430176,
            3.0682621002197266,
        ]
    elif MLP_RELU2:
        # relu2 trajectory: torch FLOAT64 (GPT2_ACT=relu2 scripts/gen_expected_losses.py
        # --dtype float64), not fp32. relu2 on GELU-pretrained weights is
        # ill-conditioned (residual stream ~1.2e5 from layer 3), and torch's own fp32
        # run drifts from its fp64 run by 0.0126 at step 9 (3.367974 vs 3.380598),
        # more than LOSS_STEP_TOL. Mojo fp32 (GPU 3.37966, CPU 3.38055) lands within
        # 1e-3 of fp64, so the gate keeps the full-strength 0.01 against the true
        # trajectory. Step 0 differs from the debug state's fp32 loss by 1e-5 (the
        # stale-state guard allows 1e-3). See docs/ai/w5_w6_zero_init_proj_and_relu2_2026-10-30.md.
        expected_losses = [
            7.184733928491164,
            6.57912622894239,
            6.156180415699619,
            5.7834021372001345,
            5.452741764643019,
            5.0208772006836355,
            4.616522495663473,
            4.157646495279576,
            3.7607664251102446,
            3.380597985334229,
        ]
    else:
        expected_losses = [
            5.26998233795166,
            4.059586048126221,
            3.374852418899536,
            2.8005142211914062,
            2.315274715423584,
            1.8489989042282104,
            1.3946499824523926,
            0.9988086223602295,
            0.6243725419044495,
            0.37682223320007324,
        ]

    # Stale-state guard. Step 0 of the list is a plain forward pass on the debug
    # state's own batch, so the loss stored in the state file must equal it
    # (llm.c's own reference differs by 3e-5). A state left by an older
    # train_gpt2.py holds a different batch (e.g. 5.3544) and would otherwise
    # surface as ten unrelated-looking LOSS MISMATCH lines; fail here, loudly,
    # naming the cause and the fix instead.
    if abs(expected_loss[unsafe_offset=0] - expected_losses[0]) > 1e-3:
        print("=========================================================")
        print(
            "STALE DEBUG STATE: gpt2_124M_debug_state.bin stores loss",
            expected_loss[unsafe_offset=0],
        )
        print("but the expected trajectory starts at", expected_losses[0])
        print(
            "The state file was written by an older train_gpt2.py (different"
            " batch)."
        )
        print("This is NOT a trainer regression. Regenerate it with:")
        print(
            "    pixi run python train_gpt2.py --device cpu --num_iterations 0"
        )
        print("=========================================================")
        return False

    var steps_failed = 0
    var last_step_line = String()
    var last_loss = Float32(0.0)
    var last_ref = Float32(0.0)
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
                dst_ptr=rebind[Pointer[Scalar[DType.float32], MutAnyOrigin]](
                    host_logits.unsafe_ptr().as_unsafe_any_origin()
                ),
                src_ptr=rebind[Pointer[Scalar[DType.float32], ImmutAnyOrigin]](
                    model.acts.logits.as_unsafe_any_origin()
                ),
                size=B * T * V_p,
            )
            ctx.synchronize()

            var logits_ok: Bool = True
            var print_count = 0
            # How many logits the check below actually compared, reported so
            # the gate can't silently pass on an empty comparison.
            var logits_compared = 0

            for b in range(B):
                for t in range(T):
                    for v in range(V):
                        var idx_py = (b * T + t) * V + v
                        var idx_mj = (b * T + t) * V_p + v

                        if log.at(DEBUG) and print_count < 3:
                            log.raw(
                                DEBUG,
                                "PyTorch: "
                                + String(expected_logits[unsafe_offset=idx_py])
                                + " Mojo: "
                                + String(host_logits[idx_mj]),
                            )
                            print_count += 1

                        # We only check logits that have a non-negligible impact on softmax (expected_logits > -10.0)
                        if expected_logits[unsafe_offset=idx_py] > -10.0:
                            logits_compared += 1
                            if (
                                abs(
                                    expected_logits[unsafe_offset=idx_py]
                                    - host_logits[idx_mj]
                                )
                                >= logits_tol
                            ):
                                log.raw(
                                    QUIET,
                                    "MISMATCH AT INDEX (Py="
                                    + String(idx_py)
                                    + ", Mj="
                                    + String(idx_mj)
                                    + "): expected (Py) "
                                    + String(
                                        expected_logits[unsafe_offset=idx_py]
                                    )
                                    + ", got (Mj) "
                                    + String(host_logits[idx_mj]),
                                )
                                logits_ok = False
                                break
                    if not logits_ok:
                        break
                if not logits_ok:
                    break

            var logits_fields: List[Field] = [
                text("gate", gate),
                text("check", "logits"),
                text("status", "ok" if logits_ok else "fail"),
                num("compared", logits_compared),
                num("tol", logits_tol),
            ]
            var logits_line = _row(
                log,
                "logits",
                String(logits_compared)
                + " compared "
                + (_ok(log) if logits_ok else log.style.bad("MISMATCH")),
            )
            log.emit(
                "check",
                NORMAL if logits_ok else QUIET,
                logits_fields,
                pretty=logits_line,
                plain=logits_line,
            )
            allok = allok and logits_ok
            _ = host_logits^

            # compare the achieved loss
            var loss_ok = (
                abs(model.mean_loss - expected_loss[unsafe_offset=0]) < loss_tol
            )
            var loss_line = _row(
                log,
                "loss",
                _ffmt(Float64(model.mean_loss), 4)
                + "  "
                + log.style.dim("ref")
                + " "
                + _ffmt(Float64(expected_loss[unsafe_offset=0]), 4)
                + " "
                + (_ok(log) if loss_ok else log.style.bad("MISMATCH")),
            )
            var loss_fields: List[Field] = [
                text("gate", gate),
                text("check", "loss"),
                text("status", "ok" if loss_ok else "fail"),
                num("loss", model.mean_loss),
                num("ref", expected_loss[unsafe_offset=0]),
                num("tol", loss_tol),
            ]
            log.emit(
                "check",
                NORMAL if loss_ok else QUIET,
                loss_fields,
                pretty=loss_line,
                plain=loss_line,
            )
            if not loss_ok:
                allok = False

            # Run backward pass before checking gradients
            model.zero_gradients()
            model.backward()

            # finally check all the gradients
            var gradoks = Array[Bool, 16](fill=False)

            gradoks[0] = check_tensor(
                ctx,
                model.grads.wte,
                expected_grads.wte,
                V * C,
                "dwte",
                GRAD_ATOL,
                GRAD_RTOL,
                GRAD_L2_FACTOR,
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
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
                log,
                gate,
                checked,
            )

            var grads_passed = 0
            for i in range(16):
                allok = allok and gradoks[i]
                if gradoks[i]:
                    grads_passed += 1
            var grads_ok = grads_passed == 16
            var grads_line = _row(
                log,
                "gradients",
                "16 tensors "
                + _ok(log) if grads_ok else log.style.bad(
                    String(16 - grads_passed) + " of 16 tensors FAILED"
                ),
            )
            var grads_fields: List[Field] = [
                text("gate", gate),
                text("check", "grads"),
                text("status", "ok" if grads_ok else "fail"),
                num("tensors", 16),
                num("failed", 16 - grads_passed),
            ]
            # In pretty on a terminal this replaces the counter line.
            log.emit(
                "check",
                NORMAL if grads_ok else QUIET,
                grads_fields,
                pretty=grads_line,
                plain=grads_line,
            )
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

        var step_line = _row(
            log,
            "10 steps",
            "step "
            + String(step)
            + "  loss "
            + _ffmt(Float64(actual_loss), 4)
            + "  "
            + log.style.dim("ref")
            + " "
            + _ffmt(Float64(expected_loss_val), 4)
            + " "
            + (
                _ok(log) if loss_step_ok else log.style.bad(
                    "MISMATCH diff="
                    + String(loss_step_diff)
                    + " > tol="
                    + String(LOSS_STEP_TOL)
                )
            ),
        )
        var trajectory_fields: List[Field] = [
            text("gate", gate),
            num("step", step),
            text("status", "ok" if loss_step_ok else "fail"),
            num("loss", actual_loss),
            num("ref", expected_loss_val),
            num("diff", loss_step_diff),
        ]
        # verbose shows every step; a mismatch always shows.
        log.emit(
            "trajectory",
            VERBOSE if loss_step_ok else QUIET,
            trajectory_fields,
            pretty=step_line,
            plain=step_line,
        )
        if loss_step_ok:
            log.redraw(NORMAL, step_line)
        else:
            steps_failed += 1
        last_step_line = step_line
        last_loss = actual_loss
        last_ref = expected_loss_val

    var steps_ok = steps_failed == 0
    var steps_fields: List[Field] = [
        text("gate", gate),
        text("check", "steps"),
        text("status", "ok" if steps_ok else "fail"),
        num("steps", 10),
        num("failed", steps_failed),
        num("loss", last_loss),
        num("ref", last_ref),
        num("tol", LOSS_STEP_TOL),
    ]
    log.emit(
        "check",
        NORMAL if steps_ok else QUIET,
        steps_fields,
        pretty=last_step_line,
        plain=last_step_line,
    )

    x.unsafe_free()
    y.unsafe_free()
    expected_logits.unsafe_free()
    expected_loss.unsafe_free()
    expected_grads_memory.unsafe_free()
    return allok


def _gate_heading(log: Log, gate: String) raises:
    var fields: List[Field] = [text("gate", gate)]
    var heading = log.style.heading(
        gate.upper() + " gate", "GPT-2 124M against the PyTorch reference"
    )
    log.emit("gate", NORMAL, fields, pretty=heading, plain=heading)


def _gate_verdict(log: Log, gate: String, ok: Bool) raises:
    var fields: List[Field] = [
        text("gate", gate),
        text("status", "pass" if ok else "fail"),
    ]
    var verdict = "  " + (
        log.style.good("passed") if ok else log.style.bad("FAILED")
    )
    log.emit("result", QUIET, fields, pretty=verdict, plain=verdict)


def main() raises:
    var log = Log()
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
            _gate_heading(log, "gpu")
            # GPU version can run with dedicated tolerances
            var gpu_ok: Bool
            if recompute:
                gpu_ok = run_test["gpu", True](100.0, 0.1)
            else:
                gpu_ok = run_test["gpu", False](100.0, 0.1)
            _gate_verdict(log, "gpu", gpu_ok)
            if not gpu_ok:
                exit(1)
        else:
            print(
                "No accelerator detected on this host; GPU tests unavailable."
            )
            exit(1)
    else:
        _gate_heading(log, "cpu")
        # CPU version has higher precision, let's use dedicated tolerances
        var cpu_ok: Bool
        if recompute:
            cpu_ok = run_test["cpu", True](100.0, 0.08)
        else:
            cpu_ok = run_test["cpu", False](100.0, 0.08)
        _gate_verdict(log, "cpu", cpu_ok)
        if not cpu_ok:
            exit(1)
