# ===----------------------------------------------------------------------=== #
# Untied LM head, slice S0: layout, loaders and file-format refusals.
#
# Run with:  make test-mojo                                  (tied build)
#            make test-mojo TEST_DEFINES="-D LLMM_UNTIED_HEAD=1"   (untied build)
#
# docs/ai/untied_lm_head_design_2026-10-30.md. The trainer is compiled once per
# flag value, so each build runs only its own half of this file; the other half
# prints a SKIP line. Cross-build equality (untied-with-idle-head == tied) is a
# process-level comparison and lives in `make check-untied-idle`, not here.
#
# Every planted value is distinct from wte (head = 1000 + index), so a loader
# that dropped the head, copied the wrong block or read the wrong offset cannot
# pass by accident.
# ===----------------------------------------------------------------------=== #

from std.testing import assert_equal, assert_true, TestSuite
from std.python import Python

from max.gpu.host import DeviceContext

from llmm.checkpointing import write_model_checkpoint
from llmm.memory import MutMemPtr, heap_alloc
from train_gpt2 import GPT2, GPT2_DTYPE, UNTIED_HEAD, NUM_PARAMETER_TENSORS

comptime CKPT = "gpt2_tiny_untied_head.bin"
# Tiny model: L=1, C=16, NH=1, maxT=16, V=60 real rows padded to V_p=64.
comptime L = 1
comptime C = 16
comptime MAXT = 16
comptime V = 60
comptime VP = 64
comptime HEAD_N = VP * C
comptime N16 = (
    VP * C
    + MAXT * C
    + L * (C + C + 3 * C * C + 3 * C + C * C + C + C + C)
    + L * (4 * C * C + 4 * C + 4 * C * C + C)
    + 2 * C
)


def _write_file(path: String, version: Int, with_head: Bool, extra: Int) raises:
    """Tied (16 tensors) or untied (+ planted head) synthetic fp32 file; wte
    padding rows are zero, as in real files. `extra` appends junk elements."""
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    var header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = version
    header[2] = MAXT
    header[3] = V
    header[4] = L
    header[5] = 1
    header[6] = C
    header[7] = VP
    np.random.seed(5)
    var w = np.random.normal(0, 0.05, N16).astype(np.float32)
    for i in range(V * C, VP * C):
        w[i] = 0.0  # wte padding rows
    var f = builtins.open(path, "wb")
    _ = f.write(header.tobytes())
    _ = f.write(w.tobytes())
    if with_head:
        var h = np.zeros(HEAD_N, dtype=np.float32)
        for i in range(V * C):
            h[i] = 1000.0 + Float64(i)
        _ = f.write(h.tobytes())
    if extra > 0:
        _ = f.write(np.zeros(extra, dtype=np.float32).tobytes())
    f.close()


def _remove(path: String):
    try:
        var os = Python.import_module("os")
        _ = os.remove(path)
    except:
        pass


def _load_raises(path: String, needle: String) raises -> Bool:
    var ctx = DeviceContext(api="cpu")
    try:
        var model = GPT2["cpu", 1](path, rank=0, zero_stage=0, ctx=ctx)
        _ = model^
    except e:
        return needle in String(e)
    return False


def test_layout_constants() raises:
    comptime if UNTIED_HEAD:
        assert_equal(NUM_PARAMETER_TENSORS, 17)
    else:
        assert_equal(NUM_PARAMETER_TENSORS, 16)


def test_tied_file_loads_head_as_wte_copy() raises:
    comptime if not UNTIED_HEAD:
        print("SKIP test_tied_file_loads_head_as_wte_copy: tied build")
        return
    _write_file(CKPT, 3, False, 0)
    var ctx = DeviceContext(api="cpu")
    var model = GPT2["cpu", 1](CKPT, rank=0, zero_stage=0, ctx=ctx)
    _remove(CKPT)
    assert_equal(model.num_parameters, N16 + HEAD_N)
    var seen_nonzero = 0
    for i in range(HEAD_N):
        # head == wte bitwise, including the zero padding rows
        assert_equal(
            model.params_memory[unsafe_offset=N16 + i],
            model.params_memory[unsafe_offset=i],
        )
        if model.params_memory[unsafe_offset=N16 + i] != 0:
            seen_nonzero += 1
    assert_true(seen_nonzero > V * C // 2, "head copy looks empty")
    for i in range(V * C, VP * C):
        assert_equal(model.params_memory[unsafe_offset=N16 + i], 0)


def test_untied_file_loads_planted_head() raises:
    comptime if not UNTIED_HEAD:
        print("SKIP test_untied_file_loads_planted_head: tied build")
        return
    _write_file(CKPT, 13, True, 0)
    var ctx = DeviceContext(api="cpu")
    var model = GPT2["cpu", 1](CKPT, rank=0, zero_stage=0, ctx=ctx)
    _remove(CKPT)
    assert_equal(model.num_parameters, N16 + HEAD_N)
    for i in range(V * C):
        assert_equal(
            model.params_memory[unsafe_offset=N16 + i],
            Scalar[GPT2_DTYPE](1000 + i),
        )
    for i in range(V * C, VP * C):
        assert_equal(model.params_memory[unsafe_offset=N16 + i], 0)
    # The head must differ from wte (otherwise this test proves nothing).
    assert_true(
        model.params_memory[unsafe_offset=N16 + 1]
        != model.params_memory[unsafe_offset=1],
        "planted head equals wte",
    )


def test_untied_file_roundtrip_through_trainer_config() raises:
    """The trainer's own _checkpoint_config + the library writer reproduce the
    untied file byte for byte (version 13, head block last)."""
    comptime if not UNTIED_HEAD:
        print("SKIP test_untied_file_roundtrip: tied build")
        return
    _write_file(CKPT, 13, True, 0)
    var ctx = DeviceContext(api="cpu")
    var model = GPT2["cpu", 1](CKPT, rank=0, zero_stage=0, ctx=ctx)
    var cfg = model._checkpoint_config()
    assert_equal(cfg.untied_head, 1)
    assert_equal(cfg.num_parameters(), model.num_parameters)
    write_model_checkpoint(
        "gpt2_tiny_untied_head_out.bin",
        cfg,
        model.params_memory,
        model.num_parameters,
    )
    var builtins = Python.import_module("builtins")
    var a = builtins.open(CKPT, "rb").read()
    var b = builtins.open("gpt2_tiny_untied_head_out.bin", "rb").read()
    assert_true(Bool(py=a == b), "round-trip file differs from the original")
    _remove(CKPT)
    _remove("gpt2_tiny_untied_head_out.bin")


def test_tied_build_refuses_untied_file() raises:
    comptime if UNTIED_HEAD:
        print("SKIP test_tied_build_refuses_untied_file: untied build")
        return
    _write_file(CKPT, 13, True, 0)
    var refused = _load_raises(CKPT, "Invalid version")
    _remove(CKPT)
    assert_true(refused, "tied build accepted a version-13 file")


def test_wrong_length_files_refused_in_both_builds() raises:
    # Extended tied file (the pre-fix reader ignored trailing bytes).
    _write_file(CKPT, 3, False, 4)
    var ext = _load_raises(CKPT, "expected")
    # An untied-sized blob behind a TIED header: tied build must not load it
    # as tied-and-drop-the-head; the untied build must not either.
    _write_file(CKPT, 3, True, 0)
    var tied_hdr_untied_blob = _load_raises(CKPT, "expected")
    _remove(CKPT)
    assert_true(ext, "extended tied file was accepted")
    assert_true(tied_hdr_untied_blob, "tied header over untied blob accepted")
    comptime if UNTIED_HEAD:
        _write_file(CKPT, 13, True, 4)
        var uext = _load_raises(CKPT, "expected")
        # untied header but only the 16-tensor blob (truncated head)
        _write_file(CKPT, 13, False, 0)
        var trunc = _load_raises(CKPT, "expected")
        _remove(CKPT)
        assert_true(uext, "extended untied file was accepted")
        assert_true(trunc, "untied file without a head block was accepted")


comptime GB = 2
comptime GT = 8


def _fwd_bwd(
    mut model: GPT2["cpu", 1],
    x: MutMemPtr[DType.int32],
    y: MutMemPtr[DType.int32],
    grads: MutMemPtr[DType.float32],
) raises -> Float32:
    """One forward + zero_gradients + backward; copies the flat gradient buffer
    (same layout as the parameters) to `grads` and returns the loss."""
    model.forward(x, y, GB, GT)
    model.zero_gradients()
    model.backward()
    model.ctx.synchronize()
    var host = model.ctx.enqueue_create_host_buffer[GPT2_DTYPE](
        model.num_parameters
    )
    model.grads_buf.enqueue_copy_to(host)
    model.ctx.synchronize()
    for i in range(model.num_parameters):
        grads[unsafe_offset=i] = host[i].cast[DType.float32]()
    _ = host^
    return model.mean_loss


def test_untied_head_gradient_split_matches_tied() raises:
    """Planted unit test for the S2 routing, mirroring
    tests/test_untied_reference.py::test_untied_copy_of_wte_matches_tied_
    forward_and_grad_split. Start from a tied .bin, so head == wte bitwise.
    The tied reference is the same build with the head idle (`head_tied`
    True; `make check-untied-idle` proves that equals the tied build's
    process bitwise). Expect:
      * loss bitwise equal (the head reads a bitwise copy of wte);
      * rows of the batch's INPUT tokens (encoder) are 'present'; every
        vocab tile also holds absent rows (a whole-tile misroute is visible);
      * absent rows: untied dwte == 0 exactly and dlm_head == tied dwte;
      * present rows: untied dwte + dlm_head == tied dwte within 1e-6, and
        dwte != 0 so the check is not vacuous.
    A head gradient routed into wte (mutant) makes untied dwte nonzero on the
    absent rows and fails the zero check."""
    comptime if not UNTIED_HEAD:
        print("SKIP test_untied_head_gradient_split: tied build")
        return
    _write_file(CKPT, 3, False, 0)
    var x = heap_alloc[Scalar[DType.int32]](GB * GT)
    var y = heap_alloc[Scalar[DType.int32]](GB * GT)
    var present = heap_alloc[Int](VP)
    for r in range(VP):
        present[unsafe_offset=r] = 0
    for i in range(GB * GT):
        # inputs from 16 distinct rows (1, 4, ..., 46), so every 8-row vocab
        # tile (the default K=8 tiling of V_p=64) holds absent rows: a tile's
        # head gradient routed into wte must hit an absent row. Targets spread
        # over the whole vocab.
        x[unsafe_offset=i] = Int32(i * 3 + 1)
        y[unsafe_offset=i] = Int32((i * 29 + 11) % V)
        present[unsafe_offset=Int(x[unsafe_offset=i])] = 1
    var ctx = DeviceContext(api="cpu")
    var tied = GPT2["cpu", 1](CKPT, rank=0, zero_stage=0, ctx=ctx)
    var untied = GPT2["cpu", 1](CKPT, rank=0, zero_stage=0, ctx=ctx)
    _remove(CKPT)
    tied.head_tied = True
    assert_true(not untied.head_tied, "untied model has an idle head")
    var gt = heap_alloc[Float32](tied.num_parameters)
    var gu = heap_alloc[Float32](untied.num_parameters)
    var loss_t = _fwd_bwd(tied, x, y, gt)
    var loss_u = _fwd_bwd(untied, x, y, gu)
    assert_true(loss_t == loss_u, "untied loss != tied loss for head == wte")

    var abs_rows = 0
    var pres_rows = 0
    var abs_head_nonzero = 0
    for r in range(VP):
        for c in range(C):
            var i = r * C + c
            var dwte_t = gt[unsafe_offset=i]
            var dwte_u = gu[unsafe_offset=i]
            var dhead_u = gu[unsafe_offset=N16 + i]
            if present[unsafe_offset=r] == 0:
                assert_true(
                    dwte_u == 0.0, "absent row has encoder gradient in wte"
                )
                assert_true(
                    dhead_u == dwte_t, "absent row: dlm_head != tied dwte"
                )
                if dhead_u != 0.0:
                    abs_head_nonzero += 1
            else:
                assert_true(
                    abs(dwte_u + dhead_u - dwte_t) <= 1e-6,
                    "present row: dwte + dlm_head != tied dwte",
                )
        if present[unsafe_offset=r] == 0:
            abs_rows += 1
        else:
            pres_rows += 1
    # Non-vacuity: the planted batch has both kinds of rows, the head gradient
    # is real on absent rows, and the encoder gradient is real on present rows.
    assert_equal(pres_rows, GB * GT)
    for t in range(VP // 8):
        var absent_in_tile = 0
        for r in range(8 * t, 8 * t + 8):
            if present[unsafe_offset=r] == 0:
                absent_in_tile += 1
        assert_true(absent_in_tile > 0, "a vocab tile has no absent row")
    assert_true(abs_head_nonzero > 100, "head gradient on absent rows is empty")
    var enc_nonzero = 0
    for r in range(VP):
        if present[unsafe_offset=r] == 0:
            continue
        for c in range(C):
            if gu[unsafe_offset=r * C + c] != 0.0:
                enc_nonzero += 1
    assert_true(
        enc_nonzero > GB * GT * C // 2, "encoder gradient on wte is empty"
    )
    x.unsafe_free()
    y.unsafe_free()
    present.unsafe_free()
    gt.unsafe_free()
    gu.unsafe_free()


def _head_step(
    opt: String,
    head_mult: Float32,
    adam_mult: Float32,
    x: MutMemPtr[DType.int32],
    y: MutMemPtr[DType.int32],
    dst: MutMemPtr[DType.float32],
) raises -> Int:
    """One forward/backward/update from the tied CKPT (head == wte copy) with
    `-l_head head_mult` (and `-l_adam adam_mult` under Muon); the updated flat
    parameters go to `dst`. Weight decay is nonzero so a head the optimizer
    touched at lr 0 would still have to stay put (decay is lr-scaled)."""
    var ctx = DeviceContext(api="cpu")
    var model = GPT2["cpu", 1](CKPT, rank=0, zero_stage=0, ctx=ctx)
    if opt == "muon":
        model.enable_muon(Float32(0.95), adam_mult)
    model.head_lr_mult = head_mult
    var g = heap_alloc[Float32](model.num_parameters)
    _ = _fwd_bwd(model, x, y, g)
    g.unsafe_free()
    model.update(
        t=UInt32(1),
        learning_rate=Scalar[DType.float32](1e-3),
        weight_decay=Scalar[DType.float32](0.1),
    )
    model.ctx.synchronize()
    var n = model.num_parameters
    for i in range(n):
        dst[unsafe_offset=i] = model.params_memory[unsafe_offset=i].cast[
            DType.float32
        ]()
    return n


def test_untied_head_lr_mult() raises:
    """-l_head reaches the head's optimizer step and only the head, under both
    AdamW (`_adamw_flat`) and Muon (`_update_hybrid`), W=1:
      * -l_head 0 freezes the head bitwise (wd > 0, so a head updated with the
        body lr would move) while the body moves;
      * the body is bitwise independent of -l_head;
      * the sentinel (-1) equals an explicit -l_head 1 bitwise (AdamW) and an
        explicit -l_head equal to -l_adam bitwise (Muon, -l_adam 2);
      * -l_head 2 moves every head element twice as far as -l_head 1 (one
        AdamW step from zero moments is linear in the lr, decay included)."""
    comptime if not UNTIED_HEAD:
        print("SKIP test_untied_head_lr_mult: tied build")
        return
    _write_file(CKPT, 3, False, 0)
    var x = heap_alloc[Scalar[DType.int32]](GB * GT)
    var y = heap_alloc[Scalar[DType.int32]](GB * GT)
    for i in range(GB * GT):
        x[unsafe_offset=i] = Int32((i * 7) % 8)
        y[unsafe_offset=i] = Int32((i * 29 + 11) % V)
    var NT = N16 + HEAD_N
    var p0 = heap_alloc[Float32](NT)  # -l_head 0
    var p1 = heap_alloc[Float32](NT)  # sentinel
    var p1e = heap_alloc[Float32](NT)  # explicit 1 (or = l_adam under Muon)
    var p2 = heap_alloc[Float32](NT)  # -l_head 2 (AdamW) / 4 (Muon, l_adam 2)
    var init = heap_alloc[Float32](N16)
    var ctx0 = DeviceContext(api="cpu")
    var m0 = GPT2["cpu", 1](CKPT, rank=0, zero_stage=0, ctx=ctx0)
    for i in range(N16):
        init[unsafe_offset=i] = m0.params_memory[unsafe_offset=i].cast[
            DType.float32
        ]()
        # The loader copied the tied file's wte into the head.
        if i < HEAD_N:
            assert_true(
                m0.params_memory[unsafe_offset=N16 + i]
                == m0.params_memory[unsafe_offset=i],
                "head != wte copy at load",
            )
    _ = m0^
    for o in range(2):
        var opt = "adamw" if o == 0 else "muon"
        var am = Float32(1.0) if o == 0 else Float32(2.0)
        assert_equal(_head_step(opt, Float32(0.0), am, x, y, p0), NT)
        _ = _head_step(opt, Float32(-1.0), am, x, y, p1)
        _ = _head_step(opt, am, am, x, y, p1e)
        _ = _head_step(opt, Float32(2.0) * am, am, x, y, p2)
        var body_moved = 0
        var head_moved = 0
        for i in range(N16):
            assert_true(
                p0[unsafe_offset=i] == p1[unsafe_offset=i]
                and p2[unsafe_offset=i] == p1[unsafe_offset=i],
                opt + ": body depends on -l_head",
            )
            if p1[unsafe_offset=i] != init[unsafe_offset=i]:
                body_moved += 1
        for i in range(HEAD_N):
            var h0 = init[unsafe_offset=i]  # head starts as the wte copy
            var j = N16 + i
            assert_true(
                p0[unsafe_offset=j] == h0, opt + ": -l_head 0 moved the head"
            )
            assert_true(
                p1e[unsafe_offset=j] == p1[unsafe_offset=j],
                opt + ": sentinel -l_head != explicit equal multiplier",
            )
            var d1 = p1[unsafe_offset=j] - h0
            var d2 = p2[unsafe_offset=j] - h0
            if d1 != 0.0:
                head_moved += 1
            assert_true(
                abs(d2 - 2.0 * d1) <= 1e-3 * abs(d1) + 1e-9,
                opt + ": -l_head 2x did not move the head 2x",
            )
        assert_true(body_moved > N16 // 2, opt + ": body did not move")
        # Real rows only; the 4 padding rows have zero gradient and value.
        assert_true(head_moved > (V * C) // 2, opt + ": head did not move")
    _remove(CKPT)
    x.unsafe_free()
    y.unsafe_free()
    p0.unsafe_free()
    p1.unsafe_free()
    p1e.unsafe_free()
    p2.unsafe_free()
    init.unsafe_free()


def test_untied_grad_norm_covers_head() raises:
    """With the head training, the clip norm is the L2 norm over all 17
    gradient tensors; with the head idle it is the body-only norm (S0's
    bit-identity with the tied build). The planted batch makes the two differ
    by far more than the tolerance, so a 16-tensor norm cannot pass."""
    comptime if not UNTIED_HEAD:
        print("SKIP test_untied_grad_norm_covers_head: tied build")
        return
    _write_file(CKPT, 3, False, 0)
    var x = heap_alloc[Scalar[DType.int32]](GB * GT)
    var y = heap_alloc[Scalar[DType.int32]](GB * GT)
    for i in range(GB * GT):
        x[unsafe_offset=i] = Int32((i * 7) % 8)
        y[unsafe_offset=i] = Int32((i * 29 + 11) % V)
    var ctx = DeviceContext(api="cpu")
    var NT = N16 + HEAD_N
    for idle in range(2):
        var model = GPT2["cpu", 1](CKPT, rank=0, zero_stage=0, ctx=ctx)
        model.head_tied = idle == 1
        var g = heap_alloc[Float32](NT)
        _ = _fwd_bwd(model, x, y, g)
        var body = Float64(0.0)
        var head = Float64(0.0)
        for i in range(N16):
            body += Float64(g[unsafe_offset=i]) ** 2
        for i in range(HEAD_N):
            head += Float64(g[unsafe_offset=N16 + i]) ** 2
        var norm = Float64(model.calculate_grad_norm())
        var want = (body + head) ** 0.5 if idle == 0 else body**0.5
        var other = body**0.5 if idle == 0 else (body + head) ** 0.5
        assert_true(head == 0.0 if idle == 1 else head > 0.0, "head grad")
        if idle == 0:
            assert_true(abs(other - want) > 1e-2 * want, "norms not separable")
        assert_true(
            abs(norm - want) <= 1e-5 * want,
            "grad norm " + String(norm) + " != " + String(want),
        )
        g.unsafe_free()
        _ = model^
    _remove(CKPT)
    x.unsafe_free()
    y.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    _remove(CKPT)
