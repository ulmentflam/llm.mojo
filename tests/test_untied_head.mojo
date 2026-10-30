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
            model.params_memory[unsafe_offset=N16 + i], Float32(1000 + i)
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    _remove(CKPT)
