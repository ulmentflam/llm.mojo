from std.memory import alloc
from std.python import Python
from std.testing import assert_equal, assert_almost_equal, TestSuite

from llmm.dataloader import DataLoader
from llmm.memory import MutMemPtr
from llmm.checkpointing import (
    CheckpointConfig,
    TrainingState,
    write_model_checkpoint,
    read_model_checkpoint,
    peek_model_header,
    write_state_checkpoint,
    read_state_checkpoint,
    make_training_state,
    restore_dataloader_state,
    MODEL_MAGIC,
    STATE_MAGIC,
    VERSION_FP32,
)


# ===----------------------------------------------------------------------=== #
# Test helpers
# ===----------------------------------------------------------------------=== #


# A small but structurally complete GPT-2 config for fast round-trip tests.
def _tiny_config() -> CheckpointConfig:
    return CheckpointConfig(
        max_seq_len=4,
        vocab_size=8,
        num_layer=2,
        num_heads=2,
        channels=4,
        padded_vocab_size=8,
    )


def _fill_pattern(ptr: MutMemPtr[DType.float32], n: Int, scale: Float32):
    for i in range(n):
        ptr.store(i, Float32(i) * scale - 1.0)


def _remove(path: String):
    try:
        var os = Python.import_module("os")
        _ = os.remove(path)
    except:
        pass


# ===----------------------------------------------------------------------=== #
# CheckpointConfig
# ===----------------------------------------------------------------------=== #


def test_num_parameters() raises:
    var config = _tiny_config()
    # Hand-computed sum of the 16 parameter-tensor sizes for the tiny config.
    assert_equal(config.num_parameters(), 544)


def test_config_equality() raises:
    var a = _tiny_config()
    var b = _tiny_config()
    assert_equal(a == b, True)
    var c = CheckpointConfig(4, 8, 2, 2, 4, 16)  # different padded_vocab_size
    assert_equal(a == c, False)
    assert_equal(a != c, True)


# ===----------------------------------------------------------------------=== #
# Model checkpoint round-trip
# ===----------------------------------------------------------------------=== #


def test_model_checkpoint_roundtrip() raises:
    var path = String("test_ckpt_model.bin")
    var config = _tiny_config()
    var n = config.num_parameters()

    var params = alloc[Float32](n)
    _fill_pattern(params, n, 0.5)

    write_model_checkpoint(path, config, params, n)

    # The header alone should recover the config (and therefore num_parameters).
    var header = peek_model_header(path)
    assert_equal(header.version, VERSION_FP32)
    assert_equal(header.config == config, True)
    assert_equal(header.config.num_parameters(), n)

    # Read the params back into a fresh buffer and compare element-wise.
    var restored = alloc[Float32](n)
    var read_header = read_model_checkpoint(path, restored, n)
    assert_equal(read_header.config == config, True)
    for i in range(n):
        assert_almost_equal(restored.load(i), params.load(i), atol=0.0)

    params.free()
    restored.free()
    _remove(path)


def test_model_checkpoint_bad_magic() raises:
    # A state checkpoint has STATE_MAGIC, not MODEL_MAGIC, so reading it as a
    # model checkpoint must raise.
    assert_equal(MODEL_MAGIC != STATE_MAGIC, True)

    var path = String("test_ckpt_badmagic.bin")
    var m = alloc[Float32](4)
    var v = alloc[Float32](4)
    for i in range(4):
        m.store(i, 1.0)
        v.store(i, 2.0)
    var state = TrainingState(0, 1, 0, 0, 0, 0, 0, 0, 0)
    write_state_checkpoint(path, state, m, v, 4)

    var out = alloc[Float32](4)
    var raised = False
    try:
        _ = read_model_checkpoint(path, out, 4)
    except:
        raised = True
    assert_equal(raised, True)

    m.free()
    v.free()
    out.free()
    _remove(path)


def test_model_checkpoint_buffer_too_small() raises:
    var path = String("test_ckpt_small.bin")
    var config = _tiny_config()
    var n = config.num_parameters()
    var params = alloc[Float32](n)
    _fill_pattern(params, n, 0.25)
    write_model_checkpoint(path, config, params, n)

    var out = alloc[Float32](n - 1)
    var raised = False
    try:
        _ = read_model_checkpoint(path, out, n - 1)
    except:
        raised = True
    assert_equal(raised, True)

    params.free()
    out.free()
    _remove(path)


# ===----------------------------------------------------------------------=== #
# State checkpoint round-trip
# ===----------------------------------------------------------------------=== #


def test_state_checkpoint_roundtrip() raises:
    var path = String("test_ckpt_state.bin")
    var n = 17  # Deliberately not a power of two.

    var m = alloc[Float32](n)
    var v = alloc[Float32](n)
    _fill_pattern(m, n, 0.1)
    _fill_pattern(v, n, 0.3)

    var state = TrainingState(
        step=42,
        num_processes=4,
        process_rank=2,
        use_master_weights=0,
        should_shuffle=1,
        sampler_rng_state=UInt64(1337),
        shuffle_rng_state=UInt64(0xDEADBEEFCAFEF00D),
        current_shard_idx=3,
        current_sample_idx=11,
    )

    write_state_checkpoint(path, state, m, v, n)

    var m_out = alloc[Float32](n)
    var v_out = alloc[Float32](n)
    var restored = read_state_checkpoint(path, m_out, v_out, n)

    # Scalar header fields, including the 64-bit RNG slots.
    assert_equal(restored.step, 42)
    assert_equal(restored.num_processes, 4)
    assert_equal(restored.process_rank, 2)
    assert_equal(restored.use_master_weights, 0)
    assert_equal(restored.should_shuffle, 1)
    assert_equal(restored.sampler_rng_state, UInt64(1337))
    assert_equal(restored.shuffle_rng_state, UInt64(0xDEADBEEFCAFEF00D))
    assert_equal(restored.current_shard_idx, 3)
    assert_equal(restored.current_sample_idx, 11)

    # Moment payloads.
    for i in range(n):
        assert_almost_equal(m_out.load(i), m.load(i), atol=0.0)
        assert_almost_equal(v_out.load(i), v.load(i), atol=0.0)

    m.free()
    v.free()
    m_out.free()
    v_out.free()
    _remove(path)


def test_state_checkpoint_bad_magic() raises:
    # A model checkpoint must not read back as a state checkpoint.
    var path = String("test_ckpt_state_badmagic.bin")
    var config = _tiny_config()
    var n = config.num_parameters()
    var params = alloc[Float32](n)
    _fill_pattern(params, n, 0.5)
    write_model_checkpoint(path, config, params, n)

    var m_out = alloc[Float32](4)
    var v_out = alloc[Float32](4)
    var raised = False
    try:
        _ = read_state_checkpoint(path, m_out, v_out, 4)
    except:
        raised = True
    assert_equal(raised, True)

    params.free()
    m_out.free()
    v_out.free()
    _remove(path)


# ===----------------------------------------------------------------------=== #
# DataLoader integration
# ===----------------------------------------------------------------------=== #


def _setup_loader_data() raises:
    var sys = Python.import_module("sys")
    sys.path.append(".")
    var utils = Python.import_module("data.utils")
    var py_tokens = Python.evaluate("[i for i in range(100)]")
    utils.write_datafile("test_ckpt_loader.bin", py_tokens, "gpt-2")


def test_dataloader_capture_restore() raises:
    _setup_loader_data()

    # Run a loader forward a few batches, then snapshot its position.
    var loader = DataLoader("test_ckpt_loader.bin", batch_size=2, seq_len=4)
    loader.next_batch()
    loader.next_batch()
    loader.next_batch()

    var path = String("test_ckpt_loader_state.bin")
    var state = make_training_state(loader, step=3, sampler_rng_state=UInt64(7))

    # Persist the (empty) optimizer payload alongside the captured position so
    # the round-trip exercises the on-disk header fields, not just the struct.
    var m = alloc[Float32](1)
    var v = alloc[Float32](1)
    m.store(0, 0.0)
    v.store(0, 0.0)
    write_state_checkpoint(path, state, m, v, 1)

    # What batch would the original loader produce next?
    loader.next_batch()
    var expected_first = loader.inputs.load(0)
    var expected_last = loader.inputs.load(7)

    # Restore into a fresh loader from the persisted state.
    var m_out = alloc[Float32](1)
    var v_out = alloc[Float32](1)
    var restored = read_state_checkpoint(path, m_out, v_out, 1)
    assert_equal(restored.step, 3)
    assert_equal(restored.current_sample_idx, 3)

    var resumed = DataLoader("test_ckpt_loader.bin", batch_size=2, seq_len=4)
    restore_dataloader_state(resumed, restored)
    resumed.next_batch()

    assert_equal(resumed.inputs.load(0), expected_first)
    assert_equal(resumed.inputs.load(7), expected_last)

    m.free()
    v.free()
    m_out.free()
    v_out.free()
    loader.close()
    resumed.close()
    _remove(path)


def _cleanup_loader_data():
    _remove("test_ckpt_loader.bin")


def main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        _cleanup_loader_data()
        raise e^
    _cleanup_loader_data()
