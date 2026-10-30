from std.python import Python
from std.testing import assert_equal, assert_almost_equal, TestSuite

from llmm.dataloader import DataLoader
from llmm.memory import MutMemPtr, heap_alloc
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
    ARCH_FLAG_ROTARY,
    STATE_FLAGS_SLOT,
    MODEL_FLAGS_SLOT,
)
from train_gpt2 import check_arch_flags, ARCH_FLAGS


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
        ptr.unsafe_store(i, Float32(i) * scale - 1.0)


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

    var params = heap_alloc[Float32](n)
    _fill_pattern(params, n, 0.5)

    write_model_checkpoint(path, config, params, n)

    # The header alone should recover the config (and therefore num_parameters).
    var header = peek_model_header(path)
    assert_equal(header.version, VERSION_FP32)
    assert_equal(header.config == config, True)
    assert_equal(header.config.num_parameters(), n)

    var restored = heap_alloc[Float32](n)
    var read_header = read_model_checkpoint(path, restored, n)
    assert_equal(read_header.config == config, True)
    for i in range(n):
        assert_almost_equal(
            restored.unsafe_load(i), params.unsafe_load(i), atol=0.0
        )

    params.unsafe_free()
    restored.unsafe_free()
    _remove(path)


def test_model_checkpoint_bad_magic() raises:
    # A state checkpoint has STATE_MAGIC, not MODEL_MAGIC, so reading it as a
    # model checkpoint must raise.
    assert_equal(MODEL_MAGIC != STATE_MAGIC, True)

    var path = String("test_ckpt_badmagic.bin")
    var m = heap_alloc[Float32](4)
    var v = heap_alloc[Float32](4)
    for i in range(4):
        m.unsafe_store(i, 1.0)
        v.unsafe_store(i, 2.0)
    var state = TrainingState(0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    write_state_checkpoint(path, state, m, v, 4)

    var out = heap_alloc[Float32](4)
    var raised = False
    try:
        _ = read_model_checkpoint(path, out, 4)
    except:
        raised = True
    assert_equal(raised, True)

    m.unsafe_free()
    v.unsafe_free()
    out.unsafe_free()
    _remove(path)


def test_model_checkpoint_buffer_too_small() raises:
    var path = String("test_ckpt_small.bin")
    var config = _tiny_config()
    var n = config.num_parameters()
    var params = heap_alloc[Float32](n)
    _fill_pattern(params, n, 0.25)
    write_model_checkpoint(path, config, params, n)

    var out = heap_alloc[Float32](n - 1)
    var raised = False
    try:
        _ = read_model_checkpoint(path, out, n - 1)
    except:
        raised = True
    assert_equal(raised, True)

    params.unsafe_free()
    out.unsafe_free()
    _remove(path)


# ===----------------------------------------------------------------------=== #
# State checkpoint round-trip
# ===----------------------------------------------------------------------=== #


def test_state_checkpoint_roundtrip() raises:
    var path = String("test_ckpt_state.bin")
    var n = 17  # Deliberately not a power of two.

    var m = heap_alloc[Float32](n)
    var v = heap_alloc[Float32](n)
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
        optimizer_kind=1,
    )

    write_state_checkpoint(path, state, m, v, n)

    var m_out = heap_alloc[Float32](n)
    var v_out = heap_alloc[Float32](n)
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
    assert_equal(restored.optimizer_kind, 1)

    # Moment payloads.
    for i in range(n):
        assert_almost_equal(m_out.unsafe_load(i), m.unsafe_load(i), atol=0.0)
        assert_almost_equal(v_out.unsafe_load(i), v.unsafe_load(i), atol=0.0)

    m.unsafe_free()
    v.unsafe_free()
    m_out.unsafe_free()
    v_out.unsafe_free()
    _remove(path)


def test_state_checkpoint_world_size_mismatch_raises() raises:
    # A state file written by rank 1 of 2 must not load at world size 1 (it
    # used to read a silent prefix), nor into the wrong rank; the matching
    # world/rank and the unchecked default still load.
    var path = String("test_ckpt_state_world.bin")
    var n = 8
    var m = heap_alloc[Float32](n)
    var v = heap_alloc[Float32](n)
    _fill_pattern(m, n, 0.1)
    _fill_pattern(v, n, 0.3)
    var state = TrainingState(
        step=5,
        num_processes=2,
        process_rank=1,
        use_master_weights=0,
        should_shuffle=0,
        sampler_rng_state=UInt64(0),
        shuffle_rng_state=UInt64(0),
        current_shard_idx=0,
        current_sample_idx=0,
        optimizer_kind=0,
    )
    write_state_checkpoint(path, state, m, v, n)
    var m_out = heap_alloc[Float32](n)
    var v_out = heap_alloc[Float32](n)

    var raised_world = False
    try:
        _ = read_state_checkpoint(
            path, m_out, v_out, n, expect_num_processes=1, expect_process_rank=0
        )
    except e:
        raised_world = "world size" in String(e)
    assert_equal(raised_world, True)

    var raised_rank = False
    try:
        _ = read_state_checkpoint(
            path, m_out, v_out, n, expect_num_processes=2, expect_process_rank=0
        )
    except e:
        raised_rank = "rank 1 of 2" in String(e)
    assert_equal(raised_rank, True)

    var ok = read_state_checkpoint(
        path, m_out, v_out, n, expect_num_processes=2, expect_process_rank=1
    )
    assert_equal(ok.step, 5)
    var unchecked = read_state_checkpoint(path, m_out, v_out, n)
    assert_equal(unchecked.num_processes, 2)

    m.unsafe_free()
    v.unsafe_free()
    m_out.unsafe_free()
    v_out.unsafe_free()
    _remove(path)


def test_state_checkpoint_shard_size_mismatch_raises() raises:
    # A ZeRO-0 state file (full-size moments) resumed at ZeRO>=1 at the SAME
    # world size and rank passes the header checks, and used to load a silent
    # prefix of m and v. The trailing-bytes check must refuse it; a short file
    # (the reverse direction) already raised in read_and_copy.
    var path = String("test_ckpt_state_shard.bin")
    var n = 8
    var m = heap_alloc[Float32](n)
    var v = heap_alloc[Float32](n)
    _fill_pattern(m, n, 0.1)
    _fill_pattern(v, n, 0.3)
    var state = TrainingState(
        step=5,
        num_processes=2,
        process_rank=0,
        use_master_weights=0,
        should_shuffle=0,
        sampler_rng_state=UInt64(0),
        shuffle_rng_state=UInt64(0),
        current_shard_idx=0,
        current_sample_idx=0,
        optimizer_kind=0,
    )
    write_state_checkpoint(path, state, m, v, n)
    # 2n: the short-file probe below reads n + 1 elements of m before the v
    # read runs out of bytes.
    var m_out = heap_alloc[Float32](2 * n)
    var v_out = heap_alloc[Float32](2 * n)

    var raised_long = False
    try:
        _ = read_state_checkpoint(
            path,
            m_out,
            v_out,
            n // 2,
            expect_num_processes=2,
            expect_process_rank=0,
        )
    except e:
        raised_long = "bytes, expected" in String(e)
    assert_equal(raised_long, True)

    var raised_short = False
    try:
        _ = read_state_checkpoint(path, m_out, v_out, n + 1)
    except e:
        raised_short = "bytes, expected" in String(e)
    assert_equal(raised_short, True)

    var ok = read_state_checkpoint(
        path, m_out, v_out, n, expect_num_processes=2, expect_process_rank=0
    )
    assert_equal(ok.step, 5)


def test_state_checkpoint_compact_v_roundtrip() raises:
    """Muon layout: m is full-size, v is compact (fewer elements)."""
    var path = String("test_ckpt_state_compact.bin")
    var n = 17
    var nv = 6
    var m = heap_alloc[Float32](n)
    var v = heap_alloc[Float32](nv)
    _fill_pattern(m, n, 0.1)
    _fill_pattern(v, nv, 0.3)
    var state = TrainingState(
        step=7,
        num_processes=1,
        process_rank=0,
        use_master_weights=0,
        should_shuffle=0,
        sampler_rng_state=UInt64(5),
        shuffle_rng_state=UInt64(6),
        current_shard_idx=0,
        current_sample_idx=1,
        optimizer_kind=1,
    )
    write_state_checkpoint(path, state, m, v, n, nv)
    var m_out = heap_alloc[Float32](n)
    var v_out = heap_alloc[Float32](nv)
    var restored = read_state_checkpoint(path, m_out, v_out, n, nv)
    assert_equal(restored.step, 7)
    assert_equal(restored.optimizer_kind, 1)
    for i in range(n):
        assert_almost_equal(m_out.unsafe_load(i), m.unsafe_load(i), atol=0.0)
    for i in range(nv):
        assert_almost_equal(v_out.unsafe_load(i), v.unsafe_load(i), atol=0.0)
    m.unsafe_free()
    v.unsafe_free()
    m_out.unsafe_free()
    v_out.unsafe_free()
    _remove(path)


def _owner_state(zero_stage: Int, rank: Int, S: Int) -> TrainingState:
    return TrainingState(
        step=3,
        num_processes=2,
        process_rank=rank,
        use_master_weights=0,
        should_shuffle=0,
        sampler_rng_state=UInt64(1),
        shuffle_rng_state=UInt64(2),
        current_shard_idx=0,
        current_sample_idx=4,
        optimizer_kind=1,
        zero_stage=zero_stage,
        muon_ownership_version=1,
        muon_shard_elems=S,
    )


def _raises_with(
    path: String,
    m_out: MutMemPtr[DType.float32],
    v_out: MutMemPtr[DType.float32],
    mu_out: MutMemPtr[DType.float32],
    n: Int,
    nv: Int,
    S: Int,
    owner: Bool,
    needle: String,
) -> Bool:
    try:
        if owner:
            _ = read_state_checkpoint(
                path,
                m_out,
                v_out,
                n,
                v_num_parameters=nv,
                expect_num_processes=2,
                expect_process_rank=1,
                muon_m_out=Optional[MutMemPtr[DType.float32]](mu_out),
                expect_muon_shard_elems=S,
            )
        else:
            _ = read_state_checkpoint(
                path,
                m_out,
                v_out,
                n,
                v_num_parameters=nv,
                expect_num_processes=2,
                expect_process_rank=1,
            )
    except e:
        return needle in String(e)
    return False


def test_state_checkpoint_sharded_muon_t6() raises:
    """T6 (S4), file level: owner-local muon_m roundtrips bit-exactly, and a
    resume across ownership version, S, owner/replicated layout, or a file one
    element longer or shorter raises. Runs for every ZeRO stage that shares the
    owner layout (S2/S3 only need to be added to `stages`)."""
    var stages: List[Int] = [1, 2, 3]
    for si in range(len(stages)):
        var path = String("test_ckpt_state_owner.bin")
        # S7: under sharded Muon m, v are both compact (A_r elements).
        var n = 5
        var nv = 5
        var S = 12
        var m = heap_alloc[Float32](n)
        var v = heap_alloc[Float32](nv)
        var mu = heap_alloc[Float32](S)
        _fill_pattern(m, n, 0.1)
        _fill_pattern(v, nv, 0.3)
        _fill_pattern(mu, S, 0.7)
        var st = _owner_state(stages[si], 1, S)
        write_state_checkpoint(
            path,
            st,
            m,
            v,
            n,
            nv,
            muon_m_memory=Optional[MutMemPtr[DType.float32]](mu),
        )
        var m_out = heap_alloc[Float32](n)
        var v_out = heap_alloc[Float32](nv)
        var mu_out = heap_alloc[Float32](S)
        var r = read_state_checkpoint(
            path,
            m_out,
            v_out,
            n,
            v_num_parameters=nv,
            expect_num_processes=2,
            expect_process_rank=1,
            muon_m_out=Optional[MutMemPtr[DType.float32]](mu_out),
            expect_muon_shard_elems=S,
        )
        assert_equal(r.muon_ownership_version, 1)
        assert_equal(r.muon_shard_elems, S)
        assert_equal(r.zero_stage, stages[si])
        for i in range(S):
            assert_almost_equal(
                mu_out.unsafe_load(i), mu.unsafe_load(i), atol=0.0
            )
        for i in range(nv):
            assert_almost_equal(
                v_out.unsafe_load(i), v.unsafe_load(i), atol=0.0
            )
        # Wrong S, wrong layout expectation (replicated reader on owner file).
        assert_equal(
            _raises_with(
                path, m_out, v_out, mu_out, n, nv, S + 4, True, "shard_elems"
            ),
            True,
        )
        assert_equal(
            _raises_with(
                path, m_out, v_out, mu_out, n, nv, S, False, "replicated layout"
            ),
            True,
        )
        # Different ownership version in the header.
        var bad = _owner_state(stages[si], 1, S)
        bad.muon_ownership_version = 2
        write_state_checkpoint(
            path,
            bad,
            m,
            v,
            n,
            nv,
            muon_m_memory=Optional[MutMemPtr[DType.float32]](mu),
        )
        assert_equal(
            _raises_with(
                path,
                m_out,
                v_out,
                mu_out,
                n,
                nv,
                S,
                True,
                "muon_ownership_version=2",
            ),
            True,
        )
        # A file with no owner payload (ZeRO-0 / W=1 Muon, AdamW) at an owner run.
        var plain = _owner_state(0, 1, 0)
        plain.muon_ownership_version = 0
        write_state_checkpoint(path, plain, m, v, n, nv)
        assert_equal(
            _raises_with(
                path,
                m_out,
                v_out,
                mu_out,
                n,
                nv,
                S,
                True,
                "no owner-local Muon momentum",
            ),
            True,
        )
        # Exact length: one element (4 bytes) longer / shorter, both directions.
        write_state_checkpoint(
            path,
            st,
            m,
            v,
            n,
            nv,
            muon_m_memory=Optional[MutMemPtr[DType.float32]](mu),
        )
        assert_equal(
            _raises_with(
                path,
                m_out,
                v_out,
                mu_out,
                n,
                nv + 1,
                S,
                True,
                "bytes, expected",
            ),
            True,
        )
        assert_equal(
            _raises_with(
                path, m_out, v_out, mu_out, n, nv, S + 1, True, "shard_elems"
            ),
            True,
        )
        write_state_checkpoint(
            path,
            st,
            m,
            v,
            n,
            nv + 1,
            muon_m_memory=Optional[MutMemPtr[DType.float32]](mu),
        )  # a longer v blob
        assert_equal(
            _raises_with(
                path, m_out, v_out, mu_out, n, nv, S, True, "bytes, expected"
            ),
            True,
        )
        # A pre-S7 owner file (flat shard-sized m, 8 elements) at a trimmed
        # run (m = A_r = 5) must be refused by length, not read as a prefix.
        var m_old = heap_alloc[Float32](8)
        _fill_pattern(m_old, 8, 0.1)
        write_state_checkpoint(
            path,
            st,
            m_old,
            v,
            8,
            nv,
            muon_m_memory=Optional[MutMemPtr[DType.float32]](mu),
        )
        assert_equal(
            _raises_with(
                path, m_out, v_out, mu_out, n, nv, S, True, "bytes, expected"
            ),
            True,
        )
        m_old.unsafe_free()
        write_state_checkpoint(
            path,
            st,
            m,
            v,
            n,
            nv,
            muon_m_memory=Optional[MutMemPtr[DType.float32]](mu),
        )
        # Truncated by one element.
        var short = open(path, "r").read_bytes()
        var f = open(path, "w")
        f.write_bytes(short[: len(short) - 4])
        f.close()
        assert_equal(
            _raises_with(
                path,
                m_out,
                v_out,
                mu_out,
                n,
                nv + 1,
                S,
                True,
                "bytes, expected",
            ),
            True,
        )
        m.unsafe_free()
        v.unsafe_free()
        mu.unsafe_free()
        m_out.unsafe_free()
        v_out.unsafe_free()
        mu_out.unsafe_free()
        _remove(path)


def test_state_header_plain_files_unchanged() raises:
    """AdamW / replicated-Muon files carry zeros in slots 41-43 (byte-identical
    to the pre-S4 layout) and are exactly header + (m + v) * 4 bytes."""
    var path = String("test_ckpt_state_plain_hdr.bin")
    var n = 6
    var m = heap_alloc[Float32](n)
    var v = heap_alloc[Float32](n)
    _fill_pattern(m, n, 0.1)
    _fill_pattern(v, n, 0.3)
    var st = _owner_state(0, 0, 0)
    st.muon_ownership_version = 0
    st.num_processes = 1
    write_state_checkpoint(path, st, m, v, n)
    var bytes = open(path, "r").read_bytes()
    assert_equal(len(bytes), 256 * 4 + 2 * n * 4)
    for k in range(41 * 4, 44 * 4):
        assert_equal(Int(bytes[k]), 0)
    m.unsafe_free()
    v.unsafe_free()
    _remove(path)


def test_state_checkpoint_bad_magic() raises:
    # A model checkpoint must not read back as a state checkpoint.
    var path = String("test_ckpt_state_badmagic.bin")
    var config = _tiny_config()
    var n = config.num_parameters()
    var params = heap_alloc[Float32](n)
    _fill_pattern(params, n, 0.5)
    write_model_checkpoint(path, config, params, n)

    var m_out = heap_alloc[Float32](4)
    var v_out = heap_alloc[Float32](4)
    var raised = False
    try:
        _ = read_state_checkpoint(path, m_out, v_out, 4)
    except:
        raised = True
    assert_equal(raised, True)

    params.unsafe_free()
    m_out.unsafe_free()
    v_out.unsafe_free()
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

    var loader = DataLoader("test_ckpt_loader.bin", batch_size=2, seq_len=4)
    loader.next_batch()
    loader.next_batch()
    loader.next_batch()

    var path = String("test_ckpt_loader_state.bin")
    var state = make_training_state(loader, step=3, sampler_rng_state=UInt64(7))

    # Persist the (empty) optimizer payload alongside the captured position so
    # the round-trip exercises the on-disk header fields, not just the struct.
    var m = heap_alloc[Float32](1)
    var v = heap_alloc[Float32](1)
    m.unsafe_store(0, 0.0)
    v.unsafe_store(0, 0.0)
    write_state_checkpoint(path, state, m, v, 1)

    # What batch would the original loader produce next?
    loader.next_batch()
    var expected_first = loader.inputs.unsafe_load(0)
    var expected_last = loader.inputs.unsafe_load(7)

    var m_out = heap_alloc[Float32](1)
    var v_out = heap_alloc[Float32](1)
    var restored = read_state_checkpoint(path, m_out, v_out, 1)
    assert_equal(restored.step, 3)
    assert_equal(restored.current_sample_idx, 3)

    var resumed = DataLoader("test_ckpt_loader.bin", batch_size=2, seq_len=4)
    restore_dataloader_state(resumed, restored)
    resumed.next_batch()

    assert_equal(resumed.inputs.unsafe_load(0), expected_first)
    assert_equal(resumed.inputs.unsafe_load(7), expected_last)

    m.unsafe_free()
    v.unsafe_free()
    m_out.unsafe_free()
    v_out.unsafe_free()
    loader.close()
    resumed.close()
    _remove(path)


def test_dataloader_capture_restore_shuffled() raises:
    """The same round trip with shuffling ON -- the path a real run uses.

    `test_dataloader_capture_restore` above builds both loaders with
    `should_shuffle` left at its default of False, so it only ever exercises
    the trivial "seek to current_sample_idx" branch of
    `restore_dataloader_state`. The branch that matters for a real resume --
    reseeding MT19937 and replaying `shuffle_rng_state - intra_draws` draws so
    the permutation lands exactly where it left off (llmm/checkpointing.mojo,
    `if loader.should_shuffle and state.should_shuffle == 1`) -- was never
    entered by any test.

    That branch is what makes commit 15b8330's batch-order parity with llm.c
    survive a restart, and this pipeline restarts on every supervisor event.
    A wrong replay count produces a valid-looking but different batch order
    after every resume, silently.

    VALIDATED AGAINST THE BUG. Dropping the `- intra_draws` correction from
    `restore_dataloader_state`'s replay count and rebuilding: this test fails
    (`left: 56, right: 80` -- the resumed loader lands on a different sample),
    while `test_dataloader_capture_restore` above still passes, because it
    never enters the branch at all. `llmm/checkpointing.mojo` was restored and
    verified byte-identical to HEAD afterwards.
    """
    _setup_loader_data()

    var loader = DataLoader(
        "test_ckpt_loader.bin", batch_size=2, seq_len=4, should_shuffle=True
    )
    loader.next_batch()
    loader.next_batch()
    loader.next_batch()

    var path = String("test_ckpt_loader_shuffled_state.bin")
    var state = make_training_state(loader, step=3, sampler_rng_state=UInt64(7))
    var m = heap_alloc[Float32](1)
    var v = heap_alloc[Float32](1)
    m.unsafe_store(0, 0.0)
    v.unsafe_store(0, 0.0)
    write_state_checkpoint(path, state, m, v, 1)

    # Ground truth: what the ORIGINAL, uninterrupted shuffling loader produces
    # next. The resumed loader has to reproduce this exactly, which it can only
    # do by replaying the right number of draws off the same seed.
    loader.next_batch()
    var expected_first = loader.inputs.unsafe_load(0)
    var expected_last = loader.inputs.unsafe_load(7)

    var m_out = heap_alloc[Float32](1)
    var v_out = heap_alloc[Float32](1)
    var restored = read_state_checkpoint(path, m_out, v_out, 1)
    assert_equal(restored.step, 3)
    assert_equal(restored.should_shuffle, 1)

    var resumed = DataLoader(
        "test_ckpt_loader.bin", batch_size=2, seq_len=4, should_shuffle=True
    )
    restore_dataloader_state(resumed, restored)
    resumed.next_batch()

    assert_equal(resumed.inputs.unsafe_load(0), expected_first)
    assert_equal(resumed.inputs.unsafe_load(7), expected_last)

    m.unsafe_free()
    v.unsafe_free()
    m_out.unsafe_free()
    v_out.unsafe_free()
    loader.close()
    resumed.close()
    _remove(path)


# ===----------------------------------------------------------------------=== #
# Architecture flags word (rotary): state header slot 44 / model header slot 20
# ===----------------------------------------------------------------------=== #


def _header_word(path: String, slot: Int) raises -> Int:
    var np = Python.import_module("numpy")
    var h = np.fromfile(path, dtype=np.int32, count=256)
    return Int(py=h[slot])


def _tiny_state(flags: Int) raises -> TrainingState:
    return TrainingState(
        step=7,
        num_processes=1,
        process_rank=0,
        use_master_weights=0,
        should_shuffle=0,
        sampler_rng_state=UInt64(1),
        shuffle_rng_state=UInt64(2),
        current_shard_idx=0,
        current_sample_idx=0,
        optimizer_kind=0,
        flags=flags,
    )


def test_state_flags_roundtrip_and_default_stays_version_1() raises:
    var path = String("test_ckpt_state_flags.bin")
    var n = 5
    var m = heap_alloc[Float32](n)
    var v = heap_alloc[Float32](n)
    _fill_pattern(m, n, 0.1)
    _fill_pattern(v, n, 0.2)
    var m_out = heap_alloc[Float32](n)
    var v_out = heap_alloc[Float32](n)

    # Flagged: version 2, slot 44 carries the bit, and it reads back.
    write_state_checkpoint(path, _tiny_state(ARCH_FLAG_ROTARY), m, v, n)
    assert_equal(_header_word(path, 1), 2)
    assert_equal(_header_word(path, STATE_FLAGS_SLOT), ARCH_FLAG_ROTARY)
    var restored = read_state_checkpoint(path, m_out, v_out, n)
    assert_equal(restored.flags, ARCH_FLAG_ROTARY)
    assert_equal(restored.step, 7)

    # Default (flags 0): still the version-1 header with slot 44 untouched, so
    # default-build files are byte-compatible with files from before the word.
    write_state_checkpoint(path, _tiny_state(0), m, v, n)
    assert_equal(_header_word(path, 1), 1)
    assert_equal(_header_word(path, STATE_FLAGS_SLOT), 0)
    assert_equal(read_state_checkpoint(path, m_out, v_out, n).flags, 0)

    m.unsafe_free()
    v.unsafe_free()
    m_out.unsafe_free()
    v_out.unsafe_free()
    _remove(path)


def test_model_flags_roundtrip() raises:
    var path = String("test_ckpt_model_flags.bin")
    var cfg = _tiny_config()
    var n = cfg.num_parameters()
    var params = heap_alloc[Float32](n)
    _fill_pattern(params, n, 0.5)

    write_model_checkpoint(path, cfg, params, n, arch_flags=ARCH_FLAG_ROTARY)
    assert_equal(_header_word(path, MODEL_FLAGS_SLOT), ARCH_FLAG_ROTARY)
    assert_equal(peek_model_header(path).arch_flags, ARCH_FLAG_ROTARY)

    write_model_checkpoint(path, cfg, params, n)
    assert_equal(_header_word(path, MODEL_FLAGS_SLOT), 0)
    assert_equal(peek_model_header(path).arch_flags, 0)

    params.unsafe_free()
    _remove(path)


def _refused(got: Int, strict: Bool) -> Bool:
    try:
        check_arch_flags("x", got, strict)
    except:
        return True
    return False


def test_arch_flag_mismatch_is_refused_for_this_build() raises:
    # `ARCH_FLAGS` is this build's flags (1 under -D LLMM_ROTARY=1, else 0), so
    # the same test file pins both directions when run under either build
    # (`make test-mojo [MOJO_DEFINES="-D LLMM_ROTARY=1"]`).
    var other = 1 - ARCH_FLAGS
    assert_equal(_refused(ARCH_FLAGS, True), False)
    assert_equal(_refused(ARCH_FLAGS, False), False)
    # Resume (strict): a flag mismatch is refused in BOTH directions.
    assert_equal(_refused(other, True), True)
    # Initialising from a model .bin (non-strict): a flagged file is refused by
    # a build without the flag; an unflagged (pretrained) file is accepted by a
    # flagged build.
    assert_equal(_refused(other, False), ARCH_FLAGS == 0)


def _cleanup_loader_data():
    _remove("test_ckpt_loader.bin")


def main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        _cleanup_loader_data()
        raise e^
    _cleanup_loader_data()
