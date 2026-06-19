from std.testing import assert_equal, TestSuite
from std.python import Python
from llmm.dataloader import DataLoader


def setup_test_files() raises:
    # Set up python path so local modules like data.utils can be imported
    var sys = Python.import_module("sys")
    sys.path.append(".")

    # Use python interop to generate the token shard files
    var utils = Python.import_module("data.utils")

    # Generate mock tokens [0, 1, 2, ..., 99]
    var py_tokens = Python.evaluate("[i for i in range(100)]")

    # Write a GPT-2 token file (UInt16 tokens, magic = 20240520)
    utils.write_datafile("test_gpt2_data.bin", py_tokens, "gpt-2")

    # Write a Llama-3 token file (UInt32 tokens, magic = 20240801)
    utils.write_datafile("test_llama3_data.bin", py_tokens, "llama-3")

    # Generate shard files for testing multi-shard transitions
    var py_tokens_shard0 = Python.evaluate("[i for i in range(24)]")
    var py_tokens_shard1 = Python.evaluate("[i for i in range(100, 124)]")
    utils.write_datafile("test_shard_0.bin", py_tokens_shard0, "gpt-2")
    utils.write_datafile("test_shard_1.bin", py_tokens_shard1, "gpt-2")

    print("Test binary data files created successfully!")


def cleanup_test_files() raises:
    var os = Python.import_module("os")
    try:
        _ = os.remove("test_gpt2_data.bin")
    except:
        pass
    try:
        _ = os.remove("test_llama3_data.bin")
    except:
        pass
    try:
        _ = os.remove("test_shard_0.bin")
    except:
        pass
    try:
        _ = os.remove("test_shard_1.bin")
    except:
        pass


def test_dataloader_gpt2() raises:
    # batch_size = 2, seq_len = 4
    # With 100 tokens, we should have plenty of batches
    var loader = DataLoader("test_gpt2_data.bin", batch_size=2, seq_len=4)

    # Check loaded metadata
    assert_equal(loader.magic, 20240520)
    assert_equal(loader.token_size, 2)
    assert_equal(loader.num_tokens, 100)

    # First batch:
    # Process rank 0, batch size 2, seq len 4
    # inputs should be:
    # Row 0: [0, 1, 2, 3]
    # Row 1: [4, 5, 6, 7]
    # targets should be:
    # Row 0: [1, 2, 3, 4]
    # Row 1: [5, 6, 7, 8]
    loader.next_batch()

    # Verify inputs
    assert_equal(loader.inputs.load(0), 0)
    assert_equal(loader.inputs.load(1), 1)
    assert_equal(loader.inputs.load(2), 2)
    assert_equal(loader.inputs.load(3), 3)
    assert_equal(loader.inputs.load(4), 4)
    assert_equal(loader.inputs.load(5), 5)
    assert_equal(loader.inputs.load(6), 6)
    assert_equal(loader.inputs.load(7), 7)

    # Verify targets
    assert_equal(loader.targets.load(0), 1)
    assert_equal(loader.targets.load(1), 2)
    assert_equal(loader.targets.load(2), 3)
    assert_equal(loader.targets.load(3), 4)
    assert_equal(loader.targets.load(4), 5)
    assert_equal(loader.targets.load(5), 6)
    assert_equal(loader.targets.load(6), 7)
    assert_equal(loader.targets.load(7), 8)

    # Second batch (advances by batch_size * seq_len = 8 tokens)
    # inputs should be:
    # Row 0: [8, 9, 10, 11]
    # Row 1: [12, 13, 14, 15]
    loader.next_batch()
    assert_equal(loader.inputs.load(0), 8)
    assert_equal(loader.inputs.load(7), 15)
    assert_equal(loader.targets.load(0), 9)
    assert_equal(loader.targets.load(7), 16)


def test_dataloader_llama3() raises:
    var loader = DataLoader("test_llama3_data.bin", batch_size=3, seq_len=2)

    # Check loaded metadata
    assert_equal(loader.magic, 20240801)
    assert_equal(loader.token_size, 4)
    assert_equal(loader.num_tokens, 100)

    # First batch: batch_size=3, seq_len=2
    # inputs:
    # Row 0: [0, 1]
    # Row 1: [2, 3]
    # Row 2: [4, 5]
    loader.next_batch()

    assert_equal(loader.inputs.load(0), 0)
    assert_equal(loader.inputs.load(1), 1)
    assert_equal(loader.inputs.load(2), 2)
    assert_equal(loader.inputs.load(3), 3)
    assert_equal(loader.inputs.load(4), 4)
    assert_equal(loader.inputs.load(5), 5)

    assert_equal(loader.targets.load(0), 1)
    assert_equal(loader.targets.load(1), 2)
    assert_equal(loader.targets.load(2), 3)
    assert_equal(loader.targets.load(3), 4)
    assert_equal(loader.targets.load(4), 5)
    assert_equal(loader.targets.load(5), 6)


def test_dataloader_shuffle() raises:
    # With B=2, T=3, total_batch_size = 6 tokens.
    # Shard has 100 tokens. (100 - 1) // 6 = 16 samples.
    var loader = DataLoader(
        "test_gpt2_data.bin", batch_size=2, seq_len=3, should_shuffle=True
    )

    assert_equal(len(loader.intra_shard_indices), 16)

    # Run next_batch a few times to ensure indices are loaded and do not crash
    loader.next_batch()
    loader.next_batch()

    # Verify that the values loaded are within [0, 100]
    for i in range(6):
        var val = loader.inputs.load(i)
        assert_equal(val >= 0 and val < 100, True)


def test_dataloader_shards() raises:
    # Matches "test_shard_*.bin". Glob result matches test_shard_0.bin and test_shard_1.bin
    var loader = DataLoader("test_shard_*.bin", batch_size=2, seq_len=3)

    assert_equal(len(loader.files), 2)
    assert_equal(loader.current_shard_idx, 0)
    assert_equal(loader.shard_num_samples, 3)  # (24-1)//6 = 3

    # Sample 0
    loader.next_batch()
    assert_equal(loader.inputs.load(0), 0)
    assert_equal(loader.inputs.load(5), 5)

    # Sample 1
    loader.next_batch()
    assert_equal(loader.inputs.load(0), 6)
    assert_equal(loader.inputs.load(5), 11)

    # Sample 2
    loader.next_batch()
    assert_equal(loader.inputs.load(0), 12)
    assert_equal(loader.inputs.load(5), 17)

    # Next batch should trigger auto-advance to shard 1
    loader.next_batch()
    assert_equal(loader.current_shard_idx, 1)
    assert_equal(loader.inputs.load(0), 100)
    assert_equal(loader.inputs.load(5), 105)

    # Shard 1, Sample 1
    loader.next_batch()
    assert_equal(loader.inputs.load(0), 106)
    assert_equal(loader.inputs.load(5), 111)


def test_dataloader_distributed() raises:
    # Shard 0 tokens are [0..23]
    # num_processes = 2, batch_size = 1, seq_len = 4
    # Total batch size = 2 * 1 * 4 = 8 tokens
    # Shard samples = (24 - 1) // 8 = 2 samples

    var loader_rank0 = DataLoader(
        "test_shard_0.bin",
        batch_size=1,
        seq_len=4,
        process_rank=0,
        num_processes=2,
    )
    var loader_rank1 = DataLoader(
        "test_shard_0.bin",
        batch_size=1,
        seq_len=4,
        process_rank=1,
        num_processes=2,
    )

    assert_equal(loader_rank0.shard_num_samples, 2)
    assert_equal(loader_rank1.shard_num_samples, 2)

    # Batch 0
    loader_rank0.next_batch()
    loader_rank1.next_batch()

    # Rank 0 inputs: [0, 1, 2, 3]
    assert_equal(loader_rank0.inputs.load(0), 0)
    assert_equal(loader_rank0.inputs.load(3), 3)

    # Rank 1 inputs: [4, 5, 6, 7]
    assert_equal(loader_rank1.inputs.load(0), 4)
    assert_equal(loader_rank1.inputs.load(3), 7)

    # Batch 1
    loader_rank0.next_batch()
    loader_rank1.next_batch()

    # Rank 0 inputs: [8, 9, 10, 11]
    assert_equal(loader_rank0.inputs.load(0), 8)
    assert_equal(loader_rank0.inputs.load(3), 11)

    # Rank 1 inputs: [12, 13, 14, 15]
    assert_equal(loader_rank1.inputs.load(0), 12)
    assert_equal(loader_rank1.inputs.load(3), 15)


def main() raises:
    setup_test_files()
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        cleanup_test_files()
        raise e^
    cleanup_test_files()
