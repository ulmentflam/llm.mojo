from std.memory import alloc

from llmm.io import read_and_copy, write_buffer
from llmm.memory import MutMemPtr
from llmm.dataloader import DataLoader


# The on-disk format mirrors Karpathy's llm.c (train_gpt2.cu) so checkpoints are
# interchangeable in structure. Two files are produced per checkpoint:
#
#   * model file  ("model_XXXXXXXX.bin"): 256-int header + the parameter blob.
#   * state file  ("state_XXXXXXXX_YYYYY.bin"): 256-int header + AdamW m & v
#                  moments. Optimizer/dataloader bookkeeping (step, rng, shard
#                  position) lives in the header, exactly like llm.c's
#                  `save_state` / `load_state`.
#
# Splitting the two means a finished run ships only the model file, while a
# resumable run keeps the (much larger) state file alongside it.


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #


comptime CHECKPOINT_HEADER_SIZE = 256  # Number of Int32 slots in every header.

# Model checkpoint header. Matches GPT2_MAGIC in train_gpt2.mojo (20240520) and
# the version convention from llm.c: 3 => fp32 params, 5 => bf16 params.
comptime MODEL_MAGIC = 20240520
comptime VERSION_FP32 = 3
comptime VERSION_BF16 = 5

# State checkpoint header. 20240527 is llm.c's `save_state` magic.
comptime STATE_MAGIC = 20240527
comptime STATE_VERSION = 1

comptime AUTO_VERSION = -1  # Sentinel: derive the version from the param dtype.


# ===----------------------------------------------------------------------=== #
# Configuration mirror
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct CheckpointConfig(Copyable, Movable):
    """Model shape persisted in the checkpoint header.

    Field-for-field a mirror of `GPT2Config` in train_gpt2.mojo. Kept here so the
    `llmm` library has no dependency on the top-level training entry point.
    """

    var max_seq_len: Int  # Max sequence length (e.g. 1024).
    var vocab_size: Int  # Vocab size (e.g. 50257).
    var num_layer: Int  # Number of layers (e.g. 12).
    var num_heads: Int  # Number of heads (e.g. 12).
    var channels: Int  # Number of channels (e.g. 768).
    var padded_vocab_size: Int  # Padded vocab size (%128 == 0, e.g. 50304).

    def num_parameters(self) -> Int:
        """Total number of model parameters implied by this config.

        Mirrors `GPT2.allocate_parameters`'s `param_sizes` sum so a reader can
        size its buffer from the header alone.
        """
        var max_T = self.max_seq_len
        var V_p = self.padded_vocab_size
        var L = self.num_layer
        var C = self.channels

        var total = 0
        total += V_p * C  # wte             (V_p, C)
        total += max_T * C  # wpe            (max_T, C)
        total += L * C  # ln_1_gamma         (L, C)
        total += L * C  # ln_1_beta          (L, C)
        total += L * (3 * C) * C  # qkv_weight        (L, 3C, C)
        total += L * (3 * C)  # qkv_bias           (L, 3C)
        total += L * C * C  # attn_proj_weight  (L, C, C)
        total += L * C  # attn_proj_bias     (L, C)
        total += L * C  # ln_2_gamma         (L, C)
        total += L * C  # ln_2_beta          (L, C)
        total += L * (4 * C) * C  # fc_weight         (L, 4C, C)
        total += L * (4 * C)  # fc_bias            (L, 4C)
        total += L * C * (4 * C)  # proj_weight       (L, C, 4C)
        total += L * C  # proj_bias          (L, C)
        total += C  # ln_f_gamma             (C,)
        total += C  # ln_f_beta              (C,)
        return total

    def __eq__(self, other: Self) -> Bool:
        return (
            self.max_seq_len == other.max_seq_len
            and self.vocab_size == other.vocab_size
            and self.num_layer == other.num_layer
            and self.num_heads == other.num_heads
            and self.channels == other.channels
            and self.padded_vocab_size == other.padded_vocab_size
        )

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct ModelHeader(Copyable, Movable):
    """Parsed model-checkpoint header: config plus its on-disk param version."""

    var config: CheckpointConfig
    var version: Int


@fieldwise_init
struct TrainingState(Copyable, Movable):
    """Resumable bookkeeping persisted in the state-checkpoint header.

    Layout follows llm.c's `save_state`: scalars in the header, optimizer
    moments in the payload (written separately).
    """

    var step: Int  # Next optimization step to run.
    var num_processes: Int  # World size the checkpoint was written with.
    var process_rank: Int  # Rank that owns this state shard.
    var use_master_weights: Int  # 1 if master fp32 weights were saved.
    var should_shuffle: Int  # 1 if the dataloader was shuffling.
    var sampler_rng_state: UInt64  # Generation/sampling RNG state.
    var shuffle_rng_state: UInt64  # Dataloader shuffle RNG state.
    var current_shard_idx: Int  # Dataloader shard position.
    var current_sample_idx: Int  # Dataloader intra-shard sample position.


# ===----------------------------------------------------------------------=== #
# Low-level byte IO
# ===----------------------------------------------------------------------=== #


@always_inline
def _version_for_dtype(dtype: DType) -> Int:
    if dtype == DType.bfloat16 or dtype == DType.float16:
        return VERSION_BF16
    return VERSION_FP32


# ===----------------------------------------------------------------------=== #
# Model checkpoint
# ===----------------------------------------------------------------------=== #


def write_model_checkpoint[
    dtype: DType = DType.float32,
](
    path: String,
    config: CheckpointConfig,
    params: MutMemPtr[dtype],
    num_parameters: Int,
    version: Int = AUTO_VERSION,
) raises:
    """Write a model checkpoint: 256-int header followed by the parameter blob.

    `params` must point to a host buffer of `num_parameters` contiguous
    elements, laid out in the canonical parameter order (wte, wpe, ln_1_gamma,
    ... ln_f_beta). When training on GPU, copy the device parameter buffer to a
    host buffer first, exactly as `GPT2.allocate_parameters` does when reading.
    """
    var resolved_version = version
    if resolved_version == AUTO_VERSION:
        resolved_version = _version_for_dtype(dtype)

    var header = alloc[Int32](CHECKPOINT_HEADER_SIZE)
    for i in range(CHECKPOINT_HEADER_SIZE):
        header.store(i, Int32(0))
    header.store(0, Int32(MODEL_MAGIC))
    header.store(1, Int32(resolved_version))
    header.store(2, Int32(config.max_seq_len))
    header.store(3, Int32(config.vocab_size))
    header.store(4, Int32(config.num_layer))
    header.store(5, Int32(config.num_heads))
    header.store(6, Int32(config.channels))
    header.store(7, Int32(config.padded_vocab_size))

    var file = open(path, "w")
    write_buffer[DType.int32](file, header, CHECKPOINT_HEADER_SIZE)
    write_buffer[dtype](file, params, num_parameters)
    file.close()
    header.free()


def read_model_header(mut file: FileHandle) raises -> ModelHeader:
    """Read and validate a 256-int model header from an already-open file."""
    var header = alloc[Int32](CHECKPOINT_HEADER_SIZE)
    read_and_copy[DType.int32](file, header, CHECKPOINT_HEADER_SIZE)

    var magic = Int(header.load(0))
    var version = Int(header.load(1))
    if magic != MODEL_MAGIC:
        header.free()
        raise Error(
            "Checkpoint error: bad model magic number in header: "
            + String(magic)
        )
    if version != VERSION_FP32 and version != VERSION_BF16:
        header.free()
        raise Error(
            "Checkpoint error: unsupported model version in header: "
            + String(version)
        )

    var config = CheckpointConfig(
        max_seq_len=Int(header.load(2)),
        vocab_size=Int(header.load(3)),
        num_layer=Int(header.load(4)),
        num_heads=Int(header.load(5)),
        channels=Int(header.load(6)),
        padded_vocab_size=Int(header.load(7)),
    )
    header.free()
    return ModelHeader(config^, version)


def peek_model_header(path: String) raises -> ModelHeader:
    """Open `path`, read & validate the model header, then close it.

    Use this to discover the config (and therefore `num_parameters()`) before
    allocating the destination buffer for `read_model_checkpoint`.
    """
    var file = open(path, "r")
    var header = read_model_header(file)
    file.close()
    return header^


def read_model_checkpoint[
    dtype: DType = DType.float32,
](
    path: String,
    params_out: MutMemPtr[dtype],
    capacity: Int,
) raises -> ModelHeader:
    """Read a model checkpoint's params into `params_out`; return its header.

    `capacity` is the number of elements `params_out` can hold and must be at
    least `header.config.num_parameters()` (call `peek_model_header` first to
    size the buffer).
    """
    var file = open(path, "r")
    var header = read_model_header(file)
    var num_parameters = header.config.num_parameters()
    if capacity < num_parameters:
        file.close()
        raise Error(
            "Checkpoint error: params buffer too small (need "
            + String(num_parameters)
            + ", got "
            + String(capacity)
            + ")"
        )
    read_and_copy[dtype](file, params_out, num_parameters)
    file.close()
    return header^


# ===----------------------------------------------------------------------=== #
# Optimizer / training-state checkpoint
# ===----------------------------------------------------------------------=== #


def write_state_checkpoint[
    dtype: DType = DType.float32,
](
    path: String,
    state: TrainingState,
    m_memory: MutMemPtr[dtype],
    v_memory: MutMemPtr[dtype],
    shard_num_parameters: Int,
) raises:
    """Write the optimizer/training state: 256-int header + AdamW m & v moments.

    `shard_num_parameters` is the per-rank moment count
    (`GPT2.optimizer_num_parameters`); under ZeRO each rank writes its own shard
    to its own `state_<step>_<rank>.bin`, matching llm.c.
    """
    var header = alloc[Int32](CHECKPOINT_HEADER_SIZE)
    for i in range(CHECKPOINT_HEADER_SIZE):
        header.store(i, Int32(0))
    header.store(0, Int32(STATE_MAGIC))
    header.store(1, Int32(STATE_VERSION))
    header.store(2, Int32(state.num_processes))
    header.store(3, Int32(state.process_rank))
    header.store(4, Int32(state.use_master_weights))
    header.store(5, Int32(state.should_shuffle))
    header.store(10, Int32(state.step))

    # 64-bit fields share the same buffer; an aligned UInt64 view writes pairs
    # of Int32 slots (index k => slots 2k, 2k+1), matching llm.c's byte offsets.
    var u64_view = header.bitcast[UInt64]()
    u64_view.store(10, state.sampler_rng_state)  # slots 20-21
    u64_view.store(11, state.shuffle_rng_state)  # slots 22-23
    u64_view.store(15, UInt64(state.current_shard_idx))  # slots 30-31
    u64_view.store(16, UInt64(state.current_sample_idx))  # slots 32-33

    var file = open(path, "w")
    write_buffer[DType.int32](file, header, CHECKPOINT_HEADER_SIZE)
    write_buffer[dtype](file, m_memory, shard_num_parameters)
    write_buffer[dtype](file, v_memory, shard_num_parameters)
    file.close()
    header.free()


def read_state_header(mut file: FileHandle) raises -> TrainingState:
    """Read and validate a 256-int state header from an already-open file."""
    var header = alloc[Int32](CHECKPOINT_HEADER_SIZE)
    read_and_copy[DType.int32](file, header, CHECKPOINT_HEADER_SIZE)

    var magic = Int(header.load(0))
    var version = Int(header.load(1))
    if magic != STATE_MAGIC:
        header.free()
        raise Error(
            "Checkpoint error: bad state magic number in header: "
            + String(magic)
        )
    if version != STATE_VERSION:
        header.free()
        raise Error(
            "Checkpoint error: unsupported state version in header: "
            + String(version)
        )

    var u64_view = header.bitcast[UInt64]()
    var state = TrainingState(
        step=Int(header.load(10)),
        num_processes=Int(header.load(2)),
        process_rank=Int(header.load(3)),
        use_master_weights=Int(header.load(4)),
        should_shuffle=Int(header.load(5)),
        sampler_rng_state=u64_view.load(10),
        shuffle_rng_state=u64_view.load(11),
        current_shard_idx=Int(u64_view.load(15)),
        current_sample_idx=Int(u64_view.load(16)),
    )
    header.free()
    return state^


def read_state_checkpoint[
    dtype: DType = DType.float32,
](
    path: String,
    m_out: MutMemPtr[dtype],
    v_out: MutMemPtr[dtype],
    shard_num_parameters: Int,
) raises -> TrainingState:
    """Read the optimizer state into `m_out`/`v_out`; return the TrainingState.

    Both buffers must hold at least `shard_num_parameters` elements.
    """
    var file = open(path, "r")
    var state = read_state_header(file)
    read_and_copy[dtype](file, m_out, shard_num_parameters)
    read_and_copy[dtype](file, v_out, shard_num_parameters)
    file.close()
    return state^


# ===----------------------------------------------------------------------=== #
# DataLoader integration
# ===----------------------------------------------------------------------=== #


def make_training_state(
    loader: DataLoader,
    step: Int,
    sampler_rng_state: UInt64 = 0,
    use_master_weights: Int = 0,
) raises -> TrainingState:
    """Snapshot a DataLoader's position into a `TrainingState` for checkpointing.
    """
    return TrainingState(
        step=step,
        num_processes=loader.num_processes,
        process_rank=loader.process_rank,
        use_master_weights=use_master_weights,
        should_shuffle=1 if loader.should_shuffle else 0,
        sampler_rng_state=sampler_rng_state,
        shuffle_rng_state=loader.shuffle_rng_state,
        current_shard_idx=loader.current_shard_idx,
        current_sample_idx=loader.current_sample_idx,
    )


def restore_dataloader_state(
    mut loader: DataLoader, state: TrainingState
) raises:
    """Reposition `loader` to the shard/sample recorded in `state`.

    After this, the next `loader.next_batch()` returns the batch the original
    run would have produced next. The loader must be constructed with the same
    files, batch size, and sequence length as when the state was written.

    Note: for a shuffling loader this restores the RNG state and shard/sample
    indices and regenerates the intra-shard permutation. The default training
    loop does not shuffle, in which case resumption is exact.
    """
    loader.shuffle_rng_state = state.shuffle_rng_state
    loader.current_shard_idx = state.current_shard_idx
    _ = loader._load_shard(state.current_shard_idx)
    loader.current_sample_idx = state.current_sample_idx
    if loader.should_shuffle:
        loader._prepare_intra_shard_indices()
