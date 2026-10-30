from std.os.path import getsize
from std.sys import size_of
from llmm.io import read_and_copy, write_buffer
from llmm.memory import MutMemPtr, heap_alloc
from llmm.dataloader import DataLoader, RNG_SEED
from llmm.rand import MT19937


# The on-disk format mirrors Karpathy's llm.c (train_gpt2.cu) so checkpoints are
# interchangeable in structure. Two files are produced per checkpoint:
#
#   * model file  ("model_XXXXXXXX.bin"): 256-int header + the parameter blob.
#   * state file  ("state_XXXXXXXX_YYYYY.bin"): 256-int header + AdamW m & v
#                  moments. Optimizer/dataloader bookkeeping (step, rng, shard
#                  position) lives in the header, exactly like llm.c's
#                  `save_state` / `load_state`.
#
# Header slots used by the state file (Int32 indices): 0 magic, 1 version,
# 2 num_processes, 3 process_rank, 4 use_master_weights, 5 should_shuffle,
# 10 step, 20-21 sampler rng, 22-23 shuffle draws, 30-31 shard idx, 32-33
# sample idx (all llm.c-compatible), and 40 optimizer_kind (0 AdamW, 1 Muon;
# absent = 0 in files written before Muon). Muon keeps its momentum in the `m`
# payload slot and leaves `v` unused for the matrices.
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
# Untied-LM-head model files (-D LLMM_UNTIED_HEAD=1): the tied version + 10. The
# blob is the 16-tensor tied blob followed by one [padded_vocab, C] head block.
# New version numbers (not a header flag) so every reader that only knows 3/5
# refuses the file loudly instead of loading it as tied and dropping the head.
comptime VERSION_FP32_UNTIED = 13
comptime VERSION_BF16_UNTIED = 15

# State checkpoint header. 20240527 is llm.c's `save_state` magic.
comptime STATE_MAGIC = 20240527
comptime STATE_VERSION = 1

comptime AUTO_VERSION = -1  # Sentinel: derive the version from the param dtype.


# ===----------------------------------------------------------------------=== #
# Configuration mirror
# ===----------------------------------------------------------------------=== #


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
    # 1 when the model carries a separate lm_head [padded_vocab, C] tensor as a
    # 17th block after ln_f_beta (model-file versions 13/15), else 0 (tied).
    var untied_head: Int

    def __init__(
        out self,
        max_seq_len: Int,
        vocab_size: Int,
        num_layer: Int,
        num_heads: Int,
        channels: Int,
        padded_vocab_size: Int,
        untied_head: Int = 0,
    ):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_layer = num_layer
        self.num_heads = num_heads
        self.channels = channels
        self.padded_vocab_size = padded_vocab_size
        self.untied_head = untied_head

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
        if self.untied_head != 0:
            total += V_p * C  # lm_head       (V_p, C)
        return total

    def __eq__(self, other: Self) -> Bool:
        return (
            self.max_seq_len == other.max_seq_len
            and self.vocab_size == other.vocab_size
            and self.num_layer == other.num_layer
            and self.num_heads == other.num_heads
            and self.channels == other.channels
            and self.padded_vocab_size == other.padded_vocab_size
            and self.untied_head == other.untied_head
        )

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct ModelHeader(Copyable, Movable):
    """Parsed model-checkpoint header: config plus its on-disk param version."""

    var config: CheckpointConfig
    var version: Int


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
    var optimizer_kind: Int  # 0 = AdamW, 1 = Muon (header slot 40).
    # Sharded Muon (ZeRO>=1, W>1) only; all 0 otherwise, so AdamW and
    # ZeRO-0/W=1 Muon files stay byte-identical to before S4.
    var zero_stage: Int  # header slot 41 (informational).
    # Slot 42: 0 = no owner payload, 1 = matrix-granular round-robin
    # owner(k, l) = (k*L + l) mod W. A nonzero value means a `muon_m`
    # blob of `muon_shard_elems` fp32 elements follows m and v.
    var muon_ownership_version: Int
    var muon_shard_elems: Int  # slot 43: S, the per-rank owner shard.
    # Slot 44: 1 = the writer was the untied-LM-head build (17-tensor layout),
    # 0 = tied. Slot 45: that run's untie step s_u (-1 = never untie, 0 =
    # untied from construction, >0 = untie at that step); always 0 in tied
    # files, so they stay byte-identical.
    var untied_layout: Int
    var untie_step: Int

    def __init__(
        out self,
        step: Int,
        num_processes: Int,
        process_rank: Int,
        use_master_weights: Int,
        should_shuffle: Int,
        sampler_rng_state: UInt64,
        shuffle_rng_state: UInt64,
        current_shard_idx: Int,
        current_sample_idx: Int,
        optimizer_kind: Int,
        zero_stage: Int = 0,
        muon_ownership_version: Int = 0,
        muon_shard_elems: Int = 0,
        untied_layout: Int = 0,
        untie_step: Int = 0,
    ):
        self.step = step
        self.num_processes = num_processes
        self.process_rank = process_rank
        self.use_master_weights = use_master_weights
        self.should_shuffle = should_shuffle
        self.sampler_rng_state = sampler_rng_state
        self.shuffle_rng_state = shuffle_rng_state
        self.current_shard_idx = current_shard_idx
        self.current_sample_idx = current_sample_idx
        self.optimizer_kind = optimizer_kind
        self.zero_stage = zero_stage
        self.muon_ownership_version = muon_ownership_version
        self.muon_shard_elems = muon_shard_elems
        self.untied_layout = untied_layout
        self.untie_step = untie_step


# ===----------------------------------------------------------------------=== #
# Low-level byte IO
# ===----------------------------------------------------------------------=== #


@always_inline
def _version_for_dtype(dtype: DType, untied_head: Int = 0) -> Int:
    if dtype == DType.bfloat16 or dtype == DType.float16:
        return VERSION_BF16_UNTIED if untied_head != 0 else VERSION_BF16
    return VERSION_FP32_UNTIED if untied_head != 0 else VERSION_FP32


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
        resolved_version = _version_for_dtype(dtype, config.untied_head)

    var header = heap_alloc[Int32](CHECKPOINT_HEADER_SIZE)
    for i in range(CHECKPOINT_HEADER_SIZE):
        header.unsafe_store(i, Int32(0))
    header.unsafe_store(0, Int32(MODEL_MAGIC))
    header.unsafe_store(1, Int32(resolved_version))
    header.unsafe_store(2, Int32(config.max_seq_len))
    header.unsafe_store(3, Int32(config.vocab_size))
    header.unsafe_store(4, Int32(config.num_layer))
    header.unsafe_store(5, Int32(config.num_heads))
    header.unsafe_store(6, Int32(config.channels))
    header.unsafe_store(7, Int32(config.padded_vocab_size))

    var file = open(path, "w")
    write_buffer[DType.int32](file, header, CHECKPOINT_HEADER_SIZE)
    write_buffer[dtype](file, params, num_parameters)
    file.close()
    header.unsafe_free()


def read_model_header(mut file: FileHandle) raises -> ModelHeader:
    """Read and validate a 256-int model header from an already-open file."""
    var header = heap_alloc[Int32](CHECKPOINT_HEADER_SIZE)
    read_and_copy[DType.int32](file, header, CHECKPOINT_HEADER_SIZE)

    var magic = Int(header.unsafe_load(0))
    var version = Int(header.unsafe_load(1))
    if magic != MODEL_MAGIC:
        header.unsafe_free()
        raise Error(
            "Checkpoint error: bad model magic number in header: "
            + String(magic)
        )
    var untied = (
        version == VERSION_FP32_UNTIED or version == VERSION_BF16_UNTIED
    )
    if version != VERSION_FP32 and version != VERSION_BF16 and not untied:
        header.unsafe_free()
        raise Error(
            "Checkpoint error: unsupported model version in header: "
            + String(version)
        )

    var config = CheckpointConfig(
        max_seq_len=Int(header.unsafe_load(2)),
        vocab_size=Int(header.unsafe_load(3)),
        num_layer=Int(header.unsafe_load(4)),
        num_heads=Int(header.unsafe_load(5)),
        channels=Int(header.unsafe_load(6)),
        padded_vocab_size=Int(header.unsafe_load(7)),
        untied_head=1 if untied else 0,
    )
    header.unsafe_free()
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
    size the buffer). The file must be exactly header + `num_parameters()`
    elements long: trailing bytes mean a layout this reader does not know (for
    example an untied blob behind a tied header) and are refused, not ignored.
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
    var expect_bytes = (
        CHECKPOINT_HEADER_SIZE * size_of[DType.int32]()
        + num_parameters * size_of[dtype]()
    )
    var file_bytes = Int(getsize(path))
    if file_bytes != expect_bytes:
        file.close()
        raise Error(
            "Checkpoint error: model file "
            + path
            + " is "
            + String(file_bytes)
            + " bytes, expected "
            + String(expect_bytes)
            + " (header version "
            + String(header.version)
            + ", num_parameters="
            + String(num_parameters)
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
    v_num_parameters: Int = -1,
    muon_m_memory: Optional[MutMemPtr[dtype]] = None,
) raises:
    """Write the optimizer/training state: 256-int header + AdamW m & v moments.

    When `state.muon_ownership_version != 0`, `muon_m_memory` (required) holds
    this rank's owner-local Muon momentum, `state.muon_shard_elems` elements,
    appended after v. Masters are not saved (rebuilt from params on resume).

    `shard_num_parameters` is the per-rank moment count
    (`GPT2.optimizer_num_parameters`); under ZeRO each rank writes its own shard
    to its own `state_<step>_<rank>.bin`, matching llm.c.
    `v_num_parameters` is the v blob length; -1 (default) means the same as m.
    Muon stores a compact v (AdamW-owned ranges only), so it passes less.
    """
    var n_v = shard_num_parameters if v_num_parameters < 0 else v_num_parameters
    var header = heap_alloc[Int32](CHECKPOINT_HEADER_SIZE)
    for i in range(CHECKPOINT_HEADER_SIZE):
        header.unsafe_store(i, Int32(0))
    header.unsafe_store(0, Int32(STATE_MAGIC))
    header.unsafe_store(1, Int32(STATE_VERSION))
    header.unsafe_store(2, Int32(state.num_processes))
    header.unsafe_store(3, Int32(state.process_rank))
    header.unsafe_store(4, Int32(state.use_master_weights))
    header.unsafe_store(5, Int32(state.should_shuffle))
    header.unsafe_store(10, Int32(state.step))
    header.unsafe_store(40, Int32(state.optimizer_kind))
    header.unsafe_store(41, Int32(state.zero_stage))
    header.unsafe_store(42, Int32(state.muon_ownership_version))
    header.unsafe_store(43, Int32(state.muon_shard_elems))
    header.unsafe_store(44, Int32(state.untied_layout))
    header.unsafe_store(45, Int32(state.untie_step))

    # 64-bit fields share the same buffer; an aligned UInt64 view writes pairs
    # of Int32 slots (index k => slots 2k, 2k+1), matching llm.c's byte offsets.
    var u64_view = header.unsafe_bitcast[UInt64]()
    u64_view.unsafe_store(10, state.sampler_rng_state)  # slots 20-21
    u64_view.unsafe_store(11, state.shuffle_rng_state)  # slots 22-23
    u64_view.unsafe_store(15, UInt64(state.current_shard_idx))  # slots 30-31
    u64_view.unsafe_store(16, UInt64(state.current_sample_idx))  # slots 32-33

    var file = open(path, "w")
    write_buffer[DType.int32](file, header, CHECKPOINT_HEADER_SIZE)
    write_buffer[dtype](file, m_memory, shard_num_parameters)
    write_buffer[dtype](file, v_memory, n_v)
    if state.muon_ownership_version != 0:
        if not muon_m_memory:
            file.close()
            header.unsafe_free()
            raise Error(
                "Checkpoint error: muon_ownership_version set but no"
                " muon_m_memory given"
            )
        write_buffer[dtype](file, muon_m_memory.value(), state.muon_shard_elems)
    file.close()
    header.unsafe_free()


def read_state_header(mut file: FileHandle) raises -> TrainingState:
    """Read and validate a 256-int state header from an already-open file."""
    var header = heap_alloc[Int32](CHECKPOINT_HEADER_SIZE)
    read_and_copy[DType.int32](file, header, CHECKPOINT_HEADER_SIZE)

    var magic = Int(header.unsafe_load(0))
    var version = Int(header.unsafe_load(1))
    if magic != STATE_MAGIC:
        header.unsafe_free()
        raise Error(
            "Checkpoint error: bad state magic number in header: "
            + String(magic)
        )
    if version != STATE_VERSION:
        header.unsafe_free()
        raise Error(
            "Checkpoint error: unsupported state version in header: "
            + String(version)
        )

    var u64_view = header.unsafe_bitcast[UInt64]()
    var state = TrainingState(
        step=Int(header.unsafe_load(10)),
        num_processes=Int(header.unsafe_load(2)),
        process_rank=Int(header.unsafe_load(3)),
        use_master_weights=Int(header.unsafe_load(4)),
        should_shuffle=Int(header.unsafe_load(5)),
        sampler_rng_state=u64_view.unsafe_load(10),
        shuffle_rng_state=u64_view.unsafe_load(11),
        current_shard_idx=Int(u64_view.unsafe_load(15)),
        current_sample_idx=Int(u64_view.unsafe_load(16)),
        optimizer_kind=Int(header.unsafe_load(40)),
        zero_stage=Int(header.unsafe_load(41)),
        muon_ownership_version=Int(header.unsafe_load(42)),
        muon_shard_elems=Int(header.unsafe_load(43)),
        untied_layout=Int(header.unsafe_load(44)),
        untie_step=Int(header.unsafe_load(45)),
    )
    header.unsafe_free()
    return state^


def read_state_checkpoint[
    dtype: DType = DType.float32,
](
    path: String,
    m_out: MutMemPtr[dtype],
    v_out: MutMemPtr[dtype],
    shard_num_parameters: Int,
    v_num_parameters: Int = -1,
    expect_num_processes: Int = -1,
    expect_process_rank: Int = -1,
    muon_m_out: Optional[MutMemPtr[dtype]] = None,
    expect_muon_shard_elems: Int = 0,
) raises -> TrainingState:
    """Read the optimizer state into `m_out`/`v_out`; return the TrainingState.

    `m_out` must hold `shard_num_parameters` elements, `v_out`
    `v_num_parameters` (-1 = same as m).

    A state file is one rank's shard of the AdamW moments, so it is only valid
    for the same world size and rank that wrote it. When `expect_num_processes`
    / `expect_process_rank` are >= 0 the header's `num_processes` /
    `process_rank` are compared against them BEFORE any payload is read, and a
    mismatch raises. Without the check a resume at a different world size
    silently read a prefix of a larger shard file (read_and_copy only errors on
    a short file). Negative (default) skips the comparison.

    Sharded Muon: pass `muon_m_out` (and `expect_muon_shard_elems` = S) to read
    the owner-local momentum blob. The header's ownership version must be 1 and
    its S must match; a file with no owner payload (ZeRO-0 / W=1 Muon, AdamW)
    is refused when `muon_m_out` is given, and an owner file is refused when it
    is not. The file length must equal header + m + v (+ muon_m) exactly: a
    longer or shorter file raises with both byte counts (the header does not
    record the m/v lengths, so this is what catches a ZeRO-0 file resumed at
    ZeRO>=1, or a different world size, in either direction).
    """
    var n_v = shard_num_parameters if v_num_parameters < 0 else v_num_parameters
    var file = open(path, "r")
    var state = read_state_header(file)
    if (
        expect_num_processes >= 0
        and state.num_processes != expect_num_processes
    ) or (
        expect_process_rank >= 0 and state.process_rank != expect_process_rank
    ):
        file.close()
        raise Error(
            "Cannot resume: state file "
            + path
            + " was written by rank "
            + String(state.process_rank)
            + " of "
            + String(state.num_processes)
            + " process(es) but this run is rank "
            + String(expect_process_rank)
            + " of "
            + String(expect_num_processes)
            + "; optimizer state is sharded per rank and cannot be resumed at"
            " a different world size"
        )
    var want_owner = Bool(muon_m_out)
    var have_owner = state.muon_ownership_version != 0
    if want_owner != have_owner:
        file.close()
        raise Error(
            "Cannot resume: state file "
            + path
            + (
                " has no owner-local Muon momentum (written by AdamW, or by Muon at ZeRO-0 / one process) but this run is sharded Muon (ZeRO>=1, W>1)" if want_owner else " holds owner-local Muon momentum (sharded Muon, ZeRO>=1) but this run expects the replicated layout (ZeRO-0 or one process)"
            )
            + "; resume with the same -z and world size that wrote it"
        )
    if want_owner and (
        state.muon_ownership_version != 1
        or state.muon_shard_elems != expect_muon_shard_elems
    ):
        file.close()
        raise Error(
            "Cannot resume: state file "
            + path
            + " has muon_ownership_version="
            + String(state.muon_ownership_version)
            + " shard_elems="
            + String(state.muon_shard_elems)
            + " but this run needs version 1 shard_elems="
            + String(expect_muon_shard_elems)
            + " (different ownership rule, model shape or world size)"
        )
    var n_mu = expect_muon_shard_elems if want_owner else 0
    var elem = size_of[dtype]()
    var head_bytes = CHECKPOINT_HEADER_SIZE * size_of[DType.int32]()
    var expect_bytes = head_bytes + (shard_num_parameters + n_v + n_mu) * elem
    var file_bytes = Int(getsize(path))
    if file_bytes != expect_bytes:
        file.close()
        raise Error(
            "Cannot resume: state file "
            + path
            + " is "
            + String(file_bytes)
            + " bytes, expected "
            + String(expect_bytes)
            + " (m="
            + String(shard_num_parameters)
            + ", v="
            + String(n_v)
            + ", muon_m="
            + String(n_mu)
            + " elements); it was written with a different optimizer shard"
            " layout (ZeRO stage or world size, or by a sharded-Muon build"
            " before the flat m/master trim, whose m blob was shard-sized"
            " instead of A_r)"
        )
    read_and_copy[dtype](file, m_out, shard_num_parameters)
    read_and_copy[dtype](file, v_out, n_v)
    if want_owner:
        read_and_copy[dtype](file, muon_m_out.value(), n_mu)
    # Backstop only: the exact getsize() check above already rejects a short
    # or long file (e.g. a ZeRO-0 state resumed at ZeRO>=1 at the same world
    # size, which would otherwise load a prefix of m and v silently). This
    # catches a file that grew between the size check and the read.
    var trailing = file.read_bytes(1)
    file.close()
    if len(trailing) != 0:
        raise Error(
            "Cannot resume: state file "
            + path
            + " holds more than the expected "
            + String(shard_num_parameters)
            + " m + v elements per moment; it was written with a different"
            " optimizer shard size (ZeRO stage or world size)"
        )
    return state^


# ===----------------------------------------------------------------------=== #
# DataLoader integration
# ===----------------------------------------------------------------------=== #


def make_training_state(
    loader: DataLoader,
    step: Int,
    sampler_rng_state: UInt64 = 0,
    use_master_weights: Int = 0,
    optimizer_kind: Int = 0,
    untied_layout: Int = 0,
    untie_step: Int = 0,
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
        # The mt19937 shuffle state is 624 words and does not fit this u64
        # field; store the draw count instead and replay it on restore.
        shuffle_rng_state=loader.shuffle_draws,
        current_shard_idx=loader.current_shard_idx,
        current_sample_idx=loader.current_sample_idx,
        optimizer_kind=optimizer_kind,
        untied_layout=untied_layout,
        untie_step=untie_step,
    )


def restore_dataloader_state(
    mut loader: DataLoader, state: TrainingState
) raises:
    """Reposition `loader` to the shard/sample recorded in `state`.

    After this, the next `loader.next_batch()` returns the batch the original
    run would have produced next. The loader must be constructed with the same
    files, batch size, and sequence length as when the state was written.

    Note: for a shuffling loader, `state.shuffle_rng_state` holds the count
    of mt19937 draws the original run had consumed. Restoration reseeds the
    generator, replays the draws that preceded the current shard's intra
    permutation, then regenerates that permutation, leaving both the batch
    order and the generator position exact. A checkpoint written by a
    non-shuffling run (`state.should_shuffle == 0`) has no stream to replay;
    restoring it into a shuffling loader starts a fresh shuffle stream.
    """
    loader.current_shard_idx = state.current_shard_idx
    _ = loader._load_shard(state.current_shard_idx)
    loader.current_sample_idx = state.current_sample_idx
    if loader.should_shuffle and state.should_shuffle == 1:
        var intra_draws = UInt64(
            loader.shard_num_samples - 1
        ) if loader.shard_num_samples > 1 else UInt64(0)
        var replay = state.shuffle_rng_state - intra_draws
        loader.shuffle_rng = MT19937(UInt32(RNG_SEED + loader.process_rank))
        for _ in range(Int(replay)):
            _ = loader.shuffle_rng.randint32()
        loader.shuffle_draws = replay
        loader._prepare_intra_shard_indices()
    elif loader.should_shuffle:
        loader._prepare_intra_shard_indices()
