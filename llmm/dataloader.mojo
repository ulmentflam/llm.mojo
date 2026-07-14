from std.memory import alloc
from std.python import Python

from llmm.sampler import random_permutation
from llmm.memory import ImmutKernelPtr, MutMemPtr


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime HEADER_SIZE = 256
comptime GPT2_MAGIC = 20240520
comptime LLAMA3_MAGIC = 20240801


comptime GLOB = "glob"
comptime RNG_SEED = 42  # The meaning of life, the universe, and everything.


# ===----------------------------------------------------------------------=== #
# File resolution
# ===----------------------------------------------------------------------=== #


def resolve_data_files(filename_pattern: String) raises -> List[String]:
    """Resolve a data path/glob pattern into a sorted list of shard files.

    A pattern containing `*`/`?` is expanded via CPython's `glob` module.
    That first Python call initializes the interpreter, which is NOT safe to
    do concurrently from the bare per-rank host threads that multi-rank
    training spawns (racing `Py_Initialize`/import crashes libpython in
    `PyImport_ImportModule`). Callers that fan out to per-rank DataLoaders must
    resolve ONCE on the main thread and hand each rank the plain list this
    returns, keeping the rank threads free of Python interop entirely.

    A literal (wildcard-free) path skips Python altogether.
    """
    var files = List[String]()
    if "*" not in filename_pattern and "?" not in filename_pattern:
        files.append(filename_pattern)
        return files^

    var glob = Python.import_module(GLOB)
    var py_files = glob.glob(filename_pattern)
    py_files.sort()

    if len(py_files) == 0:
        raise Error(
            "DataLoader error: No files matched the pattern: "
            + filename_pattern
        )

    for i in range(len(py_files)):
        files.append(String(py_files[i]))
    return files^


# ===----------------------------------------------------------------------=== #
# DataLoader
# ===----------------------------------------------------------------------=== #


struct DataLoader:
    # Variables related to distributed training.
    var process_rank: Int
    var num_processes: Int

    # Batch and token information.
    var batch_size: Int  # Our B
    var seq_len: Int  # Our T
    var num_tokens: Int  # Total number of tokens across all shards
    var shard_num_samples: Int  # Total samples in the current shard per process
    var current_shard_tokens: Int  # Total tokens in the current shard

    # Shards and current position.
    var files: List[String]
    var current_shard_idx: Int
    var current_sample_idx: Int

    var tokens_file: FileHandle

    # Data buffers.
    var inputs: MutMemPtr[DType.int32]
    var targets: MutMemPtr[DType.int32]

    # Random shuffle variables.
    var should_shuffle: Bool
    var shuffle_rng_state: UInt64
    var shard_indices: List[Int]
    var intra_shard_indices: List[Int]

    # Sizes in bytes.
    var total_batch_size_bytes: Int
    var local_batch_offset_bytes: Int
    var header_bytes: Int
    var file_size_bytes: Int

    # Datatype metadata.
    var token_size: Int
    var magic: Int32
    var version: Int32
    var has_allocated: Bool

    def __init__(
        out self,
        filename_pattern: String,
        batch_size: Int,
        seq_len: Int,
        process_rank: Int = 0,
        num_processes: Int = 1,
        should_shuffle: Bool = False,
    ) raises:
        """Construct from a path or glob pattern (resolves via CPython `glob`).

        Convenience for single-threaded callers. Multi-rank training must NOT
        use this from rank threads — resolve with `resolve_data_files` on the
        main thread and use the `files:` overload below.
        """
        self = DataLoader(
            resolve_data_files(filename_pattern),
            batch_size,
            seq_len,
            process_rank,
            num_processes,
            should_shuffle,
        )

    def __init__(
        out self,
        var files: List[String],
        batch_size: Int,
        seq_len: Int,
        process_rank: Int = 0,
        num_processes: Int = 1,
        should_shuffle: Bool = False,
    ) raises:
        """Construct from a pre-resolved shard file list (no Python interop)."""
        # Initialize scalar fields.
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_tokens = 0
        self.shard_num_samples = 0
        self.current_shard_tokens = 0

        self.current_shard_idx = 0
        self.current_sample_idx = 0
        self.should_shuffle = should_shuffle
        self.shuffle_rng_state = UInt64(RNG_SEED + process_rank)
        self.file_size_bytes = 0
        self.magic = 0
        self.version = 0
        self.token_size = 0

        self.total_batch_size_bytes = 0
        self.local_batch_offset_bytes = 0
        self.header_bytes = HEADER_SIZE * 4  # sizeof(int) = 4 bytes.

        # Initialize collection fields. `files` is already resolved (globbing,
        # if any, happened on the main thread via `resolve_data_files`), so no
        # Python interop occurs here — this constructor is safe to run from the
        # per-rank host threads that multi-rank training spawns.
        self.files = files^
        self.shard_indices = List[Int]()
        self.intra_shard_indices = List[Int]()

        if len(self.files) == 0:
            raise Error("DataLoader error: empty file list")

        # Allocate buffers for the input and target tokens.
        self.inputs = alloc[Scalar[DType.int32]](batch_size * seq_len)
        self.targets = alloc[Scalar[DType.int32]](batch_size * seq_len)
        self.has_allocated = True

        for i in range(len(self.files)):
            self.shard_indices.append(i)

        self.tokens_file = open(self.files[0], "r")

        # Now that all fields are initialized, we can safely call self methods
        self._load_shard_metadata()

        self.total_batch_size_bytes = (
            num_processes * batch_size * seq_len * self.token_size
        )
        self.local_batch_offset_bytes = (
            process_rank * batch_size * seq_len * self.token_size
        )

        # Inspect and validate all shards.
        var num_tokens = 0
        for shard_index in range(len(self.files)):
            var shard_ntok = self._load_shard(shard_index)
            var min_required = num_processes * batch_size * seq_len + 1
            if shard_ntok < min_required:
                raise Error(
                    "DataLoader error: Shard is too small for batch size (need "
                    + String(min_required)
                    + " tokens, got "
                    + String(shard_ntok)
                    + ")"
                )
            num_tokens += shard_ntok

        self.num_tokens = num_tokens

        self.reset()

    def _load_shard_metadata(mut self) raises:
        # Read header: 256 Int32 integers = 1024 bytes
        _ = self.tokens_file.seek(0)
        var header_bytes = self.tokens_file.read_bytes(self.header_bytes)
        if len(header_bytes) < self.header_bytes:
            raise Error(
                "DataLoader error: Header is too short in file "
                + self.files[self.current_shard_idx]
            )

        var header_ptr = header_bytes.unsafe_ptr().bitcast[Int32]()
        self.magic = header_ptr.load(0)
        assert (
            self.magic == GPT2_MAGIC or self.magic == LLAMA3_MAGIC
        ), "DataLoader error: Invalid magic number in header: " + String(
            self.magic
        )
        self.version = header_ptr.load(1)
        self.current_shard_tokens = Int(header_ptr.load(2))

        # Determine token size and format from magic number.
        if self.magic == GPT2_MAGIC:
            self.token_size = 2  # GPT-2 format is UInt16.
        elif self.magic == LLAMA3_MAGIC:
            self.token_size = 4  # Llama-3 format is UInt32
        else:
            raise Error(
                "DataLoader error: Invalid magic number in header: "
                + String(self.magic)
            )

    def _load_shard(mut self, shard_index: Int) -> Int:
        var actual_shard_idx = shard_index
        if self.should_shuffle:
            actual_shard_idx = self.shard_indices[shard_index]

        var filename = self.files[actual_shard_idx]

        # Reopen file handle
        try:
            self.tokens_file.close()
        except:
            pass
        try:
            self.tokens_file = open(filename, "r")
            self._load_shard_metadata()
        except e:
            # Propagate raising error.
            try:
                # NOTE: This is mainly for testing purposes.
                raise e^
            except:
                pass

        self.file_size_bytes = (
            self.header_bytes + self.current_shard_tokens * self.token_size
        )

        # -1 token worth of bytes due to us taking batch_size * seq_len + 1 tokens but moving by batch_size * seq_len.
        self.shard_num_samples = (
            self.current_shard_tokens * self.token_size - self.token_size
        ) // self.total_batch_size_bytes
        return self.current_shard_tokens

    def _prepare_intra_shard_indices(mut self):
        self.intra_shard_indices = List[Int]()
        for i in range(self.shard_num_samples):
            self.intra_shard_indices.append(i)
        random_permutation(self.intra_shard_indices, self.shuffle_rng_state)

    def reset(mut self) raises:
        self.current_shard_idx = 0
        self.current_sample_idx = 0

        if self.should_shuffle:
            random_permutation(self.shard_indices, self.shuffle_rng_state)

        _ = self._load_shard(self.current_shard_idx)

        if self.should_shuffle:
            self._prepare_intra_shard_indices()

    def _advance(mut self) raises:
        if self.current_shard_idx == len(self.files) - 1:
            self.reset()
            return

        self.current_shard_idx = (self.current_shard_idx + 1) % len(self.files)
        self.current_sample_idx = 0
        _ = self._load_shard(self.current_shard_idx)

        if self.should_shuffle:
            self._prepare_intra_shard_indices()

    def _load_batch(mut self) raises:
        if self.should_shuffle and len(self.intra_shard_indices) == 0:
            raise Error(
                "DataLoader error: Shuffle requested but intra_shard_indices"
                " not prepared"
            )

        var idx = self.current_sample_idx
        if self.should_shuffle:
            idx = self.intra_shard_indices[self.current_sample_idx]

        var global_batch_offset_bytes = idx * self.total_batch_size_bytes
        var current_offset = (
            self.header_bytes
            + global_batch_offset_bytes
            + self.local_batch_offset_bytes
        )

        var B = self.batch_size
        var T = self.seq_len

        var tokens_to_read = (
            B * T + 1
        )  # +1 because we need to read one extra token for the targets.
        var bytes_to_read = tokens_to_read * self.token_size

        _ = self.tokens_file.seek(UInt64(current_offset))
        var bytes_read = self.tokens_file.read_bytes(bytes_to_read)
        if len(bytes_read) < bytes_to_read:
            raise Error(
                "DataLoader error: Failed to read enough bytes from file"
            )

        var raw_ptr = bytes_read.unsafe_ptr()

        if self.token_size == 2:
            var ptr_u16 = raw_ptr.bitcast[UInt16]()
            for i in range(B * T):
                self.inputs.store(i, Int32(ptr_u16.load(i)))
                self.targets.store(i, Int32(ptr_u16.load(i + 1)))
        else:
            var ptr_u32 = raw_ptr.bitcast[UInt32]()
            for i in range(B * T):
                self.inputs.store(i, Int32(ptr_u32.load(i)))
                self.targets.store(i, Int32(ptr_u32.load(i + 1)))

    def next_batch(mut self) raises:
        if self.current_sample_idx >= self.shard_num_samples:
            self._advance()
        self._load_batch()
        self.current_sample_idx += 1

    def close(mut self):
        try:
            self.tokens_file.close()
        except:
            pass

    def __del__(deinit self):
        try:
            self.tokens_file.close()
        except:
            pass
        if self.has_allocated:
            self.inputs.free()
            self.targets.free()
