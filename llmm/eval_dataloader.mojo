from std.memory import alloc

from llmm.memory import MutMemPtr, MutKernelPtr

# Loader for multiple-choice completion-style evals (HellaSwag, MMLU-style),
# mirroring llm.c's EvalLoader (llmc/dataloader.h). Each example is a shared
# context plus ASSUMED_NUM_COMPLETIONS candidate endings; the model scores
# each completion's average per-token loss, and the lowest-loss completion is
# its prediction. See data/utils.py's write_evalfile for the exact on-disk
# format this reads (256-int32 header, then one uint16 stream per example:
# <START_EXAMPLE>=65535, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>, <LABEL>,
# <NUM_COMPLETIONS>, <NUM><CONTEXT_TOKENS>, then <NUM><COMPLETION_TOKENS>
# repeated NUM_COMPLETIONS times).


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime EVAL_HEADER_SIZE = 256
comptime EVAL_MAGIC = 20240522
comptime EVAL_VERSION = 1
comptime START_EXAMPLE_DELIM = 65535
comptime ASSUMED_NUM_COMPLETIONS = 4


# ===----------------------------------------------------------------------=== #
# EvalDataLoader
# ===----------------------------------------------------------------------=== #


struct EvalDataLoader:
    # Variables related to distributed evaluation.
    var process_rank: Int
    var num_processes: Int

    # Batch and shape information.
    var batch_size: Int  # Our B
    var seq_len: Int  # Our T
    var can_fit_examples: Int  # B // ASSUMED_NUM_COMPLETIONS

    # Dataset and work-partition information.
    var num_examples: Int  # Total examples in the file, across all processes
    var num_batches: Int  # Batches needed to cover this process's share
    var start_example_index: Int  # Inclusive start of this process's range
    var end_example_index: Int  # Exclusive end of this process's range
    var current_example_index: Int  # The next example we would read

    # File handle and header info.
    var eval_file: FileHandle
    var header_bytes: Int
    var longest_example_bytes: Int

    # Batch tensors (row-major, batch_size * seq_len elements).
    var inputs: MutMemPtr[DType.int32]  # (B, T)
    var targets: MutMemPtr[DType.int32]  # (B, T) — inputs shifted left by 1
    var mask: MutMemPtr[DType.uint8]  # (B, T) — 1 at completion-token positions
    var label: MutMemPtr[
        DType.int32
    ]  # (can_fit_examples,) — correct completion index

    var has_allocated: Bool

    def __init__(
        out self,
        filename: String,
        batch_size: Int,
        seq_len: Int,
        process_rank: Int = 0,
        num_processes: Int = 1,
    ) raises:
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.header_bytes = EVAL_HEADER_SIZE * 4  # sizeof(int32) = 4 bytes.

        if batch_size // ASSUMED_NUM_COMPLETIONS == 0:
            raise Error(
                "EvalDataLoader error: batch size "
                + String(batch_size)
                + " is < "
                + String(ASSUMED_NUM_COMPLETIONS)
                + " (HINT: increase batch size, e.g. -b 16)"
            )
        self.can_fit_examples = batch_size // ASSUMED_NUM_COMPLETIONS

        self.eval_file = open(filename, "r")
        var header_bytes = self.eval_file.read_bytes(self.header_bytes)
        if len(header_bytes) < self.header_bytes:
            raise Error(
                "EvalDataLoader error: header is too short in file " + filename
            )
        var header_ptr = header_bytes.unsafe_ptr().bitcast[Int32]()
        var magic = header_ptr.load(0)
        if magic != EVAL_MAGIC:
            raise Error("EvalDataLoader error: bad magic number in " + filename)
        var version = header_ptr.load(1)
        if version != EVAL_VERSION:
            raise Error("EvalDataLoader error: bad version in " + filename)
        self.num_examples = Int(header_ptr.load(2))
        self.longest_example_bytes = Int(header_ptr.load(3))
        if self.num_examples < num_processes:
            raise Error(
                "EvalDataLoader error: fewer examples ("
                + String(self.num_examples)
                + ") than processes ("
                + String(num_processes)
                + ")"
            )

        self.num_batches = 0
        self.start_example_index = 0
        self.end_example_index = 0
        self.current_example_index = 0

        self.inputs = alloc[Scalar[DType.int32]](batch_size * seq_len)
        self.targets = alloc[Scalar[DType.int32]](batch_size * seq_len)
        self.mask = alloc[Scalar[DType.uint8]](batch_size * seq_len)
        self.label = alloc[Scalar[DType.int32]](self.can_fit_examples)
        self.has_allocated = True

        self.reset()

    def reset(mut self) raises:
        # Partition num_examples across processes: process 0 gets
        # [0, examples_per_process), process 1 gets [examples_per_process,
        # 2*examples_per_process), etc, with the last process's range clamped
        # to num_examples.
        var examples_per_process = (
            self.num_examples + self.num_processes - 1
        ) // self.num_processes
        self.num_batches = (
            examples_per_process + self.can_fit_examples - 1
        ) // self.can_fit_examples
        self.start_example_index = examples_per_process * self.process_rank
        self.end_example_index = examples_per_process * (self.process_rank + 1)
        if self.end_example_index > self.num_examples:
            self.end_example_index = self.num_examples

        # Seek to the start of the examples, then read-and-discard every
        # example before this process's start index. (llm.c's C reference
        # uses a relative fseek past each example's payload instead; we read
        # and discard the payload bytes here since this repo's I/O — see
        # llmm/dataloader.mojo — only ever does absolute seeks. Examples are
        # small (bounded by longest_example_bytes), so this costs nothing
        # measurable for a one-shot eval pass.)
        _ = self.eval_file.seek(UInt64(self.header_bytes))
        for i in range(self.start_example_index):
            _ = self._read_example_payload(i)
        self.current_example_index = self.start_example_index

    def _read_example_payload(
        mut self, expected_index: Int
    ) raises -> List[UInt16]:
        # Reads one example's 3-uint16 sub-header plus its payload, validates
        # the delimiter/index, and returns the payload as a uint16 list
        # (label, num_completions, context_length, context tokens, then each
        # completion as [length, tokens...]).
        var subheader_bytes = self.eval_file.read_bytes(6)  # 3 * sizeof(uint16)
        if len(subheader_bytes) < 6:
            raise Error("EvalDataLoader error: truncated example sub-header")
        var subheader_ptr = subheader_bytes.unsafe_ptr().bitcast[UInt16]()
        var start_delim = subheader_ptr.load(0)
        var example_bytes = Int(subheader_ptr.load(1))
        var example_index = Int(subheader_ptr.load(2))
        if start_delim != START_EXAMPLE_DELIM:
            raise Error("EvalDataLoader error: bad <START_EXAMPLE> delimiter")
        if example_index != expected_index:
            raise Error(
                "EvalDataLoader error: <EXAMPLE_INDEX> mismatch (expected "
                + String(expected_index)
                + ", got "
                + String(example_index)
                + ")"
            )
        var payload_bytes = example_bytes - 6
        if payload_bytes <= 0:
            raise Error(
                "EvalDataLoader error: non-positive example payload size"
            )
        var raw_bytes = self.eval_file.read_bytes(payload_bytes)
        if len(raw_bytes) < payload_bytes:
            raise Error("EvalDataLoader error: truncated example payload")
        var raw_ptr = raw_bytes.unsafe_ptr().bitcast[UInt16]()
        var num_u16 = payload_bytes // 2
        var out = List[UInt16]()
        for i in range(num_u16):
            out.append(raw_ptr.load(i))
        return out^

    def _next_example(mut self, example_batch_index: Int) raises:
        # Populates inputs/targets/mask/label for one example, at rows
        # [example_batch_index * ASSUMED_NUM_COMPLETIONS, +ASSUMED_NUM_COMPLETIONS)
        # of the batch.
        var T = self.seq_len
        var batch_dim_offset = example_batch_index * ASSUMED_NUM_COMPLETIONS
        var buf = self._read_example_payload(self.current_example_index)

        var label = Int(buf[0])
        if label < 0 or label >= ASSUMED_NUM_COMPLETIONS:
            raise Error("EvalDataLoader error: label out of range")
        self.label[example_batch_index] = Int32(label)

        var num_completions = Int(buf[1])
        if num_completions != ASSUMED_NUM_COMPLETIONS:
            raise Error(
                "EvalDataLoader error: expected "
                + String(ASSUMED_NUM_COMPLETIONS)
                + " completions, got "
                + String(num_completions)
            )

        var context_length = Int(buf[2])
        if context_length <= 0 or context_length >= T:
            raise Error(
                "EvalDataLoader error: context_length out of range (T="
                + String(T)
                + ")"
            )

        # The context is shared: write it into every completion row.
        for c in range(num_completions):
            var row = batch_dim_offset + c
            for i in range(context_length):
                self.inputs[row * T + i] = Int32(buf[3 + i])

        # Completions follow the context in each row; targets are inputs
        # shifted left by one (standard next-token prediction), and mask=1
        # marks every position where a completion token is being predicted.
        var cursor = 3 + context_length
        for c in range(num_completions):
            var row = batch_dim_offset + c
            var completion_length = Int(buf[cursor])
            if completion_length <= 0 or context_length + completion_length > T:
                raise Error(
                    "EvalDataLoader error: completion doesn't fit in T="
                    + String(T)
                )
            for i in range(completion_length):
                var tok = Int32(buf[cursor + 1 + i])
                self.inputs[row * T + context_length + i] = tok
                self.targets[row * T + context_length + i - 1] = tok
                self.mask[row * T + context_length + i - 1] = 1
            cursor += 1 + completion_length

        self.current_example_index += 1

    def next_batch(mut self) raises:
        var B = self.batch_size
        var T = self.seq_len
        for i in range(B * T):
            self.mask[i] = 0
        for i in range(self.can_fit_examples):
            if self.current_example_index >= self.end_example_index:
                break  # This process has exhausted its work.
            self._next_example(i)

    def close(mut self):
        try:
            self.eval_file.close()
        except:
            pass

    def __del__(deinit self):
        try:
            self.eval_file.close()
        except:
            pass
        if self.has_allocated:
            self.inputs.free()
            self.targets.free()
            self.mask.free()
            self.label.free()


def eval_stat_correct(
    loader: EvalDataLoader, losses: MutKernelPtr[DType.float32]
) -> Int:
    """Given per-token losses (B*T, from a forward pass on `loader`'s current
    batch), returns how many examples in this batch the model got right: the
    completion with the lowest masked-average loss matches the label.

    Mirrors llm.c's evalloader_stat_losses.
    """
    var B = loader.batch_size
    var T = loader.seq_len
    var correct = 0
    for i in range(loader.can_fit_examples):
        var min_loss = Float32(0.0)
        var min_loss_index = -1
        var active = False
        for c in range(ASSUMED_NUM_COMPLETIONS):
            var row = i * ASSUMED_NUM_COMPLETIONS + c
            var total_loss = Float32(0.0)
            var count = 0
            for t in range(T):
                if loader.mask[row * T + t] != 0:
                    active = True
                    total_loss += losses[row * T + t]
                    count += 1
            var avg_loss = Float32(0.0)
            if count > 0:
                avg_loss = total_loss / Float32(count)
            if c == 0 or avg_loss < min_loss:
                min_loss = avg_loss
                min_loss_index = c
        if active and min_loss_index == Int(loader.label[i]):
            correct += 1
    return correct
