from std.memory import alloc

from .io import read_and_copy

# ===----------------------------------------------------------------------=== #
# GPT-2 Tokenizer (decode only)
#
# Loads the binary vocab written by train_gpt2.py / llm.c and maps token ids
# back to raw byte strings for unconditional generation.
# ===----------------------------------------------------------------------=== #


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #


comptime HEADER_SIZE = 256

comptime TOKENIZER_MAGIC = 20240520
comptime TOKENIZER_MAGIC_LEGACY = 20240328

comptime GPT2_VOCAB_SIZE = 50257
comptime GPT2_EOT_TOKEN = 50256


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def _is_printable_or_space(byte_val: Int) -> Bool:
    """
    Mirrors C isprint() || isspace() for single-byte GPT-2 tokens.
    """
    # \t, \n, \v, \f
    if byte_val == 9 or byte_val == 10 or byte_val == 11 or byte_val == 12:
        return True
    # \r, space
    if byte_val == 13 or byte_val == 32:
        return True
    # Printable ASCII range (33-126)
    return byte_val >= 33 and byte_val <= 126


def _read_token_string(mut file: FileHandle, length: Int) raises -> String:
    var token_bytes = file.read_bytes(length)
    if len(token_bytes) < length:
        raise Error("Tokenizer error: truncated token bytes")
    # GPT-2 tokens are raw byte strings; preserve bytes exactly like llm.c char*.
    return String(unsafe_from_utf8=token_bytes)


def safe_print(piece: String) raises:
    """Print a decoded token, skipping non-printable single-byte tokens."""
    var piece_bytes = piece.as_bytes()
    if len(piece_bytes) == 0:
        return
    if len(piece_bytes) == 1:
        var byte_val = Int(piece_bytes[0])
        if not _is_printable_or_space(byte_val):
            return
    print(piece, end="")


# ===----------------------------------------------------------------------=== #
# Tokenizer
# ===----------------------------------------------------------------------=== #


struct Tokenizer:
    var vocab_size: Int
    var token_table: List[String]
    var has_initialized: Bool
    var eot_token: Int

    def __init__(out self, filename: String, quiet: Bool = False):
        self.vocab_size = 0
        self.token_table = List[String]()
        self.has_initialized = False
        self.eot_token = GPT2_EOT_TOKEN

        var file: FileHandle
        try:
            file = open(filename, "r")
        except:
            if not quiet:
                # This block was copied directly from Karpathy's llm.c implementation.
                print("---")
                print("WARNING: Failed to open the tokenizer file " + filename)
                print("The Tokenizer is a new feature added April 14 2024.")
                print("Re-run `python train_gpt2.py` to write it")
                print("---")
            return

        try:
            self._load(file)
            self.has_initialized = True
        except e:
            if not quiet:
                print("---")
                print("WARNING: Failed to load tokenizer file " + filename)
                print(String(e))
                print("---")
        finally:
            try:
                file.close()
            except:
                pass

    def _load(mut self, mut file: FileHandle) raises:
        var header = alloc[Int32](HEADER_SIZE)
        read_and_copy[DType.int32](file, header, HEADER_SIZE)

        var magic = Int(header.load(0))
        if magic != TOKENIZER_MAGIC and magic != TOKENIZER_MAGIC_LEGACY:
            header.free()
            raise Error(
                "Tokenizer error: bad magic number in header: " + String(magic)
            )

        var version = Int(header.load(1))
        self.vocab_size = Int(header.load(2))

        if version == 1:
            if self.vocab_size != GPT2_VOCAB_SIZE:
                header.free()
                raise Error(
                    "Tokenizer error: unexpected vocab size for version 1: "
                    + String(self.vocab_size)
                )
            self.eot_token = GPT2_EOT_TOKEN
        elif version == 2:
            self.eot_token = Int(header.load(3))
        else:
            header.free()
            raise Error(
                "Tokenizer error: bad version in header: " + String(version)
            )

        header.free()

        self.token_table = List[String]()
        self.token_table.reserve(self.vocab_size)
        for _ in range(self.vocab_size):
            var length_bytes = file.read_bytes(1)
            if len(length_bytes) < 1:
                raise Error("Tokenizer error: truncated token length byte")
            var length = Int(length_bytes[0])
            if length <= 0:
                raise Error("Tokenizer error: token length must be positive")
            self.token_table.append(_read_token_string(file, length))

    def decode(self, token_id: Int) -> String:
        if not self.has_initialized:
            return String()
        if token_id < 0 or token_id >= self.vocab_size:
            return String()
        return self.token_table[token_id]
