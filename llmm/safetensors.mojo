from std.math import ceildiv
from std.memory import alloc

from llmm.checkpointing import CheckpointConfig
from llmm.io import get_dtype_size
from llmm.memory import MutMemPtr

# ===----------------------------------------------------------------------=== #
# Safetensors reader (pure Mojo, no MAX/Python bridge)
#
# The safetensors container is a small, stable, publicly-documented format:
#   - 8 bytes: little-endian u64 header length N
#   - N bytes: UTF-8 JSON header, one entry per tensor plus an optional
#     "__metadata__" key:
#       {"<name>": {"dtype": "BF16", "shape": [V, C], "data_offsets": [a, b]}}
#     `data_offsets` are byte offsets into the tensor-data region, relative to
#     its start (i.e. relative to file offset 8+N), not to the file start.
#   - the raw tensor bytes, back to back, packed per `data_offsets`.
#
# MAX ships a C API for this (max/c/safetensors.h, M_loadSafetensors et al.,
# backed by libmax.so), but there is no Mojo-language binding for it in this
# SDK release (unlike cuBLAS, which ships as a precompiled _cublas.mojoc
# package) — the only Mojo-facing piece is `weights_registry.mojoc`'s
# `WeightsRegistry`, which supports read-only `__getitem__` lookup but exposes
# no public constructor from raw file paths; that registry is populated
# internally by MAX's own graph-compilation pipeline, not by user Mojo code
# loading an arbitrary checkpoint standalone. Hand-rolling FFI bindings
# against the C ABI would mean guessing struct layouts MAX hasn't published a
# Mojo contract for. The format itself is simple and stable, so parsing it
# natively here is the safer choice: no undocumented ABI surface, and it
# reads/writes through this repo's own `MutMemPtr`/`FileHandle` conventions
# exactly like the rest of `llmm/`.
# ===----------------------------------------------------------------------=== #

comptime _HEADER_LEN_BYTES = 8


@fieldwise_init
struct TensorInfo(Copyable, Movable):
    """One tensor's metadata, as parsed from the safetensors JSON header."""

    var dtype: String  # safetensors dtype string, e.g. "BF16", "F32".
    var shape: List[Int]
    var offset_begin: Int  # Byte offset, relative to the data region's start.
    var offset_end: Int  # Exclusive.

    def num_bytes(self) -> Int:
        return self.offset_end - self.offset_begin

    def num_elements(self) -> Int:
        var n = 1
        for d in self.shape:
            n *= d
        return n


# ===----------------------------------------------------------------------=== #
# Minimal JSON parsing — scoped to exactly what a safetensors header contains
# (a flat object of objects; values are strings, integers, or int arrays).
# Not a general-purpose JSON parser. Every parse function takes the current
# cursor via `mut pos: Int` and leaves it just past what it consumed, rather
# than returning `(value, new_pos)` tuples.
# ===----------------------------------------------------------------------=== #


def _skip_ws(text: String, mut pos: Int) -> None:
    var bytes = text.as_bytes()
    while pos < len(bytes) and (
        bytes[pos] == 32  # space
        or bytes[pos] == 9  # tab
        or bytes[pos] == 10  # \n
        or bytes[pos] == 13  # \r
    ):
        pos += 1


def _expect(text: String, mut pos: Int, ch: Int) raises -> None:
    var bytes = text.as_bytes()
    if pos >= len(bytes) or Int(bytes[pos]) != ch:
        raise Error(
            "safetensors header: expected '"
            + chr(ch)
            + "' at byte "
            + String(pos)
        )
    pos += 1


def _parse_json_string(text: String, mut pos: Int) raises -> String:
    """Parse a JSON string starting at `text[pos] == '"'`.

    Handles the escapes safetensors headers actually use (`\\"`, `\\\\`,
    `\\/`); anything else raises rather than silently mis-parsing.
    """
    var bytes = text.as_bytes()
    _expect(text, pos, 34)  # '"'
    var out = String()
    while pos < len(bytes) and Int(bytes[pos]) != 34:
        var b = Int(bytes[pos])
        if b == 92:  # backslash
            pos += 1
            if pos >= len(bytes):
                raise Error("safetensors header: truncated escape sequence")
            var esc = Int(bytes[pos])
            if esc == 34 or esc == 92 or esc == 47:
                out += chr(esc)
            else:
                raise Error(
                    "safetensors header: unsupported escape \\"
                    + chr(esc)
                    + " (tensor names/keys are expected to be plain ASCII)"
                )
        else:
            out += chr(b)
        pos += 1
    _expect(text, pos, 34)
    return out^


def _parse_json_int(text: String, mut pos: Int) raises -> Int:
    var bytes = text.as_bytes()
    var neg = False
    if pos < len(bytes) and Int(bytes[pos]) == 45:  # '-'
        neg = True
        pos += 1
    var start = pos
    var value = 0
    while pos < len(bytes) and Int(bytes[pos]) >= 48 and Int(bytes[pos]) <= 57:
        value = value * 10 + (Int(bytes[pos]) - 48)
        pos += 1
    if pos == start:
        raise Error(
            "safetensors header: expected integer at byte " + String(pos)
        )
    return -value if neg else value


def _parse_json_int_array(text: String, mut pos: Int) raises -> List[Int]:
    _skip_ws(text, pos)
    _expect(text, pos, 91)  # '['
    var out = List[Int]()
    _skip_ws(text, pos)
    var bytes = text.as_bytes()
    if pos < len(bytes) and Int(bytes[pos]) == 93:  # ']' — empty array
        pos += 1
        return out^
    while True:
        out.append(_parse_json_int(text, pos))
        _skip_ws(text, pos)
        bytes = text.as_bytes()
        if pos < len(bytes) and Int(bytes[pos]) == 44:  # ','
            pos += 1
            _skip_ws(text, pos)
            continue
        break
    _expect(text, pos, 93)  # ']'
    return out^


def _skip_json_value(text: String, mut pos: Int) raises -> None:
    """Skip one arbitrary JSON value (used for the `__metadata__` entry,
    whose contents we don't need)."""
    var bytes = text.as_bytes()
    _skip_ws(text, pos)
    if pos >= len(bytes):
        raise Error("safetensors header: unexpected end of header")
    var b = Int(bytes[pos])
    if b == 34:  # string
        _ = _parse_json_string(text, pos)
        return
    elif b == 123 or b == 91:  # object or array — bracket-depth skip
        var open_b = b
        var close_b = 125 if b == 123 else 93
        var depth = 0
        while pos < len(bytes):
            var c = Int(bytes[pos])
            if c == 34:  # string inside — skip it properly so braces in
                # string content (none expected here, but be safe) don't
                # confuse the depth count.
                _ = _parse_json_string(text, pos)
                continue
            if c == open_b:
                depth += 1
                pos += 1
            elif c == close_b:
                depth -= 1
                pos += 1
                if depth == 0:
                    return
            else:
                pos += 1
        raise Error("safetensors header: unterminated object/array")
    else:  # number / true / false / null — read until a delimiter
        while pos < len(bytes) and Int(bytes[pos]) not in (
            44,
            125,
            93,
            32,
            9,
            10,
            13,
        ):  # , } ] and whitespace
            pos += 1


def _parse_tensor_info(text: String, mut pos: Int) raises -> TensorInfo:
    _skip_ws(text, pos)
    _expect(text, pos, 123)  # '{'
    var dtype = String("")
    var shape = List[Int]()
    var offsets = List[Int]()
    _skip_ws(text, pos)
    var bytes = text.as_bytes()
    while not (pos < len(bytes) and Int(bytes[pos]) == 125):
        var key = _parse_json_string(text, pos)
        _skip_ws(text, pos)
        _expect(text, pos, 58)  # ':'
        _skip_ws(text, pos)
        if key == "dtype":
            dtype = _parse_json_string(text, pos)
        elif key == "shape":
            shape = _parse_json_int_array(text, pos)
        elif key == "data_offsets":
            offsets = _parse_json_int_array(text, pos)
        else:
            _skip_json_value(text, pos)
        _skip_ws(text, pos)
        bytes = text.as_bytes()
        if pos < len(bytes) and Int(bytes[pos]) == 44:  # ','
            pos += 1
            _skip_ws(text, pos)
            continue
        break
    _expect(text, pos, 125)  # '}'
    if len(offsets) != 2:
        raise Error(
            "safetensors header: tensor entry missing 2-element data_offsets"
        )
    return TensorInfo(dtype^, shape^, offsets[0], offsets[1])


def _parse_header(text: String) raises -> Dict[String, TensorInfo]:
    var tensors = Dict[String, TensorInfo]()
    var pos = 0
    _skip_ws(text, pos)
    _expect(text, pos, 123)  # '{'
    _skip_ws(text, pos)
    var bytes = text.as_bytes()
    if pos < len(bytes) and Int(bytes[pos]) == 125:  # '}' — no tensors
        return tensors^
    while True:
        var name = _parse_json_string(text, pos)
        _skip_ws(text, pos)
        _expect(text, pos, 58)  # ':'
        _skip_ws(text, pos)
        if name == "__metadata__":
            _skip_json_value(text, pos)
        else:
            tensors[name] = _parse_tensor_info(text, pos)
        _skip_ws(text, pos)
        bytes = text.as_bytes()
        if pos < len(bytes) and Int(bytes[pos]) == 44:  # ','
            pos += 1
            _skip_ws(text, pos)
            continue
        break
    _expect(text, pos, 125)  # '}'
    return tensors^


# ===----------------------------------------------------------------------=== #
# SafetensorsFile
# ===----------------------------------------------------------------------=== #


comptime SAFETENSORS_DTYPE_BF16 = "BF16"
comptime SAFETENSORS_DTYPE_F32 = "F32"


def _safetensors_element_size(dtype: String) raises -> Int:
    if dtype == SAFETENSORS_DTYPE_BF16:
        return 2
    if dtype == SAFETENSORS_DTYPE_F32:
        return 4
    raise Error("safetensors: unsupported tensor dtype '" + dtype + "'")


struct SafetensorsFile:
    """A parsed safetensors header, ready to read individual tensors by name.

    Mirrors `llmm.checkpointing`'s peek-then-read shape: construct once to
    discover every tensor's name/dtype/shape (cheap — only the JSON header is
    read), then call `read_tensor` per tensor to pull its bytes off disk.
    """

    var path: String
    var data_start: Int  # Absolute file offset where tensor bytes begin.
    var tensors: Dict[String, TensorInfo]

    def __init__(out self, path: String) raises:
        self.path = path
        self.tensors = Dict[String, TensorInfo]()
        self.data_start = 0

        var file = open(path, "r")
        var len_bytes = file.read_bytes(_HEADER_LEN_BYTES)
        if len(len_bytes) < _HEADER_LEN_BYTES:
            file.close()
            raise Error("safetensors error: truncated header length in " + path)
        var header_len = 0
        for i in range(_HEADER_LEN_BYTES):
            header_len |= Int(len_bytes[i]) << (8 * i)

        var header_bytes = file.read_bytes(header_len)
        file.close()
        if len(header_bytes) < header_len:
            raise Error("safetensors error: truncated JSON header in " + path)

        self.data_start = _HEADER_LEN_BYTES + header_len
        var header_json = String(unsafe_from_utf8=header_bytes)
        self.tensors = _parse_header(header_json)

    def get_info(self, name: String) raises -> TensorInfo:
        if name not in self.tensors:
            raise Error("safetensors error: tensor not found: " + name)
        return self.tensors[name].copy()

    def read_tensor[
        dtype: DType
    ](
        self,
        name: String,
        dest: MutMemPtr[dtype],
        transpose_rows: Int = -1,
        transpose_cols: Int = -1,
    ) raises:
        """Read tensor `name` into `dest` (already-sized for its element count).

        If `transpose_rows`/`transpose_cols` are given (both >= 0), the source
        is treated as a row-major (transpose_cols, transpose_rows) matrix (HF's
        export transposes `nn.Linear`-style weights relative to our own
        internal layout — see `scripts/export_to_hf.py`) and un-transposed
        while copying, so `dest` ends up row-major (transpose_rows,
        transpose_cols).
        """
        var info = self.get_info(name)
        var elem_size = _safetensors_element_size(info.dtype)
        if elem_size != get_dtype_size(dtype):
            raise Error(
                "safetensors error: tensor '"
                + name
                + "' has dtype "
                + info.dtype
                + " ("
                + String(elem_size)
                + " bytes/elem), but the destination buffer expects "
                + String(get_dtype_size(dtype))
                + " bytes/elem — rebuild with a matching precision"
            )
        var n = info.num_elements()
        if info.num_bytes() != n * elem_size:
            raise Error(
                "safetensors error: tensor '"
                + name
                + "' byte range doesn't match its declared shape"
            )

        var file = open(self.path, "r")
        var seek_pos = self.data_start + info.offset_begin
        _ = file.seek(UInt64(seek_pos))
        var raw = file.read_bytes(info.num_bytes())
        file.close()
        if len(raw) < info.num_bytes():
            raise Error(
                "safetensors error: truncated tensor data for '" + name + "'"
            )
        var src_ptr = raw.unsafe_ptr().bitcast[Scalar[dtype]]()

        if transpose_rows < 0:
            for i in range(n):
                dest.store(i, src_ptr.load(i))
            return

        # Source is (transpose_cols, transpose_rows) row-major; write dest as
        # (transpose_rows, transpose_cols) row-major: dest[r, c] = src[c, r].
        if transpose_rows * transpose_cols != n:
            raise Error(
                "safetensors error: transpose dims ("
                + String(transpose_rows)
                + ", "
                + String(transpose_cols)
                + ") don't match tensor '"
                + name
                + "'s element count ("
                + String(n)
                + ")"
            )
        for r in range(transpose_rows):
            for c in range(transpose_cols):
                dest.store(
                    r * transpose_cols + c,
                    src_ptr.load(c * transpose_rows + r),
                )


# ===----------------------------------------------------------------------=== #
# HuggingFace config.json — a flat GPT2Config JSON, sibling to model.safetensors
# ===----------------------------------------------------------------------=== #


def _find_hf_config_field(text: String, key: String) raises -> Int:
    """Parse `"<key>": <int>` anywhere in a flat top-level JSON object and
    return the int. Not a general JSON object walker — config.json's fields
    we care about are always flat integers at the top level, so a linear
    key-then-colon-then-int scan is sufficient and avoids re-parsing the
    whole (much larger, dropout/activation-function-laden) document.
    """
    var bytes = text.as_bytes()
    var needle = '"' + key + '"'
    var needle_bytes = needle.as_bytes()
    var i = 0
    while i + len(needle_bytes) <= len(bytes):
        var matched = True
        for j in range(len(needle_bytes)):
            if bytes[i + j] != needle_bytes[j]:
                matched = False
                break
        if matched:
            var pos = i + len(needle_bytes)
            _skip_ws(text, pos)
            _expect(text, pos, 58)  # ':'
            _skip_ws(text, pos)
            return _parse_json_int(text, pos)
        i += 1
    raise Error(
        "HuggingFace config.json: missing or non-integer field '" + key + "'"
    )


def read_hf_gpt2_config(config_json_path: String) raises -> CheckpointConfig:
    """Read a HuggingFace GPT2Config JSON (the `config.json` published
    alongside `model.safetensors`, e.g. by `scripts/export_to_hf.py`) into our
    own `CheckpointConfig`.

    `padded_vocab_size` isn't a HF concept — HF's export drops our padding
    columns entirely (see `allocate_parameters_from_safetensors`) — so it's
    recomputed here with the same %128 rounding `SCRATCH_PADDED_VOCAB_SIZE`
    uses for a from-scratch d12 build (50257 -> 50304).
    """
    var file = open(config_json_path, "r")
    var raw = file.read_bytes()  # size defaults to -1: read to EOF.
    file.close()
    var text = String(unsafe_from_utf8=raw)

    var vocab_size = _find_hf_config_field(text, "vocab_size")
    var num_layer = _find_hf_config_field(text, "n_layer")
    var num_heads = _find_hf_config_field(text, "n_head")
    var channels = _find_hf_config_field(text, "n_embd")
    var max_seq_len = _find_hf_config_field(text, "n_positions")
    var padded_vocab_size = ceildiv(vocab_size, 128) * 128

    return CheckpointConfig(
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        num_layer=num_layer,
        num_heads=num_heads,
        channels=channels,
        padded_vocab_size=padded_vocab_size,
    )
