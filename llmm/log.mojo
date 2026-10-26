# ===----------------------------------------------------------------------=== #
# Structured run output: one event, four formats, four levels
# ===----------------------------------------------------------------------=== #
#
# Every reportable thing the trainer and `make verify` print (a step, a val
# loss, a check, a sample) is an event: a kind, a level and typed fields, plus
# the call site's own pretty and plain renderings, since those are layouts.
# `Log.emit` prints it in the format llmm/term.mojo picked:
#
#   llm     step step=21/40 loss=4.520491 ms=233.71 tok_s=1093
#   json    {"event": "step", "step": 21, "steps": 40, "loss": 4.520491, ...}
#   pretty  the call site's compact line (colored on a terminal)
#   plain   the call site's llm.c line
#
#   LLMM_LOG_LEVEL=quiet|normal|verbose|debug  (or 0-3; default normal)
#       quiet: results and failures. verbose: extra fields (norm, lr,
#       z-scores, MFU, per-tensor gradient stats). debug: element dumps.
#   LLMM_LOG_FILE=path  also append every event, up to verbose, as JSON lines
#       to `path`, whatever the screen format.
#   LLMM_LOG_TIME=1     time since the run began: a `t` field in llm/json, a
#       `[   12.3s]` prefix in pretty/plain.
#
# scripts/test_log.py mirrors this for `make test-mojo`, and
# scripts/train_log.py parses every format back.

from std.math import isinf, isnan
from std.os import getenv
from std.time import perf_counter_ns

from llmm.term import Style, log_format, redraw_in_place, REDRAW

comptime QUIET = 0
comptime NORMAL = 1
comptime VERBOSE = 2
comptime DEBUG = 3


def log_level() -> Int:
    var raw = getenv("LLMM_LOG_LEVEL", "normal")
    if raw == "quiet" or raw == "0":
        return QUIET
    if raw == "verbose" or raw == "2":
        return VERBOSE
    if raw == "debug" or raw == "3":
        return DEBUG
    return NORMAL


@fieldwise_init
struct Field(Copyable, Movable):
    """One `key=value` of an event.

    `level` hides the field below that log level (llm and json formats), so a
    step line stays short at `normal` and gains norm/lr/MFU at `verbose`.
    """

    var key: String
    var value: String
    var number: Bool
    var level: Int


def num(key: String, value: Int, level: Int = NORMAL) -> Field:
    return Field(key, String(value), True, level)


def num(key: String, value: Float64, level: Int = NORMAL) -> Field:
    return Field(key, String(value), True, level)


def num(key: String, value: Float32, level: Int = NORMAL) -> Field:
    return Field(key, String(value), True, level)


def num(key: String, formatted: String, level: Int = NORMAL) -> Field:
    """A number already formatted by the caller (fixed decimals, say)."""
    return Field(key, formatted, True, level)


def text(key: String, value: String, level: Int = NORMAL) -> Field:
    return Field(key, value, False, level)


def _append(mut out: List[UInt8], escape: StaticString):
    for b in escape.as_bytes():
        out.append(b)


def _json_string(value: String) -> String:
    var out = List[UInt8]()
    var hex = StaticString("0123456789abcdef").as_bytes()
    out.append(UInt8(ord('"')))
    for byte in value.as_bytes():
        var b = Int(byte)
        if b == ord('"'):
            _append(out, '\\"')
        elif b == ord("\\"):
            _append(out, "\\\\")
        elif b == ord("\n"):
            _append(out, "\\n")
        elif b == ord("\r"):
            _append(out, "\\r")
        elif b == ord("\t"):
            _append(out, "\\t")
        elif b < 0x20:
            _append(out, "\\u00")
            out.append(hex[b >> 4])
            out.append(hex[b & 15])
        else:
            out.append(byte)
    out.append(UInt8(ord('"')))
    return String(unsafe_from_utf8=out)


def _is_finite_number(value: String) -> Bool:
    return not (
        value == "nan" or value == "inf" or value == "-inf" or value == ""
    )


def _llm_value(field: Field) -> String:
    if field.number:
        return field.value
    for byte in field.value.as_bytes():
        var b = Int(byte)
        var bare = (
            (b >= ord("a") and b <= ord("z"))
            or (b >= ord("A") and b <= ord("Z"))
            or (b >= ord("0") and b <= ord("9"))
            or b == ord("_")
            or b == ord(".")
            or b == ord("/")
            or b == ord(":")
            or b == ord("%")
            or b == ord("+")
            or b == ord("-")
        )
        if not bare:
            return _json_string(field.value)
    if field.value.byte_length() == 0:
        return '""'
    return field.value


def _json_value(field: Field) -> String:
    if field.number:
        return field.value if _is_finite_number(field.value) else "null"
    return _json_string(field.value)


struct Log(Copyable, Movable):
    var format: String
    var level: Int
    var file: String
    var timestamps: Bool
    var start_ns: Int
    var style: Style
    var enabled: Bool  # rank 0 reports; other ranks stay silent

    def __init__(out self, enabled: Bool = True):
        self.format = log_format()
        self.level = log_level()
        self.file = getenv("LLMM_LOG_FILE")
        var t = getenv("LLMM_LOG_TIME")
        self.timestamps = t != "" and t != "0"
        self.start_ns = perf_counter_ns()
        self.style = Style.detect()
        self.enabled = enabled

    def plain(self) -> Bool:
        return self.format == "plain"

    def pretty(self) -> Bool:
        return self.format == "pretty"

    def llm(self) -> Bool:
        return self.format == "llm"

    def json(self) -> Bool:
        return self.format == "json"

    def at(self, level: Int) -> Bool:
        """Whether this rank prints events at `level`."""
        return self.enabled and level <= self.level

    def _seconds(self) -> String:
        var tenths = Int((perf_counter_ns() - self.start_ns) // 100_000_000)
        return String(tenths // 10) + "." + String(tenths % 10)

    def _json_line(
        self, kind: String, fields: List[Field], level: Int
    ) -> String:
        var out = String('{"event": ') + _json_string(kind)
        if self.timestamps:
            out += ', "t": ' + self._seconds()
        for f in fields:
            if f.level <= level:
                out += ", " + _json_string(f.key) + ": " + _json_value(f)
        return out + "}"

    def emit(
        self,
        kind: String,
        level: Int,
        fields: List[Field],
        pretty: String = "",
        plain: String = "",
    ) raises:
        """Report one event; an empty pretty/plain text prints nothing there."""
        if not self.enabled:
            return
        if self.file != "" and level <= max(self.level, VERBOSE):
            with open(self.file, "a") as f:
                f.write(
                    self._json_line(kind, fields, max(self.level, VERBOSE))
                    + "\n"
                )
        if level > self.level:
            return
        if self.json():
            print(self._json_line(kind, fields, self.level))
        elif self.llm():
            var out = kind
            if self.timestamps:
                out += " t=" + self._seconds()
            for f in fields:
                if f.level <= self.level:
                    out += " " + f.key + "=" + _llm_value(f)
            print(out)
        else:
            var body = pretty if self.pretty() else plain
            if body.byte_length() == 0:
                return
            if self.timestamps:
                var stamp = "[" + self._seconds() + "s] "
                while stamp.byte_length() < 11:
                    stamp = " " + stamp
                body = self.style.dim(stamp) + body
            # Clear a counter line a redraw may have left the cursor on.
            print((REDRAW if redraw_in_place() else "") + body)

    def redraw(self, level: Int, body: String):
        """Redraw a pretty in-place counter (a terminal only)."""
        if self.at(level) and self.pretty() and redraw_in_place():
            print(REDRAW + body, end="")

    def raw(self, level: Int, line: String):
        """Free text (a failure dump, say) outside json, where events carry it.
        """
        if self.at(level) and not self.json():
            print((REDRAW if redraw_in_place() else "") + line)
