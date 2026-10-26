# ===----------------------------------------------------------------------=== #
# Terminal output style shared by the trainer, the verify gate and inference
# ===----------------------------------------------------------------------=== #
#
# Which output format to use (llmm/log.mojo renders the events):
#
#   LLMM_LOG=llm     key=value lines for coding agents and LLMs: no color,
#                    glyphs or redraws (`agent` is an alias)
#   LLMM_LOG=pretty  the compact colored layout the README demo shows
#   LLMM_LOG=json    one JSON object per event
#   LLMM_LOG=plain   llm.c's original lines; scripts/benchmark_zero.py sets it
#   NO_COLOR=1       keep the layout, drop the colors (https://no-color.org)
#
# Unset, the format is `llm` when an LLM harness is detected and `pretty`
# otherwise: an explicit LLMM_LOG wins, then detection, then the default.
# Color and counters redrawn in place also need a terminal, so piped pretty
# output is the same layout as plain text.

from std.ffi import external_call
from std.os import getenv

comptime _STDOUT_FD = 1

# Set by LLM coding harnesses in the shells they run commands in. AI_AGENT is
# the cross-tool convention; the rest are per-tool. scripts/test_log.py keeps
# the same list.
comptime _LLM_HARNESS_VARS: List[StaticString] = [
    "AI_AGENT",
    "CLAUDECODE",
    "CODEX_SANDBOX",
    "GEMINI_CLI",
    "OPENCODE",
    "CURSOR_AGENT",
]

comptime _ACCENT = "\x1b[38;2;255;106;50m"
comptime _GOOD = "\x1b[38;2;74;222;128m"
comptime _BAD = "\x1b[38;2;248;113;113m"
comptime _DIM = "\x1b[38;2;148;155;170m"
comptime _BOLD = "\x1b[1m"
comptime _RESET = "\x1b[0m"

# Clear the current line and return to its start, for counters redrawn in place.
comptime REDRAW = "\r\x1b[2K"


def stdout_is_tty() -> Bool:
    return external_call["isatty", Int32](Int32(_STDOUT_FD)) == 1


def llm_harness_detected() -> Bool:
    for name in materialize[_LLM_HARNESS_VARS]():
        if getenv(String(name)) != "":
            return True
    return False


def log_format() -> String:
    """`llm`, `pretty`, `json` or `plain`; see the header for the order."""
    var requested = getenv("LLMM_LOG")
    if requested == "agent":
        return "llm"
    if (
        requested == "llm"
        or requested == "pretty"
        or requested == "json"
        or requested == "plain"
    ):
        return requested
    return "llm" if llm_harness_detected() else "pretty"


def plain_output() -> Bool:
    """Whether to print llm.c's original lines (setup tables and all)."""
    return log_format() == "plain"


def pretty_output() -> Bool:
    return log_format() == "pretty"


def redraw_in_place() -> Bool:
    """Whether counters may redraw their line in place (a terminal only)."""
    return pretty_output() and stdout_is_tty()


@fieldwise_init
struct Style(Copyable, ImplicitlyCopyable, Movable):
    """Color helpers that collapse to plain text when color is off."""

    var color: Bool

    @staticmethod
    def detect() -> Self:
        return Self(redraw_in_place() and getenv("NO_COLOR") == "")

    def _wrap(self, code: StaticString, text: String) -> String:
        if not self.color:
            return text
        return String(code) + text + _RESET

    def accent(self, text: String) -> String:
        return self._wrap(_ACCENT, text)

    def good(self, text: String) -> String:
        return self._wrap(_GOOD, text)

    def bad(self, text: String) -> String:
        return self._wrap(_BAD, text)

    def dim(self, text: String) -> String:
        return self._wrap(_DIM, text)

    def bold(self, text: String) -> String:
        return self._wrap(_BOLD, text)

    def heading(self, title: String, detail: String) -> String:
        """A section line: `▸ title  detail`."""
        return (
            self.accent("▸") + " " + self.bold(title) + "  " + self.dim(detail)
        )


def pad_left(text: String, width: Int) -> String:
    var out = text
    while out.byte_length() < width:
        out = " " + out
    return out


def pad_right(text: String, width: Int) -> String:
    var out = text
    while out.byte_length() < width:
        out += " "
    return out
