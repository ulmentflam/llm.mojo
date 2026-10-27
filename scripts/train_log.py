"""Read a trainer log back, whatever LLMM_LOG format wrote it (or llm.c).

The trainer reports steps and val losses in four formats (llmm/log.mojo), and
llm.c's CPU trainer in a fifth:

    plain   step 3/40 | loss 4.450004 (nanz)| … | 230.12 ms | … | 1105 tok/s
            val loss 4.3308616
    pretty    step  3/40  loss 4.4500    230.1 ms    1105 tok/s
              val loss  4.3309
    llm     step step=3 steps=40 loss=4.450004 ms=230.12 tok_s=1105
            val step=20 loss=4.3308616
    json    {"event": "step", "step": 3, "steps": 40, "loss": 4.450004, ...}
            {"event": "val", "step": 20, "loss": 4.3308616}
    llm.c   step 2: train loss 4.450004 (took 412.31 ms)
            val loss 4.330861

All of them (mixed, even) become the same events:

    ("step",   {"step": 3, "steps": 40, "loss": 4.45, "ms": 230.12, "tok_s": 1105})
    ("val",    {"step": 20, "loss": 4.3308616})
    ("sample", {"step": 40, "text": "SENATOR MARGARET:\\n…"})

llm.c counts steps from 0 and prints no total or tok/s, so its steps are
renumbered from 1 with `steps` and `tok_s` left out. A plain, pretty or llm.c
val line carries no step; it gets the last step seen, which is the step it was
evaluated after. The `generating:` / `---` sample block of plain and llm.c
output becomes one sample event. Color codes are stripped first, so a captured
terminal session parses too.

`read_events` and `steps_and_vals` read a whole log; `LogReader` takes it a
line at a time (scripts/log_view.py renders a live run with it).
"""

import json
import re
from collections.abc import Iterator

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_PLAIN_STEP = re.compile(
    r"^step (\d+)/(\d+) \| loss ([0-9.naif+-]+).*?\| ([0-9.]+) ms .*?\| (\d+) tok/s"
)
_PRETTY_STEP = re.compile(
    r"^\s+step\s+(\d+)/(\d+)\s+loss ([0-9.naif+-]+)\s+([0-9.]+) ms\s+(\d+) tok/s"
)
_LLMC_STEP = re.compile(
    r"^step (\d+): train loss ([0-9.naif+-]+) \(took ([0-9.]+) ms\)"
)
_TEXT_VAL = re.compile(r"^\s*val loss\s+([0-9.naif+-]+)")
_LLM_PAIR = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')


def _number(text: str):
    try:
        return int(text)
    except ValueError:
        return float(text)


def _llm_fields(line: str) -> dict:
    fields = {}
    for key, value in _LLM_PAIR.findall(line):
        fields[key] = json.loads(value) if value.startswith('"') else _number(value)
    return fields


class LogReader:
    """Turns log lines into (kind, fields) events, one line at a time."""

    def __init__(self) -> None:
        self.last_step = 0
        self._sample: list[str] | None = None  # inside a generating: block
        self._sample_open = False  # past its opening ---

    @property
    def in_sample(self) -> bool:
        """Inside a generating: block (its text may still be arriving)."""
        return self._sample is not None

    @property
    def in_sample_text(self) -> bool:
        """Past the block's opening ---, so new text is sample text."""
        return self._sample is not None and self._sample_open

    def feed(self, raw: str) -> list[tuple[str, dict]]:
        line = _ANSI.sub("", raw).replace("\r", "").rstrip("\n")
        if self._sample is not None:
            return self._feed_sample(line)
        if line.strip() == "generating:":
            self._sample, self._sample_open = [], False
            return []

        kind: str | None = None
        fields: dict = {}
        if line.startswith("{"):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return []
            kind = event.pop("event", None)
            fields = event
        elif line.startswith(("step step=", "step t=")):
            kind, fields = "step", _llm_fields(line)
        elif line.startswith(("val step=", "val t=")):
            kind, fields = "val", _llm_fields(line)
        elif line.startswith(("sample step=", "sample t=")):
            kind, fields = "sample", _llm_fields(line)
        elif m := _PLAIN_STEP.match(line) or _PRETTY_STEP.match(line):
            kind = "step"
            fields = {
                "step": int(m[1]),
                "steps": int(m[2]),
                "loss": float(m[3]),
                "ms": float(m[4]),
                "tok_s": int(m[5]),
            }
        elif m := _LLMC_STEP.match(line):
            kind = "step"
            fields = {"step": int(m[1]) + 1, "loss": float(m[2]), "ms": float(m[3])}
        elif m := _TEXT_VAL.match(line):
            kind, fields = "val", {"step": self.last_step, "loss": float(m[1])}

        if kind == "step":
            self.last_step = int(fields["step"])
        if kind in ("step", "val", "sample"):
            return [(kind, fields)]
        return []

    def _feed_sample(self, line: str) -> list[tuple[str, dict]]:
        assert self._sample is not None
        if line.strip() == "---":
            if not self._sample_open:
                self._sample_open = True
                return []
            text = "\n".join(self._sample)
            self._sample = None
            return [("sample", {"step": self.last_step, "text": text})]
        self._sample.append(line)
        return []


def read_events(
    text: str, kinds: tuple[str, ...] = ("step", "val")
) -> Iterator[tuple[str, dict]]:
    reader = LogReader()
    for raw in text.splitlines():
        for kind, fields in reader.feed(raw):
            if kind in kinds:
                yield kind, fields


def steps_and_vals(text: str) -> tuple[list[dict], list[dict]]:
    steps, vals = [], []
    for kind, fields in read_events(text):
        (steps if kind == "step" else vals).append(fields)
    return steps, vals
