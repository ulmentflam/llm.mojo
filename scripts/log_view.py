#!/usr/bin/env python3
"""Run a GPT-2 trainer and show its run in the compact layout.

For the llm.c comparison video (docs/bench-vs-llmc.*): llm.c and llm.mojo
print differently, so both run through this viewer and read alike. It runs the
command under a pseudo-terminal (so llm.c's printf flushes per line rather
than at exit), reads each line with scripts/train_log.py (llm.mojo is asked
for LLMM_LOG=plain, llm.c needs nothing) and prints the same layout as
llm.mojo's own pretty format:

    ▸ llm.c  train_gpt2.c · CPU · 10 threads
      step  1/41  loss 4.6778  1794.9 ms  143 tok/s
      val loss  4.3309
    ▸ sample  after 20 steps
    …
      done  41 steps · val loss 4.2915 · 129.5 s

Step time is each trainer's own measurement; tok/s is computed here the same
way for both, tokens per step / step time, so the two columns compare. Lines
that look like errors pass through in red, and the exit code is the
trainer's.

Recording a terminal (vhs) loads the CPU the trainer is measured on, and not
evenly: llm.c's OpenMP threads lose far more to it than llm.mojo does. So the
video is made in two passes. LOG_VIEW_RECORD=<file> saves every line this
prints with its time since launch; LOG_VIEW_REPLAY=<file> then prints a saved
run back at its original timing instead of running anything, for the camera.

Usage: log_view.py --title NAME --detail TEXT --tokens B*T --steps N
                   [--show "command as displayed"] [--cwd DIR] -- command...
"""

import argparse
import codecs
import collections
import json
import os
import pty
import re
import subprocess
import sys
import textwrap
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_log import LogReader  # noqa: E402

RECORD = os.environ.get("LOG_VIEW_RECORD", "")
REPLAY = os.environ.get("LOG_VIEW_REPLAY", "")
# A recording keeps its colors for the replay that films it.
COLOR = (sys.stdout.isatty() or RECORD != "") and os.environ.get("NO_COLOR", "") == ""
FAILURE = re.compile(r"[Ee]rror|FAIL|Traceback|panic|Segmentation|[Aa]bort")
WIDTH = 46  # wrap width, inside a ~50-column pane


def paint(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def accent(t: str) -> str:
    return paint("38;2;255;106;50", t)


def good(t: str) -> str:
    return paint("38;2;74;222;128", t)


def bad(t: str) -> str:
    return paint("38;2;248;113;113", t)


def dim(t: str) -> str:
    return paint("38;2;148;155;170", t)


def bold(t: str) -> str:
    return paint("1", t)


def heading(title: str, detail: str) -> str:
    return f"{accent('▸')} {bold(title)}  {dim(detail)}"


class Out:
    """Prints text, and with LOG_VIEW_RECORD saves each write with its time."""

    def __init__(self) -> None:
        self.start = time.time()
        self.file = open(RECORD, "w") if RECORD else None

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        if self.file:
            entry = {"t": round(time.time() - self.start, 3), "w": text}
            self.file.write(json.dumps(entry) + "\n")

    def line(self, text: str = "") -> None:
        self.write(text + "\n")

    def close(self, rc: int) -> None:
        if self.file:
            self.file.write(
                json.dumps({"t": time.time() - self.start, "rc": rc}) + "\n"
            )
            self.file.close()


EOT = "<|endoftext|>"


class SampleStream:
    """Prints a sample as its tokens arrive, as a terminal would show them.

    Each <|endoftext|> becomes a line break, lines are indented under the
    `▸ sample` heading and word-wrapped to WIDTH, and blank lines are dropped.
    """

    def __init__(self, out: Out) -> None:
        self.out = out
        self.active = False
        self.col = 0  # 0 = at the start of a line

    def begin(self, step: int) -> None:
        self.out.line()
        self.out.line(heading("sample", f"after {step} steps"))
        self.active, self.col = True, 0

    def text(self, chunk: str) -> None:
        for i, part in enumerate(chunk.replace(EOT, "\n").split("\n")):
            if i > 0 and self.col:
                self.out.write("\n")
                self.col = 0
            for word in re.findall(r"\s*\S+|\s+", part):
                if self.col == 0:
                    word = word.lstrip()
                    if not word:
                        continue
                    self.out.write("  ")
                    self.col = 2
                elif self.col + len(word) > WIDTH + 2 and word.strip():
                    self.out.write("\n  ")
                    self.col = 2
                    word = word.lstrip()
                self.out.write(word)
                self.col += len(word)

    def end(self) -> None:
        if self.col:
            self.out.write("\n")
        self.out.line()
        self.active, self.col = False, 0


def unfinished_marker(text: str) -> int:
    """Length of a trailing partial <|endoftext|> or closing --- in `text`."""
    for n in range(min(len(EOT) - 1, len(text)), 0, -1):
        if EOT.startswith(text[-n:]):
            return n
    stripped = text.rstrip("-")
    return len(text) - len(stripped) if stripped == "" else 0


class View:
    def __init__(self, out: Out, tokens: int, steps: int) -> None:
        self.out = out
        self.stream = SampleStream(out)
        self.tokens = tokens
        self.steps = steps
        self.width = len(str(steps))
        self.step_ms: list[float] = []
        self.last_val: float | None = None

    def show(self, kind: str, f: dict) -> None:
        if kind == "step":
            ms = float(f["ms"])
            self.step_ms.append(ms)
            tok_s = self.tokens * 1000.0 / ms if ms > 0 else 0.0
            step = str(f["step"]).rjust(self.width)
            self.out.line(
                f"  {dim('step')} {step}/{f.get('steps', self.steps)}"
                f"  {dim('loss')} {float(f['loss']):.4f}"
                f"  {dim(f'{ms:6.1f} ms')}  {tok_s:4.0f} {dim('tok/s')}"
            )
        elif kind == "val":
            self.last_val = float(f["loss"])
            self.out.line(f"  {accent('val loss')}  {bold(f'{self.last_val:.4f}')}")
        elif kind == "sample" and self.stream.active:
            self.stream.end()  # its text was streamed as it arrived
        elif kind == "sample":
            self.out.line()
            self.out.line(heading("sample", f"after {f['step']} steps"))
            for doc in str(f["text"]).split("<|endoftext|>"):
                for line in doc.splitlines():
                    for wrapped in textwrap.wrap(line.strip(), WIDTH):
                        self.out.line(f"  {wrapped}")
            self.out.line()

    def done(self, seconds: float) -> None:
        parts = [f"{len(self.step_ms)} steps"]
        if self.last_val is not None:
            parts.append(f"val loss {self.last_val:.4f}")
        parts.append(f"{seconds:.1f} s")
        self.out.line(f"  {good('done')}  " + dim(" · ").join(parts))


def run(cmd: list[str], view: View, cwd: str | None) -> int:
    out = view.out
    env = dict(os.environ, LLMM_LOG="plain")
    master, slave = pty.openpty()
    start = time.time()
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=slave, stderr=slave, env=env, cwd=cwd
    )
    os.close(slave)
    reader = LogReader()
    decode = codecs.getincrementaldecoder("utf-8")(errors="replace").decode
    pending = ""
    streamed = 0  # characters of pending's partial line already streamed
    # Lines not rendered as events, kept so a failed run can show why: a
    # loader or OpenMP abort matches no error pattern.
    unshown: collections.deque[str] = collections.deque(maxlen=20)
    while True:
        try:
            data = os.read(master, 4096)
        except OSError:  # EIO: the child closed the terminal
            break
        if not data:
            break
        pending += decode(data)
        while "\n" in pending:
            raw, pending = pending.split("\n", 1)
            line = raw.rstrip("\r")
            if reader.in_sample_text and line != "---":
                view.stream.text(line[streamed:] + "\n")
            streamed = 0
            was_in_sample = reader.in_sample
            events = reader.feed(line)
            if reader.in_sample and not was_in_sample:
                view.stream.begin(reader.last_step)
            for kind, fields in events:
                view.show(kind, fields)
            if events:
                continue
            if reader.in_sample:
                continue
            if FAILURE.search(line):
                out.line(bad(line))
            else:
                unshown.append(line)
        # Stream the sample's partial line now, holding back an unfinished
        # <|endoftext|> or closing --- until the rest of it arrives.
        if reader.in_sample_text and pending:
            safe = len(pending) - unfinished_marker(pending)
            if safe > streamed:
                view.stream.text(pending[streamed:safe].replace("\r", ""))
                streamed = safe
    if pending:
        unshown.append(pending.rstrip("\r"))
    os.close(master)
    rc = proc.wait()
    if rc == 0:
        view.done(time.time() - start)
    else:
        out.line(f"  {bad('FAILED')}  {dim(f'exit {rc}; last output:')}")
        for line in unshown:
            out.line(f"    {line}")
    return rc


def replay(path: str) -> int:
    """Print a LOG_VIEW_RECORD file back at the pace it was recorded."""
    start = time.time()
    rc = 0
    with open(path) as f:
        for row in f:
            entry = json.loads(row)
            delay = entry["t"] - (time.time() - start)
            if delay > 0:
                time.sleep(delay)
            if "rc" in entry:
                rc = entry["rc"]
            elif "w" in entry:
                sys.stdout.write(entry["w"])
                sys.stdout.flush()
            else:  # a recording made before partial writes were saved
                print(entry["line"], flush=True)
    return rc


def main() -> int:
    if REPLAY:
        return replay(REPLAY)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--title", required=True)
    ap.add_argument("--detail", default="")
    ap.add_argument("--tokens", type=int, required=True, help="tokens per step, B*T")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--show", help="the command as displayed (default: the real one)")
    ap.add_argument("--cwd", help="directory to run the command in")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("no command given after --")
    out = Out()
    out.line(heading(args.title, args.detail))
    shown = textwrap.wrap(
        args.show or " ".join(cmd),
        WIDTH,
        initial_indent="  $ ",
        subsequent_indent="      ",
        break_on_hyphens=False,
    )
    for line in shown:
        out.line(dim(line))
    rc = run(cmd, View(out, args.tokens, args.steps), args.cwd)
    out.close(rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
