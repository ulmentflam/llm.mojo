#!/usr/bin/env python3
"""Run one compiled Mojo test file and report it as log events.

`make test-mojo` calls this per tests/test_*.mojo. It runs the test binary,
reads Mojo's TestSuite output and reports one event per file in the format and
level the trainer and `make verify` use (llmm/log.mojo, which this mirrors):

    LLMM_LOG        llm | pretty | json | plain  (agent = llm)
                    llm is the default under an LLM coding harness, pretty
                    otherwise; plain passes the raw TestSuite output through
    LLMM_LOG_LEVEL  quiet | normal (default) | verbose | debug
                    quiet: failures and the summary; verbose: every test case;
                    debug: each file's raw output too
    LLMM_LOG_FILE   also append every event as a JSON line to this file
    LLMM_LOG_TIME   1 = prefix each event with the time since the run began

pretty:  ▸ test_zero                    14 passed · 1 skipped  4s
llm:     test file=test_zero status=pass passed=14 skipped=1 failed=0 s=4
json:    {"event": "test", "file": "test_zero", "status": "pass", ...}

Tests that return early behind an NVIDIA gate print "SKIP <file>::<test>:
<reason>" and still count as passed in TestSuite's own summary; here they
count as skipped. A file that fails, times out or crashes always reports its
full raw output, at every level.

    test_log.py --build <test file> -- <command that compiles it...>
    test_log.py <test file> -- <command that runs the binary...>
    test_log.py --summary <tally file>

The per-file counts are appended to $TEST_TALLY for the closing summary, and
$TEST_RUN_START (epoch seconds) anchors the LLMM_LOG_TIME clock.
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

# ----------------------------------------------------------------------------
# Logging: a Python mirror of llmm/log.mojo and llmm/term.mojo's
# log_format(). Keep them in step.
# ----------------------------------------------------------------------------

AGENT_MARKERS = (
    "AI_AGENT",
    "CLAUDECODE",
    "CODEX_SANDBOX",
    "GEMINI_CLI",
    "OPENCODE",
    "CURSOR_AGENT",
)
LEVELS = {"quiet": 0, "normal": 1, "verbose": 2, "debug": 3}
QUIET, NORMAL, VERBOSE, DEBUG = range(4)


def log_format() -> str:
    fmt = os.environ.get("LLMM_LOG", "")
    if fmt == "agent":
        return "llm"
    if fmt in ("llm", "pretty", "json", "plain"):
        return fmt
    return "llm" if any(os.environ.get(v) for v in AGENT_MARKERS) else "pretty"


def log_level() -> int:
    raw = os.environ.get("LLMM_LOG_LEVEL", "normal")
    return LEVELS.get(raw, int(raw) if raw.isdigit() else NORMAL)


FORMAT = log_format()
LEVEL = log_level()
LOG_FILE = os.environ.get("LLMM_LOG_FILE", "")
TIMESTAMPS = os.environ.get("LLMM_LOG_TIME", "") not in ("", "0")
RUN_START = float(os.environ.get("TEST_RUN_START", time.time()))
TTY = sys.stdout.isatty() and FORMAT == "pretty"  # redraw in place only here
COLOR = TTY and os.environ.get("NO_COLOR", "") == ""


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


def llm_value(value) -> str:
    text = str(value)
    return text if re.fullmatch(r"[\w./:%+-]+", text) else json.dumps(text)


def emit(kind: str, level: int, pretty: str, **fields) -> None:
    """Report one event in the screen format, and to LLMM_LOG_FILE."""
    elapsed = time.time() - RUN_START
    if LOG_FILE and level <= max(LEVEL, VERBOSE):
        event: dict[str, object] = {"event": kind}
        if TIMESTAMPS:
            event["t"] = round(elapsed, 1)
        event.update(fields)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
    if level > LEVEL:
        return
    redraw = "\r\033[2K" if TTY else ""
    if FORMAT == "json":
        event: dict[str, object] = {"event": kind}
        if TIMESTAMPS:
            event["t"] = round(elapsed, 1)
        event.update(fields)
        print(json.dumps(event), flush=True)
    elif FORMAT == "llm":
        pairs = [f"t={elapsed:.1f}"] if TIMESTAMPS else []
        # A failure's output follows the line as indented raw text.
        pairs += [f"{k}={llm_value(v)}" for k, v in fields.items() if k != "output"]
        print(" ".join([kind, *pairs]), flush=True)
    else:
        prefix = dim(f"[{elapsed:7.1f}s] ") if TIMESTAMPS else ""
        print(redraw + prefix + pretty, flush=True)


def emit_raw(lines: list[str]) -> None:
    """A failure's (or debug's) raw output, indented under its event."""
    if FORMAT == "json":
        return  # carried in the event's `output` field instead
    sys.stdout.writelines("    " + t for t in lines)
    sys.stdout.flush()


def status_line(text: str) -> None:
    """A transient 'compiling…' / 'running…' line, pretty terminals only."""
    if TTY and LEVEL >= NORMAL:
        print("\r\033[2K" + text, end="", flush=True)


# ----------------------------------------------------------------------------
# TestSuite parsing
# ----------------------------------------------------------------------------

RESULT = re.compile(r"^\s+(PASS|FAIL|SKIP) \[\s*[\d.]+\s*\] (\S+)")
# Files with their own main() (test_encoder_row_sparse) print "<test> PASSED".
OWN_PASS = re.compile(r"^(test_\w+) PASSED$")
GATED = re.compile(r"^SKIP (\S+?)(?:::(\S+))?: (.*)$")
NAME_WIDTH = 28


def duration(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def line(name: str, status: str, seconds: float) -> str:
    return f"{accent('▸')} {bold(name.ljust(NAME_WIDTH))} {status}  {dim(duration(seconds))}"


def record(passed: int, skipped: int, failed: int, status: str) -> None:
    tally = os.environ.get("TEST_TALLY")
    if tally:
        with open(tally, "a") as f:
            f.write(f"{passed} {skipped} {failed} {status}\n")


def build_file(path: str, cmd: list[str]) -> int:
    """Compile quietly; on failure report the file and show the compiler."""
    name = os.path.basename(path).removesuffix(".mojo")
    start = float(os.environ.get("TEST_START", time.time()))
    if FORMAT == "plain":
        return subprocess.call(cmd)
    status_line(f"{accent('▸')} {bold(name)} {dim('compiling…')}")
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    if proc.returncode != 0:
        why = "compile timed out" if proc.returncode == 124 else "compile failed"
        elapsed = time.time() - start
        out = proc.stdout.splitlines(True)
        emit(
            "test",
            QUIET,
            line(name, bad(f"FAILED · {why}"), elapsed),
            file=name,
            status="fail",
            reason=why,
            s=round(elapsed),
            output="".join(out),
        )
        emit_raw(out)
        record(0, 0, 0, "fail")
    return proc.returncode


def run_file(path: str, cmd: list[str]) -> int:
    name = os.path.basename(path).removesuffix(".mojo")
    start = float(os.environ.get("TEST_START", time.time()))
    if FORMAT != "plain":
        status_line(f"{accent('▸')} {bold(name)} {dim('running…')}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    raw: list[str] = []
    assert proc.stdout is not None
    for text in proc.stdout:
        raw.append(text)
        if FORMAT == "plain":
            sys.stdout.write(text)
            sys.stdout.flush()
    rc = proc.wait()
    elapsed = time.time() - start

    if FORMAT == "plain":
        return rc

    outcomes: dict[str, str] = {}
    gated: dict[str, str] = {}  # test -> reason
    file_skip = None
    for text in raw:
        if m := RESULT.match(text):
            outcomes[m[2]] = m[1]
        elif m := OWN_PASS.match(text.rstrip("\n")):
            outcomes[m[1]] = "PASS"
        elif m := GATED.match(text.rstrip("\n")):
            if m[2]:
                gated[m[2]] = m[3]
            else:
                file_skip = m[3]

    # A gated early return reports PASS; a TestSuite skip reports SKIP. Either
    # way the reason comes from the test's own SKIP line when it printed one.
    def case_status(test: str, outcome: str) -> str:
        if outcome == "FAIL":
            return "fail"
        return "skip" if outcome == "SKIP" or test in gated else "pass"

    statuses = {t: case_status(t, o) for t, o in outcomes.items()}
    skip_reasons = Counter(
        gated.get(t, "skipped by the suite") for t, s in statuses.items() if s == "skip"
    )
    failed = sum(s == "fail" for s in statuses.values())
    skipped = sum(s == "skip" for s in statuses.values())
    passed = len(statuses) - failed - skipped
    secs = round(elapsed)

    if rc != 0 or failed:
        why = (
            "timed out"
            if rc == 124
            else f"{failed} of {len(outcomes)} failed"
            if failed
            else f"exit {rc}"
        )
        emit(
            "test",
            QUIET,
            line(name, bad(f"FAILED · {why}"), elapsed),
            file=name,
            status="fail",
            reason=why,
            passed=passed,
            skipped=skipped,
            failed=failed,
            s=secs,
            output="".join(raw),
        )
        emit_raw(raw)
        status = "fail"
    elif file_skip is not None and not outcomes:
        emit(
            "test",
            NORMAL,
            line(name, dim(f"skipped · {file_skip}"), elapsed),
            file=name,
            status="skip",
            reason=file_skip,
            s=secs,
        )
        status = "skip"
    else:
        pretty = good(f"{passed} passed") if outcomes else good("passed")
        if skipped:
            pretty += dim(f" · {skipped} skipped")
        if not outcomes:
            pretty += dim(" · no per-test report")
        emit(
            "test",
            NORMAL,
            line(name, pretty, elapsed),
            file=name,
            status="pass",
            passed=passed,
            skipped=skipped,
            failed=0,
            s=secs,
        )
        for reason, n in skip_reasons.items():
            emit(
                "skip",
                NORMAL,
                dim(f"    skip  {n} test{'s' if n != 1 else ''}: {reason}"),
                file=name,
                count=n,
                reason=reason,
            )
        status = "ok"

    for test, case in statuses.items():
        mark = {"pass": good("ok  "), "skip": dim("skip"), "fail": bad("FAIL")}[case]
        emit("case", VERBOSE, f"    {mark}  {test}", file=name, test=test, status=case)
    if status != "fail" and LEVEL >= DEBUG:
        emit_raw(raw)

    record(passed, skipped, failed, status)
    return rc


def summary(tally: str) -> int:
    if FORMAT == "plain":
        return 0
    passed = skipped = failed = files = files_failed = files_skipped = 0
    if os.path.exists(tally):
        with open(tally) as f:
            for row in f:
                p, s, x, status = row.split()
                passed, skipped, failed = (
                    passed + int(p),
                    skipped + int(s),
                    failed + int(x),
                )
                files += 1
                files_failed += status == "fail"
                files_skipped += status == "skip"
    parts = [good(f"{passed} passed")]
    if skipped:
        parts.append(dim(f"{skipped} skipped"))
    parts.append(bad(f"{failed} failed") if failed else dim("0 failed"))
    detail = f"{files} files"
    if files_skipped:
        detail += f", {files_skipped} skipped"
    if files_failed:
        detail += ", " + bad(f"{files_failed} failed")
    if FORMAT == "pretty" and LEVEL >= NORMAL:
        print()
    emit(
        "tests",
        QUIET,
        f"{accent('▸')} {bold('mojo tests')}  "
        + dim(" · ").join(parts)
        + "  "
        + dim(detail),
        result="fail" if files_failed else "pass",
        files=files,
        files_skipped=files_skipped,
        files_failed=files_failed,
        passed=passed,
        skipped=skipped,
        failed=failed,
        s=round(time.time() - RUN_START),
    )
    return 0


def main() -> int:
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--summary":
        return summary(args[1])
    build = bool(args) and args[0] == "--build"
    if build:
        args = args[1:]
    if len(args) < 3 or args[1] != "--":
        print(__doc__, file=sys.stderr)
        return 2
    return (build_file if build else run_file)(args[0], args[2:])


if __name__ == "__main__":
    sys.exit(main())
