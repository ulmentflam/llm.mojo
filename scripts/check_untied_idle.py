"""Untied build with an idle head must equal the tied build (S0 gate).

    check_untied_idle.py <tied_bin> <untied_bin> [workdir]

Runs the same deterministic 3-step CPU training (d6 from scratch, B=2 T=64,
`-gc 1e-3` so gradient clipping fires on every step) with both trainers and
requires:
  * every step's loss and norm identical (exact string equality of the log's
    verbose json fields),
  * norm > gc on every step, read back from LLMM_LOG_FILE (otherwise clipping
    never fired and a wrong grad-norm range would be invisible),
  * the untied model file is the tied one plus a trailing head block: the
    untied header says version 13, the 16-tensor prefix is byte-identical to
    the tied blob (`cmp`), and the head block equals the trained wte (head :=
    wte is written while the head is tied; no update reached the idle head),
  * the optimizer state files agree on the 16-tensor prefix (m, v).

Mutant check (manual, see docs/ai/untied_lm_head_design_2026-10-30.md S0): make
`calculate_grad_norm` reduce over all N17 on the CPU path and rebuild the
untied binary; this script must then FAIL on the norm/loss comparison.
"""

import json
import os
import struct
import subprocess
import sys

GC = 1e-3
STEPS = 3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(binary, out, idle=False):
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, "log.jsonl")
    if os.path.exists(log):
        os.remove(log)
    lib = next(
        os.path.join(ROOT, ".pixi/envs/default/lib", f)
        for f in sorted(os.listdir(os.path.join(ROOT, ".pixi/envs/default/lib")))
        if f.startswith("libpython3") and f.endswith(".so")
    )
    data = os.path.join(ROOT, "data/.fineweb10B/fineweb_train_000001.bin")
    env = dict(
        os.environ,
        LLMM_USE_CPU="1",
        MOJO_PYTHON_LIBRARY=lib,
        LLMM_LOG="json",
        LLMM_LOG_LEVEL="verbose",
        LLMM_LOG_FILE=log,
    )
    extra = []
    if idle:
        # The untied build trains its head by default (slice S2); -untie_at 1.0
        # keeps it tied to wte and idle (never untie), the S0 contract.
        extra = ["-untie_at", "1.0"]

    args = [
        binary,
        "-e",
        "d6",
        "-i",
        data,
        "-j",
        data,
        "-o",
        out,
        "-n",
        str(STEPS),
        "-b",
        "2",
        "-t",
        "64",
        "-d",
        "128",
        "-l",
        "3e-4",
        "-gc",
        str(GC),
        "-v",
        "0",
        "-s",
        "0",
        "-g",
        "0",
        "-x",
        str(STEPS),
    ] + extra
    subprocess.run(args, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    steps = []
    for line in open(log):
        ev = json.loads(line)
        if ev.get("event") == "step":
            steps.append((ev["step"], ev["loss"], ev["norm"]))
    return steps


def header(path):
    with open(path, "rb") as f:
        return struct.unpack("<256i", f.read(1024))


def main():
    tied, untied = sys.argv[1], sys.argv[2]
    work = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "build/untied_idle")
    st = run(tied, os.path.join(work, "tied"))
    su = run(untied, os.path.join(work, "untied"), idle=True)
    ok = True
    print("tied   steps:", st)
    print("untied steps:", su)
    if len(st) != STEPS or len(su) != STEPS:
        print("FAIL: expected", STEPS, "step events from each run")
        ok = False
    if any(n <= GC for _, _, n in st + su):
        print("FAIL: norm <= gc on some step; clipping did not fire")
        ok = False
    if st != su:
        print("FAIL: loss/norm differ between tied and untied-idle")
        ok = False

    mt = os.path.join(work, "tied", f"model_{STEPS}.bin")
    mu = os.path.join(work, "untied", f"model_{STEPS}.bin")
    ht, hu = header(mt), header(mu)
    if ht[1] != 3 or hu[1] != 13:
        print("FAIL: versions", ht[1], hu[1], "(want 3 and 13)")
        ok = False
    tb, ub = open(mt, "rb").read(), open(mu, "rb").read()
    head_bytes = hu[7] * hu[6] * 4
    if len(ub) != len(tb) + head_bytes:
        print(
            "FAIL: untied file size", len(ub), "!= tied", len(tb), "+ head", head_bytes
        )
        ok = False
    body = ub[1024 : len(tb)]
    if ub[: len(tb)][1024:] != tb[1024:]:
        print("FAIL: 16-tensor prefix of the model file differs")
        ok = False
    # Tied-phase files carry head := wte (the live tied weights), not the stale
    # zero init, so infer/eval/export of the file run the model that trained.
    if ub[len(tb) :] != tb[1024 : 1024 + head_bytes]:
        print("FAIL: idle head block is not a copy of the trained wte")
        ok = False
    del body

    # State: header (1024 B) + m + v; compare the 16-tensor prefix of each.
    tsb = open(os.path.join(work, "tied", f"state_{STEPS}_0.bin"), "rb").read()
    usb = open(os.path.join(work, "untied", f"state_{STEPS}_0.bin"), "rb").read()
    n16 = (len(tb) - 1024) // 4
    n17 = (len(ub) - 1024) // 4
    if len(tsb) != 1024 + 8 * n16 or len(usb) != 1024 + 8 * n17:
        print("FAIL: state file sizes", len(tsb), len(usb))
        ok = False
    else:
        tm, tv = tsb[1024 : 1024 + 4 * n16], tsb[1024 + 4 * n16 :]
        um, uv = (
            usb[1024 : 1024 + 4 * n16],
            usb[1024 + 4 * n17 : 1024 + 4 * n17 + 4 * n16],
        )
        if tm != um or tv != uv:
            print("FAIL: optimizer m/v 16-tensor prefix differs")
            ok = False
        if any(usb[1024 + 4 * n16 : 1024 + 4 * n17]) or any(
            usb[1024 + 4 * n17 + 4 * n16 :]
        ):
            print("FAIL: idle head m/v not zero")
            ok = False
    print("result gate=untied-idle-equals-tied status=" + ("pass" if ok else "fail"))
    sys.exit(0 if ok else 1)


main()
