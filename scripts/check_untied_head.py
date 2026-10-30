"""Untied head update gate (slice S2): the head trains, and stays put when idle.

    check_untied_head.py <tied_bin> <untied_bin> [workdir]

1. Runs the tied trainer for 1 step from the d6 descriptor to get a nonzero,
   non-degenerate tied model_1.bin (version 3, head == wte by construction).
2. Starts the untied trainer from that .bin (loader copies wte into the head;
   head != 0) with weight decay `-c 0.1` and runs 2 steps, twice:
     active: head must differ from wte after training, the head's AdamW m and v
             must be nonzero and wte's m/v nonzero too (the encoder row-sparse
             gradient is the only writer of dwte).
     idle  (LLMM_UNTIED_IDLE_HEAD=1, the tie phase): head must equal wte
             bitwise, and the head's m/v must be exactly zero even though the
             weight decay is > 0 (a head updated while idle would decay).
The active run additionally checks the 16-tensor body moved (the head update
did not replace the body update).
"""

import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = 2


def env_for(extra):
    lib = next(
        os.path.join(ROOT, ".pixi/envs/default/lib", f)
        for f in sorted(os.listdir(os.path.join(ROOT, ".pixi/envs/default/lib")))
        if f.startswith("libpython3") and f.endswith(".so")
    )
    e = dict(os.environ, LLMM_USE_CPU="1", MOJO_PYTHON_LIBRARY=lib, LLMM_LOG="json")
    e.update(extra)
    return e


def train(binary, out, start, steps, extra=None, wd="0.1"):
    os.makedirs(out, exist_ok=True)
    data = os.path.join(ROOT, "data/.fineweb10B/fineweb_train_000001.bin")
    args = [
        binary, "-e", start, "-i", data, "-j", data, "-o", out,
        "-n", str(steps), "-b", "2", "-t", "64", "-d", "128",
        "-l", "3e-4", "-c", wd, "-gc", "1.0", "-v", "0", "-s", "0", "-g", "0",
        "-x", str(steps),
    ]  # fmt: skip
    subprocess.run(
        args, cwd=ROOT, env=env_for(extra or {}), check=True, stdout=subprocess.DEVNULL
    )


def floats(buf, off, n):
    return struct.unpack(f"<{n}f", buf[off : off + 4 * n])


def main():
    tied, untied = sys.argv[1], sys.argv[2]
    work = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "build/untied_head")
    start_dir = os.path.join(work, "start")
    train(tied, start_dir, "d6", 1)
    start = os.path.join(start_dir, "model_1.bin")
    hdr = struct.unpack("<256i", open(start, "rb").read(1024))
    assert hdr[1] == 3, hdr[1]
    V_p, C = hdr[7], hdr[6]  # padded vocab, channels
    H = V_p * C
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            print("FAIL:", msg)
            ok = False

    for name, extra in (("active", {}), ("idle", {"LLMM_UNTIED_IDLE_HEAD": "1"})):
        out = os.path.join(work, name)
        train(untied, out, start, STEPS, extra)
        mb = open(os.path.join(out, f"model_{STEPS}.bin"), "rb").read()
        sb = open(os.path.join(out, f"state_{STEPS}_0.bin"), "rb").read()
        h = struct.unpack("<256i", mb[:1024])
        check(h[1] == 13, f"{name}: model version {h[1]} != 13")
        n17 = (len(mb) - 1024) // 4
        n16 = n17 - H
        wte = mb[1024 : 1024 + 4 * H]
        head = mb[1024 + 4 * n16 :]
        check(len(head) == 4 * H, f"{name}: head block size")
        # State: header 1024 B, then m[n17], v[n17] (fp32 ZeRO-0 layout).
        check(len(sb) == 1024 + 8 * n17, f"{name}: state size {len(sb)}")
        m = sb[1024 : 1024 + 4 * n17]
        v = sb[1024 + 4 * n17 :]
        head_m, head_v = m[4 * n16 :], v[4 * n16 :]
        wte_m, wte_v = m[: 4 * H], v[: 4 * H]
        start_wte = open(start, "rb").read()[1024 : 1024 + 4 * H]
        check(any(start_wte), f"{name}: start wte is all zero (degenerate start)")
        if name == "active":
            check(head != wte, "active: head == wte after training (head not updated)")
            check(head != start_wte, "active: head never moved from its wte copy")
            check(any(head_m) and any(head_v), "active: head m/v are zero")
            check(any(wte_m) and any(wte_v), "active: wte m/v are zero")
            check(wte != start_wte, "active: wte (encoder) never moved")
            # The head update is the only writer of the head: it moved in lockstep
            # with a nonzero, finite gradient.
            hv = floats(head, 0, 4096)
            check(all(x == x for x in hv), "active: NaN in head")
        else:
            check(head == wte, "idle: head != wte (head updated while idle)")
            check(not any(head_m) and not any(head_v), "idle: head m/v nonzero")
            check(any(wte_m) and any(wte_v), "idle: wte m/v are zero")
        print(f"{name}: n16={n16} H={H} head==wte={head == wte}")
    print("result gate=untied-head-update status=" + ("pass" if ok else "fail"))
    sys.exit(0 if ok else 1)


main()
