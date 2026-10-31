"""`-untie_at` gates (slice S4), all deterministic fp32 CPU runs of a d6 model.

    check_untie_at.py <untied_bin> <tied_bin> <s2_untied_bin> [workdir]

* tied-file   -untie_at 0.5, 6 steps, checkpoint every step: model_1..3 carry
              head == wte bitwise (version 13); model_4.. have head != wte; head
              m/v are exactly zero at step 3 and nonzero at step 6.
* bias-corr   the head's first post-untie update (step 3 -> 4) equals a
              hand-computed AdamW step with t = 1 (m = (1-b1) g, v = (1-b2) g^2,
              update lr * g / (|g| + eps) + decoupled decay), recomputed from the
              checkpoint files in float64; a counter of t = 4 would be off by far
              more than the tolerance.
* from-start  `-untie_at 0` equals the S2 binary (no flag) bit for bit: model
              file, state file minus header slots 44/45, step losses/norms.
* never       `-untie_at 1.0` model file prefix equals the tied build's.
* resume      checkpoints at step 2 (before s_u) and 4 (after): resume each and
              compare model/state files of every later step against the
              continuous run (bitwise).
* refusals    resume with `-untie_at 0.7`, with another `-x`, a tied state file in
              the untied build, an untied state file in the tied build, and the
              flag itself in the tied build: each exits nonzero with a message.
Run for both -opt adamw and -opt muon.
"""

import json
import math
import os
import shutil
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = 6
LR = 3e-4
WD = 0.1
GC = 1.0
B1, B2, EPS = 0.9, 0.999, 1e-8
ok = True


def check(cond, msg):
    global ok
    if not cond:
        print("FAIL:", msg)
        ok = False
    return cond


def env_for(log=None):
    lib = next(
        os.path.join(ROOT, ".pixi/envs/default/lib", f)
        for f in sorted(os.listdir(os.path.join(ROOT, ".pixi/envs/default/lib")))
        if f.startswith("libpython3") and f.endswith(".so")
    )
    e = dict(os.environ, LLMM_USE_CPU="1", MOJO_PYTHON_LIBRARY=lib, LLMM_LOG="json")
    e.pop("LLMM_UNTIE_AT", None)
    if log:
        e["LLMM_LOG_FILE"] = log
        e["LLMM_LOG_LEVEL"] = "verbose"
    return e


def run(
    binary,
    out,
    opt,
    extra=(),
    steps=STEPS,
    n=1,
    resume=False,
    start="d6",
    fail_ok=False,
):
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, "log.jsonl")
    if os.path.exists(log):
        os.remove(log)
    data = os.path.join(ROOT, "data/.fineweb10B/fineweb_train_000001.bin")
    args = [
        binary, "-e", start, "-i", data, "-j", data, "-o", out,
        "-n", str(n), "-b", "2", "-t", "64", "-d", "128",
        "-l", str(LR), "-c", str(WD), "-gc", str(GC), "-v", "0", "-s", "0",
        "-g", "0", "-x", str(steps), "-opt", opt,
    ] + list(extra)  # fmt: skip
    if resume:
        args += ["-y", "1"]
    p = subprocess.run(
        args,
        cwd=ROOT,
        env=env_for(log),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not fail_ok and p.returncode != 0:
        print(p.stdout[-3000:])
        raise SystemExit("run failed: " + " ".join(args))
    return p


def steps_of(out):
    r = []
    for line in open(os.path.join(out, "log.jsonl")):
        ev = json.loads(line)
        if ev.get("event") == "step":
            r.append((ev["step"], ev["loss"], ev["norm"], ev.get("lr")))
    return r


def events(out, name):
    return [
        json.loads(ln)
        for ln in open(os.path.join(out, "log.jsonl"))
        if json.loads(ln).get("event") == name
    ]


def rd(path):
    return open(path, "rb").read()


def hdr(b):
    return struct.unpack("<256i", b[:1024])


def f32(b):
    import array

    a = array.array("f")
    a.frombytes(b)
    return a


def split_model(mb):
    h = hdr(mb)
    V_p, C = h[7], h[6]
    H = V_p * C
    return h, H, mb[1024 : 1024 + 4 * H], mb[len(mb) - 4 * H :], mb


def state_parts(sb, n17, opt):
    """m, v byte strings. Muon v is compact (AdamW ranges only), so only the
    head's tail is compared by position for both."""
    m = sb[1024 : 1024 + 4 * n17]
    v = sb[1024 + 4 * n17 :]
    return m, v


def nz(b):
    return any(b)


def same_state_modulo_untie_slots(a, b):
    ha, hb = bytearray(a[:1024]), bytearray(b[:1024])
    for s in (44, 45):
        ha[4 * s : 4 * s + 4] = b"\0\0\0\0"
        hb[4 * s : 4 * s + 4] = b"\0\0\0\0"
    return ha == hb and a[1024:] == b[1024:]


def cmp_files(a, b, what, ignore_untie_slots=False):
    x, y = rd(a), rd(b)
    if ignore_untie_slots:
        return check(same_state_modulo_untie_slots(x, y), what + " differ")
    return check(x == y, what + " differ (%d vs %d bytes)" % (len(x), len(y)))


def main():
    untied, tied, s2 = sys.argv[1:4]  # s2 '-' = compare against the same binary
    work = sys.argv[4] if len(sys.argv) > 4 else os.path.join(ROOT, "build/untie_at")
    shutil.rmtree(work, ignore_errors=True)
    for opt in ("adamw", "muon"):
        print("=== -opt", opt)
        w = os.path.join(work, opt)

        # ---- continuous run, s_u = 3 -------------------------------------
        cont = os.path.join(w, "cont")
        run(untied, cont, opt, ["-untie_at", "0.5"])
        ev = events(cont, "untie")
        check(
            len(ev) == 1 and ev[0].get("step") == 3,
            "untie event once at step 3: %s" % ev,
        )
        mb, H = b"", 0
        for k in range(1, 7):
            mb = rd(os.path.join(cont, "model_%d.bin" % k))
            h, H, wte, head, _ = split_model(mb)
            check(h[1] == 13, "model_%d version %d" % (k, h[1]))
            if k <= 3:
                check(head == wte, "model_%d (tied phase) head != wte" % k)
            else:
                check(head != wte, "model_%d head still == wte after untie" % k)
        n17 = (len(mb) - 1024) // 4
        H4 = 4 * H
        for k, want in ((3, False), (6, True)):
            sb = rd(os.path.join(cont, "state_%d_0.bin" % k))
            sh = hdr(sb)
            check(
                sh[1] == 2 and sh[44] == 2 and sh[45] == 3,
                "state_%d version/slots 44/45 = %d/%d/%d" % (k, sh[1], sh[44], sh[45]),
            )
            m = sb[1024 : 1024 + 4 * n17]
            v = sb[1024 + 4 * n17 :]
            hm, hv = m[-H4:], v[-H4:]
            check(
                nz(hm) == want and nz(hv) == want,
                "step %d head m/v nonzero == %s" % (k, want),
            )
        # ---- bias correction: step 3 -> 4 hand-computed (AdamW path) -----
        if opt == "adamw":
            lr = [s for s in steps_of(cont) if s[0] == 4][0][3]
            h3 = f32(split_model(rd(os.path.join(cont, "model_3.bin")))[3])
            h4 = f32(split_model(rd(os.path.join(cont, "model_4.bin")))[3])
            sb = rd(os.path.join(cont, "state_4_0.bin"))
            m = f32(sb[1024 + 4 * n17 - H4 : 1024 + 4 * n17])
            v = f32(sb[len(sb) - H4 :])
            worst = 0.0
            worst_t4 = 0.0
            nbad = 0
            cnt = 0
            for i in range(0, H, 7):
                g = m[i] / (1 - B1)
                if abs(v[i] - (1 - B2) * g * g) > 1e-4 * (1 - B2) * g * g + 1e-30:
                    nbad += 1
                # t = 1: mhat = m/(1-b1), vhat = v/(1-b2)
                for t, store in ((1, "ok"), (4, "t4")):
                    mh = m[i] / (1 - B1**t) if t == 1 else m[i] / (1 - B1**t)
                    vh = v[i] / (1 - B2**t)
                    exp = h3[i] - lr * (mh / (math.sqrt(vh) + EPS) + WD * h3[i])
                    err = abs(exp - h4[i])
                    if store == "ok":
                        worst = max(worst, err)
                    else:
                        worst_t4 = max(worst_t4, err)
                cnt += 1
            check(
                nbad == 0,
                "head v != (1-b2) g^2 on %d/%d sampled elements" % (nbad, cnt),
            )
            check(
                worst < 1e-7,
                "head update differs from hand-computed t=1 step: %g" % worst,
            )
            check(
                worst_t4 > 20 * max(worst, 1e-9),
                "t=4 not distinguishable (%g vs %g)" % (worst_t4, worst),
            )
            print(
                "bias-corr: max |update - hand t=1| = %.3g  (t=4 would be %.3g)"
                % (worst, worst_t4)
            )

        # ---- resume across the transition --------------------------------
        for r in (2, 4):
            rd_ = os.path.join(w, "resume%d" % r)
            os.makedirs(rd_, exist_ok=True)
            shutil.copy(os.path.join(cont, "model_%d.bin" % r), rd_)
            shutil.copy(os.path.join(cont, "state_%d_0.bin" % r), rd_)
            p = run(untied, rd_, opt, ["-untie_at", "0.5"], resume=True)
            gaps = []
            for k in range(r + 1, 7):
                for stem in ("model_%d.bin", "state_%d_0.bin"):
                    a = os.path.join(cont, stem % k)
                    b = os.path.join(rd_, stem % k)
                    cmp_files(a, b, "resume@%d %s" % (r, stem % k))
                    x, y = rd(a), rd(b)
                    gaps.append(0 if x == y else len(x))
            print(
                "resume@%d: steps %d..6 model/state bytes differing: %s"
                % (r, r + 1, gaps)
            )
            # Resume at r=2 must replay the untie; r=4 must not.
            check(
                len(events(rd_, "untie")) == (1 if r < 3 else 0),
                "resume@%d untie event count" % r,
            )
            ra = events(rd_, "resume")
            check(len(ra) == 1, "resume event missing")
            if ra:
                print(
                    "resume event:",
                    {
                        k: v
                        for k, v in ra[0].items()
                        if k in ("step", "head_tied", "untie_step")
                    },
                )

        # ---- refusals ----------------------------------------------------
        def refuse(binary, extra, src_dir, ck, tag, steps=STEPS, expect=None):
            d = os.path.join(w, "ref_" + tag)
            os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(src_dir, "model_%d.bin" % ck), d)
            shutil.copy(os.path.join(src_dir, "state_%d_0.bin" % ck), d)
            p = run(binary, d, opt, extra, steps=steps, resume=True, fail_ok=True)
            good = check(p.returncode != 0, "refusal %s: exit code 0" % tag)
            if good and expect:
                check(
                    expect in p.stdout,
                    "refusal %s: message lacks %r:\n%s"
                    % (tag, expect, p.stdout[-800:]),
                )
            line = [
                x
                for x in p.stdout.splitlines()
                if "annot resume" in x or "refus" in x.lower() or "Invalid" in x
            ]
            print(
                "refusal %-14s rc=%d  %s"
                % (tag, p.returncode, (line or p.stdout.splitlines()[-1:])[:1])
            )
            return p

        refuse(
            untied, ["-untie_at", "0.7"], cont, 2, "untie_at0.7", expect="untie step"
        )
        refuse(
            untied,
            ["-untie_at", "0.5"],
            cont,
            2,
            "other_x",
            steps=8,
            expect="untie step",
        )
        refuse(untied, [], cont, 4, "no_flag", expect="untie step")

        tcont = os.path.join(w, "tied_cont")
        run(tied, tcont, opt, [], steps=3)
        refuse(
            untied,
            ["-untie_at", "0.5"],
            tcont,
            2,
            "tied_into_untied",
            steps=3,
            expect="tied",
        )
        refuse(tied, [], cont, 2, "untied_into_tied", expect="untied")
        p = run(
            tied,
            os.path.join(w, "tied_flag"),
            opt,
            ["-untie_at", "0.5"],
            steps=2,
            fail_ok=True,
        )
        check(p.returncode != 0, "tied build accepted -untie_at")

        # ---- -untie_at 0 == S2 untied-from-start -------------------------
        a = os.path.join(w, "from0")
        b = os.path.join(w, "s2")
        run(untied, a, opt, ["-untie_at", "0"], steps=3)
        if s2 == "-":
            print(
                "(no S2 binary given: -untie_at 0 is compared with the same binary's default)"
            )
        run(s2 if s2 != "-" else untied, b, opt, [], steps=3)
        cmp_files(
            os.path.join(a, "model_3.bin"),
            os.path.join(b, "model_3.bin"),
            "-untie_at 0 vs S2 model",
        )
        cmp_files(
            os.path.join(a, "state_3_0.bin"),
            os.path.join(b, "state_3_0.bin"),
            "-untie_at 0 vs S2 state",
            True,
        )
        check(steps_of(a) == steps_of(b), "-untie_at 0 vs S2 step losses/norms")
        print("untie_at0 == S2:", steps_of(a))
        c = os.path.join(w, "default")
        run(untied, c, opt, [], steps=3)
        cmp_files(
            os.path.join(c, "model_3.bin"),
            os.path.join(b, "model_3.bin"),
            "default (no flag) vs S2 model",
        )
        cmp_files(
            os.path.join(c, "state_3_0.bin"),
            os.path.join(b, "state_3_0.bin"),
            "default vs S2 state",
            True,
        )

        # ---- -untie_at 1.0 == tied build ---------------------------------
        a = os.path.join(w, "never")
        run(untied, a, opt, ["-untie_at", "1.0"], steps=3)
        t = os.path.join(w, "tied3")
        run(tied, t, opt, [], steps=3)
        ub, tb = rd(os.path.join(a, "model_3.bin")), rd(os.path.join(t, "model_3.bin"))
        check(
            ub[1024 : len(tb)] == tb[1024:],
            "-untie_at 1.0 16-tensor prefix != tied build",
        )
        check(
            ub[len(tb) :] == tb[1024 : 1024 + len(ub) - len(tb)],
            "-untie_at 1.0 head block != wte",
        )
        check(steps_of(a) == steps_of(t), "-untie_at 1.0 losses/norms != tied build")

    print("result gate=untie-at status=" + ("pass" if ok else "fail"))
    sys.exit(0 if ok else 1)


main()
