"""Compare two trainer logs (e.g. AdamW vs Muon) as markdown tables.

    pixi run python scripts/compare_runs.py A.jsonl B.jsonl [--names adamw muon]
    pixi run python scripts/compare_runs.py --table RUN1.jsonl RUN2.jsonl ...   # N-run sweep summary

Parsing is delegated to scripts/train_log.py (any LLMM_LOG format). Prints: val
loss at each eval step, train loss averaged over fixed step windows, mean
steady-state ms/step and tok/s (step 1 and 2 excluded as warmup), and the first
step at which B's val loss is <= A's final val loss.
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_log import steps_and_vals  # noqa: E402


def load(path):
    steps, vals = steps_and_vals(Path(path).read_text())
    return steps, vals


def table(paths, tail=50, skip=2):
    """One row per run: val losses, mean train loss over the last `tail` steps, ms/step, status."""
    rows = []
    for p in paths:
        steps, vals = load(p)
        name = Path(p).stem.removeprefix("muon_")
        vd = {v["step"]: v["loss"] for v in vals}
        evals = sorted(vd)
        last = max((r["step"] for r in steps), default=0)
        losses = [r["loss"] for r in steps]
        bad = any(x != x for x in losses) or any(
            r["loss"] > 8 for r in steps if r["step"] > 50
        )
        total = steps[0]["steps"] if steps and "steps" in steps[0] else None
        status = (
            "diverged"
            if bad
            else ("ok" if total and last >= total else f"partial@{last}")
        )
        tl = [r["loss"] for r in steps if r["step"] > last - tail]
        ms = [r["ms"] for r in steps if r["step"] > skip]
        rows.append(
            (
                name,
                [vd.get(s) for s in evals],
                evals,
                statistics.mean(tl) if tl else None,
                statistics.mean(ms) if ms else None,
                status,
            )
        )
    steps_all = sorted({s for r in rows for s in r[2]})

    def f(x, d=4):
        return "" if x is None else f"{x:.{d}f}"

    print(
        "| run | "
        + " | ".join(f"val@{s}" for s in steps_all)
        + " | train last50 | ms/step | status |"
    )
    print("|---|" + "---:|" * (len(steps_all) + 2) + "---|")
    for name, v, ev, tl, ms, st in rows:
        d = dict(zip(ev, v))
        print(
            f"| {name} | "
            + " | ".join(f(d.get(s)) for s in steps_all)
            + f" | {f(tl)} | {f(ms, 1)} | {st} |"
        )


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--table":
        return table(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument(
        "--skip", type=int, default=2, help="warmup steps excluded from timing"
    )
    args = ap.parse_args()
    na, nb = args.names
    (sa, va), (sb, vb) = load(args.a), load(args.b)
    vad = {v["step"]: v["loss"] for v in va}
    vbd = {v["step"]: v["loss"] for v in vb}

    print(f"| step | val {na} | val {nb} | delta ({nb}-{na}) |")
    print("|---:|---:|---:|---:|")
    for s in sorted(set(vad) | set(vbd)):
        x, y = vad.get(s), vbd.get(s)
        d = f"{y - x:+.4f}" if x is not None and y is not None else ""
        print(
            f"| {s} | {x if x is None else f'{x:.4f}'} | {y if y is None else f'{y:.4f}'} | {d} |"
        )

    def win(steps):
        out = {}
        for r in steps:
            w = (r["step"] - 1) // args.window
            out.setdefault(w, []).append(r["loss"])
        return {w: statistics.mean(v) for w, v in out.items()}

    wa, wb = win(sa), win(sb)
    print(f"\n| steps | train {na} | train {nb} | delta ({nb}-{na}) |")
    print("|---|---:|---:|---:|")
    for w in sorted(set(wa) | set(wb)):
        x, y = wa.get(w), wb.get(w)
        d = f"{y - x:+.4f}" if x is not None and y is not None else ""
        lo, hi = w * args.window + 1, (w + 1) * args.window
        print(
            f"| {lo}-{hi} | {x if x is None else f'{x:.4f}'} | {y if y is None else f'{y:.4f}'} | {d} |"
        )

    print("\n| run | mean ms/step | tok/s | steps timed |")
    print("|---|---:|---:|---:|")
    for n, st in ((na, sa), (nb, sb)):
        t = [r for r in st if r["step"] > args.skip]
        if t:
            ms = statistics.mean(r["ms"] for r in t)
            print(
                f"| {n} | {ms:.1f} | {statistics.mean(r['tok_s'] for r in t):.0f} | {len(t)} |"
            )

    if va:
        target = va[-1]["loss"]
        hit = next((v for v in vb if v["loss"] <= target), None)
        msg = (
            f"step {hit['step']} (val {hit['loss']:.4f})"
            if hit
            else "never within the logged evals"
        )
        print(
            f"\nFirst {nb} val <= {na} final val ({target:.4f}, step {va[-1]['step']}): {msg}"
        )


if __name__ == "__main__":
    main()
