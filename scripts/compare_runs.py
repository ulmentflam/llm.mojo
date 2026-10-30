"""Compare two trainer logs (e.g. AdamW vs Muon) as markdown tables.

    pixi run python scripts/compare_runs.py A.jsonl B.jsonl [--names adamw muon]

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


def main():
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
