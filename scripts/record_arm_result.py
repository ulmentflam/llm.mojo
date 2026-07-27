#!/usr/bin/env python3
"""Record one finished precision arm's numbers as a JSON fragment.

One file per arm under docs/ai/v2_arm_results/. The README table is then
regenerated from whichever fragments exist (see update_readme_results.py), so
an arm appears the moment it finishes rather than the table being rewritten by
hand six times — and a half-finished pipeline produces a table that is honest
about which arms are in it.

Everything here is parsed from artifacts the run already wrote: the training
log for loss and throughput, the `make eval` output for HellaSwag. Nothing is
passed in by a human at publish time, so the table cannot drift from the run.
"""

import argparse
import json
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "ai" / "v2_arm_results"

STEP_RE = re.compile(
    r"^step (\d+)/(\d+) \| loss ([0-9.]+).*?\| ([0-9.]+) ms .*?\| (\d+) tok/s",
    re.M,
)
VAL_RE = re.compile(r"^val loss ([0-9.]+)", re.M)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — the CI the README already quotes for HellaSwag.

    Normal-approximation intervals misbehave badly near 0 or 1 and at small n;
    Wilson does not, which is why the existing benchmark_eval.py uses it.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (center - half, center + half)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, help="e.g. 124M-bf16")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--eval", required=True, help='"k/n = acc" from make eval')
    ap.add_argument("--repo", required=True, help="HuggingFace repo id")
    args = ap.parse_args()

    log = (args.run_dir / "train.log").read_text(errors="replace")

    steps = STEP_RE.findall(log)
    if not steps:
        raise SystemExit(f"no step lines parsed out of {args.run_dir}/train.log")
    last = steps[-1]
    final_step, total_steps = int(last[0]), int(last[1])
    final_loss = float(last[2])

    # Steady-state throughput: median over the last 500 logged steps. The mean
    # is skewed by checkpoint-write stalls and by however much of the run
    # shared the box; the median over a long tail is not.
    tail = steps[-500:]
    ms = sorted(float(s[3]) for s in tail)
    toks = sorted(int(s[4]) for s in tail)
    mid = len(ms) // 2

    vals = VAL_RE.findall(log)
    if not vals:
        raise SystemExit(f"no val loss lines in {args.run_dir}/train.log")

    m = re.match(r"(\d+)/(\d+) = ([0-9.]+)", args.eval.strip())
    if not m:
        raise SystemExit(f"could not parse --eval {args.eval!r}")
    k, n = int(m.group(1)), int(m.group(2))
    lo, hi = wilson_ci(k, n)

    scale, precision = args.arm.split("-", 1)
    record = {
        "arm": args.arm,
        "scale": scale,
        "precision": precision,
        "run_dir": str(args.run_dir),
        "hf_repo": args.repo,
        "final_step": final_step,
        "total_steps": total_steps,
        "final_train_loss": final_loss,
        "final_val_loss": float(vals[-1]),
        "hellaswag_k": k,
        "hellaswag_n": n,
        "hellaswag_acc": k / n,
        "hellaswag_ci95": [lo, hi],
        "median_ms_per_step": ms[mid],
        "median_tok_per_s": toks[mid],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.arm}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[record_arm_result] wrote {out}")
    print(
        f"[record_arm_result] {args.arm}: val {record['final_val_loss']:.4f}  "
        f"HellaSwag {k}/{n} = {100 * k / n:.2f}%  "
        f"[{100 * lo:.1f}%, {100 * hi:.1f}%]  "
        f"{record['median_tok_per_s']} tok/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
