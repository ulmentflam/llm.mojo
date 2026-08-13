#!/usr/bin/env python3
"""Precision-comparison figures: bf16 vs fp8 (vs nvfp4, when present) twins.

Renders the from-scratch FineWeb-10B precision-pair results into figures/:

  1. `precision_valloss_*.png` — validation-loss curves over training for each
     scale (one panel per model size), one line per training precision. All
     runs in a panel share the identical recipe (data, schedule, tokens/step,
     hardware, seed) and differ ONLY in GEMM precision, so the curves are an
     apples-to-apples numerics comparison, not a tuning comparison.
  2. `precision_hellaswag_*.png` — final HellaSwag acc_norm per run with
     Wilson 95% CIs (same scoring path as scripts/benchmark_eval.py; see that
     script's docstring for the acc vs acc_norm distinction and why we only
     draw CIs for results where we hold the exact (k, n)).

Data sources are the training-run logs on /data (val loss is printed every
250 steps by train_gpt2.mojo) plus the exact HellaSwag counts from `make
eval`. Runs whose log or eval is missing are skipped with a notice, so the
same command re-renders richer figures as new precision arms (e.g. nvfp4)
finish. The hardware slug in the filename comes from the box the script runs
on — these runs and figures are workstation-max artifacts.

Usage:
    pixi run python scripts/benchmark_precision.py
"""

import os
import glob
import json
import re
import sys

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.benchmark_eval import wilson_ci
from scripts.benchmark_train import (
    FIGURES,
    GRID_COLOR,
    TEXT_GRAY,
    TEXT_INK,
    _slug,
    hardware_info,
)

matplotlib.use("Agg")

RUNS_ROOT = "/data/llm.mojo/runs/v2"

# Fixed per-precision hues (never cycled, never reassigned): bf16 wears the
# llm.mojo family blue from benchmark_train.py (it is the baseline precision);
# fp8/nvfp4 get their own reserved steps. The trio passes the OKLab
# adjacent-pair CVD check (worst-case deltaE 15.3, normal-vision >= 23).
PRECISION_COLORS = {
    "bf16": "#2a78d6",
    "fp8": "#c2571f",
    "nvfp4": "#7b52ae",
}

# (scale label, precision, run dir, HellaSwag (k, n) from `make eval` — None
# until the eval has been run, which also skips the run in the eval figure.)
#
# Read from docs/ai/v2_arm_results/*.json, the same records that generate the
# README table, so a figure can never disagree with the table it sits under.
# The v1 arms these replaced are retracted; see
# docs/ai/v1_precision_table_retraction.md.
ARM_RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs",
    "ai",
    "v2_arm_results",
)


def _load_runs():
    runs = []
    for path in sorted(glob.glob(os.path.join(ARM_RESULTS, "*.json"))):
        d = json.load(open(path))
        runs.append(
            (
                d["scale"],
                d["precision"],
                os.path.basename(d["run_dir"]),
                (d["hellaswag_k"], d["hellaswag_n"]),
            )
        )
    order = {"bf16": 0, "fp8": 1, "nvfp4": 2}
    runs.sort(key=lambda r: (0 if r[0] == "124M" else 1, order.get(r[1], 9)))
    return tuple(runs)


RUNS = _load_runs()

VAL_RE = re.compile(r"^val loss ([0-9.]+)")
STEP_RE = re.compile(r"^step (\d+)/(\d+)")


def parse_val_losses(log_path):
    """(step, val_loss) series from a train.log. The trainer prints `val loss`
    lines between step lines (every 250 steps; step 0 = the pre-training
    eval), so each val loss is attributed to the most recent step seen. On
    checkpoint resume the log replays a few steps; later duplicates win,
    matching what the final model actually trained through.
    """
    points = {}
    last_step = 0
    total = None
    with open(log_path) as f:
        for line in f:
            m = STEP_RE.match(line)
            if m:
                last_step = int(m.group(1))
                total = int(m.group(2))
                continue
            m = VAL_RE.match(line)
            if m:
                points[last_step] = float(m.group(1))
    steps = sorted(points)
    return steps, [points[s] for s in steps], total


def collect():
    runs = []
    for scale, precision, run_dir, eval_kn in RUNS:
        log = os.path.join(RUNS_ROOT, run_dir, "train.log")
        if not os.path.exists(log):
            print(f"[skip] {scale} {precision}: no {log}")
            continue
        steps, losses, total = parse_val_losses(log)
        # An arm that has only just launched would put a legend entry with no
        # visible line inside the converged-tail ylim — wait for enough of the
        # curve to say something before drawing it.
        if len(steps) < 20:
            print(f"[skip] {scale} {precision}: only {len(steps)} val points so far")
            continue
        complete = total is not None and steps[-1] >= total
        runs.append(
            {
                "scale": scale,
                "precision": precision,
                "steps": steps,
                "losses": losses,
                "complete": complete,
                "eval_kn": eval_kn,
            }
        )
    return runs


def render_valloss(runs, hw, out_path):
    scales = sorted({r["scale"] for r in runs}, key=lambda s: float(s[:-1]))
    fig, axes = plt.subplots(
        1, len(scales), figsize=(5.2 * len(scales), 4.4), sharex=True
    )
    if len(scales) == 1:
        axes = [axes]
    for ax, scale in zip(axes, scales):
        panel = [r for r in runs if r["scale"] == scale]
        for r in panel:
            label = r["precision"] + ("" if r["complete"] else " (in progress)")
            ax.plot(
                r["steps"],
                r["losses"],
                color=PRECISION_COLORS[r["precision"]],
                linewidth=2,
                label=label,
                zorder=3,
            )
        # Selective direct labels: final value only, at the line ends. The
        # arms converge to near-identical losses (that is the finding), so the
        # labels are stacked in loss order to keep them from colliding.
        for rank, r in enumerate(sorted(panel, key=lambda r: -r["losses"][-1])):
            ax.annotate(
                f"{r['precision']} {r['losses'][-1]:.4f}",
                (r["steps"][-1], r["losses"][-1]),
                textcoords="offset points",
                xytext=(8, 14 - 13 * rank),
                fontsize=8.5,
                color=TEXT_INK,
            )
        ax.set_title(f"GPT-2 {scale}", fontsize=11, color=TEXT_INK, fontweight="bold")
        ax.set_xlabel("step", color=TEXT_GRAY, fontsize=9.5)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_GRAY, labelsize=9)
        # The interesting part is the converged tail, not the step-0 cliff
        # from ~11 nats at random init.
        ax.set_ylim(2.8, 5.0)
        # Room on the right for the stacked end-of-line labels.
        ax.set_xlim(left=-400, right=max(r["steps"][-1] for r in panel) * 1.24)
        if len(panel) >= 2:
            ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_INK)
    axes[0].set_ylabel("FineWeb val loss", color=TEXT_GRAY, fontsize=9.5)
    fig.suptitle(
        "Training precision does not change what the model learns",
        fontsize=13,
        color=TEXT_INK,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.925,
        "val loss every 250 steps · identical recipe per panel "
        "(10B FineWeb tokens, 458,752 tokens/step, 1 epoch)",
        ha="center",
        fontsize=9,
        color=TEXT_GRAY,
    )
    fig.text(
        0.5,
        0.885,
        f"7x {hw['gpu']}",
        ha="center",
        fontsize=8.5,
        color=TEXT_GRAY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def render_hellaswag(runs, hw, out_path):
    entries = [r for r in runs if r["eval_kn"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs, heights, errs, colors, labels = [], [], [], [], []
    x = 0.0
    prev_scale = None
    for r in entries:
        if prev_scale is not None and r["scale"] != prev_scale:
            x += 0.7  # spacer between scale groups
        prev_scale = r["scale"]
        k, n = r["eval_kn"]
        phat, lo, hi = wilson_ci(k, n)
        xs.append(x)
        heights.append(phat * 100)
        errs.append((phat * 100 - lo * 100, hi * 100 - phat * 100))
        colors.append(PRECISION_COLORS[r["precision"]])
        labels.append(f"{r['scale']}\n{r['precision']}")
        x += 1.0
    ax.bar(
        xs,
        heights,
        width=0.62,
        color=colors,
        zorder=3,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.errorbar(
        xs,
        heights,
        yerr=list(zip(*errs)),
        fmt="none",
        ecolor=TEXT_INK,
        elinewidth=1.1,
        capsize=3,
        zorder=4,
    )
    for xi, h, (_, up) in zip(xs, heights, errs):
        # Above the upper CI cap, not the bar top, so label and whisker
        # never collide.
        ax.annotate(
            f"{h:.2f}%",
            (xi, h + up),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
            color=TEXT_INK,
        )
    ax.axhline(25, color=TEXT_GRAY, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(
        "random baseline 25%",
        (xs[-1] + 0.45, 25),
        ha="right",
        va="bottom",
        fontsize=8,
        color=TEXT_GRAY,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9, color=TEXT_INK)
    ax.set_ylabel("HellaSwag accuracy (acc_norm, %)", color=TEXT_GRAY, fontsize=9.5)
    ax.set_ylim(24, max(heights) + 4)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_GRAY, labelsize=9)
    ax.set_title(
        "Wilson 95% CIs · n=10,042 · scored with `make eval` (acc_norm)",
        fontsize=9.5,
        color=TEXT_GRAY,
        pad=10,
    )
    fig.suptitle(
        "HellaSwag by training precision",
        fontsize=13,
        color=TEXT_INK,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    hw = hardware_info()
    runs = collect()
    if not runs:
        sys.exit("no runs found")
    stamp = hw["date"].split(" ")[0]
    gpu_slug = _slug(hw["gpu"])
    os.makedirs(FIGURES, exist_ok=True)
    render_valloss(
        runs, hw, os.path.join(FIGURES, f"precision_valloss_{stamp}_{gpu_slug}.png")
    )
    render_hellaswag(
        runs, hw, os.path.join(FIGURES, f"precision_hellaswag_{stamp}_{gpu_slug}.png")
    )
    # Sidecar carrying the source data verbatim, per the repo's figure
    # reproducibility contract (see the `figures` target in the Makefile).
    for stem in ("precision_valloss", "precision_hellaswag"):
        sidecar = os.path.join(FIGURES, f"{stem}_{stamp}_{gpu_slug}.json")
        with open(sidecar, "w") as fh:
            json.dump(
                {
                    "hardware": hw,
                    "source": "docs/ai/v2_arm_results/*.json",
                    "runs": [
                        {
                            "scale": r["scale"],
                            "precision": r["precision"],
                            "final_val_loss": r["losses"][-1]
                            if r.get("losses")
                            else None,
                            "steps": len(r["steps"]),
                            "complete": r["complete"],
                            "eval_kn": r["eval_kn"],
                        }
                        for r in runs
                    ],
                },
                fh,
                indent=2,
            )
        print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
