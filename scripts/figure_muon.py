"""Render the Muon-vs-AdamW result figure (GPT-2 124M, FineWeb, bf16).

Three panels, all computed from logs already committed in the tree (no GPU):
  (a) val loss vs step, 600 steps, 3 init seeds (one thin line per arm x seed)
  (b) val loss vs step, 2000 steps, seed 42
  (c) 300-step Muon sweep: val@300 vs learning rate, plus the later-stage
      winners at the chosen lr and the AdamW control

Sources: docs/ai/data/muon_vs_adamw_{adamw,muon}.jsonl,
docs/ai/data/muon_seeds_*.jsonl, sweeps/muon_*.jsonl. Writes
figures/muon_vs_adamw_124m_2026-10-30.png plus a .json sidecar of the plotted data.

Usage:
    pixi run python scripts/figure_muon.py
"""

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_log import steps_and_vals  # noqa: E402

DATA = ROOT / "docs" / "ai" / "data"
SWEEPS = ROOT / "sweeps"
OUT = ROOT / "figures" / "muon_vs_adamw_124m_2026-10-30.png"

ADAMW, MUON, TUNED = "#2a78d6", "#eb6834", "#1baf7a"
INK, GRAY, GRID = "#1a1a19", "#5f5e56", "#e5e4dd"
ARMS = [
    ("adamw", "AdamW", ADAMW),
    ("muon", "Muon, AdamW's recipe", MUON),
    ("muon_tuned", "Muon, tuned", TUNED),
]


def curve(path: Path) -> dict[int, float]:
    _, vals = steps_and_vals(path.read_text())
    return {v["step"]: v["loss"] for v in vals}


def setup(path: Path) -> dict:
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            d = json.loads(line)
            if d.get("event") == "setup":
                return d
    raise ValueError(path)


def seed_path(arm: str, seed: int, steps: int) -> Path:
    if steps == 600 and seed == 42 and arm != "muon_tuned":
        return DATA / f"muon_vs_adamw_{arm}.jsonl"
    return DATA / f"muon_seeds_s{seed}_{arm}_{steps}.jsonl"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.xaxis.label.set_color(GRAY)
    ax.yaxis.label.set_color(GRAY)


def lines(ax, runs, final_step, ymin):
    """runs: {arm: [curve,...]}; thin per-seed lines, end labels with arm mean."""
    ends = {}
    for arm, label, color in ARMS:
        cs = runs[arm]
        multi = len(cs) > 1
        for c in cs:
            xs = sorted(s for s in c if s > 0)
            ax.plot(
                xs,
                [c[s] for s in xs],
                color=color,
                lw=1.2 if multi else 2,
                alpha=0.7 if multi else 1,
                marker="o",
                ms=6 if not multi else 4,
                markeredgewidth=0,
                zorder=3,
                label=label if c is cs[0] else None,
            )
        ends[arm] = statistics.mean(c[final_step] for c in cs)
    ax.set_xlim(0, final_step * 1.22)
    ax.set_ylim(ymin, 7.0)
    # end labels, nudged apart if closer than ~0.18 in loss
    for arm, label, _ in ARMS:
        ax.annotate(
            f"{ends[arm]:.3f}",
            (final_step, ends[arm]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=INK,
            bbox=dict(fc="white", ec="none", pad=1.5),
        )
    ax.set_xlabel("step")
    ax.set_ylabel("val loss")
    return ends


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), dpi=200)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        style(ax)
    a, b, c = axes
    sidecar: dict = {}

    # (a) 600 steps x 3 seeds
    runs_a = {
        arm: [curve(seed_path(arm, s, 600)) for s in (42, 1, 2)] for arm, _, _ in ARMS
    }
    ends_a = lines(a, runs_a, 600, 4.0)
    a.set_title("600 steps, 3 init seeds", loc="left", fontsize=10.5, color=INK)
    a.text(
        0.98,
        0.97,
        "step 0 (val ~11.0) omitted;\ny-axis starts at 4.0",
        transform=a.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=GRAY,
    )
    a.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK)
    sidecar["a_end_mean_600"] = ends_a

    # (b) 2000 steps, seed 42
    runs_b = {arm: [curve(seed_path(arm, 42, 2000))] for arm, _, _ in ARMS}
    ends_b = lines(b, runs_b, 2000, 3.5)
    b.set_title("2000 steps, seed 42", loc="left", fontsize=10.5, color=INK)
    ref = ends_b["adamw"]
    b.axhline(ref, color=GRAY, lw=1, ls=":", zorder=2)
    tuned = runs_b["muon_tuned"][0]
    pts = sorted(s for s in tuned if s > 0)
    cross = next(s for s in pts if tuned[s] <= ref)
    b.annotate(
        f"tuned Muon passes AdamW's\nfinal {ref:.3f} by step {cross}",
        (cross, tuned[cross]),
        xytext=(1000, 5.35),
        fontsize=8,
        color=GRAY,
        arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8),
    )
    b.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK)
    sidecar["b_end_2000"] = ends_b
    sidecar["b_cross_step"] = cross

    # (c) sweep
    ladder = sorted(
        (setup(p)["lr"], curve(p)[300]) for p in SWEEPS.glob("muon_lr*.jsonl")
    )
    lrs, v = zip(*ladder)
    c.plot(
        lrs,
        v,
        color=MUON,
        lw=2,
        marker="o",
        ms=6,
        markeredgewidth=0,
        zorder=3,
        label="Muon lr ladder (mu 0.95)",
    )
    chosen = min(ladder, key=lambda t: abs(t[0] - 4.8e-3))
    c.plot(
        [chosen[0]],
        [chosen[1]],
        marker="o",
        ms=11,
        mfc="none",
        mec=INK,
        mew=1.2,
        zorder=4,
        ls="none",
    )
    c.annotate(
        f"chosen lr 4.8e-3\n({chosen[1]:.3f})",
        chosen,
        xytext=(10, 28),
        textcoords="offset points",
        ha="left",
        fontsize=8,
        color=INK,
    )
    winners = [
        ("mu0.9", "mu 0.9"),
        ("wd0", "+ wd 0"),
        ("gc0", "+ grad-clip off"),
    ]
    win_vals = {}
    for i, (stem, lab) in enumerate(winners):
        val = curve(SWEEPS / f"muon_{stem}.jsonl")[300]
        win_vals[stem] = val
        c.plot(
            [4.8e-3],
            [val],
            marker="o",
            ms=7,
            color=TUNED,
            markeredgewidth=0,
            zorder=5,
            ls="none",
            label="later-stage winners (at lr 4.8e-3)" if i == 0 else None,
        )
        c.annotate(
            f"{lab}: {val:.3f}",
            (4.8e-3, val),
            xytext=(14, (0, -12, -24)[i]),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    ctrl = curve(SWEEPS / "muon_adamw6e-4.jsonl")[300]
    c.axhline(ctrl, color=ADAMW, lw=1.2, ls=":", zorder=2)
    c.annotate(
        f"AdamW control (lr 6e-4): {ctrl:.3f}",
        (2.0e-2, ctrl),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK,
    )
    c.set_xscale("log")
    c.set_xlim(2.2e-4, 2.2e-2)
    c.set_ylim(4.8, 6.35)
    c.set_xlabel("peak learning rate")
    c.set_ylabel("val loss at step 300")
    c.set_title("Muon sweep, 300 steps", loc="left", fontsize=10.5, color=INK)
    c.legend(
        loc="upper right",
        fontsize=8,
        frameon=False,
        labelcolor=INK,
        bbox_to_anchor=(1.0, 0.86),
    )
    sidecar["c_ladder"] = [{"lr": lr, "val300": x} for lr, x in ladder]
    sidecar["c_winners"] = win_vals
    sidecar["c_adamw_control"] = ctrl

    fig.suptitle(
        "Muon vs AdamW, GPT-2 124M from scratch (validation loss, lower is better)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.015,
        "GPT-2 124M from scratch, FineWeb, bf16, B=32 T=1024 (131k tokens/step), "
        "cosine to zero; single GB10.",
        fontsize=8,
        color=GRAY,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white")
    OUT.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(json.dumps(sidecar, indent=1))


if __name__ == "__main__":
    main()
