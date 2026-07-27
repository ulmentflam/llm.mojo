"""Render the ZeRO memory-measurement figure: exact accounting vs nvidia-smi.

The point of this figure is a methodology lesson, not a benchmark result.

BACKGROUND FOR A READER WHO HAS NOT MET ZeRO. Training a model on several
GPUs at once normally keeps a full copy of everything on every GPU. ZeRO
("Zero Redundancy Optimizer") removes that duplication in stages: stage 1
splits the optimizer state across GPUs, stage 2 also splits the gradients,
stage 3 also splits the parameters themselves. Each stage should therefore
use strictly less memory per GPU than the one before it.

The trap this figure documents: the obvious way to check that -- reading
`nvidia-smi` -- CANNOT SEE most of it. The CUDA caching allocator commits
device memory in 256 MiB chunks, so `nvidia-smi` only ever moves in 256 MiB
steps. Real savings of tens of MiB either vanish entirely or get rounded up
into a 256 MiB jump that overstates them. Both failure directions are on
this figure.

Plotting savings RELATIVE TO STAGE 0 rather than absolute totals is
deliberate: the fixed ~696 MiB CUDA-context overhead that `nvidia-smi`
includes and the in-process accounting does not cancels exactly in the
difference, so the two instruments become directly comparable and every
`nvidia-smi` bar lands exactly on a 256 MiB gridline.

Data: Team M's pre-change world-2 baseline (`LLMM_MEM_REPORT`), measured on
workstation-max. This script only renders; it runs nothing.

Usage:
    pixi run python scripts/figure_zero_blindness.py
"""

import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.benchmark_train import (  # noqa: E402
    GRID_COLOR,
    TEXT_GRAY,
    TEXT_INK,
)

FIGURES = os.path.join(ROOT, "figures")

# Categorical slots 1 and 2 of the project palette. Colour encodes WHICH
# INSTRUMENT produced the number, and nothing else. Validated on a white
# surface: adjacent CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 contrast.
COLOR_EXACT = "#2a78d6"  # blue  -- the trainer's own byte-exact accounting
COLOR_SMI = "#eb6834"  # orange -- nvidia-smi

# The CUDA caching allocator's commit granularity, in MiB. Team M observed
# every one of the 16 driver readings in this baseline to be an exact
# integer multiple of this -- not one fell off the grid.
COMMIT_MIB = 256

STAGE_LABELS = {
    0: "Stage 0\n(DDP)",
    1: "Stage 1\n(+opt shard)",
    2: "Stage 2\n(+grad shard)",
    3: "Stage 3\n(+param shard)",
}

SRC_SMALL = os.path.join(ROOT, "zero", "bench", "bench_zero_world2.json")
SRC_LARGE = os.path.join(ROOT, "zero", "bench", "bench_zero_world2_b8t1024.json")


def _slug(s):
    import re

    return re.sub(r"[^A-Za-z0-9.]+", "-", s).strip("-") or "x"


def _gpu_token(name):
    """First five words identify the product; the full name is too long."""
    return "-".join(name.split()[:5]) if name else "unknown-gpu"


def load(path):
    with open(path) as fh:
        return json.load(fh)


def rows_by_key(doc):
    """Index the `ok` results by (precision, stage)."""
    out = {}
    for r in doc["results"]:
        if r.get("status") == "ok":
            out[(r["precision"], r["stage"])] = r
    return out


def series_for(doc, precision):
    """Savings vs stage 0, per instrument, for one precision.

    Returns (stages, exact_saved, smi_saved, base_exact, base_smi). Savings
    are positive numbers: how much LESS memory this stage uses than stage 0.
    """
    rows = rows_by_key(doc)
    stages = sorted(s for (p, s) in rows if p == precision)
    base = rows[(precision, 0)]
    base_exact = base["mem_report"]["exact_total_mib_max"]
    base_smi = base["peak_mem_mib_max_delta"]
    exact_saved = []
    smi_saved = []
    for s in stages:
        r = rows[(precision, s)]
        exact_saved.append(base_exact - r["mem_report"]["exact_total_mib_max"])
        smi_saved.append(base_smi - r["peak_mem_mib_max_delta"])
    return stages, exact_saved, smi_saved, base_exact, base_smi


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_GRAY)
    ax.tick_params(colors=TEXT_GRAY, labelsize=9)


def panel_savings(ax, doc, precision, annotate, deltas=()):
    """One precision: exact vs nvidia-smi savings, on the 256 MiB grid."""
    stages, exact, smi, base_exact, base_smi = series_for(doc, precision)
    xs = list(range(len(stages)))
    width = 0.38

    top = max(max(exact), max(smi))
    ymax = (int(top // COMMIT_MIB) + 2) * COMMIT_MIB

    # The y ticks ARE the allocator's commit grid, so every nvidia-smi bar
    # tip lands exactly on a tick. That coincidence is the whole argument.
    ticks = list(range(0, ymax + 1, COMMIT_MIB))
    ax.set_yticks(ticks)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ymax)

    for offset, vals, color in (
        (-width / 2, exact, COLOR_EXACT),
        (+width / 2, smi, COLOR_SMI),
    ):
        for x, v in zip(xs, vals):
            ax.bar(
                x + offset,
                v,
                width=width,
                color=color,
                edgecolor=TEXT_INK,
                linewidth=1.1,
                zorder=3,
            )
            ax.annotate(
                f"{v:,.1f}" if v % 1 else f"{v:,.0f}",
                (x + offset, v),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                va="bottom",
                fontsize=7.8,
                color=TEXT_INK,
                zorder=5,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels([STAGE_LABELS.get(s, f"Stage {s}") for s in stages])
    for lbl in ax.get_xticklabels():
        lbl.set_color(TEXT_INK)
    _style_axes(ax)

    ax.set_title(
        f"{precision}   ·   Stage 0 baseline: "
        f"{base_exact:,.0f} MiB exact / {base_smi:,.0f} MiB nvidia-smi",
        color=TEXT_GRAY,
        fontsize=9,
        pad=8,
    )

    # Vertical span arrows call out a step the instrument reports between
    # two adjacent stages, against the real change over the same step.
    for d in deltas:
        ax.annotate(
            "",
            xy=(d["x"], d["y1"]),
            xytext=(d["x"], d["y0"]),
            textcoords="data",
            zorder=6,
            arrowprops=dict(
                arrowstyle="<->",
                color=TEXT_INK,
                linewidth=1.1,
                shrinkA=0,
                shrinkB=0,
            ),
        )
        ax.text(
            d["x"],
            d["y1"] + 34,
            d["text"],
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=TEXT_INK,
            zorder=6,
        )

    for note in annotate:
        ax.annotate(
            note["text"],
            xy=note["xy"],
            xytext=note["xytext"],
            textcoords="data",
            fontsize=8.2,
            color=TEXT_INK,
            ha=note.get("ha", "center"),
            va="center",
            zorder=6,
            arrowprops=dict(
                arrowstyle="-",
                color=TEXT_GRAY,
                linewidth=0.9,
                shrinkA=2,
                shrinkB=2,
            ),
            bbox=dict(
                boxstyle="round,pad=0.32",
                facecolor="white",
                edgecolor=TEXT_GRAY,
                linewidth=0.7,
            ),
        )
    return dict(
        stages=stages,
        exact_saved_mib=exact,
        smi_saved_mib=smi,
        stage0_exact_mib=base_exact,
        stage0_smi_mib=base_smi,
    )


def panel_scale(ax, doc_large):
    """Context: how big is a 150 MiB saving at a production-ish shape?"""
    rows = rows_by_key(doc_large)
    r = rows[("fp32", 3)]
    mr = r["mem_report"]
    total = mr["exact_total_mib_max"]
    acts = mr["classes_mib_max"]["activations"]
    logits = mr["tensors_mib_max"]["logits"]
    target = mr["buffers_mib_max"]["embed_window_buf"]

    items = [
        ("Total footprint, one GPU", total),
        ("      of which: all activations", acts),
        ("            of which: the logits tensor †", logits),
        ("What this campaign removes", target),
    ]
    labels = [i[0] for i in items]
    vals = [i[1] for i in items]
    ys = list(range(len(items)))[::-1]

    # On a log axis a bar must run from the axis floor to its value, so the
    # width is (value - floor), not the value.
    left = 80.0
    ax.barh(
        ys,
        [v - left for v in vals],
        height=0.34,
        left=left,
        color=COLOR_EXACT,
        edgecolor=TEXT_INK,
        linewidth=1.1,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xlim(left, total * 2.6)
    ax.set_ylim(-0.72, len(items) - 0.30)
    ax.set_yticks([])
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("MiB, log scale", color=TEXT_GRAY, fontsize=9)

    # Labels sit ABOVE each bar, anchored at the axis start, so long names
    # never collide with the neighbouring panel or with the bars.
    for y, v, lbl in zip(ys, vals, labels):
        ax.text(
            left * 1.04,
            y + 0.26,
            lbl,
            ha="left",
            va="bottom",
            fontsize=8.4,
            color=TEXT_INK,
            zorder=5,
        )
        ax.annotate(
            f"{v:,.0f} MiB",
            (v, y),
            textcoords="offset points",
            xytext=(6, 0),
            ha="left",
            va="center",
            fontsize=8.4,
            color=TEXT_INK,
            zorder=5,
        )

    ax.set_title(
        "For scale — B=8, T=1024, fp32, stage 3\n"
        f"the ~150 MiB target is {100 * target / total:.1f}% of this GPU's "
        "footprint",
        color=TEXT_GRAY,
        fontsize=9,
        pad=8,
    )
    return dict(
        shape="B=8,T=1024,fp32,stage3",
        total_mib=total,
        activations_mib=acts,
        logits_mib=logits,
        campaign_target_mib=target,
    )


def render(out_path=None):

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    small = load(SRC_SMALL)
    large = load(SRC_LARGE)

    fig = plt.figure(figsize=(15.2, 6.0))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.15, 1.15, 1.05],
        left=0.052,
        right=0.988,
        top=0.735,
        bottom=0.145,
        wspace=0.22,
    )
    ax_fp32 = fig.add_subplot(gs[0, 0])
    ax_bf16 = fig.add_subplot(gs[0, 1])
    ax_scale = fig.add_subplot(gs[0, 2])
    fig.patch.set_facecolor("white")
    for ax in (ax_fp32, ax_bf16, ax_scale):
        ax.set_facecolor("white")

    # Annotations are positioned in data coordinates against the known
    # collision structure of this baseline.
    fp32_notes = [
        {
            "text": "stages 2 and 3: one nvidia-smi reading,\n"
            "two footprints 60.0 MiB apart",
            "xy": (3.19, 768),
            "xytext": (1.72, 1075),
            "ha": "center",
        },
    ]
    fp32_deltas = [
        {
            "x": 1.50,
            "y0": 512,
            "y1": 768,
            "text": "256 MiB step\nreal: 87.0 MiB",
        }
    ]
    bf16_notes = [
        {
            "text": "stages 1, 2 and 3: one nvidia-smi reading,\n"
            "three footprints spanning 73.5 MiB",
            "xy": (3.19, 768),
            "xytext": (1.70, 1085),
            "ha": "center",
        },
    ]

    d_fp32 = panel_savings(ax_fp32, small, "fp32", fp32_notes, fp32_deltas)
    d_bf16 = panel_savings(ax_bf16, small, "bf16", bf16_notes)
    d_scale = panel_scale(ax_scale, large)

    ax_fp32.set_ylabel(
        "MiB saved vs Stage 0  (MiB = 2²⁰ bytes)",
        color=TEXT_INK,
        fontsize=10,
    )

    fig.suptitle(
        "nvidia-smi cannot resolve what ZeRO sharding actually saves",
        color=TEXT_INK,
        fontsize=14,
        fontweight="bold",
        y=0.972,
    )
    flags = small["flags"]
    fig.text(
        0.5,
        0.917,
        "GPT-2 124M · world size 2 · "
        f"B={flags['b']} T={flags['t']} per rank · "
        f"{small['gpu_count']}× {small['gpu']} · "
        f"{small['host']} · {small['generated'][:10]}",
        color=TEXT_GRAY,
        fontsize=9,
        ha="center",
    )
    fig.text(
        0.5,
        0.874,
        "Left & centre: y-axis ticks are the allocator's 256 MiB commit "
        "granularity — every nvidia-smi bar lands exactly on a tick, "
        "every exact-accounting bar does not.",
        color=TEXT_GRAY,
        fontsize=8.6,
        ha="center",
    )

    legend = fig.legend(
        handles=[
            Patch(
                facecolor=COLOR_EXACT,
                edgecolor=TEXT_INK,
                linewidth=1.1,
                label="exact in-process accounting (LLMM_MEM_REPORT)",
            ),
            Patch(
                facecolor=COLOR_SMI,
                edgecolor=TEXT_INK,
                linewidth=1.1,
                label="nvidia-smi",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.845),
        frameon=False,
        fontsize=9.2,
        ncol=2,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_INK)

    fig.text(
        0.988,
        0.035,
        "† logits size comes from the model's activation size table, not "
        "a direct buffer read, and is already counted inside 'all "
        "activations'.\n"
        "All values measured, per rank (max over 2 ranks). nvidia-smi is a "
        "peak-during-run delta; exact accounting is read at the steady "
        "phase. GPU collectives are verified at world size 2 only.",
        color=TEXT_GRAY,
        fontsize=7.2,
        ha="right",
        va="bottom",
        style="italic",
    )

    if out_path is None:
        gen = small["generated"]
        stamp = f"{gen[:10]}_{gen[11:16].replace(':', '')}"
        name = (
            f"zero_mem_blindness_w{small['world_size']}"
            f"_b{flags['b']}_t{flags['t']}_{stamp}"
            f"_{_gpu_token(small['gpu'])}_{_slug(small['host'])}.png"
        )
        out_path = os.path.join(FIGURES, name)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)

    # Reproducibility contract: the data that made the figure travels with
    # it, at the same stem. This figure draws on two source documents, so
    # the sidecar carries both verbatim plus the exact plotted series.
    sidecar = os.path.splitext(out_path)[0] + ".json"
    with open(sidecar, "w") as fh:
        json.dump(
            {
                "figure": "zero_mem_blindness",
                "rendered": datetime.datetime.now().isoformat(timespec="seconds"),
                "rendered_by": "scripts/figure_zero_blindness.py",
                "provenance": {
                    "all_values": "MEASURED - Team M's LLMM_MEM_REPORT "
                    "baseline plus nvidia-smi sampling. Nothing on this "
                    "figure is modelled or interpolated.",
                    "logits_tensor": "one step removed - read from the "
                    "model's activation size table rather than a live "
                    "DeviceBuffer",
                    "commit_granularity_mib": COMMIT_MIB,
                    "note": "The 256 MiB granularity and the ~696-698 MiB "
                    "nvidia-smi offset are findings computed by arithmetic "
                    "from the observed data, not independently instrumented.",
                },
                "plotted": {
                    "fp32": d_fp32,
                    "bf16": d_bf16,
                    "scale_panel": d_scale,
                },
                "sources": {
                    os.path.relpath(SRC_SMALL, ROOT): small,
                    os.path.relpath(SRC_LARGE, ROOT): large,
                },
            },
            fh,
            indent=2,
        )
    plt.close(fig)
    print(out_path)
    print(sidecar)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output PNG path (default: auto-named)")
    args = ap.parse_args()
    render(args.out)


if __name__ == "__main__":
    main()
