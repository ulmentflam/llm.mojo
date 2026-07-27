"""Render the vocab-parallel communication break-even figure.

BACKGROUND FOR A READER MEETING THIS COLD. When a model is trained on N
GPUs at once, the GPUs must exchange data every step. Two designs are
compared here for the LM head (the final layer that turns hidden states
into one score per vocabulary entry):

  * TODAY'S APPROACH. The token-embedding table is sharded across GPUs and
    gathered back whenever it is needed. What crosses the wire is therefore
    proportional to the SIZE OF THE TABLE, and does not depend on how many
    tokens are in the batch.
  * VOCAB-PARALLEL (the Megatron-style alternative). Each GPU permanently
    owns a slice of the vocabulary, so the table never moves; instead the
    GPUs exchange per-token quantities. What crosses the wire is therefore
    proportional to the NUMBER OF TOKENS, and does not depend on the table.

One cost is fixed and one grows with batch size, so there is a crossover.
Below it vocab-parallel moves less data; above it, more. This figure shows
where that crossover sits relative to the shapes we actually train at.

EVERYTHING ON THIS FIGURE IS MODELLED -- closed-form byte counts from Team
V's cost model, with no measured input. That is stated on the figure face
rather than only in this docstring. It needs no measurement: the two
designs move bytes over the same links by the same mechanism, so the link
rate cancels in the ratio.

Usage:
    pixi run python scripts/figure_vocab_breakeven.py
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
SRC = os.path.join(
    ROOT, "docs", "ai", "data", "vocab_parallel_cost_model_2026-07-27.json"
)

# Diverging use of palette slots 1 and 2: which side of the break-even a
# point falls on. Blue = vocab-parallel moves less; orange = it moves more.
COLOR_WIN = "#2a78d6"
COLOR_LOSE = "#eb6834"


def _slug(s):
    import re

    return re.sub(r"[^A-Za-z0-9.]+", "-", s).strip("-") or "x"


def load():
    with open(SRC) as fh:
        return json.load(fh)


def points(doc):
    """Every (global_tokens, ratio) pair the model file contains.

    The ratio depends only on global tokens per micro-step -- not on how
    that total is split into ranks, batch or sequence length -- so the
    shape records and the crossover sweep all lie on one line.
    """
    seen = {}
    for rec in doc["shapes"]["fp32"]:
        seen[rec["global_tokens"]] = rec["ratio"]
    for rec in doc["sweep_for_plotting"]["points"]:
        seen[rec["global_tokens"]] = rec["ratio"]
    xs = sorted(seen)
    return xs, [seen[x] for x in xs]


def named_shape(doc, b, t, n):
    for rec in doc["shapes"]["fp32"]:
        if rec["B"] == b and rec["T"] == t and rec["N"] == n:
            return rec
    raise KeyError(f"no shape record for B={b} T={t} N={n}")


def render(out_path=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    doc = load()
    consts = doc["constants"]
    w = consts["z3_embed_window_elements"]
    c = consts["C"]
    v_p = consts["V_p"]

    # The model's exact break-even, solved from its own closed form:
    # ratio = global_tokens * (C + 3) / W, so ratio == 1 at W / (C + 3).
    # This is the analytic crossover of the committed formula, not a
    # sampled data point, and not an interpolation between samples.
    crossover = w / (c + 3.0)

    xs, ys = points(doc)
    bench = named_shape(doc, 4, 64, 7)
    prod = named_shape(doc, 32, 1024, 7)
    swing = prod["ratio"] / bench["ratio"]

    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    fig.subplots_adjust(left=0.088, right=0.975, top=0.745, bottom=0.135)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Split the line at the analytic crossover. The model is exactly linear
    # in global tokens, so this join is exact rather than interpolated.
    left_x = [x for x in xs if x < crossover] + [crossover]
    left_y = [y for x, y in zip(xs, ys) if x < crossover] + [1.0]
    right_x = [crossover] + [x for x in xs if x >= crossover]
    right_y = [1.0] + [y for x, y in zip(xs, ys) if x >= crossover]

    ax.axhspan(1.0, 1e3, color=GRID_COLOR, alpha=0.55, zorder=0)

    ax.axhline(1.0, color=TEXT_GRAY, linewidth=1.0, zorder=2)
    ax.axvline(crossover, color=TEXT_GRAY, linewidth=1.0, zorder=2)

    ax.plot(
        left_x,
        left_y,
        color=COLOR_WIN,
        linewidth=2.0,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        right_x,
        right_y,
        color=COLOR_LOSE,
        linewidth=2.0,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        [x for x in xs if x < crossover],
        [y for x, y in zip(xs, ys) if x < crossover],
        linestyle="none",
        marker="o",
        markersize=4.6,
        markerfacecolor=COLOR_WIN,
        markeredgecolor="white",
        markeredgewidth=1.4,
        zorder=5,
    )
    ax.plot(
        [x for x in xs if x >= crossover],
        [y for x, y in zip(xs, ys) if x >= crossover],
        linestyle="none",
        marker="o",
        markersize=4.6,
        markerfacecolor=COLOR_LOSE,
        markeredgecolor="white",
        markeredgewidth=1.4,
        zorder=5,
    )

    for rec, color, label, off, ha in (
        (bench, COLOR_WIN, "bench shape\nB=4  T=64  N=7", (18, -40), "left"),
        (
            prod,
            COLOR_LOSE,
            "production shape\nB=32  T=1024  N=7",
            (-26, 52),
            "right",
        ),
    ):
        ax.plot(
            [rec["global_tokens"]],
            [rec["ratio"]],
            marker="o",
            markersize=11,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=2.0,
            zorder=6,
        )
        factor = (
            f"{1 / rec['ratio']:.0f}× LESS communication"
            if rec["ratio"] < 1
            else f"{rec['ratio']:.2f}× MORE communication"
        )
        ax.annotate(
            f"{label}\n{factor}",
            (rec["global_tokens"], rec["ratio"]),
            textcoords="offset points",
            xytext=off,
            ha=ha,
            va="center",
            fontsize=9,
            color=TEXT_INK,
            zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.36",
                facecolor="white",
                edgecolor=TEXT_GRAY,
                linewidth=0.8,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=TEXT_GRAY,
                linewidth=0.9,
                shrinkA=2,
                shrinkB=8,
            ),
        )

    ax.annotate(
        f"break-even: {crossover:,.0f} global tokens\n"
        f"(the rule of thumb is V_p = {v_p:,}; the extra 1.6%\n"
        "is the model's three per-token all-reduces)",
        (crossover, 0.0135),
        textcoords="offset points",
        xytext=(14, 0),
        ha="left",
        va="center",
        fontsize=8.6,
        color=TEXT_INK,
        zorder=7,
    )

    ax.text(
        0.038,
        0.80,
        f"same code, same technique —\na {swing:.0f}× swing driven purely by shape",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9.6,
        color=TEXT_INK,
        style="italic",
        zorder=7,
    )

    ax.set_xlim(min(xs) * 0.55, max(xs) * 2.4)
    ax.set_ylim(0.006, 22)
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.9, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_GRAY)
    ax.tick_params(colors=TEXT_GRAY, labelsize=9)

    ax.set_xlabel(
        "Global tokens per micro-step  (N ranks × B batch × T sequence)",
        color=TEXT_INK,
        fontsize=10.5,
    )
    ax.set_ylabel(
        "Communication volume, vocab-parallel ÷ today\n"
        "(per rank, per micro-step; 1.0 = the two are equal)",
        color=TEXT_INK,
        fontsize=10.5,
    )

    legend = ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=COLOR_WIN,
                linewidth=2.0,
                marker="o",
                markersize=6,
                markeredgecolor="white",
                label="vocab-parallel moves less data (ratio < 1)",
            ),
            Line2D(
                [],
                [],
                color=COLOR_LOSE,
                linewidth=2.0,
                marker="o",
                markersize=6,
                markeredgecolor="white",
                label="vocab-parallel moves more data (ratio > 1)",
            ),
        ],
        loc="upper left",
        frameon=False,
        fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_INK)

    fig.suptitle(
        "Vocab-parallelism loses at our token count — on any cluster",
        color=TEXT_INK,
        fontsize=14,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.905,
        "GPT-2 124M · V_p = 50,304 · C = 768 · the ratio depends ONLY on "
        "global tokens per micro-step: the rank count N cancels out, and so "
        "does how the tokens\nsplit between batch and sequence length. "
        "All 20 shapes in the source model lie on this one line — so a "
        "different world size cannot rescue it.",
        color=TEXT_GRAY,
        fontsize=8.8,
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.836,
        "MODELLED — closed-form byte counts from Team V's cost model. "
        "No measured quantity appears on this figure,\nand none is needed: "
        "both designs move bytes over the same links, so the link rate "
        "cancels in the ratio.",
        color=TEXT_INK,
        fontsize=8.8,
        ha="center",
        va="top",
    )

    fig.text(
        0.975,
        0.028,
        "The model is deliberately CHARITABLE to vocab-parallelism: it "
        "ignores latency and barrier costs, and omits the extra B×T×C "
        "forward all-reduce a vocab-parallel encoder would need.\n"
        "Both omissions push the true break-even further LEFT, so the "
        "region where vocab-parallelism loses is if anything larger than "
        "drawn. Volumes are per rank, per micro-step; the ratio is "
        "dtype-independent.",
        color=TEXT_GRAY,
        fontsize=7.2,
        ha="right",
        va="bottom",
        style="italic",
    )

    if out_path is None:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        name = (
            f"vocab_parallel_breakeven_n2-8_{stamp}"
            f"_modelled_{_slug(os.uname().nodename)}.png"
        )
        out_path = os.path.join(FIGURES, name)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)

    sidecar = os.path.splitext(out_path)[0] + ".json"
    with open(sidecar, "w") as fh:
        json.dump(
            {
                "figure": "vocab_parallel_breakeven",
                "rendered": datetime.datetime.now().isoformat(timespec="seconds"),
                "rendered_by": "scripts/figure_vocab_breakeven.py",
                "provenance": {
                    "all_values": "MODELLED - closed-form, zero measured "
                    "input. Nothing on this figure was measured.",
                    "crossover_global_tokens": crossover,
                    "crossover_derivation": "ratio = global_tokens * (C+3) "
                    "/ W, so ratio == 1 at W/(C+3). Analytic solution of "
                    "the source model's own formula, not an interpolation "
                    "between sampled points.",
                    "rule_of_thumb_V_p": v_p,
                },
                "plotted": {
                    "global_tokens": xs,
                    "ratio": ys,
                    "bench_shape": bench,
                    "production_shape": prod,
                    "swing": swing,
                },
                "source": {os.path.relpath(SRC, ROOT): doc},
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
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()
    render(args.out)


if __name__ == "__main__":
    main()
