#!/usr/bin/env python3
"""HellaSwag eval comparison: llm.mojo vs Karpathy's llm.c reference.

Runs `make eval` (or reads its cached result) to get our HellaSwag accuracy on
`log124M/model_19552.bin` (GPT-2 124M, d12, trained from scratch on FineWeb
classic 10B tokens), computes a Wilson score confidence interval for it (the
normal approximation ±1.96·SE is a poor fit for a proportion this close to the
edge of "well-calibrated" at n=10042), and compares it against three published
reference points, each cited to its primary source:

  1. Karpathy's llm.c reproduction of the IDENTICAL setup (124M, d12, 10B
     FineWeb tokens) — github.com/karpathy/llm.c discussion #481: "We get to
     29.9 here". This is the only reference point that's a true apples-to-
     apples comparison (same architecture, same dataset, same token budget),
     so it's the one we test statistical significance against.
  2. The original OpenAI-pretrained GPT-2 124M checkpoint, evaluated with
     llm.c's own "completion style" scoring (third_party/llm.c/dev/data/
     hellaswag.py's docstring: "this script: 10042 acc_norm: 0.2955") — same
     methodology as ours, different training regime (WebText, not FineWeb).
  3. GPT-3 Small (124M), per the GPT-3 paper Appendix H, cited in the same
     llm.c discussion: 33.7% — trained on 300B tokens (30x our budget) with
     an unspecified/likely-different eval methodology. Scale context only;
     not a methodology-matched comparison.

Metric note: our `eval_stat_correct` (llmm/eval_dataloader.mojo) and llm.c's
`evalloader_stat_losses` (third_party/llm.c/llmc/dataloader.h) are both
length-normalized — the predicted completion is the one with the lowest
AVERAGE per-masked-token loss, not raw summed loss. This is "acc_norm" in
Eleuther-harness terminology, NOT the un-normalized "acc" (llm.c's own
hellaswag.py docstring reports both for the same checkpoint — acc 0.2859 vs
acc_norm 0.2955 — a ~1pp gap, so getting this right matters). Karpathy's
"29.9" is very likely acc_norm too: train_gpt2.cu's training loop only ever
computes accuracy via `evalloader_stat_losses`, which is the same length-
normalized computation, and discussion #481's plot is generated from exactly
that training loop.

Usage:
    python scripts/benchmark_eval.py                   # runs `make eval` fresh
    python scripts/benchmark_eval.py --k 2965 --n 10042 # skip the run, use given counts
"""

import argparse
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.benchmark_train import (
    FIGURES,
    TEXT_GRAY,
    TEXT_INK,
    GRID_COLOR,
    FAMILY_COLORS,
    hardware_info,
    _slug,
)

matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Reference:
    label: str
    pct: float
    source: str
    # Same completion-style/acc_norm methodology AND same architecture/dataset
    # budget as ours (True), vs. included as scale/methodology context only
    # (False) — see wilson_ci()'s docstring for why we don't fabricate a CI
    # for any of these from a bare cited percentage either way.
    comparable: bool


REFERENCES: tuple[Reference, ...] = (
    Reference(
        label="Karpathy llm.c\n(same setup: 10B FineWeb)",
        pct=29.9,
        source="github.com/karpathy/llm.c discussion #481",
        comparable=True,  # same architecture, dataset, token budget, metric
    ),
    Reference(
        label="GPT-2 124M original\n(OpenAI WebText checkpoint)",
        pct=29.55,
        source='third_party/llm.c/dev/data/hellaswag.py ("this script: acc_norm 0.2955")',
        comparable=True,  # same completion-style/acc_norm methodology
    ),
    Reference(
        label="GPT-3 Small 124M\n(300B tokens, 30x our budget)",
        pct=33.7,
        source="GPT-3 paper Appendix H, via llm.c discussion #481",
        comparable=False,  # different training scale AND likely different eval methodology
    ),
)


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion — accurate at any n/p,
    unlike the normal approximation (p_hat ± z*SE), which under/overshoots
    near 0 or 1 and is a worse fit than Wilson even at moderate p far from the
    edges. We only compute this for OUR result, where we have the exact (k,
    n) — NOT for cited reference percentages, where we only have a rounded
    point estimate and no guarantee of the same n or raw count. Fabricating a
    CI for those from incomplete public information would overstate their
    precision; the honest comparison is: does the reference point fall inside
    OUR interval.
    """
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return phat, center - half, center + half


def two_proportion_p_value(k, n, ref_pct):
    """Illustrative two-proportion z-test, ASSUMING the reference used n=10042
    too (the standard HellaSwag val split size) and treating its rounded
    percentage as if it were an exact count. This is a secondary, explicitly-
    caveated supplement to the Wilson-CI comparison above, not a substitute
    for it — we don't actually know the reference's true n or raw count.
    """
    p1 = k / n
    p2 = ref_pct / 100
    n2 = n  # assumption, stated above
    p_pool = (k + p2 * n2) / (n + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z_stat = (p1 - p2) / se
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_stat) / math.sqrt(2))))
    return z_stat, p_value


def run_make_eval():
    """Runs `make eval` and parses its final "HellaSwag: k / n = acc" line."""
    print(
        "Running `make eval` (this scores the full 10,042-example HellaSwag "
        "val split — a few minutes)..."
    )
    out = subprocess.run(
        ["make", "eval"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout
    m = re.search(r"HellaSwag:\s*(\d+)\s*/\s*(\d+)\s*=\s*([\d.]+)", out)
    if not m:
        print(out[-2000:])
        raise RuntimeError("could not parse `make eval` output for the HellaSwag line")
    return int(m.group(1)), int(m.group(2))


def render(k, n, outpath, info):
    phat, lo, hi = wilson_ci(k, n)
    phat_pct, lo_pct, hi_pct = phat * 100, lo * 100, hi * 100

    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)

    ax.set_ylim(0, max(38, hi_pct + 4))
    ax.set_ylabel("HellaSwag accuracy (acc_norm, %)", color=TEXT_GRAY, fontsize=9.5)
    ax.tick_params(axis="y", colors=TEXT_GRAY, labelsize=9)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)

    # Our measured result: one bar with a real 95% Wilson CI as error bars —
    # this is the only value on the chart with sampling uncertainty drawn,
    # because it's the only one we have raw (k, n) for.
    bar_color = FAMILY_COLORS["llm.mojo"]
    ax.bar(
        [0],
        [phat_pct],
        width=0.5,
        color=bar_color,
        zorder=3,
        yerr=[[phat_pct - lo_pct], [hi_pct - phat_pct]],
        error_kw=dict(ecolor=TEXT_INK, elinewidth=1.6, capsize=6, zorder=4),
    )
    ax.text(
        0,
        hi_pct + 0.5,
        f"{phat_pct:.2f}%  (95% CI {lo_pct:.1f}–{hi_pct:.1f})",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TEXT_INK,
        zorder=5,
    )
    ax.set_xticks([0])
    ax.set_xticklabels(
        ["llm.mojo (ours)\nfrom-scratch, 10B FineWeb"], fontsize=9.5, color=TEXT_INK
    )
    ax.set_xlim(-1.0, 1.0)

    # Reference points: dashed/dotted horizontal lines, not bars — these are
    # cited percentages, not our own measurements, and drawing them as bars
    # would visually imply the same evidentiary weight as the CI'd bar above.
    # Karpathy's 29.9 and the GPT-2-original 29.55 are only 0.35pp apart, far
    # too close to label inline at the line without the text colliding — so
    # labels live in a fixed-position legend box (axes-fraction coordinates,
    # decoupled from data values) instead; linestyle order ties a legend line
    # back to its axhline (both iterate REFERENCES in the same fixed order).
    LINESTYLES = ["--", "-.", ":"]
    for ref, ls in zip(REFERENCES, LINESTYLES):
        color = TEXT_INK if ref.comparable else TEXT_GRAY
        alpha = 0.85 if ref.comparable else 0.55
        ax.axhline(
            ref.pct, color=color, linestyle=ls, linewidth=1.3, alpha=alpha, zorder=2
        )

    legend_lines = [
        f"{ref.label.splitlines()[0]}: {ref.pct:.2f}%" for ref in REFERENCES
    ]
    legend_text = "\n".join(legend_lines)
    ax.text(
        0.985,
        0.985,
        legend_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=TEXT_GRAY,
        linespacing=1.9,
        bbox=dict(facecolor="white", edgecolor=GRID_COLOR, boxstyle="round,pad=0.5"),
        zorder=6,
    )

    fig.suptitle(
        "HellaSwag accuracy — llm.mojo vs published references",
        fontsize=13,
        color=TEXT_INK,
        x=0.02,
        ha="left",
    )
    sig_z, sig_p = two_proportion_p_value(k, n, REFERENCES[0].pct)
    subtitle = (
        f"n={n:,} val examples  |  Karpathy's {REFERENCES[0].pct:.1f}% falls inside our "
        f"95% CI (not significantly different, illustrative p={sig_p:.2f})"
    )
    ax.set_title(subtitle, fontsize=9.5, color=TEXT_GRAY, pad=10)

    footer = (
        f"llm.mojo: {k}/{n} correct, log124M/model_19552.bin (GPT-2 124M, d12, from-scratch, "
        f"FineWeb classic 10B tokens, bf16). Reproduce: make eval. "
        f"{info['platform']}, {info['date'][:10]}."
    )
    import textwrap

    fig.supxlabel(
        textwrap.fill(footer, width=100), fontsize=7.5, color=TEXT_GRAY, style="italic"
    )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    print(f"\nEval comparison chart written to {outpath}")
    return phat_pct, lo_pct, hi_pct, sig_z, sig_p


def default_output(info):
    date = info["date"][:16].replace(" ", "_").replace(":", "")
    hw = _slug(info["gpu"]) if info["gpu"] != "none" else _slug(info["cpu"])
    name = f"hellaswag_eval_{date}_{hw}_{_slug(info['platform'])}.png"
    return os.path.join(FIGURES, name)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--k", type=int, default=None, help="correct count (skip running `make eval`)"
    )
    parser.add_argument(
        "--n", type=int, default=None, help="total examples (skip running `make eval`)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output PNG path (default: auto, under figures/)",
    )
    args = parser.parse_args()

    if args.k is not None and args.n is not None:
        k, n = args.k, args.n
    else:
        k, n = run_make_eval()

    info = hardware_info()
    outpath = args.out or default_output(info)
    phat, lo, hi, z, p = render(k, n, outpath, info)

    print(f"\nk/n = {k}/{n} = {phat:.4f}%")
    print(f"Wilson 95% CI: [{lo:.4f}%, {hi:.4f}%]")
    print(f"Karpathy's {REFERENCES[0].pct}% inside CI: {lo <= REFERENCES[0].pct <= hi}")
    print(
        f"Illustrative two-proportion z={z:.3f}, p={p:.3f} (not significant if p > 0.05)"
    )


if __name__ == "__main__":
    main()
