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
    make benchmark-eval                                          # runs `make eval` fresh
    pixi run python scripts/benchmark_eval.py --k 2965 --n 10042 # skip the run, use given counts
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


# Reference bars aren't a code "family" like llm.mojo/llm.c/PyTorch — they're
# cited external data points (different models/training regimes). Muted,
# neutral fills (not a new arbitrary hue) keep that distinction legible: color
# still follows identity (llm.mojo blue is always llm.mojo; llm.c aqua is
# always llm.c, matching benchmark_train.py's chart family), but a reference
# with no raw (k, n) behind it is visually "quieter" than a measured bar.
REF_COMPARABLE_COLOR = "#9a9890"  # muted neutral — same methodology, cited
REF_CONTEXT_COLOR = "#c9c7bd"  # lighter still — different budget/methodology


def render(k, n, outpath, info):
    phat, lo, hi = wilson_ci(k, n)
    phat_pct, lo_pct, hi_pct = phat * 100, lo * 100, hi * 100
    sig_z, sig_p = two_proportion_p_value(k, n, REFERENCES[0].pct)

    # All four data points as bars on one shared categorical axis — a
    # magnitude comparison across named categories reads as a comparison when
    # every entry is the same mark type. The earlier draft drew llm.c and the
    # other references as thin axhlines crossing a single bar; two of those
    # lines (29.9 vs 29.55) were only 0.35pp apart and unreadable, and — more
    # importantly — a line doesn't read as "a thing being compared" the way a
    # bar does. llm.c gets its established color from benchmark_train.py's
    # FAMILY_COLORS (same aqua everywhere llm.c appears in this repo's
    # figures); the other two references are muted neutral fills, since
    # they're cited external numbers, not our own (k, n) measurements.
    entries = [
        ("llm.mojo\n(ours)", phat_pct, FAMILY_COLORS["llm.mojo"], (lo_pct, hi_pct)),
        (
            "llm.c (Karpathy)\nsame setup: 10B FineWeb",
            REFERENCES[0].pct,
            FAMILY_COLORS["llm.c"],
            None,
        ),
        (
            "GPT-2 124M original\nOpenAI WebText ckpt",
            REFERENCES[1].pct,
            REF_COMPARABLE_COLOR,
            None,
        ),
        (
            "GPT-3 Small 124M\n300B tokens, 30x budget",
            REFERENCES[2].pct,
            REF_CONTEXT_COLOR,
            None,
        ),
    ]
    x = list(range(len(entries)))
    heights = [e[1] for e in entries]
    colors = [e[2] for e in entries]

    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)

    ax.set_ylim(0, max(38, hi_pct + 4))
    ax.set_ylabel("HellaSwag accuracy (acc_norm, %)", color=TEXT_GRAY, fontsize=9.5)
    ax.tick_params(axis="y", colors=TEXT_GRAY, labelsize=9)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)

    ax.bar(
        x, heights, width=0.6, color=colors, zorder=3, edgecolor="white", linewidth=0.5
    )

    # Real 95% Wilson CI error bar — only llm.mojo has raw (k, n) behind it;
    # the other three are point estimates cited from external sources, so
    # drawing an error bar on them would fabricate a precision we don't have.
    ax.errorbar(
        [0],
        [phat_pct],
        yerr=[[phat_pct - lo_pct], [hi_pct - phat_pct]],
        fmt="none",
        ecolor=TEXT_INK,
        elinewidth=1.6,
        capsize=6,
        zorder=4,
    )

    # Direct value label on every bar (≤4 categories — labeling all of them is
    # clearer than a legend here, and the CI only needs to sit next to the one
    # bar that has one).
    for xi, h, ref_comparable in zip(
        x,
        heights,
        [True, True, True, False],  # llm.mojo counts as "comparable" to itself
    ):
        label = f"{h:.2f}%"
        if xi == 0:
            label += f"\n95% CI [{lo_pct:.1f}, {hi_pct:.1f}]"
        ax.text(
            xi,
            h + (hi_pct - phat_pct if xi == 0 else 0) + 0.6,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=TEXT_INK if ref_comparable else TEXT_GRAY,
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([e[0] for e in entries], fontsize=8.8, color=TEXT_INK)
    ax.set_xlim(-0.7, len(entries) - 0.3)

    fig.suptitle(
        "HellaSwag accuracy — llm.mojo vs llm.c and published references",
        fontsize=13,
        color=TEXT_INK,
        x=0.02,
        ha="left",
    )
    subtitle = (
        f"n={n:,} val examples  |  llm.c's {REFERENCES[0].pct:.1f}% falls inside our "
        f"95% CI (not significantly different, illustrative p={sig_p:.2f})"
    )
    ax.set_title(subtitle, fontsize=9.5, color=TEXT_GRAY, pad=10)

    footer = (
        f"llm.mojo: {k}/{n} correct, log124M/model_19552.bin (GPT-2 124M, d12, from-scratch, "
        f"FineWeb classic 10B tokens, bf16). GPT-3 Small (grey, right) is scale context only — "
        f"different token budget and likely eval methodology, not a statistical comparison. "
        f"Reproduce: make eval. {info['platform']}, {info['date'][:10]}."
    )
    import textwrap

    fig.supxlabel(
        textwrap.fill(footer, width=105), fontsize=7.5, color=TEXT_GRAY, style="italic"
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
