"""Render the modded-nanogpt adoption campaign figures and results table.

RE-RUN INSTRUCTIONS (when the ranking queue has drained):

    make figure-adoption-collect   # copy finished sweeps/*.jsonl from the item worktrees
    make figure-adoption           # render PNG (light + dark) + JSON, refresh the report tables

`--collect` (the first step) is the only part that touches worktree paths: it
copies every finished ranking log (w<N>_*.jsonl and timing_*.jsonl, completed
runs only) from <root>/../llm.mojo-wt/*/sweeps/ into docs/ai/data/adoption_*.jsonl.
Override the search root with ADOPTION_WORKTREES. After that, rendering reads
docs/ai/data/adoption_*.jsonl only (no GPU, no worktree paths), so a fresh
checkout of the committed data reproduces the figure.

A new arm needs no code change to appear in the plots: an unknown
adoption_<stem>.jsonl is picked up with its item taken from the w<N> prefix.
Add it to ARMS only to give it a readable label, flags and note. Arms listed in
ARMS with no log yet are rendered as "pending" rows in the report table.

Writes figures/modded_nanogpt_adoption_2026-10-30.png (light surface),
..._dark.png, and a .json sidecar of every plotted number, and rewrites the
tables between the <!-- adoption-*:start/end --> markers in
docs/ai/modded_nanogpt_adoption_results_2026-10-30.md.

Panels: (a) val loss vs step per arm, (b) val@300 delta vs the base with the
single-seed noise band, (c) W1 AdamW-range lr multiplier ladder, (d) QK-norm max
attention logit against the cap, (e) train loss vs step time, (f) gelu vs relu2
step time. Logs are read with scripts/train_log.py only.

Usage:
    pixi run python scripts/figure_adoption.py [--collect | --table-only]
"""

import json
import os
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_log import steps_and_vals  # noqa: E402

DATA = ROOT / "docs" / "ai" / "data"
FIG = ROOT / "figures" / "modded_nanogpt_adoption_2026-10-30.png"
DOC = ROOT / "docs" / "ai" / "modded_nanogpt_adoption_results_2026-10-30.md"
PREFIX = "adoption_"

BASE = "w1_base"
NOISE = 0.02  # single-seed, 300-step val noise band (plan doc section 1)
QK_CAP = 8.0
# Arms whose ms/step the coordinator knows were measured while another job shared
# the GPU. Empty unless the ledger says so; the script also flags unstable step
# times from the log itself (see unstable()).
# The plan doc's section 6 records that w7_qknorm and w8_rotary shared the GPU
# with a verifier compile for part of their runs.
CONTAMINATED: set[str] = {"w7_qknorm", "w8_rotary"}

# stem -> (item, label, flags, note). Item key is also the colour/marker key.
ARMS: dict[str, tuple[str, str, str, str]] = {
    "w1_base": (
        "base",
        "base (Muon mu 0.9)",
        "`-opt muon -l 4.8e-3 -mu 0.9 -c 0.1 -gc 1.0 -u 50`",
        "reference; sweep-doc mu0.9 row was 5.0343",
    ),
    "w1_mult0.125": (
        "W1",
        "W1 adam x0.125",
        "`-l_adam 0.125`",
        "AdamW ranges at 6e-4: the sweep win was partly the AdamW-range rate",
    ),
    "w1_mult2": ("W1", "W1 adam x2", "`-l_adam 2`", ""),
    "w1_mult4": ("W1", "W1 adam x4", "`-l_adam 4`", ""),
    "w1_mult8": ("W1", "W1 adam x8", "`-l_adam 8`", ""),
    "w2_ramp085_090": (
        "W2",
        "W2 ramp .85-.90",
        "`-mu_start 0.85 -mu 0.9 -mu_warmup 300`",
        "",
    ),
    "w2_ramp085_095": (
        "W2",
        "W2 ramp .85-.95",
        "`-mu_start 0.85 -mu 0.95 -mu_warmup 300`",
        "flat 0.95 was 5.0732 in the sweep",
    ),
    "w3_wsd_u0_c040_lin": (
        "W3",
        "W3 wsd u0 c.40",
        "`-k wsd -u 0 -cooldown 0.4 -decay_shape linear`",
        "",
    ),
    "w3_wsd_u0_c045_lin": (
        "W3",
        "W3 wsd u0 c.45",
        "`-k wsd -u 0 -cooldown 0.45 -decay_shape linear`",
        "",
    ),
    "w3_wsd_u50_c040_lin": (
        "W3",
        "W3 wsd u50 c.40",
        "`-k wsd -u 50 -cooldown 0.4 -decay_shape linear`",
        "",
    ),
    "w3_wsd_u0_c040_sqrt": (
        "W3",
        "W3 wsd u0 c.40 sqrt",
        "`-k wsd -u 0 -cooldown 0.4 -decay_shape sqrt`",
        "",
    ),
    "w56_relu2": (
        "W56",
        "W6 relu2",
        "`-D LLMM_MLP_ACT=relu2`",
        "at the noise edge alone",
    ),
    "w56_zeroinit": ("W56", "W5 zero-init proj", "`-D LLMM_ZERO_INIT_PROJ=1`", ""),
    "w56_both": (
        "W56",
        "W5+W6 both",
        "`-D LLMM_ZERO_INIT_PROJ=1 -D LLMM_MLP_ACT=relu2`",
        "",
    ),
    "w56_smallinit": ("W56", "W5 small-init proj", "`-D LLMM_ZERO_INIT_PROJ=2`", ""),
    "w7_qknorm": ("W7", "W7 QK-norm", "`-D LLMM_QK_NORM=1`", "default attention scale"),
    "w7_qknorm_s012": (
        "W7",
        "W7 QK-norm s0.12",
        "`-D LLMM_QK_NORM=1 -attn_scale 0.12`",
        "",
    ),
    "w8_rotary": ("W8", "W8 rotary", "`-D LLMM_ROTARY=1`", "largest single gain"),
    "w9_untied0": (
        "W9",
        "W9 untied head",
        "`-D LLMM_UNTIED_HEAD=1 -untie_at 0`",
        "tied arm is the base",
    ),
    "w9_untied0_lhead0.125": (
        "W9",
        "W9 untied, head x0.125",
        "`-D LLMM_UNTIED_HEAD=1 -untie_at 0 -l_head 0.125`",
        "",
    ),
}

# Items in fixed palette-slot order (never re-sorted by rank). Base is ink.
ITEMS = ["W1", "W2", "W3", "W56", "W7", "W8", "W9"]
ITEM_NAME = {
    "base": "base",
    "W1": "W1 AdamW-range lr",
    "W2": "W2 momentum ramp",
    "W3": "W3 WSD cooldown",
    "W56": "W5/W6 init, relu2",
    "W7": "W7 QK-norm",
    "W8": "W8 rotary",
    "W9": "W9 untied head",
}
MARKERS = {
    "base": "o",
    "W1": "s",
    "W2": "^",
    "W3": "D",
    "W56": "v",
    "W7": "P",
    "W8": "X",
    "W9": "h",
}
# palette.md categorical slots 1,2,3,4,5,7,8-order-safe subset: blue, orange, aqua,
# yellow, magenta, green, violet. Adjacent pairs validated light and dark.
SLOTS = {
    "light": [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
    ],
    "dark": [
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
    ],
}
THEMES = {
    "light": dict(
        bg="#fcfcfb", ink="#0b0b0b", sub="#52514e", grid="#e5e4dd", band="#d9d8d0"
    ),
    "dark": dict(
        bg="#1a1a19", ink="#f0efec", sub="#c3c2b7", grid="#34332f", band="#3a3935"
    ),
}


def item_of(stem: str) -> str:
    if stem in ARMS:
        return ARMS[stem][0]
    head = stem.split("_")[0].upper()
    return head if head in ITEMS else "base"


# ----------------------------------------------------------------- collect


def collect() -> None:
    root = Path(os.environ.get("ADOPTION_WORKTREES", ROOT.parent))
    best: dict[str, Path] = {}
    for p in sorted(root.glob("*/sweeps/*.jsonl")):
        n = p.name
        ranking = n[0] == "w" and n[1:2].isdigit() and n.endswith(".jsonl")
        if not (ranking or n.startswith("timing_")):
            continue
        if not complete(p):
            print(f"skip (unfinished): {p}")
            continue
        if n not in best or p.stat().st_mtime > best[n].stat().st_mtime:
            best[n] = p
    DATA.mkdir(parents=True, exist_ok=True)
    for n, p in sorted(best.items()):
        dst = DATA / (PREFIX + n)
        shutil.copyfile(p, dst)
        print(f"{p} -> {dst.relative_to(ROOT)}")
    print(f"collected {len(best)} logs")


def complete(p: Path) -> bool:
    steps, vals = steps_and_vals(p.read_text())
    if not steps:
        return False
    if p.name.startswith("timing_"):
        return steps[-1]["step"] == steps[-1]["steps"]
    return bool(vals) and vals[-1]["step"] == steps[-1]["steps"]


# -------------------------------------------------------------------- load


def load(path: Path) -> dict:
    steps, vals = steps_and_vals(path.read_text())
    ms = [s["ms"] for s in steps if s["step"] > 1]  # step 1 pays compile/warmup
    med = statistics.median(ms)
    last50 = [s["loss"] for s in steps[-50:]]
    return dict(
        val={v["step"]: v["loss"] for v in vals},
        train_last50=statistics.mean(last50),
        ms=med,
        ms_all=ms,
        unstable=unstable(ms, med),
        qk=[(s["step"], s["qk_max"]) for s in steps if s.get("qk_max") is not None],
        setup=_setup(path),
    )


def unstable(ms: list[float], med: float) -> bool:
    """A step-time series that spikes is a sign another job shared the GPU."""
    spikes = sum(1 for m in ms if m > 1.15 * med)
    return spikes / len(ms) > 0.02 or max(ms) > 1.5 * med


def _setup(path: Path) -> dict:
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            d = json.loads(line)
            if d.get("event") == "setup":
                return d
    return {}


def load_all() -> tuple[dict[str, dict], dict[str, dict]]:
    arms, timing = {}, {}
    for p in sorted(DATA.glob(PREFIX + "*.jsonl")):
        stem = p.name[len(PREFIX) : -len(".jsonl")]
        if stem.startswith("timing_"):
            timing[stem] = load(p)
        else:
            arms[stem] = load(p)
    return arms, timing


def contaminated(stem: str, a: dict) -> bool:
    return stem in CONTAMINATED or a["unstable"]


# -------------------------------------------------------------------- plot


def render(arms: dict, timing: dict, mode: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    th = THEMES[mode]
    colors = {"base": th["ink"], **dict(zip(ITEMS, SLOTS[mode]))}

    def style(ax):
        ax.set_facecolor(th["bg"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(th["sub"])
        ax.grid(True, color=th["grid"], linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=th["sub"], labelsize=8)
        ax.xaxis.label.set_color(th["sub"])
        ax.yaxis.label.set_color(th["sub"])

    fig, axes = plt.subplots(3, 2, figsize=(13, 15), dpi=200)
    fig.patch.set_facecolor(th["bg"])
    for ax in axes.flat:
        style(ax)
    (a, b), (c, d), (e, f) = axes
    base300: float = arms[BASE]["val"][300]

    def label(stem):
        return ARMS[stem][1] if stem in ARMS else stem

    def title(ax, t):
        ax.set_title(t, loc="left", fontsize=10.5, color=th["ink"], fontweight="bold")

    # per-item dash cycle so arms in one item stay distinguishable
    seen: dict[str, int] = {}
    dashes = ["-", "--", ":", "-."]
    look = {}
    for stem in arms:
        it = item_of(stem)
        k = seen.get(it, 0)
        seen[it] = k + 1
        look[stem] = dashes[k % 4]

    # (a) val vs step
    for stem, arm in arms.items():
        it = item_of(stem)
        xs = [s for s in (100, 200, 300) if s in arm["val"]]
        is_base = stem == BASE
        a.plot(
            xs,
            [arm["val"][s] for s in xs],
            color=colors[it],
            ls=look[stem],
            lw=3 if is_base else 1.4,
            marker=MARKERS[it],
            ms=8 if is_base else 5,
            markeredgewidth=0,
            alpha=1 if is_base else 0.9,
            zorder=5 if is_base else 3,
        )
    best = min(arms, key=lambda s: arms[s]["val"][300]) if arms else None
    for stem in {BASE, best} - {None}:
        if stem in arms:
            a.annotate(
                f"{label(stem)} {arms[stem]['val'][300]:.3f}",
                (300, arms[stem]["val"][300]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=th["ink"],
            )
    a.set_xlim(80, 390)
    a.set_xticks([100, 200, 300])
    a.set_ylim(4.65, 6.35)
    a.set_xlabel("step")
    a.set_ylabel("val loss (lower is better)")
    title(a, "(a) Validation loss, every arm (step 0, val ~11.0, omitted)")
    handles = [
        Line2D(
            [], [], color=colors[i], marker=MARKERS[i], ms=6, lw=1.6, label=ITEM_NAME[i]
        )
        for i in ["base"] + ITEMS
        if any(item_of(s) == i for s in arms)
    ]
    a.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        frameon=False,
        labelcolor=th["ink"],
    )

    # (b) delta vs base
    rows = sorted(
        ((s, x["val"][300] - base300) for s, x in arms.items() if s != BASE),
        key=lambda t: t[1],
    )
    ys = range(len(rows))
    b.axvspan(-NOISE, NOISE, color=th["band"], zorder=1)
    b.axvline(0, color=th["ink"], lw=1.2, zorder=2)
    b.barh(
        list(ys),
        [r[1] for r in rows],
        color=[colors[item_of(r[0])] for r in rows],
        height=0.62,
        zorder=3,
    )
    for y, (s, dv) in zip(ys, rows):
        b.annotate(
            f"{dv:+.3f}",
            (dv, y),
            xytext=(4 if dv >= 0 else -4, 0),
            textcoords="offset points",
            ha="left" if dv >= 0 else "right",
            va="center",
            fontsize=8,
            color=th["ink"],
        )
    b.set_yticks(list(ys))
    b.set_yticklabels([label(r[0]) for r in rows], fontsize=8, color=th["ink"])
    b.invert_yaxis()
    lo = min([r[1] for r in rows] + [-0.1])
    hi = max([r[1] for r in rows] + [0.1])
    b.set_xlim(lo * 1.35, hi * 1.25)
    b.set_xlabel(
        f"val@300 minus base ({base300:.4f}); negative is better\ngrey band: +/-{NOISE} single-seed noise"
    )
    title(b, "(b) val@300 versus the base, sorted")

    # (c) W1 ladder
    ladder = sorted(
        (x["setup"].get("adam_lr_mult", 1.0), x["val"][300], s)
        for s, x in arms.items()
        if item_of(s) in ("W1", "base")
        and "adam_lr_mult" in x["setup"]
        and x["setup"]["adam_lr_mult"] > 0
    )
    import math

    lx = [math.log2(m) for m, _, _ in ladder]
    c.axhspan(base300 - NOISE, base300 + NOISE, color=th["band"], zorder=1)
    c.plot(
        lx,
        [v for _, v, _ in ladder],
        color=colors["W1"],
        lw=2,
        marker="s",
        ms=7,
        markeredgewidth=0,
        zorder=3,
    )
    for x, (m, v, _) in zip(lx, ladder):
        c.annotate(
            f"{v:.3f}",
            (x, v),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=th["ink"],
        )
    c.set_xticks(lx)
    c.set_xticklabels([f"{m:g}" for m, _, _ in ladder])
    c.set_xlabel("AdamW-range lr multiplier (log2 axis, x1 = shared rate)")
    c.set_ylabel("val loss at step 300")
    title(c, "(c) W1: the AdamW-range rate must stay at the shared rate")

    # (d) QK-norm max logit
    for stem, arm in arms.items():
        if not arm["qk"]:
            continue
        xs, ys_ = zip(*arm["qk"])
        d.plot(
            xs,
            ys_,
            color=colors[item_of(stem)],
            lw=1.0,
            alpha=0.55,
            zorder=2,
            ls=look[stem],
        )
        wins = [
            (w, max(v for s, v in arm["qk"] if w - 100 < s <= w))
            for w in (100, 200, 300)
        ]
        d.plot(
            *zip(*wins),
            color=colors[item_of(stem)],
            marker=MARKERS[item_of(stem)],
            ms=8,
            lw=0,
            markeredgewidth=0,
            zorder=4,
        )
        for w, v in wins:
            d.annotate(
                f"{v:.2f}",
                (w, v),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=th["ink"],
            )
        d.annotate(
            label(stem),
            (xs[-1], ys_[-1]),
            xytext=(-4, -14),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=th["ink"],
        )
    d.axhline(QK_CAP, color=th["ink"], lw=1.2, ls="--", zorder=3)
    d.annotate(
        f"cap {QK_CAP:g}",
        (2, QK_CAP),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=8,
        color=th["ink"],
    )
    d.set_ylim(0, 9.5)
    d.set_xlabel("step (line: per-step; markers: max per 100-step window)")
    d.set_ylabel("max attention logit")
    title(d, "(d) QK-norm bounds the attention logit below the cap")

    # (e) loss vs ms
    offs = [(7, 6), (7, -12), (7, 14), (7, -22)]
    for k, (stem, arm) in enumerate(sorted(arms.items(), key=lambda t: t[1]["ms"])):
        it = item_of(stem)
        bad = contaminated(stem, arm)
        e.scatter(
            arm["ms"],
            arm["train_last50"],
            s=70 if stem == BASE else 48,
            marker=MARKERS[it],
            facecolors="none" if bad else colors[it],
            edgecolors=colors[it],
            linewidths=1.5,
            zorder=3,
        )
        e.annotate(
            label(stem) + ("*" if bad else ""),
            (arm["ms"], arm["train_last50"]),
            xytext=offs[k % 4],
            textcoords="offset points",
            fontsize=7,
            color=th["ink"],
            ha="left",
        )
    xs_e = [x["ms"] for x in arms.values()]
    ys_e = [x["train_last50"] for x in arms.values()]
    e.set_ylim(min(ys_e) - 0.07, max(ys_e) + 0.05)
    e.set_xlim(min(xs_e) - 40, max(xs_e) + 160)
    e.set_xlabel("median ms/step (steps 2-300)")
    e.set_ylabel("train loss, mean of last 50 steps")
    e.text(
        0.99,
        0.99,
        "hollow, * : step times unstable in the log (another job likely\nshared the GPU); read ms/step as an upper bound",
        transform=e.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=th["sub"],
    )
    title(e, "(e) Loss gain against step-time cost")

    # (f) timing
    groups = {"gelu": [], "relu2": []}
    for stem, t in timing.items():
        for g in groups:
            if g in stem:
                groups[g].append(
                    (stem, t["ms_all"][1:] if len(t["ms_all"]) > 2 else t["ms_all"])
                )
    if all(groups.values()):
        data_ = [[m for _, ms_ in groups[g] for m in ms_] for g in groups]
        bp = f.boxplot(
            data_,
            tick_labels=list(groups),
            widths=0.45,
            patch_artist=True,
            showfliers=False,
        )
        for box, g in zip(bp["boxes"], groups):
            box.set(facecolor=th["band"], edgecolor=th["sub"])
        for part in ("whiskers", "caps", "medians"):
            for ln in bp[part]:
                ln.set(color=th["ink"])
        for i, g in enumerate(groups, 1):
            for j, (stem, ms_) in enumerate(groups[g]):
                f.scatter(
                    [i + (j - 0.5) * 0.14] * len(ms_),
                    ms_,
                    s=8,
                    color=colors["W56"],
                    alpha=0.6,
                    zorder=4,
                )
        med = [statistics.median(x) for x in data_]
        f.set_title("", loc="left")
        title(
            f,
            f"(f) Step time, gelu vs relu2 ({(med[1] / med[0] - 1) * 100:+.1f}% median)",
        )
        f.set_ylabel("ms/step (30-step runs, 2 repeats each)")
    else:
        f.text(
            0.5,
            0.5,
            "pending: gelu vs relu2 timing runs\n(adoption_timing_*.jsonl)",
            ha="center",
            va="center",
            transform=f.transAxes,
            color=th["sub"],
            fontsize=10,
        )
        title(f, "(f) Step time, gelu vs relu2")
        f.set_xticks([])
        f.set_yticks([])

    fig.suptitle(
        "modded-nanogpt adoption campaign, GPT-2 124M, 300-step Muon rankings (seed 42)",
        x=0.01,
        ha="left",
        fontsize=13,
        color=th["ink"],
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.008,
        "FineWeb, bf16, B=32 T=1024 (131k tokens/step), base: Muon lr 4.8e-3 mu 0.9, cosine, u50; single GB10. "
        "Single-seed: treat anything inside the grey band as unranked.",
        fontsize=8,
        color=th["sub"],
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=th["bg"])
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------- tables


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def results_table(arms: dict) -> str:
    base300 = arms[BASE]["val"][300] if BASE in arms else float("nan")
    head = (
        "| arm | flags | val@100 | val@200 | val@300 | vs base | train last-50 | ms/step | note |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---|"
    )
    lines = [head]
    stems = list(ARMS) + [s for s in arms if s not in ARMS]
    for s in stems:
        _, lab, flags, note = ARMS.get(s, (item_of(s), s, "(see log setup)", ""))
        if s not in arms:
            lines.append(
                f"| {lab} | {flags} | pending | pending | pending | | | | queued |"
            )
            continue
        x = arms[s]
        v = x["val"]
        delta = "" if s == BASE else f"{v[300] - base300:+.4f}"
        bad = contaminated(s, x)
        ms = f"{x['ms']:.0f}" + ("*" if bad else "")
        lines.append(
            f"| {lab} | {flags} | {fmt(v[100])} | {fmt(v[200])} | {fmt(v[300])} | {delta} | "
            f"{fmt(x['train_last50'])} | {ms} | {note} |"
        )
    lines.append("")
    lines.append(
        "`*` ms/step unstable in the log (spikes above 1.15x the median, or a step above 1.5x): "
        "another job probably shared the GPU, so read it as an upper bound."
    )
    return "\n".join(lines)


def timing_table(timing: dict) -> str:
    if not timing:
        return "Timing runs pending."
    lines = [
        "| run | median ms/step | min | max | steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for s, t in sorted(timing.items()):
        m = t["ms_all"]
        lines.append(
            f"| {s} | {t['ms']:.1f} | {min(m):.1f} | {max(m):.1f} | {len(m) + 1} |"
        )
    return "\n".join(lines)


def splice(text: str, tag: str, body: str) -> str:
    a, z = f"<!-- adoption-{tag}:start -->", f"<!-- adoption-{tag}:end -->"
    if a not in text or z not in text:
        print(f"warning: markers for {tag} not found in {DOC.name}")
        return text
    head, rest = text.split(a, 1)
    _, tail = rest.split(z, 1)
    return f"{head}{a}\n{body}\n{z}{tail}"


def update_doc(arms: dict, timing: dict) -> None:
    if not DOC.exists():
        return
    t = DOC.read_text()
    t = splice(t, "table", results_table(arms))
    t = splice(t, "timing", timing_table(timing))
    DOC.write_text(t)
    print(f"updated tables in {DOC.relative_to(ROOT)}")


def main() -> None:
    if "--collect" in sys.argv:
        collect()
        return
    arms, timing = load_all()
    if BASE not in arms:
        sys.exit(f"missing {PREFIX}{BASE}.jsonl; run --collect first")
    update_doc(arms, timing)
    if "--table-only" in sys.argv:
        return
    render(arms, timing, "light", FIG)
    render(arms, timing, "dark", FIG.with_name(FIG.stem + "_dark.png"))
    base300 = arms[BASE]["val"][300]
    side = {
        "noise_band": NOISE,
        "qk_cap": QK_CAP,
        "arms": {
            s: dict(
                item=item_of(s),
                val=x["val"],
                delta_vs_base=x["val"][300] - base300,
                train_last50=x["train_last50"],
                ms_median=x["ms"],
                ms_unstable=contaminated(s, x),
                qk_max_per_100=[
                    max(v for st, v in x["qk"] if w - 100 < st <= w)
                    for w in (100, 200, 300)
                ]
                if x["qk"]
                else None,
            )
            for s, x in arms.items()
        },
        "timing": {
            s: dict(ms_median=t["ms"], ms_min=min(t["ms_all"]), ms_max=max(t["ms_all"]))
            for s, t in timing.items()
        },
        "pending": [s for s in ARMS if s not in arms],
    }
    FIG.with_suffix(".json").write_text(json.dumps(side, indent=2) + "\n")
    print(f"wrote {FIG.with_suffix('.json')}")


if __name__ == "__main__":
    main()
