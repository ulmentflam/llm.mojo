"""Render the headline run of the modded-nanogpt adoption campaign: tokens to val 3.28.

Two Muon runs of the recommended recipe (GPT-2 124M, bf16, FineWeb, 524,288
tokens/step, val on the 10.5M-token set via -m 320) against the in-house AdamW
524k run:

  * `-x 3800`: docs/ai/data/adoption_headline_tokens_to_328.jsonl, steps 0-3800.
  * `-x 5700`: docs/ai/data/adoption_headline_x5700.jsonl, resumed from step 2000
    of the first run (the stable phase is identical at both horizons), steps
    2001-5700. Its 0-2000 prefix is the first log's.
  * AdamW baseline: docs/ai/data/adoption_adamw_baseline_val_m20.json, extracted
    from the run's training log (-m 20, 655,360-token val prefix).

    make figure-adoption-headline            # PNG (light + dark), JSON sidecar, doc table
    python3 scripts/figure_adoption_headline.py --extract-baseline   # one-off, reads an ignored log

`--extract-baseline` is the only step that reads outside the tree (the AdamW
run's .autosentry/logs/process.log, override with ADOPTION_BASELINE_LOG); it
writes the committed JSON. Rendering reads committed files only, through
scripts/train_log.py.

Panels: (a) val vs tokens, log x; (b) val vs step zoom with both cooldowns;
(c) learning rate vs step, aligned with (b); (d) max attention logit vs the cap.
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import figure_adoption as fa  # noqa: E402
from scripts.train_log import steps_and_vals  # noqa: E402

DATA = fa.DATA
DOC = fa.DOC
FIG = ROOT / "figures" / "modded_nanogpt_adoption_headline_2026-11-02.png"
LOG_3800 = DATA / "adoption_headline_tokens_to_328.jsonl"
LOG_5700 = DATA / "adoption_headline_x5700.jsonl"
BASELINE = DATA / "adoption_adamw_baseline_val_m20.json"
BASELINE_LOG = Path(
    os.environ.get(
        "ADOPTION_BASELINE_LOG",
        "/home/evan/workspace/llm.mojo/.autosentry/logs/process.log",
    )
)
TPS = 524_288
TARGET = 3.28
LEVELS = (3.29, 3.285, 3.28)
CAP = 8.0
SPLIT = 2000  # -x 5700 resumed from step 2000 of the -x 3800 run
COOLDOWN = {3800: 2280, 5700: 3420}  # floor(x * 0.6)
# facts measured outside the logs (docs/ai/data has no copy of these runs)
ADAMW_FINAL = dict(
    step=19552,
    tokens=19552 * TPS,
    m20=3.2806964,
    m320=3.2900271,
    note="log124M/model_19552.bin evaluated with this repo's trainer at -m 20 and -m 320",
)
RUNS = {
    "x5700": ("recipe, -x 5700", 0),
    "x3800": ("recipe, -x 3800", 1),
    "adamw": ("AdamW 524k (20-batch val)", 2),
}
COL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a"],
    "dark": ["#3987e5", "#d95926", "#199e70"],
}


# ---------------------------------------------------------------- baseline


def extract_baseline() -> None:
    """Pair each `val loss` line with the step line before it; last occurrence wins.

    Resumes repeat steps, so the later reading of a step replaces the earlier
    one. A val line before any step line of its segment (right after a resume
    or at startup) is skipped.
    """
    last = None
    vals: dict[int, float] = {}
    for line in BASELINE_LOG.read_text(errors="replace").splitlines():
        if line.startswith("Resumed from") or line.startswith("| resume"):
            last = None
            continue
        m = re.match(r"step (\d+)/", line)
        if m:
            last = int(m.group(1))
            continue
        m = re.match(r"val loss ([0-9.]+)", line)
        if m and last is not None:
            vals[last] = float(m.group(1))
    pts = [dict(step=s, tokens=s * TPS, val=v) for s, v in sorted(vals.items())]
    BASELINE.write_text(
        json.dumps(
            dict(
                source=".autosentry/logs/process.log of the in-house AdamW 524k run (gitignored)",
                method="val line paired with the preceding step line; last occurrence of a step wins; val lines before the first step of a resume segment skipped",
                val_batches=20,
                val_tokens=20 * 32 * 1024,
                tokens_per_step=TPS,
                points=pts,
            ),
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {BASELINE} ({len(pts)} points)")


# -------------------------------------------------------------------- load


def load() -> dict:
    s38, v38 = steps_and_vals(LOG_3800.read_text())
    s57, v57 = steps_and_vals(LOG_5700.read_text())
    val38 = {v["step"]: v["loss"] for v in v38}
    val57 = {v["step"]: v["loss"] for v in v38 if v["step"] <= SPLIT}
    val57.update({v["step"]: v["loss"] for v in v57})  # step 2000 reproduced
    lr38 = {s["step"]: s["lr"] for s in s38}
    lr57 = {s["step"]: s["lr"] for s in s38 if s["step"] <= SPLIT}
    lr57.update({s["step"]: s["lr"] for s in s57})
    qk38 = {s["step"]: s["qk_max"] for s in s38 if s.get("qk_max") is not None}
    qk57 = {s["step"]: s["qk_max"] for s in s38 if s["step"] <= SPLIT and "qk_max" in s}
    qk57.update({s["step"]: s["qk_max"] for s in s57 if s.get("qk_max") is not None})
    import statistics

    ms = lambda ss: statistics.median(s["ms"] for s in ss if s["step"] > 1)  # noqa: E731
    base = json.loads(BASELINE.read_text())
    ada = {p["step"]: p["val"] for p in base["points"]}
    assert val57[SPLIT] == val38[SPLIT] or abs(val57[SPLIT] - val38[SPLIT]) < 1e-6
    return dict(
        x3800=dict(val=val38, lr=lr38, qk=qk38, ms=ms(s38), steps=3800),
        x5700=dict(
            val=val57, lr=lr57, qk=qk57, ms=ms(s57), steps=5700
        ),  # ms: branch only
        adamw=dict(val=ada, steps=19552),
    )


def first_le(
    val: dict[int, float], level: float, min_step: int = 1
) -> tuple[int, float, int] | None:
    for s in sorted(val):
        if s >= min_step and val[s] <= level:
            prev = max((p for p in val if p < s), default=0)
            return s, val[s], prev
    return None


def tok_b(step: int) -> float:
    return step * TPS / 1e9


def summary(R) -> dict:
    out: dict = {"levels": list(LEVELS), "runs": {}}
    for k, r in R.items():
        d = {}
        for lv in LEVELS:
            f = first_le(r["val"], lv)
            d[f"first_le_{lv}"] = (
                None
                if f is None
                else dict(
                    step=f[0],
                    tokens_B=tok_b(f[0]),
                    val=f[1],
                    bracket_tokens_B=[tok_b(f[2]), tok_b(f[0])],
                )
            )
        last = max(r["val"])
        low = min(r["val"], key=r["val"].get)
        d["final"] = dict(step=last, tokens_B=tok_b(last), val=r["val"][last])
        d["min"] = dict(step=low, tokens_B=tok_b(low), val=r["val"][low])
        if "qk" in r:
            d["qk_max_peak"] = max(r["qk"].values())
            d["median_ms_per_step"] = r["ms"]
        out["runs"][k] = d
    out["adamw_final_checkpoint"] = ADAMW_FINAL
    out["adamw_bias_m20_minus_m320_at_final"] = ADAMW_FINAL["m20"] - ADAMW_FINAL["m320"]
    r5 = R["x5700"]["val"]
    f = first_le(r5, ADAMW_FINAL["m320"])
    assert f is not None
    out["recipe_passes_adamw_final_m320"] = dict(
        threshold=ADAMW_FINAL["m320"],
        first_eval_step=f[0],
        tokens_B=tok_b(f[0]),
        val=f[1],
        previous_eval_step=f[2],
        previous_val=r5[f[2]],
        token_ratio_vs_adamw_final=ADAMW_FINAL["tokens"] / (f[0] * TPS),
    )
    out["recipe_final_vs_adamw_final_m320"] = r5[max(r5)] - ADAMW_FINAL["m320"]
    out["cooldown_start_steps"] = COOLDOWN
    return out


# -------------------------------------------------------------------- plot


def render(R, S, mode: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    th = fa.THEMES[mode]
    col = dict(zip(RUNS, COL[mode]))
    fig = plt.figure(figsize=(14, 12.5), dpi=200)
    fig.patch.set_facecolor(th["bg"])
    outer = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.28, wspace=0.2)
    a = fig.add_subplot(outer[0, :])
    left = outer[1, 0].subgridspec(2, 1, height_ratios=[3, 1.1], hspace=0.08)
    b = fig.add_subplot(left[0])
    c = fig.add_subplot(left[1], sharex=b)
    d = fig.add_subplot(outer[1, 1])

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

    def title(ax, t):
        ax.set_title(t, loc="left", fontsize=10.5, color=th["ink"], fontweight="bold")

    for ax in (a, b, c, d):
        style(ax)
    ink, sub = th["ink"], th["sub"]
    r57, r38, ada = R["x5700"], R["x3800"], R["adamw"]
    f90 = S["runs"]["x5700"]["first_le_3.29"]
    f28 = S["runs"]["x5700"]["first_le_3.28"]
    pas = S["recipe_passes_adamw_final_m320"]
    m320 = ADAMW_FINAL["m320"]

    # ---- (a) val vs tokens, log x
    def xy(r, lo=0):
        ss = [s for s in sorted(r["val"]) if s > lo]
        return [tok_b(s) for s in ss], [r["val"][s] for s in ss]

    x, y = xy(ada)
    a.plot(x, y, color=col["adamw"], lw=2, ls=(0, (5, 3)), zorder=3)
    x, y = xy(r38)
    a.plot(x, y, color=col["x3800"], lw=2.4, zorder=4)
    x, y = xy(r57)
    a.plot(x, y, color=col["x5700"], lw=2.4, zorder=5)
    a.axhline(TARGET, color=sub, lw=1.2, ls=":", zorder=2)
    a.text(
        0.165,
        TARGET - 0.006,
        "val 3.28 (target)",
        fontsize=8,
        color=sub,
        va="top",
        ha="left",
    )
    a.plot(
        [tok_b(ADAMW_FINAL["step"])],
        [m320],
        marker="D",
        ms=10,
        mfc=th["bg"],
        mec=ink,
        mew=2,
        ls="none",
        zorder=7,
    )
    a.annotate(
        f"AdamW final checkpoint on the same val set\n(-m 320): {m320:.4f} at 10.25B",
        (tok_b(ADAMW_FINAL["step"]), m320),
        xytext=(3.0, 3.92),
        textcoords="data",
        ha="left",
        va="top",
        fontsize=8.5,
        color=ink,
        arrowprops=dict(arrowstyle="-", color=sub, lw=1),
    )
    a.plot(
        [tok_b(pas["first_eval_step"])],
        [pas["val"]],
        marker="o",
        ms=8,
        color=col["x5700"],
        mec=th["bg"],
        mew=1.5,
        ls="none",
        zorder=8,
    )
    a.annotate(
        f"-x 5700 reaches {f90['val']:.4f} at {f90['tokens_B']:.2f}B,\n"
        f"below AdamW's 10.25B value ({pas['token_ratio_vs_adamw_final']:.1f}x fewer tokens);\n"
        f"first <= 3.28 at {f28['tokens_B']:.2f}B ({f28['val']:.4f}),\n"
        f"final {S['runs']['x5700']['final']['val']:.4f} at {S['runs']['x5700']['final']['tokens_B']:.2f}B",
        (pas["tokens_B"], pas["val"]),
        xytext=(3.0, 4.45),
        textcoords="data",
        ha="left",
        va="top",
        fontsize=8.5,
        color=ink,
        arrowprops=dict(arrowstyle="-", color=sub, lw=1, relpos=(0.0, 0.0)),
    )
    f38 = S["runs"]["x3800"]["final"]
    a.annotate(
        f"-x 3800: {f38['val']:.4f} at {f38['tokens_B']:.2f}B,\nnever crossed 3.28",
        (f38["tokens_B"], f38["val"]),
        xytext=(1.99, 3.225),
        textcoords="data",
        ha="right",
        va="center",
        fontsize=8.5,
        color=ink,
    )
    la = S["runs"]["adamw"]["final"]
    a.annotate(
        "AdamW, its own 20-batch val: lowest 3.2807 at 10.09B,\nnever <= 3.28; this val reads 0.0093 below the 320-batch val",
        (la["tokens_B"], la["val"]),
        xytext=(3.0, 4.19),
        textcoords="data",
        ha="left",
        va="top",
        fontsize=8.5,
        color=ink,
    )
    a.set_xscale("log")
    a.set_xlim(0.15, 12)
    a.set_ylim(3.15, 4.5)
    a.set_xticks([0.2, 0.5, 1, 2, 3, 5, 10])
    a.set_xticklabels(["0.2", "0.5", "1", "2", "3", "5", "10"])
    a.set_xlabel("training tokens (billions, log scale)")
    a.set_ylabel("val loss (lower is better)")
    hs = [
        Line2D(
            [],
            [],
            color=col["x5700"],
            lw=2.4,
            label="recipe, -x 5700 (val: 320 batches)",
        ),
        Line2D(
            [],
            [],
            color=col["x3800"],
            lw=2.4,
            label="recipe, -x 3800 (val: 320 batches)",
        ),
        Line2D(
            [],
            [],
            color=col["adamw"],
            lw=2,
            ls=(0, (5, 3)),
            label="AdamW 524k, val: 20 batches (not the same val set)",
        ),
        Line2D(
            [],
            [],
            color=ink,
            lw=0,
            marker="D",
            mfc=th["bg"],
            mew=2,
            ms=8,
            label="same AdamW checkpoint, 320 batches",
        ),
    ]
    a.legend(
        handles=hs,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.2),
        fontsize=8.5,
        frameon=False,
        labelcolor=ink,
    )
    title(
        a,
        "(a) Val loss vs tokens: the recipe passes AdamW's final 320-batch value at 2.88B tokens",
    )

    # ---- (b) zoom
    lo, hi = SPLIT, 5750
    for r, k, z in ((r38, "x3800", 4), (r57, "x5700", 5)):
        ss = [s for s in sorted(r["val"]) if s >= lo]
        b.plot(
            ss,
            [r["val"][s] for s in ss],
            color=col[k],
            lw=2.2,
            marker="o",
            ms=3.5,
            zorder=z,
        )
    b.axhline(TARGET, color=sub, lw=1.2, ls=":", zorder=2)
    b.axhline(m320, color=ink, lw=1, ls=(0, (1, 2)), zorder=2)
    b.text(lo + 40, TARGET - 0.002, "3.28", fontsize=8, color=sub, va="top")
    b.text(
        lo + 40,
        m320 + 0.0025,
        "3.2900: AdamW final checkpoint, same val set",
        fontsize=8,
        color=ink,
        va="bottom",
    )
    for x0, k in ((COOLDOWN[3800], "x3800"), (COOLDOWN[5700], "x5700")):
        for ax in (b, c):
            ax.axvline(x0, color=col[k], lw=1.2, ls="--", alpha=0.8, zorder=1)
    b.text(
        COOLDOWN[3800] + 25,
        3.335,
        f"-x 3800 cooldown starts, step {COOLDOWN[3800]}",
        ha="left",
        va="bottom",
        fontsize=8,
        color=ink,
    )
    b.text(
        COOLDOWN[5700] + 25,
        3.57,
        f"-x 5700 cooldown starts, step {COOLDOWN[5700]}",
        ha="left",
        va="bottom",
        fontsize=8,
        color=ink,
    )
    b.plot(
        [f28["step"]],
        [f28["val"]],
        marker="o",
        ms=9,
        color=col["x5700"],
        mec=th["bg"],
        mew=1.5,
        ls="none",
        zorder=8,
    )
    b.annotate(
        f"first <= 3.28: step {f28['step']}\n({f28['tokens_B']:.3f}B, {f28['val']:.4f}),\nbracket {f28['bracket_tokens_B'][0]:.3f}B to {f28['bracket_tokens_B'][1]:.3f}B",
        (f28["step"], f28["val"]),
        xytext=(5000, 3.232),
        textcoords="data",
        ha="right",
        va="center",
        fontsize=8.5,
        color=ink,
        arrowprops=dict(arrowstyle="-", color=sub, lw=1),
    )
    b.annotate(
        f"-x 3800 final {f38['val']:.4f},\ndid not cross",
        (3800, f38["val"]),
        xytext=(8, -2),
        textcoords="offset points",
        va="top",
        fontsize=8.5,
        color=ink,
    )
    b.annotate(
        "runs agree to step 2280; then -x 3800 decays, -x 5700 does not",
        (2300, 3.556),
        xytext=(2330, 3.617),
        va="top",
        fontsize=8.5,
        color=ink,
        arrowprops=dict(arrowstyle="-", color=sub, lw=1),
    )
    b.set_ylim(3.19, 3.62)
    b.set_xlim(lo, hi)
    b.set_ylabel("val loss (320 batches)")
    plt.setp(b.get_xticklabels(), visible=False)
    title(b, "(b) Zoom, steps 2000-5700: the crossing falls in the -x 5700 cooldown")

    # ---- (c) lr
    for r, k in ((r38, "x3800"), (r57, "x5700")):
        ss = [s for s in sorted(r["lr"]) if s >= lo]
        c.plot(ss, [r["lr"][s] * 1e3 for s in ss], color=col[k], lw=2.2, zorder=4)
    c.set_ylim(0, 5.4)
    c.set_xlabel("step (524,288 tokens each)")
    c.set_ylabel("lr (1e-3)")
    c.text(
        hi - 30,
        4.9,
        "WSD: constant 4.8e-3, then linear to 0.1x",
        ha="right",
        va="top",
        fontsize=8,
        color=sub,
    )

    # ---- (d) qk_max
    for r, k in ((r38, "x3800"), (r57, "x5700")):
        ss = sorted(r["qk"])
        d.plot(
            ss,
            [r["qk"][s] for s in ss],
            color=col[k],
            lw=1.2,
            alpha=0.9,
            zorder=4 if k == "x5700" else 3,
        )
    pk = max(S["runs"][k]["qk_max_peak"] for k in ("x3800", "x5700"))
    d.axhline(CAP, color=ink, lw=1.5, zorder=5)
    d.text(
        30,
        CAP + 0.04,
        f"cap 8.0 (unit-RMS rows, head dim 64); run peak {pk:.3f}",
        fontsize=8.5,
        color=ink,
        va="bottom",
    )
    d.set_ylim(5.4, 8.3)
    d.set_xlim(0, 5750)
    d.set_xlabel("step")
    d.set_ylabel("max attention logit (qk_max)")
    d.legend(
        handles=[
            Line2D(
                [],
                [],
                color=col["x5700"],
                lw=2,
                label="recipe, -x 5700 (0-2000 shared)",
            ),
            Line2D([], [], color=col["x3800"], lw=2, label="recipe, -x 3800"),
        ],
        loc="lower right",
        fontsize=8.5,
        frameon=False,
        labelcolor=ink,
    )
    title(d, "(d) QK-norm keeps the max logit under the cap for 5700 steps")

    fig.suptitle(
        "Tokens to val 3.28: Muon recipe vs AdamW, GPT-2 124M, FineWeb, 524,288 tokens/step",
        x=0.01,
        ha="left",
        fontsize=13,
        color=ink,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.006,
        "One seed per horizon. Recipe: WSD cooldown 0.4, momentum ramp, rotary, QK-norm, Muon lr 4.8e-3, pre-fusion kernels (21a2e4c). -x 5700 resumed from step 2000 of the -x 3800 run\n"
        "(its steps 2001-3800 are discarded). Only the diamond in (a) compares on the same val set; AdamW's earlier points use a 20-batch val whose bias there is unmeasured. Single GB10.",
        fontsize=7.5,
        color=sub,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.075)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=th["bg"])
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------- table


def cell(f) -> str:
    if f is None:
        return "not reached"
    return f"{f['step']} / {f['tokens_B']:.3f}B / {f['val']:.5f}"


def table(S) -> str:
    L = [
        "| run | val set | first <= 3.29 (step / tokens / val) | first <= 3.285 | first <= 3.28 | final | lowest |",
        "|---|---|---|---|---|---|---|",
    ]
    names = {
        "x5700": ("recipe `-x 5700`", "10.5M tokens (`-m 320`)"),
        "x3800": ("recipe `-x 3800`", "10.5M tokens (`-m 320`)"),
        "adamw": ("AdamW 524k, 19,552 steps", "655k tokens (`-m 20`)"),
    }
    for k in ("x5700", "x3800", "adamw"):
        r = S["runs"][k]
        fin, mn = r["final"], r["min"]
        L.append(
            f"| {names[k][0]} | {names[k][1]} | "
            + " | ".join(cell(r[f"first_le_{lv}"]) for lv in LEVELS)
            + f" | {fin['step']} / {fin['tokens_B']:.3f}B / {fin['val']:.5f} | "
            + f"{mn['step']} / {mn['tokens_B']:.3f}B / {mn['val']:.5f} |"
        )
    af = ADAMW_FINAL
    L.append(
        f"| AdamW final checkpoint, re-evaluated | 10.5M tokens (`-m 320`) | not measured | not measured | not measured (final {af['m320']:.5f}) | {af['step']} / {af['tokens'] / 1e9:.3f}B / {af['m320']:.5f} | n/a |"
    )
    L.append(
        f"| AdamW final checkpoint, re-evaluated | 655k tokens (`-m 20`) | | | | {af['step']} / {af['tokens'] / 1e9:.3f}B / {af['m20']:.5f} | |"
    )
    p = S["recipe_passes_adamw_final_m320"]
    L.append("")
    L.append(
        f"Same-val comparison: the recipe's first eval at or below AdamW's final 320-batch value "
        f"({p['threshold']:.4f}) is step {p['first_eval_step']} ({p['tokens_B']:.3f}B, {p['val']:.5f}); the eval before it, "
        f"step {p['previous_eval_step']}, read {p['previous_val']:.5f}. That is {p['token_ratio_vs_adamw_final']:.2f}x fewer tokens than AdamW's {af['tokens'] / 1e9:.2f}B. "
        f"The recipe's final value is {S['recipe_final_vs_adamw_final_m320']:+.4f} against AdamW's final checkpoint on that val set. "
        f"The 20-batch val reads {-S['adamw_bias_m20_minus_m320_at_final']:.4f} below the 320-batch val at AdamW's final checkpoint; the bias at earlier checkpoints is unmeasured, "
        "so the AdamW crossing rows are on a different val set from the recipe rows."
    )
    return "\n".join(L)


def main() -> None:
    if "--extract-baseline" in sys.argv:
        extract_baseline()
        return
    R = load()
    S = summary(R)
    if DOC.exists():
        DOC.write_text(fa.splice(DOC.read_text(), "headline", table(S)))
        print(f"updated headline table in {DOC.name}")
    if "--table-only" in sys.argv:
        return
    render(R, S, "light", FIG)
    render(R, S, "dark", FIG.with_name(FIG.stem + "_dark.png"))
    S["data"] = dict(
        x5700={
            k: {str(s): v for s, v in R["x5700"][k].items()}
            for k in ("val", "lr", "qk")
        },
        x3800={
            k: {str(s): v for s, v in R["x3800"][k].items()}
            for k in ("val", "lr", "qk")
        },
        adamw={"val": {str(s): v for s, v in R["adamw"]["val"].items()}},
    )
    FIG.with_suffix(".json").write_text(json.dumps(S, indent=1) + "\n")
    print(f"wrote {FIG.with_suffix('.json')}")


if __name__ == "__main__":
    main()
