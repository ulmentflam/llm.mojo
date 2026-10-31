"""Render the three-seed confirmation of the combined adoption recipe.

Six 600-step runs (GPT-2 124M, bf16, Muon, FineWeb, 131k tokens/step): arm
`base` (the campaign base) and arm `combined` (the recommended recipe: WSD
cooldown, momentum ramp, rotary, QK-norm) at init seeds 42, 1, 2. Logs are
docs/ai/data/adoption_seed_<arm>_s<seed>_600.jsonl.

    make figure-adoption-seeds-collect   # copy finished logs from the run checkout
    make figure-adoption-seeds           # PNG (light + dark), JSON sidecar, report table

`--collect` is the only step that reads outside the tree (default source: the
main checkout's docs/ai/data; override with ADOPTION_SEEDS_SRC). Rendering reads
the committed logs through scripts/train_log.py and works on partial data: seeds
missing from either arm are left out of the paired panels and listed as pending.

Panels: (a) val vs step, per-seed thin lines + per-arm mean; (b) val@600 paired
by seed; (c) gain (base minus combined) at every val step with the min/max band;
(d) train loss last-50 mean and median ms/step per run.
"""

import json
import os
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import figure_adoption as fa  # noqa: E402

DATA = fa.DATA
DOC = fa.DOC
FIG = ROOT / "figures" / "modded_nanogpt_adoption_seeds_2026-10-31.png"
SRC = Path(
    os.environ.get("ADOPTION_SEEDS_SRC", "/home/evan/workspace/llm.mojo/docs/ai/data")
)
SEEDS = (42, 1, 2)
STEPS = (100, 200, 300, 400, 500, 600)
RAMP_END, COOLDOWN_START = 300, 360  # mu_warmup 300; 600 * (1 - 0.4)
ARMS = {
    "base": ("base (cosine, u50, mu 0.9)", 0),
    "combined": ("combined (WSD + ramp + rotary + QK-norm)", 1),
}
MARK = {42: "o", 1: "s", 2: "^"}
# palette.md slots 1 (blue) and 2 (orange): validated adjacent pair, light and dark
ARM_COLOR = {
    "light": {"base": "#2a78d6", "combined": "#eb6834"},
    "dark": {"base": "#3987e5", "combined": "#d95926"},
}


def path_of(arm: str, seed: int) -> Path:
    return DATA / f"adoption_seed_{arm}_s{seed}_600.jsonl"


def collect() -> None:
    n = 0
    for arm in ARMS:
        for seed in SEEDS:
            name = path_of(arm, seed).name
            src = SRC / name
            if not src.exists():
                print(f"missing: {src}")
            elif not fa.complete(src):
                print(f"skip (unfinished): {src}")
            else:
                shutil.copyfile(src, DATA / name)
                n += 1
                print(f"{src} -> docs/ai/data/{name}")
    print(f"collected {n} logs")


def load_runs() -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {a: {} for a in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            p = path_of(arm, seed)
            if p.exists():
                runs[arm][seed] = fa.load(p)
    return runs


def pairs(runs) -> list[int]:
    return [s for s in SEEDS if s in runs["base"] and s in runs["combined"]]


def mean_curve(runs, arm, steps=STEPS):
    return {
        k: statistics.mean(r["val"][k] for r in runs[arm].values())
        for k in steps
        if runs[arm]
    }


def spread(xs) -> float:
    return max(xs) - min(xs)


def summary(runs) -> dict:
    ps = pairs(runs)
    out: dict = {
        "pairs": ps,
        "pending": [f"{a}_s{s}" for a in ARMS for s in SEEDS if s not in runs[a]],
    }
    out["gain_by_step"] = (
        {
            k: [runs["base"][s]["val"][k] - runs["combined"][s]["val"][k] for s in ps]
            for k in STEPS
        }
        if ps
        else {}
    )
    for arm in ARMS:
        rs = runs[arm]
        if rs:
            v6 = [r["val"][600] for r in rs.values()]
            out[arm] = dict(
                seeds=sorted(rs),
                val600_mean=statistics.mean(v6),
                val600_spread=spread(v6),
                ms_mean=statistics.mean(r["ms"] for r in rs.values()),
            )
    if ps:
        g = out["gain_by_step"][600]
        out["gain600_mean"], out["gain600_min"], out["gain600_max"] = (
            statistics.mean(g),
            min(g),
            max(g),
        )
        med = {a: statistics.median(r["ms"] for r in runs[a].values()) for a in ARMS}
        out["ms_delta_pct"] = (
            med["combined"] / med["base"] - 1
        ) * 100  # median of runs
        if 42 in ps:
            out["ms_delta_pct_s42"] = (
                runs["combined"][42]["ms"] / runs["base"][42]["ms"] - 1
            ) * 100
    return out


def render(runs, summ, mode: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    th = fa.THEMES[mode]
    col = ARM_COLOR[mode]
    fig = plt.figure(figsize=(13, 10.5), dpi=200)
    fig.patch.set_facecolor(th["bg"])
    gs = fig.add_gridspec(2, 2)
    a, b, c = (fig.add_subplot(gs[i]) for i in range(3))
    gd = gs[1, 1].subgridspec(1, 2, wspace=0.35)
    d1, d2 = fig.add_subplot(gd[0]), fig.add_subplot(gd[1])

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

    def pending(ax, msg):
        ax.text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=th["sub"],
            fontsize=10,
        )

    for ax in (a, b, c, d1, d2):
        style(ax)
    ps = summ["pairs"]

    # (a) val vs step
    for arm in ARMS:
        for seed, r in runs[arm].items():
            xs = [k for k in STEPS if k in r["val"]]
            a.plot(
                xs,
                [r["val"][k] for k in xs],
                color=col[arm],
                lw=1.0,
                alpha=0.6,
                marker=MARK[seed],
                ms=5,
                markeredgewidth=0,
                zorder=3,
            )
        if runs[arm]:
            m = mean_curve(runs, arm)
            a.plot(list(m), list(m.values()), color=col[arm], lw=3, zorder=4)
            a.annotate(
                f"{m[600]:.3f}",
                (600, m[600]),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=th["ink"],
            )
    a.set_xlim(50, 690)
    lo6 = min((r["val"][600] for arm in ARMS for r in runs[arm].values()), default=4.2)
    a.set_ylim(lo6 - 0.1, 6.15)
    a.set_xlabel("step")
    a.set_ylabel("val loss (lower is better)")
    handles = [Line2D([], [], color=col[k], lw=3, label=ARMS[k][0]) for k in ARMS]
    handles += [
        Line2D([], [], color=th["sub"], lw=0, marker=MARK[s], ms=6, label=f"seed {s}")
        for s in SEEDS
    ]
    a.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        frameon=False,
        labelcolor=th["ink"],
    )
    title(a, "(a) Val loss, three seeds per arm (thick: arm mean)")

    # (b) paired val@600
    if ps:
        for i, s in enumerate(ps):
            vb, vc = runs["base"][s]["val"][600], runs["combined"][s]["val"][600]
            b.plot([i, i], [vc, vb], color=th["sub"], lw=1.5, zorder=2)
            for arm, v in (("base", vb), ("combined", vc)):
                b.plot(
                    [i],
                    [v],
                    marker=MARK[s],
                    ms=9,
                    color=col[arm],
                    markeredgecolor=th["bg"],
                    markeredgewidth=1.5,
                    ls="none",
                    zorder=4,
                )
            b.annotate(
                f"-{vb - vc:.3f}",
                (i, (vb + vc) / 2),
                xytext=(9, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=th["ink"],
            )
        xm = len(ps) - 0.1
        for arm in ARMS:
            v = [runs[arm][s]["val"][600] for s in ps]
            b.plot(
                [xm + 0.35 * ARMS[arm][1]] * 2,
                [min(v), max(v)],
                color=col[arm],
                lw=3,
                solid_capstyle="butt",
                zorder=3,
            )
            b.plot(
                [xm + 0.35 * ARMS[arm][1]],
                [statistics.mean(v)],
                marker="_",
                ms=16,
                mew=2.5,
                color=th["ink"],
                zorder=5,
                ls="none",
            )
        b.set_xticks(list(range(len(ps))) + [xm + 0.175])
        b.set_xticklabels([f"seed {s}" for s in ps] + ["mean, range"])
        b.set_xlim(-0.6, xm + 0.9)
        g = summ["gain600_mean"]
        b.text(
            0.03,
            0.04,
            f"mean gain {g:.3f}, per-seed {summ['gain600_min']:.3f} to {summ['gain600_max']:.3f}\n"
            f"seed range: base {summ['base']['val600_spread']:.3f}, "
            f"combined {summ['combined']['val600_spread']:.3f}",
            transform=b.transAxes,
            fontsize=9,
            color=th["ink"],
            va="bottom",
            bbox=dict(fc=th["bg"], ec=th["grid"], pad=4),
            zorder=6,
        )
        b.set_ylabel("val loss at step 600")
    else:
        pending(b, "pending: needs a seed with both arms")
    title(b, "(b) Step-600 val loss, paired by seed (labels: gain)")

    # (c) gain by step
    if ps:
        gains = summ["gain_by_step"]
        lo = [min(gains[k]) for k in STEPS]
        hi = [max(gains[k]) for k in STEPS]
        mn = [statistics.mean(gains[k]) for k in STEPS]
        c.axhline(0, color=th["sub"], lw=1, zorder=2)
        c.axvline(RAMP_END, color=th["sub"], lw=1, ls=":", alpha=0.6, zorder=1)
        c.axvline(COOLDOWN_START, color=th["sub"], lw=1, ls=":", alpha=0.6, zorder=1)
        c.fill_between(
            STEPS, lo, hi, color=th["band"], zorder=2, label="min to max over seeds"
        )
        for i, s in enumerate(ps):
            c.plot(
                STEPS,
                [gains[k][i] for k in STEPS],
                ls="none",
                marker=MARK[s],
                ms=5,
                color=th["sub"],
                markeredgewidth=0,
                zorder=3,
            )
        c.plot(STEPS, mn, color=col["combined"], lw=3, zorder=4, label="mean gain")
        c.annotate(
            f"{mn[-1]:.3f}",
            (600, mn[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=th["ink"],
        )
        top = max(hi + [0.05])
        c.text(
            RAMP_END - 6,
            top,
            "momentum ramp ends (300)",
            rotation=90,
            ha="right",
            va="top",
            fontsize=7.5,
            color=th["sub"],
        )
        c.text(
            COOLDOWN_START + 6,
            top,
            "cooldown starts (360)",
            rotation=90,
            ha="left",
            va="top",
            fontsize=7.5,
            color=th["sub"],
        )
        c.set_xlim(50, 680)
        c.legend(loc="best", fontsize=8, frameon=False, labelcolor=th["ink"])
        c.set_xlabel("step")
        c.set_ylabel("gain = base minus combined val loss")
    else:
        pending(c, "pending: needs a seed with both arms")
    title(c, "(c) When the combined recipe pulls ahead")

    # (d) train last-50 and ms/step
    for ax, key, lab in (
        (d1, "train_last50", "train loss, last-50 mean"),
        (d2, "ms", "median ms/step"),
    ):
        for arm in ARMS:
            x0 = ARMS[arm][1]
            vals = []
            for j, (s, r) in enumerate(sorted(runs[arm].items())):
                vals.append(r[key])
                ax.plot(
                    [x0 + (j - 1) * 0.12],
                    [r[key]],
                    marker=MARK[s],
                    ms=8,
                    color=col[arm],
                    markeredgecolor=th["bg"],
                    markeredgewidth=1.2,
                    ls="none",
                    zorder=4,
                )
                if key == "ms" and fa.unstable(r["ms_all"], r["ms"]):
                    ax.annotate(
                        "*",
                        (x0 + (j - 1) * 0.12, r[key]),
                        xytext=(5, 3),
                        textcoords="offset points",
                        color=th["ink"],
                        fontsize=10,
                    )
            if vals:
                ax.plot(
                    [x0 - 0.3, x0 + 0.3],
                    [statistics.mean(vals)] * 2,
                    color=th["ink"],
                    lw=2,
                    zorder=3,
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["base", "combined"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel(lab)
    allms = [r["ms"] for arm in ARMS for r in runs[arm].values()]
    if allms:
        d2.set_ylim(0.9 * min(allms), 1.1 * max(allms))
    d1.set_title(
        "(d) Train loss, last-50 mean",
        loc="left",
        fontsize=10.5,
        color=th["ink"],
        fontweight="bold",
    )
    d2.set_title(
        f"step time {summ['ms_delta_pct']:+.1f}% (median of runs)"
        if ps
        else "median ms/step",
        loc="left",
        fontsize=10.5,
        color=th["ink"],
        fontweight="bold",
    )

    n_pend = len(summ["pending"])
    fig.suptitle(
        "Combined recipe vs campaign base, three init seeds, GPT-2 124M, 600 steps",
        x=0.01,
        ha="left",
        fontsize=13,
        color=th["ink"],
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.008,
        "FineWeb, bf16, B=32 T=1024 (131k tokens/step), Muon lr 4.8e-3; seeds 42, 1, 2 via LLMM_INIT_SEED, same data order; single GB10. "
        "\nHorizontal ticks in (b), (d) are means; * unstable step time."
        + (f"  PENDING: {', '.join(summ['pending'])}." if n_pend else ""),
        fontsize=7.5,
        color=th["sub"],
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=th["bg"])
    plt.close(fig)
    print(f"wrote {out}")


def table(runs, summ) -> str:
    h = (
        "| run | val@100 | val@200 | val@300 | val@400 | val@500 | val@600 | train last-50 | ms/step |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    L = [h]
    for arm in ARMS:
        for s in SEEDS:
            r = runs[arm].get(s)
            if r is None:
                L.append(
                    f"| {arm} s{s} | pending | pending | pending | pending | pending | pending | | |"
                )
                continue
            star = "*" if fa.unstable(r["ms_all"], r["ms"]) else ""
            cells = " | ".join(fa.fmt(r["val"][k]) for k in STEPS)
            L.append(
                f"| {arm} s{s} | {cells} | {fa.fmt(r['train_last50'])} | {r['ms']:.0f}{star} |"
            )
        if runs[arm]:
            m = mean_curve(runs, arm)
            L.append(
                f"| **{arm} mean** | "
                + " | ".join(fa.fmt(m[k]) for k in STEPS)
                + f" | {fa.fmt(statistics.mean(r['train_last50'] for r in runs[arm].values()))}"
                f" | {summ[arm]['ms_mean']:.0f} |"
            )
            L.append(
                f"| {arm} spread (max-min) | "
                + " | ".join(
                    fa.fmt(spread([r["val"][k] for r in runs[arm].values()]))
                    for k in STEPS
                )
                + " | | |"
            )
    if summ["pairs"]:
        for s in summ["pairs"]:
            L.append(
                f"| gain s{s} (base - combined) | "
                + " | ".join(
                    f"{runs['base'][s]['val'][k] - runs['combined'][s]['val'][k]:+.4f}"
                    for k in STEPS
                )
                + " | | |"
            )
        gb = summ["gain_by_step"]
        L.append(
            "| **mean gain** | "
            + " | ".join(f"{statistics.mean(gb[k]):+.4f}" for k in STEPS)
            + f" | | {summ['ms_delta_pct']:+.1f}% (median) |"
        )
        L.append(
            "| gain min to max | "
            + " | ".join(f"{min(gb[k]):+.4f} to {max(gb[k]):+.4f}" for k in STEPS)
            + " | | |"
        )
    L.append("")
    L.append(
        "`*` step time unstable in the log (another job probably shared the GPU): upper bound."
    )
    return "\n".join(L)


def main() -> None:
    if "--collect" in sys.argv:
        collect()
        return
    runs = load_runs()
    if not any(runs.values()):
        sys.exit("no adoption_seed_*_600.jsonl logs; run --collect first")
    summ = summary(runs)
    # qk_max in the step log is proof the rotary + QK-norm binary ran
    for arm, want in (("base", False), ("combined", True)):
        for seed, r in runs[arm].items():
            if bool(r["qk"]) != want:
                sys.exit(
                    f"{arm} s{seed}: qk_max {'missing' if want else 'present'} in log: wrong binary"
                )
    if DOC.exists():
        t = DOC.read_text()
        t = fa.splice(t, "seeds", table(runs, summ))
        DOC.write_text(t)
        print(f"updated seed table in {DOC.name}")
    if "--table-only" in sys.argv:
        return
    render(runs, summ, "light", FIG)
    render(runs, summ, "dark", FIG.with_name(FIG.stem + "_dark.png"))
    side = dict(
        summ,
        runs={
            f"{a}_s{s}": dict(
                val=r["val"],
                train_last50=r["train_last50"],
                ms_median=r["ms"],
                ms_unstable=fa.unstable(r["ms_all"], r["ms"]),
                qk_max_logged=bool(r["qk"]),
            )
            for a in ARMS
            for s, r in runs[a].items()
        },
    )
    FIG.with_suffix(".json").write_text(json.dumps(side, indent=2) + "\n")
    print(f"wrote {FIG.with_suffix('.json')}")


if __name__ == "__main__":
    main()
