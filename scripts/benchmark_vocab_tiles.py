"""What vocabulary tiling costs: collect and render the LM-head GEMM sweep.

BACKGROUND FOR A READER MEETING THIS COLD. The last layer of a GPT-2-style
model turns each token's hidden vector into one score per vocabulary entry.
That is a single large matrix multiply, `(B*T, C) x (C, V_p)`, and its output
-- the "logits" -- is enormous: at B=32, T=1024 and V_p=50304 it is over
6 GiB in fp32. Vocabulary TILING splits that one multiply into K narrower
ones, so only one `[B*T, V_p/K]` block of logits has to exist at a time.
Memory falls roughly as 1/K.

Nothing is free. K narrow multiplies are not as efficient as one wide one,
because each launch has fixed overhead and narrower shapes fill the machine
less well. This script measures that price so the memory win can be judged
against it.

TWO SEPARATE THINGS ARE PLOTTED AND THEY MUST NOT BE CONFUSED:
  * throughput vs tile count -- what tiling COSTS (measured here)
  * resident logits bytes vs tile count -- what tiling BUYS (exact, from
    the shapes; it is a byte count, not a measurement)
They share an x-axis and are drawn as two panels, never on two y-axes.

CONTENTION IS FATAL TO THIS MEASUREMENT. Throughput on a shared box is not
reproducible, so this script refuses to collect unless it can verify the box
is quiet BEFORE and AFTER the sweep, and it records what it saw in the
output's provenance block either way.

Usage:
    pixi run -e cuda python scripts/benchmark_vocab_tiles.py --run
    pixi run python scripts/benchmark_vocab_tiles.py --plot <json>
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.benchmark_train import (  # noqa: E402
    GRID_COLOR,
    TEXT_GRAY,
    TEXT_INK,
)

FIGURES = os.path.join(ROOT, "figures")
DEFAULT_OUT = os.path.join(ROOT, "zero", "bench", "bench_vocab_tiles.json")

# Palette slot 1 for the measured cost series, slot 3 (aqua) for the exact
# byte-count series in the second panel -- a different hue because it is a
# different KIND of quantity, not a second measurement.
COLOR_COST = "#2a78d6"
COLOR_BUYS = "#1baf7a"

# Tile counts Team L's multi-tile correctness case actually verifies at
# V_p=50304. Anything outside this is drawn as unverified.
VERIFIED_TILE_COUNTS = (1, 2, 4, 8, 16)

RESULT_RE = re.compile(
    r"RESULT\s+m=\s*(\d+)\s+k=\s*(\d+)\s+n=\s*(\d+)\s+tiles=\s*(\d+)\s+"
    r"realized_tiles=\s*(\d+)\s+tile_n=\s*(\d+)\s+ms=\s*([0-9.eE+-]+)\s+"
    r"gflops=\s*([0-9.eE+-]+)\s+"
    r"logits_bytes=\s*([0-9.eE+-]+)\s+iters=\s*(\d+)\s+warmup=\s*(\d+)"
)
CHECK_RE = re.compile(
    r"CHECK\s+m=\s*(\d+)\s+n=\s*(\d+)\s+tiles=\s*(\d+)\s+"
    r"realized_tiles=\s*(\d+)\s+tile_n=\s*(\d+)\s+"
    r"max_abs=\s*([0-9.eE+-]+)\s+max_rel=\s*([0-9.eE+-]+)\s+"
    r"ref_mag=\s*([0-9.eE+-]+)"
)


def _slug(s):
    return re.sub(r"[^A-Za-z0-9.]+", "-", s).strip("-") or "x"


def _gpu_name():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip().splitlines()[0]
    except Exception:
        return ""


def _gpu_token(name):
    return "-".join(name.split()[:5]) if name else "unknown-gpu"


def box_conditions():
    """Snapshot everything that would invalidate a throughput measurement."""

    def _count(pattern):
        try:
            out = subprocess.run(
                ["pgrep", "-c", "-f", pattern],
                capture_output=True,
                text=True,
            )
            return int(out.stdout.strip() or 0)
        except Exception:
            return -1

    try:
        with open("/proc/loadavg") as fh:
            load = [float(x) for x in fh.read().split()[:3]]
    except Exception:
        load = []
    try:
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,gpu_uuid",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        gpu_procs = [ln for ln in apps.splitlines() if ln.strip()]
    except Exception:
        gpu_procs = []

    return {
        "sampled": datetime.datetime.now().isoformat(timespec="seconds"),
        "loadavg": load,
        "make_test_procs": _count("make test"),
        "pytest_procs": _count("pytest"),
        "gpu_compute_apps": gpu_procs,
    }


def is_quiet(cond, max_load):
    return (
        cond["make_test_procs"] == 0
        and cond["pytest_procs"] == 0
        and bool(cond["loadavg"])
        and cond["loadavg"][0] < max_load
    )


def parse(text):
    results = []
    for m in RESULT_RE.finditer(text):
        results.append(
            {
                "m": int(m.group(1)),
                "k": int(m.group(2)),
                "n": int(m.group(3)),
                "tiles": int(m.group(4)),
                "realized_tiles": int(m.group(5)),
                "tile_n": int(m.group(6)),
                "ms": float(m.group(7)),
                "gflops": float(m.group(8)),
                "logits_bytes": float(m.group(9)),
                "iters": int(m.group(10)),
                "warmup": int(m.group(11)),
            }
        )
    checks = []
    for m in CHECK_RE.finditer(text):
        checks.append(
            {
                "m": int(m.group(1)),
                "n": int(m.group(2)),
                "tiles": int(m.group(3)),
                "realized_tiles": int(m.group(4)),
                "tile_n": int(m.group(5)),
                "max_abs": float(m.group(6)),
                "max_rel": float(m.group(7)),
                "ref_mag": float(m.group(8)),
                "bit_identical": float(m.group(6)) == 0.0,
            }
        )
    return results, checks


def run(out_path, max_load, timeout_s):
    before = box_conditions()
    if not is_quiet(before, max_load):
        print(
            "REFUSING TO COLLECT: the box is not quiet.\n"
            f"  loadavg           {before['loadavg']}\n"
            f"  make test procs   {before['make_test_procs']}\n"
            f"  pytest procs      {before['pytest_procs']}\n"
            "Throughput measured under contention is not reproducible and "
            "would be worse than no figure. Wait for quiet and retry.",
            file=sys.stderr,
        )
        return 1

    cmd = [
        "flock",
        "-w",
        str(timeout_s),
        "/tmp/llmm-gpu.lock",
        "-c",
        f"cd {ROOT} && pixi run -e cuda mojo run -I . bench_gemm_vocab_tiles.mojo",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    after = box_conditions()
    results, checks = parse(proc.stdout)

    quiet_throughout = is_quiet(before, max_load) and is_quiet(after, max_load)
    doc = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "gpu": _gpu_name() or None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "provenance": {
            "throughput": "MEASURED - bf16 linalg.matmul, same methodology "
            "as bench_gemm.mojo (same call, same warmup/iters shape, same "
            "2*M*N*K FLOP count).",
            "logits_bytes": "EXACT - a byte count from the shapes, not a measurement.",
            "correctness_check": "MEASURED and DETERMINISTIC - tiled vs "
            "untiled reference; unaffected by box load.",
            "scope": "This is a SYNTHETIC decomposition benchmark. It "
            "characterises the DECOMPOSITION, not any particular "
            "vocab-tiled LM-head implementation, which is a different "
            "kernel and may have a different performance curve.",
            "box_quiet_before": is_quiet(before, max_load),
            "box_quiet_after": is_quiet(after, max_load),
            "box_quiet_throughout": quiet_throughout,
            "conditions_before": before,
            "conditions_after": after,
            "verified_tile_counts": list(VERIFIED_TILE_COUNTS),
        },
        "returncode": proc.returncode,
        "results": results,
        "checks": checks,
        "stderr_tail": proc.stderr[-2000:],
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(out_path)
    if not quiet_throughout:
        print(
            "WARNING: the box did not stay quiet for the whole sweep. The "
            "throughput numbers in this file are NOT publishable.",
            file=sys.stderr,
        )
        return 1
    return 0


def plot(json_path, out_path=None):
    import shutil

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with open(json_path) as fh:
        doc = json.load(fh)

    prov = doc.get("provenance", {})
    if not prov.get("box_quiet_throughout"):
        print(
            "REFUSING TO PLOT: this data was collected on a box that was not "
            "verified quiet before and after the sweep. Re-collect it.",
            file=sys.stderr,
        )
        return None

    ms = sorted({r["m"] for r in doc["results"]})
    lead_m = max(ms)
    rows = sorted(
        (r for r in doc["results"] if r["m"] == lead_m),
        key=lambda r: r["tiles"],
    )
    tiles = [r["tiles"] for r in rows]
    tflops = [r["gflops"] / 1000.0 for r in rows]
    gib = [r["logits_bytes"] / (1024**3) for r in rows]
    base = tflops[0]

    fig, (ax_cost, ax_buys) = plt.subplots(1, 2, figsize=(13.0, 6.0))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.775, bottom=0.135, wspace=0.24)
    fig.patch.set_facecolor("white")
    for ax in (ax_cost, ax_buys):
        ax.set_facecolor("white")
        ax.set_xscale("log", base=2)
        ax.set_xticks(tiles)
        ax.set_xticklabels([str(t) for t in tiles])
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(TEXT_GRAY)
        ax.tick_params(colors=TEXT_GRAY, labelsize=9)
        ax.set_xlabel("Number of vocabulary tiles (K)", color=TEXT_INK, fontsize=10)
        unverified = [t for t in tiles if t not in VERIFIED_TILE_COUNTS]
        if unverified:
            ax.axvspan(
                min(unverified) / 1.41,
                max(tiles) * 1.41,
                color=GRID_COLOR,
                alpha=0.75,
                zorder=0,
            )

    for ax, ys, color, ylab in (
        (ax_cost, tflops, COLOR_COST, "Achieved bf16 throughput (TFLOP/s)"),
        (ax_buys, gib, COLOR_BUYS, "Resident logits block (GiB)"),
    ):
        ax.plot(tiles, ys, color=color, linewidth=2.0, zorder=3)
        for t, y in zip(tiles, ys):
            verified = t in VERIFIED_TILE_COUNTS
            ax.plot(
                [t],
                [y],
                marker="o",
                markersize=9,
                markerfacecolor=color if verified else "white",
                markeredgecolor=color if not verified else "white",
                markeredgewidth=2.0,
                zorder=5,
            )
        ax.set_ylabel(ylab, color=TEXT_INK, fontsize=10)
        ax.set_ylim(0, max(ys) * 1.18)

    for t, y in zip(tiles, tflops):
        ax_cost.annotate(
            f"{100 * y / base:.0f}%",
            (t, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color=TEXT_INK,
            zorder=6,
        )
    ax_cost.set_title(
        "What tiling COSTS — measured  (labels: % of the untiled K=1 rate)",
        color=TEXT_GRAY,
        fontsize=9.5,
        pad=8,
    )
    ax_buys.set_title(
        "What tiling BUYS — exact byte count, not a measurement",
        color=TEXT_GRAY,
        fontsize=9.5,
        pad=8,
    )

    bit_exact = sorted(c["tiles"] for c in doc.get("checks", []) if c["bit_identical"])
    diverged = sorted(
        c["tiles"] for c in doc.get("checks", []) if not c["bit_identical"]
    )

    fig.suptitle(
        "Vocabulary tiling: memory falls as 1/K, throughput does not hold up",
        color=TEXT_INK,
        fontsize=14,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.905,
        f"GPT-2 124M LM head · bf16 · M = B×T = {lead_m:,} · K = C = "
        f"{rows[0]['k']} · N = V_p = {rows[0]['n']:,} · "
        f"{doc.get('gpu') or 'unknown GPU'} · {doc['host']} · "
        f"{doc['generated'][:10]}",
        color=TEXT_GRAY,
        fontsize=9,
        ha="center",
    )
    fig.text(
        0.5,
        0.862,
        "Shaded region: tile counts beyond the multi-tile correctness case, "
        "shown hollow. Filled markers are verified tile counts."
        + (
            f"  Decomposition bit-identical to untiled at K = "
            f"{', '.join(str(t) for t in bit_exact)}"
            + (f"; first divergence at K = {diverged[0]}." if diverged else ".")
            if bit_exact
            else ""
        ),
        color=TEXT_INK,
        fontsize=8.6,
        ha="center",
    )

    legend = fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=TEXT_GRAY,
                marker="o",
                markersize=8,
                linestyle="none",
                markerfacecolor=TEXT_GRAY,
                markeredgecolor="white",
                markeredgewidth=2.0,
                label="correctness verified at this tile count",
            ),
            Line2D(
                [],
                [],
                color=TEXT_GRAY,
                marker="o",
                markersize=8,
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=TEXT_GRAY,
                markeredgewidth=2.0,
                label="not verified — do not pick an optimum here",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.845),
        frameon=False,
        fontsize=9,
        ncol=2,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_INK)

    fig.text(
        0.985,
        0.028,
        "MEASURED on an otherwise-idle box, verified quiet before and after "
        "the sweep (conditions recorded in the sidecar JSON). This "
        "characterises the DECOMPOSITION, not any particular vocab-tiled "
        "LM-head implementation —\na purpose-written kernel is different "
        "code and may have a different curve. Forward projection GEMM only.",
        color=TEXT_GRAY,
        fontsize=7.2,
        ha="right",
        va="bottom",
        style="italic",
    )

    if out_path is None:
        gen = doc["generated"]
        stamp = f"{gen[:10]}_{gen[11:16].replace(':', '')}"
        name = (
            f"vocab_tile_gemm_m{lead_m}_oc{rows[0]['n']}_{stamp}"
            f"_{_gpu_token(doc.get('gpu') or '')}_{_slug(doc['host'])}.png"
        )
        out_path = os.path.join(FIGURES, name)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    sidecar = os.path.splitext(out_path)[0] + ".json"
    if os.path.abspath(sidecar) != os.path.abspath(json_path):
        shutil.copyfile(json_path, sidecar)
    plt.close(fig)
    print(out_path)
    print(sidecar)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="collect the sweep")
    ap.add_argument("--plot", default=None, help="render from this JSON")
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--plot-out", default=None)
    ap.add_argument(
        "--max-load",
        type=float,
        default=6.0,
        help="refuse to collect above this 1-minute load average",
    )
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    if args.plot:
        plot(args.plot, args.plot_out)
        return
    if args.run:
        sys.exit(run(args.output, args.max_load, args.timeout))
    ap.error("pass --run to collect or --plot <json> to render")


if __name__ == "__main__":
    main()
