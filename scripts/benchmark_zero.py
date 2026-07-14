#!/usr/bin/env python3
"""Collect and chart per-ZeRO-stage training benchmark data at a fixed world size.

Run side: drives the training binary once per (precision, stage), parses
per-step timing/loss from stdout, samples per-GPU memory via ``nvidia-smi``
during the run, and writes a machine-readable JSON.

Plot side (``--plot bench.json``): renders the README chart from a previously
collected JSON — peak per-GPU memory and mean step time per stage, fp32 vs
bf16 — into a dated, hardware-stamped PNG under ``figures/`` (with the source
JSON copied alongside), matching scripts/benchmark_train.py's conventions.

ZeRO's headline number is the memory-per-stage curve: stage 1 shards the
optimizer state, stage 2 also shards gradients, stage 3 also shards parameters,
so peak per-GPU memory should fall as the stage rises. Identical ``-b/-t/-d``
flags are used across stages so the numbers are comparable; those flags are
recorded in the JSON.

Example:
    python scripts/benchmark_zero.py \
        --world-size 8 --stages 0,1,2,3 -b 4 -t 64 --steps 12 \
        --fp32-binary build/train_gpt2 \
        --output bench_zero_world8.json

Each (precision, stage) entry records: stage, precision, world_size, the b/t/d
flags, mean_step_ms (excluding the first 2 warmup steps), tokens_per_sec,
peak per-GPU memory (MiB, max sampled during the run), status (ok/crashed/…),
and the per-step losses actually observed. Stages that crash are recorded with
status="crashed" and the tail of stderr, so the JSON is an honest snapshot even
when a stage is broken.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import threading
import time

# "step 3/12 | loss 5.213456 (…z)| norm … | 41.83 ms | … tok/s"
_STEP_RE = re.compile(
    r"step\s+(\d+)/\d+\s*\|\s*loss\s+([-\d.]+).*?\|\s*([\d.]+)\s*ms\s*\|"
)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _query_gpu_mem_mib():
    """Return a list of used-memory MiB per visible GPU, or [] on failure."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    vals = []
    for line in out.strip().splitlines():
        line = line.strip()
        if line:
            try:
                vals.append(int(line))
            except ValueError:
                pass
    return vals


class _MemSampler(threading.Thread):
    """Poll per-GPU used memory until stopped; keep the max seen per GPU."""

    def __init__(self, interval_s: float = 0.25):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._stop_evt = threading.Event()
        self.peak_per_gpu = {}

    def run(self):
        while not self._stop_evt.is_set():
            for idx, mib in enumerate(_query_gpu_mem_mib()):
                if mib > self.peak_per_gpu.get(idx, 0):
                    self.peak_per_gpu[idx] = mib
            self._stop_evt.wait(self.interval_s)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=5.0)


def _run_stage(binary, world_size, stage, b, t, d, steps, timeout_s, extra):
    root = _repo_root()
    env = dict(os.environ)
    env["WORLD_SIZE"] = str(world_size)
    # run_train_gpt2.sh execs build/train_gpt2; allow a non-default binary via
    # the BIN override the bf16 runner understands, else call the binary direct.
    cmd = [
        binary,
        "-e",
        extra["load"],
        "-i",
        extra["train_data"],
        "-j",
        extra["val_data"],
        "-b",
        str(b),
        "-t",
        str(t),
        "-x",
        str(steps),
        "-z",
        str(stage),
        "-pn",
        str(world_size),
        "-v",
        "1000",
        "-s",
        "0",
    ]
    if d is not None:
        cmd += ["-d", str(d)]
    # libpython wiring (mirrors run_train_gpt2.sh) so the binary can be called
    # directly with a per-precision path.
    if "MOJO_PYTHON_LIBRARY" not in env:
        import glob

        for pat in ("libpython3*.so", "libpython3*.dylib"):
            hits = glob.glob(os.path.join(root, ".pixi/envs/default/lib", pat))
            if hits:
                env["MOJO_PYTHON_LIBRARY"] = hits[0]
                break

    # Per-GPU baseline BEFORE launch, so another team's jobs (e.g. pinned to
    # GPUs 4-7 on this shared box) are subtracted out and the reported delta is
    # this process's own contribution.
    baseline = _query_gpu_mem_mib()
    sampler = _MemSampler()
    sampler.start()
    t0 = time.time()
    status = "ok"
    stderr_tail = ""
    stdout = ""
    try:
        proc = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        stdout = proc.stdout
        if proc.returncode != 0:
            status = "crashed"
            # The Mojo runtime prints "Unhandled exception …" to stdout, so fall
            # back to the stdout tail when stderr is empty.
            tail_src = (proc.stderr or "").strip() or (stdout or "").strip()
            stderr_tail = "\n".join(tail_src.splitlines()[-8:])
    except subprocess.TimeoutExpired as e:
        status = "timeout"
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    finally:
        wall = time.time() - t0
        sampler.stop()

    steps_seen = []
    for m in _STEP_RE.finditer(stdout):
        steps_seen.append(
            {
                "step": int(m.group(1)),
                "loss": float(m.group(2)),
                "ms": float(m.group(3)),
            }
        )

    # Mean step time excluding the first 2 warmup steps.
    timed = [s["ms"] for s in steps_seen[2:]]
    mean_step_ms = sum(timed) / len(timed) if timed else None
    tokens_per_step = world_size * b * t
    tokens_per_sec = tokens_per_step / (mean_step_ms / 1000.0) if mean_step_ms else None
    if status == "ok" and not steps_seen:
        status = "no-steps"

    peak = sampler.peak_per_gpu
    # Delta over the pre-launch baseline, floored at 0, so co-tenant jobs on
    # other GPUs don't inflate the number. This is the headline ZeRO metric.
    delta = {}
    for idx, mib in peak.items():
        base = baseline[idx] if idx < len(baseline) else 0
        delta[idx] = max(0, mib - base)
    touched = {k: v for k, v in delta.items() if v > 0}
    return {
        "stage": stage,
        "precision": extra["precision"],
        "world_size": world_size,
        "flags": {"b": b, "t": t, "d": d, "steps": steps},
        "status": status,
        "mean_step_ms": mean_step_ms,
        "tokens_per_sec": tokens_per_sec,
        "tokens_per_step": tokens_per_step,
        "peak_mem_mib_per_gpu_raw": {str(k): v for k, v in sorted(peak.items())},
        "peak_mem_mib_per_gpu_delta": {str(k): v for k, v in sorted(delta.items())},
        "peak_mem_mib_max_delta": max(delta.values()) if delta else None,
        "num_gpus_touched": len(touched),
        "losses": [s["loss"] for s in steps_seen],
        "wall_s": round(wall, 1),
        "stderr_tail": stderr_tail,
    }


# Chart style: mirrors scripts/benchmark_train.py's house rules (fixed llm.mojo
# hue; precision encoded via hatching over the same hue, never by color; text
# and scaffolding in neutral inks, never a series hue; value label on each bar).
LLM_MOJO_BLUE = "#2a78d6"
TEXT_INK = "#1a1a19"
TEXT_GRAY = "#5f5e56"
GRID_COLOR = "#e5e4dd"
LEGEND_FILL = "#cccccc"

_STAGE_LABELS = {
    0: "Stage 0\n(DDP)",
    1: "Stage 1\n(+opt shard)",
    2: "Stage 2\n(+grad shard)",
    3: "Stage 3\n(+param shard)",
}


def _gpu_name_token():
    """Short sanitized GPU product token for the figure filename."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        name = out.strip().splitlines()[0]
        # "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition" is too
        # long for a filename; the first five words identify the product.
        return "-".join(name.split()[:5])
    except Exception:
        return "unknown-gpu"


def _grouped_bars(ax, stages, fp32_vals, bf16_vals, fmt):
    """One panel of paired fp32/bf16 bars with per-bar value labels."""
    width = 0.38
    xs = list(range(len(stages)))
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for offset, vals, precision in (
        (-width / 2, fp32_vals, "fp32"),
        (+width / 2, bf16_vals, "bf16"),
    ):
        for x, v in zip(xs, vals):
            if v is None:
                continue
            bar = ax.bar(
                x + offset,
                v,
                width=width,
                color=LLM_MOJO_BLUE,
                edgecolor=TEXT_INK,
                linewidth=1.2,
                zorder=3,
            )[0]
            if precision == "bf16":
                # White hatch + ink outline needs two artists (hatch lines
                # draw in the edgecolor) — same gotcha as benchmark_train.py.
                bar.set_hatch("///")
                bar.set_edgecolor("white")
                bar.set_linewidth(0)
                ax.bar(
                    x + offset,
                    v,
                    width=width,
                    fill=False,
                    edgecolor=TEXT_INK,
                    linewidth=1.2,
                    zorder=4,
                )
            ax.annotate(
                fmt(v),
                (x + offset, v),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=TEXT_INK,
                zorder=5,
            )

    top = max(v for v in fp32_vals + bf16_vals if v is not None)
    ax.set_ylim(0, top * 1.14)
    ax.set_xticks(xs)
    ax.set_xticklabels([_STAGE_LABELS.get(s, f"Stage {s}") for s in stages])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_GRAY)
    ax.tick_params(colors=TEXT_GRAY, labelsize=9)
    for lbl in ax.get_xticklabels():
        lbl.set_color(TEXT_INK)


def plot(json_path, out_path=None):
    import shutil

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    root = _repo_root()
    with open(json_path) as f:
        doc = json.load(f)

    ok = [r for r in doc["results"] if r["status"] == "ok"]
    skipped = [
        (r["precision"], r["stage"]) for r in doc["results"] if r["status"] != "ok"
    ]
    stages = sorted({r["stage"] for r in ok})
    if not stages:
        raise SystemExit(f"{json_path}: no successful stage runs to plot")

    def series(precision, field):
        by_stage = {r["stage"]: r for r in ok if r["precision"] == precision}
        vals = []
        for s in stages:
            r = by_stage.get(s)
            if r is None:
                vals.append(None)
            elif field == "mem":
                vals.append(max(r["peak_mem_mib_per_gpu_delta"].values()))
            else:
                vals.append(r[field])
        return vals

    fig, (ax_mem, ax_ms) = plt.subplots(1, 2, figsize=(11.5, 5.4), layout="constrained")
    fig.patch.set_facecolor("white")
    for ax in (ax_mem, ax_ms):
        ax.set_facecolor("white")

    _grouped_bars(
        ax_mem,
        stages,
        series("fp32", "mem"),
        series("bf16", "mem"),
        fmt=lambda v: f"{v:,.0f}",
    )
    ax_mem.set_title(
        "Peak GPU memory (MiB per GPU, above idle)", color=TEXT_INK, fontsize=11
    )

    _grouped_bars(
        ax_ms,
        stages,
        series("fp32", "mean_step_ms"),
        series("bf16", "mean_step_ms"),
        fmt=lambda v: f"{v:.1f}",
    )
    ax_ms.set_title("Mean step time (ms)", color=TEXT_INK, fontsize=11)

    flags = doc["flags"]
    world = doc["world_size"]
    date = doc["generated"][:10]
    # One rank per GPU: name what the ranks ran on, not the box's GPU count.
    gpu_product = _gpu_name_token().replace("-", " ")
    subtitle = (
        f"{world}× {gpu_product} · B={flags['b']} T={flags['t']} per rank"
        f" · {flags['steps']} steps (first 2 excluded) · {date}"
    )
    if skipped:
        subtitle += "  ·  not run: " + ", ".join(f"{p} z{s}" for p, s in skipped)
    fig.suptitle(
        f"ZeRO stages — GPT-2 124M training, WORLD_SIZE={world}",
        color=TEXT_INK,
        fontsize=13,
        fontweight="semibold",
    )
    ax_mem.text(
        0.0,
        1.10,
        subtitle,
        transform=ax_mem.transAxes,
        color=TEXT_GRAY,
        fontsize=9.5,
        ha="left",
    )
    fig.legend(
        handles=[
            Patch(
                facecolor=LEGEND_FILL, edgecolor=TEXT_INK, linewidth=1.2, label="fp32"
            ),
            Patch(
                facecolor=LEGEND_FILL,
                edgecolor=TEXT_INK,
                linewidth=1.2,
                hatch="///",
                label="bf16",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(0.99, 1.0),
        frameon=False,
        fontsize=9.5,
    )

    if out_path is None:
        hhmm = doc["generated"][11:16].replace(":", "")
        stem = (
            f"benchmark_zero_w{world}_b{flags['b']}_t{flags['t']}"
            f"_{date}_{hhmm}_{_gpu_name_token()}_{doc['host']}"
        )
        out_path = os.path.join(root, "figures", f"{stem}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    json_copy = os.path.splitext(out_path)[0] + ".json"
    if os.path.abspath(json_copy) != os.path.abspath(json_path):
        shutil.copyfile(json_path, json_copy)
    print(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--plot",
        metavar="JSON",
        help="skip running; render the chart from this benchmark JSON",
    )
    ap.add_argument(
        "--plot-out",
        default=None,
        help="output PNG path for --plot (default: auto under figures/)",
    )
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--stages", default="0,1,2,3")
    ap.add_argument("-b", type=int, default=4)
    ap.add_argument("-t", type=int, default=64)
    ap.add_argument("-d", type=int, default=None, help="total batch size")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=400)
    ap.add_argument(
        "--fp32-binary",
        default="build/train_gpt2",
        help="WORLD_SIZE-built fp32 binary, or '' to skip fp32",
    )
    ap.add_argument(
        "--bf16-binary",
        default="build/train_gpt2_bf16",
        help="WORLD_SIZE-built bf16 binary, or '' to skip bf16",
    )
    ap.add_argument("--load-fp32", default="gpt2_124M.bin")
    ap.add_argument("--load-bf16", default="gpt2_124M_bf16.bin")
    ap.add_argument(
        "--train-data",
        default="./data/.tinyshakespeare/tiny_shakespeare_train.bin",
    )
    ap.add_argument(
        "--val-data",
        default="./data/.tinyshakespeare/tiny_shakespeare_val.bin",
    )
    ap.add_argument("--output", default="bench_zero_world8.json")
    args = ap.parse_args()

    if args.plot:
        plot(args.plot, args.plot_out)
        return

    root = _repo_root()
    stages = [int(s) for s in args.stages.split(",") if s.strip() != ""]

    precisions = []
    if args.fp32_binary:
        precisions.append(("fp32", args.fp32_binary, args.load_fp32))
    if args.bf16_binary and os.path.exists(os.path.join(root, args.bf16_binary)):
        precisions.append(("bf16", args.bf16_binary, args.load_bf16))

    results = []
    for precision, binary, load in precisions:
        bin_abs = os.path.join(root, binary)
        if not os.path.exists(bin_abs):
            print(f"skip {precision}: {binary} not found", file=sys.stderr)
            continue
        for stage in stages:
            print(
                f"== {precision} world={args.world_size} stage={stage} ==",
                file=sys.stderr,
            )
            entry = _run_stage(
                bin_abs,
                args.world_size,
                stage,
                args.b,
                args.t,
                args.d,
                args.steps,
                args.timeout,
                {
                    "precision": precision,
                    "load": load,
                    "train_data": args.train_data,
                    "val_data": args.val_data,
                },
            )
            print(
                f"   status={entry['status']} "
                f"mean_step_ms={entry['mean_step_ms']} "
                f"peak_mem_delta={entry['peak_mem_mib_max_delta']}",
                file=sys.stderr,
            )
            results.append(entry)

    host = os.uname().nodename
    doc = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": host,
        "gpu_count": len(_query_gpu_mem_mib()),
        "world_size": args.world_size,
        "flags": {"b": args.b, "t": args.t, "d": args.d, "steps": args.steps},
        "note": (
            "Run-side data only (no plotting). mean_step_ms excludes the first"
            " 2 warmup steps. peak_mem_mib_* sampled via nvidia-smi during the"
            " run. Stages with status!='ok' did not train."
        ),
        "results": results,
    }
    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(root, out_path)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
