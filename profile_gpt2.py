#!/usr/bin/env python3
"""Profile the llm.mojo GPT-2 training step with NVIDIA Nsight Compute (ncu).

This is the llm.mojo analogue of llm.c's profile_gpt2cu.py. It runs `ncu` over
the build/profile_gpt2 harness (one forward / backward / update iteration),
parses the per-kernel metrics, buckets kernels into operation families and — for
kernels whose names carry it — forward/backward/optimizer phases, then prints a
per-kernel table plus family and phase summaries.

Typical use (via the Makefile):

    make profile-ncu

Direct use:

    # capture + analyze in one shot
    pixi run -e cuda python profile_gpt2.py --exe build/profile_gpt2

    # re-analyze a capture without re-running ncu
    python profile_gpt2.py --input profile_ncu.csv

Notes:
  * DRAM/tensor metrics require GPU performance-counter access. Without it ncu
    reports "n/a" for those metrics (timing still works); pass --sudo to run ncu
    with elevated privileges, or analyze a capture that already has them.
  * MAX lowers the GEMMs to cutlass kernels whose names do not encode forward vs
    backward, so those land in the "matmul" family and the "unattributed" phase.
"""

import argparse
import csv
import io
import os
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field

# ncu metrics we collect. Mirrors the core set llm.c's profile script uses:
# wall time, DRAM read/write volume, and tensor-core utilization.
DEFAULT_METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
]

METRIC_TIME = "gpu__time_duration.sum"
METRIC_DRAM_R = "dram__bytes_read.sum"
METRIC_DRAM_W = "dram__bytes_write.sum"
METRIC_TENSOR = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"

# Operation family classification, by substring of the (mangled) kernel name.
# Order matters — first match wins. Tuned to the names MAX emits for llmm's
# kernels (e.g. "llmm_attention_attention_bwd_...") and to cutlass GEMMs.
FAMILY_RULES = [
    ("attention", ("attention",)),
    ("matmul", ("cutlass", "gemm", "matmul")),
    ("layernorm", ("layernorm",)),
    ("encoder", ("encoder", "wte", "wpe")),
    ("softmax", ("softmax",)),
    ("gelu", ("gelu",)),
    ("classifier", ("fused_classifier", "fused_cl", "crossentropy", "classifier")),
    ("optimizer", ("adamw", "global_norm")),
    ("split/merge", ("split", "merge")),
    ("elementwise", ("elementwise",)),
]


@dataclass
class KernelInstance:
    """One kernel launch parsed from the ncu CSV (metrics keyed by name)."""

    name: str
    grid: str
    block: str
    metrics: dict = field(default_factory=dict)


@dataclass
class KernelAgg:
    """All launches of a kernel name, summed."""

    name: str
    family: str
    calls: int = 0
    time_ns: float = 0.0
    dram_r: float = 0.0
    dram_w: float = 0.0
    tensor_sum: float = 0.0
    tensor_n: int = 0
    phase: str = ""


def classify_family(name):
    low = name.lower()
    for family, needles in FAMILY_RULES:
        if any(n in low for n in needles):
            return family
    return "other"


def classify_phase(name, family):
    """Best-effort training-phase bucket. Only names that encode it are split."""
    low = name.lower()
    if family == "optimizer":
        return "optimizer"
    if family == "classifier":
        return "classifier"
    if "_bwd" in low or "backward" in low:
        return "backward"
    if "_fwd" in low or "forward" in low:
        return "forward"
    return "unattributed"


def find_ncu(explicit):
    if explicit:
        return explicit
    found = shutil.which("ncu")
    if found:
        return found
    for cand in ("/usr/local/cuda/bin/ncu", "/opt/nvidia/nsight-compute/ncu"):
        if os.path.exists(cand):
            return cand
    return None


def run_ncu(ncu, exe, exe_args, metrics, use_sudo, extra_args, cwd=None):
    """Run ncu over `exe exe_args...` and return its CSV output as text."""
    cmd = []
    if use_sudo:
        cmd += ["sudo"]
    cmd += [
        ncu,
        "--csv",
        "--target-processes",
        "all",
        "--metrics",
        ",".join(metrics),
    ]
    cmd += extra_args
    cmd += [exe] + list(exe_args)
    print("[profile] running:", " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        if "ERR_NVGPUCTRPERM" in (proc.stdout + proc.stderr):
            sys.stderr.write(
                "\n[profile] ncu lacks GPU performance-counter access. "
                "Re-run with --sudo, or grant access:\n"
                "  https://developer.nvidia.com/ERR_NVGPUCTRPERM\n"
            )
        raise SystemExit(proc.returncode)
    # ncu prints the profiled program's stdout interleaved; the CSV starts at the
    # quoted "ID" header line.
    return proc.stdout


def parse_csv(text):
    """Parse ncu --csv long-format output into per-kernel-instance records."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            start = i
            break
    if start is None:
        raise SystemExit("no ncu CSV table found in input")

    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    # Each kernel launch (one "ID") spans several rows, one per metric.
    kernels: dict[str, KernelInstance] = {}
    for row in reader:
        kid = row["ID"]
        k = kernels.get(kid)
        if k is None:
            k = KernelInstance(
                name=row["Kernel Name"],
                grid=row.get("Grid Size", ""),
                block=row.get("Block Size", ""),
            )
            kernels[kid] = k
        k.metrics[row["Metric Name"]] = _num(row["Metric Value"])
    return list(kernels.values())


def _num(raw):
    """Parse an ncu metric value ("92,288", "n/a", "") into a float or None."""
    if raw is None:
        return None
    raw = raw.strip().replace(",", "")
    if raw == "" or raw.lower() == "n/a":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def aggregate(kernels):
    """Aggregate kernel instances by name."""
    agg: dict[str, KernelAgg] = {}
    for k in kernels:
        name = k.name
        m = k.metrics
        a = agg.get(name)
        if a is None:
            a = KernelAgg(name=name, family=classify_family(name))
            agg[name] = a
        a.calls += 1
        a.time_ns += m.get(METRIC_TIME) or 0.0
        a.dram_r += m.get(METRIC_DRAM_R) or 0.0
        a.dram_w += m.get(METRIC_DRAM_W) or 0.0
        t = m.get(METRIC_TENSOR)
        if t is not None:
            a.tensor_sum += t
            a.tensor_n += 1
    for a in agg.values():
        a.phase = classify_phase(a.name, a.family)
    return sorted(agg.values(), key=lambda a: a.time_ns, reverse=True)


def short_name(name, width=44):
    """Trim the mangled hash suffix and clip to width for display."""
    n = name
    if n.startswith("void "):
        n = n[5:]
    # drop a trailing "_<8+ hex>" content hash if present
    if "_" in n:
        head, _, tail = n.rpartition("_")
        if len(tail) >= 8 and all(c in "0123456789abcdef" for c in tail.lower()):
            n = head
    if len(n) > width:
        n = n[: width - 1] + "…"
    return n


def fmt_us(ns):
    return f"{ns / 1e3:10.2f}"


def fmt_gbps(bytes_total, time_ns):
    if not bytes_total or not time_ns:
        return "       -"
    gbps = (bytes_total / 1e9) / (time_ns / 1e9)
    return f"{gbps:8.1f}"


def print_table(kernels, total_ns):
    print()
    print("Per-kernel breakdown (sorted by total time):")
    print("-" * 118)
    print(
        f"{'kernel':46} {'family':12} {'calls':>5} {'time(us)':>11} "
        f"{'%':>6} {'DRAM r GB/s':>11} {'DRAM w GB/s':>11} {'tensor%':>8}"
    )
    print("-" * 118)
    for a in kernels:
        pct = 100.0 * a.time_ns / total_ns if total_ns else 0.0
        tensor = f"{a.tensor_sum / a.tensor_n:7.1f}" if a.tensor_n else "      -"
        print(
            f"{short_name(a.name):46} {a.family:12} {a.calls:>5} "
            f"{fmt_us(a.time_ns)} {pct:6.1f} "
            f"{fmt_gbps(a.dram_r, a.time_ns)}   "
            f"{fmt_gbps(a.dram_w, a.time_ns)}   {tensor}"
        )
    print("-" * 118)


def print_group(title, key, kernels, total_ns):
    groups = defaultdict(lambda: {"time_ns": 0.0, "calls": 0})
    for a in kernels:
        g = groups[getattr(a, key)]
        g["time_ns"] += a.time_ns
        g["calls"] += a.calls
    print()
    print(f"{title}:")
    print("-" * 52)
    print(f"{key:16} {'calls':>6} {'time(us)':>14} {'%':>8}")
    print("-" * 52)
    for name, g in sorted(
        groups.items(), key=lambda kv: kv[1]["time_ns"], reverse=True
    ):
        pct = 100.0 * g["time_ns"] / total_ns if total_ns else 0.0
        print(f"{name:16} {g['calls']:>6} {g['time_ns'] / 1e3:>14.2f} {pct:>8.1f}")
    print("-" * 52)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--exe", default="build/profile_gpt2", help="profile harness binary"
    )
    ap.add_argument("--target", default="gpu", help="harness target arg (cpu/gpu)")
    ap.add_argument(
        "--exe-args",
        default=None,
        help="full argument string for --exe (overrides --target); use this to "
        "profile a different binary, e.g. llm.c's train_gpt2cu",
    )
    ap.add_argument("--cwd", default=None, help="working directory to run --exe in")
    ap.add_argument("--ncu", default=None, help="path to ncu (default: auto-detect)")
    ap.add_argument(
        "--input",
        default=None,
        help="parse an existing ncu --csv capture instead of running ncu",
    )
    ap.add_argument("--output", default=None, help="save the raw ncu CSV here")
    ap.add_argument(
        "--metrics",
        default=None,
        help="comma-separated ncu metrics (default: time, DRAM r/w, tensor%%)",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="pass `--set full` to ncu (many metrics, much slower)",
    )
    ap.add_argument(
        "--sudo",
        action="store_true",
        help="run ncu under sudo (for perf-counter access)",
    )
    ap.add_argument(
        "--launch-count",
        type=int,
        default=None,
        help="profile only the first N kernel launches (ncu --launch-count). "
        "Use for harnesses with no step-limit flag (e.g. llm.c's fp32 CUDA "
        "build runs a full epoch) so ncu doesn't replay hundreds of steps.",
    )
    args = ap.parse_args()

    metrics = args.metrics.split(",") if args.metrics else DEFAULT_METRICS

    if args.input:
        with open(args.input) as f:
            text = f.read()
    else:
        ncu = find_ncu(args.ncu)
        if not ncu:
            raise SystemExit("ncu not found; install Nsight Compute or pass --ncu PATH")
        if not os.path.exists(args.exe):
            raise SystemExit(f"{args.exe} not found; run `make build-profile` first")
        extra = ["--set", "full"] if args.full else []
        if args.launch_count is not None:
            extra += ["--launch-count", str(args.launch_count)]
        exe_args = shlex.split(args.exe_args) if args.exe_args else [args.target]
        text = run_ncu(ncu, args.exe, exe_args, metrics, args.sudo, extra, cwd=args.cwd)
        if args.output:
            with open(args.output, "w") as f:
                f.write(text)
            print(f"[profile] raw ncu CSV saved to {args.output}", file=sys.stderr)

    kernels = aggregate(parse_csv(text))
    total_ns = sum(a.time_ns for a in kernels)

    print_table(kernels, total_ns)
    print_group("By operation family", "family", kernels, total_ns)
    print_group(
        "By training phase (name-attributable only)", "phase", kernels, total_ns
    )

    print()
    n_calls = sum(a.calls for a in kernels)
    print(
        f"Total: {len(kernels)} distinct kernels, {n_calls} launches, "
        f"{total_ns / 1e3:.2f} us on GPU"
    )
    if not any(a.dram_r or a.dram_w for a in kernels):
        print(
            "(DRAM/tensor columns blank: ncu had no perf-counter access — see --sudo)"
        )


if __name__ == "__main__":
    main()
