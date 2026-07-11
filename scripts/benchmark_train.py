#!/usr/bin/env python3
"""Comparative training-loop benchmark: llm.mojo vs llm.c vs PyTorch.

Runs each implementation/configuration for many training steps, collects the
per-step train-loop time (forward + backward + optimizer update), and renders a
bar chart of the mean per-step time (ms on the Y axis; model + configuration on
the X axis). CPU and GPU are rendered as *separate* figures so each can be read
on its own and so this runs anywhere.

Configurations
  CPU  (fp32 throughout — llm.mojo CPU training is fp32 by policy):
    llm.mojo (fp32) | llm.c (OpenMP) | llm.c (1 thread) | PyTorch (fp32)
  GPU (NVIDIA, requires CUDA):
    llm.mojo (fp32) | llm.mojo (bf16) | llm.c CUDA (fp32) | llm.c CUDA (bf16)
    | PyTorch (fp32) | PyTorch (bf16)
  Metal (Apple Silicon, fp32 + bf16):
    llm.mojo fp32 (build/profile_gpt2 gpu) | llm.mojo bf16 (build/profile_gpt2_bf16 gpu)
    | PyTorch MPS fp32 | PyTorch MPS bf16
    NOTE: llm.c has no Metal port — the baseline is PyTorch on MPS, not llm.c.
    A --cooldown-s pause (default 30 s) runs between arms to prevent M4 Max thermal
    throttling from skewing later arms (see P16 in metal_port_gotchas_and_optimizations.md).

Device selection:
  --device cpu    only the CPU figure (works on macOS / any box, no CUDA)
  --device gpu    only the NVIDIA GPU figure (requires CUDA + the binaries)
  --device metal  Apple Silicon Metal GPU figure (llm.mojo + PyTorch MPS)
  --device auto   CPU always; on Apple Silicon: Metal; on NVIDIA: GPU (default)

Prereqs (built by the Makefile): build/profile_gpt2 (llm.mojo fp32 harness),
build/profile_gpt2_bf16 (llm.mojo bf16 harness, used for both GPU/CUDA and Metal
modes), the llm.c binaries third_party/llm.c/{train_gpt2, train_gpt2cu,
train_gpt2fp32cu} (CPU/GPU modes only), and PyTorch in the pixi env. The llm.c
inputs are staged into build/llmc/ with their magic numbers patched to the values
llm.c expects (the byte layout is otherwise identical). The PyTorch reference is
llm.c's train_gpt2.py run with random d12 (=GPT-2 124M) weights for CUDA mode, or
the repo-local train_gpt2.py with --device mps for Metal mode (fp32 and bf16).
"""

import argparse
import datetime
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import textwrap
import time
import matplotlib
import matplotlib.pyplot as plt

# Legend: precision hatch key.
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLMC = os.path.join(ROOT, "third_party", "llm.c")
TORCH_REF = os.path.join(LLMC, "train_gpt2.py")
# Local train_gpt2.py (used for the PyTorch MPS reference in Metal mode — it
# already has --device mps support with proper torch.mps.synchronize() timing).
TORCH_TRAIN_PY = os.path.join(ROOT, "train_gpt2.py")
STAGE = os.path.join(ROOT, "build", "llmc")
DATA = os.path.join(ROOT, "data", ".tinyshakespeare")
FIGURES = os.path.join(ROOT, "figures")

MOJO_FP32_BIN = os.path.join(ROOT, "build", "profile_gpt2")
MOJO_BF16_BIN = os.path.join(ROOT, "build", "profile_gpt2_bf16")

# Apples-to-apples config: identical hyperparameters everywhere so the bars are
# directly comparable. Set from --batch-size / --seq-len in main(); every config
# (Mojo harness env, llm.c CUDA -b/-t, PyTorch flags, and — via LLMC_B/LLMC_T —
# the llm.c CPU reference) is fed the same pair. The defaults (B=4, T=64) match
# what llm.c's CPU reference historically hardcoded. For Metal mode the default
# is B=4, T=1024 (the real training config, ~6.5 s/step) but can be overridden
# with --batch-size / --seq-len just like the other modes.
B, T = 4, 64
WARMUP = 5  # leading steps to drop (allocation / first-touch / clock spin-up)
# Default step counts per mode. Metal fp32 at B=4 T=1024 is ~6.5 s/step, so a
# small default (10 measured steps after warmup = ~70 s) is appropriate. Override
# on the command line with --metal-steps N.
DEFAULT_METAL_STEPS = 10

# llm.c expects different magic numbers than llm.mojo writes; the byte layout is
# otherwise the same, so we patch a copy. (model, tokenizer.)
MAGIC_MODEL = 20240326
MAGIC_TOKENIZER = 20240328

# Per-implementation-family colors: a FIXED assignment (never cycled, never by
# position) so the same implementation always reads as the same hue across
# every chart variant (e.g. PyTorch is #eda100 on the Metal chart and the CUDA
# chart alike, even when llm.c isn't present). Precision (fp32 vs bf16) is
# encoded separately via hatching over the same hue — never by color.
FAMILY_COLORS = {
    "llm.mojo": "#2a78d6",  # blue
    "llm.c": "#1baf7a",  # aqua
    "PyTorch": "#eda100",  # yellow (below 3:1 contrast on white — acceptable
    # only because every bar carries a direct value label; keep those labels).
}
FAMILY_COLOR_FALLBACK = "#008300"  # green, for a hypothetical 4th implementation

# Ink / neutral palette — text and chart scaffolding never borrow a series hue.
TEXT_INK = "#1a1a19"  # titles, value labels, bar edges
TEXT_GRAY = "#5f5e56"  # subtitle, spines, ticks, error bars
GRID_COLOR = "#e5e4dd"  # recessive horizontal gridlines, behind the bars
LEGEND_FILL = "#cccccc"  # neutral swatch fill for the fp32/bf16 precision key

# Known accelerator chip -> human-meaningful platform/product name. Used to
# replace the machine hostname in filenames and figure text (e.g. any box
# with an NVIDIA GB10 reports as "DGX Spark", regardless of its hostname).
KNOWN_ACCELERATORS = {
    "GB10": "DGX Spark",
}


def libpython():
    import glob

    for pat in ("libpython3*.so", "libpython3*.dylib"):
        hits = glob.glob(os.path.join(ROOT, ".pixi/envs/default/lib", pat))
        if hits:
            return hits[0]
    return ""


def is_apple_silicon():
    """True on macOS aarch64 (M1/M2/M3/M4 etc.)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def apple_chip_name():
    """Return the Apple chip name, e.g. 'Apple M4 Max'.

    On Apple Silicon, ``machdep.cpu.brand_string`` is the most reliable sysctl
    key — it returns the full chip name that the CPU information panel shows.
    Falls back to system_profiler for completeness.
    """
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        for line in out.splitlines():
            if "Chip:" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _ordinal(day):
    # 1->1st, 2->2nd, 3->3rd, 4->4th ... 11/12/13 are always "th".
    if 11 <= day % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def hardware_info():
    now = datetime.datetime.now()
    # On Apple Silicon there is no NVIDIA GPU; we populate "gpu" with the Apple
    # chip name so the filename slug and figure subtitle are self-describing
    # (e.g. "Apple M4 Max" → slug "Apple-M4-Max") and have_gpu() stays False.
    if is_apple_silicon():
        gpu_field = apple_chip_name()
    else:
        gpu_field = gpu_name() or "none"

    info = {
        # `date` keeps the timestamp for unique output filenames; `date_display`
        # is the human date shown on the figure itself (e.g. "July 31st, 2026").
        "date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date_display": f"{now.strftime('%B')} {_ordinal(now.day)}, {now.year}",
        # NOTE: `host` (the machine's actual hostname) is kept only for the
        # console summary — it must never appear in a filename or in the figure
        # itself. Use `platform` (derived below) for anything user-facing.
        "host": platform.node(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cores": str(os.cpu_count()),
        "gpu": gpu_field,
    }
    try:
        if sys.platform == "darwin":
            info["cpu"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        elif sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    # Human-meaningful platform/silicon description, computed last so it can
    # see the final "cpu"/"gpu" fields (e.g. the CPU sysctl override above).
    info["platform"] = platform_description(info)
    return info


def gpu_name():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        return out.splitlines()[0] if out else ""
    except Exception:
        return ""


def have_gpu():
    return bool(gpu_name())


def platform_description(info):
    """Human-meaningful platform/silicon description, used everywhere a
    machine hostname would otherwise leak into a filename or figure (e.g.
    "DGX Spark" for an NVIDIA GB10 box, "Mac M4 Max" for Apple Silicon).

    Never returns socket.gethostname()/platform.node() — falls back to
    "<os>-<arch>" (e.g. "linux-aarch64") when the silicon isn't recognized.
    """
    if is_apple_silicon():
        chip = info.get(
            "gpu", ""
        )  # populated with apple_chip_name(), e.g. "Apple M4 Max"
        if chip and chip != "none":
            short = chip[len("Apple ") :] if chip.startswith("Apple ") else chip
            return f"Mac {short}"
        return "Mac"
    gpu = info.get("gpu", "") or ""
    for chip, label in KNOWN_ACCELERATORS.items():
        if chip in gpu:
            return label
    return f"{sys.platform}-{platform.machine()}"


def _slug(s):
    return re.sub(r"[^A-Za-z0-9.]+", "-", s).strip("-") or "x"


def default_output(info, device):
    """A self-describing path under figures/ so runs on different hardware/dates/
    hyperparameters never overwrite and are identifiable at a glance, e.g.
    figures/benchmark_gpu_b4_t1024_2026-06-30_2127_NVIDIA-GB10_DGX-Spark.png

    The trailing component is a human-meaningful platform/silicon description
    (see platform_description()), never the machine's hostname.
    """
    date = info["date"][:16].replace(" ", "_").replace(":", "")  # YYYY-MM-DD_HHMM
    hw = _slug(info["gpu"]) if info["gpu"] != "none" else _slug(info["cpu"])
    name = f"benchmark_{device}_b{B}_t{T}_{date}_{hw}_{_slug(info['platform'])}.png"
    return os.path.join(FIGURES, name)


# --------------------------------------------------------------------------- #
# Running + parsing
# --------------------------------------------------------------------------- #


def _run(cmd, env=None, cwd=None, timeout=2400):
    e = dict(os.environ)
    if env:
        e.update(env)
    # errors="replace": llm.c's fp32cu binary samples generated text at its
    # last step regardless of -s (upstream `step > 0 && step % sample_every
    # == 0 || last_step` always fires on the final step), and printed GPT-2
    # byte-level-BPE token bytes are not guaranteed to be valid UTF-8 on
    # their own — a strict decode intermittently crashes the whole benchmark
    # depending on which token got sampled. We only regex-parse the timing
    # lines (which are always plain ASCII), so lossy-replacing any stray
    # invalid bytes in the (unused) sample text is harmless.
    p = subprocess.run(
        cmd,
        env=e,
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    return p.stdout + "\n" + p.stderr


# llm.mojo harness: "... | forward: 0.11s | backward: 0.30s | update: 0.03s"
_MOJO_RE = re.compile(
    r"forward:\s*([\d.]+)s\s*\|\s*backward:\s*([\d.]+)s\s*\|\s*update:\s*([\d.]+)s"
)
_LLMC_CPU_RE = re.compile(r"took\s*([\d.]+)\s*ms")  # "(took 1790.12 ms)"
_LLMC_CUDA_RE = re.compile(r"\|\s*([\d.]+)\s*ms\s*\|")  # bf16cu: "| 26.14 ms |"
_LLMC_FP32CU_RE = re.compile(r"\(([\d.]+)\s*ms,")  # fp32cu: "(40.74 ms, ...)"
_TORCH_RE = re.compile(r"\(([\d.]+)\s*ms\s*\|")  # torch: "(47.55 ms | ...)"


def bench_mojo(binary, target, steps):
    if not os.path.exists(binary):
        print(f"  (skip: {os.path.basename(binary)} not built)", flush=True)
        return []
    env = {
        "LLMM_PROFILE_B": str(B),
        "LLMM_PROFILE_T": str(T),
        "LLMM_PROFILE_LAYERS": "12",
        "LLMM_PROFILE_STEPS": str(steps),
        "MOJO_PYTHON_LIBRARY": libpython(),
    }
    out = _run([binary, target], env=env)
    return [
        (float(f) + float(b) + float(u)) * 1000.0 for f, b, u in _MOJO_RE.findall(out)
    ]


def bench_llmc_cpu(threads):
    # CPU train_gpt2 runs ~40 steps in build/llmc. Upstream hardcodes B=4, T=64;
    # our build reads LLMC_B/LLMC_T (defaulting to 4/64 when unset), so pass the
    # benchmark's B/T to keep the CPU bars apples-to-apples with the rest.
    out = _run(
        [os.path.join(LLMC, "train_gpt2")],
        env={"OMP_NUM_THREADS": str(threads), "LLMC_B": str(B), "LLMC_T": str(T)},
        cwd=STAGE,
    )
    return [float(x) for x in _LLMC_CPU_RE.findall(out)]


def bench_llmc_cuda_bf16(steps):
    out = _run(
        [
            os.path.join(LLMC, "train_gpt2cu"),
            "-e",
            "gpt2_124M_bf16.bin",
            "-i",
            "dev/data/tinyshakespeare/tiny_shakespeare_train.bin",
            "-j",
            "dev/data/tinyshakespeare/tiny_shakespeare_val.bin",
            "-b",
            str(B),
            "-t",
            str(T),
            "-x",
            str(steps),
            "-v",
            "0",
            "-s",
            "0",
            "-l",
            "0",  # no val/sample/lr-warmup noise
        ],
        cwd=STAGE,
    )
    return [float(x) for x in _LLMC_CUDA_RE.findall(out)]


def bench_llmc_cuda_fp32(steps):
    # fp32cu has no -x (runs one epoch) and loads gpt2_124M.bin unconditionally.
    # Point both data flags at the small val bin to keep the epoch short, then
    # cap the parsed samples to `steps` downstream.
    out = _run(
        [
            os.path.join(LLMC, "train_gpt2fp32cu"),
            "-i",
            "dev/data/tinyshakespeare/tiny_shakespeare_val.bin",
            "-j",
            "dev/data/tinyshakespeare/tiny_shakespeare_val.bin",
            "-b",
            str(B),
            "-t",
            str(T),
            "-v",
            "10000",
            "-s",
            "10000",  # push val/sample past the run
        ],
        cwd=STAGE,
    )
    return [float(x) for x in _LLMC_FP32CU_RE.findall(out)][: steps + WARMUP]


def bench_torch(device, dtype, steps):
    cmd = [
        sys.executable,
        TORCH_REF,
        "--model",
        "d12",  # random GPT-2 124M, no HF download
        "--write_tensors",
        "0",
        "--num_iterations",
        str(steps),
        "--batch_size",
        str(B),
        "--sequence_length",
        str(T),
        # One fwd/bwd/update per measured step (grad_accum=1), matching every other
        # config. The ref asserts total_batch_size % (B*T) == 0 and derives
        # grad_accum = total_batch_size // (B*T); pin it to B*T so this holds at any
        # T (the default 256 only divides at T=64).
        "--total_batch_size",
        str(B * T),
        "--dtype",
        dtype,
        "--device",
        device,
        "--inference_only",
        "0",
        "--overfit_single_batch",
        "1",
        "--val_loss_every",
        "0",
        "--sample_every",
        "0",
        "--tensorcores",
        "1",
        "--compile",
        "0",
    ]
    out = _run(cmd, cwd=STAGE)
    return [float(x) for x in _TORCH_RE.findall(out)]


def bench_torch_mps(steps, dtype="float32"):
    """Benchmark our local train_gpt2.py on Apple MPS (Metal Performance Shaders).

    Uses the repo-local train_gpt2.py (not llm.c's reference) because it already
    has --device mps support with torch.mps.synchronize() for accurate timing.
    The output format matches _TORCH_RE: '({ms:.2f} ms | {tok/s:.0f} tok/s)'.

    dtype: "float32" or "bfloat16". The repo-local train_gpt2.py supports both via
    torch.amp.autocast(device_type='mps', dtype=...) — see the --dtype flag.
    """
    if not os.path.exists(TORCH_TRAIN_PY):
        print(
            f"  (skip PyTorch MPS {dtype}: {TORCH_TRAIN_PY} not found)",
            flush=True,
        )
        return []
    cmd = [
        sys.executable,
        TORCH_TRAIN_PY,
        "--device",
        "mps",
        "--model",
        "d12",  # random GPT-2 124M weights, no HF download
        "--write_tensors",
        "0",
        "--num_iterations",
        str(steps),
        "--batch_size",
        str(B),
        "--sequence_length",
        str(T),
        # grad_accum = total_batch_size // (B*T) = 1 — one fwd+bwd+update/step.
        "--total_batch_size",
        str(B * T),
        "--dtype",
        dtype,
        "--inference_only",
        "0",
        "--overfit_single_batch",
        "1",
        "--val_loss_every",
        "0",
        "--sample_every",
        "0",
        "--tensorcores",
        "0",
        "--compile",
        "0",
    ]
    out = _run(cmd, cwd=ROOT, timeout=3600)
    samples = [float(x) for x in _TORCH_RE.findall(out)]
    if not samples:
        # Surface the failure clearly so the Metal run still emits the Mojo half.
        print(
            f"  WARNING: PyTorch MPS {dtype} arm returned no timing samples.\n"
            "  This may mean train_gpt2.py --device mps is not yet fully "
            "implemented or MPS is unavailable.\n"
            "  llm.mojo Metal timings will still be reported.\n"
            "  Captured output (last 20 lines):\n" + "\n".join(out.splitlines()[-20:]),
            flush=True,
        )
    return samples


# --------------------------------------------------------------------------- #
# Staging llm.c inputs (magic-patched copies + data symlinks) into build/llmc
# --------------------------------------------------------------------------- #


def _patch_magic(src, dst, magic):
    shutil.copyfile(src, dst)
    with open(dst, "r+b") as f:
        f.seek(0)
        f.write(struct.pack("<i", magic))


def stage_llmc(need_cpu, need_gpu):
    os.makedirs(os.path.join(STAGE, "dev", "data", "tinyshakespeare"), exist_ok=True)
    # token files: llm.c's magic already matches llm.mojo's, so plain symlinks.
    for split in ("train", "val"):
        link = os.path.join(
            STAGE, "dev", "data", "tinyshakespeare", f"tiny_shakespeare_{split}.bin"
        )
        tgt = os.path.join(DATA, f"tiny_shakespeare_{split}.bin")
        if not os.path.lexists(link):
            os.symlink(tgt, link)
    tok = os.path.join(STAGE, "gpt2_tokenizer.bin")
    if not os.path.exists(tok):
        _patch_magic(os.path.join(ROOT, "gpt2_tokenizer.bin"), tok, MAGIC_TOKENIZER)
    # fp32 model: needed by llm.c CPU and llm.c fp32 CUDA.
    if need_cpu or need_gpu:
        m = os.path.join(STAGE, "gpt2_124M.bin")
        if not os.path.exists(m):
            _patch_magic(os.path.join(ROOT, "gpt2_124M.bin"), m, MAGIC_MODEL)
    # bf16 model: needed by llm.c bf16 CUDA.
    if need_gpu:
        m = os.path.join(STAGE, "gpt2_124M_bf16.bin")
        if not os.path.exists(m):
            _patch_magic(os.path.join(ROOT, "gpt2_124M_bf16.bin"), m, MAGIC_MODEL)


# --------------------------------------------------------------------------- #
# Stats + plot
# --------------------------------------------------------------------------- #


def trim(samples):
    return samples[WARMUP:] if len(samples) > WARMUP else samples


def summarize(name, label, family, precision, samples):
    import statistics

    s = trim(samples)
    if not s:
        return None
    mean = statistics.mean(s)
    return {
        "name": name,
        "label": label,
        "family": family,
        "precision": precision,
        "n": len(s),
        "mean": mean,
        "median": statistics.median(s),
        "std": statistics.pstdev(s) if len(s) > 1 else 0.0,
        "min": min(s),
        "tok_s": (B * T) / mean * 1000.0,
        "samples": s,
    }


def _hardware_subtitle(info):
    """Build a de-duplicated hardware/date subtitle from info's platform/gpu/cpu
    fields.

    On Apple Silicon, `platform` ("Mac M4 Max"), `gpu` ("Apple M4 Max"), and
    `cpu` (also "Apple M4 Max" from the sysctl brand string) all encode the
    same chip name — naively joining them repeats "M4 Max" three times. This
    collapses any component whose chip name is already implied by `platform`
    (or by `gpu`, for the cpu check) into a single mention, folding the core
    count onto whichever component survives.
    """

    def norm(s):
        return re.sub(r"\s+", " ", (s or "")).strip().lower()

    def chip(s):
        return s[len("Apple ") :] if s.startswith("Apple ") else s

    platform_str = info.get("platform", "") or ""
    gpu = info.get("gpu", "") or ""
    cpu = info.get("cpu", "") or ""
    cores = info.get("cores", "")
    date_display = info.get("date_display", "")

    parts = []
    if platform_str:
        parts.append(platform_str)

    gpu_redundant = gpu and gpu != "none" and norm(chip(gpu)) in norm(platform_str)
    if gpu and gpu != "none" and not gpu_redundant:
        parts.append(gpu)

    cpu_redundant = cpu and (
        norm(chip(cpu)) in norm(platform_str) or norm(chip(cpu)) in norm(gpu)
    )
    if cpu and not cpu_redundant:
        parts.append(f"{cpu} ({cores} cores)" if cores else cpu)
    elif cores and parts and "cores)" not in parts[0]:
        # Core count would otherwise be lost if cpu was folded away as a dup.
        parts[0] = f"{parts[0]} ({cores} cores)"

    if date_display:
        parts.append(date_display)
    return "  |  ".join(parts)


def plot_bars(device, series, info, outpath, footer=None):

    matplotlib.use("Agg")

    series = [s for s in series if s]
    if not series:
        return
    n = len(series)
    # Content-scaled width (with a floor so a 2-bar Metal chart doesn't look
    # cramped); constrained layout + bbox_inches="tight" on save (below) do the
    # rest of the work to guarantee nothing — bars, title, or subtitle — clips.
    fig, ax = plt.subplots(figsize=(max(8.0, 1.8 * n), 6.0), layout="constrained")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xs = range(n)
    means = [s["mean"] for s in series]
    stds = [s["std"] for s in series]
    colors = [FAMILY_COLORS.get(s["family"], FAMILY_COLOR_FALLBACK) for s in series]

    # Recessive horizontal gridlines behind the bars.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)

    bars = ax.bar(
        xs,
        means,
        width=0.62,
        yerr=stds,
        color=colors,
        capsize=4,
        ecolor=TEXT_GRAY,
        error_kw=dict(elinewidth=1.2, capthick=1.2),
        edgecolor=TEXT_INK,
        linewidth=1.2,
        zorder=3,
    )
    # Precision stays encoded by texture, not color: bf16 bars get a subtle
    # white 45-degree hatch over the SAME implementation hue used for fp32.
    # Matplotlib gotcha: hatch lines are drawn in the edgecolor, so getting
    # WHITE hatch lines with an INK outline takes two artists at the SAME
    # position — the hatched bar itself (edgecolor=white, linewidth=0 so no
    # white border shows) plus an unfilled ink-outline rectangle on top.
    # NOTE: ax.bar() defaults to align="center" (x is the bar's CENTER) while
    # Rectangle.get_x() returns its LEFT edge — the overlay must convert, or
    # it lands half a bar-width off its base bar.
    for b, s in zip(bars, series):
        if s["precision"] == "bf16":
            b.set_hatch("///")
            b.set_edgecolor("white")
            b.set_linewidth(0)
            ax.bar(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                width=b.get_width(),
                fill=False,
                edgecolor=TEXT_INK,
                linewidth=1.2,
                zorder=4,
            )

    ax.set_xticks(list(xs))
    ax.set_xticklabels([s["label"] for s in series])
    ax.set_ylabel("train-loop time per step (ms)", color=TEXT_INK)

    # Headroom above the tallest bar+error so the two-line value/throughput
    # label never collides with the error bar cap or the axes top.
    top = max(m + sd for m, sd in zip(means, stds))
    ax.set_ylim(0, top * 1.15)

    # Recessive spines/ticks; drop the top and right spines entirely.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT_GRAY)
    ax.spines["bottom"].set_color(TEXT_GRAY)
    ax.tick_params(colors=TEXT_GRAY)

    # Annotate each bar with the mean and throughput, in ink (never series color).
    for b, s in zip(bars, series):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + s["std"] + top * 0.02,
            f"{s['mean']:.1f} ms\n{s['tok_s']:.0f} tok/s",
            ha="center",
            va="bottom",
            fontsize=8,
            color=TEXT_INK,
        )

    dev_label = {"gpu": "GPU", "cpu": "CPU", "metal": "Metal GPU"}.get(
        device, device.upper()
    )
    # Hyperparameters belong on the figure itself so a saved PNG is self-describing:
    # batch, sequence length, and the resulting tokens/step the bars are measured at.
    cfg = f"B={B}, T={T}  ({B * T} tok/step)"
    # "(lower is better)" lives on the subtitle, not the title: the title must
    # end clear of the right-margin legend column (constrained layout centers
    # the suptitle across the FULL figure width, legend column included).
    title = f"GPT-2 124M {dev_label} training-loop time — {cfg}"
    # Machine/date details move to a smaller, de-duplicated, wrapped subtitle
    # line rather than crowding (and overflowing) the main title.
    sub = _hardware_subtitle(info) + "  |  lower is better"
    sub_wrapped = textwrap.fill(sub, width=max(60, int(fig.get_figwidth() * 11)))

    # Left-anchored so the title can never run under the right-margin legend.
    fig.suptitle(title, fontsize=13, color=TEXT_INK, x=0.02, ha="left")
    ax.set_title(sub_wrapped, fontsize=9.5, color=TEXT_GRAY, pad=8)

    # Precision legend only when both fp32 and bf16 appear — otherwise the x
    # tick labels already carry the precision, and a legend is redundant.
    # Placed OUTSIDE the axes (constrained layout reserves space for it) so it
    # can never collide with the value annotation above the tallest bar.
    precisions = {s["precision"] for s in series}
    if {"fp32", "bf16"} <= precisions:
        legend = [
            Patch(facecolor=LEGEND_FILL, edgecolor=TEXT_INK, label="fp32"),
            Patch(facecolor=LEGEND_FILL, edgecolor=TEXT_INK, hatch="///", label="bf16"),
        ]
        fig.legend(
            handles=legend,
            fontsize=9,
            loc="outside right upper",
            frameon=False,
        )

    if footer:
        # supxlabel participates in constrained layout, so the figure grows to
        # fit the footer instead of overlapping the x tick labels (the old
        # fig.text(0.5, 0.0, ...) was invisible to the layout engine). Wrap so
        # a long footer becomes extra lines rather than overflowing the width.
        footer_wrapped = textwrap.fill(
            footer, width=max(80, int(fig.get_figwidth() * 13))
        )
        fig.supxlabel(
            footer_wrapped,
            fontsize=7.5,
            color=TEXT_GRAY,
            style="italic",
        )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    print(f"\n{dev_label} bar chart written to {outpath}")


def print_summary(info, device, rows, footer=None):
    dev_label = {"gpu": "GPU", "cpu": "CPU", "metal": "METAL GPU"}.get(
        device, device.upper()
    )
    print("=" * 84)
    print(f"  {dev_label}    Date: {info['date']}")
    print(f"  Host:  {info['host']}  ({info['os']})")
    print(f"  CPU:   {info['cpu']}  ({info['cores']} cores)")
    print(f"  GPU:   {info['gpu']}")
    print("=" * 84)
    print(
        f"  {'configuration':28s} {'n':>4s} {'mean ms':>10s} "
        f"{'median':>9s} {'std':>8s} {'tok/s':>9s}"
    )
    print("-" * 84)
    for r in rows:
        flat = r["label"].replace("\n", " ")
        print(
            f"  {flat:28s} {r['n']:>4d} {r['mean']:>10.2f} "
            f"{r['median']:>9.2f} {r['std']:>8.2f} {r['tok_s']:>9.0f}"
        )
    print("=" * 84)
    if footer:
        print(f"  Note: {footer}")


# --------------------------------------------------------------------------- #
# Config tables
# --------------------------------------------------------------------------- #


def cpu_series(steps, cores):
    print(f"[cpu] llm.mojo fp32  (B={B} T={T} x{steps}) ...", flush=True)
    yield summarize(
        "cpu/llm.mojo",
        "llm.mojo\nfp32",
        "llm.mojo",
        "fp32",
        bench_mojo(MOJO_FP32_BIN, "cpu", steps),
    )
    print(f"[cpu] llm.c  (OpenMP, {cores} threads) ...", flush=True)
    yield summarize(
        "cpu/llm.c-omp",
        "llm.c\n(OpenMP)",
        "llm.c",
        "fp32",
        bench_llmc_cpu(cores),
    )
    print("[cpu] llm.c  (1 thread, no OpenMP) ...", flush=True)
    yield summarize(
        "cpu/llm.c-1t",
        "llm.c\n(1 thread)",
        "llm.c",
        "fp32",
        bench_llmc_cpu(1),
    )
    print(f"[cpu] PyTorch fp32  (B={B} T={T} x{steps}) ...", flush=True)
    yield summarize(
        "cpu/torch",
        "PyTorch\nfp32",
        "PyTorch",
        "fp32",
        bench_torch("cpu", "float32", steps),
    )


def gpu_series(steps):
    print(f"[gpu] llm.mojo fp32  (B={B} T={T} x{steps}) ...", flush=True)
    yield summarize(
        "gpu/llm.mojo-fp32",
        "llm.mojo\nfp32",
        "llm.mojo",
        "fp32",
        bench_mojo(MOJO_FP32_BIN, "gpu", steps),
    )
    print(f"[gpu] llm.mojo bf16  (B={B} T={T} x{steps}) ...", flush=True)
    yield summarize(
        "gpu/llm.mojo-bf16",
        "llm.mojo\nbf16",
        "llm.mojo",
        "bf16",
        bench_mojo(MOJO_BF16_BIN, "gpu", steps),
    )
    print(f"[gpu] llm.c CUDA fp32  (B={B} T={T}) ...", flush=True)
    yield summarize(
        "gpu/llm.c-fp32",
        "llm.c CUDA\nfp32",
        "llm.c",
        "fp32",
        bench_llmc_cuda_fp32(steps),
    )
    print(f"[gpu] llm.c CUDA bf16  (B={B} T={T} x{steps}) ...", flush=True)
    yield summarize(
        "gpu/llm.c-bf16",
        "llm.c CUDA\nbf16",
        "llm.c",
        "bf16",
        bench_llmc_cuda_bf16(steps),
    )
    print(f"[gpu] PyTorch fp32  (B={B} T={T} x{steps}) ...", flush=True)
    yield summarize(
        "gpu/torch-fp32",
        "PyTorch\nfp32",
        "PyTorch",
        "fp32",
        bench_torch("cuda", "float32", steps),
    )
    print(f"[gpu] PyTorch bf16  (B={B} T={T} x{steps}) ...", flush=True)
    yield summarize(
        "gpu/torch-bf16",
        "PyTorch\nbf16",
        "PyTorch",
        "bf16",
        bench_torch("cuda", "bfloat16", steps),
    )


def metal_series(steps, cooldown_s=30):
    """Benchmark series for Apple Silicon Metal GPU — 4 arms: fp32 + bf16 for both
    llm.mojo and PyTorch MPS.

    llm.c has NO Metal port — the baseline is PyTorch on MPS (Metal Performance
    Shaders) via the repo-local train_gpt2.py.

    The llm.mojo fp32 arm uses build/profile_gpt2 (the fp32 harness) with target
    'gpu': on Apple Silicon, Mojo's GPU backend dispatches to Metal automatically.
    The llm.mojo bf16 arm uses build/profile_gpt2_bf16 (the -D LLMM_BF16 build),
    which auto-loads gpt2_124M_bf16.bin.

    PyTorch MPS bf16 uses train_gpt2.py --device mps --dtype bfloat16 with
    torch.amp.autocast(device_type='mps', dtype=torch.bfloat16).

    cooldown_s: seconds to sleep between arms so each starts comparably cool
    (M4 Max hits the thermal throttle cliff at ~8 s of sustained load — P16 in
    docs/ai/metal_port_gotchas_and_optimizations.md).
    """
    print(
        "NOTE: llm.c has no Metal port — baseline is PyTorch MPS "
        "(train_gpt2.py --device mps).",
        flush=True,
    )
    if cooldown_s > 0:
        print(
            f"  Thermal discipline: {cooldown_s}s cooldown between arms "
            "(M4 Max throttle cliff ~8 s, P16).",
            flush=True,
        )

    print(f"[metal] llm.mojo fp32  (B={B} T={T} x{steps}) ...", flush=True)
    yield summarize(
        "metal/llm.mojo-fp32",
        "llm.mojo\nfp32",
        "llm.mojo",
        "fp32",
        bench_mojo(MOJO_FP32_BIN, "gpu", steps),
    )

    if cooldown_s > 0:
        print(f"  Cooling down {cooldown_s}s ...", flush=True)
        time.sleep(cooldown_s)

    print(f"[metal] llm.mojo bf16  (B={B} T={T} x{steps}) ...", flush=True)
    yield summarize(
        "metal/llm.mojo-bf16",
        "llm.mojo\nbf16",
        "llm.mojo",
        "bf16",
        bench_mojo(MOJO_BF16_BIN, "gpu", steps),
    )

    if cooldown_s > 0:
        print(f"  Cooling down {cooldown_s}s ...", flush=True)
        time.sleep(cooldown_s)

    print(f"[metal] PyTorch MPS fp32  (B={B} T={T} x{steps}) ...", flush=True)
    yield summarize(
        "metal/torch-mps-fp32",
        "PyTorch MPS\nfp32",
        "PyTorch",
        "fp32",
        bench_torch_mps(steps, "float32"),
    )

    if cooldown_s > 0:
        print(f"  Cooling down {cooldown_s}s ...", flush=True)
        time.sleep(cooldown_s)

    print(f"[metal] PyTorch MPS bf16  (B={B} T={T} x{steps}) ...", flush=True)
    yield summarize(
        "metal/torch-mps-bf16",
        "PyTorch MPS\nbf16",
        "PyTorch",
        "bf16",
        bench_torch_mps(steps, "bfloat16"),
    )


def main():
    # B/T are module globals read by the bench_* helpers; the arg defaults below
    # reference them, so the declaration must precede first use.
    global B, T
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--device",
        choices=["cpu", "gpu", "metal", "auto"],
        default="auto",
        help=(
            "cpu: CPU comparison only; "
            "gpu: NVIDIA GPU (requires CUDA); "
            "metal: Apple Silicon Metal GPU (llm.mojo fp32+bf16, PyTorch MPS fp32+bf16); "
            "auto: CPU always, plus Metal on Apple Silicon or GPU on NVIDIA (default)"
        ),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=B,
        help=f"batch size B for every config (default {B})",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=T,
        help=f"sequence length T for every config (default {T}). T must be "
        "<= 1024 (GPT-2's max). CPU fp32 cost scales steeply with T.",
    )
    ap.add_argument("--cpu-steps", type=int, default=40)
    ap.add_argument("--gpu-steps", type=int, default=40)
    ap.add_argument(
        "--metal-steps",
        type=int,
        default=DEFAULT_METAL_STEPS,
        help=(
            f"measured steps per arm for the Metal run (default {DEFAULT_METAL_STEPS}). "
            "fp32 Metal at B=4 T=1024 is ~6.5 s/step; keep this modest. "
            "All four arms (llm.mojo fp32/bf16, PyTorch MPS fp32/bf16) use this count."
        ),
    )
    ap.add_argument(
        "--cooldown-s",
        type=int,
        default=30,
        help=(
            "seconds to sleep between Metal benchmark arms (default 30). "
            "The M4 Max hits the thermal throttle cliff at ~8 s of sustained load "
            "(P16 in docs/ai/metal_port_gotchas_and_optimizations.md); a per-arm "
            "cooldown ensures each arm starts comparably cool. Set to 0 to skip."
        ),
    )
    ap.add_argument("--output-cpu", default=None, help="output PNG for the CPU figure")
    ap.add_argument("--output-gpu", default=None, help="output PNG for the GPU figure")
    ap.add_argument(
        "--output-metal", default=None, help="output PNG for the Metal GPU figure"
    )
    ap.add_argument(
        "--stage-only",
        action="store_true",
        help="only stage llm.c inputs into build/llmc (magic-patched), then exit",
    )
    args = ap.parse_args()

    # Pin the module globals to the requested config (see note at top of main()).
    B, T = args.batch_size, args.seq_len
    if not 0 < T <= 1024:
        ap.error(f"--seq-len must be in 1..1024 (GPT-2's max), got {T}")

    # Determine which modes to run.
    # Metal and GPU are mutually exclusive in auto mode: Apple Silicon has Metal,
    # NVIDIA Linux boxes have GPU. Explicit --device gpu on Apple will also run
    # as Metal (since the Mojo 'gpu' target dispatches to Metal on Apple Silicon).
    on_apple = is_apple_silicon()
    do_metal = args.device == "metal" or (args.device in ("gpu", "auto") and on_apple)
    do_gpu = (
        args.device == "gpu" or (args.device == "auto" and have_gpu() and not on_apple)
    ) and not do_metal
    do_cpu = args.device in ("cpu", "auto")

    if args.device == "gpu" and not have_gpu() and not on_apple:
        print("warning: --device gpu but no NVIDIA GPU detected", file=sys.stderr)

    # Stage llm.c inputs only when needed (Metal mode has no llm.c dependency).
    if do_cpu or do_gpu:
        stage_llmc(need_cpu=do_cpu, need_gpu=do_gpu)
    if args.stage_only:
        print(f"staged llm.c inputs into {STAGE}")
        return

    info = hardware_info()
    cores = int(info["cores"])

    if do_cpu:
        series = [s for s in cpu_series(args.cpu_steps, cores) if s]
        if series:
            print_summary(info, "cpu", series)
            plot_bars(
                "cpu", series, info, args.output_cpu or default_output(info, "cpu")
            )

    if do_gpu:
        series = [s for s in gpu_series(args.gpu_steps) if s]
        if series:
            print_summary(info, "gpu", series)
            plot_bars(
                "gpu", series, info, args.output_gpu or default_output(info, "gpu")
            )

    if do_metal:
        cooldown_s = args.cooldown_s
        metal_footer = (
            f"Thermal discipline: {cooldown_s}s cooldown between each of the 4 arms "
            f"(M4 Max throttle cliff ~8 s, P16). "
            "llm.c has no Metal port — baseline is PyTorch MPS."
            if cooldown_s > 0
            else "llm.c has no Metal port — baseline is PyTorch MPS."
        )
        series = [s for s in metal_series(args.metal_steps, cooldown_s=cooldown_s) if s]
        if series:
            print_summary(info, "metal", series, footer=metal_footer)
            plot_bars(
                "metal",
                series,
                info,
                args.output_metal or default_output(info, "metal"),
                footer=metal_footer,
            )
        else:
            print(
                "Metal benchmark produced no results. "
                "Ensure build/profile_gpt2 and build/profile_gpt2_bf16 are built "
                "(make build-profile build-profile-bf16).",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
