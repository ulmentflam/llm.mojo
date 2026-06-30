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
  GPU:
    llm.mojo (fp32) | llm.mojo (bf16) | llm.c CUDA (fp32) | llm.c CUDA (bf16)
    | PyTorch (fp32) | PyTorch (bf16)

Device selection:
  --device cpu   only the CPU figure (works on macOS / any box, no CUDA)
  --device gpu   only the GPU figure (requires an NVIDIA GPU + the binaries)
  --device auto  CPU always; GPU too iff an NVIDIA GPU is detected (default)

Prereqs (built by the Makefile): build/profile_gpt2 (llm.mojo fp32 harness),
build/profile_gpt2_bf16 (llm.mojo bf16 harness), the llm.c binaries
third_party/llm.c/{train_gpt2, train_gpt2cu, train_gpt2fp32cu}, and PyTorch in
the pixi env. The llm.c inputs are staged into build/llmc/ with their magic
numbers patched to the values llm.c expects (the byte layout is otherwise
identical). The PyTorch reference is llm.c's train_gpt2.py run with random d12
(=GPT-2 124M) weights, so it needs no network.
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
import matplotlib
import matplotlib.pyplot as plt

# Legend: precision hatch key.
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLMC = os.path.join(ROOT, "third_party", "llm.c")
TORCH_REF = os.path.join(LLMC, "train_gpt2.py")
STAGE = os.path.join(ROOT, "build", "llmc")
DATA = os.path.join(ROOT, "data", ".tinyshakespeare")
FIGURES = os.path.join(ROOT, "figures")

MOJO_FP32_BIN = os.path.join(ROOT, "build", "profile_gpt2")
MOJO_BF16_BIN = os.path.join(ROOT, "build", "profile_gpt2_bf16")

# Apples-to-apples config: identical hyperparameters everywhere so the bars are
# directly comparable. Set from --batch-size / --seq-len in main(); every config
# (Mojo harness env, llm.c CUDA -b/-t, PyTorch flags, and — via LLMC_B/LLMC_T —
# the llm.c CPU reference) is fed the same pair. The defaults (B=4, T=64) match
# what llm.c's CPU reference historically hardcoded.
B, T = 4, 64
WARMUP = 5  # leading steps to drop (allocation / first-touch / clock spin-up)

# llm.c expects different magic numbers than llm.mojo writes; the byte layout is
# otherwise the same, so we patch a copy. (model, tokenizer.)
MAGIC_MODEL = 20240326
MAGIC_TOKENIZER = 20240328

# Per-implementation-family colors; bf16 bars are hatched to read at a glance.
FAMILY_COLORS = {
    "llm.mojo": "#1f77b4",
    "llm.c": "#d62728",
    "PyTorch": "#ff7f0e",
}


def libpython():
    import glob

    for pat in ("libpython3*.so", "libpython3*.dylib"):
        hits = glob.glob(os.path.join(ROOT, ".pixi/envs/default/lib", pat))
        if hits:
            return hits[0]
    return ""


def _ordinal(day):
    # 1->1st, 2->2nd, 3->3rd, 4->4th ... 11/12/13 are always "th".
    if 11 <= day % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def hardware_info():
    now = datetime.datetime.now()
    info = {
        # `date` keeps the timestamp for unique output filenames; `date_display`
        # is the human date shown on the figure itself (e.g. "July 31st, 2026").
        "date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date_display": f"{now.strftime('%B')} {_ordinal(now.day)}, {now.year}",
        "host": platform.node(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cores": str(os.cpu_count()),
        "gpu": gpu_name() or "none",
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


def _slug(s):
    return re.sub(r"[^A-Za-z0-9.]+", "-", s).strip("-") or "x"


def default_output(info, device):
    """A self-describing path under figures/ so runs on different hardware/dates/
    hyperparameters never overwrite and are identifiable at a glance, e.g.
    figures/benchmark_gpu_b4_t1024_2026-06-30_2127_NVIDIA-GB10_spark-c265.png
    """
    date = info["date"][:16].replace(" ", "_").replace(":", "")  # YYYY-MM-DD_HHMM
    hw = _slug(info["gpu"]) if info["gpu"] != "none" else _slug(info["cpu"])
    name = f"benchmark_{device}_b{B}_t{T}_{date}_{hw}_{_slug(info['host'])}.png"
    return os.path.join(FIGURES, name)


# --------------------------------------------------------------------------- #
# Running + parsing
# --------------------------------------------------------------------------- #


def _run(cmd, env=None, cwd=None, timeout=2400):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(
        cmd, env=e, cwd=cwd, capture_output=True, text=True, timeout=timeout
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


def plot_bars(device, series, info, outpath):

    matplotlib.use("Agg")

    series = [s for s in series if s]
    if not series:
        return
    n = len(series)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.7 * n), 6.0))

    xs = range(n)
    means = [s["mean"] for s in series]
    stds = [s["std"] for s in series]
    colors = [FAMILY_COLORS.get(s["family"], "#7f7f7f") for s in series]
    hatches = ["//" if s["precision"] == "bf16" else "" for s in series]

    bars = ax.bar(
        xs, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.6
    )
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([s["label"] for s in series])
    ax.set_ylabel("train-loop time per step (ms)")
    ax.set_ylim(0, max(m + sd for m, sd in zip(means, stds)) * 1.18)

    # Annotate each bar with the mean and throughput.
    for b, s in zip(bars, series):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(means) * 0.012 + s["std"],
            f"{s['mean']:.1f} ms\n{s['tok_s']:.0f} tok/s",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    sub = (
        f"{info['date_display']}   |   {info['gpu']}   |   {info['cpu']}  "
        f"({info['cores']} cores)   |   {info['host']}"
    )
    dev_label = "GPU" if device == "gpu" else "CPU"
    # Hyperparameters belong on the figure itself so a saved PNG is self-describing:
    # batch, sequence length, and the resulting tokens/step the bars are measured at.
    cfg = f"B={B}, T={T}  ({B * T} tok/step)"
    fig.suptitle(
        f"GPT-2 124M {dev_label} training-loop time  —  {cfg}  (lower is better)\n"
        + sub,
        fontsize=11,
    )

    legend = [
        Patch(facecolor="#cccccc", edgecolor="black", label="fp32"),
        Patch(facecolor="#cccccc", edgecolor="black", hatch="//", label="bf16"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="upper right")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=130)
    print(f"\n{dev_label} bar chart written to {outpath}")


def print_summary(info, device, rows):
    print("=" * 84)
    print(f"  {device.upper()}    Date: {info['date']}")
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


def main():
    # B/T are module globals read by the bench_* helpers; the arg defaults below
    # reference them, so the declaration must precede first use.
    global B, T
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--device", choices=["cpu", "gpu", "auto"], default="auto")
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
    ap.add_argument("--output-cpu", default=None, help="output PNG for the CPU figure")
    ap.add_argument("--output-gpu", default=None, help="output PNG for the GPU figure")
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

    do_cpu = args.device in ("cpu", "auto")
    do_gpu = args.device == "gpu" or (args.device == "auto" and have_gpu())
    if args.device == "gpu" and not have_gpu():
        print("warning: --device gpu but no NVIDIA GPU detected", file=sys.stderr)

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


if __name__ == "__main__":
    main()
