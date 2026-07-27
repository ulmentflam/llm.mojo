#!/usr/bin/env python3
"""Convert an llm.mojo model checkpoint to a Hugging Face GPT2LMHeadModel export.

Why this script exists
-----------------------
llm.mojo uses its own checkpoint magic number (20240520, see MODEL_MAGIC in
llmm/checkpointing.mojo) to distinguish its checkpoint format from upstream
llm.c's own checkpoints (magic 20240326), even though the 256-int32 header
layout and parameter blob order are otherwise byte-identical (same fields:
maxT, V, L, H, C, Vp; same parameter order: wte, wpe, ln1w/b, qkvw/b,
attprojw/b, ln2w/b, fcw/b, fcprojw/b, lnfw/b). This is intentional and
permanent -- llmm/checkpointing.mojo's MODEL_MAGIC must not be changed to
match llm.c's.

This script makes a *patched copy* of an llm.mojo checkpoint with llm.c's
original magic number written in, solely so upstream llm.c tooling
(third_party/llm.c/dev/eval/export_hf.py) can read it, and then runs that
converter to produce a standard Hugging Face `transformers` export
(safetensors, GPT2LMHeadModel-compatible). It NEVER modifies the source
checkpoint -- it operates on a temporary copy only.

Usage
-----
  python scripts/export_to_hf.py --input log124M/model_19552.bin --output /tmp/hf_export

This is a thin wrapper; the real conversion logic (header parsing, tensor
reshaping, bf16/fp32 handling) lives in
third_party/llm.c/dev/eval/export_hf.py and is unchanged/untouched here.
"""

import argparse
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# See llmm/checkpointing.mojo: MODEL_MAGIC (llm.mojo) vs. llm.c's own magic,
# which third_party/llm.c/dev/eval/export_hf.py hardcodes as the only magic
# it accepts.
LLM_MOJO_MAGIC = 20240520
LLM_C_MAGIC = 20240326

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_HF_PY = REPO_ROOT / "third_party" / "llm.c" / "dev" / "eval" / "export_hf.py"

# The four bias tensors produced by `matmul_bias_bwd`, in HF naming. These are
# checked for non-zeroness on export because a scratch overrun in that kernel
# once shipped six published checkpoints whose `c_attn`/`c_fc` biases were
# bit-exactly zero -- never updated by a single optimizer step -- and nothing
# between the trainer and the Hub noticed
# (docs/ai/dbias_scratch_overrun_silent_zero_bug.md). GPT-2 initialises biases
# to exactly 0, so "still zero after training" is decisive rather than
# suggestive, and upload is the last point at which it is cheap to catch.
MATMUL_BIAS_SUFFIXES = (
    "attn.c_attn.bias",
    "attn.c_proj.bias",
    "mlp.c_fc.bias",
    "mlp.c_proj.bias",
)


def verify_export(output: Path) -> bool:
    """Check an export is loadable, finite, and not silently missing gradients.

    Deliberately CPU-only and dependency-light: it must be runnable while the
    GPUs are busy training, which is exactly when exports happen.
    """
    from safetensors import safe_open

    path = output / "model.safetensors"
    if not path.is_file():
        print(f"[verify] FAIL: {path} does not exist")
        return False

    ok = True
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        nonfinite = []
        dead = []
        for k in keys:
            t = f.get_tensor(k).float()
            if not t.isfinite().all():
                nonfinite.append(k)
            if k.endswith(MATMUL_BIAS_SUFFIXES) and not t.any():
                dead.append(k)

    print(f"[verify] {len(keys)} tensors in {path}")

    if nonfinite:
        print(f"[verify] FAIL: {len(nonfinite)} tensor(s) contain NaN/Inf:")
        for k in nonfinite[:8]:
            print(f"           {k}")
        ok = False

    if dead:
        print(
            f"[verify] FAIL: {len(dead)} matmul bias tensor(s) are entirely zero,"
            " i.e. never received a gradient:"
        )
        for k in dead[:8]:
            print(f"           {k}")
        print(
            "         See docs/ai/dbias_scratch_overrun_silent_zero_bug.md."
            " Do NOT publish this checkpoint."
        )
        ok = False

    if ok:
        print("[verify] OK: all tensors finite, all matmul biases trained")
    return ok


def make_llmc_compatible_copy(src: Path, dst: Path) -> None:
    """Copy `src` to `dst`, then patch only the first int32 (the magic
    number) from LLM_MOJO_MAGIC to LLM_C_MAGIC. The source file is opened
    read-only and is never written to; only `dst` is modified.
    """
    shutil.copyfile(src, dst)

    with open(dst, "r+b") as f:
        (magic,) = struct.unpack("<i", f.read(4))
        if magic != LLM_MOJO_MAGIC:
            raise ValueError(
                f"expected llm.mojo magic {LLM_MOJO_MAGIC} at the start of "
                f"{src}, found {magic}. Refusing to patch a file that "
                "doesn't look like an llm.mojo checkpoint."
            )
        f.seek(0)
        f.write(struct.pack("<i", LLM_C_MAGIC))

    print(
        f"[export_to_hf] patched copy magic {LLM_MOJO_MAGIC} -> {LLM_C_MAGIC} in {dst}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="llm.mojo model_*.bin checkpoint"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="output directory for the HF export"
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="dtype to export weights as (passed through to export_hf.py)",
    )
    parser.add_argument(
        "--spin",
        action="store_true",
        help="run upstream's post-export generation smoke test (needs a free "
        "CUDA device, accelerate, and flash-attention; off by default)",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="skip the CPU-side check that the export is finite and its matmul "
        "biases actually trained",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input checkpoint not found: {args.input}")
    if not EXPORT_HF_PY.is_file():
        parser.error(
            f"could not find llm.c's export_hf.py at {EXPORT_HF_PY} "
            "(is the third_party/llm.c submodule checked out?)"
        )

    with tempfile.TemporaryDirectory(prefix="llm_mojo_hf_export_") as tmpdir:
        patched = Path(tmpdir) / args.input.name
        make_llmc_compatible_copy(args.input, patched)

        cmd = [
            sys.executable,
            str(EXPORT_HF_PY),
            "--input",
            str(patched),
            "--output",
            str(args.output),
            "--dtype",
            args.dtype,
            # `spin` is upstream's post-export generation smoke test. It is off
            # by default here because it needs a free CUDA device, `accelerate`,
            # and a flash-attention build -- none of which hold while the box is
            # training, and its absence used to surface as a traceback AFTER a
            # perfectly good export had already been written. argparse declares
            # it `type=bool`, so any non-empty string is True and "" is False.
            "--spin",
            "1" if args.spin else "",
        ]
        print(f"[export_to_hf] running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return result.returncode

    if args.verify:
        return 0 if verify_export(args.output) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
