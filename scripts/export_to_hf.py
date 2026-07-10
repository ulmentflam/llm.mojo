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
        ]
        print(f"[export_to_hf] running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
