#!/usr/bin/env python3
"""Write a copy of an llm.mojo model or state checkpoint with its architecture
flags word corrected. Only that one int32 changes.

Why: model header slot 20 and state header slot 44 record which architecture a
file was trained with (llmm/checkpointing.mojo ARCH_FLAG_*: 1 rotary, 2 untied
head [state files], 4 QK-norm). QK-norm only got its bit after the first
rotary+QK-norm checkpoints were written, so those files record rotary (1) but
not QK-norm, and a build without -D LLMM_QK_NORM=1 loads them and computes the
wrong model without complaint. This writes a corrected COPY; the input is only
read. Refuses an unknown magic or version, and an output that exists.

  python scripts/set_checkpoint_arch_flags.py model_5700.bin out/model_5700.bin \
      --flags rotary,qk-norm
  python scripts/set_checkpoint_arch_flags.py state_5700_0.bin out/state_5700_0.bin \
      --flags rotary,qk-norm

A flagged file is version 2 for state files (version 1 is the unflagged
format), so a version-1 state file cannot be given flags here (that would
change more than the flag word), and a version-2 state file cannot be set to 0.
"""

import argparse
import shutil
import struct
import sys
from pathlib import Path

MODEL_MAGIC = 20240520
STATE_MAGIC = 20240527
# Model versions llmm/checkpointing.mojo reads: 3 fp32, 5 bf16, +10 untied head.
MODEL_VERSIONS = (3, 5, 13, 15)
STATE_VERSIONS = (1, 2)
MODEL_FLAGS_SLOT = 20
STATE_FLAGS_SLOT = 44
FLAG_BITS = {"rotary": 1, "untied": 2, "qk-norm": 4}
MODEL_FLAG_MASK = FLAG_BITS["rotary"] | FLAG_BITS["qk-norm"]


def describe(flags: int) -> str:
    names = [n for n, b in FLAG_BITS.items() if flags & b]
    rest = flags & ~sum(FLAG_BITS.values())
    if rest:
        names.append(f"unknown({rest})")
    return f"{flags} ({'|'.join(names) if names else 'none'})"


def parse_flags(text: str) -> int:
    if text.strip().lstrip("-").isdigit():
        return int(text)
    out = 0
    for part in text.split(","):
        part = part.strip()
        if part in ("", "none"):
            continue
        if part not in FLAG_BITS:
            raise SystemExit(f"unknown flag '{part}'; known: {', '.join(FLAG_BITS)}")
        out |= FLAG_BITS[part]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument(
        "--flags",
        required=True,
        help="new flags word: comma list of rotary,qk-norm,untied (untied: state "
        "files only) or an integer",
    )
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = ap.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise SystemExit("refusing to write over the input; give a different output")
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists (use --force to overwrite)")

    with open(args.input, "rb") as f:
        head = f.read(256 * 4)
    if len(head) < 256 * 4:
        raise SystemExit(f"{args.input}: shorter than a 256-int header")
    header = struct.unpack("<256i", head)
    magic, version = header[0], header[1]
    if magic == MODEL_MAGIC:
        kind, slot, versions = "model", MODEL_FLAGS_SLOT, MODEL_VERSIONS
    elif magic == STATE_MAGIC:
        kind, slot, versions = "state", STATE_FLAGS_SLOT, STATE_VERSIONS
    else:
        raise SystemExit(
            f"{args.input}: unknown magic {magic} (want {MODEL_MAGIC} model or "
            f"{STATE_MAGIC} state)"
        )
    if version not in versions:
        raise SystemExit(
            f"{args.input}: unsupported {kind} version {version} (known: {versions})"
        )

    new = parse_flags(args.flags)
    known = sum(FLAG_BITS.values())
    if new & ~known or new < 0:
        raise SystemExit(f"flags {new} has unknown bits")
    if kind == "model" and new & ~MODEL_FLAG_MASK:
        raise SystemExit(
            "model files carry rotary and qk-norm only (an untied model file is "
            "marked by its version, not a flag)"
        )
    old = header[slot]
    if kind == "state":
        if version == 1 and new != 0:
            raise SystemExit(
                "state version 1 is the unflagged format; setting flags would "
                "also change the version word, which this tool does not do"
            )
        if version == 2 and new == 0:
            raise SystemExit("a version-2 state file must carry nonzero flags")
        if version == 2 and (new & FLAG_BITS["untied"]) != (old & FLAG_BITS["untied"]):
            raise SystemExit("refusing to change the untied bit of a state file")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, args.output)
    with open(args.output, "r+b") as f:
        f.seek(slot * 4)
        f.write(struct.pack("<i", new))

    print(f"{kind} file {args.input} (version {version}), header slot {slot}")
    print(f"  old flags: {describe(old)}")
    print(f"  new flags: {describe(new)}")
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
