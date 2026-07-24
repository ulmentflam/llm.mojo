#!/usr/bin/env python3
"""Launch a multi-GPU ZeRO training run from a zero/*.json config.

A config bundles the three knobs a ZeRO run needs — compile-time world size,
runtime ZeRO stage, and precision (which selects the training binary) — plus
any llm.c-style training flags, so `make train-zero ZERO_CONFIG=zero/zero2.json`
reproduces a run without hand-assembling build + runner invocations. Baseline
configs for stages 1-3 live in zero/ (see zero/README.md for the schema).

The launcher shells out to make (the project convention for reproducible runs):
first `make build* WORLD_SIZE=N` for the config's precision, then the matching
train target with `-z <stage> -pn <world_size>` and the config's train_flags.
Extra flags after `--` are appended last, so they override the config, e.g.:

    python scripts/train_zero.py --config zero/zero2.json -- -x 50 -o log_z2
"""

import argparse
import json
import os
import subprocess
import sys

# precision -> (build target, train target); precision picks the binary.
_PRECISION_TARGETS = {
    "fp32": ("build", "train"),
    "bf16": ("build-bf16", "train-bf16"),
    "fp8": ("build-fp8", "train-fp8"),
    "fp4": ("build-fp4", "train-fp4"),
}


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="zero/zero2.json")
    ap.add_argument(
        "extra",
        nargs="*",
        help="training flags appended after the config's train_flags "
        "(use -- before them)",
    )
    args = ap.parse_args()

    root = _repo_root()
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(root, config_path)
    with open(config_path) as f:
        config = json.load(f)

    stage = int(config["zero_stage"])
    world_size = int(config["world_size"])
    precision = config.get("precision", "bf16")
    if precision not in _PRECISION_TARGETS:
        raise SystemExit(
            f"{args.config}: unknown precision {precision!r} "
            f"(expected one of {sorted(_PRECISION_TARGETS)})"
        )
    build_target, train_target = _PRECISION_TARGETS[precision]

    train_args = ["-z", str(stage), "-pn", str(world_size)]
    for flag, value in config.get("train_flags", {}).items():
        train_args += [f"-{flag}", str(value)]
    train_args += args.extra

    build_cmd = ["make", build_target, f"WORLD_SIZE={world_size}"]
    train_cmd = ["make", train_target, "ARGS=" + " ".join(train_args)]
    for cmd in (build_cmd, train_cmd):
        print("+ " + " ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
