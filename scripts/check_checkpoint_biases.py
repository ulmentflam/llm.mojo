#!/usr/bin/env python3
"""Decide, from a checkpoint alone, whether its biases ever received a gradient.

GPT-2 initialises every bias to exactly 0 (train_gpt2.mojo:2048 -- "weights ~
N(0, 0.02), biases 0"). So a bias tensor that is still bit-exactly zero after
22,345 optimizer steps was never updated. That makes this a direct test of the
dbias scratch-overrun bug rather than an inference from the kernel analysis.

The four matmul biases (qkvb, attprojb, fcprojb, fcb) go through
`matmul_bias_bwd`, the kernel that overran its scratch. The three layernorm
biases (ln1b, ln2b, lnfb) have their own backward and are the control: if the
layernorm biases are non-zero and the matmul biases are zero in the same file,
the fault is specific to the matmul path and is not, say, a checkpoint-writer
bug or a dead optimizer.

Usage: check_biases.py <checkpoint.bin> [...]
"""

import struct
import sys

import numpy as np

HEADER_INTS = 256


def tensor_layout(maxT, V, L, C, Vp):
    """(name, element count) in checkpoint order -- see llmm/checkpointing.mojo."""
    return [
        ("wte", Vp * C),
        ("wpe", maxT * C),
        ("ln1w", L * C),
        ("ln1b", L * C),
        ("qkvw", L * 3 * C * C),
        ("qkvb", L * 3 * C),
        ("attprojw", L * C * C),
        ("attprojb", L * C),
        ("ln2w", L * C),
        ("ln2b", L * C),
        ("fcw", L * 4 * C * C),
        ("fcb", L * 4 * C),
        ("fcprojw", L * C * 4 * C),
        ("fcprojb", L * C),
        ("lnfw", C),
        ("lnfb", C),
    ]


MATMUL_BIASES = {"qkvb", "attprojb", "fcb", "fcprojb"}
LAYERNORM_BIASES = {"ln1b", "ln2b", "lnfb"}


def check(path):
    with open(path, "rb") as f:
        header = struct.unpack(f"<{HEADER_INTS}i", f.read(HEADER_INTS * 4))
        magic, version, maxT, V, L, H, C = header[:7]
        Vp = header[7]
        payload = f.read()

    total = sum(n for _, n in tensor_layout(maxT, V, L, C, Vp))
    itemsize = len(payload) / total
    if abs(itemsize - 2) < 1e-9:
        dtype, kind = np.uint16, "bf16"
    elif abs(itemsize - 4) < 1e-9:
        dtype, kind = np.float32, "fp32"
    else:
        raise SystemExit(f"{path}: {len(payload)} bytes / {total} params = {itemsize}")

    raw = np.frombuffer(payload, dtype=dtype)
    print(f"\n=== {path}")
    print(
        f"    magic={magic} version={version} L={L} C={C} Vp={Vp} params={total} {kind}"
    )

    off = 0
    verdicts = {}
    for name, n in tensor_layout(maxT, V, L, C, Vp):
        chunk = raw[off : off + n]
        off += n
        if name not in MATMUL_BIASES and name not in LAYERNORM_BIASES:
            continue
        # bf16 zero and fp32 zero are both the all-bits-zero pattern, so this
        # works without decoding the float format.
        nonzero = int(np.count_nonzero(chunk))
        verdicts[name] = (nonzero, n)
        tag = "matmul   " if name in MATMUL_BIASES else "layernorm"
        state = (
            "ALL ZERO -- never updated" if nonzero == 0 else f"{nonzero}/{n} non-zero"
        )
        print(f"    {tag} {name:9s} {state}")

    frozen = [k for k in MATMUL_BIASES if verdicts.get(k, (1, 1))[0] == 0]
    ln_live = [k for k in LAYERNORM_BIASES if verdicts.get(k, (0, 1))[0] > 0]
    if len(frozen) == len(MATMUL_BIASES) and len(ln_live) == len(LAYERNORM_BIASES):
        print(
            "    VERDICT: frozen-bias run (matmul biases dead, layernorm biases alive)"
        )
    elif not frozen:
        print("    VERDICT: healthy -- every matmul bias received gradients")
    else:
        print(
            f"    VERDICT: mixed/unexpected -- frozen={sorted(frozen)} ln_live={sorted(ln_live)}"
        )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        check(p)
