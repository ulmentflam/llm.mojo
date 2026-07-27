"""Regenerate the 10-step reference loss trajectory used by ``test_gpt2.mojo``.

WHY THIS FILE EXISTS
--------------------
``test_gpt2.mojo`` hardcodes a list of ten expected losses and compares the
Mojo trainer's overfitting run against it. That list used to be produced by a
script living at ``/Users/evanowen/Workspace/scripts/llmm-metal-probes/
gen_expected_losses.py`` -- a path that is not in this repository and that
nobody else can run. The list it left behind was wrong, and because the
generator was missing there was no way to tell that it was wrong, or to fix it:

* Its step-0 value was 5.354427, while the reference loss actually stored in
  ``gpt2_124M_debug_state.bin`` is 5.2678394. The comment above the list
  asserted the two were the same ("Step-0 must be ~5.354 (debug-state
  reference)"). They were not.
* Consequently ``make verify-gpu`` / ``verify-gpu-tf32`` / ``verify-cpu``
  reported ``LOSS MISMATCH`` on all ten steps and exited non-zero on *every*
  commit, including baselines predating any of the work it was blamed on.

A gate that fails for a bogus reason is as harmful as one that passes without
testing anything -- arguably worse, because a permanently red gate teaches
people to ignore it, and then it cannot warn them when something real breaks.

SO: THE ONE RULE FOR REGENERATING THIS LIST
-------------------------------------------
**The expected losses must come from PyTorch, never from the Mojo trainer's
own output.** Pasting the Mojo numbers in would make the comparison compare
the implementation against itself -- a tautology that can never fail, which
looks green forever while testing nothing. If this list ever needs updating,
run THIS script and paste THIS script's output. That is the whole point of
committing it.

Usage
-----
    pixi run python scripts/gen_expected_losses.py
    pixi run python scripts/gen_expected_losses.py --device cuda

It prints a ready-to-paste Mojo list literal.

Configuration (must stay in sync with ``model.update()`` in test_gpt2.mojo)
--------------------------------------------------------------------------
    optimizer      AdamW
    lr             1e-4
    betas          (0.9, 0.999)
    eps            1e-8
    weight_decay   0.01   -- uniform, applied to ALL parameters (no no-decay
                             group for biases/LayerNorm, matching the Mojo
                             trainer, which decays everything)
    grad clipping  none
    steps          10, all on the SAME batch (deliberate overfit)
    batch          x/y read from gpt2_124M_debug_state.bin (B=4, T=64)
    dtype          float32, TF32 disabled (strict IEEE, matching
                   `make verify-gpu`'s -D LLMM_NO_TF32=1)
    init           GPT.from_pretrained("gpt2"), the same initialisation
                   train_gpt2.py uses to write the debug state
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# train_gpt2.py guards its training driver behind `if __name__ == "__main__"`,
# so importing it for the model definition is side-effect free.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gpt2 import GPT  # noqa: E402

STATE_FILE = "gpt2_124M_debug_state.bin"
MAGIC_EXPECTED = 20240520
VERSION_EXPECTED = 3
NUM_STEPS = 10

# Optimizer hyperparameters. Changing any of these invalidates the list.
LR = 1e-4
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01


def read_debug_state(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (x, y, reference_loss) from the debug state file.

    Layout, mirroring ``write_state`` in train_gpt2.py and the reads in
    test_gpt2.mojo:335-354:
        header  256 x int32   (magic, version, B, T, ...)
        x       B*T  x int32
        y       B*T  x int32
        logits  B*T*V x float32
        loss    1    x float32
        ... gradients, activations (not needed here)
    """
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        magic, version, batch, seq = (
            int(header[0]),
            int(header[1]),
            int(header[2]),
            int(header[3]),
        )
        if magic != MAGIC_EXPECTED:
            raise SystemExit(
                f"{path}: magic {magic} != {MAGIC_EXPECTED}. This is llm.c's\n"
                "downloaded debug state, which has no activation tensors.\n"
                "Regenerate it first with: pixi run python train_gpt2.py"
            )
        if version != VERSION_EXPECTED:
            raise SystemExit(f"{path}: version {version} != {VERSION_EXPECTED}")

        x = np.frombuffer(f.read(batch * seq * 4), dtype=np.int32).reshape(batch, seq)
        y = np.frombuffer(f.read(batch * seq * 4), dtype=np.int32).reshape(batch, seq)

        # Skip the logits block to reach the stored reference loss. V is the
        # UNPADDED vocab size, matching test_gpt2.mojo's `V = config.vocab_size`.
        vocab = 50257
        f.seek(batch * seq * vocab * 4, 1)
        (ref_loss,) = struct.unpack("<f", f.read(4))

    return x, y, ref_loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device (default: cpu, for reproducibility across boxes)",
    )
    parser.add_argument("--state-file", default=STATE_FILE)
    args = parser.parse_args()

    # Strict IEEE fp32. TF32 would drift the trajectory by ~1e-2, which is the
    # same order as the tolerance this list is compared against.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    state_path = Path(args.state_file)
    if not state_path.exists():
        raise SystemExit(
            f"{state_path} not found. Generate it with:\n"
            "  pixi run python train_gpt2.py"
        )

    x_np, y_np, ref_loss = read_debug_state(state_path)
    print(f"batch from {state_path}: B={x_np.shape[0]} T={x_np.shape[1]}")
    print(f"reference step-0 loss stored in the file: {ref_loss:.7f}")

    device = torch.device(args.device)
    x = torch.from_numpy(x_np.astype(np.int64)).to(device)
    y = torch.from_numpy(y_np.astype(np.int64)).to(device)

    model = GPT.from_pretrained("gpt2")
    model.to(device)
    model.train()
    model.float()

    # Uniform weight decay over every parameter -- deliberately NOT the
    # split decay/no-decay grouping used for real training runs, because the
    # Mojo trainer's adamw_update applies one decay to everything.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )

    losses: list[float] = []
    for step in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        # No gradient clipping: the Mojo trainer's verify path does not clip.
        optimizer.step()
        losses.append(float(loss.item()))
        print(f"step {step}: loss {losses[-1]:.7f}")

    # Self-check. Step 0 is a plain forward pass on the same batch and the same
    # initial weights that produced the file, so it must reproduce the stored
    # reference loss. If it does not, something upstream has changed and the
    # list must NOT be pasted anywhere.
    drift = abs(losses[0] - ref_loss)
    print(
        f"\nstep-0 vs stored reference: {losses[0]:.7f} vs {ref_loss:.7f} "
        f"(drift {drift:.2e})"
    )
    if drift > 1e-3:
        raise SystemExit(
            "FAIL: step-0 loss does not reproduce the reference loss stored in\n"
            f"{state_path} (drift {drift:.2e} > 1e-3). The model init, the batch,\n"
            "or the file is inconsistent. Do NOT paste this output."
        )
    print("OK: step 0 reproduces the stored reference loss.")

    print("\nPaste into test_gpt2.mojo:\n")
    print("    var expected_losses: List[Float32] = [")
    for value in losses:
        print(f"        {value!r},")
    print("    ]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
