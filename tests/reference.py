"""PyTorch AdamW reference implementation + fixture dumper.

Two modes:
  1. As a library — imported by `tests/test_adamw_equivalence.py` to compute
     expected trajectories in-process from a torch.optim.AdamW.
  2. As a CLI — `python -m tests.reference dump` writes deterministic
     fixtures (params/grads/expected state) to `tests/fixtures/*.npz` for
     the Mojo-side tests that have no Python interpreter at runtime.

The two paths share `simulate()` so the bytes a Mojo test compares against
are guaranteed to match what an in-process pytest run would compute.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class AdamWParams:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_scale: float = 1.0


@dataclass(frozen=True)
class Case:
    """A single deterministic test case.

    Fixed seed + shape + dtype + step count + AdamW hyperparams uniquely
    determines the params/grads stream and the expected post-step state.
    """

    name: str
    n: int
    steps: int
    dtype: str  # "float32" | "bfloat16" | "float16"
    seed: int
    hp: AdamWParams

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]

    @property
    def np_dtype(self) -> np.dtype:
        # bfloat16 has no numpy dtype — store as uint16 view alongside an fp32 copy.
        return {
            "float32": np.dtype("float32"),
            "bfloat16": np.dtype("uint16"),
            "float16": np.dtype("float16"),
        }[self.dtype]


# Default catalogue. Add cases here as the optimizer grows new code paths.
CASES: tuple[Case, ...] = (
    Case("fp32_small", n=128, steps=20, dtype="float32", seed=0, hp=AdamWParams()),
    Case("fp32_unaligned", n=131, steps=10, dtype="float32", seed=1, hp=AdamWParams()),
    Case(
        "fp32_wd0",
        n=64,
        steps=10,
        dtype="float32",
        seed=2,
        hp=AdamWParams(weight_decay=0.0),
    ),
    Case("fp32_zero_grad", n=64, steps=5, dtype="float32", seed=3, hp=AdamWParams()),
    Case("bf16_small", n=128, steps=20, dtype="bfloat16", seed=4, hp=AdamWParams()),
)


def _zero_grad_override(case: Case) -> bool:
    return case.name == "fp32_zero_grad"


def _make_inputs(case: Case) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return (initial_params, [grad_step_0, grad_step_1, ...])."""
    g = torch.Generator().manual_seed(case.seed)
    init_params = torch.randn(case.n, generator=g, dtype=torch.float32).to(
        case.torch_dtype
    )
    grads: list[torch.Tensor] = []
    for _ in range(case.steps):
        if _zero_grad_override(case):
            grads.append(torch.zeros(case.n, dtype=case.torch_dtype))
        else:
            grads.append(
                torch.randn(case.n, generator=g, dtype=torch.float32).to(
                    case.torch_dtype
                )
            )
    return init_params, grads


def simulate(case: Case) -> dict[str, np.ndarray]:
    """Run torch.optim.AdamW for `case.steps` steps, returning fixture bytes.

    Returned arrays:
      init_params  : (n,)
      grads        : (steps, n)
      final_params : (n,)
      final_m      : (n,)  exp_avg
      final_v      : (n,)  exp_avg_sq
      trajectory   : (steps, n)  params after each step (handy for first-divergence diffs)
    """
    init_params, grads = _make_inputs(case)
    param = init_params.clone().detach().requires_grad_(True)

    opt = torch.optim.AdamW(
        [param],
        lr=case.hp.lr,
        betas=(case.hp.beta1, case.hp.beta2),
        eps=case.hp.eps,
        weight_decay=case.hp.weight_decay,
    )

    trajectory = []
    for g in grads:
        opt.zero_grad(set_to_none=False)
        # Inject the deterministic gradient and apply grad_scale exactly the
        # way the Mojo kernel does (multiplies grad by grad_scale during load).
        param.grad = (case.hp.grad_scale * g.to(torch.float32)).to(param.dtype)
        opt.step()
        trajectory.append(param.detach().clone().to(torch.float32).numpy())

    state = opt.state[param]
    return {
        "init_params": _to_storable(init_params, case),
        "grads": _to_storable(torch.stack(grads), case),
        "final_params": _to_storable(param.detach(), case),
        "final_m": state["exp_avg"].to(torch.float32).numpy(),
        "final_v": state["exp_avg_sq"].to(torch.float32).numpy(),
        "trajectory": np.stack(trajectory),
    }


def _to_storable(t: torch.Tensor, case: Case) -> np.ndarray:
    """numpy has no bfloat16 — round-trip through uint16 view for storage."""
    if case.dtype == "bfloat16":
        return t.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
    return t.to(case.torch_dtype).cpu().numpy()


def dump(cases: Iterable[Case] = CASES, out_dir: Path = FIXTURES_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for case in cases:
        payload = simulate(case)
        path = out_dir / f"{case.name}.npz"
        # pyrefly: ignore[bad-argument-type]  numpy's `savez(file, *args, **kwds)`
        # accepts arbitrary keyword-named arrays; pyrefly's stub mis-maps
        # **payload against `allow_pickle`.
        np.savez(path, **payload)
        manifest.append(
            {
                "name": case.name,
                "path": path.name,
                "n": case.n,
                "steps": case.steps,
                "dtype": case.dtype,
                "seed": case.seed,
                "hp": asdict(case.hp),
            }
        )
        print(f"wrote {path} ({case.n} params x {case.steps} steps, {case.dtype})")

    # A small JSON-ish manifest lets the Mojo loader iterate fixtures without
    # hardcoding the catalogue in two places.
    import json

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="AdamW reference + fixture tool")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dump", help="write reference fixtures to tests/fixtures/")
    args = p.parse_args()
    if args.cmd == "dump":
        dump()


if __name__ == "__main__":
    main()
