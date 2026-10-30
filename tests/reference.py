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

from tests._dtypes import NP_STORAGE_DTYPES, TORCH_DTYPES, to_storage


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
        return TORCH_DTYPES[self.dtype]

    @property
    def np_dtype(self) -> np.dtype:
        # bfloat16 has no numpy dtype — stored as a uint16 view (see tests/_dtypes.py).
        return NP_STORAGE_DTYPES[self.dtype]


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
    return to_storage(t.detach().to(torch.float32).cpu().numpy(), case.dtype)


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


# ---------------------------------------------------------------------------
# Muon
# ---------------------------------------------------------------------------
# The Mojo `muon_update` kernel implements exactly this math (explicit torch,
# not torch.optim.Muon, whose details differ):
#   g = grad_scale * G;  M = mu*M + g;  U = g + mu*M (nesterov) else M
#   X = bf16(U);  X /= (||X||_F + 1e-7);  transpose if rows > cols
#   5x: A = X X^T; B = b*A + c*A@A; X = a*X + B@X      (a,b,c = 3.4445,-4.7750,2.0315)
#   O = X (transposed back);  W -= lr * (0.2*sqrt(max(r,c)) * O + wd * W)

NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315


@dataclass(frozen=True)
class MuonParams:
    lr: float = 0.02
    momentum: float = 0.95
    weight_decay: float = 0.1
    grad_scale: float = 1.0
    ns_steps: int = 5
    nesterov: bool = True


@dataclass(frozen=True)
class MuonCase:
    name: str
    dtype: str  # "float32" | "bfloat16"
    rows: int
    cols: int
    steps: int
    hp: MuonParams
    seed: int

    @property
    def torch_dtype(self) -> torch.dtype:
        return TORCH_DTYPES[self.dtype]


MUON_CASES: tuple[MuonCase, ...] = (
    MuonCase("muon_fp32_64x64", "float32", 64, 64, 6, MuonParams(), 10),
    MuonCase("muon_fp32_48x96", "float32", 48, 96, 6, MuonParams(), 11),
    MuonCase("muon_fp32_96x48", "float32", 96, 48, 6, MuonParams(), 12),
    MuonCase("muon_fp32_256x768", "float32", 256, 768, 3, MuonParams(), 13),
    MuonCase("muon_bf16_64x64", "bfloat16", 64, 64, 6, MuonParams(), 14),
    MuonCase("muon_bf16_96x48", "bfloat16", 96, 48, 6, MuonParams(), 15),
    MuonCase("muon_bf16_256x768", "bfloat16", 256, 768, 3, MuonParams(), 16),
)


def newton_schulz(
    u: torch.Tensor, steps: int = 5, ns_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Orthogonalise fp32 `u` (rows x cols); returns fp32 O of the same shape."""
    rows, cols = u.shape
    x = u.to(ns_dtype)
    x = x / (x.float().norm() + 1e-7).to(ns_dtype)
    if rows > cols:
        x = x.T
    for _ in range(steps):
        a = x @ x.T
        b = NS_B * a + NS_C * (a @ a)
        x = NS_A * x + b @ x
    if rows > cols:
        x = x.T
    return x.float()


def _muon_inputs(case: MuonCase) -> tuple[torch.Tensor, list[torch.Tensor]]:
    g = torch.Generator().manual_seed(case.seed)
    shape = (case.rows, case.cols)
    init = (0.5 * torch.randn(shape, generator=g)).to(case.torch_dtype)
    grads = [
        torch.randn(shape, generator=g).to(case.torch_dtype) for _ in range(case.steps)
    ]
    return init, grads


def _muon_run(case: MuonCase, ns_dtype: torch.dtype):
    init, grads = _muon_inputs(case)
    hp = case.hp
    w = init.clone()
    m = torch.zeros(case.rows, case.cols, dtype=torch.float32)
    scale = 0.2 * (max(case.rows, case.cols) ** 0.5)
    traj, moms = [], []
    for G in grads:
        g = hp.grad_scale * G.float()
        m = hp.momentum * m + g
        u = g + hp.momentum * m if hp.nesterov else m
        o = newton_schulz(u, hp.ns_steps, ns_dtype)
        wf = w.float()
        w = (wf - hp.lr * (scale * o + hp.weight_decay * wf)).to(case.torch_dtype)
        traj.append(w.float().numpy())
        moms.append(m.numpy().copy())
    return init, grads, np.stack(traj), np.stack(moms)


def simulate_muon(case: MuonCase) -> dict[str, np.ndarray]:
    """Torch Muon reference.

    Returns init_params (storage dtype), grads (steps, r, c; storage dtype),
    trajectory (steps, r, c) fp32 params after each step, momentum (steps, r, c)
    fp32, trajectory_fp32ns (same loop with fp32 Newton-Schulz) and
    spread (scalar): max|trajectory - trajectory_fp32ns|.
    """
    init, grads, traj, moms = _muon_run(case, torch.bfloat16)
    _, _, traj32, _ = _muon_run(case, torch.float32)
    return {
        "init_params": _to_storable(init, case),  # type: ignore[arg-type]
        "grads": _to_storable(torch.stack(grads), case),  # type: ignore[arg-type]
        "trajectory": traj,
        "momentum": moms,
        "trajectory_fp32ns": traj32,
        "spread": np.asarray(np.max(np.abs(traj - traj32)), dtype=np.float32),
    }


def muon_tolerance(case: MuonCase, ref: dict[str, np.ndarray]) -> float:
    """atol = 2*max(spread_bf16_vs_fp32, 1e-6) + rtol*max|W|.

    The kernel's bf16 GEMMs round differently from torch's, so the allowed
    error is derived from how far torch's own bf16 NS already sits from fp32 NS.
    rtol is 1e-3 (fp32) / 8e-3 (bf16: one param ulp).
    """
    rtol = 8e-3 if case.dtype == "bfloat16" else 1e-3
    spread = float(ref["spread"])
    wmax = float(np.max(np.abs(ref["trajectory"])))
    return 2 * max(spread, 1e-6) + rtol * wmax


def dump_muon(out_dir: Path = FIXTURES_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in MUON_CASES:
        path = out_dir / f"{case.name}.npz"
        # pyrefly: ignore[bad-argument-type]
        np.savez(path, **simulate_muon(case))
        print(f"wrote {path} ({case.rows}x{case.cols} x {case.steps} steps)")


def main() -> None:
    p = argparse.ArgumentParser(description="AdamW reference + fixture tool")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dump", help="write reference fixtures to tests/fixtures/")
    args = p.parse_args()
    if args.cmd == "dump":
        dump()
        dump_muon()


if __name__ == "__main__":
    main()
