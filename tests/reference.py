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
    """Orthogonalise fp32 `u` (rows x cols); returns fp32 O of the same shape.

    Models what the kernel does with cuBLAS(Lt) bf16 GEMMs: operands are bf16,
    products accumulate in fp32, the `b*A + c*A@A` / `a*X + B@X` epilogues
    (GEMM beta) are fp32, and each GEMM output is rounded to bf16 once. The
    1/(||X||+eps) scale is fp32 applied to bf16 X and rounded once (the norm
    itself is never rounded to bf16). With ns_dtype=float32 nothing rounds.
    """
    rows, cols = u.shape

    def rnd(t: torch.Tensor) -> torch.Tensor:
        return t.to(ns_dtype).float()

    x = rnd(u)
    x = rnd(x / (x.norm() + 1e-7))
    if rows > cols:
        x = x.T
    for _ in range(steps):
        a = rnd(x @ x.T)
        b = rnd(NS_B * a + NS_C * (a @ a))
        x = rnd(NS_A * x + b @ x)
    if rows > cols:
        x = x.T
    return x.contiguous()


def effective_hp(case: MuonCase) -> MuonParams:
    """Hyperparameters as the kernel sees them.

    The op contract passes lr/momentum/weight_decay/grad_scale as
    `Scalar[dtype]`, so a bf16 case hands the kernel bf16-rounded values
    (momentum 0.95 -> 0.94921875) that it then promotes to fp32. The reference
    must use the same values or a correct kernel fails the momentum check.
    """
    hp = case.hp

    def rnd(v: float) -> float:
        return float(torch.tensor(v).to(case.torch_dtype).float())

    return MuonParams(
        lr=rnd(hp.lr),
        momentum=rnd(hp.momentum),
        weight_decay=rnd(hp.weight_decay),
        grad_scale=rnd(hp.grad_scale),
        ns_steps=hp.ns_steps,
        nesterov=hp.nesterov,
    )


def muon_scale(rows: int, cols: int) -> float:
    """Moonlight RMS rule."""
    return 0.2 * max(rows, cols) ** 0.5


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
    hp = effective_hp(case)
    w = init.clone()
    m = torch.zeros(case.rows, case.cols, dtype=torch.float32)
    scale = muon_scale(case.rows, case.cols)
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


def _torch_muon_run(case: MuonCase, direction_k: int | None = None):
    """Drive the stock `torch.optim.Muon` with our spec's hyperparameters.

    torch convention differences (see tests/test_muon_crosscheck.py header):
    its buffer is (1-mu)*ours (lerp form) and Nesterov update is (1-mu)*U, a
    constant factor NS normalises away. Grads are pre-multiplied by grad_scale.
    The torch param is fp32 holding storage-dtype values and is rounded to the
    storage dtype after every step, so a bf16 case rounds once per step and
    keeps an fp32 momentum buffer like the kernel (torch.optim.Muon on bf16
    params would round W twice and store the buffer in bf16).
    With `direction_k` set, W is zeroed before that step and its (W1 - 0) is
    returned as the update (wd*W vanishes), else None.
    """
    init, grads = _muon_inputs(case)
    hp = effective_hp(case)
    p = torch.nn.Parameter(init.float().clone())
    opt = torch.optim.Muon(
        [p],
        lr=hp.lr,
        weight_decay=hp.weight_decay,
        momentum=hp.momentum,
        nesterov=hp.nesterov,
        ns_steps=hp.ns_steps,
        adjust_lr_fn="match_rms_adamw",
    )
    traj, moms, w1 = [], [], None
    for k, G in enumerate(grads):
        if direction_k == k:
            p.data.zero_()
        p.grad = hp.grad_scale * G.float()
        opt.step()
        p.data = p.data.to(case.torch_dtype).float()
        if direction_k == k:
            w1 = p.data.numpy().copy()
        traj.append(p.data.numpy().copy())
        buf = opt.state[p]["momentum_buffer"]
        moms.append((buf / (1.0 - hp.momentum)).numpy().copy())
    return init, grads, np.stack(traj), np.stack(moms), w1


def simulate_muon_torch_optim(case: MuonCase) -> dict[str, np.ndarray]:
    """Same keys as `simulate_muon`, produced by stock `torch.optim.Muon`.

    `momentum` is rescaled by 1/(1-mu) to our convention. `spread` is the
    larger of our NS-rounding spread and the measured torch-vs-ours trajectory
    gap, so `muon_tolerance` becomes 2x the observed independent-implementation
    gap (+ the usual rtol term).
    """
    init, grads, traj, moms, _ = _torch_muon_run(case)
    ours = simulate_muon(case)
    gap = float(np.max(np.abs(traj - ours["trajectory"])))
    out = dict(ours)
    out.update(
        trajectory=traj,
        momentum=moms,
        spread=np.asarray(max(float(ours["spread"]), gap), dtype=np.float32),
    )
    return out


def torch_optim_direction(case: MuonCase, k: int) -> np.ndarray:
    """Orthogonalised update O of step k from stock torch.optim.Muon
    (W zeroed before the step; O = -W1 / (lr * 0.2*sqrt(max)))."""
    hp = effective_hp(case)
    *_, w1 = _torch_muon_run(case, direction_k=k)
    assert w1 is not None
    return -w1 / (hp.lr * muon_scale(case.rows, case.cols))


def muon_tolerance(case: MuonCase, ref: dict[str, np.ndarray]) -> float:
    """Trajectory atol = 2*max(spread_bf16_vs_fp32, 1e-6) + rtol*max|W|.

    rtol is 1e-3 (fp32) / 8e-3 (bf16: one param ulp). This is a whole-loop
    smoke test (momentum feedback, weight decay, store path), NOT the NS gate:
    with lr=0.02 one Muon step moves W by less than a bf16 ulp at |W|~2, so in
    bf16 a kernel that skips Newton-Schulz or uses the wrong scale stays inside
    this atol. `muon_direction_tolerance` is what pins the update direction.
    """
    rtol = 8e-3 if case.dtype == "bfloat16" else 1e-3
    spread = float(ref["spread"])
    wmax = float(np.max(np.abs(ref["trajectory"])))
    return 2 * max(spread, 1e-6) + rtol * wmax


def muon_direction_inputs(
    case: MuonCase, k: int
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """(m before step k, grad k in storage dtype, fp32 U of step k) under the
    kernel's effective hyperparameters. k >= 1 makes M != g, so a kernel that
    drops the Nesterov term points somewhere else (at k=0 U is just (1+mu)g)."""
    hp = effective_hp(case)
    _, grads = _muon_inputs(case)
    m = torch.zeros(case.rows, case.cols, dtype=torch.float32)
    for G in grads[:k]:
        m = hp.momentum * m + hp.grad_scale * G.float()
    g = hp.grad_scale * grads[k].float()
    m_new = hp.momentum * m + g
    u = g + hp.momentum * m_new if hp.nesterov else m_new
    return m.numpy().copy(), _to_storable(grads[k], case), u  # type: ignore[arg-type]


def direction_metrics(got: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """(cosine similarity, relative Frobenius error ||got-ref||/||ref||), fp64."""
    a = np.asarray(got, dtype=np.float64).ravel()
    b = np.asarray(ref, dtype=np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return cos, float(np.linalg.norm(a - b) / np.linalg.norm(b))


MUON_MIN_COS = 0.995


def muon_direction_tolerance(u: torch.Tensor, ns_steps: int = 5) -> float:
    """Allowed relative-Frobenius error of the kernel's O against bf16 NS.

    Five bf16 NS iterations amplify rounding (a=3.44): this reference sits
    ~0.01-0.015 from fp32 NS, but a correct NS that merely rounds at other
    points (bf16 norm, every elementwise op rounded, as plain torch bf16 code
    does) sits 0.03-0.04 from it (measured on MUON_CASES). So the bound is
    max(0.10, 5 * spread_vs_fp32), ~2.5x above that. Measured wrong kernels
    sit far outside: 4 NS steps ~0.30, no Nesterov ~0.38-0.60, no NS ~0.85,
    transpose not undone ~1.4, a wrong scale rule >= 0.278 (sqrt(min/max)
    or Keller's max(1, r/c)^0.5).
    """
    o16 = newton_schulz(u, ns_steps, torch.bfloat16)
    o32 = newton_schulz(u, ns_steps, torch.float32)
    return max(0.10, 5.0 * direction_metrics(o16.numpy(), o32.numpy())[1])


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
