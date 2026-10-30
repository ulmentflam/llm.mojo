"""Independent cross-check of our Muon against stock `torch.optim.Muon`.

Torch-only, CPU, no Mojo kernel. torch 2.13.0 facts (quoted from
torch/optim/_muon.py, nothing else is used):

  * EPS = 1e-7, (a, b, c) = (3.4445, -4.7750, 2.0315), ns_steps = 5 -- same as ours.
  * Momentum is the LERP form:
        buf.lerp_(grad, 1 - momentum)                 # buf = mu*buf + (1-mu)*g
        update = grad.lerp(buf, momentum) if nesterov else buf
    i.e. buf_torch = (1-mu) * M_ours and update_torch = (1-mu) * U_ours. (The
    docstring writes B = mu*B + g, but the code is lerp.) The (1-mu) factor is
    a constant that Newton-Schulz's Frobenius normalisation removes, so the
    direction is equivalent; only the bf16 rounding of differently scaled
    inputs and the eps clamp differ (negligible). We compare momentum after
    dividing torch's buffer by (1-mu).
  * Newton-Schulz (_zeropower_via_newtonschulz):
        ortho_grad = grad.bfloat16(); if rows > cols: ortho_grad = ortho_grad.T
        ortho_grad.div_(ortho_grad.norm().clamp(min=eps))      # norm in bf16, clamp (not +eps)
        per step: G = X @ X.T; GU = torch.addmm(G, G, G, beta=b, alpha=c)
                  X = torch.addmm(X, GU, X, beta=a)            # all bf16 tensors
    Same op sequence as ours; rounding differs: torch's norm and each addmm
    output are bf16 with bf16 accumulate-epilogue, ours keeps an fp32 norm and
    fp32 epilogues (one bf16 rounding per GEMM). Output stays bf16.
  * adjust_lr_fn in {None/"original", "match_rms_adamw"}; "original" is
    sqrt(max(1, A/B)) (Keller); "match_rms_adamw" is 0.2*sqrt(max(A, B)) ==
    our muon_scale. A, B = param.shape[:2] (rows, cols).
  * Weight decay: param.mul_(1 - lr*wd) with the UNADJUSTED lr, then
    param.add_(update, alpha=-adjusted_lr). Ours: W -= lr*(scale*O + wd*W)
    = W*(1 - lr*wd) - lr*scale*O. Identical (no cross term, since both apply
    the decay to the old W).
  * Constraints: params must be 2D; no grad_scale knob (we pre-multiply grads);
    torch computes in the param dtype (bf16 params would round param twice per
    step), so the harness keeps an fp32 param holding storage-dtype values and
    rounds to storage after each step, matching the kernel's single round.

Residual non-equivalences (documented, not hidden): bf16 norm vs fp32 norm and
bf16 epilogues in NS. They show up as the NS gap measured below.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.optim import _muon as tmuon

from tests.reference import (
    MUON_CASES,
    MuonCase,
    direction_metrics,
    effective_hp,
    muon_scale,
    muon_direction_inputs,
    newton_schulz,
    simulate_muon,
    simulate_muon_torch_optim,
    torch_optim_direction,
)

NS_SHAPES = [(48, 96), (96, 48), (256, 768), (64, 64)]
# Measured on this machine (torch 2.13.0 CPU): max-abs / rel-Frobenius of
# ours vs torch's NS, see test messages. Tolerance = 2x measured, <= 2e-2 rel.
# Measured rel-Frobenius 0.0120 / 0.0124 / 0.0153 / 0.0087; max-abs 5.9e-3 /
# 8.8e-3 / 4.4e-3 / 7.8e-3. 2x rel would exceed the 2e-2 ceiling for 256x768,
# so the ceiling binds there. The gap is rounding-noise amplified by a=3.44
# over 5 steps (bf16 norm + bf16 epilogues in torch vs fp32 in ours), the same
# 0.01-0.015 band reference.muon_direction_tolerance documents vs fp32 NS.
NS_REL_TOL = {(48, 96): 0.0120, (96, 48): 0.0124, (256, 768): 0.0153, (64, 64): 0.0087}
NS_ABS_TOL = {(48, 96): 5.9e-3, (96, 48): 8.8e-3, (256, 768): 4.4e-3, (64, 64): 7.8e-3}
HARD_REL_CEILING = 2e-2


def _torch_ns(u: torch.Tensor) -> torch.Tensor:
    return tmuon._zeropower_via_newtonschulz(
        u.clone(), (tmuon.DEFAULT_A, tmuon.DEFAULT_B, tmuon.DEFAULT_C), 5, tmuon.EPS
    ).float()


def _u(shape: tuple[int, int]) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + shape[0] + shape[1])
    return torch.randn(shape, generator=g).to(torch.bfloat16).float()


def test_constants_match():
    assert (tmuon.DEFAULT_A, tmuon.DEFAULT_B, tmuon.DEFAULT_C) == (
        3.4445,
        -4.7750,
        2.0315,
    )
    assert tmuon.EPS == 1e-7 and tmuon.DEFAULT_NS_STEPS == 5


@pytest.mark.parametrize("shape", NS_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_newton_schulz_vs_torch(shape):
    u = _u(shape)
    ours = newton_schulz(u, 5).numpy()
    theirs = _torch_ns(u).numpy()
    max_abs = float(np.max(np.abs(ours - theirs)))
    _, rel = direction_metrics(ours, theirs)
    msg = f"{shape}: max_abs={max_abs:.4e} rel_frob={rel:.4e}"
    print(msg)
    assert rel <= min(2 * NS_REL_TOL[shape], HARD_REL_CEILING), msg
    assert max_abs <= 2 * NS_ABS_TOL[shape], msg


def test_adjust_lr_is_our_scale():
    for r, c in [(48, 96), (96, 48), (64, 64), (256, 768)]:
        got = tmuon._adjust_lr(1.0, "match_rms_adamw", torch.Size((r, c)))
        assert got == pytest.approx(muon_scale(r, c), rel=1e-12)


# 2x measured max|ours - torch| over the trajectory. fp32: 7.9e-4/7.9e-4/8.0e-4/
# 7.0e-4; bf16: 1 ulp (7.8e-3 / 1.56e-2 / 1.17e-2 at |W|~1-4), floored below at
# 2 ulp since a one-step bf16 flip is rounding, not a bug.
TRAJ_TOL = {
    "muon_fp32_64x64": 1.6e-3,
    "muon_fp32_48x96": 1.6e-3,
    "muon_fp32_96x48": 1.6e-3,
    "muon_fp32_256x768": 1.4e-3,
    "muon_bf16_64x64": 1.56e-2,
    "muon_bf16_96x48": 3.1e-2,
    "muon_bf16_256x768": 2.3e-2,
}
MOM_TOL = 1e-5  # momentum agrees to ~1e-6 after rescaling


@pytest.mark.parametrize("case", MUON_CASES, ids=lambda c: c.name)
def test_trajectory_vs_torch_optim(case: MuonCase):
    ours = simulate_muon(case)
    theirs = simulate_muon_torch_optim(case)
    # simulate_muon_torch_optim's "spread" folds the gap in; recompute it raw.
    d = np.abs(ours["trajectory"] - theirs["trajectory"])
    per_step = d.reshape(case.steps, -1).max(axis=1)
    m_err = float(np.max(np.abs(ours["momentum"] - theirs["momentum"])))
    wmax = float(np.max(np.abs(ours["trajectory"])))
    msg = (
        f"{case.name}: traj max|d| per step={per_step.tolist()} "
        f"momentum max|d|={m_err:.3e} wmax={wmax:.3f}"
    )
    print(msg)
    ulp = 2.0**-7 * 2 ** np.floor(np.log2(wmax)) if case.dtype == "bfloat16" else 0.0
    tol = TRAJ_TOL.get(case.name, 0.0)
    assert float(per_step.max()) <= max(tol, 2 * ulp), msg
    assert float(per_step.max()) <= 0.05, msg  # plausibility ceiling
    assert m_err <= MOM_TOL * (1 + float(np.max(np.abs(ours["momentum"])))), msg


# 2x worst measured rel-Frobenius over k in {0,1} (cos was >= 0.9997 everywhere).
DIR_REL_TOL = {
    "muon_fp32_64x64": 0.0365,
    "muon_fp32_48x96": 0.0398,
    "muon_fp32_96x48": 0.0402,
    "muon_fp32_256x768": 0.0309,
    "muon_bf16_64x64": 0.0368,
    "muon_bf16_96x48": 0.0350,
    "muon_bf16_256x768": 0.0298,
}


@pytest.mark.parametrize("k", [0, 1])
@pytest.mark.parametrize("case", MUON_CASES, ids=lambda c: c.name)
def test_direction_vs_torch_optim(case: MuonCase, k: int):
    _, _, u = muon_direction_inputs(case, k)
    o_ours = newton_schulz(u, case.hp.ns_steps).numpy()
    o_torch = torch_optim_direction(case, k)
    cos, rel = direction_metrics(o_torch, o_ours)
    msg = f"{case.name} k={k}: cos={cos:.5f} rel_frob={rel:.4f}"
    print(msg)
    assert np.all(np.isfinite(o_torch)), msg
    assert cos >= 0.999, msg
    assert rel <= DIR_REL_TOL.get(case.name, 0.0), msg


def test_hp_rounding_is_shared():
    """Both sides see the dtype-rounded hyperparameters (bf16 mu=0.94921875)."""
    c = next(c for c in MUON_CASES if c.dtype == "bfloat16")
    assert effective_hp(c).momentum == 0.94921875
