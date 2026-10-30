"""Independent cross-check of our Muon against Keller Jordan's reference package.

Torch-only, CPU, no Mojo kernel. Third source next to our own torch reference
and stock `torch.optim.Muon` (tests/test_muon_crosscheck.py). Library:
`muon-optimizer` (https://github.com/KellerJordan/Muon, import name `muon`).
Facts, quoted from `inspect.getsource(muon)`:

  * zeropower_via_newtonschulz5(G, steps):
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = G.bfloat16()
        if G.size(-2) > G.size(-1): X = X.mT
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)   # bf16 norm, +eps (no clamp)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X
        (transpose back if rows > cols; returns bf16). `steps` has no default
        (we pass 5) and eps is hard-coded 1e-7 -- same as ours. Unlike ours,
        EVERY op is a separate bf16 tensor op (norm rounded to bf16, the
        division, b*A, c*A@A, their sum, a*X, ... each round), whereas our
        reference/kernel round once per GEMM epilogue with an fp32 norm.
  * muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
        momentum.lerp_(grad, 1 - beta)                 # buf = mu*buf + (1-mu)*g
        update = grad.lerp_(momentum, beta) if nesterov else momentum
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
        update *= max(1, update.size(-2) / update.size(-1))**0.5   # inside muon_update
    So buf_keller = (1-mu)*M_ours and update = (1-mu)*U_ours (NS removes the
    constant); momentum is compared after dividing by (1-mu).
  * SingleDeviceMuon.step (per p; no use_muon flags, any 2-D param works):
        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
        p.mul_(1 - group["lr"] * group["weight_decay"])
        p.add_(update.reshape(p.shape), alpha=-group["lr"])
    The buffer is zeros_like(p) (fp32 here, as in the torch.optim harness).

Convention differences (documented, NOT bugs):
  1. Scale rule. Keller scales O by max(1, rows/cols)^0.5 ("spectral norm"
     units); ours is Moonlight's 0.2*sqrt(max(rows, cols)) (AdamW-RMS
     matching). The direction check therefore recovers O with Keller's own
     factor, and the trajectory check hands SingleDeviceMuon
         lr_keller = lr * 0.2*sqrt(max(r,c)) / max(1, r/c)^0.5
     so that the per-step update magnitude equals ours.
  2. Weight decay. Keller decays by (1 - lr_keller*wd); ours by (1 - lr*wd).
     With the rescaled lr_keller these differ (lr_keller/lr = 0.2*sqrt(max)/
     max(1,r/c)^0.5, e.g. 1.6 at 64x64), so the trajectory comparison runs with
     wd = 0 for BOTH sides. Weight decay is covered by the fp32 trajectory gate
     against our own reference and torch.optim.Muon (identical decay rule),
     and by the no-weight-decay mutant in test_muon_equivalence.
  3. Rounding. Keller's all-bf16 NS differs from our single-rounding NS by
     ~0.03-0.04 rel-Frobenius (the same band tests/reference.py documents for
     "other rounding points"), larger than vs torch.optim.Muon (0.009-0.015),
     which keeps GEMM epilogues fused. That is why the NS ceiling below is
     8e-2, not the 2e-2 used for torch.optim.Muon: 2x the measured gap
     (0.027-0.038) is 0.054-0.076.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from muon import muon_update, zeropower_via_newtonschulz5

from tests.reference import (
    MUON_CASES,
    MuonCase,
    direction_metrics,
    effective_hp,
    keller_direction,
    keller_wd0_case,
    muon_direction_inputs,
    newton_schulz,
    simulate_muon,
    simulate_muon_keller,
)
from tests.test_muon_equivalence import (
    _check_direction,
    _mutant_step,
    _recover_direction,
)

NS_SHAPES = [(48, 96), (96, 48), (256, 768), (64, 64)]
# Measured (torch 2.13 CPU, ours vs zeropower_via_newtonschulz5, identical bf16
# input): rel-Frobenius 0.0364 / 0.0380 / 0.0270 / 0.0308; max-abs 0.0176 /
# 0.0195 / 0.0098 / 0.0146. Tolerance = 2x measured.
NS_REL_TOL = {(48, 96): 0.0364, (96, 48): 0.0380, (256, 768): 0.0270, (64, 64): 0.0308}
NS_ABS_TOL = {(48, 96): 0.0176, (96, 48): 0.0195, (256, 768): 0.0098, (64, 64): 0.0146}
HARD_REL_CEILING = 8e-2  # see header, point 3


def _u(shape: tuple[int, int]) -> torch.Tensor:
    g = torch.Generator().manual_seed(2000 + shape[0] + shape[1])
    return torch.randn(shape, generator=g).to(torch.bfloat16).float()


@pytest.mark.parametrize("shape", NS_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_newton_schulz_vs_keller(shape):
    u = _u(shape)
    ours = newton_schulz(u, 5).numpy()
    theirs = zeropower_via_newtonschulz5(u.clone(), steps=5).float().numpy()
    max_abs = float(np.max(np.abs(ours - theirs)))
    _, rel = direction_metrics(ours, theirs)
    msg = f"{shape}: max_abs={max_abs:.4e} rel_frob={rel:.4e}"
    print(msg)
    assert rel <= min(2 * NS_REL_TOL[shape], HARD_REL_CEILING), msg
    assert max_abs <= 2 * NS_ABS_TOL[shape], msg


def test_muon_update_scale_rule():
    """Keller's muon_update folds max(1, r/c)^0.5 into its output."""
    for r, c in [(48, 96), (96, 48), (64, 64)]:
        g = torch.Generator().manual_seed(r + c)
        grad = torch.randn(r, c, generator=g)
        # nesterov at buf=0: buf=(1-mu)g, update = g.lerp(buf, mu) = (1-mu^2)g
        raw = zeropower_via_newtonschulz5(grad * (1 - 0.95**2), steps=5).float()
        got = muon_update(grad.clone(), torch.zeros(r, c), beta=0.95).float()
        # NS is scale-invariant up to bf16 rounding, so compare directions.
        cos, _ = direction_metrics(got.numpy(), raw.numpy())
        assert cos >= 0.99
        ratio = float(got.norm() / raw.norm())
        assert ratio == pytest.approx(max(1.0, r / c) ** 0.5, rel=2e-2)


# 2x measured max|ours - keller| over the wd=0 trajectory (rescaled lr). fp32
# measured 1.34e-3 / 1.71e-3 / 1.57e-3 / 1.27e-3; bf16 = 1 ulp (floored at 2 ulp
# below, since a one-step bf16 flip is rounding, not a bug).
TRAJ_TOL = {
    "muon_fp32_64x64": 2.7e-3,
    "muon_fp32_48x96": 3.5e-3,
    "muon_fp32_96x48": 3.2e-3,
    "muon_fp32_256x768": 2.6e-3,
}
MOM_TOL = 1e-5  # measured ~1e-6 after rescaling by 1/(1-mu)


@pytest.mark.parametrize("case", MUON_CASES, ids=lambda c: c.name)
def test_trajectory_vs_keller(case: MuonCase):
    case0 = keller_wd0_case(case)
    ours = simulate_muon(case0)
    theirs = simulate_muon_keller(case)
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
    assert float(per_step.max()) <= max(TRAJ_TOL.get(case.name, 0.0), 2 * ulp), msg
    assert float(per_step.max()) <= 0.05, msg  # plausibility ceiling
    assert m_err <= MOM_TOL * (1 + float(np.max(np.abs(ours["momentum"])))), msg


# 2x worst measured rel-Frobenius over k in {0,1}; cos was >= 0.9992 everywhere.
DIR_REL_TOL = {
    "muon_fp32_64x64": 0.0695,
    "muon_fp32_48x96": 0.0795,
    "muon_fp32_96x48": 0.0719,
    "muon_fp32_256x768": 0.0572,
    "muon_bf16_64x64": 0.0631,
    "muon_bf16_96x48": 0.0742,
    "muon_bf16_256x768": 0.0541,
}


@pytest.mark.parametrize("k", [0, 1])
@pytest.mark.parametrize("case", MUON_CASES, ids=lambda c: c.name)
def test_direction_vs_keller(case: MuonCase, k: int):
    """One SingleDeviceMuon step from W=0, wd=0; O recovered with Keller's own
    scale (convention difference 1), compared with our NS of the same U."""
    _, _, u = muon_direction_inputs(case, k)
    o_ours = newton_schulz(u, case.hp.ns_steps).numpy()
    o_keller = keller_direction(case, k)
    cos, rel = direction_metrics(o_keller, o_ours)
    msg = f"{case.name} k={k}: cos={cos:.5f} rel_frob={rel:.4f}"
    print(msg)
    assert np.all(np.isfinite(o_keller)), msg
    assert cos >= 0.999, msg
    assert rel <= DIR_REL_TOL[case.name], msg


def test_hp_rounding_is_shared():
    c = next(c for c in MUON_CASES if c.dtype == "bfloat16")
    assert effective_hp(c).momentum == 0.94921875


@pytest.mark.parametrize("case", MUON_CASES, ids=lambda c: c.name)
def test_keller_direction_gate_rejects_mutants(case: MuonCase):
    """The Keller-sourced direction gate (the one test_muon_equivalence applies
    with expected_source="keller") accepts a correct kernel with other rounding
    points and rejects skip-NS, 4-step NS, no-Nesterov, transpose-not-undone."""
    _, _, u = muon_direction_inputs(case, 1)
    o_ref = keller_direction(case, 1)
    _check_direction(
        _recover_direction(_mutant_step(case, 1, "correct_other_rounding"), case),
        u,
        case,
        "correct kernel with other rounding points",
        o_ref,
    )
    mutants = ["skip_ns", "ns4", "no_nesterov"]
    if case.rows > case.cols:
        mutants.append("transpose_not_undone")
    for variant in mutants:
        o_got = _recover_direction(_mutant_step(case, 1, variant), case)
        with pytest.raises(AssertionError):
            _check_direction(o_got, u, case, variant, o_ref)
