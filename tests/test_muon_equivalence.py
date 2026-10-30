"""Muon equivalence: Mojo `muon_update` vs the explicit torch reference.

Tolerance is derived per case from the bf16-vs-fp32 Newton-Schulz spread
(see `tests.reference.muon_tolerance`), not from the AdamW tolerances.
Measured spreads (max|bf16-NS traj - fp32-NS traj|) are recorded in
`test_tolerance_not_absurd`'s bound below.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests._dtypes import from_storage
from tests.kernels import muon
from tests.reference import (
    MUON_CASES,
    MuonCase,
    MuonParams,
    muon_tolerance,
    newton_schulz,
    simulate_muon,
)


@pytest.mark.parametrize("case", MUON_CASES, ids=lambda c: c.name)
def test_matches_torch_trajectory(case: MuonCase):
    """Stepwise diff against torch — fails on the first step that drifts."""
    ref = simulate_muon(case)
    atol = muon_tolerance(case, ref)

    params = ref["init_params"].copy()
    m = np.zeros((case.rows, case.cols), dtype=np.float32)
    for step in range(case.steps):
        out = muon.step(
            params=params,
            m=m,
            grads=ref["grads"][step],
            t=step + 1,
            hp=case.hp,
            dtype_name=case.dtype,
        )
        got = from_storage(out.params, case.dtype)
        max_abs = float(np.max(np.abs(got - ref["trajectory"][step])))
        assert max_abs <= atol, (
            f"{case.name} params diverged at step {step + 1}: "
            f"max|Δ|={max_abs:.3e} > atol={atol:.3e}"
        )
        m_err = float(np.max(np.abs(out.m - ref["momentum"][step])))
        assert m_err <= 1e-5 * (1 + float(np.max(np.abs(ref["momentum"][step])))), (
            f"{case.name} momentum diverged at step {step + 1}: max|Δ|={m_err:.3e}"
        )
        params, m = out.params, out.m


@pytest.mark.parametrize("case", MUON_CASES, ids=lambda c: c.name)
def test_tolerance_not_absurd(case: MuonCase):
    """The derived atol must stay small relative to the per-step update size."""
    ref = simulate_muon(case)
    atol = muon_tolerance(case, ref)
    step_size = case.hp.lr * 0.2 * max(case.rows, case.cols) ** 0.5 * 0.5
    assert atol < 0.5 * step_size * case.steps + 1e-2, (
        f"{case.name}: atol={atol:.3e}, spread={float(ref['spread']):.3e}"
    )


@pytest.mark.parametrize("shape", [(48, 96), (96, 48), (64, 192)])
def test_orthogonality_of_reference(shape):
    """fp32 Newton-Schulz output has singular values roughly in [0.6, 1.3].

    Rectangular only: a square gaussian has singular values near 0, which 5
    NS steps cannot lift into the band."""
    g = torch.Generator().manual_seed(7)
    u = torch.randn(*shape, generator=g)
    o = newton_schulz(u, 5, torch.float32)
    s = torch.linalg.svdvals(o)
    assert float(s.min()) >= 0.6 and float(s.max()) <= 1.3, (
        f"singular values [{float(s.min()):.3f}, {float(s.max()):.3f}]"
    )


@pytest.mark.parametrize("shape", [(64, 64), (48, 96)])
def test_zero_grad_zero_wd_is_identity(shape):
    """grads=0, wd=0, m=0: params and momentum must not move."""
    hp = MuonParams(weight_decay=0.0)
    rng = np.random.default_rng(99)
    params0 = rng.standard_normal(shape).astype(np.float32)
    out = muon.step(
        params=params0,
        m=np.zeros(shape, dtype=np.float32),
        grads=np.zeros(shape, dtype=np.float32),
        t=1,
        hp=hp,
        dtype_name="float32",
    )
    np.testing.assert_allclose(out.params, params0, atol=0, rtol=0)
    np.testing.assert_allclose(out.m, np.zeros(shape), atol=0, rtol=0)
