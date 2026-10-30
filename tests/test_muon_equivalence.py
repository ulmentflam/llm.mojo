"""Muon equivalence: Mojo `muon_update` vs the explicit torch reference.

Two gates, because one is not enough:
  * `test_update_direction` pins the orthogonalised update O itself. One step
    from W=0 (so W1 = -lr*scale*O and weight decay drops out) recovers the
    kernel's O, compared by cosine and relative Frobenius error against the
    bf16-NS reference (`tests.reference.muon_direction_tolerance`).
  * `test_matches_torch_trajectory` runs the multi-step loop (momentum
    feedback, weight decay, bf16 store). Its atol cannot see NS in bf16: one
    step at lr=0.02 is below a bf16 ulp of |W|~2.
`test_tolerance_not_absurd` proves both gates reject known-wrong kernels
(skipped NS, wrong scale, transpose not undone, no Nesterov, no weight decay)
and accept a correct one that rounds at different points.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests._dtypes import from_storage, to_storage
from tests.kernels import muon
from tests.reference import (
    MUON_CASES,
    MUON_MIN_COS,
    NS_A,
    NS_B,
    NS_C,
    MuonCase,
    MuonParams,
    direction_metrics,
    effective_hp,
    muon_direction_inputs,
    muon_direction_tolerance,
    muon_scale,
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


def _recover_direction(w1: np.ndarray, case: MuonCase) -> np.ndarray:
    """O from one step taken at W=0: W1 = -lr * scale * O (wd*W vanishes)."""
    hp = effective_hp(case)
    return -w1 / (hp.lr * muon_scale(case.rows, case.cols))


def _check_direction(o_got: np.ndarray, u: torch.Tensor, case: MuonCase, what: str):
    o_ref = newton_schulz(u, case.hp.ns_steps, torch.bfloat16).numpy()
    cos, rel = direction_metrics(o_got, o_ref)
    rel_tol = muon_direction_tolerance(u, case.hp.ns_steps)
    assert np.all(np.isfinite(o_got)), f"{what}: non-finite update"
    assert cos >= MUON_MIN_COS and rel <= rel_tol, (
        f"{what}: cos={cos:.5f} (min {MUON_MIN_COS}) rel_frob={rel:.4f} (max {rel_tol:.4f})"
    )


@pytest.mark.parametrize("k", [0, 1])
@pytest.mark.parametrize("case", MUON_CASES, ids=lambda c: c.name)
def test_update_direction(case: MuonCase, k: int):
    """The kernel's orthogonalised update matches bf16 Newton-Schulz."""
    m_prev, grad, u = muon_direction_inputs(case, k)
    zeros = np.zeros((case.rows, case.cols), dtype=np.float32)
    out = muon.step(
        params=to_storage(zeros, case.dtype),
        m=m_prev,
        grads=grad,
        t=k + 1,
        hp=case.hp,
        dtype_name=case.dtype,
    )
    o_got = _recover_direction(from_storage(out.params, case.dtype), case)
    _check_direction(o_got, u, case, f"{case.name} step {k + 1}")


# --- gate self-test -------------------------------------------------------


def _ns_torch_bf16(u: torch.Tensor, steps: int) -> torch.Tensor:
    """A *correct* NS that rounds where plain torch bf16 code does (norm
    rounded to bf16, every elementwise op rounded). The gate must accept it."""
    rows, cols = u.shape
    x = u.to(torch.bfloat16)
    x = x / (x.float().norm() + 1e-7).to(torch.bfloat16)
    if rows > cols:
        x = x.T
    for _ in range(steps):
        a = x @ x.T
        b = NS_B * a + NS_C * (a @ a)
        x = NS_A * x + b @ x
    if rows > cols:
        x = x.T
    return x.float()


def _mutant_step(case: MuonCase, k: int, variant: str) -> np.ndarray:
    """W1 a (possibly wrong) kernel would write for one step from W=0."""
    hp = effective_hp(case)
    m_prev, _, u = muon_direction_inputs(case, k)
    r, c = case.rows, case.cols
    scale = muon_scale(r, c)
    o = None
    if variant == "correct_other_rounding":
        o = _ns_torch_bf16(u, 5)
    elif variant == "skip_ns":
        x = u.to(torch.bfloat16).float()
        o = x / (x.norm() + 1e-7)
    elif variant == "ns4":
        o = newton_schulz(u, 4)
    elif variant == "transpose_not_undone":
        # NS on X^T, result written back row-major without transposing.
        o = newton_schulz(u, 5).T.contiguous().reshape(r, c)
    elif variant == "no_nesterov":
        # U = M instead of g + mu*M. Invert u = (1+mu)g + mu^2 M_prev for g.
        mu, mp = hp.momentum, torch.from_numpy(m_prev)
        g = (u - mu * mu * mp) / (1 + mu)
        o = newton_schulz(mu * mp + g, 5)
    elif variant == "scale_min":
        o = newton_schulz(u, 5)
        scale = 0.2 * min(r, c) ** 0.5
    elif variant == "scale_keller":
        o = newton_schulz(u, 5)
        scale = max(1.0, r / c) ** 0.5
    else:
        raise ValueError(variant)
    w1 = (-hp.lr * scale * o).to(case.torch_dtype).float().numpy()
    return w1


@pytest.mark.parametrize("case", MUON_CASES, ids=lambda c: c.name)
def test_tolerance_not_absurd(case: MuonCase):
    """Both gates reject known-wrong kernels and accept a correct one.

    A gate that cannot fail on these is decoration (a flat atol in this repo
    once buried three real backward bugs)."""
    _, _, u = muon_direction_inputs(case, 1)
    _check_direction(
        _recover_direction(_mutant_step(case, 1, "correct_other_rounding"), case),
        u,
        case,
        "correct kernel with other rounding points",
    )
    mutants = ["skip_ns", "ns4", "no_nesterov", "scale_keller"]
    if case.rows > case.cols:
        mutants.append("transpose_not_undone")
    if case.rows != case.cols:
        mutants.append("scale_min")
    for variant in mutants:
        o_got = _recover_direction(_mutant_step(case, 1, variant), case)
        with pytest.raises(AssertionError):
            _check_direction(o_got, u, case, variant)

    # Trajectory gate: dropping weight decay must fail it in fp32. (In bf16
    # the per-step change is sub-ulp, which is why the direction gate exists.)
    if case.dtype == "float32":
        ref = simulate_muon(case)
        atol = muon_tolerance(case, ref)
        hp = effective_hp(case)
        w = torch.from_numpy(ref["init_params"].copy())
        m = torch.zeros(case.rows, case.cols)
        worst = 0.0
        for step in range(case.steps):
            g = hp.grad_scale * torch.from_numpy(ref["grads"][step])
            m = hp.momentum * m + g
            o = newton_schulz(g + hp.momentum * m, hp.ns_steps)
            w = w - hp.lr * muon_scale(case.rows, case.cols) * o
            worst = max(
                worst, float(np.max(np.abs(w.numpy() - ref["trajectory"][step])))
            )
        assert worst > atol, (
            f"no-weight-decay kernel passes: {worst:.3e} <= atol {atol:.3e}"
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
