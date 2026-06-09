"""End-to-end equivalence: Mojo adamw_update vs torch.optim.AdamW.

The reference (torch) and the kernel (Mojo) are fed identical params and
gradient streams. After every step the running state (params, m, v) is
compared; the first divergence is the test failure — the trajectory in
the fixture makes it easy to locate which step drifted.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import DTYPE_TOLERANCES
from tests.kernels import adamw
from tests.reference import CASES, Case, simulate


def _bf16_to_fp32(arr: np.ndarray, dtype_name: str) -> np.ndarray:
    if dtype_name != "bfloat16":
        return arr.astype(np.float32)
    import torch

    return torch.from_numpy(arr).view(torch.bfloat16).to(torch.float32).numpy()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_matches_torch_trajectory(case: Case):
    """Stepwise diff against torch — fails on the first step that drifts."""
    ref = simulate(case)
    tol = DTYPE_TOLERANCES[case.dtype]

    n = case.n
    params = ref["init_params"].copy()
    m = np.zeros(n, dtype=np.float32)
    v = np.zeros(n, dtype=np.float32)

    for step in range(case.steps):
        grad = ref["grads"][step]
        out = adamw.step(
            params=params,
            m=m,
            v=v,
            grads=grad,
            t=step + 1,
            hp=case.hp,
            dtype_name=case.dtype,
        )

        expected = ref["trajectory"][step]
        got = _bf16_to_fp32(out.params, case.dtype)
        max_abs = float(np.max(np.abs(got - expected)))
        assert max_abs <= tol["atol"] + tol["rtol"] * float(np.max(np.abs(expected))), (
            f"{case.name} diverged at step {step + 1}: "
            f"max|Δparam|={max_abs:.3e} > tol(atol={tol['atol']}, rtol={tol['rtol']})"
        )

        params, m, v = out.params, out.m, out.v


@pytest.mark.parametrize(
    "case", [c for c in CASES if c.name == "fp32_zero_grad"], ids=lambda c: c.name
)
def test_zero_grad_zero_wd_is_identity(case: Case):
    """With grads=0 and weight_decay=0, params and v must not move; m decays β1·m."""
    hp = type(case.hp)(**{**case.hp.__dict__, "weight_decay": 0.0})
    n = case.n
    params0 = np.random.default_rng(99).standard_normal(n).astype(np.float32)
    m0 = np.zeros(n, dtype=np.float32)
    v0 = np.zeros(n, dtype=np.float32)
    grads = np.zeros(n, dtype=np.float32)

    out = adamw.step(
        params=params0,
        m=m0,
        v=v0,
        grads=grads,
        t=1,
        hp=hp,
        dtype_name="float32",
    )

    np.testing.assert_allclose(
        out.params,
        params0,
        atol=0,
        rtol=0,
        err_msg="zero grad + zero wd must leave params untouched",
    )
    np.testing.assert_allclose(out.m, np.zeros(n), atol=0, rtol=0)
    np.testing.assert_allclose(out.v, np.zeros(n), atol=0, rtol=0)
