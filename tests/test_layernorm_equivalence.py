"""Equivalence: Mojo layernorm vs PyTorch and vs Modular's kernel.

The forward is checked against torch and Modular's production layer_norm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tests._dtypes import DTYPE_TOLERANCES, TORCH_DTYPES, from_storage, to_storage
from tests.kernels import layernorm


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int  # B
    seq_len: int  # T
    channels: int  # C
    dtype: str  # "float32" | "bfloat16" | "float16"
    epsilon: float
    seed: int


CASES: tuple[Case, ...] = (
    Case(
        "fp32_small",
        batch_size=4,
        seq_len=8,
        channels=64,
        dtype="float32",
        epsilon=1e-5,
        seed=0,
    ),
    Case(
        "fp32_odd_size",
        batch_size=2,
        seq_len=7,
        channels=767,
        dtype="float32",
        epsilon=1e-5,
        seed=1,
    ),
    Case(
        "fp32_gpt2_shape",
        batch_size=1,
        seq_len=4,
        channels=768,
        dtype="float32",
        epsilon=1e-5,
        seed=2,
    ),
    Case(
        "fp32_gpt2_full",
        batch_size=2,
        seq_len=1024,
        channels=768,
        dtype="float32",
        epsilon=1e-5,
        seed=4,
    ),
    Case(
        "bf16_small",
        batch_size=4,
        seq_len=8,
        channels=64,
        dtype="bfloat16",
        epsilon=1e-5,
        seed=3,
    ),
)


def _make_inputs(case: Case) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = torch.Generator().manual_seed(case.seed)
    bt = case.batch_size * case.seq_len
    x = torch.randn(bt, case.channels, generator=g) * 2.0
    gamma = torch.randn(case.channels, generator=g) * 0.1 + 1.0
    beta = torch.randn(case.channels, generator=g) * 0.1

    # Round-trip through the kernel dtype
    x = x.to(TORCH_DTYPES[case.dtype]).to(torch.float32)
    gamma = gamma.to(TORCH_DTYPES[case.dtype]).to(torch.float32)
    beta = beta.to(TORCH_DTYPES[case.dtype]).to(torch.float32)

    return x.numpy(), gamma.numpy(), beta.numpy()


def _reference(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float
) -> np.ndarray:
    x_t = torch.from_numpy(x)
    g_t = torch.from_numpy(gamma)
    b_t = torch.from_numpy(beta)
    return torch.nn.functional.layer_norm(x_t, (x.shape[-1],), g_t, b_t, eps).numpy()


def _run_kernel(case: Case, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray):
    bt = case.batch_size * case.seq_len
    mean = np.zeros(bt, dtype=np.float32)
    rstd = np.zeros(bt, dtype=np.float32)

    return layernorm.forward(
        x=to_storage(x.reshape(bt, case.channels), case.dtype),
        gamma=to_storage(gamma, case.dtype),
        beta=to_storage(beta, case.dtype),
        mean=mean,
        rstd=rstd,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        channels=case.channels,
        epsilon=case.epsilon,
        dtype_name=case.dtype,
    )


def _run_kernel_bwd(
    case: Case,
    d_out: np.ndarray,
    x: np.ndarray,
    gamma: np.ndarray,
    mean: np.ndarray,
    rstd: np.ndarray,
    d_gamma: np.ndarray | None = None,
    d_beta: np.ndarray | None = None,
):
    bt = case.batch_size * case.seq_len
    d_x = np.zeros_like(x)
    # d_gamma/d_beta accumulate (the kernel adds into them), so callers can
    # pass carried-over buffers to exercise that; default to a fresh zero.
    if d_gamma is None:
        d_gamma = np.zeros(case.channels, dtype=np.float32)
    if d_beta is None:
        d_beta = np.zeros(case.channels, dtype=np.float32)

    return layernorm.backward(
        d_output=to_storage(d_out.reshape(bt, case.channels), case.dtype),
        x=to_storage(x.reshape(bt, case.channels), case.dtype),
        gamma=to_storage(gamma, case.dtype),
        mean=mean,
        rstd=rstd,
        d_x=to_storage(d_x.reshape(bt, case.channels), case.dtype),
        d_gamma=d_gamma,
        d_beta=d_beta,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        channels=case.channels,
        dtype_name=case.dtype,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_forward_matches_torch(case: Case):
    x, gamma, beta = _make_inputs(case)
    expected = _reference(x, gamma, beta, case.epsilon)

    got, _, _ = _run_kernel(case, x, gamma, beta)

    bt = case.batch_size * case.seq_len
    got_v = from_storage(got, case.dtype).reshape(bt, case.channels)

    tol = DTYPE_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        got_v,
        expected,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: forward output diverged from torch",
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_forward_matches_modular(case: Case):
    x, gamma, beta = _make_inputs(case)
    bt = case.batch_size * case.seq_len

    theirs = layernorm.modular_forward(
        to_storage(x, case.dtype).reshape(bt, case.channels),
        to_storage(gamma, case.dtype),
        to_storage(beta, case.dtype),
        case.epsilon,
        case.dtype,
    )

    got, _, _ = _run_kernel(case, x, gamma, beta)
    got_v = from_storage(got, case.dtype).reshape(bt, case.channels)

    tol = DTYPE_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        got_v,
        from_storage(theirs, case.dtype).reshape(bt, case.channels),
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: forward output diverged from Modular's layer_norm",
    )


def test_mean_rstd_correctness():
    """Verify that the intermediate mean and rstd match torch's internal stats."""
    case = next(c for c in CASES if c.name == "fp32_small")
    x, gamma, beta = _make_inputs(case)

    # Torch reference for stats
    x_t = torch.from_numpy(x)
    mean_expected = x_t.mean(dim=-1).numpy()
    var_expected = x_t.var(dim=-1, unbiased=False).numpy()
    rstd_expected = 1.0 / np.sqrt(var_expected + case.epsilon)

    _, got_mean, got_rstd = _run_kernel(case, x, gamma, beta)

    np.testing.assert_allclose(
        got_mean, mean_expected, atol=1e-6, err_msg="mean diverged"
    )
    np.testing.assert_allclose(
        got_rstd, rstd_expected, atol=1e-6, err_msg="rstd diverged"
    )


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_backward_matches_torch(case: Case):
    x, gamma, beta = _make_inputs(case)

    x_t = torch.from_numpy(x).requires_grad_(True)
    g_t = torch.from_numpy(gamma).requires_grad_(True)
    b_t = torch.from_numpy(beta).requires_grad_(True)

    y_t = torch.nn.functional.layer_norm(x_t, (case.channels,), g_t, b_t, case.epsilon)

    # Generate random gradient for y
    g = torch.Generator().manual_seed(case.seed + 100)
    dy_t = torch.randn_like(y_t, generator=g)

    y_t.backward(dy_t)

    # Get reference stats for the kernel
    with torch.no_grad():
        mean_ref = x_t.mean(dim=-1).numpy()
        var_ref = x_t.var(dim=-1, unbiased=False).numpy()
        rstd_ref = 1.0 / np.sqrt(var_ref + case.epsilon)

    # Run Mojo kernel
    got_dx, got_dgamma, got_dbeta = _run_kernel_bwd(
        case, dy_t.detach().numpy(), x, gamma, mean_ref, rstd_ref
    )

    bt = case.batch_size * case.seq_len
    got_dx_v = from_storage(got_dx, case.dtype).reshape(bt, case.channels)

    tol = DTYPE_TOLERANCES[case.dtype]

    # Check dX
    assert x_t.grad is not None
    np.testing.assert_allclose(
        got_dx_v,
        x_t.grad.numpy(),
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: dX diverged",
    )

    # Check dGamma
    assert g_t.grad is not None
    np.testing.assert_allclose(
        got_dgamma,
        g_t.grad.numpy(),
        atol=tol["atol"] * 100,
        rtol=tol["rtol"] * 10,
        err_msg=f"{case.name}: dGamma diverged",
    )

    # Check dBeta
    assert b_t.grad is not None
    np.testing.assert_allclose(
        got_dbeta,
        b_t.grad.numpy(),
        atol=tol["atol"] * 100,
        rtol=tol["rtol"] * 10,
        err_msg=f"{case.name}: dBeta diverged",
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_backward_accumulates(case: Case):
    """The parameter grads add into d_gamma/d_beta (the reduction replaced the
    old atomic accumulate-into-buffer), so a second pass carrying the prior
    grads forward doubles them. dX is overwritten, so it stays unchanged."""
    x, gamma, beta = _make_inputs(case)

    g = torch.Generator().manual_seed(case.seed + 100)
    dy = torch.randn(case.batch_size * case.seq_len, case.channels, generator=g)
    dy = dy.to(TORCH_DTYPES[case.dtype]).to(torch.float32).numpy()

    x_t = torch.from_numpy(x)
    mean_ref = x_t.mean(dim=-1).numpy()
    var_ref = x_t.var(dim=-1, unbiased=False).numpy()
    rstd_ref = 1.0 / np.sqrt(var_ref + case.epsilon)

    dx1, dgamma1, dbeta1 = _run_kernel_bwd(case, dy, x, gamma, mean_ref, rstd_ref)

    # Second pass carrying the prior parameter grads forward.
    dx2, dgamma2, dbeta2 = _run_kernel_bwd(
        case,
        dy,
        x,
        gamma,
        mean_ref,
        rstd_ref,
        d_gamma=dgamma1.copy(),
        d_beta=dbeta1.copy(),
    )

    bt = case.batch_size * case.seq_len
    dx1_v = from_storage(dx1, case.dtype).reshape(bt, case.channels)
    dx2_v = from_storage(dx2, case.dtype).reshape(bt, case.channels)

    tol = DTYPE_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        dx2_v, dx1_v, atol=0, rtol=0, err_msg=f"{case.name}: dX must overwrite"
    )
    np.testing.assert_allclose(
        dgamma2,
        2.0 * dgamma1,
        atol=tol["atol"] * 100,
        rtol=tol["rtol"] * 10,
        err_msg=f"{case.name}: dGamma did not accumulate",
    )
    np.testing.assert_allclose(
        dbeta2,
        2.0 * dbeta1,
        atol=tol["atol"] * 100,
        rtol=tol["rtol"] * 10,
        err_msg=f"{case.name}: dBeta did not accumulate",
    )


# ---------------------------------------------------------------------------
# Fused Residual Forward Tests
# ---------------------------------------------------------------------------


def _make_fused_inputs(
    case: Case,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = torch.Generator().manual_seed(case.seed)
    bt = case.batch_size * case.seq_len
    x1 = torch.randn(bt, case.channels, generator=g) * 2.0
    x2 = torch.randn(bt, case.channels, generator=g) * 1.5
    gamma = torch.randn(case.channels, generator=g) * 0.1 + 1.0
    beta = torch.randn(case.channels, generator=g) * 0.1

    # Round-trip through the kernel dtype
    x1 = x1.to(TORCH_DTYPES[case.dtype]).to(torch.float32)
    x2 = x2.to(TORCH_DTYPES[case.dtype]).to(torch.float32)
    gamma = gamma.to(TORCH_DTYPES[case.dtype]).to(torch.float32)
    beta = beta.to(TORCH_DTYPES[case.dtype]).to(torch.float32)

    return x1.numpy(), x2.numpy(), gamma.numpy(), beta.numpy()


def _run_fused_residual_kernel(
    case: Case, x1: np.ndarray, x2: np.ndarray, gamma: np.ndarray, beta: np.ndarray
):
    bt = case.batch_size * case.seq_len
    mean = np.zeros(bt, dtype=np.float32)
    rstd = np.zeros(bt, dtype=np.float32)

    return layernorm.layernorm_fused_residual_forward(
        x1=to_storage(x1.reshape(bt, case.channels), case.dtype),
        x2=to_storage(x2.reshape(bt, case.channels), case.dtype),
        gamma=to_storage(gamma, case.dtype),
        beta=to_storage(beta, case.dtype),
        mean=mean,
        rstd=rstd,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        channels=case.channels,
        epsilon=case.epsilon,
        dtype_name=case.dtype,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_layernorm_fused_residual_forward(case: Case):
    x1, x2, gamma, beta = _make_fused_inputs(case)
    residual_expected = x1 + x2
    # Round-trip the addition through the kernel dtype since the kernel does the addition in kernel dtype
    residual_expected_torch = (
        torch.from_numpy(residual_expected)
        .to(TORCH_DTYPES[case.dtype])
        .to(torch.float32)
    )
    normed_expected = _reference(
        residual_expected_torch.numpy(), gamma, beta, case.epsilon
    )

    # Reference mean and rstd of residual_expected
    mean_expected = residual_expected_torch.mean(dim=-1).numpy()
    var_expected = residual_expected_torch.var(dim=-1, unbiased=False).numpy()
    rstd_expected = 1.0 / np.sqrt(var_expected + case.epsilon)

    # Got
    got_res, got_normed, got_mean, got_rstd = _run_fused_residual_kernel(
        case, x1, x2, gamma, beta
    )

    bt = case.batch_size * case.seq_len
    got_res_v = from_storage(got_res, case.dtype).reshape(bt, case.channels)
    got_normed_v = from_storage(got_normed, case.dtype).reshape(bt, case.channels)

    tol = DTYPE_TOLERANCES[case.dtype]
    # Check residual
    np.testing.assert_allclose(
        got_res_v,
        residual_expected_torch.numpy(),
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: fused residual output diverged",
    )

    # Check normed
    np.testing.assert_allclose(
        got_normed_v,
        normed_expected,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: fused normed output diverged",
    )

    # Check mean and rstd
    np.testing.assert_allclose(
        got_mean,
        mean_expected,
        atol=1e-5 if case.dtype == "float32" else 1e-2,
        rtol=1e-5 if case.dtype == "float32" else 1e-2,
        err_msg=f"{case.name}: fused mean diverged",
    )
    np.testing.assert_allclose(
        got_rstd,
        rstd_expected,
        atol=1e-5 if case.dtype == "float32" else 1e-2,
        rtol=1e-5 if case.dtype == "float32" else 1e-2,
        err_msg=f"{case.name}: fused rstd diverged",
    )


# ---------------------------------------------------------------------------
# Fused Residual Backward Tests
# ---------------------------------------------------------------------------


def _run_fused_residual_bwd(
    case: Case,
    d_out: np.ndarray,
    residual: np.ndarray,
    gamma: np.ndarray,
    mean: np.ndarray,
    rstd: np.ndarray,
    d_inp1: np.ndarray | None = None,
    d_inp2: np.ndarray | None = None,
    d_gamma: np.ndarray | None = None,
    d_beta: np.ndarray | None = None,
):
    bt = case.batch_size * case.seq_len
    if d_inp1 is None:
        d_inp1 = np.zeros_like(residual)
    if d_inp2 is None:
        d_inp2 = np.zeros_like(residual)
    if d_gamma is None:
        d_gamma = np.zeros(case.channels, dtype=np.float32)
    if d_beta is None:
        d_beta = np.zeros(case.channels, dtype=np.float32)
    d_residual = np.zeros_like(residual)

    return layernorm.layernorm_fused_residual_backward(
        d_inp1=to_storage(d_inp1.reshape(bt, case.channels), case.dtype),
        d_inp2=to_storage(d_inp2.reshape(bt, case.channels), case.dtype),
        d_output=to_storage(d_out.reshape(bt, case.channels), case.dtype),
        residual=to_storage(residual.reshape(bt, case.channels), case.dtype),
        gamma=to_storage(gamma, case.dtype),
        mean=mean,
        rstd=rstd,
        d_gamma=d_gamma,
        d_beta=d_beta,
        d_residual=to_storage(d_residual.reshape(bt, case.channels), case.dtype),
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        channels=case.channels,
        dtype_name=case.dtype,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_layernorm_fused_residual_backward(case: Case):
    x1, x2, gamma, beta = _make_fused_inputs(case)

    got_res, _, got_mean, got_rstd = _run_fused_residual_kernel(
        case, x1, x2, gamma, beta
    )

    bt = case.batch_size * case.seq_len
    residual_v = from_storage(got_res, case.dtype).reshape(bt, case.channels)

    x1_t = torch.from_numpy(x1).requires_grad_(True)
    x2_t = torch.from_numpy(x2).requires_grad_(True)
    g_t = torch.from_numpy(gamma).requires_grad_(True)
    b_t = torch.from_numpy(beta).requires_grad_(True)

    res_t = x1_t + x2_t
    normed_t = torch.nn.functional.layer_norm(
        res_t, (case.channels,), g_t, b_t, case.epsilon
    )

    g = torch.Generator().manual_seed(case.seed + 200)
    dy_t = torch.randn_like(normed_t, generator=g)

    normed_t.backward(dy_t)

    got_d_inp1, got_d_inp2, got_dgamma, got_dbeta, _ = _run_fused_residual_bwd(
        case,
        dy_t.detach().numpy(),
        residual_v,
        gamma,
        got_mean,
        got_rstd,
    )

    got_d_inp1_v = from_storage(got_d_inp1, case.dtype).reshape(bt, case.channels)
    got_d_inp2_v = from_storage(got_d_inp2, case.dtype).reshape(bt, case.channels)

    tol = DTYPE_TOLERANCES[case.dtype]

    assert x1_t.grad is not None
    np.testing.assert_allclose(
        got_d_inp1_v,
        x1_t.grad.numpy(),
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: fused d_inp1 diverged",
    )

    assert x2_t.grad is not None
    np.testing.assert_allclose(
        got_d_inp2_v,
        x2_t.grad.numpy(),
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: fused d_inp2 diverged",
    )

    assert g_t.grad is not None
    np.testing.assert_allclose(
        got_dgamma,
        g_t.grad.numpy(),
        atol=tol["atol"] * 100,
        rtol=tol["rtol"] * 10,
        err_msg=f"{case.name}: fused dGamma diverged",
    )

    assert b_t.grad is not None
    np.testing.assert_allclose(
        got_dbeta,
        b_t.grad.numpy(),
        atol=tol["atol"] * 100,
        rtol=tol["rtol"] * 10,
        err_msg=f"{case.name}: fused dBeta diverged",
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_layernorm_fused_residual_backward_accumulates(case: Case):
    """d_inp1/d_inp2 and d_gamma/d_beta accumulate; d_residual is scratch."""
    x1, x2, gamma, beta = _make_fused_inputs(case)

    got_res, _, got_mean, got_rstd = _run_fused_residual_kernel(
        case, x1, x2, gamma, beta
    )

    bt = case.batch_size * case.seq_len
    residual_v = from_storage(got_res, case.dtype).reshape(bt, case.channels)

    g = torch.Generator().manual_seed(case.seed + 200)
    dy = torch.randn(bt, case.channels, generator=g)
    dy = dy.to(TORCH_DTYPES[case.dtype]).to(torch.float32).numpy()

    d_inp1_1, d_inp2_1, dgamma1, dbeta1, _ = _run_fused_residual_bwd(
        case, dy, residual_v, gamma, got_mean, got_rstd
    )

    d_inp1_2, d_inp2_2, dgamma2, dbeta2, _ = _run_fused_residual_bwd(
        case,
        dy,
        residual_v,
        gamma,
        got_mean,
        got_rstd,
        d_inp1=from_storage(d_inp1_1, case.dtype).reshape(bt, case.channels),
        d_inp2=from_storage(d_inp2_1, case.dtype).reshape(bt, case.channels),
        d_gamma=dgamma1.copy(),
        d_beta=dbeta1.copy(),
    )

    d_inp1_1_v = from_storage(d_inp1_1, case.dtype).reshape(bt, case.channels)
    d_inp2_1_v = from_storage(d_inp2_1, case.dtype).reshape(bt, case.channels)
    d_inp1_2_v = from_storage(d_inp1_2, case.dtype).reshape(bt, case.channels)
    d_inp2_2_v = from_storage(d_inp2_2, case.dtype).reshape(bt, case.channels)

    tol = DTYPE_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        d_inp1_2_v,
        2.0 * d_inp1_1_v,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: d_inp1 did not accumulate",
    )
    np.testing.assert_allclose(
        d_inp2_2_v,
        2.0 * d_inp2_1_v,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: d_inp2 did not accumulate",
    )
    np.testing.assert_allclose(
        dgamma2,
        2.0 * dgamma1,
        atol=tol["atol"] * 100,
        rtol=tol["rtol"] * 10,
        err_msg=f"{case.name}: fused dGamma did not accumulate",
    )
    np.testing.assert_allclose(
        dbeta2,
        2.0 * dbeta1,
        atol=tol["atol"] * 100,
        rtol=tol["rtol"] * 10,
        err_msg=f"{case.name}: fused dBeta did not accumulate",
    )
