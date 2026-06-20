"""Equivalence: Mojo split_{fwd,bwd} / merge_{fwd,bwd} vs NumPy layout refs.

These ops are pure layout transforms used around attention:

  *  split_fwd: (B, T, 3*C) qkv interleave -> three (B, NH, T, HD) head buffers
  *  merge_fwd: one head buffer -> (B, T, C) merged layout
  *  split_bwd / merge_bwd: exact inverses for gradient scatter

Properties beyond plain closeness:

  *  split then merge on q recovers the q slice of the original qkv layout
  *  round-trip backward passes are inverses of their forward partners

Inputs are round-tripped through the kernel dtype so the kernel and the
reference consume bit-identical values (float16 is not supported).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tests._dtypes import DTYPE_TOLERANCES, TORCH_DTYPES, from_storage, to_storage
from tests.kernels import split_merge

NUM_SPLITS = 3


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int
    dtype: str  # "float32" | "bfloat16"
    seed: int


CASES: tuple[Case, ...] = (
    Case(
        "fp32_small",
        batch_size=2,
        seq_len=3,
        num_heads=2,
        head_dim=4,
        dtype="float32",
        seed=0,
    ),
    # Odd head_dim: tail paths in vectorized CPU / GPU kernels.
    Case(
        "fp32_odd_head",
        batch_size=2,
        seq_len=7,
        num_heads=3,
        head_dim=5,
        dtype="float32",
        seed=1,
    ),
    # GPT-2 attention shape (scaled-down batch).
    Case(
        "fp32_gpt2_attn",
        batch_size=1,
        seq_len=16,
        num_heads=12,
        head_dim=64,
        dtype="float32",
        seed=2,
    ),
    Case(
        "bf16_small",
        batch_size=2,
        seq_len=4,
        num_heads=2,
        head_dim=8,
        dtype="bfloat16",
        seed=3,
    ),
)


def _ids(case: Case) -> str:
    return case.name


def _channels(case: Case) -> int:
    return case.num_heads * case.head_dim


def _make_qkv(case: Case) -> np.ndarray:
    """Deterministic qkv flat buffer in kernel layout."""
    B, T = case.batch_size, case.seq_len
    C = _channels(case)
    size = B * T * NUM_SPLITS * C
    raw = np.arange(size, dtype=np.float32)
    td = TORCH_DTYPES[case.dtype]
    return torch.from_numpy(raw).to(td).to(torch.float32).numpy()


def _split_fwd_ref(
    qkv: np.ndarray, case: Case
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B, T, NH, HD = case.batch_size, case.seq_len, case.num_heads, case.head_dim
    C = _channels(case)
    qkv3 = qkv.reshape(B, T, NUM_SPLITS, C)
    outs = []
    for s in range(NUM_SPLITS):
        part = qkv3[:, :, s, :].reshape(B, T, NH, HD)
        part = np.transpose(part, (0, 2, 1, 3))
        outs.append(part.reshape(-1))
    return outs[0], outs[1], outs[2]


def _merge_fwd_ref(x: np.ndarray, case: Case) -> np.ndarray:
    B, T, NH, HD = case.batch_size, case.seq_len, case.num_heads, case.head_dim
    C = _channels(case)
    x4 = x.reshape(B, NH, T, HD)
    merged = np.transpose(x4, (0, 2, 1, 3))
    return merged.reshape(B, T, C).reshape(-1)


def _split_bwd_ref(
    d_q: np.ndarray, d_k: np.ndarray, d_v: np.ndarray, case: Case
) -> np.ndarray:
    B, T, NH, HD = case.batch_size, case.seq_len, case.num_heads, case.head_dim
    C = _channels(case)
    d_src = np.zeros((B, T, NUM_SPLITS, C), dtype=d_q.dtype)
    for s, d_part in enumerate((d_q, d_k, d_v)):
        part = d_part.reshape(B, NH, T, HD)
        part = np.transpose(part, (0, 2, 1, 3))
        d_src[:, :, s, :] = part.reshape(B, T, C)
    return d_src.reshape(-1)


def _merge_bwd_ref(d_merged: np.ndarray, case: Case) -> np.ndarray:
    B, T, NH, HD = case.batch_size, case.seq_len, case.num_heads, case.head_dim
    C = _channels(case)
    dm = d_merged.reshape(B, T, C).reshape(B, T, NH, HD)
    dm = np.transpose(dm, (0, 2, 1, 3))
    return dm.reshape(-1)


def _torch_split_grad(case: Case, qkv: np.ndarray) -> np.ndarray:
    """d(qkv) when loss = sum of split(q,k,v) tiles."""
    td = TORCH_DTYPES[case.dtype]
    B, T, NH, HD = case.batch_size, case.seq_len, case.num_heads, case.head_dim
    C = _channels(case)
    qkv_t = (
        torch.from_numpy(qkv).to(td).reshape(B, T, NUM_SPLITS, C).requires_grad_(True)
    )
    parts = []
    for s in range(NUM_SPLITS):
        p = qkv_t[:, :, s, :].reshape(B, T, NH, HD).permute(0, 2, 1, 3)
        parts.append(p)
    loss = parts[0].sum()
    for part in parts[1:]:
        loss = loss + part.sum()
    loss.backward()
    assert qkv_t.grad is not None
    return qkv_t.grad.to(torch.float32).numpy().reshape(-1)


def _torch_merge_grad(case: Case) -> np.ndarray:
    """d(head) when loss = sum(merge(head))."""
    td = TORCH_DTYPES[case.dtype]
    B, T, NH, HD = case.batch_size, case.seq_len, case.num_heads, case.head_dim
    q = torch.zeros(B, NH, T, HD, dtype=td, requires_grad=True)
    merged = q.permute(0, 2, 1, 3).reshape(B, T, NH * HD)
    merged.sum().backward()
    assert q.grad is not None
    return q.grad.to(torch.float32).numpy().reshape(-1)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_split_forward_matches_reference(case: Case) -> None:
    qkv = _make_qkv(case)
    qkv_storage = to_storage(qkv, case.dtype)
    dst0, dst1, dst2 = split_merge.split_forward(
        src=qkv_storage,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        num_heads=case.num_heads,
        head_dim=case.head_dim,
        dtype_name=case.dtype,
    )
    exp0, exp1, exp2 = _split_fwd_ref(qkv, case)
    tol = DTYPE_TOLERANCES[case.dtype]
    for got, exp in zip(
        (
            from_storage(dst0, case.dtype),
            from_storage(dst1, case.dtype),
            from_storage(dst2, case.dtype),
        ),
        (exp0, exp1, exp2),
        strict=True,
    ):
        np.testing.assert_allclose(got, exp, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_merge_forward_matches_reference(case: Case) -> None:
    qkv = _make_qkv(case)
    q, _, _ = _split_fwd_ref(qkv, case)
    q_storage = to_storage(q, case.dtype)
    merged = split_merge.merge_forward(
        src=q_storage,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        num_heads=case.num_heads,
        head_dim=case.head_dim,
        dtype_name=case.dtype,
    )
    expected = _merge_fwd_ref(q, case)
    tol = DTYPE_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        from_storage(merged, case.dtype),
        expected,
        atol=tol["atol"],
        rtol=tol["rtol"],
    )


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_split_backward_matches_reference(case: Case) -> None:
    qkv = _make_qkv(case)
    q, k, v = _split_fwd_ref(qkv, case)
    d_q = to_storage(q, case.dtype)
    d_k = to_storage(k, case.dtype)
    d_v = to_storage(v, case.dtype)
    d_src = split_merge.split_backward(
        d_dst0=d_q,
        d_dst1=d_k,
        d_dst2=d_v,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        num_heads=case.num_heads,
        head_dim=case.head_dim,
        dtype_name=case.dtype,
    )
    expected = _split_bwd_ref(q, k, v, case)
    tol = DTYPE_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        from_storage(d_src, case.dtype),
        expected,
        atol=tol["atol"],
        rtol=tol["rtol"],
    )


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_merge_backward_matches_reference(case: Case) -> None:
    qkv = _make_qkv(case)
    q, _, _ = _split_fwd_ref(qkv, case)
    merged = _merge_fwd_ref(q, case)
    d_merged = to_storage(merged, case.dtype)
    d_src = split_merge.merge_backward(
        d_dst=d_merged,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        num_heads=case.num_heads,
        head_dim=case.head_dim,
        dtype_name=case.dtype,
    )
    expected = _merge_bwd_ref(merged, case)
    tol = DTYPE_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        from_storage(d_src, case.dtype),
        expected,
        atol=tol["atol"],
        rtol=tol["rtol"],
    )


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_backward_matches_torch_autograd(case: Case) -> None:
    qkv = _make_qkv(case)
    head_size = case.batch_size * case.num_heads * case.seq_len * case.head_dim
    merged_size = case.batch_size * case.seq_len * _channels(case)
    ones_head = np.ones(head_size, dtype=np.float32)
    ones_merged = np.ones(merged_size, dtype=np.float32)
    tol = DTYPE_TOLERANCES[case.dtype]

    d_src = split_merge.split_backward(
        d_dst0=to_storage(ones_head, case.dtype),
        d_dst1=to_storage(ones_head, case.dtype),
        d_dst2=to_storage(ones_head, case.dtype),
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        num_heads=case.num_heads,
        head_dim=case.head_dim,
        dtype_name=case.dtype,
    )
    np.testing.assert_allclose(
        from_storage(d_src, case.dtype),
        _torch_split_grad(case, qkv),
        atol=tol["atol"],
        rtol=tol["rtol"],
    )

    d_head = split_merge.merge_backward(
        d_dst=to_storage(ones_merged, case.dtype),
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        num_heads=case.num_heads,
        head_dim=case.head_dim,
        dtype_name=case.dtype,
    )
    np.testing.assert_allclose(
        from_storage(d_head, case.dtype),
        _torch_merge_grad(case),
        atol=tol["atol"],
        rtol=tol["rtol"],
    )


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_split_merge_roundtrip(case: Case) -> None:
    qkv = _make_qkv(case)
    qkv_storage = to_storage(qkv, case.dtype)
    q, k, v = split_merge.split_forward(
        src=qkv_storage,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        num_heads=case.num_heads,
        head_dim=case.head_dim,
        dtype_name=case.dtype,
    )
    merged_q = split_merge.merge_forward(
        src=q,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        num_heads=case.num_heads,
        head_dim=case.head_dim,
        dtype_name=case.dtype,
    )
    qkv3 = from_storage(qkv_storage, case.dtype).reshape(
        case.batch_size, case.seq_len, NUM_SPLITS, _channels(case)
    )
    expected_q = qkv3[:, :, 0, :]
    tol = DTYPE_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        from_storage(merged_q, case.dtype).reshape(expected_q.shape),
        expected_q,
        atol=tol["atol"],
        rtol=tol["rtol"],
    )
