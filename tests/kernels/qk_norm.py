from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tests._max_bridge import MutableBuf, ReadTensor, ScalarArg, run_custom_op

if TYPE_CHECKING:
    from max.driver import Device


def _scalars(
    batch_size: int, num_heads: int, seq_len: int, head_dim: int, q_mul: float
) -> list[ScalarArg]:
    return [
        ScalarArg(batch_size, "int64"),
        ScalarArg(num_heads, "int64"),
        ScalarArg(seq_len, "int64"),
        ScalarArg(head_dim, "int64"),
        ScalarArg(q_mul, "float32"),
    ]


def forward(
    *,
    q: np.ndarray,  # (B, NH, T, HD) head layout, kernel dtype (storage form)
    k: np.ndarray,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    q_mul: float,
    dtype_name: str,
    do_norm: bool = True,
    device: "Device | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """qk_rows_fwd: q <- q_mul * rmsnorm(q), k <- rmsnorm(k) (in place)."""
    q_out, k_out = run_custom_op(
        kernel_name="qk_rows_fwd",
        args=[
            MutableBuf(q.copy(), dtype_name),
            MutableBuf(k.copy(), dtype_name),
            *_scalars(batch_size, num_heads, seq_len, head_dim, q_mul),
        ],
        device=device,
        parameters={"do_norm": do_norm},
    )
    return q_out, k_out


def backward(
    *,
    d_q: np.ndarray,  # grads w.r.t. the normalized/scaled q, k (head layout)
    d_k: np.ndarray,
    qkv: np.ndarray,  # (B, T, 3C) pre-norm token-layout activation
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    q_mul: float,
    dtype_name: str,
    do_norm: bool = True,
    device: "Device | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """qk_rows_bwd: grads w.r.t. the raw q, k rows (in place)."""
    dq_out, dk_out = run_custom_op(
        kernel_name="qk_rows_bwd",
        args=[
            MutableBuf(d_q.copy(), dtype_name),
            MutableBuf(d_k.copy(), dtype_name),
            ReadTensor(qkv, dtype_name),
            *_scalars(batch_size, num_heads, seq_len, head_dim, q_mul),
        ],
        device=device,
        parameters={"do_norm": do_norm},
    )
    return dq_out, dk_out
