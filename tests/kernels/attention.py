from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tests._max_bridge import (
    MutableBuf,
    ReadTensor,
    ScalarArg,
    run_custom_op,
)

if TYPE_CHECKING:
    from max.driver import Device


def forward(
    *,
    q: np.ndarray,  # (B, NH, T, HS) kernel dtype
    k: np.ndarray,  # (B, NH, T, HS) kernel dtype
    v: np.ndarray,  # (B, NH, T, HS) kernel dtype
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype_name: str,
    output: np.ndarray | None = None,
    l_vec: np.ndarray | None = None,
    use_soft_exp: bool = False,
    use_conditional_rescale: bool = False,
    device: "Device | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One attention_fwd pass end-to-end through MAX.

    `output` is the mutable output buffer (kernel dtype, shape (B, NH, T, HS)).
    `l_vec` is the mutable log-sum-exp buffer (float32, shape (B, NH, T)).
    Returns (output, l_vec).
    """
    if output is None:
        output = np.zeros_like(q)
    if l_vec is None:
        l_vec = np.zeros((batch_size, num_heads, seq_len), dtype=np.float32)

    (out_probs, out_l_vec) = run_custom_op(
        kernel_name="attention_fwd",
        args=[
            MutableBuf(output, dtype_name),
            ReadTensor(q, dtype_name),
            ReadTensor(k, dtype_name),
            ReadTensor(v, dtype_name),
            MutableBuf(l_vec, "float32"),
            ScalarArg(batch_size, "int64"),
            ScalarArg(num_heads, "int64"),
            ScalarArg(seq_len, "int64"),
            ScalarArg(head_dim, "int64"),
        ],
        device=device,
        parameters={
            "use_soft_exp": use_soft_exp,
            "use_conditional_rescale": use_conditional_rescale,
        },
    )
    return out_probs, out_l_vec


def backward(
    *,
    d_q: np.ndarray,
    d_k: np.ndarray,
    d_v: np.ndarray,
    d_output: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    output: np.ndarray,
    l_vec: np.ndarray,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype_name: str,
    use_soft_exp: bool = False,
    device: "Device | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One attention_bwd pass end-to-end through MAX.

    Returns (d_q, d_k, d_v).
    """
    (out_d_q, out_d_k, out_d_v) = run_custom_op(
        kernel_name="attention_bwd",
        args=[
            MutableBuf(d_q, dtype_name),
            MutableBuf(d_k, dtype_name),
            MutableBuf(d_v, dtype_name),
            ReadTensor(d_output, dtype_name),
            ReadTensor(q, dtype_name),
            ReadTensor(k, dtype_name),
            ReadTensor(v, dtype_name),
            ReadTensor(output, dtype_name),
            ReadTensor(l_vec, "float32"),
            ScalarArg(batch_size, "int64"),
            ScalarArg(num_heads, "int64"),
            ScalarArg(seq_len, "int64"),
            ScalarArg(head_dim, "int64"),
        ],
        device=device,
        parameters={
            "use_soft_exp": use_soft_exp,
        },
    )
    return out_d_q, out_d_k, out_d_v
