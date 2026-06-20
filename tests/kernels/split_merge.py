from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tests._dtypes import NP_STORAGE_DTYPES
from tests._max_bridge import MutableBuf, ReadTensor, ScalarArg, run_custom_op

if TYPE_CHECKING:
    from max.driver import Device


def _scalars(
    batch_size: int, seq_len: int, num_heads: int, head_dim: int
) -> list[ScalarArg]:
    return [
        ScalarArg(batch_size, "int64"),
        ScalarArg(seq_len, "int64"),
        ScalarArg(num_heads, "int64"),
        ScalarArg(head_dim, "int64"),
    ]


def _head_size(batch_size: int, seq_len: int, num_heads: int, head_dim: int) -> int:
    return batch_size * num_heads * seq_len * head_dim


def _qkv_size(
    batch_size: int, seq_len: int, num_heads: int, head_dim: int, num_splits: int = 3
) -> int:
    channels = num_heads * head_dim
    return batch_size * seq_len * num_splits * channels


def split_forward(
    *,
    src: np.ndarray,  # flat (B, T, 3*C) qkv layout, kernel dtype (storage form)
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_name: str,
    dst0: np.ndarray | None = None,
    dst1: np.ndarray | None = None,
    dst2: np.ndarray | None = None,
    device: "Device | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """split_fwd: qkv -> (q, k, v) in head layout (B, NH, T, HD) flattened."""
    storage = NP_STORAGE_DTYPES[dtype_name]
    size = _head_size(batch_size, seq_len, num_heads, head_dim)
    if dst0 is None:
        dst0 = np.zeros(size, dtype=storage)
    if dst1 is None:
        dst1 = np.zeros(size, dtype=storage)
    if dst2 is None:
        dst2 = np.zeros(size, dtype=storage)
    dst0, dst1, dst2 = run_custom_op(
        kernel_name="split_fwd",
        args=[
            MutableBuf(dst0, dtype_name),
            MutableBuf(dst1, dtype_name),
            MutableBuf(dst2, dtype_name),
            ReadTensor(src, dtype_name),
            *_scalars(batch_size, seq_len, num_heads, head_dim),
        ],
        device=device,
    )
    return dst0, dst1, dst2


def split_backward(
    *,
    d_dst0: np.ndarray,
    d_dst1: np.ndarray,
    d_dst2: np.ndarray,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_name: str,
    d_src: np.ndarray | None = None,
    device: "Device | None" = None,
) -> np.ndarray:
    """split_bwd: scatter head-layout split grads back into qkv layout."""
    storage = NP_STORAGE_DTYPES[dtype_name]
    if d_src is None:
        d_src = np.zeros(
            _qkv_size(batch_size, seq_len, num_heads, head_dim), dtype=storage
        )
    (d_src,) = run_custom_op(
        kernel_name="split_bwd",
        args=[
            MutableBuf(d_src, dtype_name),
            ReadTensor(d_dst0, dtype_name),
            ReadTensor(d_dst1, dtype_name),
            ReadTensor(d_dst2, dtype_name),
            *_scalars(batch_size, seq_len, num_heads, head_dim),
        ],
        device=device,
    )
    return d_src


def merge_forward(
    *,
    src: np.ndarray,  # flat head layout (B, NH, T, HD)
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_name: str,
    dst: np.ndarray | None = None,
    device: "Device | None" = None,
) -> np.ndarray:
    """merge_fwd: head layout -> (B, T, C) flattened."""
    storage = NP_STORAGE_DTYPES[dtype_name]
    channels = num_heads * head_dim
    if dst is None:
        dst = np.zeros(batch_size * seq_len * channels, dtype=storage)
    (dst,) = run_custom_op(
        kernel_name="merge_fwd",
        args=[
            MutableBuf(dst, dtype_name),
            ReadTensor(src, dtype_name),
            *_scalars(batch_size, seq_len, num_heads, head_dim),
        ],
        device=device,
    )
    return dst


def merge_backward(
    *,
    d_dst: np.ndarray,  # flat (B, T, C) merged-layout gradient
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_name: str,
    d_src: np.ndarray | None = None,
    device: "Device | None" = None,
) -> np.ndarray:
    """merge_bwd: scatter merged-layout grad back into head layout."""
    storage = NP_STORAGE_DTYPES[dtype_name]
    if d_src is None:
        d_src = np.zeros(
            _head_size(batch_size, seq_len, num_heads, head_dim), dtype=storage
        )
    (d_src,) = run_custom_op(
        kernel_name="merge_bwd",
        args=[
            MutableBuf(d_src, dtype_name),
            ReadTensor(d_dst, dtype_name),
            *_scalars(batch_size, seq_len, num_heads, head_dim),
        ],
        device=device,
    )
    return d_src
