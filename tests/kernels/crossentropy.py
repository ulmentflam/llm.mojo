from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tests._max_bridge import MutableBuf, ReadTensor, ScalarArg, run_custom_op

if TYPE_CHECKING:
    from max.driver import Device


def forward(
    *,
    probs: np.ndarray,  # (B*T*Vp,) flattened, kernel dtype
    targets: np.ndarray,  # (B*T,) int32, values in [0, V)
    batch_size: int,
    seq_len: int,
    vocab_size_padded: int,
    dtype_name: str,
    losses: np.ndarray | None = None,
    device: "Device | None" = None,
) -> np.ndarray:
    """One crossentropy_ohe_fwd pass end-to-end through MAX.

    `losses` is the mutable output buffer (always float32, shape (B*T,));
    pass one in to test what the kernel does to pre-existing contents,
    otherwise a zeroed buffer is used. Returns the post-execution losses.
    """
    if losses is None:
        losses = np.zeros(batch_size * seq_len, dtype=np.float32)
    (out_losses,) = run_custom_op(
        kernel_name="crossentropy_ohe_fwd",
        args=[
            MutableBuf(losses, "float32"),
            ReadTensor(probs, dtype_name),
            ReadTensor(targets, "int32"),
            ScalarArg(batch_size, "int64"),
            ScalarArg(seq_len, "int64"),
            ScalarArg(vocab_size_padded, "int64"),
        ],
        device=device,
    )
    return out_losses


def backward(
    *,
    d_losses: np.ndarray,  # (B*T,) float32 upstream gradient
    probs: np.ndarray,  # (B*T*Vp,) flattened, kernel dtype
    targets: np.ndarray,  # (B*T,) int32, values in [0, V)
    batch_size: int,
    seq_len: int,
    vocab_size_padded: int,
    dtype_name: str,
    d_probs: np.ndarray | None = None,
    device: "Device | None" = None,
) -> np.ndarray:
    """One crossentropy_ohe_bwd pass end-to-end through MAX.

    `d_probs` is the mutable output buffer (kernel dtype, shape
    (B*T*Vp,)). The kernel writes only the target column of each row and
    leaves every other entry untouched, so callers must zero (or
    sentinel-fill) the buffer themselves. Returns the post-execution
    d_probs.
    """
    if d_probs is None:
        d_probs = np.zeros_like(probs)
    (out_d_probs,) = run_custom_op(
        kernel_name="crossentropy_ohe_bwd",
        args=[
            ReadTensor(d_losses, "float32"),
            MutableBuf(d_probs, dtype_name),
            ReadTensor(probs, dtype_name),
            ReadTensor(targets, "int32"),
            ScalarArg(batch_size, "int64"),
            ScalarArg(seq_len, "int64"),
            ScalarArg(vocab_size_padded, "int64"),
        ],
        device=device,
    )
    return out_d_probs
