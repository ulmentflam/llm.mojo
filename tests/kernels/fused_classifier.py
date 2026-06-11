from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tests._max_bridge import MutableBuf, ReadTensor, ScalarArg, run_custom_op

if TYPE_CHECKING:
    from max.driver import Device


def forward_backward(
    *,
    logits: np.ndarray,  # (B*T*Vp,) flattened, kernel dtype (storage form)
    d_losses: np.ndarray,  # (B*T,) float32 upstream gradient
    targets: np.ndarray,  # (B*T,) int32, values in [0, V)
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    vocab_size_padded: int,
    dtype_name: str,
    losses: np.ndarray | None = None,
    device: "Device | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One fused_classifier training pass end-to-end through MAX.

    `logits` is consumed IN PLACE: the post-execution buffer holds
    d_logits = (p - onehot(target)) * d_loss over [0, V) of each row, with
    the padded tail zeroed. Callers that need the original logits again
    must pass a copy. Returns (d_logits, losses).
    """
    if losses is None:
        losses = np.zeros(batch_size * seq_len, dtype=np.float32)
    out_d_logits, out_losses = run_custom_op(
        kernel_name="fused_classifier",
        args=[
            MutableBuf(logits, dtype_name),
            MutableBuf(losses, "float32"),
            ReadTensor(d_losses, "float32"),
            ReadTensor(targets, "int32"),
            ScalarArg(int(batch_size), "int64"),
            ScalarArg(int(seq_len), "int64"),
            ScalarArg(int(vocab_size), "int64"),
            ScalarArg(int(vocab_size_padded), "int64"),
        ],
        device=device,
    )
    return out_d_logits, out_losses


def forward(
    *,
    logits: np.ndarray,  # (B*T*Vp,) flattened, kernel dtype (storage form)
    targets: np.ndarray,  # (B*T,) int32, values in [0, V)
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    vocab_size_padded: int,
    dtype_name: str,
    losses: np.ndarray | None = None,
    device: "Device | None" = None,
) -> np.ndarray:
    """One fused_classifier_fwd eval pass: losses only, logits read-only.

    Returns the post-execution losses (float32, shape (B*T,)).
    """
    if losses is None:
        losses = np.zeros(batch_size * seq_len, dtype=np.float32)
    (out_losses,) = run_custom_op(
        kernel_name="fused_classifier_fwd",
        args=[
            MutableBuf(losses, "float32"),
            ReadTensor(logits, dtype_name),
            ReadTensor(targets, "int32"),
            ScalarArg(int(batch_size), "int64"),
            ScalarArg(int(seq_len), "int64"),
            ScalarArg(int(vocab_size), "int64"),
            ScalarArg(int(vocab_size_padded), "int64"),
        ],
        device=device,
    )
    return out_losses
