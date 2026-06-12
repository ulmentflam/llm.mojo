from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tests._max_bridge import MutableBuf, ReadTensor, run_custom_op

if TYPE_CHECKING:
    from max.driver import Device


def forward(
    *,
    x: np.ndarray,  # (rows, cols) kernel dtype (storage form)
    dtype_name: str,
    output: np.ndarray | None = None,
    device: "Device | None" = None,
) -> np.ndarray:
    """One gelu_fwd pass end-to-end through MAX.

    output = gelu(x), llm.c tanh approximation. x is left untouched (the
    pre-activation survives for the backward). Returns the post-execution
    output.
    """
    if output is None:
        output = np.zeros_like(x)
    (out,) = run_custom_op(
        kernel_name="gelu_fwd",
        args=[
            MutableBuf(output, dtype_name),
            ReadTensor(x, dtype_name),
        ],
        device=device,
    )
    return out


def backward(
    *,
    x: np.ndarray,  # (rows, cols) PRE-activation, kernel dtype (storage form)
    dtype_name: str,
    output: np.ndarray | None = None,
    device: "Device | None" = None,
) -> np.ndarray:
    """One gelu_bwd pass end-to-end through MAX.

    output = gelu'(x), the LOCAL derivative at the pre-activation; the op
    takes no upstream gradient, so callers contract with d_out themselves.
    Returns the post-execution output.
    """
    if output is None:
        output = np.zeros_like(x)
    (out,) = run_custom_op(
        kernel_name="gelu_bwd",
        args=[
            MutableBuf(output, dtype_name),
            ReadTensor(x, dtype_name),
        ],
        device=device,
    )
    return out
