from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tests._max_bridge import MutableBuf, ReadTensor, ScalarArg, run_custom_op
from tests._dtypes import NP_STORAGE_DTYPES

if TYPE_CHECKING:
    from max.driver import Device


def _scalars(batch_size: int, seq_len: int, channels: int, output_channels: int):
    return [
        ScalarArg(batch_size, "int64"),
        ScalarArg(seq_len, "int64"),
        ScalarArg(channels, "int64"),
        ScalarArg(output_channels, "int64"),
    ]


def _dummy(dtype_name: str) -> np.ndarray:
    """Placeholder for a tensor the requested instantiation never touches
    (comptime-dead in the kernel); rank-2 to satisfy the op signature."""
    return np.zeros((1, 1), dtype=NP_STORAGE_DTYPES[dtype_name])


def _dummy_bias(dtype_name: str) -> np.ndarray:
    """Rank-1 placeholder when has_bias=False (comptime-dead in the kernel)."""
    return np.zeros(1, dtype=NP_STORAGE_DTYPES[dtype_name])


def forward(
    *,
    x: np.ndarray,  # (B*T, C) kernel dtype (storage form)
    weight: np.ndarray,  # (OC, C)
    bias: np.ndarray,  # (OC,)
    batch_size: int,
    seq_len: int,
    channels: int,
    output_channels: int,
    dtype_name: str,
    use_gelu: bool = False,
    has_bias: bool = True,
    device: "Device | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(out, pre_gelu) through the registered matmul_fwd op.

    out = x @ weight^T (+ bias when has_bias); with use_gelu, pre_gelu holds
    that value and out gets gelu(pre_gelu).
    """
    rows = batch_size * seq_len
    storage = NP_STORAGE_DTYPES[dtype_name]
    out = np.zeros((rows, output_channels), dtype=storage)
    pre_gelu = (
        np.zeros((rows, output_channels), dtype=storage)
        if use_gelu
        else _dummy(dtype_name)
    )
    bias_arg = bias if has_bias else _dummy_bias(dtype_name)
    out, pre_gelu = run_custom_op(
        kernel_name="matmul_fwd",
        args=[
            MutableBuf(out, dtype_name),
            MutableBuf(pre_gelu, dtype_name),
            ReadTensor(x, dtype_name),
            ReadTensor(weight, dtype_name),
            ReadTensor(bias_arg, dtype_name),
            *_scalars(batch_size, seq_len, channels, output_channels),
        ],
        parameters={"use_gelu": use_gelu, "has_bias": has_bias},
        device=device,
    )
    return out, pre_gelu


def backward(
    *,
    d_output: np.ndarray,  # (B*T, OC) kernel dtype (storage form)
    x: np.ndarray,  # (B*T, C)
    weight: np.ndarray,  # (OC, C)
    batch_size: int,
    seq_len: int,
    channels: int,
    output_channels: int,
    dtype_name: str,
    use_gelu: bool = False,
    accumulate: bool = True,
    has_bias: bool = True,
    pre_gelu: np.ndarray
    | None = None,  # (B*T, C) pre-activation of x; required when use_gelu
    d_input: np.ndarray | None = None,
    d_weight: np.ndarray | None = None,
    d_bias: np.ndarray | None = None,
    device: "Device | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(d_input, d_weight, d_bias) through matmul_bwd.

    With accumulate (llm.c beta=1 contract) the op ACCUMULATES into
    d_weight/d_bias; d_input is always overwritten. Pass zeroed buffers
    (the default) for plain gradients, or prior-step grads to test
    accumulation. When has_bias=False, d_bias is a comptime-dead dummy.
    """
    rows = batch_size * seq_len
    storage = NP_STORAGE_DTYPES[dtype_name]
    if use_gelu and pre_gelu is None:
        raise ValueError("use_gelu backward requires the forward's pre_gelu")
    if pre_gelu is None:
        pre_gelu = _dummy(dtype_name)
    if d_input is None:
        d_input = np.zeros((rows, channels), dtype=storage)
    if d_weight is None:
        d_weight = np.zeros((output_channels, channels), dtype=storage)
    if d_bias is None:
        d_bias = (
            np.zeros(output_channels, dtype=storage)
            if has_bias
            else _dummy_bias(dtype_name)
        )
    scratch = np.zeros((output_channels, rows), dtype=storage)
    d_input, d_weight, d_bias, _ = run_custom_op(
        kernel_name="matmul_bwd",
        args=[
            MutableBuf(d_input, dtype_name),
            MutableBuf(d_weight, dtype_name),
            MutableBuf(d_bias, dtype_name),
            MutableBuf(scratch, dtype_name),
            ReadTensor(d_output, dtype_name),
            ReadTensor(x, dtype_name),
            ReadTensor(weight, dtype_name),
            ReadTensor(pre_gelu, dtype_name),
            *_scalars(batch_size, seq_len, channels, output_channels),
        ],
        parameters={
            "use_gelu": use_gelu,
            "accumulate": accumulate,
            "has_bias": has_bias,
        },
        device=device,
    )
    return d_input, d_weight, d_bias
