from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tests._max_bridge import MutableBuf, ReadTensor, ScalarArg, run_custom_op

if TYPE_CHECKING:
    from max.driver import Device


def forward(
    *,
    inp: np.ndarray,  # (batch_size, seq_len) int32
    wte: np.ndarray,  # (vocab_size, channels) kernel dtype (storage form)
    wpe: np.ndarray,  # (seq_len, channels) kernel dtype (storage form)
    dtype_name: str,
    output: np.ndarray | None = None,
    device: "Device | None" = None,
) -> np.ndarray:
    """One encoder_fwd pass end-to-end through MAX.

    Computes token embedding + position embedding.
    """
    B, T = inp.shape
    C = wte.shape[1]
    if output is None:
        output = np.zeros((B, T, C), dtype=wte.dtype)
    (out,) = run_custom_op(
        kernel_name="encoder_fwd",
        args=[
            MutableBuf(output, dtype_name),
            ReadTensor(inp, "int32"),
            ReadTensor(wte, dtype_name),
            ReadTensor(wpe, dtype_name),
            ScalarArg(B, "int64"),
            ScalarArg(T, "int64"),
            ScalarArg(C, "int64"),
        ],
        device=device,
    )
    return out


def backward(
    *,
    dwte: np.ndarray,  # (vocab_size, channels) kernel dtype (storage form)
    dwpe: np.ndarray,  # (seq_len, channels) kernel dtype (storage form)
    bucket_info: np.ndarray,  # (num_buckets, 4) int32
    workload_indices: np.ndarray,  # (B * T) int32
    dout: np.ndarray,  # (batch_size, seq_len, channels) kernel dtype (storage form)
    dtype_name: str,
    device: "Device | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One encoder_bwd pass end-to-end through MAX.

    Accumulates gradients into dwte and dwpe.
    """
    num_buckets = bucket_info.shape[0]
    B, T, C = dout.shape
    (out_dwte, out_dwpe) = run_custom_op(
        kernel_name="encoder_bwd",
        args=[
            MutableBuf(dwte, dtype_name),
            MutableBuf(dwpe, dtype_name),
            ReadTensor(bucket_info, "int32"),
            ReadTensor(workload_indices, "int32"),
            ReadTensor(dout, dtype_name),
            ScalarArg(num_buckets, "int64"),
            ScalarArg(B, "int64"),
            ScalarArg(T, "int64"),
            ScalarArg(C, "int64"),
        ],
        device=device,
    )
    return out_dwte, out_dwpe
