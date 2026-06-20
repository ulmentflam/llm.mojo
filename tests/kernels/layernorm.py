from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tests._max_bridge import (
    MutableBuf,
    ReadTensor,
    ScalarArg,
    _from_device_buffer,
    _to_device_buffer,
    max_dtype,
    pick_device,
    run_custom_op,
)

if TYPE_CHECKING:
    from max.driver import Device


def forward(
    *,
    x: np.ndarray,  # (B*T, C) flattened, kernel dtype (storage form)
    gamma: np.ndarray,  # (C,) flattened, kernel dtype (storage form)
    beta: np.ndarray,  # (C,) flattened, kernel dtype (storage form)
    mean: np.ndarray,  # (B*T,) float32
    rstd: np.ndarray,  # (B*T,) float32
    batch_size: int,
    seq_len: int,
    channels: int,
    epsilon: float,
    dtype_name: str,
    output: np.ndarray | None = None,
    device: "Device | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One layernorm_fwd pass end-to-end through MAX.

    Returns (output, mean, rstd).
    """
    if output is None:
        output = np.zeros_like(x)
    (out, out_mean, out_rstd) = run_custom_op(
        kernel_name="layernorm_fwd",
        args=[
            MutableBuf(output, dtype_name),
            ReadTensor(x, dtype_name),
            ReadTensor(gamma, dtype_name),
            ReadTensor(beta, dtype_name),
            ScalarArg(epsilon, "float32"),
            MutableBuf(mean, "float32"),
            MutableBuf(rstd, "float32"),
            ScalarArg(batch_size, "int64"),
            ScalarArg(seq_len, "int64"),
            ScalarArg(channels, "int64"),
        ],
        device=device,
    )
    return out, out_mean, out_rstd


def layernorm_fused_residual_forward(
    *,
    x1: np.ndarray,  # (B*T, C) flattened, kernel dtype (storage form)
    x2: np.ndarray,  # (B*T, C) flattened, kernel dtype (storage form)
    gamma: np.ndarray,  # (C,) flattened, kernel dtype (storage form)
    beta: np.ndarray,  # (C,) flattened, kernel dtype (storage form)
    mean: np.ndarray,  # (B*T,) float32
    rstd: np.ndarray,  # (B*T,) float32
    batch_size: int,
    seq_len: int,
    channels: int,
    epsilon: float,
    dtype_name: str,
    residual: np.ndarray | None = None,
    normed: np.ndarray | None = None,
    device: "Device | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One layernorm_fused_residual_fwd pass end-to-end through MAX.

    Returns (residual, normed, mean, rstd).
    """
    if residual is None:
        residual = np.zeros_like(x1)
    if normed is None:
        normed = np.zeros_like(x1)
    (out_residual, out_normed, out_mean, out_rstd) = run_custom_op(
        kernel_name="layernorm_fused_residual_fwd",
        args=[
            MutableBuf(residual, dtype_name),
            MutableBuf(normed, dtype_name),
            ReadTensor(x1, dtype_name),
            ReadTensor(x2, dtype_name),
            ReadTensor(gamma, dtype_name),
            ReadTensor(beta, dtype_name),
            ScalarArg(epsilon, "float32"),
            MutableBuf(mean, "float32"),
            MutableBuf(rstd, "float32"),
            ScalarArg(batch_size, "int64"),
            ScalarArg(seq_len, "int64"),
            ScalarArg(channels, "int64"),
        ],
        device=device,
    )
    return out_residual, out_normed, out_mean, out_rstd


def layernorm_fused_residual_backward(
    *,
    d_inp1: np.ndarray,  # (B*T, C) flattened, kernel dtype (storage form)
    d_inp2: np.ndarray,  # (B*T, C) flattened, kernel dtype (storage form)
    d_output: np.ndarray,  # (B*T, C) flattened, kernel dtype (storage form)
    residual: np.ndarray,  # (B*T, C) flattened, kernel dtype (storage form)
    gamma: np.ndarray,  # (C,) flattened, kernel dtype (storage form)
    mean: np.ndarray,  # (B*T,) float32
    rstd: np.ndarray,  # (B*T,) float32
    d_gamma: np.ndarray,  # (C,) float32
    d_beta: np.ndarray,  # (C,) float32
    d_residual: np.ndarray,  # (B*T, C) flattened, kernel dtype (storage form)
    batch_size: int,
    seq_len: int,
    channels: int,
    dtype_name: str,
    device: "Device | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One layernorm_fused_residual_bwd pass end-to-end through MAX.

    Accumulates into d_inp1/d_inp2 and d_gamma/d_beta. Overwrites d_residual.
    """
    (
        out_d_inp1,
        out_d_inp2,
        out_d_gamma,
        out_d_beta,
        out_d_residual,
    ) = run_custom_op(
        kernel_name="layernorm_fused_residual_bwd",
        args=[
            MutableBuf(d_inp1, dtype_name),
            MutableBuf(d_inp2, dtype_name),
            ReadTensor(d_output, dtype_name),
            ReadTensor(residual, dtype_name),
            ReadTensor(gamma, dtype_name),
            ReadTensor(mean, "float32"),
            ReadTensor(rstd, "float32"),
            MutableBuf(d_gamma, "float32"),
            MutableBuf(d_beta, "float32"),
            MutableBuf(d_residual, dtype_name),
            ScalarArg(batch_size, "int64"),
            ScalarArg(seq_len, "int64"),
            ScalarArg(channels, "int64"),
        ],
        device=device,
    )
    return out_d_inp1, out_d_inp2, out_d_gamma, out_d_beta, out_d_residual


def backward(
    *,
    d_output: np.ndarray,  # (B*T, C) flattened
    x: np.ndarray,  # (B*T, C) flattened
    gamma: np.ndarray,  # (C,) flattened
    mean: np.ndarray,  # (B*T,) float32
    rstd: np.ndarray,  # (B*T,) float32
    d_x: np.ndarray,  # (B*T, C) flattened
    d_gamma: np.ndarray,  # (C,) float32
    d_beta: np.ndarray,  # (C,) float32
    batch_size: int,
    seq_len: int,
    channels: int,
    dtype_name: str,
    device: "Device | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One layernorm_bwd pass end-to-end through MAX."""
    (out_d_x, out_d_gamma, out_d_beta) = run_custom_op(
        kernel_name="layernorm_bwd",
        args=[
            ReadTensor(d_output, dtype_name),
            ReadTensor(x, dtype_name),
            ReadTensor(gamma, dtype_name),
            ReadTensor(mean, "float32"),
            ReadTensor(rstd, "float32"),
            MutableBuf(d_x, dtype_name),
            MutableBuf(d_gamma, "float32"),
            MutableBuf(d_beta, "float32"),
            ScalarArg(batch_size, "int64"),
            ScalarArg(seq_len, "int64"),
            ScalarArg(channels, "int64"),
        ],
        device=device,
    )
    return out_d_x, out_d_gamma, out_d_beta


_MODULAR_CACHE: dict[tuple, tuple] = {}


def modular_forward(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float,
    dtype_name: str,
    device: "Device | None" = None,
) -> np.ndarray:
    """Modular's own layernorm (`max.graph.ops.layer_norm`) over the last axis."""
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, TensorType, ops

    if device is None:
        device = pick_device()

    key = (dtype_name, x.ndim, type(device).__name__, str(device))
    cached = _MODULAR_CACHE.get(key)
    if cached is None:
        dev_ref = DeviceRef.from_device(device)
        graph = Graph(
            "modular_layernorm_reference",
            forward=lambda t, g, b: (ops.layer_norm(t, g, b, epsilon=epsilon),),
            input_types=[
                TensorType(
                    max_dtype(dtype_name),
                    shape=[f"dim{d}" for d in range(x.ndim)],
                    device=dev_ref,
                ),
                TensorType(
                    max_dtype(dtype_name),
                    shape=["C"],
                    device=dev_ref,
                ),
                TensorType(
                    max_dtype(dtype_name),
                    shape=["C"],
                    device=dev_ref,
                ),
            ],
        )
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        _MODULAR_CACHE[key] = (session, model)
    else:
        _, model = cached

    (out,) = model.execute(
        _to_device_buffer(x, dtype_name, device),
        _to_device_buffer(gamma, dtype_name, device),
        _to_device_buffer(beta, dtype_name, device),
    )
    return _from_device_buffer(out, dtype_name)
