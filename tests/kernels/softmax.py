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
    logits: np.ndarray,  # (B*T*Vp,) flattened, kernel dtype (storage form)
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    vocab_size_padded: int,
    dtype_name: str,
    probs: np.ndarray | None = None,
    device: "Device | None" = None,
) -> np.ndarray:
    """One softmax_fwd pass end-to-end through MAX.

    `probs` is the mutable output buffer (kernel dtype, shape (B*T*Vp,)).
    The kernel writes only the first V columns of each row and leaves the
    padded tail untouched, so callers must zero (or sentinel-fill) the
    buffer themselves. Returns the post-execution probs.
    """
    if probs is None:
        probs = np.zeros_like(logits)
    (out_probs,) = run_custom_op(
        kernel_name="softmax_fwd",
        args=[
            MutableBuf(probs, dtype_name),
            ReadTensor(logits, dtype_name),
            ScalarArg(int(batch_size), "int64"),
            ScalarArg(int(seq_len), "int64"),
            ScalarArg(int(vocab_size), "int64"),
            ScalarArg(int(vocab_size_padded), "int64"),
        ],
        device=device,
    )
    return out_probs


# Compiled-graph cache for the Modular reference, keyed like the bridge's
# model cache: shape + dtype + device. One entry per test case shape.
_MODULAR_CACHE: dict[tuple, tuple] = {}


def modular_forward(
    x: np.ndarray,  # (rows, V) in storage form — no padding; slice before calling
    dtype_name: str,
    device: "Device | None" = None,
) -> np.ndarray:
    """Modular's own softmax (`max.graph.ops.softmax`) over the last axis.

    This routes to the production `nn.softmax` kernels that ship with MAX,
    so it cross-checks our kernel against Modular's implementation rather
    than only against PyTorch. Their op has no V/Vp split — callers slice
    the real vocab columns out first. Returns probs in storage form.
    """
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, TensorType, ops

    if device is None:
        device = pick_device()

    key = (dtype_name, x.shape, type(device).__name__, str(device))
    cached = _MODULAR_CACHE.get(key)
    if cached is None:
        dev_ref = DeviceRef.from_device(device)
        graph = Graph(
            "modular_softmax_reference",
            forward=lambda t: (ops.softmax(t, axis=-1),),
            input_types=[
                TensorType(max_dtype(dtype_name), shape=list(x.shape), device=dev_ref)
            ],
        )
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        _MODULAR_CACHE[key] = (session, model)
    else:
        _, model = cached

    (out,) = model.execute(_to_device_buffer(x, dtype_name, device))
    return _from_device_buffer(out, dtype_name)
