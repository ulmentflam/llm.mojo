from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tests._max_bridge import MutableBuf, ReadTensor, ScalarArg, run_custom_op

if TYPE_CHECKING:
    from max.driver import Device


@dataclass
class StepResult:
    params: np.ndarray
    m: np.ndarray
    v: np.ndarray


def step(
    *,
    params: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    grads: np.ndarray,
    t: int,
    hp,  # tests.reference.AdamWParams
    dtype_name: str,
    device: "Device | None" = None,
) -> StepResult:
    """One AdamW step end-to-end through MAX.

    Inputs are numpy arrays of shape (n,); m/v are always float32 to match
    the kernel's Float32 moment storage. `params`, `m`, and `v` are mutated
    through `MutableInputTensor` buffers; the returned arrays are reads of
    those same post-execution buffers.
    """
    out_params, out_m, out_v = run_custom_op(
        kernel_name="adamw_update",
        args=[
            MutableBuf(params, dtype_name),
            MutableBuf(m, "float32"),
            MutableBuf(v, "float32"),
            ScalarArg(int(t), "uint32"),
            ReadTensor(grads, dtype_name),
            ScalarArg(hp.lr, dtype_name),
            ScalarArg(hp.beta1, dtype_name),
            ScalarArg(hp.beta2, dtype_name),
            ScalarArg(hp.eps, dtype_name),
            ScalarArg(hp.weight_decay, dtype_name),
            ScalarArg(hp.grad_scale, dtype_name),
        ],
        device=device,
    )
    return StepResult(params=out_params, m=out_m, v=out_v)
