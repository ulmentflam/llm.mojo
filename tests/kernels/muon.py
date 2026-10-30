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


def step(
    *,
    params: np.ndarray,
    m: np.ndarray,
    grads: np.ndarray,
    t: int,
    hp,  # tests.reference.MuonParams
    dtype_name: str,
    device: "Device | None" = None,
) -> StepResult:
    """One Muon step end-to-end through MAX. params/m/grads are [rows, cols];
    m is always float32. Arg order is the contract with the Mojo `muon_update`."""
    out_params, out_m = run_custom_op(
        kernel_name="muon_update",
        args=[
            MutableBuf(params, dtype_name),
            MutableBuf(m, "float32"),
            ScalarArg(t, "uint32"),
            ReadTensor(grads, dtype_name),
            ScalarArg(hp.lr, dtype_name),
            ScalarArg(hp.momentum, dtype_name),
            ScalarArg(hp.weight_decay, dtype_name),
            ScalarArg(hp.grad_scale, dtype_name),
        ],
        device=device,
    )
    return StepResult(params=out_params, m=out_m)
