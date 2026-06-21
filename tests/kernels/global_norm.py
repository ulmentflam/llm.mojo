from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from tests._max_bridge import MutableBuf, ReadTensor, ScalarArg, run_custom_op

if TYPE_CHECKING:
    from max.driver import Device


def global_norm_squared(
    *,
    output: np.ndarray,
    data: np.ndarray,
    stride: int,
    num_slices: int,
    max_num_block_sums: int,
    reset: bool,
    dtype_name: str,
    device: "Device | None" = None,
) -> np.ndarray:
    """Computes the sum of squares of a tensor's elements."""
    out_params = run_custom_op(
        kernel_name="global_norm_squared",
        args=[
            MutableBuf(output, "float32"),
            ReadTensor(data, dtype_name),
            ScalarArg(stride, "int64"),
            ScalarArg(num_slices, "int64"),
            ScalarArg(max_num_block_sums, "int64"),
            ScalarArg(int(reset), "uint32"),
        ],
        device=device,
    )
    return out_params[0][:1]
