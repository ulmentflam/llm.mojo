"""Shared dtype utilities for the test suite.

numpy has no bfloat16, so the suite-wide convention is: bf16 arrays are
stored as their raw uint16 bit patterns, and conversions round-trip
through torch views at the edges (`to_storage` / `from_storage`). This
module is the single home for that convention, the dtype-name maps, and
the per-dtype comparison tolerances. Anything that grows a second copy
of these belongs here instead.
"""

from __future__ import annotations

import numpy as np
import torch

import os

use_accelerator = os.environ.get("MAX_USE_ACCELERATOR") == "1"

# Per-dtype tolerances for comparisons against the PyTorch reference.
# These start conservative — tighten as the kernels mature. Loosen only with
# a written rationale (the looser the tolerance, the less the test catches).
DTYPE_TOLERANCES: dict[str, dict[str, float]] = {
    "float32": {
        "atol": 2e-5 if use_accelerator else 1e-6,
        "rtol": 1e-4 if use_accelerator else 1e-5,
    },
    "float16": {"atol": 1e-3, "rtol": 1e-2},
    "bfloat16": {"atol": 5e-3, "rtol": 2e-2},
}

TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# numpy storage dtype per kernel dtype; bf16 is stored as uint16 bits.
NP_STORAGE_DTYPES: dict[str, np.dtype] = {
    "float32": np.dtype("float32"),
    "bfloat16": np.dtype("uint16"),
    "float16": np.dtype("float16"),
}


def to_storage(arr_f32: np.ndarray, dtype_name: str) -> np.ndarray:
    """Round an fp32 array to `dtype_name`, returned in storage form."""
    if dtype_name == "bfloat16":
        return (
            torch.from_numpy(np.ascontiguousarray(arr_f32))
            .to(torch.bfloat16)
            .view(torch.uint16)
            .numpy()
        )
    return arr_f32.astype(NP_STORAGE_DTYPES[dtype_name])


def from_storage(arr: np.ndarray, dtype_name: str) -> np.ndarray:
    """Inverse of `to_storage`: storage bytes back to an fp32 array."""
    if dtype_name == "bfloat16":
        return torch.from_numpy(arr).view(torch.bfloat16).to(torch.float32).numpy()
    return arr.astype(np.float32)
