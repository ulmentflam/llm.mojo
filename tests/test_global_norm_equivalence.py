"""Equivalence: Mojo global_norm_squared vs PyTorch / NumPy reference.

Verify that:
  * Mojo global_norm_squared correctly computes the sum of squares of a tensor.
  * The accumulation (reset=False) accumulates on top of the existing buffer.
  * The input tensor remains unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tests._dtypes import DTYPE_TOLERANCES, TORCH_DTYPES, to_storage
from tests.kernels import global_norm


@dataclass(frozen=True)
class Case:
    """Fixed seed + shape + dtype uniquely determine the operand."""

    name: str
    num_slices: int
    cols: int
    dtype: str  # "float32" | "bfloat16" | "float16"
    seed: int


CASES: tuple[Case, ...] = (
    # Small single slice (1D case)
    Case("fp32_single_slice", num_slices=1, cols=128, dtype="float32", seed=0),
    # Multiple slices (2D case)
    Case("fp32_multiple_slices", num_slices=4, cols=256, dtype="float32", seed=1),
    # Odd sizes: cols % simd width != 0 to check the CPU vectorize tail
    Case("fp32_odd_sizes", num_slices=3, cols=33, dtype="float32", seed=2),
    # Large typical shape
    Case("fp32_large", num_slices=8, cols=1024, dtype="float32", seed=3),
    # BFloat16 testing
    Case("bf16_small", num_slices=2, cols=128, dtype="bfloat16", seed=4),
    Case("bf16_odd_sizes", num_slices=5, cols=37, dtype="bfloat16", seed=5),
    # Real GPT-2 shapes: WPE (1024, 768) and MLP (3072, 768)
    Case("fp32_gpt2_wpe", num_slices=1024, cols=768, dtype="float32", seed=6),
    Case("fp32_gpt2_mlp", num_slices=3072, cols=768, dtype="float32", seed=7),
    Case("bf16_gpt2_wpe", num_slices=1024, cols=768, dtype="bfloat16", seed=8),
)


def _ids(case: Case) -> str:
    return case.name


def _make_input(case: Case) -> np.ndarray:
    """Generate unit-scale input rounded to the target dtype, in float32."""
    g = torch.Generator().manual_seed(case.seed)
    td = TORCH_DTYPES[case.dtype]
    t = torch.randn(case.num_slices, case.cols, generator=g)
    return t.to(td).to(torch.float32).numpy()


def _run_kernel(
    case: Case,
    data: np.ndarray,
    reset: bool = True,
    output: np.ndarray | None = None,
) -> np.ndarray:
    # max_num_block_sums must be non-negative
    max_num_block_sums = 4096 + case.num_slices

    # Size output buffer to max_num_block_sums to satisfy GPU workspace size.
    if output is None:
        output = np.zeros(max_num_block_sums, dtype=np.float32)
    elif len(output) < max_num_block_sums:
        new_output = np.zeros(max_num_block_sums, dtype=np.float32)
        new_output[0] = output[0]
        output = new_output

    # Convert data to storage format (e.g. uint16 for bf16)
    data_storage = to_storage(data, case.dtype)

    # Since the array is contiguous, the stride is the number of columns
    stride = case.cols

    return global_norm.global_norm_squared(
        output=output,
        data=data_storage,
        stride=stride,
        num_slices=case.num_slices,
        max_num_block_sums=max_num_block_sums,
        reset=reset,
        dtype_name=case.dtype,
    )


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_forward_matches_reference(case: Case) -> None:
    data = _make_input(case)
    # Calculate sum of squares in double precision to avoid precision issues
    expected = np.sum(data.astype(np.float64) ** 2)

    got = _run_kernel(case, data, reset=True)

    tol = DTYPE_TOLERANCES[case.dtype]

    # For bfloat16 and float16, the accumulative error of summing many elements
    # can grow. So we scale the absolute tolerance by the number of elements.
    num_elements = case.num_slices * case.cols
    scaled_atol = tol["atol"] * np.sqrt(num_elements)

    np.testing.assert_allclose(
        got,
        expected,
        atol=scaled_atol,
        rtol=tol["rtol"],
        err_msg=f"{case.name}: global norm squared diverged from reference",
    )


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_accumulation_matches_reference(case: Case) -> None:
    """Verify that multiple passes with reset=False correctly accumulate the norm squared."""
    data1 = _make_input(case)

    # Make second input different by changing seed
    case2 = Case(
        name=case.name,
        num_slices=case.num_slices,
        cols=case.cols,
        dtype=case.dtype,
        seed=case.seed + 100,
    )
    data2 = _make_input(case2)

    expected1 = np.sum(data1.astype(np.float64) ** 2)
    expected2 = np.sum(data2.astype(np.float64) ** 2)
    expected_total = expected1 + expected2

    # First run with reset=True
    output = np.zeros(1, dtype=np.float32)

    out1 = _run_kernel(case, data1, reset=True, output=output)

    tol = DTYPE_TOLERANCES[case.dtype]
    num_elements = case.num_slices * case.cols
    scaled_atol = tol["atol"] * np.sqrt(num_elements)

    np.testing.assert_allclose(
        out1,
        expected1,
        atol=scaled_atol,
        rtol=tol["rtol"],
    )

    # Second run with reset=False, mutating output in-place
    out2 = _run_kernel(case2, data2, reset=False, output=out1)

    np.testing.assert_allclose(
        out2,
        expected_total,
        atol=scaled_atol * np.sqrt(2),
        rtol=tol["rtol"],
        err_msg=f"{case.name}: accumulation failed to match reference",
    )


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_input_survives(case: Case) -> None:
    """Verify that the input tensor is not modified by the operation."""
    data = _make_input(case)
    pristine = data.copy()

    _run_kernel(case, data, reset=True)
    np.testing.assert_array_equal(data, pristine)
