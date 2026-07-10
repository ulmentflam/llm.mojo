"""Equivalence: Mojo attention_fwd vs PyTorch.

The forward is checked against torch.nn.functional.scaled_dot_product_attention
with causal masking. We also verify the log-sum-exp (L) vector the backward
pass will consume. Properties beyond plain closeness:

  *  Causal masking: mutating K/V at positions j > t must not change the
     output or L at query position t.
  *  Position zero: with only key 0 in the causal window, the output row
     equals V[:, :, 0, :] exactly.
  *  Compile-time paths: each FA4 softmax variant is checked against a Python
     port of the same online-softmax logic (not against torch).
  *  fp16 is omitted from the parametrized cases: MAX rejects f16 graphs on
     CPU (`pick_device()` defaults to CPU; see tests/_max_bridge.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests._dtypes import DTYPE_TOLERANCES, TORCH_DTYPES, from_storage, to_storage
from tests.kernels import attention

# Attention accumulates softmax statistics across O(T) keys and O(head_dim)
# FMAs before rounding to the storage dtype, so low-precision dtypes need
# more slack than pointwise ops (see matmul_equivalence.py for the pattern).
ATTENTION_TOLERANCES: dict[str, dict[str, float]] = {
    **DTYPE_TOLERANCES,
    "bfloat16": {"atol": 1.5e-2, "rtol": 3.5e-2},
    "float16": {"atol": 5e-3, "rtol": 2e-2},
}


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    dtype: str
    seed: int
    # True only for cases that must run on the GPU accelerator to mean
    # anything (see _device_for_case below) — independent of the module's
    # shared pick_device()/MAX_USE_ACCELERATOR default, which stays CPU.
    prefer_accelerator: bool = False
    # True for cases that must run on CPU specifically, e.g. to deliberately
    # exercise attention_fwd_cpu rather than relying on it as an accidental
    # fallback when no accelerator happens to be present. Takes precedence
    # over prefer_accelerator if both are somehow set on the same case.
    force_cpu: bool = False


def _device_for_case(case: "Case"):
    """Device override for cases that specifically need the accelerator, or
    specifically need to be pinned to CPU.

    Most cases pass `device=None` to run_custom_op, which falls back to the
    shared `pick_device()` default (CPU unless MAX_USE_ACCELERATOR=1 — see
    tests/_max_bridge.py). Cases with `prefer_accelerator=True` bypass that
    default entirely: they pick the accelerator directly if one is present,
    with no env var required, and fall back to CPU (not skip/fail) so the
    suite stays green on machines with no accelerator (e.g. a Metal dev
    box, or CPU-only CI).

    IMPORTANT: on CPU, attention dispatches to attention_fwd_cpu — a
    different function that never had the f0da883 alignment bug. A CPU
    fallback run of a prefer_accelerator case keeps the test passing
    everywhere, but only the GPU run actually exercises (and proves the fix
    for) attention_fwd_gemm's _attention_softmax_causal_gpu. Don't mistake
    a green CPU-fallback run for regression coverage.

    Two ways to deliberately exercise attention_fwd_cpu (not just as an
    accidental byproduct of missing hardware) even on a machine that has an
    accelerator available:
      1. Set LLMM_TEST_FORCE_CPU=1 in the environment — pins EVERY case in
         this module (including prefer_accelerator=True ones) to CPU,
         regardless of what hardware is present. Matches this repo's
         existing LLMM_-prefixed env var convention (see LLMM_USE_CPU,
         LLMM_FORCE_PORTABLE_GPU, LLMM_DISABLE_METAL elsewhere).
      2. Cases with force_cpu=True (e.g. the *_cpu_forced variants below)
         always run on CPU, every invocation, with no env var needed — so
         CI exercises attention_fwd_cpu on these shapes by default, not
         only when a developer remembers to opt in.
    """
    from max.driver import CPU, Accelerator, accelerator_count

    if os.environ.get("LLMM_TEST_FORCE_CPU"):
        return CPU()
    if case.force_cpu:
        return CPU()
    if not case.prefer_accelerator:
        return None
    return Accelerator() if accelerator_count() > 0 else CPU()


CASES: tuple[Case, ...] = (
    Case(
        "fp32_small",
        batch_size=2,
        num_heads=4,
        seq_len=16,
        head_dim=32,
        dtype="float32",
        seed=0,
    ),
    Case(
        "fp32_odd",
        batch_size=1,
        num_heads=2,
        seq_len=7,
        head_dim=13,
        dtype="float32",
        seed=1,
    ),
    # head_dim smaller than the SIMD width: the vector loop never runs.
    Case(
        "fp32_tiny_head",
        batch_size=2,
        num_heads=2,
        seq_len=8,
        head_dim=3,
        dtype="float32",
        seed=2,
    ),
    # GPU MAX_HEAD_DIM boundary (CPU path accepts any head_dim).
    Case(
        "fp32_max_head",
        batch_size=1,
        num_heads=2,
        seq_len=8,
        head_dim=128,
        dtype="float32",
        seed=3,
    ),
    Case(
        "fp32_seq_len_one",
        batch_size=2,
        num_heads=4,
        seq_len=1,
        head_dim=32,
        dtype="float32",
        seed=4,
    ),
    # Real GPT-2 small shape (partial)
    Case(
        "fp32_gpt2",
        batch_size=1,
        num_heads=12,
        seq_len=64,
        head_dim=64,
        dtype="float32",
        seed=5,
    ),
    Case(
        "bf16_small",
        batch_size=2,
        num_heads=4,
        seq_len=16,
        head_dim=32,
        dtype="bfloat16",
        seed=6,
    ),
    # Regression cases for the GEMM-attention softmax alignment bug (fixed in
    # f0da883, see docs/ai/bf16_generation_misaligned_address_bug.md).
    # attention_fwd_gemm's vectorized bf16 softmax kernel loads/stores 8-wide
    # (bf16's SIMD width) at per-row offset row*seq_len; that offset is only
    # a multiple of 8 when seq_len itself is. Training's fixed seq_len=1024
    # always satisfied this; token-by-token generation's seq_len=1,2,3,...
    # does not, and the first failure is at seq_len=9 (CUDA_ERROR_MISALIGNED_
    # ADDRESS on real hardware). prefer_accelerator=True routes these to the
    # GPU whenever one's present (see _device_for_case) — no env var needed —
    # since that's the only device that actually reaches the fixed kernel;
    # on a CPU-only machine these still pass, but via the unrelated
    # attention_fwd_cpu path, which never had this bug.
    Case(
        "bf16_seq_len_9_unaligned",
        batch_size=2,
        num_heads=4,
        seq_len=9,
        head_dim=32,
        dtype="bfloat16",
        seed=7,
        prefer_accelerator=True,
    ),
    Case(
        "bf16_seq_len_17_unaligned",
        batch_size=2,
        num_heads=4,
        seq_len=17,
        head_dim=32,
        dtype="bfloat16",
        seed=8,
        prefer_accelerator=True,
    ),
    # Deliberate CPU-pinned counterparts of the two cases above (force_cpu=
    # True — see _device_for_case): these always run attention_fwd_cpu on
    # these exact shapes, every invocation, with no env var needed. This is
    # the "on-demand, not just an accidental fallback" verification that the
    # CPU path (which never had the alignment bug — it doesn't share
    # attention_fwd_gemm's vectorized-load code) legitimately handles
    # non-8-aligned seq_len correctly too. To force EVERY case in this file
    # onto CPU instead (not just these two), run with LLMM_TEST_FORCE_CPU=1.
    Case(
        "bf16_seq_len_9_unaligned_cpu_forced",
        batch_size=2,
        num_heads=4,
        seq_len=9,
        head_dim=32,
        dtype="bfloat16",
        seed=7,
        force_cpu=True,
    ),
    Case(
        "bf16_seq_len_17_unaligned_cpu_forced",
        batch_size=2,
        num_heads=4,
        seq_len=17,
        head_dim=32,
        dtype="bfloat16",
        seed=8,
        force_cpu=True,
    ),
)


@dataclass(frozen=True)
class KernelParams:
    name: str
    use_soft_exp: bool
    use_conditional_rescale: bool


KERNEL_PARAM_CASES: tuple[KernelParams, ...] = (
    KernelParams(
        "exact_exp_no_deferred_rescale",
        use_soft_exp=False,
        use_conditional_rescale=False,
    ),
    KernelParams(
        "soft_exp_with_deferred_rescale",
        use_soft_exp=True,
        use_conditional_rescale=True,
    ),
    KernelParams(
        "soft_exp_no_deferred_rescale",
        use_soft_exp=True,
        use_conditional_rescale=False,
    ),
    KernelParams(
        "exact_exp_with_deferred_rescale",
        use_soft_exp=False,
        use_conditional_rescale=True,
    ),
)


# FlashAttention-4 softmax helpers (must match llmm/attention.mojo).
_LN_2 = np.float32(0.69314718056)
_LOG2_EXP_MIN = np.float32(-126.0)
_C4 = np.float32(0.009618129)
_C3 = np.float32(0.055504108)
_C2 = np.float32(0.240179544)
_C1 = np.float32(0.69314718)
_C0 = np.float32(1.0)
_TAU = np.float32(8.0)
_MIN_FINITE = np.float32(np.finfo(np.float32).min)


def _kernel_exp(x: float | np.floating, *, use_soft_exp: bool) -> np.float32:
    xf = np.float32(x)
    if xf <= np.float32(-103.0):
        return np.float32(0.0)
    if not use_soft_exp:
        return np.float32(np.exp(xf))
    log2_input = xf / _LN_2
    log2_input = np.float32(max(log2_input, _LOG2_EXP_MIN))
    integer_part = np.floor(log2_input)
    fractional_part = log2_input - integer_part
    polynomial = np.float32(
        (
            ((_C4 * fractional_part + _C3) * fractional_part + _C2) * fractional_part
            + _C1
        )
        * fractional_part
        + _C0
    )
    return np.float32(np.ldexp(polynomial, np.int32(integer_part)))


def _reference_kernel_path(
    q_np: np.ndarray,
    k_np: np.ndarray,
    v_np: np.ndarray,
    *,
    dtype_name: str,
    use_soft_exp: bool,
    use_conditional_rescale: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Slow causal-attention reference that mirrors the Mojo online-softmax paths."""
    q = q_np.astype(np.float32, copy=False)
    k = k_np.astype(np.float32, copy=False)
    v = v_np.astype(np.float32, copy=False)
    b, nh, seq_len, head_dim = q.shape
    scale = np.float32(1.0 / np.sqrt(head_dim))

    out = np.zeros((b, nh, seq_len, head_dim), dtype=np.float32)
    l_vec = np.zeros((b, nh, seq_len), dtype=np.float32)

    for bi in range(b):
        for hi in range(nh):
            for ti in range(seq_len):
                q_row = q[bi, hi, ti]
                output_row = np.zeros(head_dim, dtype=np.float32)

                max_deferred = _MIN_FINITE
                max_true = _MIN_FINITE
                denom_true = np.float32(0.0)
                output_rescale = np.float32(1.0)

                for kj in range(ti + 1):
                    score = np.float32(np.dot(q_row, k[bi, hi, kj]) * scale)
                    prev_true_max = max_true
                    max_true = np.float32(max(max_true, score))
                    true_rescale = _kernel_exp(
                        prev_true_max - max_true, use_soft_exp=use_soft_exp
                    )
                    true_weight = _kernel_exp(
                        score - max_true, use_soft_exp=use_soft_exp
                    )
                    denom_true = np.float32(true_rescale * denom_true + true_weight)

                    value_row = v[bi, hi, kj]
                    if use_conditional_rescale:
                        candidate_max = np.float32(max(max_deferred, score))
                        max_delta = np.float32(candidate_max - max_deferred)
                        if max_delta > _TAU:
                            rescale_factor = _kernel_exp(
                                max_deferred - candidate_max, use_soft_exp=use_soft_exp
                            )
                            attention_weight = _kernel_exp(
                                score - candidate_max, use_soft_exp=use_soft_exp
                            )
                            output_row = (
                                output_row * rescale_factor
                                + attention_weight * value_row
                            )
                            max_deferred = candidate_max
                        else:
                            attention_weight = _kernel_exp(
                                score - max_deferred, use_soft_exp=use_soft_exp
                            )
                            output_row = output_row + attention_weight * value_row
                    else:
                        prev_deferred_max = max_deferred
                        max_deferred = np.float32(max(max_deferred, score))
                        rescale_factor = _kernel_exp(
                            prev_deferred_max - max_deferred, use_soft_exp=use_soft_exp
                        )
                        attention_weight = _kernel_exp(
                            score - max_deferred, use_soft_exp=use_soft_exp
                        )
                        output_row = (
                            output_row * rescale_factor + attention_weight * value_row
                        )

                if use_conditional_rescale:
                    epilogue_scale = (
                        _kernel_exp(max_deferred - max_true, use_soft_exp=use_soft_exp)
                        * output_rescale
                    )
                else:
                    epilogue_scale = np.float32(1.0)

                output_row = output_row * (epilogue_scale / denom_true)
                l_vec[bi, hi, ti] = max_true + np.log(denom_true)

                if dtype_name == "float32":
                    out[bi, hi, ti] = output_row
                else:
                    out[bi, hi, ti] = from_storage(
                        to_storage(output_row.reshape(1, head_dim), dtype_name),
                        dtype_name,
                    )

    return out, l_vec


def _shape(case: Case) -> tuple[int, int, int, int]:
    return case.batch_size, case.num_heads, case.seq_len, case.head_dim


def _make_inputs(case: Case) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = torch.Generator().manual_seed(case.seed)
    shape = _shape(case)
    q = torch.randn(shape, generator=g)
    k = torch.randn(shape, generator=g)
    v = torch.randn(shape, generator=g)

    q = q.to(TORCH_DTYPES[case.dtype]).to(torch.float32)
    k = k.to(TORCH_DTYPES[case.dtype]).to(torch.float32)
    v = v.to(TORCH_DTYPES[case.dtype]).to(torch.float32)

    return q.numpy(), k.numpy(), v.numpy()


def _reference(
    q_np: np.ndarray,
    k_np: np.ndarray,
    v_np: np.ndarray,
    *,
    dtype_name: str = "float32",
) -> tuple[np.ndarray, np.ndarray]:
    q = torch.from_numpy(q_np)
    k = torch.from_numpy(k_np)
    v = torch.from_numpy(v_np)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d**0.5)

    t = q.size(-2)
    mask = torch.tril(torch.ones(t, t)).view(1, 1, t, t)
    scores = scores.masked_fill(mask == 0, float("-inf"))

    max_val = torch.max(scores, dim=-1).values
    l_vec = max_val + torch.log(
        torch.sum(torch.exp(scores - max_val.unsqueeze(-1)), dim=-1)
    )

    out_np = out.numpy()
    if dtype_name != "float32":
        # Kernel stores in the storage dtype; round the torch reference the same way.
        out_np = from_storage(to_storage(out_np, dtype_name), dtype_name)

    return out_np, l_vec.numpy()


def _run_kernel(
    case: Case,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    output: np.ndarray | None = None,
    l_vec: np.ndarray | None = None,
    use_soft_exp: bool = False,
    use_conditional_rescale: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    return attention.forward(
        q=to_storage(q, case.dtype),
        k=to_storage(k, case.dtype),
        v=to_storage(v, case.dtype),
        batch_size=case.batch_size,
        num_heads=case.num_heads,
        seq_len=case.seq_len,
        head_dim=case.head_dim,
        dtype_name=case.dtype,
        output=output,
        l_vec=l_vec,
        use_soft_exp=use_soft_exp,
        use_conditional_rescale=use_conditional_rescale,
        device=_device_for_case(case),
    )


def _assert_matches_reference(
    case: Case,
    got_out_storage: np.ndarray,
    got_l: np.ndarray,
    expected_out: np.ndarray,
    expected_l: np.ndarray,
    *,
    err_prefix: str,
    tol: dict[str, float] | None = None,
) -> None:
    got_out = from_storage(got_out_storage, case.dtype).reshape(expected_out.shape)

    if tol is None:
        tol = ATTENTION_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        got_out,
        expected_out,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{err_prefix}: attention output diverged from reference",
    )

    # L is stored in fp32, but it is derived from Q/K dot products that read the
    # storage dtype; for fp16/bf16 those scores can differ from torch SDPA by
    # ~1e-3 even when the output matches after rounding.
    if tol is not None:
        l_tol = tol
    elif case.dtype != "float32":
        l_tol = ATTENTION_TOLERANCES[case.dtype]
    else:
        l_tol = DTYPE_TOLERANCES["float32"]
    np.testing.assert_allclose(
        got_l,
        expected_l,
        atol=l_tol["atol"],
        rtol=l_tol["rtol"],
        err_msg=f"{err_prefix}: log-sum-exp diverged from reference",
    )


def _make_extreme_negative_score_qkv(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Q/K with dot products that yield very negative attention scores.

    Align both on axis 0 so score_ij = scale * k_ij[0] with q_ij[0] == 1.
    Causal keys per query sweep from +30 down to -92, matching the spread
    seen in GPT-2 layer-4 head 11 that exposed the soft_exp ldexp bug.
    """
    scale = np.float32(1.0 / np.sqrt(head_dim))
    q = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
    k = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
    v = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)

    max_score = np.float32(30.0)
    min_score = np.float32(-92.0)

    for bi in range(batch_size):
        for hi in range(num_heads):
            for ti in range(seq_len):
                q[bi, hi, ti, 0] = np.float32(1.0)
                for kj in range(ti + 1):
                    if ti == 0:
                        target_score = max_score
                    else:
                        frac = np.float32(kj) / np.float32(ti)
                        target_score = max_score + frac * (min_score - max_score)
                    k[bi, hi, kj, 0] = target_score / scale
                    v[bi, hi, kj, :] = np.float32(kj + 1)

    return q, k, v


def test_kernel_exp_large_negative_delta_is_finite():
    """Regression scalar: exp inputs near -91 must not hit broken Mojo ldexp paths."""
    y = _kernel_exp(np.float32(-91.41362), use_soft_exp=True)
    assert np.isfinite(y)
    assert y >= np.float32(0.0)
    assert y < np.float32(1e-30)


def test_soft_exp_finite_on_extreme_negative_score_deltas():
    """Regression: FA4 soft_exp must stay finite on wide causal score spreads.

    Before the log2 clamp (FA4 ex2_emulation uses max(x, -127) in log2 space),
    Mojo ldexp returned garbage for exponents below -128 and attention_fwd
    produced NaNs on real GPT-2 activations (layer-4 head 11).
    """
    case = Case(
        "fp32_soft_exp_extreme_scores",
        batch_size=4,
        num_heads=12,
        seq_len=64,
        head_dim=64,
        dtype="float32",
        seed=0,
    )
    q, k, v = _make_extreme_negative_score_qkv(
        case.batch_size,
        case.num_heads,
        case.seq_len,
        case.head_dim,
    )

    expected_out, expected_l = _reference_kernel_path(
        q,
        k,
        v,
        dtype_name=case.dtype,
        use_soft_exp=True,
        use_conditional_rescale=True,
    )

    got_out_storage, got_l = _run_kernel(
        case,
        q,
        k,
        v,
        use_soft_exp=True,
        use_conditional_rescale=True,
    )

    got_out = from_storage(got_out_storage, case.dtype).reshape(expected_out.shape)
    assert np.all(np.isfinite(got_out)), (
        "soft_exp attention output contains non-finite values"
    )
    assert np.all(np.isfinite(got_l)), "soft_exp log-sum-exp contains non-finite values"

    _assert_matches_reference(
        case,
        got_out_storage,
        got_l,
        expected_out,
        expected_l,
        err_prefix=case.name,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_forward_matches_torch(case: Case):
    q, k, v = _make_inputs(case)
    expected_out, expected_l = _reference(q, k, v, dtype_name=case.dtype)

    got_out_storage, got_l = _run_kernel(case, q, k, v)

    _assert_matches_reference(
        case,
        got_out_storage,
        got_l,
        expected_out,
        expected_l,
        err_prefix=case.name,
    )


@pytest.mark.parametrize("params", KERNEL_PARAM_CASES, ids=lambda p: p.name)
def test_compile_time_paths_match_reference(params: KernelParams):
    """Each comptime softmax path must match the Python port of the same logic."""
    case = next(c for c in CASES if c.name == "fp32_small")
    q, k, v = _make_inputs(case)

    expected_out, expected_l = _reference_kernel_path(
        q,
        k,
        v,
        dtype_name=case.dtype,
        use_soft_exp=params.use_soft_exp,
        use_conditional_rescale=params.use_conditional_rescale,
    )

    got_out_storage, got_l = _run_kernel(
        case,
        q,
        k,
        v,
        use_soft_exp=params.use_soft_exp,
        use_conditional_rescale=params.use_conditional_rescale,
    )

    tol = None
    if params.use_soft_exp:
        tol = {"atol": 2e-3, "rtol": 2e-3}

    _assert_matches_reference(
        case,
        got_out_storage,
        got_l,
        expected_out,
        expected_l,
        err_prefix=params.name,
        tol=tol,
    )


def test_causal_mask_respected():
    """Mutating K/V at positions j > t must not change output or L at t."""
    case = next(c for c in CASES if c.name == "fp32_small")
    q, k, v = _make_inputs(case)
    rng = np.random.default_rng(case.seed + 1000)

    got_out_storage, got_l = _run_kernel(case, q, k, v)
    got_out = from_storage(got_out_storage, case.dtype).reshape(_shape(case))

    for t in range(case.seq_len):
        k_alt = k.copy()
        v_alt = v.copy()
        if t + 1 < case.seq_len:
            future_shape = k_alt[:, :, t + 1 :, :].shape
            k_alt[:, :, t + 1 :, :] = rng.standard_normal(future_shape).astype(
                np.float32
            )
            v_alt[:, :, t + 1 :, :] = rng.standard_normal(future_shape).astype(
                np.float32
            )

        got_out_alt_storage, got_l_alt = _run_kernel(case, q, k_alt, v_alt)
        got_out_alt = from_storage(got_out_alt_storage, case.dtype).reshape(
            _shape(case)
        )

        np.testing.assert_allclose(
            got_out[:, :, t, :],
            got_out_alt[:, :, t, :],
            atol=0,
            rtol=0,
            err_msg=f"position {t} output changed when K/V after {t} changed",
        )
        np.testing.assert_allclose(
            got_l[:, :, t],
            got_l_alt[:, :, t],
            atol=0,
            rtol=0,
            err_msg=f"position {t} L changed when K/V after {t} changed",
        )


def test_position_zero_equals_value():
    """With only key 0 in the causal window, output row 0 equals V[:, :, 0]."""
    case = next(c for c in CASES if c.name == "fp32_small")
    q, k, v = _make_inputs(case)

    got_out_storage, _ = _run_kernel(case, q, k, v)
    got_out = from_storage(got_out_storage, case.dtype).reshape(_shape(case))

    tol = ATTENTION_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        got_out[:, :, 0, :],
        v[:, :, 0, :],
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg="position 0 output is not V[:, :, 0, :]",
    )


def test_uniform_scores_average_values():
    """When every causal score is equal, output at t is the mean of V[:, :, :t+1]."""
    case = Case(
        "uniform",
        batch_size=1,
        num_heads=2,
        seq_len=5,
        head_dim=8,
        dtype="float32",
        seed=42,
    )
    shape = _shape(case)
    q = np.ones(shape, dtype=np.float32)
    k = np.ones(shape, dtype=np.float32)
    v = np.broadcast_to(
        np.arange(case.head_dim, dtype=np.float32).reshape(1, 1, 1, case.head_dim),
        shape,
    ).copy()
    for t in range(case.seq_len):
        v[:, :, t, :] += float(t)

    expected_out, expected_l = _reference(q, k, v, dtype_name=case.dtype)
    got_out_storage, got_l = _run_kernel(case, q, k, v)

    _assert_matches_reference(
        case,
        got_out_storage,
        got_l,
        expected_out,
        expected_l,
        err_prefix="uniform",
    )


def test_output_overwrites_buffer():
    """The kernel must write every output element; sentinel-filled buffers
    should not survive in any position."""
    case = next(c for c in CASES if c.name == "fp32_small")
    q, k, v = _make_inputs(case)
    shape = _shape(case)

    sentinel = np.float32(7.0)
    output_in = np.full(shape, sentinel, dtype=np.float32)

    got_out_storage, _ = _run_kernel(case, q, k, v, output=output_in)
    got_out = from_storage(got_out_storage, case.dtype).reshape(shape)

    assert not np.any(got_out == sentinel), "kernel left sentinel values in output"


def _reference_bwd(q_np, k_np, v_np, d_out_np):
    """Reference attention backward pass using PyTorch autograd."""
    q = torch.from_numpy(q_np).requires_grad_(True)
    k = torch.from_numpy(k_np).requires_grad_(True)
    v = torch.from_numpy(v_np).requires_grad_(True)
    d_out = torch.from_numpy(d_out_np)

    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d**0.5)
    t = q.size(-2)
    mask = torch.tril(torch.ones(t, t)).view(1, 1, t, t)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = probs @ v

    out.backward(d_out)
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    return q.grad.numpy(), k.grad.numpy(), v.grad.numpy()


def _run_kernel_bwd(
    case: Case,
    d_output: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    output: np.ndarray,
    l_vec: np.ndarray,
    *,
    use_soft_exp: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_q = np.zeros_like(q)
    d_k = np.zeros_like(k)
    d_v = np.zeros_like(v)
    return attention.backward(
        d_q=to_storage(d_q, case.dtype),
        d_k=to_storage(d_k, case.dtype),
        d_v=to_storage(d_v, case.dtype),
        d_output=to_storage(d_output, case.dtype),
        q=to_storage(q, case.dtype),
        k=to_storage(k, case.dtype),
        v=to_storage(v, case.dtype),
        output=to_storage(output, case.dtype),
        l_vec=l_vec,
        batch_size=case.batch_size,
        num_heads=case.num_heads,
        seq_len=case.seq_len,
        head_dim=case.head_dim,
        dtype_name=case.dtype,
        use_soft_exp=use_soft_exp,
        device=_device_for_case(case),
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_backward_matches_torch(case: Case):
    q, k, v = _make_inputs(case)
    output_storage, got_l = _run_kernel(case, q, k, v)
    output = from_storage(output_storage, case.dtype).reshape(_shape(case))

    g = torch.Generator().manual_seed(case.seed + 999)
    d_output = torch.randn(_shape(case), generator=g).numpy()

    expected_dq, expected_dk, expected_dv = _reference_bwd(q, k, v, d_output)
    got_dq_storage, got_dk_storage, got_dv_storage = _run_kernel_bwd(
        case, d_output, q, k, v, output, got_l
    )

    got_dq = from_storage(got_dq_storage, case.dtype).reshape(_shape(case))
    got_dk = from_storage(got_dk_storage, case.dtype).reshape(_shape(case))
    got_dv = from_storage(got_dv_storage, case.dtype).reshape(_shape(case))

    tol = ATTENTION_TOLERANCES[case.dtype]

    np.testing.assert_allclose(
        got_dq,
        expected_dq,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: dq diverged from torch autograd reference",
    )
    np.testing.assert_allclose(
        got_dk,
        expected_dk,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: dk diverged from torch autograd reference",
    )
    np.testing.assert_allclose(
        got_dv,
        expected_dv,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: dv diverged from torch autograd reference",
    )
