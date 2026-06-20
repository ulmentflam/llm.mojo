"""Faithful Python transcription of llmc/sampler.h (Karpathy llm.c).

Used by tests/test_sampler.mojo to verify llmm.sampler.sample_softmax matches
the reference C float32/float64 semantics (expf + double norm).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

RU32_HEX = 0x2545F4914F6CDD1D
FLOAT_CONST = 16777216.0


def expf(x: float | np.floating) -> np.float32:
    """C expf: single-precision exp in, single-precision out."""
    with np.errstate(over="ignore", under="ignore"):
        return np.float32(np.exp(np.asarray(x, dtype=np.float32)))


def _coin_after_norm(coin: float, norm: np.float64) -> np.float32:
    """C coin *= norm: float promoted to double, product truncated to float."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.float32(np.float64(np.asarray(coin, dtype=np.float32)) * norm)


def sample_softmax_c(logits: Sequence[float] | np.ndarray, coin: float) -> int:
    """Matches int sample_softmax(const float* logits, int n, float coin)."""
    logits_f32 = np.asarray(logits, dtype=np.float32)
    n = int(logits_f32.shape[0])

    norm = np.float64(0.0)
    for i in range(n):
        norm += np.float64(expf(logits_f32[i]))

    coin_f32 = _coin_after_norm(coin, norm)

    cdf = np.float32(0.0)
    for i in range(n):
        cdf = np.asarray(cdf + expf(logits_f32[i]), dtype=np.float32)
        if coin_f32 < cdf:
            return i
    return n - 1


def make_logits(n: int, seed: int) -> np.ndarray:
    """Deterministic float32 logits in a GPT-2-like range for sweep tests."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n, dtype=np.float32) * np.float32(2.0)


def coin_sweep(num: int) -> np.ndarray:
    """Coins in [0, 1) avoiding exact 1.0 (same domain as random_f32)."""
    return (np.arange(num, dtype=np.float64) / num * 0.999999).astype(np.float32)
