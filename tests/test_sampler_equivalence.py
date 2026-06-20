"""Reference checks for llmc/sampler.h sample_softmax semantics.

Mojo-vs-llm.c equality is enforced in tests/test_sampler.mojo (make test-mojo).
This file pins the Python reference with closed-form and sweep cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._sampler_reference import coin_sweep, expf, make_logits, sample_softmax_c


def test_single_class_always_index_zero() -> None:
    logits = np.array([2.5], dtype=np.float32)
    for coin in (0.0, 0.5, 0.999):
        assert sample_softmax_c(logits, coin) == 0


def test_uniform_four_way_partition() -> None:
    logits = np.zeros(4, dtype=np.float32)
    cases = (0.0, 0.24, 0.25, 0.49, 0.50, 0.74, 0.75, 0.999)
    expected = (0, 0, 1, 1, 2, 2, 3, 3)
    for coin, want in zip(cases, expected, strict=True):
        assert sample_softmax_c(logits, coin) == want


def test_peaked_distribution() -> None:
    logits = np.array([-1.0, 0.0, 10.0, -2.0, -3.0], dtype=np.float32)
    assert sample_softmax_c(logits, 0.0) == 0
    assert sample_softmax_c(logits, 0.999) == 2


@pytest.mark.parametrize("n,seed", [(32, 0), (128, 7), (512, 13), (1024, 21)])
def test_reference_coin_sweep_deterministic(n: int, seed: int) -> None:
    logits = make_logits(n, seed)
    indices = [sample_softmax_c(logits, float(coin)) for coin in coin_sweep(100)]
    assert all(0 <= i < n for i in indices)
    again = [sample_softmax_c(logits, float(coin)) for coin in coin_sweep(100)]
    assert indices == again


def test_large_vocab_like() -> None:
    logits = make_logits(50304, 42)
    for coin in coin_sweep(25):
        idx = sample_softmax_c(logits, float(coin))
        assert 0 <= idx < 50304


def test_extreme_logits() -> None:
    logits = np.array([-100.0, -50.0, 0.0, 50.0, 100.0, -100.0], dtype=np.float32)
    for coin in coin_sweep(30):
        idx = sample_softmax_c(logits, float(coin))
        assert 0 <= idx < len(logits)


def test_expf_is_float32() -> None:
    x = np.float32(88.0)
    assert expf(x) == np.float32(np.exp(x))
