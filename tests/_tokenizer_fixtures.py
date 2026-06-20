"""Helpers for tokenizer tests (Python side)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from train_gpt2 import MAGIC_NUMBER

TOKENIZER_MAGIC_LEGACY = 20240328

SAMPLE_TOKENS: tuple[bytes, ...] = (
    b"hello",
    b" world",
    b"!",
    b"\x07",
    b"\xff",
    b" the",
    b" quick",
    b" brown",
)


def write_test_tokenizer(path: str | Path) -> None:
    """Write a small version-2 tokenizer file for unit tests."""
    path = Path(path)
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC_NUMBER
    header[1] = 2
    header[2] = len(SAMPLE_TOKENS)
    header[3] = 3
    with path.open("wb") as f:
        f.write(header.tobytes())
        for token in SAMPLE_TOKENS:
            f.write(struct.pack("<B", len(token)))
            f.write(token)


def read_tokenizer_bin(path: str | Path) -> tuple[int, int, list[bytes]]:
    """Mirror llm.c / llmm/tokenizer.mojo binary load."""
    path = Path(path)
    with path.open("rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        magic = int(header[0])
        if magic not in (MAGIC_NUMBER, TOKENIZER_MAGIC_LEGACY):
            raise ValueError(f"bad tokenizer magic: {magic}")
        version = int(header[1])
        vocab_size = int(header[2])
        eot_token = 50256 if version == 1 else int(header[3])
        tokens: list[bytes] = []
        for _ in range(vocab_size):
            (length,) = struct.unpack("<B", f.read(1))
            if length <= 0:
                raise ValueError("token length must be positive")
            tokens.append(f.read(length))
    return vocab_size, eot_token, tokens
