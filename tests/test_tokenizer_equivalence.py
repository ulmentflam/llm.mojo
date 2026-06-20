"""Equivalence: Mojo Tokenizer vs Python binary reader / write_tokenizer."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests._tokenizer_fixtures import (
    SAMPLE_TOKENS,
    read_tokenizer_bin,
    write_test_tokenizer,
)

SAMPLE_TOKEN_IDS: tuple[int, ...] = tuple(range(len(SAMPLE_TOKENS)))


def _mojo_env() -> dict[str, str]:
    env = os.environ.copy()
    if "MOJO_PYTHON_LIBRARY" not in env:
        lib_dir = ".pixi/envs/default/lib"
        if os.path.isdir(lib_dir):
            for filename in os.listdir(lib_dir):
                if filename.startswith("libpython3") and (
                    filename.endswith(".dylib") or filename.endswith(".so")
                ):
                    env["MOJO_PYTHON_LIBRARY"] = os.path.join(lib_dir, filename)
                    break
    return env


def _run_mojo_bridge(
    path: Path, token_ids: tuple[int, ...]
) -> tuple[bool, int, int, dict[int, bytes]]:
    env = _mojo_env()
    env["TOKENIZER_BRIDGE_PATH"] = str(path)
    env["TOKENIZER_BRIDGE_IDS"] = ",".join(str(i) for i in token_ids)
    cmd = [
        "pixi",
        "run",
        "mojo",
        "-I",
        ".",
        "tests/_tokenizer_bridge.mojo",
    ]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            "Mojo tokenizer bridge failed "
            f"(exit {res.returncode})\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )

    lines = [line for line in res.stdout.splitlines() if line.strip()]
    has_initialized = lines[0] == "True"
    vocab_size = int(lines[1])
    eot_token = int(lines[2])
    decoded: dict[int, bytes] = {}
    for line in lines[3:]:
        if ":" not in line:
            continue
        token_id_str, *byte_strs = line.split(":", 1)
        if not token_id_str.isdigit():
            continue
        token_id = int(token_id_str)
        if byte_strs and byte_strs[0]:
            decoded[token_id] = bytes(int(b) for b in byte_strs[0].split())
        else:
            decoded[token_id] = b""
    return has_initialized, vocab_size, eot_token, decoded


@pytest.fixture(scope="module")
def test_tokenizer_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tokenizer") / "test_tokenizer.bin"
    write_test_tokenizer(path)
    return path


def test_write_tokenizer_roundtrip(test_tokenizer_bin: Path):
    vocab_size, eot_token, tokens = read_tokenizer_bin(test_tokenizer_bin)
    assert vocab_size == len(SAMPLE_TOKENS)
    assert eot_token == 3
    assert tokens == list(SAMPLE_TOKENS)


def test_mojo_metadata_matches_python(test_tokenizer_bin: Path):
    py_vocab, py_eot, _ = read_tokenizer_bin(test_tokenizer_bin)
    has_initialized, mojo_vocab, mojo_eot, _ = _run_mojo_bridge(
        test_tokenizer_bin, SAMPLE_TOKEN_IDS[:1]
    )
    assert has_initialized
    assert mojo_vocab == py_vocab
    assert mojo_eot == py_eot


def test_mojo_decode_matches_python_reader(test_tokenizer_bin: Path):
    _, _, py_tokens = read_tokenizer_bin(test_tokenizer_bin)
    _, _, _, mojo_decoded = _run_mojo_bridge(test_tokenizer_bin, SAMPLE_TOKEN_IDS)
    for token_id in SAMPLE_TOKEN_IDS:
        assert mojo_decoded[token_id] == py_tokens[token_id]


def test_mojo_decode_invalid_token_is_empty(test_tokenizer_bin: Path):
    invalid_id = len(SAMPLE_TOKENS) + 10
    _, _, _, mojo_decoded = _run_mojo_bridge(test_tokenizer_bin, (invalid_id,))
    assert mojo_decoded[invalid_id] == b""


def test_mojo_unit_suite():
    """Run the pure-Mojo tokenizer unit tests."""
    cmd = ["pixi", "run", "mojo", "-I", ".", "tests/test_tokenizer.mojo"]
    res = subprocess.run(cmd, env=_mojo_env(), capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(
            f"Mojo tokenizer unit tests failed (exit {res.returncode}).\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
