import os
import subprocess

import pytest

# Coarse, direct regression guard for the f0da883 attention-softmax alignment
# bug (docs/ai/bf16_generation_misaligned_address_bug.md): runs the actual
# checkpoint-load + B=1 generation path a handful of times via the standalone
# infer_gpt2_bf16 binary and asserts none of them crash. This complements
# tests/test_attention_equivalence.py's kernel-level equivalence cases
# (bf16_seq_len_9_unaligned / bf16_seq_len_17_unaligned), which pin down the
# exact fault; this test instead exercises the full production path (real
# checkpoint, real tokenizer, real end-to-end forward) the way the crash
# actually happened. Kept short (few trials, short generation length) so it
# stays fast.
#
# Skips (does not fail) if the inference binary or a checkpoint aren't
# present — this is a smoke test on top of already-produced artifacts
# (`make build-infer-bf16`, a trained checkpoint), not something every CI
# environment is expected to have built/downloaded.

_BIN = "build/infer_gpt2_bf16"
_CHECKPOINT = "log124M/model_19552.bin"
_GEN_LENGTH = 24
_SEEDS = (1, 2, 3, 4, 5)


def _mojo_python_library_env() -> dict:
    env = os.environ.copy()
    if "MOJO_PYTHON_LIBRARY" not in env:
        lib_dir = ".pixi/envs/default/lib"
        if os.path.exists(lib_dir):
            for filename in os.listdir(lib_dir):
                if filename.startswith("libpython3") and (
                    filename.endswith(".dylib") or filename.endswith(".so")
                ):
                    env["MOJO_PYTHON_LIBRARY"] = os.path.join(lib_dir, filename)
                    break
    return env


@pytest.mark.parametrize("seed", _SEEDS)
def test_generation_does_not_crash(seed: int) -> None:
    if not os.path.exists(_BIN):
        pytest.skip(f"{_BIN} not built — run `make build-infer-bf16` first")
    if not os.path.exists(_CHECKPOINT):
        pytest.skip(f"{_CHECKPOINT} not present — no trained checkpoint to load")

    env = _mojo_python_library_env()
    cmd = [_BIN, _CHECKPOINT, str(_GEN_LENGTH), str(seed)]
    # GPT-2's byte-level BPE can legitimately decode to a partial multi-byte
    # UTF-8 sequence at a generation boundary; errors="replace" avoids a
    # spurious UnicodeDecodeError in the test harness for output we're only
    # scanning for crash/error markers, not validating byte-for-byte.
    res = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )

    if res.returncode != 0:
        print("STDOUT:")
        print(res.stdout)
        print("STDERR:")
        print(res.stderr)

    assert res.returncode == 0, (
        f"infer_gpt2_bf16 crashed (seed={seed}, exit={res.returncode}) — "
        f"possible regression of the f0da883 misaligned-address fix.\n"
        f"STDERR: {res.stderr}"
    )
    assert "CUDA_ERROR" not in res.stderr
    assert "generating:" in res.stdout
