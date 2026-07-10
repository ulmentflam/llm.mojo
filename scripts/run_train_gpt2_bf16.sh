#!/usr/bin/env bash
# Run train_gpt2_bf16 with libpython wired for DataLoader's Python interop.
# Mirrors run_train_gpt2.sh (the fp32 launcher) for the bf16 binary. All
# training flags (-i/-j data patterns, -e model descriptor, -o output dir,
# hyperparameters, ...) are passed straight through — this script carries no
# run-specific defaults. For an example full invocation (the GPT-2 124M /
# FineWeb 10B reproduction), see docs/ai/gpt2_124m_fineweb_training_run.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${BIN:-build/train_gpt2_bf16}"

if [ -z "${MOJO_PYTHON_LIBRARY:-}" ]; then
	for lib in .pixi/envs/default/lib/libpython3*.dylib \
		.pixi/envs/default/lib/libpython3*.so; do
		if [ -f "$lib" ]; then
			export MOJO_PYTHON_LIBRARY="$lib"
			break
		fi
	done
fi

if [ -z "${MOJO_PYTHON_LIBRARY:-}" ]; then
	echo "error: libpython not found under .pixi/envs/default/lib" >&2
	echo "hint: run 'make install' from the repo root" >&2
	exit 1
fi

if [ ! -x "$ROOT/$BIN" ]; then
	echo "error: $ROOT/$BIN not found (run: make build-bf16, or set BIN=)" >&2
	exit 1
fi

exec "$ROOT/$BIN" "$@"
