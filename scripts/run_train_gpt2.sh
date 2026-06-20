#!/usr/bin/env bash
# Run train_gpt2 with libpython wired for DataLoader's Python interop.
# Use this instead of ./build/train_gpt2 directly — especially on iCloud paths
# with spaces, where an unset MOJO_PYTHON_LIBRARY breaks auto-discovery.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
	echo "hint: run 'pixi install' from the repo root" >&2
	exit 1
fi

if [ ! -x "$ROOT/build/train_gpt2" ]; then
	echo "error: $ROOT/build/train_gpt2 not found (run: make build)" >&2
	exit 1
fi

exec "$ROOT/build/train_gpt2" "$@"
