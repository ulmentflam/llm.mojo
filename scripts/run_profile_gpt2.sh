#!/usr/bin/env bash
# Run build/profile_gpt2 with libpython wired up, mirroring run_train_gpt2.sh.
# train_gpt2 links Python interop (DataLoader's glob), so the profile harness —
# which imports GPT2 from it — needs MOJO_PYTHON_LIBRARY resolved the same way,
# especially on iCloud paths with spaces where auto-discovery breaks.
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
	echo "hint: run 'make install' from the repo root" >&2
	exit 1
fi

if [ ! -x "$ROOT/build/profile_gpt2" ]; then
	echo "error: $ROOT/build/profile_gpt2 not found (run: make build-profile)" >&2
	exit 1
fi

exec "$ROOT/build/profile_gpt2" "$@"
