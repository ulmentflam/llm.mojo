"""The llmm Mojo kernels as one precompiled MAX custom-op package.

Every Python caller of a `@register`ed llmm kernel loads the same
`llmm.mojoc`: the pytest bridge (tests/_max_bridge.py) and the MAX GPT-2
scaffolds (`llmm_attention` / `llmm_gelu`). `make build-mojo` builds it;
`python -m llmm_pkg` does the same and prints its path.

The package is content-addressed: it lives at
`build/llmm_pkg/<fingerprint>/llmm.mojoc`, where the fingerprint hashes the
toolchain and every `llmm/*.mojo`. A source edit therefore resolves to a new
path instead of reusing a stale package, reverting the edit finds the old one
again, and no timestamp can make the two disagree (mtimes lie after
`git checkout` or an iCloud sync). `mojo precompile` output is not bit-stable
across identical sources, which is why the key hashes sources rather than the
package.

Why a prebuilt package at all: compiling `custom_extensions` from the SOURCE
dir makes MAX repackage it into one shared temp package
(/var/folders/.../.modular_*/mojo_pkg/, content-hashed name) on every Graph
build, rewritten non-atomically and read back immediately. That file was the
root of nondeterministic "Failed to compile the model" flakes: a process can
read its own half-written package, and concurrent pytest runs tear each
other's down (observed corrupt "invalid magic bytes" leftovers that poison
later runs until deleted). A prebuilt package is loaded directly; the temp dir
is never created.

Set LLMM_DISABLE_MEF_CACHE=1 to build into a per-process temp dir instead.
"""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "llmm"
CACHE_ROOT = REPO_ROOT / "build" / "llmm_pkg"
# The package embeds its module name from the build-time FILE NAME: any other
# name leaves the kernels under the wrong module prefix, which MAX's generated
# code then cannot resolve.
PKG_FILENAME = "llmm.mojoc"

_resolved: Path | None = None


def toolchain_fingerprint(h: hashlib._Hash) -> None:
    """Feed everything that invalidates every compiled artifact into `h`."""
    try:
        from max import _core

        h.update(f"max={_core.__version__}".encode())
    except Exception:
        h.update(b"max=unknown")
    mojo = shutil.which("mojo")
    if mojo:
        st = Path(mojo).resolve().stat()
        h.update(f"mojo={st.st_size}:{st.st_mtime_ns}".encode())


@functools.lru_cache(maxsize=1)
def package_fingerprint() -> str:
    """Hash over the toolchain and ALL of llmm/: the package is built from
    every module regardless of which kernel a caller wants."""
    h = hashlib.sha256()
    toolchain_fingerprint(h)
    for f in sorted(SOURCE_DIR.rglob("*.mojo")):
        h.update(f.relative_to(SOURCE_DIR).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def package_path() -> Path:
    """Where the package for the current sources lives (built or not)."""
    return CACHE_ROOT / package_fingerprint() / PKG_FILENAME


def ensure_llmm_package(echo_warnings: bool = False) -> Path:
    """Return the llmm package for the current sources, building it if absent.

    Memoised per process. Written via a scratch file + os.replace, so
    concurrent pytest workers never read a half-written package. With
    echo_warnings (the `make build-mojo` path), mojo's compile warnings are
    forwarded to stderr instead of swallowed.
    """
    global _resolved
    if _resolved is not None and _resolved.exists():
        return _resolved
    if os.environ.get("LLMM_DISABLE_MEF_CACHE"):
        target = Path(tempfile.mkdtemp(prefix="llmm_pkg_")) / PKG_FILENAME
    else:
        target = package_path()
        if target.exists():
            _resolved = target
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
    # Unique per-process scratch dir beside the target: same filesystem, so
    # the final os.replace is atomic, and the file keeps its required name.
    scratch = target.parent / f".pkg_build{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = scratch / PKG_FILENAME
    try:
        proc = subprocess.run(
            ["mojo", "precompile", str(SOURCE_DIR), "-o", str(tmp)],
            check=True,
            capture_output=True,
            text=True,
        )
        os.replace(tmp, target)
    except FileNotFoundError as e:
        raise RuntimeError(
            "`mojo` not on PATH; run via `pixi run` or `make` so the pixi env"
            " is active."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"mojo precompile failed:\n{e.stderr}") from e
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if echo_warnings and proc.stderr:
        noise = ("Crashpad",)
        for line in proc.stderr.splitlines():
            if not any(n in line for n in noise):
                print(line, file=sys.stderr)
    _resolved = target
    return target


if __name__ == "__main__":
    # `make build-mojo`: build (or reuse) the package and print its path.
    print(ensure_llmm_package(echo_warnings=True))
