"""pytest fixtures shared across the kernel test suite.

Dtype maps, storage conversions, and comparison tolerances live in
`tests/_dtypes.py` — import from there, not from conftest.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# Parallelism: not worth it — run the suite sequentially. The only real
# cost is first-touch model compiles, cached per (kernel, dtype,
# parameters) within one process, and each compile is already
# multi-threaded, so xdist workers oversubscribe the cores. Measured
# 2026-06-12 (96 tests, M-series 10-core): sequential 244s; `-n 6 --dist
# loadfile` 220s at best and slower on a loaded machine; `--dist load`
# 297s and per-(module, dtype) loadgroup 428s (both duplicate compiles
# across workers). If you do run parallel anyway, loadfile is the only
# sane mode, and it REQUIRES the _packaged_kernels fixture below —
# without it, concurrent compiles tear MAX's shared temp mojopkg and fail
# randomly. (The fixture matters sequentially too: MAX can race its own
# temp-package write within one process.)


@pytest.fixture(scope="session", autouse=True)
def _packaged_kernels(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Build llmm.mojopkg once per session and point the bridge at it.

    Compiling custom_extensions from the SOURCE dir makes MAX repackage it
    into one shared temp .mojopkg (/var/folders/.../.modular_*/mojo_pkg/,
    content-hashed name) on every Graph build, rewritten non-atomically and
    read back immediately. That file is the root of the nondeterministic
    "Failed to compile the model" flakes: a single process can read its own
    half-written package, and concurrent pytest runs tear each other's down
    (observed corrupt "invalid magic bytes" leftovers that poison later
    runs until deleted). A prebuilt package is loaded directly — the temp
    dir is never created (verified) — and tmp_path_factory is per-process,
    so xdist workers don't share the artifact either.
    """
    from tests import _max_bridge

    src = _max_bridge.MOJO_KERNELS_DIR
    pkg = tmp_path_factory.mktemp("llmm_pkg") / "llmm.mojopkg"
    try:
        subprocess.run(
            ["mojo", "package", str(src), "-o", str(pkg)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "`mojo` not on PATH; run the suite via `pixi run pytest` or "
            "activate the pixi env (see CLAUDE notes)."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"mojo package failed:\n{e.stderr}") from e
    _max_bridge.MOJO_KERNELS_DIR = pkg


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--regen-fixtures",
        action="store_true",
        default=False,
        help="Regenerate tests/fixtures/*.npz from tests/reference.py before running.",
    )


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    from tests.reference import FIXTURES_DIR

    return FIXTURES_DIR


@pytest.fixture(scope="session", autouse=True)
def _maybe_regen_fixtures(request: pytest.FixtureRequest, fixtures_dir: Path) -> None:
    if (
        request.config.getoption("--regen-fixtures")
        or not (fixtures_dir / "manifest.json").exists()
    ):
        from tests.reference import dump

        dump()


# `max` is a hard pixi dependency (see pixi.toml). If it's missing the
# equivalence tests should fail with an ImportError rather than skip.
