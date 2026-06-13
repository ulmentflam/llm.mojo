"""pytest fixtures shared across the kernel test suite.

Dtype maps, storage conversions, and comparison tolerances live in
`tests/_dtypes.py` — import from there, not from conftest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Speed: model compiles (4-10s per (kernel, dtype, parameters), ~30 in the
# suite) dominate a cold run. The bridge persists every compiled model as
# MEF under tests/.mef_cache/ (see tests/_max_bridge.py), so a compile is
# paid once per kernel-source change, not once per run; warm runs load in
# milliseconds. Packaging llmm.mojopkg is also lazy and cached there.
#
# Parallelism: not worth it — run the suite sequentially. With a warm MEF
# cache there is nothing left for xdist to parallelize but kernel
# execution; on a cold cache each compile is already multi-threaded, so
# workers oversubscribe the cores AND duplicate compiles for any key two
# workers both touch (each exports its own MEF, so the waste is per-run,
# not persistent). Measured 2026-06-12 pre-MEF-cache (96 tests, M-series
# 10-core): sequential 244s; `-n 6 --dist loadfile` 220s at best and
# slower on a loaded machine; `--dist load` 297s and per-(module, dtype)
# loadgroup 428s. If you do run parallel anyway, loadfile is the only sane
# mode; the bridge's prebuilt-package + atomic cache writes keep
# concurrent workers from corrupting each other.


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
