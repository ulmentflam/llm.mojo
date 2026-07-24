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
# milliseconds. Precompiling llmm.mojoc is also lazy and cached there.
#
# Parallelism: `make test-python` runs `-n auto` — worker processes use the
# machine's cores and isolate the blast radius of MAX's CPU execute crash
# (docs/ai/max_cpu_custom_op_crash_2026-07-24.md) to one worker. The
# trade-off on SMALL-core machines: on a cold cache each MAX compile is
# already multi-threaded, so workers oversubscribe and duplicate compiles
# (measured 2026-06-12 pre-MEF-cache, 96 tests, M-series 10-core:
# sequential 244s; `-n 6 --dist loadfile` 220s at best; `--dist load`
# 297s). If tuning dist modes, loadfile is the only sane one; the bridge's
# prebuilt-package + atomic cache writes keep concurrent workers from
# corrupting each other.


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--regen-fixtures",
        action="store_true",
        default=False,
        help="Regenerate tests/fixtures/*.npz from tests/reference.py before running.",
    )


def pytest_configure(config: pytest.Config) -> None:
    # PyTorch 2.x emits these when `import torch` loads `torch.jit`; the reference
    # code does not use JIT — suppress the noise across the equivalence suite.
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*torch.jit.script.*:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*torch.jit.interface.*:DeprecationWarning",
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
