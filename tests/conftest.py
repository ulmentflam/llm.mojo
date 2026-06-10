"""pytest fixtures shared across the kernel test suite.

Dtype maps, storage conversions, and comparison tolerances live in
`tests/_dtypes.py` — import from there, not from conftest.
"""

from __future__ import annotations

from pathlib import Path

import pytest


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
