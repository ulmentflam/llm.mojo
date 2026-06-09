"""pytest fixtures + tolerances shared across the AdamW test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

# Per-dtype tolerances for max-abs-diff against the PyTorch reference.
# These start conservative — tighten as the kernel matures. Loosen only with
# a written rationale (the looser the tolerance, the less the test catches).
DTYPE_TOLERANCES: dict[str, dict[str, float]] = {
    "float32": {"atol": 1e-6, "rtol": 1e-5},
    "float16": {"atol": 1e-3, "rtol": 1e-2},
    "bfloat16": {"atol": 5e-3, "rtol": 2e-2},
}


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
