"""Shared fixtures for the Shiko test suite.

The milestone entry points run real installations and, for the host phase, a real
Hermes process. Helpers live in :mod:`tests.support`; fixtures here only manage
expensive, session-scoped artifacts.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

# The repository root carries an ``__init__.py`` on purpose: Hermes imports the
# plugin directory as a package. That makes the root itself look like a package
# named "shiko", so an interpreter with its *parent* on sys.path would import the
# plugin door instead of the core. Pin the root at the front of the path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.support import EVIDENCE_ARTIFACTS, REPO_ROOT, run  # noqa: E402


@pytest.fixture(scope="session")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """Build the wheel and sdist once per session."""
    outdir = tmp_path_factory.mktemp("dist")
    run([sys.executable, "-m", "build", "--outdir", str(outdir), str(REPO_ROOT)], timeout=900)
    wheels = sorted(outdir.glob("*.whl"))
    sdists = sorted(outdir.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected one wheel, found {wheels}"
    assert len(sdists) == 1, f"expected one sdist, found {sdists}"
    return outdir, wheels[0], sdists[0]


@pytest.fixture()
def artifact_dir() -> Path:
    if EVIDENCE_ARTIFACTS.exists():
        shutil.rmtree(EVIDENCE_ARTIFACTS)
    EVIDENCE_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return EVIDENCE_ARTIFACTS
