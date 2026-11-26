"""Host revision facts for the milestone evidence log."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REAL_HOME = Path(os.environ.get("SHIKO_REAL_HERMES_HOME") or Path.home() / ".hermes")
HERMES_BIN = Path(os.environ.get("SHIKO_HERMES_BIN") or Path.home() / ".local" / "bin" / "hermes")


def hermes_versions() -> dict[str, str]:
    """Version and git revision of the installed host, for reproducibility."""
    facts: dict[str, str] = {"hermes_bin": str(HERMES_BIN), "real_home": str(REAL_HOME)}
    try:
        completed = subprocess.run(
            [str(HERMES_BIN), "--version"], capture_output=True, text=True, timeout=60, check=False
        )
        facts["hermes_version"] = " | ".join(
            line.strip() for line in (completed.stdout + completed.stderr).splitlines() if line.strip()
        )[:500]
    except OSError as exc:
        facts["hermes_version"] = f"unavailable: {exc}"
    repo = REAL_HOME / "hermes-agent"
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "log", "-1", "--format=%H %ci %s"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        facts["hermes_revision"] = completed.stdout.strip()
    except OSError as exc:
        facts["hermes_revision"] = f"unavailable: {exc}"
    return facts
