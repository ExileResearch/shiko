"""Helpers shared by the milestone entry points.

Kept out of ``conftest.py`` so the test modules can import them as ordinary
functions (``from tests.support import ...``) while the fixtures stay small.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MILESTONES = REPO_ROOT / "tests" / "milestones"
EVIDENCE_ARTIFACTS = REPO_ROOT / "notes" / "plans" / "evidence" / "artifacts" / "milestone-1"

HOST_ENV_KEYS = ("PYTHONPATH", "PYTHONHOME", "SHIKO_HOME", "SHIKO_OWNER", "HERMES_HOME")


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in HOST_ENV_KEYS}
    env.update(extra or {})
    return env


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 600,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=str(cwd or REPO_ROOT),
        env=clean_env(env),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {' '.join(args)}\n"
            f"--- stdout ---\n{completed.stdout[-4000:]}\n"
            f"--- stderr ---\n{completed.stderr[-4000:]}"
        )
    return completed


def make_venv(path: Path) -> Path:
    """Create a fresh virtual environment and return its interpreter path."""
    venv.EnvBuilder(with_pip=True, clear=True, symlinks=True).create(str(path))
    exe = path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assert exe.exists(), f"venv interpreter missing at {exe}"
    return exe


def pip_install(python: Path, *targets: str, offline: bool = False, timeout: int = 900) -> None:
    args = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--no-deps"]
    args += list(targets)
    env = {"PIP_NO_INDEX": "1"} if offline else None
    run(args, env=env, timeout=timeout)


def run_python(
    python: Path,
    script: str,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Execute a script in another interpreter; nothing is written into the repo."""
    return subprocess.run(
        [str(python), "-c", textwrap.dedent(script)],
        cwd=str(cwd or Path.home()),
        env=clean_env(env),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def last_json_line(text: str) -> dict:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:"):])
    raise AssertionError(f"no RESULT line in output:\n{text[-4000:]}")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


__all__ = [
    "EVIDENCE_ARTIFACTS",
    "HOST_ENV_KEYS",
    "MILESTONES",
    "REPO_ROOT",
    "clean_env",
    "last_json_line",
    "make_venv",
    "pip_install",
    "read_jsonl",
    "run",
    "run_python",
    "sys",
    "venv_python",
    "write_jsonl",
]
