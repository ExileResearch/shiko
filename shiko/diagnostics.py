"""Diagnostics that survive the process.

A hook that fails inside Hermes must not raise into the host, but silence is
worse: every failure is written to a JSONL log under the deployment root and
also kept in a small in-memory ring for the current process. Callers get the
same records through the result they already received.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .results import Diagnostic, Severity

MAX_RECENT = 128


class Diagnostics:
    """Append-only diagnostic log for one deployment."""

    def __init__(self, path: str | Path | None, *, owner_id: str = "") -> None:
        self.path = Path(path) if path is not None else None
        self.owner_id = owner_id
        self._recent: deque[Diagnostic] = deque(maxlen=MAX_RECENT)
        self._lock = threading.Lock()
        self._write_failure: str | None = None

    def emit(self, diagnostic: Diagnostic) -> Diagnostic:
        with self._lock:
            self._recent.append(diagnostic)
            if self.path is None:
                return diagnostic
            record = {
                "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "owner_id": self.owner_id,
                **diagnostic.to_dict(),
            }
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            except OSError as exc:  # diagnostics must never break the caller
                self._write_failure = str(exc)
        return diagnostic

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        for diagnostic in diagnostics:
            self.emit(diagnostic)

    def recent(self, *, limit: int | None = None) -> tuple[Diagnostic, ...]:
        with self._lock:
            items = tuple(self._recent)
        if limit is not None:
            items = items[-limit:]
        return items

    @property
    def write_failure(self) -> str | None:
        with self._lock:
            return self._write_failure
