"""Deployment ownership.

A data root belongs to exactly one owner. The manifest is written when the root
is created and verified every time it is opened, so two agents cannot silently
share a store, and a store cannot be adopted by a different identity later.

The owner is a *deployment* fact, not an envelope fact. Several actors (users,
subagents, tools) may write into one owner's store; their identity travels on
each envelope.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._version import __version__
from .errors import OwnershipError

MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = "shiko-deployment"
MANIFEST_VERSION = 1
_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SLUG_STRIP = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True)
class OwnerManifest:
    owner_id: str
    display_name: str = ""
    created_at: str = ""
    engine_version: str = __version__
    manifest_version: int = MANIFEST_VERSION
    format: str = MANIFEST_FORMAT

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "manifest_version": self.manifest_version,
            "owner_id": self.owner_id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "engine_version": self.engine_version,
        }


def slugify_owner(value: str, *, fallback: str = "hermes") -> str:
    """Turn a host-provided name into a valid owner id.

    Profiles, VM names and display names arrive with capitals and spaces; the
    manifest keeps one canonical spelling so reopening never depends on case.
    """
    text = (value or "").strip().lower()
    text = _SLUG_STRIP.sub("-", text).strip("-.")
    if not text or not text[0].isalnum():
        text = f"{fallback}-{text}".strip("-") if text else fallback
    return text[:64].strip("-.") or fallback


def validate_owner_id(owner_id: str) -> str:
    text = (owner_id or "").strip()
    if not _SLUG.match(text):
        raise OwnershipError(
            "owner_id must be a lowercase slug of 1-64 characters "
            "(letters, digits, dot, underscore or hyphen, starting alphanumeric)",
            detail={"owner_id": owner_id},
        )
    return text


def manifest_path(root: str | Path) -> Path:
    return Path(root) / MANIFEST_NAME


def read_manifest(root: str | Path) -> OwnerManifest | None:
    """Read a manifest, or ``None`` when the root has never been provisioned."""
    path = manifest_path(root)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OwnershipError(f"cannot read owner manifest: {exc}", detail={"path": str(path)}) from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise OwnershipError(
            "owner manifest is not valid JSON; refusing to guess the owner",
            detail={"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(data, dict) or data.get("format") != MANIFEST_FORMAT:
        raise OwnershipError("owner manifest has an unknown format", detail={"path": str(path)})
    version = data.get("manifest_version")
    if not isinstance(version, int) or version > MANIFEST_VERSION:
        raise OwnershipError(
            "owner manifest was written by a newer Shiko build",
            detail={"path": str(path), "manifest_version": version, "supported": MANIFEST_VERSION},
        )
    owner_id = data.get("owner_id")
    try:
        owner_id = validate_owner_id(str(owner_id or ""))
    except OwnershipError as exc:
        raise OwnershipError("owner manifest has an invalid owner_id", detail={"path": str(path)}) from exc
    return OwnerManifest(
        owner_id=owner_id,
        display_name=str(data.get("display_name", "")),
        created_at=str(data.get("created_at", "")),
        engine_version=str(data.get("engine_version", "")),
        manifest_version=version,
        format=MANIFEST_FORMAT,
    )


def write_manifest(root: str | Path, manifest: OwnerManifest) -> Path:
    """Atomically write a manifest with owner-only permissions."""
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=".manifest-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def ensure_manifest(
    root: str | Path, owner_id: str, *, display_name: str = "",
) -> tuple[OwnerManifest, bool]:
    """Create or verify the manifest. Returns ``(manifest, created)``.

    A root that already holds a store but no manifest is refused rather than
    adopted: the store's owner is unknowable, and guessing would merge two
    agents' memories.
    """
    owner_id = validate_owner_id(owner_id)
    root = Path(root)
    existing = read_manifest(root)
    if existing is not None:
        if existing.owner_id != owner_id:
            raise OwnershipError(
                f"data root belongs to owner '{existing.owner_id}', not '{owner_id}'",
                detail={
                    "root": str(root),
                    "expected_owner": existing.owner_id,
                    "requested_owner": owner_id,
                },
            )
        return existing, False

    if (root / "shiko.db").exists():
        raise OwnershipError(
            "data root contains a store but no owner manifest; refusing to adopt an unknown owner",
            detail={"root": str(root)},
        )

    manifest = OwnerManifest(
        owner_id=owner_id,
        display_name=(display_name or "").strip(),
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        engine_version=__version__,
    )
    write_manifest(root, manifest)
    return manifest, True
