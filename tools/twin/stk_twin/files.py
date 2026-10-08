"""File modes for what LocalRouter's image reads (it runs as uid 10001)."""

from __future__ import annotations

from pathlib import Path


def readable(out: str | Path) -> None:
    """Directories 0755 and files 0644 under `out`, whatever umask wrote them (root's in the NGC container left 0600
    pack files on the Spark)."""
    out = Path(out)
    for p in [out, *out.rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
