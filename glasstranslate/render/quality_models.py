"""Model store for the quality renderer's sidecar (LaMa, SDXL, the ControlNet).

The files live in ``<models_dir>/quality`` and are downloaded by the **app**
(Engines page progress row) - the sidecar never downloads anything itself, per
``renderer/PROTOCOL.md``.  The table of names, sizes and SHA-256 digests is a
data file, :data:`TABLE_PATH`, shared with ``glassrenderer/models.py``; a
checkout that does not carry it yet loads an empty table and every function
here keeps working (nothing is ready, nothing can be downloaded).

The streaming, verification and atomic-rename logic is the one that already
ships for the manga-ocr bundle (:mod:`glasstranslate.ocr.models`): this module
reuses it unchanged rather than growing a second copy, so both stores share
the ``.part`` handling, the size / digest checks and the name validation.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from ..ocr.models import (  # the verified downloader, reused as is
    ModelDownloadError,
    ModelFile,
    ProgressCallback,
    download_one as _download_one,
    validate_name as _validate_name,
)

__all__ = [
    "ModelDownloadError",
    "QUALITY_SUBDIR",
    "QualityModel",
    "TABLE_PATH",
    "ensure_models",
    "file_path",
    "load_table",
    "missing_files",
    "models_ready",
    "quality_dir",
    "size_label",
    "total_bytes",
]

log = logging.getLogger(__name__)

QUALITY_SUBDIR = "quality"
TABLE_PATH = Path(__file__).resolve().parent / "quality_models.json"
_REQUIRED_KEYS = ("name", "remote_path", "size", "sha256", "repo", "revision", "subdir")
# Serialises downloads so two callers never stream into the same ``.part`` file.
_DOWNLOAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class QualityModel(ModelFile):
    """One file of the quality store: a :class:`~glasstranslate.ocr.models.ModelFile`
    plus the ``'/'``-separated ``subdir`` it lives in under ``<models_dir>/quality``
    (the same layout ``glassrenderer/models.py`` loads from)."""

    subdir: str = ""


def quality_dir(models_dir: Union[str, Path]) -> Path:
    """Directory holding the quality-renderer files under ``models_dir``.

    A path that already ends in ``quality`` is taken as the store itself, so the
    app and the sidecar accept each other's value.
    """
    path = Path(models_dir)
    return path if path.name == QUALITY_SUBDIR else path / QUALITY_SUBDIR


def file_path(models_dir: Union[str, Path], spec: QualityModel) -> Path:
    """Where ``spec`` lives: ``<models_dir>/quality/<subdir>/<name>``."""
    return quality_dir(models_dir).joinpath(*spec.subdir.split("/")) / spec.name


def _validate_subdir(subdir: str) -> None:
    """Refuse subdirectories that are absolute or climb out of the store."""
    if not subdir or subdir.startswith(("/", "\\")) or "\\" in subdir or os.path.isabs(subdir):
        raise ValueError(f"invalid model subdir {subdir!r}")
    for part in subdir.split("/"):
        if not part or part in (".", "..") or ":" in part:
            raise ValueError(f"invalid model subdir {subdir!r}")


def _entry(raw: object) -> Optional[QualityModel]:
    """One table row, or None when it is incomplete or unsafe."""
    if not isinstance(raw, dict):
        return None
    if any(key not in raw for key in _REQUIRED_KEYS):
        return None
    try:
        name = str(raw["name"])
        subdir = str(raw["subdir"])
        # Never let a table entry escape the models directory.
        _validate_name(name)
        _validate_subdir(subdir)
        size = int(raw["size"])
    except (ValueError, TypeError):
        return None
    if size <= 0 or not str(raw["sha256"]).strip():
        return None
    return QualityModel(
        name=name,
        remote_path=str(raw["remote_path"]),
        size=size,
        sha256=str(raw["sha256"]).strip().lower(),
        repo=str(raw["repo"]),
        revision=str(raw["revision"]),
        subdir=subdir,
    )


def load_table(path: Optional[Union[str, Path]] = None) -> Tuple[QualityModel, ...]:
    """The model table, or ``()`` when the data file is absent or malformed.

    The file is ``{"files": [{name, remote_path, size, sha256, repo, revision,
    subdir}, ...]}``; rows missing a field, carrying a non-positive size or a
    name / subdir that could escape the target directory are dropped rather
    than trusted.
    """
    target = Path(path) if path is not None else TABLE_PATH
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.debug("no quality model table at %s (%s)", target, exc)
        return ()
    rows = data.get("files") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return ()
    entries = [_entry(row) for row in rows]
    if any(entry is None for entry in entries):
        log.warning("ignoring the quality model table in %s: %d unusable entries",
                    target, sum(1 for e in entries if e is None))
        return ()
    return tuple(e for e in entries if e is not None)


def _files(files: Optional[Sequence[QualityModel]]) -> Tuple[QualityModel, ...]:
    return tuple(files) if files is not None else load_table()


def total_bytes(files: Optional[Sequence[QualityModel]] = None) -> int:
    """Download size of the whole table in bytes (0 for an empty table)."""
    return sum(spec.size for spec in _files(files))


def size_label(files: Optional[Sequence[QualityModel]] = None) -> str:
    """``"10.0 GB"`` - the download size as the Engines page shows it."""
    return f"{total_bytes(files) / 1e9:.1f} GB"


def missing_files(
    models_dir: Union[str, Path], files: Optional[Sequence[QualityModel]] = None
) -> List[QualityModel]:
    """The table entries not present at their pinned size under ``models_dir``.

    The size test mirrors the sidecar's ``glassrenderer.models._is_present``: the two
    sides must agree on what "ready" means, or the Engines page says ready while every
    job fails with "missing".  A truncated download is therefore fetched again."""
    out: List[QualityModel] = []
    for spec in _files(files):
        path = file_path(models_dir, spec)
        if not (path.is_file() and path.stat().st_size == spec.size):
            out.append(spec)
    return out


def models_ready(
    models_dir: Union[str, Path], files: Optional[Sequence[QualityModel]] = None
) -> bool:
    """True when every table entry is present.  An empty table is never "ready":
    without the data file nothing is known about the models, so the Engines page
    must not claim they are installed."""
    known = _files(files)
    return bool(known) and not missing_files(models_dir, known)


def ensure_models(
    models_dir: Union[str, Path],
    *,
    files: Optional[Sequence[QualityModel]] = None,
    progress: Optional[ProgressCallback] = None,
    session: Optional[object] = None,
    cancel: Optional[threading.Event] = None,
) -> Path:
    """Download every missing entry and return ``<models_dir>/quality``.

    ``progress(filename, percent)`` is called as each file streams in;
    ``session`` is any ``requests.Session``-like object; setting ``cancel``
    aborts between chunks.  Raises :class:`ModelDownloadError` when the table
    is empty (nothing is known to download) or a file fails to verify.
    """
    known = _files(files)
    if not known:
        raise ModelDownloadError(
            "no quality model table is bundled with this build; the quality renderer "
            "cannot download its models"
        )
    for spec in known:
        _validate_name(spec.name)
        _validate_subdir(spec.subdir)
    target = quality_dir(models_dir)
    if not missing_files(models_dir, known):
        return target
    with _DOWNLOAD_LOCK:
        todo = missing_files(models_dir, known)  # another caller may have finished
        if not todo:
            return target
        owned: Optional[object] = None
        if session is None:
            import requests

            owned = session = requests.Session()
        try:
            for spec in todo:
                folder = file_path(models_dir, spec).parent
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise ModelDownloadError(f"cannot create {folder} ({exc})") from exc
                _download_one(session, folder, spec, progress, cancel)
        finally:
            if owned is not None:
                owned.close()  # type: ignore[attr-defined]
    log.info("quality renderer models ready in %s", target)
    return target
