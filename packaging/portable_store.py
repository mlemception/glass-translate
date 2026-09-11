"""Verify the local model store and describe exactly what the models zip ships.

Pure functions over a ``models`` directory, used by ``build_portable.py`` (steps
3 and 5 of ``docs/plans/2026-09-11-portable-bundle.md``).  Nothing here ever
downloads: a file that is absent or does not match its pinned size and SHA-256
fails the build instead.

The three stores have three owners, and this module reuses each one's own table
rather than restating it:

* quality renderer - :mod:`glasstranslate.render.quality_models` (24 files,
  sizes + digests from ``quality_models.json``),
* manga-ocr - :data:`glasstranslate.ocr.models.MANGA_OCR_FILES` (7 files),
* Argos / Sugoi packs - :mod:`glasstranslate.translate.argos`'s own directory
  and archive readers, so the build discovers exactly what the app will.

``models_dir`` is always the store root (the folder holding ``quality/``,
``manga-ocr/`` and the pack directories); manifest paths are POSIX and relative
to it.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow ``python packaging/portable_store.py`` from anywhere
    sys.path.insert(0, str(ROOT))

from glasstranslate.ocr import models as ocr_models  # noqa: E402
from glasstranslate.render import quality_models  # noqa: E402
from glasstranslate.translate import argos  # noqa: E402

__all__ = [
    "ArgosPack",
    "StoreError",
    "StoreFile",
    "copy_store",
    "discover_argos",
    "offered_pairs",
    "store_manifest",
    "uncovered_pairs",
    "verify_manga_ocr",
    "verify_quality",
]

Pair = Tuple[str, str]
PathLike = Union[str, Path]

MANGA_OCR_LICENCE = "Apache-2.0"  # both source repos, see glasstranslate/ocr/models.py

# Translation packs.  Looked up 2026-09-11; recorded verbatim, never inferred:
#
# * argospm-index's README licences *the index repository* ("Dual licensed under either
#   the MIT License or CC0") and says nothing about the packages it lists, so the packs'
#   own terms are unknown and are recorded as such rather than guessed.
# * The Sugoi conversion's model card declares ``license: other`` /
#   ``license_name: ntt-license``; that LICENSE is NTT's research licence - redistribution
#   is permitted for research only and "This data is not available for commercial use,
#   including the sale of translators trained using this data."
ARGOS_INDEX_URL = "https://github.com/argosopentech/argospm-index"
SUGOI_MODEL_URL = "https://huggingface.co/entai2965/sugoi-v4-ja-en-ctranslate2"
ARGOS_LICENCE = f"unknown - see {ARGOS_INDEX_URL}"
SUGOI_LICENCE = (
    "ntt-license (license: other) - NTT research licence: research use only, no commercial "
    "use, not for redistribution"
)
# Shipped by default because the bundle is built for personal use on the machine that
# built it.  ``--no-research-models`` leaves it out for a bundle that will be shared; the
# app can then install it locally through Engines -> Download Sugoi.
SUGOI_EXCLUSION_REASON = (
    "NTT research licence: redistribution for research only, no commercial use - run the "
    "app on a networked machine and use Engines -> Download Sugoi to install it locally"
)
# The SentencePiece path both Sugoi layouts carry (_read_sugoi_dir hard-codes it, and
# download_sugoi writes it into its metadata.json), so a renamed directory is still
# recognised as the research-licensed pack.
_SUGOI_SPM = "spm/spm.ja.nopretok.model"

_HASH_CHUNK = 1 << 20  # 1 MiB


class StoreError(RuntimeError):
    """A store file is missing or does not match its pinned size / digest."""


@dataclass(frozen=True)
class StoreFile:
    """One file the models zip ships, with the digest the build measured."""

    path: str  # POSIX, relative to the models dir
    size: int
    sha256: str
    kind: str  # quality | manga-ocr | argos | sugoi
    pair: Optional[str] = None  # "ja->en" for a translation pack, else None
    licence: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
            "kind": self.kind,
            "pair": self.pair,
            "licence": self.licence,
        }


@dataclass(frozen=True)
class ArgosPack:
    """A translation pack the store holds, as the app's ``rescan()`` sees it.

    ``redistributable`` is False for a pack whose upstream licence does not allow passing
    it on: the Sugoi conversion is NTT-licensed for research use only.  Such a pack still
    ships by default - the bundle is built for personal use on the machine that built it -
    and ``--no-research-models`` leaves it out of a bundle that will be shared.
    """

    pair: Pair
    priority: int
    kind: str  # argos | sugoi
    directory: Optional[Path] = None
    archive: Optional[Path] = None
    redistributable: bool = True
    licence_note: str = ""


def pair_label(pair: Pair) -> str:
    """``("ja", "en")`` -> ``"ja->en"`` (the manifest's string form)."""
    return f"{pair[0]}->{pair[1]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked(path: Path, rel: str, size: int, sha256: Optional[str]) -> Tuple[int, str]:
    """Size and digest of ``path``, raising :class:`StoreError` on any mismatch."""
    if not path.is_file():
        raise StoreError(f"{rel}: missing from the store ({path})")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise StoreError(f"{rel}: size mismatch (expected {size} bytes, got {actual_size})")
    digest = _sha256(path)
    if sha256 is not None and digest != sha256.lower():
        raise StoreError(f"{rel}: sha256 mismatch (expected {sha256}, got {digest})")
    return actual_size, digest


def _quality_licences() -> Dict[Tuple[str, str], str]:
    """``(subdir, name) -> licence`` from the shipped table's own ``licence`` field."""
    try:
        data = json.loads(quality_models.TABLE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    rows = data.get("files", []) if isinstance(data, dict) else []
    return {
        (str(row.get("subdir", "")), str(row.get("name", ""))): str(row["licence"])
        for row in rows
        if isinstance(row, dict) and row.get("licence")
    }


def verify_quality(
    models_dir: PathLike, files: Optional[Sequence[Any]] = None
) -> List[StoreFile]:
    """Re-hash every quality-renderer file under ``models_dir``.

    ``files`` defaults to the shipped table (``quality_models.load_table()``);
    tests pass a small synthetic one.  Raises :class:`StoreError` naming the
    first file that is absent, truncated or has the wrong digest.
    """
    table = list(files) if files is not None else list(quality_models.load_table())
    if not table:
        raise StoreError(
            f"the quality model table {quality_models.TABLE_PATH} is empty or unreadable"
        )
    licences = _quality_licences()
    rows: List[StoreFile] = []
    for spec in table:
        rel = f"{quality_models.QUALITY_SUBDIR}/{spec.subdir}/{spec.name}"
        size, digest = _checked(quality_models.file_path(models_dir, spec), rel, spec.size, spec.sha256)
        rows.append(StoreFile(rel, size, digest, "quality", None, licences.get((spec.subdir, spec.name))))
    return sorted(rows, key=lambda row: row.path)


def verify_manga_ocr(
    models_dir: PathLike, files: Sequence[Any] = ocr_models.MANGA_OCR_FILES
) -> List[StoreFile]:
    """Re-hash every manga-ocr file under ``models_dir`` (same failure mode)."""
    target = ocr_models.manga_ocr_dir(models_dir)
    rows: List[StoreFile] = []
    for spec in files:
        rel = f"{ocr_models.MANGA_OCR_SUBDIR}/{spec.name}"
        size, digest = _checked(target / spec.name, rel, spec.size, spec.sha256)
        rows.append(StoreFile(rel, size, digest, "manga-ocr", None, MANGA_OCR_LICENCE))
    return sorted(rows, key=lambda row: row.path)


def _pack_kind(directory: Optional[Path], package: Any = None) -> str:
    """``"sugoi"`` for the research-licensed conversion, ``"argos"`` otherwise.

    Both of the layouts ``glasstranslate.translate.argos`` recognises are covered: the
    conventional ``sugoi-v4-ja-en`` directory name, and any directory whose package
    (from ``_read_sugoi_dir`` or a ``download_sugoi`` metadata.json) names Sugoi's own
    SentencePiece model - so renaming the folder does not smuggle it into the zip.
    """
    if directory is not None and directory.name == argos.SUGOI_DIR_NAME:
        return "sugoi"
    return "sugoi" if getattr(package, "source_spm", "") == _SUGOI_SPM else "argos"


def _make_pack(pair: Pair, priority: int, kind: str, directory: Optional[Path],
               archive: Optional[Path]) -> ArgosPack:
    licence, _source = pack_licence(kind)
    return ArgosPack(pair, priority, kind, directory, archive,
                     redistributable=(kind != "sugoi"), licence_note=licence)


def _pack_name(pack: "ArgosPack") -> str:
    source = pack.directory if pack.directory is not None else pack.archive
    return source.name if source is not None else ""


def discover_argos(models_dir: PathLike, *, include_research_models: bool = True) -> List[ArgosPack]:
    """The translation packs under ``models_dir``, read the way the app reads them.

    **Every** recognisable directory is shipped, including one whose pair another
    pack already wins at runtime: ``ja_en/`` stays even though ``sugoi-v4-ja-en/``
    (priority 10) serves ja->en, so a later Argos index refresh still finds a
    package there.  An **archive** is shipped only when no *shipping* directory
    serves its pair, which drops ``translate-ja_en-1_1.argosmodel`` once ``ja_en/``
    is extracted - but keeps it when the only directory for that pair is the
    research-licensed one being left out (``include_research_models=False``).  A
    directory holding no recognisable pack (an empty one included) is skipped, and
    so is the stray empty ``sugoi-v4-ja-en/sugoi-v4-ja-en/`` nested inside a pack,
    which is not a direct child at all.
    """
    root = Path(models_dir)
    if not root.is_dir():
        return []
    directories: List[ArgosPack] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        pkg = argos._read_metadata_dir(child)
        if pkg is not None:
            directories.append(_make_pack(pkg.pair, pkg.priority, _pack_kind(child, pkg), child, None))
    # Only a directory that will actually ship suppresses its archive: dropping the
    # research-licensed Sugoi pack must not silently drop ja->en with it.
    served = {pack.pair for pack in directories if pack.redistributable or include_research_models}
    packs = list(directories)
    for archive in sorted(root.glob("*.argosmodel")):
        pkg = argos._read_metadata_zip(archive)
        if pkg is not None and pkg.pair not in served:
            packs.append(_make_pack(pkg.pair, pkg.priority, "argos", None, archive))
    return sorted(packs, key=lambda pack: (pack.pair, _pack_name(pack)))


def offered_pairs() -> List[Pair]:
    """Every ``X->en`` / ``en->X`` pair the Translate page offers (``X != en``)."""
    from glasstranslate.ui.bridge_fields import LANGUAGES

    pairs = set()
    for code, _name in LANGUAGES:
        if code == "en":
            continue
        pairs.add((code, "en"))
        pairs.add(("en", code))
    return sorted(pairs)


def uncovered_pairs(packs: Sequence[ArgosPack]) -> List[Pair]:
    """The offered pairs ``packs`` cannot serve, directly or through English."""
    direct = {pack.pair for pack in packs}
    sources = {src for src, dst in direct if dst == "en"}
    targets = {dst for src, dst in direct if src == "en"}
    covered = direct | {(src, dst) for src in sources for dst in targets if src != dst}
    return [pair for pair in offered_pairs() if pair not in covered]


def pack_licence(kind: str) -> Tuple[str, str]:
    """``(licence, source URL)`` for a pack of ``kind`` - see the note by the constants."""
    if kind == "sugoi":
        return SUGOI_LICENCE, SUGOI_MODEL_URL
    return ARGOS_LICENCE, ARGOS_INDEX_URL


def _pack_readme(pack: ArgosPack, root: Path) -> Optional[str]:
    """The pack's own ``README.md`` (Argos packs ship one; Sugoi does not)."""
    if pack.directory is None:
        return None
    readme = pack.directory / "README.md"
    return readme.relative_to(root).as_posix() if readme.is_file() else None


def _pack_files(models_dir: PathLike, pack: ArgosPack) -> List[StoreFile]:
    """Every file of one pack, hashed (empty directories contribute nothing)."""
    root = Path(models_dir)
    label = pair_label(pack.pair)
    licence, _source = pack_licence(pack.kind)
    sources = (
        sorted(p for p in pack.directory.rglob("*") if p.is_file())
        if pack.directory is not None
        else [pack.archive]
    )
    rows = []
    for path in sources:
        assert path is not None  # one of directory / archive is always set
        rel = path.relative_to(root).as_posix()
        rows.append(StoreFile(rel, path.stat().st_size, _sha256(path), pack.kind, label, licence))
    return rows


def _pack_entry(pack: ArgosPack, root: Path) -> Dict[str, Any]:
    """One ``argos_packs`` row of the manifest (a pack the bundle ships)."""
    licence, source = pack_licence(pack.kind)
    return {
        "pair": pair_label(pack.pair),
        "priority": pack.priority,
        "kind": pack.kind,
        "directory": pack.directory.relative_to(root).as_posix() if pack.directory else None,
        "archive": pack.archive.relative_to(root).as_posix() if pack.archive else None,
        "licence": licence,
        "source": source,
        "readme": _pack_readme(pack, root),
        "redistributable": pack.redistributable,
    }


def _excluded_entry(pack: ArgosPack, root: Path) -> Dict[str, Any]:
    """One ``excluded_packs`` row: a pack the store holds but the bundle must not ship."""
    _licence, source = pack_licence(pack.kind)
    return {
        "directory": pack.directory.relative_to(root).as_posix() if pack.directory else None,
        "pair": pair_label(pack.pair),
        "priority": pack.priority,
        "reason": SUGOI_EXCLUSION_REASON,
        "source": source,
    }


def store_manifest(
    models_dir: PathLike,
    generated: str,
    *,
    quality_files: Optional[Sequence[Any]] = None,
    manga_ocr_files: Sequence[Any] = ocr_models.MANGA_OCR_FILES,
    include_research_models: bool = True,
) -> Dict[str, Any]:
    """The ``models/MANIFEST.json`` body: every shipped file, hashed, plus coverage.

    ``generated`` is the build's ISO date, passed in so the manifest is the caller's to
    date-stamp (and so tests stay deterministic).  A pack that is not
    :attr:`ArgosPack.redistributable` ships by default (the bundle is for personal use on
    the machine that built it) and is recorded with its licence;
    ``include_research_models=False`` leaves it out of ``files`` and ``total_bytes`` and
    lists it under ``excluded_packs`` instead, which is what a shareable bundle needs.
    """
    packs = discover_argos(models_dir, include_research_models=include_research_models)
    shipped = [pack for pack in packs if pack.redistributable or include_research_models]
    excluded = [pack for pack in packs if pack not in shipped]
    rows = verify_quality(models_dir, quality_files) + verify_manga_ocr(models_dir, manga_ocr_files)
    for pack in shipped:
        rows.extend(_pack_files(models_dir, pack))
    rows.sort(key=lambda row: row.path)
    root = Path(models_dir)
    return {
        "version": 1,
        "generated": generated,
        "research_models_included": include_research_models,
        "files": [row.as_dict() for row in rows],
        "argos_packs": [_pack_entry(pack, root) for pack in shipped],
        "excluded_packs": [_excluded_entry(pack, root) for pack in excluded],
        "uncovered_pairs": [pair_label(pair) for pair in uncovered_pairs(shipped)],
        "total_bytes": sum(row.size for row in rows),
    }


def _safe_relative(root: Path, rel: str) -> Path:
    """``root/rel`` for a manifest path, refusing anything that leaves ``root``.

    The manifest is data: a hand-edited or truncated one must not be able to write
    through an absolute path, a drive letter or ``..``.
    """
    candidate = Path(rel)
    if not rel or candidate.is_absolute() or candidate.drive or rel[0] in "/\\":
        raise StoreError(f"unsafe manifest path {rel!r}: must be relative to the models dir")
    if any(part in ("..", "") for part in candidate.parts):
        raise StoreError(f"unsafe manifest path {rel!r}: must not contain '..'")
    target = (root / candidate).resolve()
    if not target.is_relative_to(root.resolve()):
        raise StoreError(f"unsafe manifest path {rel!r}: escapes {root}")
    return target


def copy_store(models_dir: PathLike, dest: PathLike, manifest: Dict[str, Any]) -> int:
    """Copy exactly the manifest's files into ``dest`` and return how many.

    Nothing else travels: a leftover ``.part`` download, a ``__pycache__`` or any
    other stray file next to a model is simply not in the manifest.  Every path is
    checked to stay under ``dest`` before anything is written.
    """
    source_root, target_root = Path(models_dir), Path(dest)
    for row in manifest["files"]:
        target = _safe_relative(target_root, str(row["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / str(row["path"]), target)
    return len(manifest["files"])
