"""Inventory of the JA/EN corpus: the only module that touches ``JA vs. EN``.

The corpus is two archives, not two folders:

* ``Jujutsu Kaisen ENG/*.cbz`` - 31 ZIPs, volumes 00-30.  Entry names carry the
  metadata (``Jujutsu Kaisen - c001 (v01) - p000 [VIZ Media] [Digital] [1r0n].png``)
  and a two-page spread is ONE wide entry named ``... - p006-p007 ...``.
* ``Jujutsu Kaisen JA.rar`` - one RAR5, non-solid, every entry STORED, 10,650 pages
  in 125 folders.  ``rarfile`` reads stored entries in pure Python, so no external
  unrar/7z binary is needed and the archive is never extracted.

Nothing here writes into the corpus: it is read-only, licensed third-party material.
No page, crop or derived image may leave ``demo/output/`` (which is gitignored).

Paths are ``pathlib.Path`` objects throughout and image bytes are decoded from memory
with ``cv2.imdecode``, never ``cv2.imread(str(path))`` - the corpus root contains a
space and a period, and OpenCV's C++ ``fopen`` mangles non-ASCII paths.  This is the
same rule as ``glasstranslate/capture/static.py`` and ``renderer/.../stages/lama.py``.

Usage::

    .venv/Scripts/python.exe demo/corpus_index.py                 # full inventory
    .venv/Scripts/python.exe demo/corpus_index.py --no-decode     # names + sizes only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO_ROOT / "JA vs. EN"
OUT_ROOT = REPO_ROOT / "demo" / "output" / "corpus"
EN_DIR_NAME = "Jujutsu Kaisen ENG"
JA_ARCHIVE_NAME = "Jujutsu Kaisen JA.rar"

PAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}
SPREAD_ASPECT = 1.15  # width/height above this is a joined two-page spread
COLOUR_METRIC_THRESHOLD = 12.0  # mean per-pixel max(BGR)-min(BGR) above this is colour
THUMB_SIDE = 256  # working size for the colour metric

# Volumes 25-30 exist only in English.
EN_ONLY_VOLUMES = (25, 26, 27, 28, 29, 30)

# Canonical JA source folder per volume, as a suffix of the folder path.  Most volumes
# have more than one candidate (24 of 25 do) and the plain "vNN" folder is often the
# wrong one: v16 has ~2x the expected pages, v17/v18/v20 are landscape screen dumps
# rather than scans, and v21/v22/v24 are colour-tinted.  Established by inventory.
CANONICAL_JA: Dict[int, str] = {
    0: "v00", 1: "v01", 2: "第02巻", 3: "第03巻",
    4: "第04巻", 5: "第05巻", 6: "第06巻",
    7: "第07巻", 8: "v08", 9: "v09", 10: "v10", 11: "v11", 12: "v12",
    13: "v13", 14: "第14巻", 15: "第15巻", 16: "v16ss",
    17: "v17b", 18: "v18b", 19: "v19", 20: "第20巻",
    21: "第21巻", 22: "第22巻", 23: "Jujutsu_Kaisen_v23",
    24: "Jujutsu_Kaisen_v24b",
}

# Folders with no EN counterpart, excluded from pairing entirely.
EXCLUDED_JA_MARKERS = ("Official Fanbook", "v15-16s")

_EN_NAME_RE = re.compile(
    r"-\s*c(?P<chapter>[0-9]+(?:\.[0-9]+)?[a-z]?)\s*\(v(?P<volume>\d+)\)"
    r"\s*-\s*p(?P<page>\d+)(?:-p(?P<page2>\d+))?",
    re.IGNORECASE,
)
_EN_FLAT_RE = re.compile(r"^(?P<page>\d+)$")  # v30 (Rillant): flat NNNN.ext names
_VOLUME_IN_FILENAME_RE = re.compile(r"\bv(\d{2})\b", re.IGNORECASE)


# --------------------------------------------------------------------- page records
@dataclass(frozen=True)
class PageRef:
    """One image entry inside one archive, plus the metadata its name yields."""

    side: str  # "ja" | "en"
    archive: str  # archive file name, relative to the corpus root
    entry: str  # internal path inside the archive
    group: str  # JA: the containing folder.  EN: the .cbz stem.
    volume: Optional[int]
    chapter: Optional[str]
    page: Optional[int]
    page2: Optional[int]  # second page number of a named spread
    is_spread: bool
    width: int
    height: int
    fmt: str
    colour_metric: float
    is_colour: bool
    crc: int  # with size, the cache identity; free from the directory listing
    size: int

    @property
    def page_id(self) -> str:
        """Stable, ASCII-safe, filesystem-safe id (also the render/cache stem).

        JA folder names are Japanese, so the id must never be derived from the entry
        path: it becomes a directory name and a cv2 filename downstream.
        """
        vol = f"v{self.volume:02d}" if self.volume is not None else "vxx"
        page = f"p{self.page:03d}" if self.page is not None else "pxxx"
        # The full CRC, not a truncation: JA entries carry no page number, so every JA
        # page of a non-canonical group is "ja_vxx_pxxx_<crc>" and the CRC is the only
        # thing separating them.  Truncating to 24 bits bought nothing and could collide.
        return f"{self.side}_{vol}_{page}_{self.crc & 0xFFFFFFFF:08x}"


@dataclass(frozen=True)
class EntryInfo:
    name: str
    crc: int
    size: int


# --------------------------------------------------------------------- archives
class Archive:
    """Read-only random access to an archive of pages."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def name(self) -> str:
        return self.path.name

    def entries(self) -> List[EntryInfo]:  # pragma: no cover - interface
        raise NotImplementedError

    def read(self, entry: str) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        return None


class ZipArchive(Archive):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._zf = zipfile.ZipFile(self.path)

    def entries(self) -> List[EntryInfo]:
        return [
            EntryInfo(i.filename, i.CRC, i.file_size)
            for i in self._zf.infolist()
            if not i.is_dir() and Path(i.filename).suffix.lower() in PAGE_EXT
        ]

    def read(self, entry: str) -> bytes:
        return self._zf.read(entry)

    def close(self) -> None:
        self._zf.close()


class RarArchive(Archive):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        try:
            import rarfile
        except ImportError as exc:  # pragma: no cover - environment problem
            raise RuntimeError(
                "reading the JA corpus needs the 'rarfile' package: "
                ".venv/Scripts/python.exe -m pip install rarfile"
            ) from exc
        self._rf = rarfile.RarFile(self.path)

    def entries(self) -> List[EntryInfo]:
        return [
            EntryInfo(i.filename, int(i.CRC or 0), int(i.file_size or 0))
            for i in self._rf.infolist()
            if not i.is_dir() and Path(i.filename).suffix.lower() in PAGE_EXT
        ]

    def read(self, entry: str) -> bytes:
        return self._rf.read(entry)

    def close(self) -> None:
        self._rf.close()


def open_archive(path: Path) -> Archive:
    path = Path(path)
    return RarArchive(path) if path.suffix.lower() == ".rar" else ZipArchive(path)


# --------------------------------------------------------------------- decoding
def decode_bgr(data: bytes) -> Optional[np.ndarray]:
    """Decode page bytes from memory.  Never ``cv2.imread`` - see the module docstring."""
    buf = np.frombuffer(data, np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def decode_gray(data: bytes) -> Optional[np.ndarray]:
    buf = np.frombuffer(data, np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf.size else None


def imread_gray(path: Path) -> Optional[np.ndarray]:
    """Read a greyscale image from disk without handing OpenCV a ``str`` path.

    ``cv2.imread(str(path))`` goes through a C++ ``fopen`` that mangles non-ASCII, and
    returns ``None`` rather than raising - so on a checkout under, say, ``C:\\Users\\
    \u3086\u3046\u305f\\project`` every read would fail silently and the harness would
    degrade instead of erroring.  Reading the bytes in Python first removes the question.
    """
    path = Path(path)
    if not path.exists():
        return None
    return decode_gray(path.read_bytes())


def write_page_bytes(path: Path, data: bytes) -> Path:
    """Materialise a page next to the harness output.  Bytes go through Python."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def colourfulness(bgr: Optional[np.ndarray]) -> float:
    """Mean per-pixel ``max(BGR) - min(BGR)`` on a thumbnail.

    A JPEG can carry mode RGB while being visually greyscale, so the channel spread is
    measured rather than trusting the decoded mode.  Keep the float, not just the
    boolean: it is what surfaced the colour-tinted v21/v22/v24 JA scans.
    """
    if bgr is None or bgr.ndim != 3:
        return 0.0
    h, w = bgr.shape[:2]
    scale = min(1.0, THUMB_SIDE / max(1, max(h, w)))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    arr = bgr.astype(np.int16)
    return float((arr.max(axis=2) - arr.min(axis=2)).mean())


def probe_size(data: bytes) -> Tuple[int, int, str]:
    """True pixel size and format from the header, without a full decode."""
    with Image.open(BytesIO(data)) as im:
        return int(im.size[0]), int(im.size[1]), (im.format or "").lower()


# --------------------------------------------------------------------- name parsing
def parse_en_name(entry: str, archive_name: str = "") -> Dict[str, Optional[object]]:
    """Volume / chapter / page / spread from an EN entry name.

    v30 is a different scanlator with flat ``NNNN.ext`` names and no chapter marker,
    so the volume falls back to the ``vNN`` token in the .cbz file name.
    """
    stem = Path(entry).stem
    match = _EN_NAME_RE.search(stem)
    if match:
        page2 = match.group("page2")
        return {
            "volume": int(match.group("volume")),
            "chapter": match.group("chapter"),
            "page": int(match.group("page")),
            "page2": int(page2) if page2 else None,
            "named_spread": page2 is not None,
        }
    volume: Optional[int] = None
    vol_match = _VOLUME_IN_FILENAME_RE.search(archive_name)
    if vol_match:
        volume = int(vol_match.group(1))
    flat = _EN_FLAT_RE.match(stem)
    return {
        "volume": volume,
        "chapter": None,
        "page": int(flat.group("page")) if flat else None,
        "page2": None,
        "named_spread": False,
    }


def ja_group(entry: str) -> str:
    """The containing folder of a JA entry - one scan source."""
    return Path(entry).parent.as_posix()


def ja_volume(group: str) -> Optional[int]:
    """The volume a canonical JA group belongs to, or None when it is not canonical."""
    for volume, marker in CANONICAL_JA.items():
        if group.rstrip("/").endswith(marker):
            return volume
    return None


def is_excluded_ja(group: str) -> bool:
    """The fanbook and the weekly-magazine dump have no EN counterpart."""
    return any(marker in group for marker in EXCLUDED_JA_MARKERS)


# --------------------------------------------------------------------- spreads
def is_spread(width: int, height: int, named_spread: bool = False) -> bool:
    """A spread is declared by the entry name or by an unusually wide page."""
    return bool(named_spread) or (height > 0 and width / height > SPREAD_ASPECT)


def split_spread(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a joined spread into ``(right, left)`` - manga reading order.

    Verified against the EN release on six spreads: the right half always lands on the
    LOWER JA page index, because the book reads right to left.
    """
    mid = img.shape[1] // 2
    return img[:, mid:], img[:, :mid]


# --------------------------------------------------------------------- enumeration
def _record(side: str, archive: Archive, info: EntryInfo, decode: bool) -> PageRef:
    data = archive.read(info.name)
    width, height, fmt = probe_size(data)
    metric = colourfulness(decode_bgr(data)) if decode else 0.0
    if side == "en":
        meta = parse_en_name(info.name, archive.name)
        group = Path(archive.path).stem
    else:
        meta = {"volume": ja_volume(ja_group(info.name)), "chapter": None,
                "page": None, "page2": None, "named_spread": False}
        group = ja_group(info.name)
    return PageRef(
        side=side,
        archive=archive.name,
        entry=info.name,
        group=group,
        volume=meta["volume"],  # type: ignore[arg-type]
        chapter=meta["chapter"],  # type: ignore[arg-type]
        page=meta["page"],  # type: ignore[arg-type]
        page2=meta["page2"],  # type: ignore[arg-type]
        is_spread=is_spread(width, height, bool(meta["named_spread"])),
        width=width,
        height=height,
        fmt=fmt,
        colour_metric=round(metric, 3),
        is_colour=metric > COLOUR_METRIC_THRESHOLD,
        crc=info.crc,
        size=info.size,
    )


def enumerate_archive(archive: Archive, side: str, *, decode: bool = True,
                      workers: int = 12, limit: Optional[int] = None) -> List[PageRef]:
    infos = sorted(archive.entries(), key=lambda i: i.name)
    if limit:
        infos = infos[:limit]
    if workers <= 1:
        return [_record(side, archive, i, decode) for i in infos]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda i: _record(side, archive, i, decode), infos))


def enumerate_corpus(corpus_root: Path = DEFAULT_CORPUS, *, decode: bool = True,
                     workers: int = 12, limit: Optional[int] = None) -> List[PageRef]:
    corpus_root = Path(corpus_root)
    pages: List[PageRef] = []
    ja_path = corpus_root / JA_ARCHIVE_NAME
    if ja_path.exists():
        ja = open_archive(ja_path)
        try:
            pages += enumerate_archive(ja, "ja", decode=decode, workers=workers, limit=limit)
        finally:
            ja.close()
    en_dir = corpus_root / EN_DIR_NAME
    for cbz in sorted(en_dir.glob("*.cbz")) if en_dir.exists() else []:
        en = open_archive(cbz)
        try:
            pages += enumerate_archive(en, "en", decode=decode, workers=workers, limit=limit)
        finally:
            en.close()
    return pages


def summarize(pages: Sequence[PageRef]) -> Dict[str, object]:
    ja = [p for p in pages if p.side == "ja"]
    en = [p for p in pages if p.side == "en"]
    en_volumes = sorted({p.volume for p in en if p.volume is not None})
    ja_volumes = sorted({p.volume for p in ja if p.volume is not None})
    groups: Dict[str, int] = {}
    for p in ja:
        groups[p.group] = groups.get(p.group, 0) + 1
    return {
        "ja_pages": len(ja),
        "en_pages": len(en),
        "ja_groups": len(groups),
        "en_volumes": en_volumes,
        "ja_canonical_volumes": ja_volumes,
        "en_only_volumes": [v for v in en_volumes if v not in ja_volumes],
        "ja_spreads": sum(1 for p in ja if p.is_spread),
        "en_spreads": sum(1 for p in en if p.is_spread),
        "ja_colour": sum(1 for p in ja if p.is_colour),
        "en_colour": sum(1 for p in en if p.is_colour),
        "excluded_ja_groups": sorted(g for g in groups if is_excluded_ja(g)),
    }


def write_index(pages: Sequence[PageRef], corpus_root: Path,
                out_path: Optional[Path] = None) -> Path:
    out_path = Path(out_path) if out_path else OUT_ROOT / "index.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus_root": str(corpus_root),
        "summary": summarize(pages),
        "pages": [asdict(p) for p in pages],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_path


def load_index(path: Optional[Path] = None) -> List[PageRef]:
    path = Path(path) if path else OUT_ROOT / "index.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [PageRef(**rec) for rec in data["pages"]]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-decode", action="store_true",
                        help="skip the colour metric (names, sizes and formats only)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None, help="first N entries per archive")
    args = parser.parse_args(argv)

    if not Path(args.corpus).exists():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 2
    pages = enumerate_corpus(args.corpus, decode=not args.no_decode,
                             workers=args.workers, limit=args.limit)
    out = write_index(pages, Path(args.corpus), args.out)
    print(json.dumps(summarize(pages), ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
