"""Structural page descriptors, version 1.

The JA and EN releases carry the SAME artwork and differ only in the lettering, so a
descriptor that keeps panel gutters and tone masses but discards small ink will score a
true page pair far above any other page.  Two channels are kept:

* ``tone`` - the greyscale page pooled to a 64x64 grid.
* ``edge`` - the Sobel gradient magnitude pooled the same way.

Both are z-normed, so a dot product is a normalised cross-correlation in [-1, 1], and
``similarity`` is their mean.  Measured on 36 ORB-verified true pairs against the best
wrong match in the same row: tone true 0.858+-0.111 vs false 0.257+-0.075 (min margin
+0.126), edge true 0.714+-0.130 vs false 0.216+-0.054 (min margin +0.104).

Two things that sound sensible and are NOT done, because they were measured:

* Row/column ink projection profiles are useless here - a wrong page scores 0.715.
* Down-weighting text-heavy tiles does not help (margin 0.050 vs 0.104 plain): lettering
  is a minority of a manga page's edge energy and the gutters already dominate.

Pure numpy/cv2 on arrays, so every function is testable on synthetic pages with no
corpus, no archive and no network.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Bumping this invalidates every cached descriptor: it is part of the cache key.
DESCRIPTOR_VERSION = 1

GRID = 64  # descriptor side; lettering is sub-pixel at this size, gutters are not
EDGE_SIDE = 512  # working long side for the Sobel, before pooling
CONTENT_PCT = 2.0  # per-cent of the peak row/column ink that still counts as content
CHANNELS: Tuple[str, ...] = ("tone", "edge")
QUANT = 127.0  # int8 quantisation of the cached descriptors

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / "demo" / "output" / "corpus" / "desc" / f"v{DESCRIPTOR_VERSION}"


@dataclass(frozen=True)
class Descriptor:
    """One page's structural signature.  ``tone``/``edge`` are unit-norm, zero-mean."""

    tone: np.ndarray
    edge: np.ndarray
    width: int
    height: int

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


# --------------------------------------------------------------------- geometry
def content_box(gray: np.ndarray) -> Tuple[int, int, int, int]:
    """The ink-bearing rectangle, so a scan border does not shift the descriptor.

    Returns ``(x, y, w, h)``; the full frame when the page is blank.
    """
    if gray.size == 0:
        return 0, 0, 0, 0
    paper = float(np.percentile(gray, 90))
    ink = gray < max(30.0, paper - 40.0)
    rows = ink.sum(axis=1).astype(np.float64)
    cols = ink.sum(axis=0).astype(np.float64)
    if rows.max() <= 0 or cols.max() <= 0:
        return 0, 0, gray.shape[1], gray.shape[0]
    ys = np.flatnonzero(rows > rows.max() * CONTENT_PCT / 100.0)
    xs = np.flatnonzero(cols > cols.max() * CONTENT_PCT / 100.0)
    if ys.size == 0 or xs.size == 0:
        return 0, 0, gray.shape[1], gray.shape[0]
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    return x0, y0, x1 - x0, y1 - y0


def _znorm(arr: np.ndarray) -> np.ndarray:
    v = arr.astype(np.float32).ravel()
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-6 else v.astype(np.float32)


def _pool(arr: np.ndarray) -> np.ndarray:
    return cv2.resize(arr.astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_AREA)


def _edge_magnitude(gray: np.ndarray) -> np.ndarray:
    side = max(gray.shape)
    if side > EDGE_SIDE:
        scale = EDGE_SIDE / side
        gray = cv2.resize(gray, (max(1, round(gray.shape[1] * scale)),
                                 max(1, round(gray.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
    f = gray.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    # log1p compresses the huge dynamic range of a hard panel border so that a single
    # black gutter cannot dominate the whole correlation.
    return np.log1p(cv2.magnitude(gx, gy))


def describe(gray: np.ndarray) -> Descriptor:
    """Descriptor of a greyscale (or BGR) page."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    x, y, cw, ch = content_box(gray)
    crop = gray[y:y + ch, x:x + cw] if cw > 0 and ch > 0 else gray
    return Descriptor(tone=_znorm(_pool(crop)),
                      edge=_znorm(_pool(_edge_magnitude(crop))),
                      width=w, height=h)


def describe_bytes(data: bytes) -> Optional[Descriptor]:
    """Descriptor straight from page bytes - decoded in memory, never from a path."""
    buf = np.frombuffer(data, np.uint8)
    gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf.size else None
    return describe(gray) if gray is not None else None


def mirror(desc: Descriptor) -> Descriptor:
    """The same page flipped horizontally, for the mirroring sanity check."""
    return Descriptor(
        tone=_znorm(desc.tone.reshape(GRID, GRID)[:, ::-1]),
        edge=_znorm(desc.edge.reshape(GRID, GRID)[:, ::-1]),
        width=desc.width, height=desc.height,
    )


# --------------------------------------------------------------------- similarity
def similarity(a: Descriptor, b: Descriptor) -> float:
    """``0.5*tone + 0.5*edge`` NCC.

    Tone has the better margin and edge the better absolute separation; together they
    cover each other (tone fails on inverted/black pages, edge on flat-tone pages).
    """
    return 0.5 * float(a.tone @ b.tone) + 0.5 * float(a.edge @ b.edge)


def mirror_similarity(a: Descriptor, b: Descriptor) -> float:
    return similarity(mirror(a), b)


def similarity_matrix(rows: Sequence[Descriptor], cols: Sequence[Descriptor],
                      flip: bool = False) -> np.ndarray:
    """``len(rows) x len(cols)`` matrix of :func:`similarity`, vectorised."""
    if not rows or not cols:
        return np.zeros((len(rows), len(cols)), np.float32)
    src = [mirror(d) for d in rows] if flip else list(rows)
    at = np.stack([d.tone for d in src])
    ae = np.stack([d.edge for d in src])
    bt = np.stack([d.tone for d in cols])
    be = np.stack([d.edge for d in cols])
    return (0.5 * (at @ bt.T) + 0.5 * (ae @ be.T)).astype(np.float32)


# --------------------------------------------------------------------- cache
def cache_key(archive: str, entry: str, crc: int, size: int) -> str:
    """Identity of one cached descriptor.

    ``crc`` and ``size`` come free from the archive's directory listing, so validating
    the whole cache costs one ``infolist()``.  Every parameter that changes the numbers
    is in the key, so bumping any of them invalidates the entry rather than silently
    mixing descriptor generations.
    """
    payload = "|".join(str(x) for x in (
        DESCRIPTOR_VERSION, archive, entry, crc, size,
        GRID, EDGE_SIDE, CONTENT_PCT, ",".join(CHANNELS),
    ))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _quantise(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """int8 + per-row scale.  Halves the cache with no measurable loss at these margins."""
    scale = np.abs(vecs).max(axis=1)
    scale[scale < 1e-9] = 1.0
    q = np.rint(vecs / scale[:, None] * QUANT).clip(-QUANT, QUANT).astype(np.int8)
    return q, scale.astype(np.float32)


def _dequantise(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    vecs = q.astype(np.float32) / QUANT * scale[:, None]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return (vecs / norms).astype(np.float32)


class DescriptorCache:
    """One ``.npz`` per (archive, page group), keyed by :func:`cache_key`."""

    def __init__(self, root: Path = CACHE_ROOT) -> None:
        self.root = Path(root)
        self._loaded: Dict[Path, Dict[str, Descriptor]] = {}

    def _path(self, archive: str, group: str) -> Path:
        digest = hashlib.sha1(group.encode("utf-8")).hexdigest()[:16]
        return self.root / Path(archive).stem / f"{digest}.npz"

    def _bucket(self, archive: str, group: str) -> Dict[str, Descriptor]:
        path = self._path(archive, group)
        if path in self._loaded:
            return self._loaded[path]
        bucket: Dict[str, Descriptor] = {}
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                tone = _dequantise(data["tone_q"], data["tone_s"])
                edge = _dequantise(data["edge_q"], data["edge_s"])
                sizes = data["size"]
                for i, key in enumerate(data["keys"].tolist()):
                    bucket[str(key)] = Descriptor(tone[i], edge[i],
                                                  int(sizes[i][0]), int(sizes[i][1]))
        self._loaded[path] = bucket
        return bucket

    def get(self, archive: str, group: str, key: str) -> Optional[Descriptor]:
        return self._bucket(archive, group).get(key)

    def put(self, archive: str, group: str, key: str, desc: Descriptor) -> None:
        self._bucket(archive, group)[key] = desc

    def flush(self, archive: str, group: str) -> Optional[Path]:
        return self.flush_path(self._path(archive, group))

    def flush_all(self) -> List[Path]:
        return [p for p in (self.flush_path(path) for path in list(self._loaded)) if p]

    def flush_path(self, path: Path) -> Optional[Path]:
        bucket = self._loaded.get(path)
        if not bucket:
            return None
        keys = sorted(bucket)
        tone_q, tone_s = _quantise(np.stack([bucket[k].tone for k in keys]))
        edge_q, edge_s = _quantise(np.stack([bucket[k].edge for k in keys]))
        size = np.array([[bucket[k].width, bucket[k].height] for k in keys], np.int32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keys=np.array(keys), tone_q=tone_q, tone_s=tone_s,
                            edge_q=edge_q, edge_s=edge_s, size=size)
        return path


def describe_entries(archive, refs: Iterable, cache: Optional[DescriptorCache] = None,
                     ) -> List[Descriptor]:
    """Descriptors for ``refs`` (``corpus_index.PageRef``), reusing the cache.

    ``archive`` is an open ``corpus_index.Archive``; only the misses are read and
    decoded, so a second run over the same pages does no image work at all.
    """
    cache = cache if cache is not None else DescriptorCache()
    out: List[Descriptor] = []
    touched: set = set()
    for ref in refs:
        key = cache_key(ref.archive, ref.entry, ref.crc, ref.size)
        desc = cache.get(ref.archive, ref.group, key)
        if desc is None:
            desc = describe_bytes(archive.read(ref.entry))
            if desc is None:
                raise ValueError(f"cannot decode {ref.entry}")
            cache.put(ref.archive, ref.group, key, desc)
            touched.add((ref.archive, ref.group))
        out.append(desc)
    for archive_name, group in touched:
        cache.flush(archive_name, group)
    return out
