"""Geometric furigana (ruby) detection over raw OCR segments.

The PaddleOCR path (``rapidocr`` running PP-OCRv5) emits ruby annotations as
separate small lines next to the text they annotate: a narrow kana column
beside a vertical column, or a short kana line above a horizontal one.
Translated on their own those lines are noise, so the plain (non-manga)
pipeline drops them with :func:`strip_furigana`.  manga-ocr ignores furigana
at recognition time and does not need this.

Manga mode has its own pixel-aware rule in ``render/layout.py``
(``_mark_furigana``), which also feeds the eraser with the ruby boxes; this
module deliberately does not replace it.  That rule is the stricter of the
two: it asks the annotated line to carry a kanji, because what it flags is
dropped from the translation and a kana column standing beside a slightly
bigger kana one is speech, not ruby.  The deliberate cost is that a kana gloss
printed beside a katakana or Latin word is not recognised there and is
translated as a line of its own; this module, which only filters text, still
takes it.

Only axis-aligned bounding boxes (``Segment.bbox``) are considered; rotated
text is left alone.
"""
from __future__ import annotations

import logging
from typing import FrozenSet, List, Sequence

from glasstranslate.core.types import Rect, Segment

log = logging.getLogger(__name__)

# A ruby line's glyph size is at most this fraction of its parent's.
FURIGANA_MAX_HEIGHT_RATIO = 0.6
# Gap between the ruby and its parent, in parent glyph sizes.
FURIGANA_MAX_GAP_RATIO = 0.6
# Parents with smaller glyphs than this cannot carry legible ruby.
MIN_PARENT_GLYPH_PX = 8
# Share of the ruby's extent along the parent that must lie beside / over it.
_MIN_OVERLAP = 0.5
_PUNCTUATION = "。、！？…・「」"


def _is_kana(ch: str) -> bool:
    """Hiragana, katakana, the prolonged-sound mark or a voicing mark.
    Mirrors ``render.layout._is_kana`` (duplicated so this package stays
    free of cv2 / the render stack)."""
    o = ord(ch)
    return 0x3041 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF or ch in "ー゛゜"


def is_kana_text(text: str) -> bool:
    """True when every letter of ``text`` is kana (spaces and Japanese
    punctuation ignored).  Kanji, Latin and empty strings are never kana."""
    letters = [c for c in text if not c.isspace() and c not in _PUNCTUATION]
    return bool(letters) and all(_is_kana(c) for c in letters)


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return min(a1, b1) - max(a0, b0)


def _beside_column(ruby: Rect, parent: Rect) -> bool:
    """Ruby of a vertical column: a narrow column immediately left or right
    of it that overlaps it vertically."""
    glyph = parent.w
    if ruby.w > FURIGANA_MAX_HEIGHT_RATIO * glyph:
        return False
    centre_x = ruby.x + ruby.w / 2.0
    if parent.x <= centre_x <= parent.x2:
        return False  # inside the column, not beside it
    gap = max(ruby.x - parent.x2, parent.x - ruby.x2)
    if gap > FURIGANA_MAX_GAP_RATIO * glyph:
        return False
    return _overlap(ruby.y, ruby.y2, parent.y, parent.y2) >= _MIN_OVERLAP * ruby.h


def _above_line(ruby: Rect, parent: Rect) -> bool:
    """Ruby of a horizontal line: a short line immediately above it that
    overlaps it horizontally.  Ruby never sits below horizontal text."""
    glyph = parent.h
    if ruby.h > FURIGANA_MAX_HEIGHT_RATIO * glyph:
        return False
    centre_y = ruby.y + ruby.h / 2.0
    if centre_y >= parent.y:
        return False
    gap = parent.y - ruby.y2
    if gap > FURIGANA_MAX_GAP_RATIO * glyph:
        return False
    return _overlap(ruby.x, ruby.x2, parent.x, parent.x2) >= _MIN_OVERLAP * ruby.w


def is_ruby_of(ruby: Rect, parent: Rect) -> bool:
    """True when ``ruby`` is placed like furigana of ``parent`` (geometry
    only; the caller checks the text).  ``parent`` is vertical text when it
    is taller than wide, horizontal otherwise."""
    if ruby.w <= 0 or ruby.h <= 0:
        return False
    if min(parent.w, parent.h) < MIN_PARENT_GLYPH_PX:
        return False
    if parent.h > parent.w:
        return _beside_column(ruby, parent)
    return _above_line(ruby, parent)


def furigana_indices(segments: Sequence[Segment]) -> FrozenSet[int]:
    """Indices of the segments judged to be furigana: kana-only text placed
    like ruby next to some other (larger) segment."""
    boxes = [s.bbox for s in segments]
    found = set()
    for i, seg in enumerate(segments):
        if not is_kana_text(seg.text):
            continue
        if any(j != i and is_ruby_of(boxes[i], parent) for j, parent in enumerate(boxes)):
            found.add(i)
    return frozenset(found)


def strip_furigana(segments: Sequence[Segment]) -> List[Segment]:
    """A new list of ``segments`` without the furigana ones (order kept).
    The input sequence and its segments are not modified."""
    ruby = furigana_indices(segments)
    if ruby:
        log.debug("dropping %d furigana line(s): %s", len(ruby), [segments[i].text for i in sorted(ruby)])
    return [s for i, s in enumerate(segments) if i not in ruby]
