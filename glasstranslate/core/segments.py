"""Segment / rectangle helpers used by the pipeline loop (split out of ``core/pipeline.py``).

Pure functions on :mod:`core.types`; ``pipeline`` re-exports them for the tests.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .types import Rect, Segment, StyledSegment, TranslatedSegment

_UNK_MARKERS = ("<unk>", "⁇")  # sentencepiece / ctranslate2 unknown-token renderings


def clean_translation(text: str) -> str:
    """Drop unknown-token markers a model emits for words it has no
    vocabulary for, and collapse the whitespace left behind.  Returns an empty
    string when nothing but markers and punctuation remains, so the caller
    can fall back to the source text instead of typesetting ``⁇``."""
    cleaned = text
    for marker in _UNK_MARKERS:
        cleaned = cleaned.replace(marker, " ")
    cleaned = " ".join(cleaned.split())
    if not cleaned.strip(" .…,!?"):
        return ""
    return cleaned


def _footprint(seg: TranslatedSegment) -> Rect:
    """The frame area a live segment depends on: its quad plus, for typeset
    blocks, the layout region (the whole bubble) and the erased patch painted
    under the lettering.  A crop that re-reads the block must cover all of it
    so the bubble is flood-filled whole.

    Free text may be lettered anywhere inside ``style.search_box`` (up to a
    few em from the source), i.e. possibly outside this footprint, so a
    change under the lettering but outside the footprint does not re-read the
    block.  That is cosmetic only (the glass is excluded from capture, so the
    lettering never feeds back into OCR) and the search box is far too large
    to serve as a crop trigger, so it is deliberately left out."""
    box = seg.styled.segment.bbox
    for extra in (seg.style.layout_box, seg.style.clean_rect):
        if extra is not None:
            box = box.union(extra)
    return box


def _offset_translated(seg: TranslatedSegment, dx: int, dy: int) -> TranslatedSegment:
    """Copy of ``seg`` with every frame coordinate (quad and style) moved."""
    inner = seg.styled.segment
    moved = Segment(
        inner.text, inner.quad + np.array([dx, dy], dtype=np.float32), inner.confidence, inner.lang_hint
    )
    return TranslatedSegment(
        styled=StyledSegment(moved, seg.styled.style.shifted(dx, dy), seg.styled.src_lang),
        translation=seg.translation,
        tgt_lang=seg.tgt_lang,
        from_cache=seg.from_cache,
    )


def _expand(rect: Rect, px: int, width: int, height: int) -> Rect:
    """Grow ``rect`` by ``px`` on every side and clamp to ``width x height``."""
    return Rect(rect.x - px, rect.y - px, rect.w + 2 * px, rect.h + 2 * px).clamp(width, height)


def _merge_rects(rects: Sequence[Rect]) -> List[Rect]:
    """Union intersecting rectangles until none intersect (order-independent)."""
    merged = [r for r in rects if r.w > 0 and r.h > 0]
    changed = True
    while changed:
        changed = False
        out: List[Rect] = []
        for r in merged:
            for i, o in enumerate(out):
                if r.intersects(o):
                    out[i] = o.union(r)
                    changed = True
                    break
            else:
                out.append(r)
        merged = out
    return merged


def _offset_segment(seg: Segment, dx: int, dy: int) -> Segment:
    """Copy of an OCR ``seg`` moved from crop to frame coordinates."""
    quad = seg.quad.copy()
    quad[:, 0] += dx
    quad[:, 1] += dy
    return Segment(text=seg.text, quad=quad, confidence=seg.confidence, lang_hint=seg.lang_hint)
