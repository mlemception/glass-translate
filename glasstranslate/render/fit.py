"""Fit a string into a box by wrapping and shrinking.  Pure, deterministic,
no Qt/PIL dependency: the caller supplies a ``measure(text, size)`` callable
returning ``(width, height)`` for that text at that font size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

Measure = Callable[[str, float], tuple[float, float]]

# Multiplicative shrink step used when nothing fits at the current size.
_SHRINK_FACTOR = 0.9
# Fraction of the box height a single line starts at (leaves room for
# ascenders/descenders that the OCR quad tends to clip).
_START_HEIGHT_FRACTION = 0.85


@dataclass(frozen=True)
class FitResult:
    """Chosen font ``size`` and the ``lines`` to draw (never empty)."""

    size: float
    lines: List[str]


def _break_word(word: str, size: float, box_w: float, measure: Measure) -> List[str]:
    """Split a single over-wide word into the fewest chunks that each fit
    ``box_w`` (a single character is never split further)."""
    chunks: List[str] = []
    current = ""
    for ch in word:
        candidate = current + ch
        if current and measure(candidate, size)[0] > box_w:
            chunks.append(current)
            current = ch
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def wrap_lines(text: str, size: float, box_w: float, measure: Measure) -> List[str]:
    """Greedy word-wrap of ``text`` at ``size`` into lines no wider than
    ``box_w``.  Words wider than the box are broken at character level.
    Text without spaces (CJK) is broken at character level as well.
    """
    words = text.split()
    if not words:
        return [text]
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if measure(candidate, size)[0] <= box_w:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        if measure(word, size)[0] <= box_w:
            current = word
        else:
            pieces = _break_word(word, size, box_w, measure)
            lines.extend(pieces[:-1])
            current = pieces[-1]
    if current:
        lines.append(current)
    return lines


def _lines_fit(lines: List[str], size: float, box_w: float, box_h: float, measure: Measure) -> bool:
    total_h = 0.0
    for line in lines:
        w, h = measure(line, size)
        if w > box_w:
            return False
        total_h += h
    return total_h <= box_h


def fit_text(
    text: str,
    box_w: float,
    box_h: float,
    measure: Measure,
    *,
    max_lines: int = 3,
    min_size: float = 6.0,
    start_size: Optional[float] = None,
) -> FitResult:
    """Choose a font size and line breaks so ``text`` fits a ``box_w`` x
    ``box_h`` box, maximising the font size.

    For every line count ``n`` in ``1..max_lines`` the search starts at
    ``min(start_size, box_h / n)`` (``start_size`` defaults to
    ``box_h * 0.85``) and shrinks by 10 % steps until the word-wrapped text
    occupies at most ``n`` lines that all fit the width and together fit the
    height.  Over-wide words are broken at character level.  The largest
    size found wins; ties prefer fewer lines, so a single line that fits at
    ``start_size`` is always returned as-is.  If nothing fits even at
    ``min_size`` the text is wrapped at ``min_size`` and returned anyway.

    The result never has empty ``lines``; empty input yields ``[""]``.
    """
    text = text.strip()
    start = start_size if start_size is not None else box_h * _START_HEIGHT_FRACTION
    start = max(start, min_size)

    if not text:
        return FitResult(start, [""])

    best: Optional[FitResult] = None
    for n in range(1, max(max_lines, 1) + 1):
        size = max(min(start, box_h / n), min_size) if n > 1 else start
        if best is not None and size <= best.size:
            break  # smaller start than what we already have: cannot improve
        while True:
            lines = wrap_lines(text, size, box_w, measure)
            if len(lines) <= n and _lines_fit(lines, size, box_w, box_h, measure):
                if best is None or size > best.size:
                    best = FitResult(size, lines)
                break
            if size <= min_size or (best is not None and size * _SHRINK_FACTOR <= best.size):
                break
            size = max(size * _SHRINK_FACTOR, min_size)

    if best is None:
        return FitResult(min_size, wrap_lines(text, min_size, box_w, measure))
    return best
