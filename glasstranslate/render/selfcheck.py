"""Reference-free self-checks on a rendered block.

Every metric in ``demo/`` compares us with the official release, so none of it
can run outside the corpus harness.  These three checks need only our own
output, which means they hold in real use - on a page nobody has an English
edition of - and they catch the defects a reader notices first:

* :func:`text_complete` - **did we draw every character we were handed?**  The
  typesetter can silently drop words when nothing fits; the reader sees half a
  sentence in a balloon.
* :func:`breaks_clean` - **does the drawn text read back as the text we were
  given?**  A break that eats its space sets ``BEINGAN ADULTISSO CONFUSING!``.
  A break is legal at a space, or mid-word behind a hyphen (either one the
  typesetter inserted or one the text already carried).
* :func:`ink_inside_bubble` - **is any source ink still readable inside the
  balloon?**  Furigana the eraser never saw survives as a stray Japanese glyph
  next to English lettering.

The first two are pure string functions - no numpy, no cv2, no page.  The
third needs the erased page and the balloon interior, both of which the render
pass already has.  Nothing here imports torch, the sidecar, or anything under
``demo/``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

# Grey level below which a pixel is ink on light paper (above ``255 - INK_THR``
# on dark paper).  Matches ``demo/typeset_reference.INK_THR`` so the self-check
# and the corpus metric agree about what ink is.
INK_THR = 128
# Ems of the interior eroded away before looking for surviving ink.  The eraser
# deliberately keeps the balloon outline and its anti-aliased fringe, so without
# this band every balloon reports its own outline as leftover source ink.
# Matches ``demo/typeset_metrics.OUTLINE_BAND_EM``.
OUTLINE_BAND_EM = 0.25


def _squeeze(text: str) -> str:
    """``text`` with every space and hyphen removed - what survives a line
    break either way, so two strings that differ only in how they were broken
    squeeze to the same thing."""
    return "".join(c for c in text if not c.isspace() and c != "-")


def drawn_text(lines: Sequence[str]) -> str:
    """The lines joined back into one string, spaces restored at the breaks.

    This is what the reader reads, not what was drawn: a hyphen the typesetter
    inserted to break a word is left in place, because :func:`breaks_clean` is
    what decides whether it was legitimate.
    """
    return " ".join(line for line in lines if line)


def text_complete(text: str, lines: Sequence[str]) -> Optional[bool]:
    """Did every character of ``text`` reach the page?

    Compared with spaces and hyphens squeezed out of both sides, so a block
    broken over four lines and hyphenated twice still compares equal to the
    string it was given.  This asks only about *completeness* - whether the
    breaks were sane is :func:`breaks_clean`.

    **None when nothing was drawn at all**, which is a different state and not
    this check's business: a block the typesetter produced no lines for was not
    lettered badly, it was not lettered, and the score already punishes that
    through ``answered``.  Measured before this returned None, all 16 blocks it
    flagged over the 50-page sample were page numbers and stray punctuation
    (``'80'``, ``'183'``, ``'!!'``) that the renderer deliberately never set -
    noise, not the fragment defect this exists to catch.
    """
    drawn = [line for line in lines if line]
    if not drawn:
        return None
    return _squeeze(text) == _squeeze(" ".join(drawn))


def breaks_clean(text: str, lines: Sequence[str]) -> Optional[bool]:
    """Does the drawn block read back as ``text``, breaking only where it may?

    Walks the lines against the source string.  A break is legal when it falls
    on a space, or mid-word behind a hyphen - one the typesetter inserted
    (``PRECIOUS-`` / ``NESS``) or one the text already carried (``SEN-`` /
    ``PAI``).  Anything else means a space was eaten or a character invented,
    which is the ``BEINGAN ADULTISSO CONFUSING!`` defect.

    **None when nothing was drawn**, for the reason :func:`text_complete` gives:
    an unlettered block is not a badly broken one, and charging it here would
    double-count the same 16 page numbers.
    """
    source = " ".join(text.split())
    drawn = [line for line in lines if line]
    if not drawn:
        return None
    position = 0
    for index, line in enumerate(drawn):
        if source.startswith(line, position):
            position += len(line)
        elif line.endswith("-") and source.startswith(line[:-1], position):
            # A hyphen the typesetter added: it is on the page but not in the
            # source, so the word continues on the next line with no space.
            position += len(line) - 1
            continue
        else:
            return False
        if index == len(drawn) - 1:
            break
        if position < len(source) and source[position] == " ":
            position += 1
        elif not line.endswith("-"):
            return False
    return position == len(source)


def ink_inside_bubble(erased_gray: np.ndarray, interior: Optional[np.ndarray],
                      em_px: float, dark: bool = False) -> Optional[int]:
    """Source ink still readable inside a balloon after the erase pass, in pixels.

    Judged on the *erased* page - before any lettering is drawn on it - inside
    the interior eroded by :data:`OUTLINE_BAND_EM` ems, so the outline the
    eraser deliberately keeps never counts.  This is what catches furigana the
    eraser never saw.

    None when the block has no interior (free text over artwork has no balloon
    to be clean inside of).  A clean balloon returns 0, which is what the gate
    treats as the invariant holding.
    """
    if interior is None:
        return None
    mask = np.asarray(interior).astype(bool)
    if not mask.any():
        return None
    margin = max(1, int(round(OUTLINE_BAND_EM * max(float(em_px), 1e-6))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    core = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    if not core.any():
        return 0
    gray = np.asarray(erased_gray)
    ink = (gray > 255 - INK_THR) if dark else (gray < INK_THR)
    return int((ink & core).sum())
