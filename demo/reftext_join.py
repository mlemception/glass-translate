"""Join the OCR'd lines of one reference block back into a single string.

The published English page breaks words across lines, and the OCR reads each
printed line separately.  Joining those lines with a space - which is what
``typeset_reference`` did - turns ``engage-`` + ``ment`` into ``ENGAGE- MENT``:
two tokens where the page has one word.  Our renderer then wraps at that fake
break, so the harness marks a typesetting defect that is really a transcription
defect.  Measured over the 50-page sample, **29 of 316 blocks** carried one.

Removing the *space* is always right.  Removing the *hyphen* is not, because a
compound can be broken at its own hyphen:

    engage- / ment      -> ENGAGEMENT      (the hyphen was the line break)
    special- / grade    -> SPECIAL-GRADE   (the hyphen belongs to the word)

No spell-checker is installed and bootstrapping a vocabulary from the corpus is
refuted (see ``docs/NOTES.md``), so the decision is made with ``pyphen``,
which carries hyphenation *patterns* rather than a word list: if the solid form
could legally be hyphenated at exactly that point, the hyphen was the break.
Measured against 14 hand-labelled cases it agrees on 12.

The two it misses both still lose the space, so the line-count repair holds
either way; only the hyphen is wrong.  ``SPECIAL-GRADE`` fuses (``specialgrade``
really does have a legal break after ``special``), and ``FUSHI-GURO`` keeps a
hyphen it should not.  A corpus-internal check - does the solid form appear
elsewhere in the reference? - was measured and is right 2/2, but only 2 of the
14 cases have any such evidence and it needs a cross-page index, so it is left
out deliberately rather than forgotten.

Pure strings: no OCR, no images, no network.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

# A line-ending hyphen is a break only when a single hyphen closes a run of
# word characters.  ``--`` (the folded em dash, see compose._FOLD_TABLE) cannot
# match: its final hyphen is preceded by a hyphen, which is not a ``\w``.
# ``search`` is leftmost-then-greedy, so the group is the WHOLE trailing word.
_BREAK = re.compile(r"(\w+)-$", re.UNICODE)
_FIRST_WORD = re.compile(r"^(\w+)", re.UNICODE)

_pyphen_cache: List[Optional[object]] = []


def _hyphenator():
    """The ``pyphen`` dictionary, or None when it is unavailable."""
    if not _pyphen_cache:
        try:
            import pyphen

            _pyphen_cache.append(pyphen.Pyphen(lang="en_US"))
        except Exception:  # pragma: no cover - pyphen is a declared dependency
            _pyphen_cache.append(None)
    return _pyphen_cache[0]


# Productive prefixes that stay hyphenated by convention, so a hyphen after one
# of them belongs to the word rather than to the line break.  ``pyphen`` cannot
# see this: ``nonsorcerers`` has a perfectly legal break after ``non``, which
# fused ``NON- SORCERERS`` three times on one page before this list existed.
_HYPHENATED_PREFIXES = frozenset({
    "non", "self", "ex", "anti", "semi", "pre", "post", "co", "multi", "pro",
    "sub", "super", "ultra", "mid", "all", "cross", "counter", "pseudo", "quasi",
})


def should_drop_hyphen(left: str, right: str) -> bool:
    """True when ``left-`` + ``right`` is one word broken across printed lines.

    ``left`` is the word fragment before the hyphen, ``right`` the fragment
    that opens the next line.  False keeps the hyphen, which is the safe answer
    whenever the evidence is absent: a wrongly kept hyphen reads as a compound,
    a wrongly dropped one fuses two words.
    """
    if not left or not right or not left.isalpha() or not right.isalpha():
        return False
    if left.lower() in _HYPHENATED_PREFIXES:
        return False
    dic = _hyphenator()
    if dic is None:
        return False
    return len(left) in set(dic.positions((left + right).lower()))


def join_reference_lines(texts: Iterable[str]) -> str:
    """One block's printed lines, in reading order, as a single string.

    A line ending in a word-hyphen joins to the next with no space, and the
    hyphen survives only when :func:`should_drop_hyphen` says it belongs to the
    word.  Every other line join is a single space, as before.
    """
    if isinstance(texts, str):  # a str is an Iterable[str] of characters
        texts = [texts]
    lines = [t.strip() for t in texts if t and t.strip()]
    if not lines:
        return ""
    out = lines[0]
    for nxt in lines[1:]:
        match = _BREAK.search(out)
        word = _FIRST_WORD.match(nxt)
        if match and word:
            if should_drop_hyphen(match.group(1), word.group(1)):
                out = out[: match.end() - 1] + nxt  # drop the hyphen, close the word
            else:
                out = out + nxt  # keep the hyphen, close the gap
            continue
        out = out + " " + nxt
    return out


# The stored ``reftext/reference_text_<page>.json`` files were bootstrapped with
# the old space-join and are never rewritten (``typeset_reference`` only writes
# when the file is absent, so hand corrections survive).  They are also
# gitignored, so they are not safely revertible.  The same repair is therefore
# applied when the text is READ, which fixes the whole corpus without touching a
# single stored file.  It is idempotent: text with no ``word- word`` left in it
# is returned unchanged.
_JOINED_BREAK = re.compile(r"(\w+)- (\w+)", re.UNICODE)


def repair_joined_text(text: str) -> str:
    """Undo an old space-join inside an already-assembled reference string."""
    def _fix(match: "re.Match[str]") -> str:
        left, right = match.group(1), match.group(2)
        return left + right if should_drop_hyphen(left, right) else left + "-" + right

    return _JOINED_BREAK.sub(_fix, text)
