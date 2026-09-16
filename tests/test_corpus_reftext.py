"""The English we letter with, and the gate that decides a re-read is better.

The reference text is OCR'd off the published English page.  Measured over the
60 paired pages: 7.1 % of blocks carry a token that cannot be a word, and the
confidence score cannot separate them - the visibly-wrong lines have a p90
confidence of **1.000** (``GHHH`` scores 1.000, ``3W`` 0.995, ``MT.`` 0.998).
So the gate is lexical, and confidence is not part of it.

Re-reading a line's own crop upscaled recovers the text (``THO T TO`` ->
``THEM TO DO``), but the *correct* re-read often scores lower than the garbage
it replaces - 0.723 and 0.574 against 0.755 and 0.704.  A re-read is therefore
accepted only when it reduces the count of tokens that cannot be words.

No corpus page, no OCR, no network: every fixture here is a literal string.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

import corpus_metrics as CM  # noqa: E402


# ------------------------------------------------------------ the detector

def test_real_lettering_has_no_suspicious_tokens() -> None:
    for text in (
        "...IF THEY",
        "HAVE TO DIE,",
        "THEN I WANT",
        "THEM TO DO",
        "SO IN THE",
        "BEST WAY",
        "POSSIBLE.",
        "SPECIAL- GRADE CURSED OBJECT RYOMEN SUKUNA.",
        "FORGET IT. IT'S DANGER- OUS. JUST HAND IT OVER.",
        "A",          # the article
        "I",          # the pronoun
    ):
        assert CM.suspicious_tokens(text) == [], f"{text!r} was called garbled"


def test_comic_noises_are_not_suspicious() -> None:
    """All-consonant strings that really are words in lettering."""
    for text in ("HMPH", "TSK", "SHH", "BRR", "PSST", "MR. GOJO", "ST. LOUIS"):
        assert CM.suspicious_tokens(text) == [], f"{text!r} was called garbled"


def test_the_known_garbled_lines_are_caught() -> None:
    assert CM.suspicious_tokens("THO T TO") == ["T"]
    assert CM.suspicious_tokens("SON I TS") == ["TS"]
    assert CM.suspicious_tokens("P OEI") == ["P"]
    assert CM.suspicious_tokens("TTS STAY") == ["TTS"]
    assert CM.suspicious_tokens("HS") == ["HS"]


def test_a_lone_letter_that_is_not_a_or_i_is_garbled() -> None:
    assert CM.suspicious_tokens("THE P WORD") == ["P"]
    assert CM.suspicious_tokens("A WORD") == []
    assert CM.suspicious_tokens("I WORD") == []


# --------------------------------------------------------------- the gate

def test_a_cleaner_reread_is_taken() -> None:
    """The measured case: two suspicious tokens become none."""
    assert CM.reread_is_better("THO T TO", "THEM TO DO") is True
    assert CM.reread_is_better("SON I TS", "sO IN THE") is True
    assert CM.reread_is_better("P OEI", "PEOPLE I") is True


def test_a_dirtier_reread_is_refused() -> None:
    """Also measured: these two re-reads are worse than what they replace."""
    assert CM.reread_is_better("INOO I", "I LNO T") is False
    assert CM.reread_is_better("I 'ON", "011... I 'ON") is False


def test_an_equally_clean_reread_is_refused() -> None:
    """No lexical reason to prefer it, so the original stands.

    This is what keeps the pass conservative: it can only ever reduce the
    number of tokens that cannot be words, never churn the text sideways.
    """
    assert CM.reread_is_better("I OS", "SO I") is False
    assert CM.reread_is_better("HAVE TO DIE,", "HAVE TO DIE,") is False


def test_an_empty_reread_is_refused() -> None:
    assert CM.reread_is_better("THO T TO", "") is False
    assert CM.reread_is_better("THO T TO", "   ") is False


def test_confidence_is_not_consulted() -> None:
    """The gate takes text only.

    The correct re-reads measured 0.723 and 0.574 against the garbage's 0.755
    and 0.704, so any confidence term would reject exactly the repairs that
    matter.  Pinned as a signature so it cannot quietly grow one.
    """
    import inspect

    params = list(inspect.signature(CM.reread_is_better).parameters)
    assert params == ["original", "reread"], f"the gate grew a parameter: {params}"


def test_held_sound_effects_are_not_garbled() -> None:
    """A letter repeated three times running is lettering, not a misread."""
    for text in ("GHHH", "BRRRR", "PFFFT", "AAAGH", "NNNGH"):
        assert CM.suspicious_tokens(text) == [], f"{text!r} was called garbled"
    # Two in a row is not enough: TTS really is a misread of "IT'S".
    assert CM.suspicious_tokens("TTS") == ["TTS"]


def test_ordinals_and_abbreviations_survive() -> None:
    """``23rd`` strips to ``rd``, which is vowel-less and is not a word.

    The token it came from is perfectly good text, so the suffixes are listed.
    """
    for text in ("23rd", "26th", "25th", "2nd", "MT. FUJI", "IS MT."):
        assert CM.suspicious_tokens(text) == [], f"{text!r} was called garbled"
