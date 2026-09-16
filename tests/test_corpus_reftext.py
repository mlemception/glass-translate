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


# -------------------------------------------- joining a block's printed lines
# The published page breaks words across lines and the OCR reads each printed
# line separately.  Joining with a space made ``engage-`` + ``ment`` into
# ``ENGAGE- MENT``: two tokens where the page has one word, which our renderer
# then wrapped at.  29 of 316 blocks in the 50-page sample carried one.

from reftext_join import join_reference_lines, should_drop_hyphen  # noqa: E402


def test_a_word_broken_across_printed_lines_is_rejoined() -> None:
    """The behavioural change: the old code produced 'engage- ment'."""
    assert join_reference_lines(["engage-", "ment of the"]) == "engagement of the"
    assert join_reference_lines(["danger-", "ous"]) == "dangerous"
    assert join_reference_lines(["remem-", "ber"]) == "remember"


def test_the_target_three_example_is_one_word_again() -> None:
    """v11_p133_808ab5 b4 read 'AUSPI- CIOUS BEAST SUMMON'."""
    assert join_reference_lines(["auspi-", "cious beast summon"]) == "auspicious beast summon"


def test_ordinary_lines_still_join_with_one_space() -> None:
    assert join_reference_lines(["the size of", "utahime's ribbon"]) == "the size of utahime's ribbon"
    assert join_reference_lines(["  a  ", "", "  b  "]) == "a b"


def test_a_folded_em_dash_is_not_a_line_break_hyphen() -> None:
    """compose._FOLD_TABLE maps the em dash to '--'; it must not fuse words."""
    assert join_reference_lines(["wait--", "he said"]) == "wait-- he said"


def test_a_non_alphabetic_stem_keeps_its_hyphen_but_still_loses_the_space() -> None:
    """Removing the space is always right; removing the hyphen is a judgement.

    ``11-`` is not a word continuing, so the hyphen stays - but ``11-seconds``
    is still better than the ``11- seconds`` the old join produced, and it is
    one token, so it cannot wrap at the fake break.
    """
    assert join_reference_lines(["for 11-", "seconds"]) == "for 11-seconds"


def test_a_single_line_and_an_empty_block_are_unchanged() -> None:
    assert join_reference_lines(["only one"]) == "only one"
    assert join_reference_lines([]) == ""
    assert join_reference_lines(["", "   "]) == ""


def test_the_hyphen_is_kept_when_the_evidence_does_not_say_to_drop_it() -> None:
    """False is the safe answer: a kept hyphen reads as a compound, a dropped
    one fuses two words."""
    assert should_drop_hyphen("", "word") is False
    assert should_drop_hyphen("word", "") is False
    assert should_drop_hyphen("11", "seconds") is False
    assert should_drop_hyphen("o'clock", "ish") is False


def test_a_compound_broken_at_its_own_hyphen_is_a_KNOWN_MISS() -> None:
    """Documented limitation, pinned so an improvement is noticed.

    'specialgrade' really does have a legal hyphenation point after 'special',
    so pyphen cannot separate this from a line break.  The space is still
    removed, so the line-count repair holds; only the hyphen is wrong.  A
    corpus-internal check would fix 'fushi-guro' but needs a cross-page index.
    """
    assert join_reference_lines(["special-", "grade"]) == "specialgrade"  # wanted: special-grade
    assert join_reference_lines(["fushi-", "guro"]) == "fushi-guro"  # wanted: fushiguro


def test_a_bare_string_is_one_line_not_a_list_of_characters() -> None:
    """``str`` satisfies ``Iterable[str]``, so this is an easy caller mistake."""
    assert join_reference_lines("engage- ment") == "engage- ment"


def test_a_productive_prefix_keeps_its_hyphen() -> None:
    """``nonsorcerers`` has a legal break after ``non``, so pyphen alone fused
    it three times on one page.  The prefix list is what stops that."""
    assert join_reference_lines(["non-", "sorcerers"]) == "non-sorcerers"
    assert join_reference_lines(["self-", "aware"]) == "self-aware"
    assert should_drop_hyphen("non", "sorcerers") is False


def test_the_whole_trailing_word_is_captured_not_a_suffix_of_it() -> None:
    """Regression: a lookbehind once cut the first letter off, so ``NON-`` was
    read as ``ON-`` and the prefix list never matched."""
    from reftext_join import _BREAK

    assert _BREAK.search("PROTECT NON-").group(1) == "NON"
    assert _BREAK.search("ENGAGE-").group(1) == "ENGAGE"
    assert _BREAK.search("wait--") is None


# The stored reftext files were bootstrapped with the old space-join and are
# never rewritten (typeset_reference only writes when the file is absent, so
# hand corrections survive) and are gitignored, so they are not safely
# revertible.  The repair is applied on READ instead.

from reftext_join import repair_joined_text  # noqa: E402


def test_an_already_joined_string_is_repaired_on_read() -> None:
    assert repair_joined_text("ENGAGE- MENT IS HARD") == "ENGAGEMENT IS HARD"
    assert repair_joined_text("FOR 11 SEC- ONDS..") == "FOR 11 SECONDS.."
    assert repair_joined_text("NON- SORCERERS...") == "NON-SORCERERS..."
    assert repair_joined_text("meet- and-greet") == "meet-and-greet"


def test_the_repair_leaves_clean_text_alone_and_is_idempotent() -> None:
    for text in ("nothing to fix here", "wait-- he said", "", "a-b c-d"):
        assert repair_joined_text(text) == text
    once = repair_joined_text("ENGAGE- MENT")
    assert repair_joined_text(once) == once


def test_the_read_repair_and_the_line_join_agree() -> None:
    """Two entry points, one decision - they must not drift apart."""
    for left, right in (("engage", "ment"), ("non", "sorcerers"), ("first", "years"),
                        ("auspi", "cious"), ("sec", "onds")):
        assert join_reference_lines([left + "-", right]) == repair_joined_text(f"{left}- {right}")
