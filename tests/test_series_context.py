"""Tests for ``glasstranslate.translate.context`` (F5-1 series-context template engine).

Pure functions only: no Qt, no I/O, no network.  The template carries a
``[Series Name]`` placeholder that is substituted case-insensitively; an empty
series name strips the placeholder and tidies the whitespace it leaves behind.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect

import pytest

from glasstranslate.translate import context as context_mod
from glasstranslate.translate.context import (
    DEFAULT_TEMPLATE,
    MAX_SERIES_CHARS,
    MAX_TEMPLATE_CHARS,
    PLACEHOLDER,
    ResolvedPrompt,
    context_key,
    normalize_series_name,
    resolve_prompt,
    validate_template,
)

TEMPLATE = (
    "Translate manga dialogue.\n"
    "\n"
    "The text comes from the series [Series Name]; use its established names.\n"
    "\n"
    "Output only the translation."
)


# ------------------------------------------------------------ substitution
def test_placeholder_is_substituted_with_the_series_name():
    result = resolve_prompt(TEMPLATE, "Frieren")

    assert "from the series Frieren; use" in result.prompt
    assert PLACEHOLDER not in result.prompt
    assert result.warning is None
    assert result.used_default is False


@pytest.mark.parametrize(
    "variant", ["[series name]", "[SERIES NAME]", "[ Series Name ]", "[Series  Name]"]
)
def test_placeholder_match_is_case_insensitive_and_tolerates_inner_spaces(variant):
    result = resolve_prompt(f"Series: {variant}. Translate.", "Oshi no Ko")

    assert result.prompt == "Series: Oshi no Ko. Translate."


def test_multiple_placeholder_occurrences_are_all_replaced():
    template = "[Series Name] rules. Stay faithful to [series name]. End [SERIES NAME]."

    result = resolve_prompt(template, "Dungeon Meshi")

    assert result.prompt == (
        "Dungeon Meshi rules. Stay faithful to Dungeon Meshi. End Dungeon Meshi."
    )


def test_series_name_with_backslash_is_inserted_verbatim():
    result = resolve_prompt(TEMPLATE, r"Re\Zero \1 \g<0>")

    assert r"from the series Re\Zero \1 \g<0>; use" in result.prompt


# --------------------------------------------------------- empty series name
def test_empty_series_name_strips_placeholder_and_collapses_whitespace():
    result = resolve_prompt(TEMPLATE, "")

    assert result.prompt == (
        "Translate manga dialogue.\n"
        "\n"
        "The text comes from the series; use its established names.\n"
        "\n"
        "Output only the translation."
    )
    assert result.warning is None
    assert result.used_default is False


def test_empty_series_name_fixes_space_before_comma_and_period():
    result = resolve_prompt("Names from [Series Name] , tone from [Series Name] .", "   ")

    assert result.prompt == "Names from, tone from."


def test_empty_series_name_collapses_blank_line_runs():
    template = "Line A.\n\n[Series Name]\n\nLine B.\n\n\n\nLine C."

    result = resolve_prompt(template, "")

    assert result.prompt == "Line A.\n\nLine B.\n\nLine C."


def test_empty_series_name_strips_trailing_spaces_per_line_and_the_whole_prompt():
    template = "\n\nSeries: [Series Name]\t \nRules:\t[Series Name]\n\n"

    result = resolve_prompt(template, "")

    assert result.prompt == "Series:\nRules:"


# ------------------------------------------------------------ fallbacks
def test_empty_template_falls_back_to_default_with_reason():
    result = resolve_prompt("   \n\t", "Frieren")

    assert result.used_default is True
    assert result.warning == "Prompt template invalid (template is empty); using the default"
    assert result.prompt == resolve_prompt(DEFAULT_TEMPLATE, "Frieren").prompt
    assert "Frieren" in result.prompt


def test_template_without_placeholder_falls_back_with_reason():
    result = resolve_prompt("Translate everything into English.", "Frieren")

    assert result.used_default is True
    assert result.warning == (
        "Prompt template invalid (template has no [Series Name] placeholder); using the default"
    )
    assert "Frieren" in result.prompt


def test_oversized_template_falls_back_with_reason():
    oversized = PLACEHOLDER + "x" * MAX_TEMPLATE_CHARS

    result = resolve_prompt(oversized, "Frieren")

    assert result.used_default is True
    assert result.warning == (
        "Prompt template invalid (template is longer than 4000 characters); using the default"
    )
    assert "xxxx" not in result.prompt


def test_template_at_the_size_cap_is_still_valid():
    at_cap = PLACEHOLDER + "x" * (MAX_TEMPLATE_CHARS - len(PLACEHOLDER))

    assert len(at_cap) == MAX_TEMPLATE_CHARS
    assert validate_template(at_cap) is None


@pytest.mark.parametrize("bad", [None, 123, b"[Series Name]", ["[Series Name]"], object()])
def test_non_string_template_falls_back_as_empty(bad):
    assert validate_template(bad) == "template is empty"

    result = resolve_prompt(bad, "Frieren")

    assert result.used_default is True
    assert result.warning == "Prompt template invalid (template is empty); using the default"


def test_validate_template_reports_each_reason():
    assert validate_template("") == "template is empty"
    assert validate_template("   ") == "template is empty"
    assert validate_template("no placeholder here") == "template has no [Series Name] placeholder"
    assert validate_template("[Series Name]" * 400) == "template is longer than 4000 characters"
    assert validate_template(TEMPLATE) is None


# ------------------------------------------------------- name normalisation
def test_series_name_is_trimmed_and_internal_whitespace_collapsed():
    assert normalize_series_name("  Frieren:   Beyond\tJourney's\n\nEnd  ") == (
        "Frieren: Beyond Journey's End"
    )


def test_series_name_is_capped_at_max_series_chars():
    assert MAX_SERIES_CHARS == 80

    capped = normalize_series_name("A" * 81)

    assert len(capped) == 80
    assert capped == "A" * 80


def test_series_name_cap_strips_a_dangling_space_after_the_cut():
    name = "A" * 79 + " " + "B" * 20

    capped = normalize_series_name(name)

    assert capped == "A" * 79
    assert not capped.endswith(" ")


def test_control_characters_are_removed_from_the_series_name():
    assert normalize_series_name("Fri\x00eren\x07 \x1b[0m Beyond\x7f\u200b") == (
        "Frieren [0m Beyond"
    )


def test_series_name_accepts_non_string_values():
    assert normalize_series_name(None) == ""
    assert normalize_series_name(42) == "42"


def test_series_name_with_only_whitespace_or_controls_is_empty():
    assert normalize_series_name("  \t\n ") == ""
    assert normalize_series_name("\x00\x01\x1f") == ""


# ------------------------------------------------------------- context key
def test_context_key_is_stable_for_the_same_inputs():
    assert context_key(TEMPLATE, "Frieren") == context_key(TEMPLATE, "Frieren")


def test_context_key_is_the_first_16_hex_chars_of_sha256_of_the_prompt():
    expected = hashlib.sha256(
        resolve_prompt(TEMPLATE, "Frieren").prompt.encode("utf-8")
    ).hexdigest()[:16]

    key = context_key(TEMPLATE, "Frieren")

    assert key == expected
    assert len(key) == 16
    int(key, 16)


def test_context_key_ignores_series_name_whitespace_differences():
    assert context_key(TEMPLATE, "  Frieren ") == context_key(TEMPLATE, "Frieren")


def test_context_key_differs_between_two_series():
    assert context_key(TEMPLATE, "Frieren") != context_key(TEMPLATE, "Dungeon Meshi")


def test_context_key_differs_between_templates():
    other = TEMPLATE + "\nKeep honorifics."

    assert context_key(TEMPLATE, "Frieren") != context_key(other, "Frieren")


def test_context_key_is_empty_for_default_template_and_empty_name():
    assert context_key(DEFAULT_TEMPLATE, "") == ""
    assert context_key(DEFAULT_TEMPLATE, "   ") == ""
    assert context_key(DEFAULT_TEMPLATE, None) == ""


def test_context_key_is_empty_when_an_invalid_template_falls_back_with_no_name():
    assert context_key("", "") == ""
    assert context_key(None, None) == ""
    assert context_key("no placeholder", "") == ""


def test_context_key_is_non_empty_for_default_template_with_a_series_name():
    key = context_key(DEFAULT_TEMPLATE, "Frieren")

    assert key != ""
    assert len(key) == 16


def test_context_key_is_non_empty_for_a_custom_template_without_a_name():
    assert context_key(TEMPLATE, "") != ""


def test_invalid_template_with_a_name_keys_like_the_default_with_that_name():
    assert context_key("", "Frieren") == context_key(DEFAULT_TEMPLATE, "Frieren")


# --------------------------------------------------------- default template
def test_default_template_validates_and_contains_the_placeholder_exactly_once():
    assert validate_template(DEFAULT_TEMPLATE) is None
    assert DEFAULT_TEMPLATE.count(PLACEHOLDER) == 1
    assert len(context_mod._PLACEHOLDER_RE.findall(DEFAULT_TEMPLATE)) == 1


def test_default_template_is_multi_line_and_under_1200_chars():
    assert "\n" in DEFAULT_TEMPLATE
    assert len(DEFAULT_TEMPLATE) < 1200
    assert DEFAULT_TEMPLATE == DEFAULT_TEMPLATE.strip()


def test_default_template_reads_well_with_and_without_a_series_name():
    named = resolve_prompt(DEFAULT_TEMPLATE, "Frieren").prompt
    unnamed = resolve_prompt(DEFAULT_TEMPLATE, "").prompt

    assert "the series Frieren;" in named
    assert "the series;" in unnamed
    assert "[" not in unnamed and "]" not in unnamed


# --------------------------------------------------------------- robustness
def test_resolve_prompt_never_raises_for_none_inputs():
    result = resolve_prompt(None, None)

    assert isinstance(result, ResolvedPrompt)
    assert result.used_default is True
    assert result.warning == "Prompt template invalid (template is empty); using the default"
    assert "None" not in result.prompt
    assert PLACEHOLDER not in result.prompt


def test_resolved_prompt_is_a_frozen_dataclass():
    result = resolve_prompt(TEMPLATE, "Frieren")

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.prompt = "mutated"  # type: ignore[misc]


def test_placeholder_constant_and_regex_agree():
    assert PLACEHOLDER == "[Series Name]"
    assert context_mod._PLACEHOLDER_RE.fullmatch(PLACEHOLDER)


def test_module_exports_the_public_api_and_stays_qt_free():
    expected = {
        "DEFAULT_TEMPLATE",
        "MAX_SERIES_CHARS",
        "MAX_TEMPLATE_CHARS",
        "PLACEHOLDER",
        "ResolvedPrompt",
        "context_key",
        "normalize_series_name",
        "resolve_prompt",
        "validate_template",
    }

    assert set(context_mod.__all__) == expected
    for name in expected:
        assert hasattr(context_mod, name)
    assert "PySide6" not in inspect.getsource(context_mod)
