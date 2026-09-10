"""Series-context prompt template engine (feature F5-1).

Pure functions only: no I/O, no UI, no network.  The user-editable template
contains a ``[Series Name]`` placeholder (matched case-insensitively, inner
spaces tolerated) that :func:`resolve_prompt` replaces with the normalised
series name.  An empty name strips the placeholder and tidies the whitespace it
leaves behind; a malformed template falls back to :data:`DEFAULT_TEMPLATE` with
a one-line warning for the status strip.  :func:`context_key` hashes the
resolved prompt so the translation cache cannot leak lines across series.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "DEFAULT_TEMPLATE",
    "MAX_SERIES_CHARS",
    "MAX_TEMPLATE_CHARS",
    "PLACEHOLDER",
    "ResolvedPrompt",
    "context_key",
    "normalize_series_name",
    "resolve_prompt",
    "validate_template",
]

PLACEHOLDER = "[Series Name]"
_PLACEHOLDER_RE = re.compile(r"\[\s*series\s+name\s*\]", re.IGNORECASE)

MAX_SERIES_CHARS = 80
MAX_TEMPLATE_CHARS = 4000

_CONTEXT_KEY_HEX_CHARS = 16
#: Unicode categories dropped from a series name: control (Cc) and format (Cf,
#: e.g. zero-width space, byte-order mark, bidi controls).  Whitespace controls
#: such as tab and newline are kept so they collapse into a single space.
_DROPPED_CATEGORIES = frozenset({"Cc", "Cf"})

_SPACE_RUN_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r" +(?=[;,.:!?])")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_LINE_RUN_RE = re.compile(r"\n{3,}")

DEFAULT_TEMPLATE = (
    "You are an expert translator of manga, light-novel and video-game dialogue.\n"
    "The text comes from the series [Series Name]; use its established character "
    "names, terminology and tone.\n"
    "\n"
    "Guidelines:\n"
    "- Translate naturally and idiomatically, the way a native speaker would "
    "actually say it; avoid stiff, literal phrasing.\n"
    "- Preserve each speaker's intent, emotion, register and personality "
    "(casual, formal, rude, cute, archaic).\n"
    "- Keep honorifics (san, kun, chan, sama, senpai, sensei) where they sound "
    "natural; otherwise adapt them.\n"
    "- Keep names, attack names and in-world terms consistent; do not localise "
    "them unless the series already does.\n"
    "- Render sound effects and interjections with natural equivalents.\n"
    "- The lines were captured by OCR: quietly correct obvious misreads, keep "
    "the same line order, and never merge, split or drop lines.\n"
    "- Output only the translation: no explanations, notes, romaji or "
    "alternative renderings."
)


@dataclass(frozen=True)
class ResolvedPrompt:
    """Outcome of :func:`resolve_prompt`.

    ``warning`` is a one-line status-strip message, set only when the user's
    template was rejected and ``used_default`` is ``True``.
    """

    prompt: str
    warning: Optional[str]
    used_default: bool


def _is_kept(char: str) -> bool:
    return char.isspace() or unicodedata.category(char) not in _DROPPED_CATEGORIES


def normalize_series_name(name: object) -> str:
    """Trim, collapse whitespace runs to one space, drop control/format
    characters and cap at :data:`MAX_SERIES_CHARS` (trailing space after the
    cut removed).  ``None`` becomes ``""``; other non-strings are ``str()``-ed."""
    text = "" if name is None else str(name)
    visible = "".join(char for char in text if _is_kept(char))
    collapsed = " ".join(visible.split())
    return collapsed[:MAX_SERIES_CHARS].rstrip()


def validate_template(template: object) -> Optional[str]:
    """Return ``None`` when *template* is usable, else a short reason."""
    if not isinstance(template, str) or not template.strip():
        return "template is empty"
    if len(template) > MAX_TEMPLATE_CHARS:
        return f"template is longer than {MAX_TEMPLATE_CHARS} characters"
    if _PLACEHOLDER_RE.search(template) is None:
        return f"template has no {PLACEHOLDER} placeholder"
    return None


def _tidy_after_removal(text: str) -> str:
    """Whitespace clean-up once an empty series name removed the placeholder."""
    text = _SPACE_RUN_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub("", text)
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _BLANK_LINE_RUN_RE.sub("\n\n", text)
    return text.strip()


def resolve_prompt(template: object, series_name: object) -> ResolvedPrompt:
    """Substitute every placeholder occurrence with the normalised series name.

    An invalid template falls back to :data:`DEFAULT_TEMPLATE` with a warning.
    An empty name strips the placeholder and collapses the whitespace left
    behind.  Never raises.
    """
    reason = validate_template(template)
    if reason is None:
        source, warning = str(template), None
    else:
        source = DEFAULT_TEMPLATE
        warning = f"Prompt template invalid ({reason}); using the default"
    name = normalize_series_name(series_name)
    if name:
        # A callable replacement keeps backslashes in the name literal.
        prompt = _PLACEHOLDER_RE.sub(lambda _match: name, source)
    else:
        prompt = _tidy_after_removal(_PLACEHOLDER_RE.sub("", source))
    return ResolvedPrompt(prompt=prompt, warning=warning, used_default=reason is not None)


_DEFAULT_EMPTY_PROMPT = resolve_prompt(DEFAULT_TEMPLATE, "").prompt


def context_key(template: object, series_name: object) -> str:
    """Short stable hash of the resolved prompt for the translation-cache key.

    Returns ``""`` when the resolved prompt is the default template with no
    series name, so default configurations keep an empty context.
    """
    prompt = resolve_prompt(template, series_name).prompt
    if prompt == _DEFAULT_EMPTY_PROMPT:
        return ""
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return digest[:_CONTEXT_KEY_HEX_CHARS]
