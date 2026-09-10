"""Online backend for Google Gemini over plain ``requests`` (no SDK).

One numbered prompt per batch - ``1. <text>\\n2. <text>`` - goes out with a
system instruction; the model answers with the same numbering and every index
that comes back replaces the matching input.  Anything that cannot be
translated (a missing line, an HTTP error, a safety block, a timeout,
cancellation) degrades to the source text so the overlay keeps working, and
:attr:`GeminiTranslator.last_error` carries a redacted, human-readable reason
for the UI.

Security: the API key travels in a header, never in the URL; it is never
logged (nor are request bodies or headers) and never part of ``last_error``.
Free-form server text is filtered down to enum-style tokens before it can
reach a log line.
"""
from __future__ import annotations

import hashlib
import logging
import random
import re
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests

from ..config.secrets import redact
from ..core.interfaces import Translator

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"
MODEL_PRESETS = (
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
API_FORMATS = ("native", "openai")
DEFAULT_TIMEOUT_S = 30.0
CONNECT_TIMEOUT_S = 5.0
DEFAULT_MAX_RETRIES = 3
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
MAX_RETRY_AFTER_S = 30.0
MAX_OUTPUT_TOKENS = 2048
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high")

ERR_NO_KEY = "Gemini API key missing (Engines → Gemini)"
ERR_CANCELLED = "Gemini request cancelled"
ERR_TIMEOUT = "Gemini request timed out"
ERR_CONNECTION = "Gemini connection failed"
ERR_REQUEST = "Gemini request failed"
ERR_BAD_RESPONSE = "Gemini returned an unreadable response"
ERR_EMPTY = "Gemini returned no text"

_BACKOFF_BASE_S = 1.0
_BACKOFF_CAP_S = 30.0
_BACKOFF_JITTER_S = 0.5
_CONTEXT_KEY_CHARS = 16
_OK_FINISH_REASONS = frozenset({"STOP", "MAX_TOKENS"})
_LEGACY_THINKING_PREFIX = "gemini-2.5"  # thinkingBudget models; 3.x use thinkingLevel
_NO_THINKING_LEVELS = frozenset({"off", "minimal"})
_REASONING_EFFORT = {"low": "low", "medium": "medium", "high": "high"}
_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)
_HTTP_HINTS = {
    400: "bad request",
    401: "invalid API key",
    403: "permission denied",
    404: "model not found",
    408: "request timeout",
    429: "rate limited",
    500: "server error",
    502: "bad gateway",
    503: "service unavailable",
    504: "gateway timeout",
}
_LANGUAGE_NAMES = {
    "auto": "the source language",
    "en": "English", "de": "German", "es": "Spanish", "fr": "French", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "ru": "Russian", "uk": "Ukrainian",
    "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "ar": "Arabic", "tr": "Turkish",
    "hi": "Hindi", "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "sv": "Swedish",
}
_NUMBERED_LINE = re.compile(r"^\s*(\d+)[.)]\s*(.*)$")
_SAFE_TOKEN = re.compile(r"^[A-Z][A-Z0-9_]{0,39}$")

_INSTRUCTIONS = (
    "You translate on-screen text from {src} to {tgt} for a live screen translator.\n"
    "The user message is a numbered list with one item per line, formatted `1. text`.\n"
    "Reply with the same numbered list (`1. translation`), one line per input, same order, "
    "same numbers; never merge, split, skip or add items.\n"
    "Translate every item into natural {tgt}. Keep names, honorifics and terminology sensible "
    "for the target audience; leave untranslatable strings (numbers, codes) as they are.\n"
    "Output only the numbered translations: no commentary, no notes, no markdown, no code fences."
)


class _GeminiError(Exception):
    """Internal failure carrying an already-redacted message and retry hints."""

    def __init__(self, message: str, *, retryable: bool = False, retry_after_s: Optional[float] = None) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.retry_after_s = retry_after_s


# ------------------------------------------------------------ pure helpers
def _normalize_base_url(url: str) -> str:
    cleaned = (url or "").strip().rstrip("/")
    if not cleaned.startswith(("http://", "https://")):
        raise ValueError("base_url must start with http:// or https://")
    return cleaned


def _language_name(code: str) -> str:
    key = (code or "").strip().lower()
    return _LANGUAGE_NAMES.get(key, key or _LANGUAGE_NAMES["auto"])


def _safe_token(value: Any) -> str:
    """Keep ``value`` only when it looks like an API enum (``RESOURCE_EXHAUSTED``),
    so free-form server text never reaches a log line or the UI."""
    return value if isinstance(value, str) and _SAFE_TOKEN.match(value) else ""


def _one_line(text: str) -> str:
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def _build_prompt(texts: Sequence[str]) -> str:
    return "\n".join("%d. %s" % (number, _one_line(text)) for number, text in enumerate(texts, start=1))


def _parse_numbered(reply: str) -> Dict[int, str]:
    """``{index: translation}`` for every ``N. text`` / ``N) text`` line; the
    first occurrence of an index wins and everything else is ignored."""
    found: Dict[int, str] = {}
    for line in reply.splitlines():
        match = _NUMBERED_LINE.match(line)
        if match is None:
            continue
        index = int(match.group(1))
        if index not in found:
            found[index] = match.group(2).strip()
    return found


def _merge(texts: Sequence[str], indices: Sequence[int], parsed: Mapping[int, str]) -> List[str]:
    """Put parsed translations back at ``indices``; missing or empty ones keep
    the source text."""
    numbers = {position: number for number, position in enumerate(indices, start=1)}
    return [(parsed.get(numbers[i]) or text) if i in numbers else text for i, text in enumerate(texts)]


def _thinking_config(model: str, level: str) -> Dict[str, Any]:
    """Native ``thinkingConfig``: 2.5 models only know ``thinkingBudget`` (0 to
    disable, omitted otherwise); newer models take ``thinkingLevel``.  Never both."""
    if model.startswith(_LEGACY_THINKING_PREFIX):
        return {"thinkingBudget": 0} if level in _NO_THINKING_LEVELS else {}
    return {"thinkingLevel": "minimal" if level == "off" else level}


def _payload(response: Any) -> Any:
    try:
        return response.json()
    except (ValueError, TypeError, AttributeError, requests.RequestException):
        return None


def _api_status(payload: Any) -> str:
    error = payload.get("error") if isinstance(payload, dict) else None
    return _safe_token(error.get("status")) if isinstance(error, dict) else ""


def _http_error(status: int, payload: Any) -> str:
    details = [hint for hint in (_HTTP_HINTS.get(status), _api_status(payload)) if hint]
    suffix = " (%s)" % "; ".join(details) if details else ""
    return "Gemini HTTP %d%s" % (status, suffix)


def _retry_after(response: Any) -> Optional[float]:
    headers = getattr(response, "headers", None)
    if not hasattr(headers, "get"):
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        raw = headers.get("retry-after")
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None  # absent or an HTTP-date: fall back to exponential backoff
    return max(0.0, min(MAX_RETRY_AFTER_S, seconds))


def _part_text(part: Any) -> str:
    if not isinstance(part, dict) or part.get("thought"):
        return ""
    text = part.get("text")
    return text if isinstance(text, str) else ""


def _extract_native_text(payload: Dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        feedback = payload.get("promptFeedback")
        reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
        raise _GeminiError("Gemini blocked the prompt: %s" % (_safe_token(reason) or "UNKNOWN"))
    first = candidates[0] if isinstance(candidates[0], dict) else {}
    finish = first.get("finishReason")
    if finish is not None and finish not in _OK_FINISH_REASONS:
        raise _GeminiError("Gemini stopped generating: %s" % (_safe_token(finish) or "UNKNOWN"))
    content = first.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    text = "".join(_part_text(part) for part in parts) if isinstance(parts, list) else ""
    if not text.strip():
        raise _GeminiError(ERR_EMPTY)
    return text


def _extract_openai_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    if first.get("finish_reason") == "content_filter":
        raise _GeminiError("Gemini blocked the prompt: CONTENT_FILTER")
    message = first.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise _GeminiError(ERR_EMPTY)
    return content


# ---------------------------------------------------------------- backend
class GeminiTranslator(Translator):
    """Gemini client speaking either the native ``generateContent`` API or the
    OpenAI-compatible ``chat/completions`` endpoint.

    ``sleep`` / ``rng`` / ``session`` are injectable for tests.  Thread-safe:
    ``set_context`` and ``translate_batch`` may run on different threads.
    """

    name = "gemini"
    device = "remote"
    offline = False

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_format: str = "native",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        thinking_level: str = "low",
        temperature: Optional[float] = None,
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        if api_format not in API_FORMATS:
            raise ValueError("api_format must be one of %s, got %r" % (API_FORMATS, api_format))
        if thinking_level not in THINKING_LEVELS:
            raise ValueError("thinking_level must be one of %s, got %r" % (THINKING_LEVELS, thinking_level))
        self._api_key = (api_key or "").strip()
        self._model = (model or "").strip() or DEFAULT_MODEL
        self._base_url = _normalize_base_url(base_url)
        self._api_format = api_format
        self._timeout_s = float(timeout_s)
        self._max_retries = max(0, int(max_retries))
        self._thinking_level = thinking_level
        self._temperature = temperature
        self._session = session if session is not None else requests.Session()
        self._sleep = sleep
        self._rng = rng
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._system_prompt = ""
        self.last_error: Optional[str] = None

    def __repr__(self) -> str:
        return "GeminiTranslator(model=%r, api_format=%r, base_url=%r, key=%s)" % (
            self._model, self._api_format, self._base_url, redact(self._api_key) or "<none>")

    # ------------------------------------------------------------ properties
    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def api_format(self) -> str:
        return self._api_format

    @property
    def thinking_level(self) -> str:
        return self._thinking_level

    # --------------------------------------------------------------- context
    def set_context(self, system_prompt: str) -> None:
        """Replace the user-supplied system prompt (series notes, glossary...).
        The built-in numbering instructions are always appended at call time."""
        with self._lock:
            self._system_prompt = (system_prompt or "").strip()

    def context_key(self) -> str:
        """Short stable digest of the current system prompt for cache keys;
        ``""`` when no prompt is set."""
        with self._lock:
            prompt = self._system_prompt
        if not prompt:
            return ""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:_CONTEXT_KEY_CHARS]

    def _system_text(self, src: str, tgt: str) -> str:
        instructions = _INSTRUCTIONS.format(src=_language_name(src), tgt=_language_name(tgt))
        with self._lock:
            custom = self._system_prompt
        return custom + "\n\n" + instructions if custom else instructions

    # -------------------------------------------------------- Translator API
    def supported_pairs(self) -> Iterable[tuple[str, str]]:
        return ()

    def supports(self, src: str, tgt: str) -> bool:
        return True  # an LLM handles any pair; failures degrade per call

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str) -> List[str]:
        texts = list(texts)
        indices = [i for i, text in enumerate(texts) if text.strip()]
        if not indices:
            return texts
        if not self._api_key:
            self.last_error = ERR_NO_KEY
            log.warning("%s; returning inputs", ERR_NO_KEY)
            return texts
        prompt = _build_prompt([texts[i] for i in indices])
        system = self._system_text(src, tgt)
        try:
            reply = self._request_with_retry(system, prompt)
        except _GeminiError as exc:
            self.last_error = exc.message
            log.warning("%s; returning inputs", exc.message)
            return texts
        self.last_error = None
        return _merge(texts, indices, _parse_numbered(reply))

    def close(self) -> None:
        """Abort in-flight retries and release the HTTP session."""
        self._cancel.set()
        try:
            self._session.close()
        except Exception:  # pragma: no cover - best effort on shutdown
            log.debug("Gemini session close failed", exc_info=True)

    # ------------------------------------------------------------- transport
    def _request_with_retry(self, system: str, prompt: str) -> str:
        url, headers, body = self._build_request(system, prompt)
        for attempt in range(self._max_retries + 1):
            if self._cancel.is_set():
                raise _GeminiError(ERR_CANCELLED)
            try:
                return self._request_once(url, headers, body)
            except _GeminiError as exc:
                if not exc.retryable or attempt >= self._max_retries:
                    raise
                if self._cancel.is_set():
                    raise _GeminiError(ERR_CANCELLED) from None
                delay = exc.retry_after_s if exc.retry_after_s is not None else self._backoff(attempt)
                log.warning("%s; retrying in %.1fs (%d/%d)", exc.message, delay, attempt + 1, self._max_retries)
                self._sleep(delay)
        raise _GeminiError(ERR_REQUEST)  # pragma: no cover - loop always returns or raises

    def _backoff(self, attempt: int) -> float:
        return min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * 2 ** attempt) + self._rng() * _BACKOFF_JITTER_S

    def _request_once(self, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> str:
        try:
            response = self._session.post(url, headers=headers, json=body, timeout=(CONNECT_TIMEOUT_S, self._timeout_s))
        except requests.Timeout as exc:
            log.debug("Gemini transport error: %s", type(exc).__name__)
            raise _GeminiError(ERR_TIMEOUT, retryable=True) from None
        except requests.ConnectionError as exc:
            log.debug("Gemini transport error: %s", type(exc).__name__)
            raise _GeminiError(ERR_CONNECTION, retryable=True) from None
        except requests.RequestException as exc:
            log.debug("Gemini transport error: %s", type(exc).__name__)
            raise _GeminiError(ERR_REQUEST) from None
        return self._handle_response(response)

    def _handle_response(self, response: Any) -> str:
        try:
            status = int(getattr(response, "status_code", 0) or 0)
        except (TypeError, ValueError):
            status = 0
        payload = _payload(response)
        if status in RETRY_STATUSES:
            raise _GeminiError(_http_error(status, payload), retryable=True, retry_after_s=_retry_after(response))
        if not 200 <= status < 300:
            raise _GeminiError(_http_error(status, payload))
        if not isinstance(payload, dict):
            raise _GeminiError(ERR_BAD_RESPONSE)
        if self._api_format == "openai":
            return _extract_openai_text(payload)
        return _extract_native_text(payload)

    # -------------------------------------------------------- request bodies
    def _build_request(self, system: str, prompt: str) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        if self._api_format == "openai":
            return self._openai_request(system, prompt)
        return self._native_request(system, prompt)

    def _native_request(self, system: str, prompt: str) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        url = "%s/v1beta/models/%s:generateContent" % (self._base_url, self._model)
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        thinking = _thinking_config(self._model, self._thinking_level)
        generation = {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            **({"temperature": self._temperature} if self._temperature is not None else {}),
            **({"thinkingConfig": thinking} if thinking else {}),
        }
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation,
            "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in _SAFETY_CATEGORIES],
        }
        return url, headers, body

    def _openai_request(self, system: str, prompt: str) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        url = self._base_url + "/v1beta/openai/chat/completions"
        headers = {"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json"}
        effort = _REASONING_EFFORT.get(self._thinking_level)
        body = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_tokens": MAX_OUTPUT_TOKENS,
            **({"temperature": self._temperature} if self._temperature is not None else {}),
            **({"reasoning_effort": effort} if effort else {}),
        }
        return url, headers, body
