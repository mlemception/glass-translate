"""Tests for glasstranslate.translate.gemini (fake ``requests.Session``; no network)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import requests

from glasstranslate.core.interfaces import Translator
from glasstranslate.translate import gemini
from glasstranslate.translate.gemini import GeminiTranslator

KEY = "AIzaFAKE-SECRET-1234567890"
JA_HELLO = "こんにちは"  # konnichiwa
JA_THANKS = "ありがとう"  # arigatou


# ---------------------------------------------------------------- fakes
@dataclass
class FakeResponse:
    status_code: int
    payload: Any = None
    headers: Dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    """Replays canned responses (or raises canned exceptions) and records calls."""

    def __init__(self, responses: Sequence[Any]) -> None:
        self._responses = list(responses)
        self.calls: List[SimpleNamespace] = []
        self.closed = False

    def post(self, url: str, *, headers: Optional[dict] = None, json: Any = None, timeout: Any = None) -> Any:
        self.calls.append(SimpleNamespace(url=url, headers=headers or {}, json=json, timeout=timeout))
        if not self._responses:
            raise AssertionError("unexpected extra request to %s" % url)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


def native_ok(text: str, finish: Optional[str] = "STOP") -> FakeResponse:
    candidate: Dict[str, Any] = {"content": {"role": "model", "parts": [{"text": text}]}}
    if finish is not None:
        candidate["finishReason"] = finish
    return FakeResponse(200, {"candidates": [candidate]})


def api_error(status: int, api_status: str, message: str = "boom", headers: Optional[dict] = None) -> FakeResponse:
    body = {"error": {"code": status, "message": message, "status": api_status}}
    return FakeResponse(status, body, headers=headers or {})


def make(responses: Sequence[Any], **kwargs: Any) -> Tuple[GeminiTranslator, FakeSession, List[float]]:
    session = FakeSession(responses)
    sleeps: List[float] = []
    kwargs.setdefault("api_key", KEY)
    tr = GeminiTranslator(session=session, sleep=sleeps.append, rng=lambda: 0.0, **kwargs)
    return tr, session, sleeps


def user_text(call: SimpleNamespace) -> str:
    return call.json["contents"][0]["parts"][0]["text"]


# ------------------------------------------------------------ constants
def test_module_constants():
    assert gemini.DEFAULT_MODEL == "gemini-3.8-flash"
    assert gemini.DEFAULT_MODEL in gemini.MODEL_PRESETS
    assert gemini.MODEL_PRESETS == (
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    )
    assert gemini.DEFAULT_BASE_URL == "https://generativelanguage.googleapis.com"
    assert gemini.API_FORMATS == ("native", "openai")
    assert gemini.DEFAULT_TIMEOUT_S == 30.0
    assert gemini.CONNECT_TIMEOUT_S == 5.0
    assert gemini.DEFAULT_MAX_RETRIES == 3
    assert gemini.RETRY_STATUSES == frozenset({408, 429, 500, 502, 503, 504})
    assert gemini.MAX_RETRY_AFTER_S == 30.0
    assert gemini.MAX_OUTPUT_TOKENS == 2048
    assert gemini.THINKING_LEVELS == ("off", "minimal", "low", "medium", "high")


def test_class_attributes_and_defaults():
    tr, _, _ = make([])
    assert isinstance(tr, Translator)
    assert tr.name == "gemini"
    assert tr.device == "remote"
    assert tr.offline is False
    assert tr.model == gemini.DEFAULT_MODEL
    assert tr.base_url == gemini.DEFAULT_BASE_URL
    assert tr.api_format == "native"
    assert tr.last_error is None
    assert list(tr.supported_pairs()) == []
    assert tr.supports("ja", "en") is True
    assert tr.supports("xx", "yy") is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_format": "grpc"},
        {"thinking_level": "max"},
        {"base_url": "generativelanguage.googleapis.com"},
        {"base_url": "ftp://example.com"},
        {"base_url": ""},
    ],
)
def test_constructor_validation(kwargs: Dict[str, Any]):
    with pytest.raises(ValueError):
        GeminiTranslator(KEY, session=FakeSession([]), **kwargs)


def test_repr_does_not_leak_key():
    tr, _, _ = make([])
    text = repr(tr)
    assert KEY not in text
    assert "gemini-3.8-flash" in text


# ------------------------------------------------------- happy path
def test_numbered_prompt_round_trip():
    tr, session, _ = make([native_ok("1. Hello\n2. Thank you")])
    out = tr.translate_batch([JA_HELLO, JA_THANKS], "ja", "en")
    assert out == ["Hello", "Thank you"]
    assert tr.last_error is None
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call.url == gemini.DEFAULT_BASE_URL + "/v1beta/models/gemini-3.8-flash:generateContent"
    assert call.timeout == (gemini.CONNECT_TIMEOUT_S, gemini.DEFAULT_TIMEOUT_S)
    body = call.json
    assert body["system_instruction"]["parts"][0]["text"].strip()
    assert body["contents"][0]["role"] == "user"
    assert "1. " + JA_HELLO in user_text(call)
    assert "2. " + JA_THANKS in user_text(call)
    assert body["generationConfig"]["maxOutputTokens"] == gemini.MAX_OUTPUT_TOKENS
    categories = {s["category"] for s in body["safetySettings"]}
    assert categories == {
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    }
    assert all(s["threshold"] == "BLOCK_NONE" for s in body["safetySettings"])


def test_api_key_sent_as_header_not_in_url():
    tr, session, _ = make([native_ok("1. Hi")])
    tr.translate_batch([JA_HELLO], "ja", "en")
    call = session.calls[0]
    assert "key=" not in call.url
    assert KEY not in call.url
    assert call.headers["x-goog-api-key"] == KEY
    assert call.headers["Content-Type"] == "application/json"


def test_model_id_is_url_quoted_into_one_path_segment():
    """The Model field is free text since 2026-09-10: a slash, space or query char must not escape
    the models/<id>:generateContent path segment."""
    tr, session, _ = make([native_ok("1. Hi")], model="team/model x?k=v")
    tr.translate_batch([JA_HELLO], "ja", "en")
    url = session.calls[0].url
    assert url.endswith("/v1beta/models/team%2Fmodel%20x%3Fk%3Dv:generateContent")
    assert "/team/" not in url and "?" not in url


def test_response_with_fewer_lines_keeps_source_for_missing():
    tr, _, _ = make([native_ok("1. A\n3. C")])
    out = tr.translate_batch(["a", "b", "c"], "ja", "en")
    assert out == ["A", "b", "C"]
    assert tr.last_error is None


def test_parser_accepts_paren_numbering_and_ignores_noise():
    tr, _, _ = make([native_ok("Sure, here you go:\n```\n 1) One\n2.   Two\n```\n")])
    assert tr.translate_batch(["x", "y"], "ja", "en") == ["One", "Two"]


def test_multi_part_reply_is_joined_and_thoughts_skipped():
    payload = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"text": "thinking...", "thought": True},
                        {"text": "1. Hel"},
                        {"text": "lo\n2. World"},
                    ]
                },
            }
        ]
    }
    tr, _, _ = make([FakeResponse(200, payload)])
    assert tr.translate_batch(["a", "b"], "ja", "en") == ["Hello", "World"]


def test_blank_texts_are_preserved_and_not_sent():
    tr, session, _ = make([native_ok("1. Hello")])
    out = tr.translate_batch(["", "   ", "hi"], "ja", "en")
    assert out == ["", "   ", "Hello"]
    assert user_text(session.calls[0]) == "1. hi"


def test_newlines_inside_a_text_become_spaces():
    tr, session, _ = make([native_ok("1. one two")])
    tr.translate_batch(["one\r\ntwo"], "ja", "en")
    assert user_text(session.calls[0]) == "1. one two"


def test_empty_texts_list_short_circuits():
    tr, session, _ = make([])
    assert tr.translate_batch([], "ja", "en") == []
    assert tr.translate_batch(["", " "], "ja", "en") == ["", " "]
    assert session.calls == []


def test_temperature_is_optional():
    tr, session, _ = make([native_ok("1. a")])
    tr.translate_batch(["x"], "ja", "en")
    assert "temperature" not in session.calls[0].json["generationConfig"]
    tr2, session2, _ = make([native_ok("1. a")], temperature=0.2)
    tr2.translate_batch(["x"], "ja", "en")
    assert session2.calls[0].json["generationConfig"]["temperature"] == 0.2


# ------------------------------------------------------- thinking config
def test_thinking_config_25_model_off_sends_budget_zero():
    tr, session, _ = make([native_ok("1. a")], model="gemini-2.5-flash", thinking_level="off")
    tr.translate_batch(["x"], "ja", "en")
    cfg = session.calls[0].json["generationConfig"]["thinkingConfig"]
    assert cfg == {"thinkingBudget": 0}
    assert "thinkingLevel" not in cfg


def test_thinking_config_25_model_low_omits_thinking():
    tr, session, _ = make([native_ok("1. a")], model="gemini-2.5-flash-lite", thinking_level="low")
    tr.translate_batch(["x"], "ja", "en")
    assert "thinkingConfig" not in session.calls[0].json["generationConfig"]


def test_thinking_config_3x_model_uses_level():
    tr, session, _ = make([native_ok("1. a")], model="gemini-3.8-flash", thinking_level="low")
    tr.translate_batch(["x"], "ja", "en")
    cfg = session.calls[0].json["generationConfig"]["thinkingConfig"]
    assert cfg == {"thinkingLevel": "low"}
    assert "thinkingBudget" not in cfg


def test_thinking_config_3x_model_off_maps_to_minimal():
    tr, session, _ = make([native_ok("1. a")], model="gemini-3.1-flash-lite", thinking_level="off")
    tr.translate_batch(["x"], "ja", "en")
    assert session.calls[0].json["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}


# --------------------------------------------------------- failures
def test_missing_api_key_makes_no_request():
    tr, session, _ = make([], api_key="   ")
    out = tr.translate_batch(["a", "b"], "ja", "en")
    assert out == ["a", "b"]
    assert session.calls == []
    assert tr.last_error == "Gemini API key missing (Engines → Gemini)"


def test_blocked_prompt_returns_inputs_and_mentions_reason():
    tr, _, _ = make([FakeResponse(200, {"promptFeedback": {"blockReason": "SAFETY"}})])
    out = tr.translate_batch(["a", "b"], "ja", "en")
    assert out == ["a", "b"]
    assert tr.last_error is not None
    assert "SAFETY" in tr.last_error


def test_unexpected_finish_reason_is_a_failure():
    tr, _, _ = make([native_ok("1. A", finish="RECITATION")])
    assert tr.translate_batch(["a"], "ja", "en") == ["a"]
    assert tr.last_error is not None and "RECITATION" in tr.last_error


def test_max_tokens_finish_still_uses_partial_output():
    tr, _, _ = make([native_ok("1. A", finish="MAX_TOKENS")])
    assert tr.translate_batch(["a", "b"], "ja", "en") == ["A", "b"]
    assert tr.last_error is None


def test_429_with_retry_after_sleeps_then_succeeds():
    tr, session, sleeps = make(
        [api_error(429, "RESOURCE_EXHAUSTED", headers={"Retry-After": "2"}), native_ok("1. A")]
    )
    assert tr.translate_batch(["a"], "ja", "en") == ["A"]
    assert sleeps == [2.0]
    assert len(session.calls) == 2
    assert tr.last_error is None


def test_retry_after_is_clamped():
    tr, _, sleeps = make([api_error(503, "UNAVAILABLE", headers={"Retry-After": "9999"}), native_ok("1. A")])
    tr.translate_batch(["a"], "ja", "en")
    assert sleeps == [gemini.MAX_RETRY_AFTER_S]


def test_backoff_without_retry_after_is_exponential():
    tr, _, sleeps = make([api_error(500, "INTERNAL"), api_error(502, "BAD_GATEWAY"), native_ok("1. A")])
    tr.translate_batch(["a"], "ja", "en")
    assert sleeps == [1.0, 2.0]


def test_400_is_not_retried():
    tr, session, sleeps = make([api_error(400, "INVALID_ARGUMENT", message="bad request body")])
    out = tr.translate_batch(["a", "b"], "ja", "en")
    assert out == ["a", "b"]
    assert len(session.calls) == 1
    assert sleeps == []
    assert tr.last_error is not None
    assert "400" in tr.last_error
    assert "INVALID_ARGUMENT" in tr.last_error
    assert "bad request body" not in tr.last_error


@pytest.mark.parametrize("status", [401, 403, 404])
def test_other_4xx_not_retried(status: int):
    tr, session, _ = make([api_error(status, "PERMISSION_DENIED")])
    assert tr.translate_batch(["a"], "ja", "en") == ["a"]
    assert len(session.calls) == 1
    assert str(status) in (tr.last_error or "")


def test_exhausts_max_retries_on_503_then_returns_inputs():
    tr, session, sleeps = make([api_error(503, "UNAVAILABLE")] * 3, max_retries=2)
    assert tr.translate_batch(["a"], "ja", "en") == ["a"]
    assert len(session.calls) == 3  # initial attempt + 2 retries
    assert len(sleeps) == 2
    assert tr.last_error is not None and "503" in tr.last_error


def test_timeout_is_retried_then_reported():
    tr, session, sleeps = make([requests.Timeout(), requests.ReadTimeout()], max_retries=1)
    assert tr.translate_batch(["a"], "ja", "en") == ["a"]
    assert len(session.calls) == 2
    assert len(sleeps) == 1
    assert tr.last_error == "Gemini request timed out"


def test_connection_error_retried_then_succeeds():
    tr, session, _ = make([requests.ConnectionError(), native_ok("1. A")])
    assert tr.translate_batch(["a"], "ja", "en") == ["A"]
    assert len(session.calls) == 2


def test_unreadable_json_body_is_a_failure():
    tr, _, _ = make([FakeResponse(200, ValueError("not json"))])
    assert tr.translate_batch(["a"], "ja", "en") == ["a"]
    assert tr.last_error is not None


def test_success_clears_last_error():
    tr, _, _ = make([api_error(400, "INVALID_ARGUMENT"), native_ok("1. A")])
    tr.translate_batch(["a"], "ja", "en")
    assert tr.last_error is not None
    tr.translate_batch(["a"], "ja", "en")
    assert tr.last_error is None


def test_cancel_before_call_makes_no_request():
    tr, session, _ = make([native_ok("1. A")])
    tr.close()
    assert tr.translate_batch(["a"], "ja", "en") == ["a"]
    assert session.calls == []
    assert session.closed is True
    assert tr.last_error == "Gemini request cancelled"


def test_cancel_during_backoff_aborts_without_sleeping():
    session = FakeSession([api_error(503, "UNAVAILABLE"), native_ok("1. A")])
    sleeps: List[float] = []
    tr = GeminiTranslator(KEY, session=session, sleep=sleeps.append, rng=lambda: 0.0)

    original_post = session.post

    def post_then_cancel(*args: Any, **kwargs: Any) -> Any:
        result = original_post(*args, **kwargs)
        tr.close()
        return result

    session.post = post_then_cancel  # type: ignore[method-assign]
    assert tr.translate_batch(["a"], "ja", "en") == ["a"]
    assert sleeps == []
    assert tr.last_error == "Gemini request cancelled"


def test_key_never_appears_in_logs_or_last_error(caplog: pytest.LogCaptureFixture):
    echo = "denied for key " + KEY
    tr, _, _ = make(
        [
            api_error(400, "INVALID_ARGUMENT", message=echo),
            requests.Timeout(echo),
            requests.Timeout(echo),
            FakeResponse(200, {"promptFeedback": {"blockReason": KEY}}),
        ],
        max_retries=1,
    )
    with caplog.at_level(logging.DEBUG):
        tr.translate_batch(["a"], "ja", "en")
        assert KEY not in (tr.last_error or "")
        tr.translate_batch(["a"], "ja", "en")
        assert KEY not in (tr.last_error or "")
        tr.translate_batch(["a"], "ja", "en")
        assert KEY not in (tr.last_error or "")
    for record in caplog.records:
        assert KEY not in record.getMessage()
        assert KEY not in str(record.args)


# ------------------------------------------------------------ base url
def test_custom_base_url_used_verbatim_with_trailing_slash_stripped():
    tr, session, _ = make([native_ok("1. A")], base_url="  https://proxy.example.com/gemini/  ")
    assert tr.base_url == "https://proxy.example.com/gemini"
    tr.translate_batch(["a"], "ja", "en")
    assert session.calls[0].url == "https://proxy.example.com/gemini/v1beta/models/gemini-3.8-flash:generateContent"


# --------------------------------------------------------- openai format
def test_openai_format_endpoint_headers_and_parsing():
    payload = {"choices": [{"message": {"role": "assistant", "content": "1. Hello\n2. Bye"}, "finish_reason": "stop"}]}
    tr, session, _ = make([FakeResponse(200, payload)], api_format="openai", thinking_level="low")
    out = tr.translate_batch([JA_HELLO, JA_THANKS], "ja", "en")
    assert out == ["Hello", "Bye"]
    call = session.calls[0]
    assert call.url == gemini.DEFAULT_BASE_URL + "/v1beta/openai/chat/completions"
    assert call.headers["Authorization"] == "Bearer " + KEY
    assert "x-goog-api-key" not in call.headers
    assert "key=" not in call.url
    body = call.json
    assert body["model"] == gemini.DEFAULT_MODEL
    assert body["max_tokens"] == gemini.MAX_OUTPUT_TOKENS
    assert body["reasoning_effort"] == "low"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "1. " + JA_HELLO in body["messages"][1]["content"]
    assert "2. " + JA_THANKS in body["messages"][1]["content"]


def test_openai_format_omits_reasoning_effort_for_off_and_minimal():
    for level in ("off", "minimal"):
        tr, session, _ = make(
            [FakeResponse(200, {"choices": [{"message": {"content": "1. A"}}]})], api_format="openai", thinking_level=level
        )
        tr.translate_batch(["a"], "ja", "en")
        assert "reasoning_effort" not in session.calls[0].json


def test_openai_format_content_filter_is_a_failure():
    payload = {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]}
    tr, _, _ = make([FakeResponse(200, payload)], api_format="openai")
    assert tr.translate_batch(["a"], "ja", "en") == ["a"]
    assert tr.last_error is not None


# -------------------------------------------------------------- context
def test_set_context_changes_context_key_and_prompt():
    tr, session, _ = make([native_ok("1. A"), native_ok("1. A")])
    assert tr.context_key() == ""
    tr.set_context("You are translating a fantasy RPG; keep item names in English.")
    key1 = tr.context_key()
    assert len(key1) == 16 and int(key1, 16) >= 0
    tr.set_context("Different context")
    key2 = tr.context_key()
    assert key2 != key1 and len(key2) == 16
    tr.translate_batch(["a"], "ja", "en")
    system = session.calls[0].json["system_instruction"]["parts"][0]["text"]
    assert "Different context" in system
    assert "1." in system  # built-in numbering instructions always appended
    tr.set_context("   ")
    assert tr.context_key() == ""
    tr.translate_batch(["a"], "ja", "en")
    system = session.calls[1].json["system_instruction"]["parts"][0]["text"]
    assert "Different context" not in system
    assert system.strip()


def test_system_prompt_mentions_languages():
    tr, session, _ = make([native_ok("1. A")])
    tr.translate_batch(["a"], "ja", "en")
    system = session.calls[0].json["system_instruction"]["parts"][0]["text"]
    assert "Japanese" in system and "English" in system
