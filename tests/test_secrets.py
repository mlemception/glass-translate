"""Tests for glasstranslate.config.secrets (tmp files only, no real user data dir)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from glasstranslate.config import secrets as sec
from glasstranslate.config.settings import user_data_dir

FAKE_KEY = "AIzaFAKE-SECRET-1234567890"


@pytest.fixture
def secrets_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the secrets store at a tmp file and clear the Gemini env override."""
    path = tmp_path / "data" / "secrets.json"
    monkeypatch.setenv("GLASSTRANSLATE_SECRETS_FILE", str(path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return path


# ------------------------------------------------------------------ paths
def test_constants():
    assert sec.SECRETS_FILE_NAME == "secrets.json"
    assert sec.GEMINI_API_KEY == "gemini_api_key"
    assert sec.GEMINI_API_KEY_ENV == "GEMINI_API_KEY"


def test_secrets_path_under_user_data_dir_without_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GLASSTRANSLATE_SECRETS_FILE", raising=False)
    assert sec.secrets_path() == user_data_dir() / sec.SECRETS_FILE_NAME
    assert sec.secrets_path().name != "config.json"


def test_secrets_path_honours_env_override(secrets_file: Path):
    assert sec.secrets_path() == secrets_file


# ------------------------------------------------------------------ reads
def test_missing_file_returns_empty_and_does_not_create_it(secrets_file: Path):
    assert sec.get_secret(sec.GEMINI_API_KEY) == ""
    assert sec.has_secret(sec.GEMINI_API_KEY) is False
    assert sec.get_gemini_api_key() == ""
    assert sec.has_gemini_api_key() is False
    assert not secrets_file.exists()
    assert not secrets_file.parent.exists()


def test_env_wins_over_file(secrets_file: Path, monkeypatch: pytest.MonkeyPatch):
    sec.set_secret(sec.GEMINI_API_KEY, "from-file-value")
    monkeypatch.setenv("GEMINI_API_KEY", "  from-env-value  ")
    assert sec.get_secret(sec.GEMINI_API_KEY, env=sec.GEMINI_API_KEY_ENV) == "from-env-value"
    assert sec.get_gemini_api_key() == "from-env-value"
    assert sec.has_gemini_api_key() is True
    # Without the env hint the file value is still what get_secret returns.
    assert sec.get_secret(sec.GEMINI_API_KEY) == "from-file-value"


def test_blank_env_falls_back_to_file(secrets_file: Path, monkeypatch: pytest.MonkeyPatch):
    sec.set_secret(sec.GEMINI_API_KEY, "from-file-value")
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    assert sec.get_gemini_api_key() == "from-file-value"


def test_file_value_is_stripped_on_read(secrets_file: Path):
    secrets_file.parent.mkdir(parents=True)
    secrets_file.write_text(json.dumps({sec.GEMINI_API_KEY: "  padded  "}), encoding="utf-8")
    assert sec.get_gemini_api_key() == "padded"


def test_corrupt_json_returns_empty_without_raising(secrets_file: Path, caplog: pytest.LogCaptureFixture):
    secrets_file.parent.mkdir(parents=True)
    secrets_file.write_text("{ this is not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="glasstranslate.config.secrets"):
        assert sec.get_secret(sec.GEMINI_API_KEY) == ""
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_non_object_json_returns_empty(secrets_file: Path):
    secrets_file.parent.mkdir(parents=True)
    secrets_file.write_text("[1, 2, 3]", encoding="utf-8")
    assert sec.get_secret(sec.GEMINI_API_KEY) == ""


def test_non_string_values_are_ignored(secrets_file: Path):
    secrets_file.parent.mkdir(parents=True)
    secrets_file.write_text(json.dumps({sec.GEMINI_API_KEY: 12345, "other": ["x"]}), encoding="utf-8")
    assert sec.get_secret(sec.GEMINI_API_KEY) == ""
    assert sec.get_secret("other") == ""


def test_unreadable_file_returns_empty_without_raising(
    secrets_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    secrets_file.parent.mkdir(parents=True)
    secrets_file.write_text("{}", encoding="utf-8")

    def boom(self: Path, *args, **kwargs) -> str:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", boom)
    with caplog.at_level(logging.WARNING, logger="glasstranslate.config.secrets"):
        assert sec.get_secret(sec.GEMINI_API_KEY) == ""
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# ----------------------------------------------------------------- writes
def test_set_writes_atomically_outside_config(secrets_file: Path, tmp_path: Path):
    written = sec.set_secret(sec.GEMINI_API_KEY, "  " + FAKE_KEY + "  ")
    assert written == secrets_file
    assert secrets_file.exists()
    assert secrets_file.parent == tmp_path / "data"
    leftovers = sorted(p.name for p in secrets_file.parent.iterdir() if p != secrets_file)
    assert leftovers == []  # no .tmp left behind
    assert not (secrets_file.parent / "config.json").exists()
    data = json.loads(secrets_file.read_text(encoding="utf-8"))
    assert data == {sec.GEMINI_API_KEY: FAKE_KEY}
    assert sec.get_gemini_api_key() == FAKE_KEY


def test_set_preserves_other_keys(secrets_file: Path):
    sec.set_secret("alpha", "1")
    sec.set_secret("beta", "2")
    sec.set_secret("alpha", "3")
    assert json.loads(secrets_file.read_text(encoding="utf-8")) == {"alpha": "3", "beta": "2"}


def test_empty_value_removes_key_and_keeps_others(secrets_file: Path):
    sec.set_secret("other_key", "keep-me")
    sec.set_secret(sec.GEMINI_API_KEY, "remove-me")
    sec.set_secret(sec.GEMINI_API_KEY, "   ")
    assert json.loads(secrets_file.read_text(encoding="utf-8")) == {"other_key": "keep-me"}
    assert sec.get_gemini_api_key() == ""
    assert sec.has_gemini_api_key() is False


def test_removing_from_missing_file_does_not_create_it(secrets_file: Path):
    assert sec.set_secret(sec.GEMINI_API_KEY, "") == secrets_file
    assert not secrets_file.exists()
    assert not secrets_file.parent.exists()


def test_set_rejects_empty_name(secrets_file: Path):
    with pytest.raises(ValueError):
        sec.set_secret("   ", "value")
    assert not secrets_file.exists()


def test_gemini_convenience_wrappers(secrets_file: Path):
    assert sec.set_gemini_api_key(FAKE_KEY) == secrets_file
    assert sec.get_gemini_api_key() == FAKE_KEY
    assert sec.has_gemini_api_key() is True


def test_set_never_logs_the_value(secrets_file: Path, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.DEBUG):
        sec.set_gemini_api_key(FAKE_KEY)
        sec.get_gemini_api_key()
    assert all(FAKE_KEY not in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------- redact
@pytest.mark.parametrize("value", ["x", "short", "abcdefghijk", FAKE_KEY])
def test_redact_never_contains_the_value(value: str):
    out = sec.redact(value)
    assert value not in out
    assert out.startswith("••••")


def test_redact_empty_is_empty():
    assert sec.redact("") == ""
    assert sec.redact("   ") == ""


def test_redact_tail_only_for_long_values():
    mask = "••••"
    assert sec.redact(FAKE_KEY) == mask + "90"
    assert sec.redact("abcdefghijkl") == mask + "kl"  # exactly 12 chars
    assert sec.redact("abcdefghijk") == mask  # 11 chars: no tail
    assert sec.redact("ab") == mask
