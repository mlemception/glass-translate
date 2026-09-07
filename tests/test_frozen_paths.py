"""Frozen-mode branches (docs/GLASS_DESIGN.md 5): models dir and logging handlers."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

import pytest

from glasstranslate.config import settings as S
from glasstranslate.ui import app as APP


def test_models_dir_dev_unchanged() -> None:
    assert not getattr(sys, "frozen", False)
    assert S.default_models_dir() == S.project_root() / "models"


def test_models_dir_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    if sys.platform == "win32":
        assert S.user_data_dir() == tmp_path / "GlassTranslate"
        assert S.default_models_dir() == tmp_path / "GlassTranslate" / "models"
    else:
        assert S.default_models_dir() == S.default_config_path().parent / "models"
    assert S.AppConfig().models_dir == str(S.default_models_dir())


def _handlers(logger: logging.Logger):
    stream = [h for h in logger.handlers if type(h) is logging.StreamHandler]
    files = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    return stream, files


def _fresh_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    return logger


def test_logging_dev_uses_stderr_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    logger = _fresh_logger("gt-test-dev")
    added = APP._configure_logging(logger, frozen=False)
    stream, files = _handlers(logger)
    assert len(stream) == 1 and files == [] and len(added) == 1
    assert not (tmp_path / "GlassTranslate" / "logs").exists()
    assert APP._configure_logging(logger, frozen=False) == []  # already configured


def test_logging_frozen_adds_rotating_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    logger = _fresh_logger("gt-test-frozen")
    APP._configure_logging(logger, frozen=True)
    stream, files = _handlers(logger)
    assert len(files) == 1 and len(stream) == 1  # stderr still exists under pytest
    handler = files[0]
    assert handler.maxBytes == 2_000_000 and handler.backupCount == 3
    expected = S.user_data_dir() / "logs" / APP.LOG_FILE_NAME
    assert Path(handler.baseFilename) == expected
    logger.warning("hello")
    handler.flush()
    assert "hello" in expected.read_text(encoding="utf-8")
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)


def test_logging_without_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A --noconsole exe has sys.stderr = None: file handler only, no StreamHandler."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(sys, "stderr", None)
    logger = _fresh_logger("gt-test-nostderr")
    APP._configure_logging(logger, frozen=False)
    stream, files = _handlers(logger)
    assert stream == [] and len(files) == 1
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)


def test_qt_message_handler_routes_to_qt_logger(caplog: pytest.LogCaptureFixture) -> None:
    from PySide6.QtCore import QtMsgType

    class Ctx:
        category = "qml"

    with caplog.at_level(logging.DEBUG, logger="qt"):
        APP._qt_message_handler(QtMsgType.QtWarningMsg, Ctx(), "boom")
    assert any(r.name == "qt" and r.levelno == logging.WARNING and "[qml] boom" in r.getMessage() for r in caplog.records)


def test_version() -> None:
    import glasstranslate

    assert glasstranslate.__version__ == "0.2.0"
