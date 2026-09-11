"""The portable bundle's path rule (docs/plans/2026-09-11-portable-bundle.md 2.1).

A ``portable.txt`` next to the running exe moves models, config, logs and secrets under
that folder (``models/``, ``config/``, ``logs/``, secrets in ``config/``); ``models_dir``
is then stored relative to the root so the folder can be moved.  Without the marker every
function returns exactly what it returned before, which is what the "unchanged" tests here
pin down alongside ``tests/test_frozen_paths.py``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from glasstranslate.config import secrets as SEC
from glasstranslate.config import settings as S

MOVED_NAME = "Glass Bündle 2"  # a space and a non-ASCII character, as the acceptance run uses


def _mark(root: Path) -> Path:
    """Create ``root`` with the portable marker in it and return the resolved root."""
    root.mkdir(parents=True, exist_ok=True)
    (root / S.PORTABLE_MARKER).write_text("GlassTranslate portable marker 0.2.0\n", encoding="utf-8")
    return root.resolve()


@pytest.fixture()
def frozen_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A frozen ``GlassTranslate.exe`` in a folder that carries the portable marker."""
    root = _mark(tmp_path / "Glass Bündle")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(root / "GlassTranslate.exe"))
    return root


# ------------------------------------------------------------------ without the marker
def test_marker_constant() -> None:
    assert S.PORTABLE_MARKER == "portable.txt"


def test_dev_checkout_is_not_portable_and_keeps_every_path() -> None:
    assert not getattr(sys, "frozen", False)
    assert S.app_dir() == S.project_root()
    assert S.portable_root() is None
    assert S.is_portable() is False
    assert S.default_models_dir() == S.project_root() / "models"
    assert S.default_config_path().name == "config.json"
    assert S.logs_dir() == S.user_data_dir() / "logs"
    assert S.cache_dir() == S.user_data_dir() / "cache"
    assert S.secrets_dir() == S.user_data_dir()
    assert S.AppConfig().models_dir == str(S.project_root() / "models")


def test_frozen_without_a_marker_keeps_the_user_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-portable frozen behaviour of tests/test_frozen_paths.py, unchanged."""
    app = tmp_path / "app"
    app.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(app / "GlassTranslate.exe"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    assert S.app_dir() == app.resolve()
    assert S.portable_root() is None and S.is_portable() is False
    if sys.platform == "win32":
        assert S.user_data_dir() == tmp_path / "local" / "GlassTranslate"
        assert S.default_models_dir() == tmp_path / "local" / "GlassTranslate" / "models"
        assert S.default_config_path() == tmp_path / "roaming" / "GlassTranslate" / "config.json"
    else:
        assert S.default_models_dir() == S.default_config_path().parent / "models"
    assert S.logs_dir() == S.user_data_dir() / "logs"
    assert S.secrets_dir() == S.user_data_dir()


def test_a_marker_directory_is_not_a_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = tmp_path / "app"
    (app / S.PORTABLE_MARKER).mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(app / "GlassTranslate.exe"))
    assert S.portable_root() is None and S.is_portable() is False


# --------------------------------------------------------------------- with the marker
def test_the_marker_moves_every_path_under_the_root(frozen_root: Path) -> None:
    assert S.app_dir() == frozen_root
    assert S.portable_root() == frozen_root
    assert S.is_portable() is True
    assert S.user_data_dir() == frozen_root
    assert S.default_config_path() == frozen_root / "config" / "config.json"
    assert S.default_models_dir() == frozen_root / "models"
    assert S.logs_dir() == frozen_root / "logs"
    assert S.cache_dir() == frozen_root / "cache"
    assert S.secrets_dir() == frozen_root / "config"
    assert S.AppConfig().models_dir == str(frozen_root / "models")


def test_a_marker_in_a_checkout_makes_the_project_root_portable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not frozen: ``app_dir()`` is the project root, so a marker there works too."""
    root = _mark(tmp_path / "checkout")
    monkeypatch.setattr(S, "project_root", lambda: root)
    assert not getattr(sys, "frozen", False)
    assert S.app_dir() == root and S.portable_root() == root
    assert S.default_models_dir() == root / "models"
    assert S.default_config_path() == root / "config" / "config.json"
    assert S.secrets_dir() == root / "config"


# ------------------------------------------------------------------ resolve_models_dir
def test_resolve_models_dir_fills_in_the_default(frozen_root: Path) -> None:
    assert S.resolve_models_dir("") == str(frozen_root / "models")
    assert S.resolve_models_dir("   ") == str(frozen_root / "models")


def test_resolve_models_dir_anchors_a_relative_path_inside_the_root(frozen_root: Path) -> None:
    assert S.resolve_models_dir("models") == str(frozen_root / "models")
    assert S.resolve_models_dir("store/models") == str(frozen_root / "store" / "models")


def test_resolve_models_dir_rejects_a_relative_escape_from_the_root(frozen_root: Path) -> None:
    """A config travelling inside the bundle must not point the app outside it."""
    for escape in ("../../evil", "..", "models/../../evil", "./../evil"):
        assert S.resolve_models_dir(escape) == str(frozen_root / "models")


def test_resolve_models_dir_drops_a_stale_absolute_path_when_portable(
    frozen_root: Path, tmp_path: Path
) -> None:
    (frozen_root / "models").mkdir()
    stale = tmp_path / "gone" / "models"
    assert S.resolve_models_dir(str(stale)) == str(frozen_root / "models")


def test_resolve_models_dir_keeps_a_stale_absolute_path_outside_portable_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A normal install keeps its configured store even when that drive is not
    plugged in: switching silently would make the next save() lose the setting."""
    root = tmp_path / "checkout"
    (root / "models").mkdir(parents=True)
    monkeypatch.setattr(S, "project_root", lambda: root)
    assert S.portable_root() is None and S.default_models_dir().is_dir()
    stale = tmp_path / "gone" / "models"
    assert S.resolve_models_dir(str(stale)) == str(stale)


def test_resolve_models_dir_keeps_an_absolute_path_that_exists(
    frozen_root: Path, tmp_path: Path
) -> None:
    (frozen_root / "models").mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert S.resolve_models_dir(str(elsewhere)) == str(elsewhere)


def test_resolve_models_dir_keeps_a_stale_path_when_the_default_is_missing_too(
    frozen_root: Path, tmp_path: Path
) -> None:
    """Nothing to fall back to: the value survives so the download can create it."""
    assert not (frozen_root / "models").exists()
    stale = tmp_path / "gone" / "models"
    assert S.resolve_models_dir(str(stale)) == str(stale)


def test_resolve_models_dir_outside_portable_mode_returns_the_value_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the marker the function is the identity for every non-empty value,
    which is exactly what a pre-portable install did."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "local"))
    assert S.portable_root() is None
    assert S.resolve_models_dir("models") == "models"  # never anchored
    assert S.resolve_models_dir("../../evil") == "../../evil"
    assert S.resolve_models_dir(str(tmp_path / "store")) == str(tmp_path / "store")
    assert S.resolve_models_dir("") == str(S.default_models_dir())  # only "" gets the default


# ------------------------------------------------------------------------- config i/o
def test_save_stores_models_dir_relative_to_the_portable_root(frozen_root: Path) -> None:
    path = S.AppConfig().save()
    assert path == frozen_root / "config" / "config.json"
    assert json.loads(path.read_text(encoding="utf-8"))["models_dir"] == "models"


def test_save_stores_an_escaping_absolute_path_absolutely(frozen_root: Path) -> None:
    """``<root>/models/../../x`` starts with the root textually but leaves it: storing
    it relative would let the next load anchor it somewhere else entirely."""
    escaping = frozen_root / "models" / ".." / ".." / "evil"
    path = S.AppConfig(models_dir=str(escaping)).save()
    assert json.loads(path.read_text(encoding="utf-8"))["models_dir"] == str(escaping)


def test_save_stores_a_dotted_path_that_stays_inside_the_root_relative(frozen_root: Path) -> None:
    inside = frozen_root / "store" / ".." / "models"
    path = S.AppConfig(models_dir=str(inside)).save()
    assert json.loads(path.read_text(encoding="utf-8"))["models_dir"] == "models"


def test_save_keeps_a_models_dir_outside_the_root_absolute(
    frozen_root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside" / "models"
    outside.mkdir(parents=True)
    path = S.AppConfig(models_dir=str(outside)).save()
    assert json.loads(path.read_text(encoding="utf-8"))["models_dir"] == str(outside)
    assert S.AppConfig.load().models_dir == str(outside)


def test_save_outside_portable_mode_writes_the_absolute_path(tmp_path: Path) -> None:
    store = tmp_path / "models"
    store.mkdir()
    path = tmp_path / "config.json"
    S.AppConfig(models_dir=str(store)).save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["models_dir"] == str(store)


def test_a_moved_portable_folder_still_finds_its_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Save under root A, move everything to root B: ``models_dir`` follows without an edit."""
    first = _mark(tmp_path / "Glass Bündle")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(first / "GlassTranslate.exe"))
    (first / "models").mkdir()
    saved = S.AppConfig(target_lang="de").save()
    assert json.loads(saved.read_text(encoding="utf-8"))["models_dir"] == "models"

    second = _mark(tmp_path / MOVED_NAME)
    (second / "models").mkdir()
    (second / "config").mkdir()
    (second / "config" / "config.json").write_text(saved.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(second / "GlassTranslate.exe"))

    loaded = S.AppConfig.load()
    assert loaded.target_lang == "de"
    assert loaded.models_dir == str(second / "models")


def test_load_replaces_a_vanished_absolute_models_dir_when_portable(
    frozen_root: Path, tmp_path: Path
) -> None:
    (frozen_root / "models").mkdir()
    path = frozen_root / "config" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"models_dir": str(tmp_path / "gone")}), encoding="utf-8")
    assert S.AppConfig.load().models_dir == str(frozen_root / "models")


def test_from_dict_stays_a_pure_parser(frozen_root: Path) -> None:
    """Only ``load`` resolves: ``from_dict`` returns what the file said."""
    assert S.AppConfig.from_dict({"models_dir": "models"}).models_dir == "models"


# ---------------------------------------------------------------------------- secrets
def test_secrets_live_in_the_config_folder_when_portable(
    frozen_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SEC.SECRETS_FILE_ENV, raising=False)
    assert SEC.secrets_path() == frozen_root / "config" / SEC.SECRETS_FILE_NAME
    assert SEC.secrets_path().parent == S.default_config_path().parent


def test_the_secrets_env_override_still_wins_when_portable(
    frozen_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "elsewhere" / "secrets.json"
    monkeypatch.setenv(SEC.SECRETS_FILE_ENV, str(override))
    assert SEC.secrets_path() == override


def test_secrets_without_the_marker_stay_in_the_user_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SEC.SECRETS_FILE_ENV, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert SEC.secrets_path() == S.user_data_dir() / SEC.SECRETS_FILE_NAME


def test_a_portable_write_lands_in_the_config_folder(
    frozen_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SEC.SECRETS_FILE_ENV, raising=False)
    monkeypatch.delenv(SEC.GEMINI_API_KEY_ENV, raising=False)
    written = SEC.set_gemini_api_key("AIzaFAKE-PORTABLE-000000")
    assert written == frozen_root / "config" / SEC.SECRETS_FILE_NAME
    assert written.is_file() and SEC.get_gemini_api_key() == "AIzaFAKE-PORTABLE-000000"
    assert not (frozen_root / SEC.SECRETS_FILE_NAME).exists()
    assert os.path.dirname(written) == str(frozen_root / "config")


# ------------------------------------------------------- Qt's own caches (ui/app.py)
def test_portable_mode_redirects_the_qml_cache_and_disables_the_pipeline_cache(
    frozen_root: Path,
) -> None:
    """Qt would put both under ``%LOCALAPPDATA%`` whatever the app's paths say."""
    from glasstranslate.ui import app as A

    env: dict = {}
    applied = A.apply_portable_qt_environment(env)
    assert applied == env
    assert env[A.QML_CACHE_ENV] == str(frozen_root / "cache" / "qmlcache")
    assert env[A.RHI_CACHE_ENV] == "1"


def test_an_explicit_qt_cache_setting_is_never_overridden(frozen_root: Path) -> None:
    from glasstranslate.ui import app as A

    env = {A.QML_CACHE_ENV: "D:/mine", A.RHI_CACHE_ENV: "0"}
    assert A.apply_portable_qt_environment(env) == {}
    assert env == {A.QML_CACHE_ENV: "D:/mine", A.RHI_CACHE_ENV: "0"}


def test_without_the_marker_qt_keeps_its_own_caches() -> None:
    from glasstranslate.ui import app as A

    assert S.is_portable() is False
    env: dict = {}
    assert A.apply_portable_qt_environment(env) == {} and env == {}
