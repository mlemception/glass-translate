"""tools/build_resources.py: digest freshness check and the QML import guard (docs/GLASS_DESIGN.md 4)."""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from tools import build_resources as B

ROOT = Path(__file__).resolve().parents[1]


def _replica(tmp_path: Path) -> Path:
    """A minimal copy of the project tree with the real qrc inputs."""
    root = tmp_path / "proj"
    shutil.copytree(ROOT / "glasstranslate" / "ui" / "qml", root / "glasstranslate" / "ui" / "qml")
    shutil.copytree(ROOT / "glasstranslate" / "render" / "fonts" / "animeace",
                    root / "glasstranslate" / "render" / "fonts" / "animeace")
    (root / "glasstranslate" / "ui" / "icons").mkdir(parents=True)
    shutil.copy2(ROOT / "glasstranslate" / "ui" / "icons" / "app.png", root / "glasstranslate" / "ui" / "icons" / "app.png")
    return root


def test_real_tree_is_fresh(resources_built: Path) -> None:
    assert resources_built == B.rc_path(ROOT) and resources_built.is_file()
    ok, msg = B.check_digest(ROOT)
    assert ok, msg
    header = resources_built.read_text(encoding="utf-8").splitlines()[0]
    assert header == B.DIGEST_PREFIX + B.compute_digest(ROOT)


def test_qrc_lists_every_input(resources_built: Path) -> None:
    entries = B.qrc_entries(ROOT)
    aliases = {alias for alias, _ in entries["/qml"]}
    assert "Theme.qml" in aliases and "qmldir" in aliases
    assert {"shaders/glass.frag.qsb", "shaders/blur.frag.qsb", "shaders/shadow.frag.qsb"} <= aliases
    assert {alias for alias, _ in entries["/fonts"]} == {"animeace2_reg.ttf", "animeace2_ital.ttf"}
    assert [alias for alias, _ in entries["/icons"]] == ["app.png"]
    text = B.qrc_path(ROOT).read_text(encoding="utf-8")
    assert 'prefix="/qml"' in text and 'prefix="/fonts"' in text and 'prefix="/icons"' in text


def test_check_detects_stale_and_missing(tmp_path: Path) -> None:
    root = _replica(tmp_path)
    assert B.check_digest(root)[0] is False  # no resources_rc.py yet
    out = B.build(root)
    assert out.is_file() and B.read_digest(out) == B.compute_digest(root)
    assert B.check_digest(root) == (True, "digest ok")
    theme = root / "glasstranslate" / "ui" / "qml" / "Theme.qml"
    theme.write_text(theme.read_text(encoding="utf-8") + "\n// touched\n", encoding="utf-8")
    ok, msg = B.check_digest(root)
    assert not ok and "stale" in msg
    B.build(root)
    assert B.check_digest(root)[0]
    # An edited shader source (not part of the qrc) also invalidates the digest.
    frag = root / "glasstranslate" / "ui" / "qml" / "shaders" / "glass.frag"
    frag.write_text(frag.read_text(encoding="utf-8") + "\n// touched\n", encoding="utf-8")
    assert not B.check_digest(root)[0]


def test_import_guard(tmp_path: Path) -> None:
    root = _replica(tmp_path)
    assert B.check_imports(root) == []
    bad = root / "glasstranslate" / "ui" / "qml" / "Bad.qml"
    bad.write_text("import QtQuick\nimport QtQuick.Controls\nItem { Button {} }\n", encoding="utf-8")
    offenders = B.check_imports(root)
    assert offenders and any(o.startswith("QtQuick.Controls ->") for o in offenders)
    B.build(root)
    assert B.check(root) == 2  # digest fine, guard fails
    bad.unlink()
    B.build(root)
    assert B.check(root) == 0


def test_cli_check_exit_codes(tmp_path: Path, resources_built: Path) -> None:
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "build_resources.py"), "--check"],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "digest ok" in proc.stdout and "qml imports ok" in proc.stdout
    root = _replica(tmp_path)
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "build_resources.py"), "--check", "--root", str(root)],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=240)
    assert proc.returncode == 1 and "missing" in proc.stdout


def test_resources_load_in_qt(qapp) -> None:
    from PySide6.QtCore import QFile

    # Registering the resources is an import-time side effect of resources_rc;
    # in a full-suite run glasstranslate.ui.control happens to import it first,
    # but a standalone run of this file must register them itself.
    importlib.import_module("glasstranslate.ui.resources_rc")

    for path in (":/qml/Theme.qml", ":/qml/qmldir", ":/qml/shaders/glass.frag.qsb", ":/qml/shaders/blur.frag.qsb",
                 ":/qml/shaders/shadow.frag.qsb", ":/fonts/animeace2_reg.ttf", ":/fonts/animeace2_ital.ttf",
                 ":/icons/app.png"):
        assert QFile.exists(path), path


# ---------------------------------------------------------------- _ensure_resources self-heal
def test_ensure_resources_rebuilds_when_digest_stale(resources_built: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A present-but-stale ``resources_rc`` (already imported) must be rebuilt and
    reloaded, unregistering the old Qt resource data before the reload."""
    from glasstranslate.ui import control as C
    from glasstranslate.ui import resources_rc as RC

    build_calls: List[None] = []
    events: List[str] = []

    def fake_build() -> Path:
        build_calls.append(None)
        return B.rc_path()

    def fake_cleanup() -> None:
        events.append("cleanup")

    def fake_reload(module: object) -> None:
        events.append("reload")

    monkeypatch.setattr(B, "check_digest", lambda: (False, "stale"))
    monkeypatch.setattr(B, "build", fake_build)
    monkeypatch.setattr(RC, "qCleanupResources", fake_cleanup)
    monkeypatch.setattr(importlib, "reload", fake_reload)

    assert C._ensure_resources() is True
    assert len(build_calls) == 1
    assert events == ["cleanup", "reload"]


def test_ensure_resources_skips_rebuild_when_digest_fresh(resources_built: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh digest must not trigger a rebuild."""
    from glasstranslate.ui import control as C

    build_calls: List[None] = []

    def fake_build() -> Path:
        build_calls.append(None)
        return B.rc_path()

    monkeypatch.setattr(B, "check_digest", lambda: (True, "digest ok"))
    monkeypatch.setattr(B, "build", fake_build)

    assert C._ensure_resources() is True
    assert build_calls == []
