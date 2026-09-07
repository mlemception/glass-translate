"""tools/build_resources.py: digest freshness check and the QML import guard (docs/GLASS_DESIGN.md 4)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


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

    for path in (":/qml/Theme.qml", ":/qml/qmldir", ":/qml/shaders/glass.frag.qsb", ":/qml/shaders/blur.frag.qsb",
                 ":/qml/shaders/shadow.frag.qsb", ":/fonts/animeace2_reg.ttf", ":/fonts/animeace2_ital.ttf",
                 ":/icons/app.png"):
        assert QFile.exists(path), path
