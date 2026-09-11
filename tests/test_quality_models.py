"""``glasstranslate.render.quality_models``: the model table, its tolerance of a missing
data file (another slice writes ``quality_models.json``), and the packaging rules that keep
the sidecar out of the exe."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from glasstranslate.render import quality_models as QM

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "GlassTranslate.spec"


def test_missing_table_loads_empty_and_every_function_tolerates_it(tmp_path: Path) -> None:
    files = QM.load_table(tmp_path / "does-not-exist.json")
    assert files == ()
    assert QM.total_bytes(files) == 0
    assert QM.missing_files(tmp_path, files) == []
    assert QM.models_ready(tmp_path, files) is False  # nothing known: never claim "ready"
    assert QM.quality_dir(tmp_path) == tmp_path / QM.QUALITY_SUBDIR
    with pytest.raises(QM.ModelDownloadError):
        QM.ensure_models(tmp_path, files=files)


def _row(name: str, size: int, subdir: str, digest: str = "ab") -> dict:
    return {"name": name, "remote_path": name, "size": size, "sha256": digest * 32,
            "repo": "org/x", "revision": "cafe", "subdir": subdir}


def _write(path: Path, *rows: dict) -> Path:
    path.write_text(json.dumps({"files": list(rows)}), encoding="utf-8")
    return path


def test_a_malformed_table_is_ignored_rather_than_crashing(tmp_path: Path) -> None:
    bad = tmp_path / "quality_models.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert QM.load_table(bad) == ()
    bad.write_text(json.dumps({"files": [{"name": "x"}]}), encoding="utf-8")
    assert QM.load_table(bad) == ()  # incomplete entries are dropped


def test_a_table_is_read_and_reports_missing_files(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "quality_models.json",
        _row("lama.pt", 400_000_000, "lama"),
        _row("config.json", 600, "lama", "cd"),
        _row("config.json", 6_599_999_400, "sdxl/unet", "ef"),  # the same name in another subdir
    )
    files = QM.load_table(path)
    assert [f.subdir for f in files] == ["lama", "lama", "sdxl/unet"]
    assert QM.total_bytes(files) == 7_000_000_000
    assert QM.size_label(files) == "7.0 GB"
    target = QM.quality_dir(tmp_path)
    assert len(QM.missing_files(tmp_path, files)) == 3
    assert QM.models_ready(tmp_path, files) is False
    assert QM.file_path(tmp_path, files[-1]) == target / "sdxl" / "unet" / "config.json"
    # Presence means "at the pinned size" (the sidecar's rule): a truncated file is missing.
    small = QM.load_table(_write(
        tmp_path / "small.json",
        _row("lama.pt", 5, "lama"),
        _row("config.json", 3, "lama", "cd"),
        _row("config.json", 7, "sdxl/unet", "ef"),
    ))
    for spec in small:
        dest = QM.file_path(tmp_path, spec)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * spec.size)
    assert QM.missing_files(tmp_path, small) == []
    assert QM.models_ready(tmp_path, small) is True
    assert QM.ensure_models(tmp_path, files=small) == target  # nothing missing: no download
    QM.file_path(tmp_path, small[0]).write_bytes(b"x")
    assert [f.name for f in QM.missing_files(tmp_path, small)] == ["lama.pt"]
    assert QM.models_ready(tmp_path, small) is False
    # A models dir that already points at the store is taken as is (the sidecar's rule).
    assert QM.quality_dir(target) == target


def test_a_table_entry_can_never_escape_the_models_directory(tmp_path: Path) -> None:
    path = tmp_path / "quality_models.json"
    assert QM.load_table(_write(path, _row("../evil.bin", 1, "lama"))) == ()
    assert QM.load_table(_write(path, _row("evil.bin", 1, "../.."))) == ()
    assert QM.load_table(_write(path, _row("evil.bin", 1, "C:/windows"))) == ()
    assert QM.load_table(_write(path, _row("evil.bin", 1, "/abs"))) == ()
    assert QM.load_table(_write(path, _row("evil.bin", 1, ""))) == ()


def test_the_shipped_table_loads_and_matches_the_sidecar_copy() -> None:
    """PROTOCOL.md: the app and ``glassrenderer/models.py`` share one table."""
    files = QM.load_table()
    assert isinstance(files, tuple)
    for spec in files:
        assert spec.name and spec.subdir and spec.size > 0 and len(spec.sha256) == 64
    if not QM.TABLE_PATH.is_file():
        pytest.skip("the model table is not delivered yet")
    assert files, "the delivered table must not be empty"
    sidecar = ROOT / "renderer" / "glassrenderer" / "models.json"
    if not sidecar.is_file():
        pytest.skip("the sidecar package is not delivered yet")
    app_rows = json.loads(QM.TABLE_PATH.read_text(encoding="utf-8"))
    side_rows = json.loads(sidecar.read_text(encoding="utf-8"))
    assert app_rows == side_rows, "the two model tables have drifted apart"


# --------------------------------------------------------------------------------- packaging
def test_spec_hides_the_quality_modules_and_ships_the_table() -> None:
    text = SPEC.read_text(encoding="utf-8")
    assert "glasstranslate.render.quality" in text
    assert "glasstranslate.render.quality_models" in text
    assert "quality_models.json" in text


def test_spec_never_collects_the_sidecar() -> None:
    text = SPEC.read_text(encoding="utf-8")
    assert "glassrenderer" in text and "renderer/" in text
