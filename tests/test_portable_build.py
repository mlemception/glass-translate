"""Tests for the portable-bundle build helpers (plan section 3, slice 6).

``packaging/`` is not an importable package (the name collides with the PyPI
``packaging`` distribution), so the three modules under test are imported off
``sys.path`` the way ``tests/test_smoke_portable.py`` imports the smoke scripts.

Nothing here touches the real 9.3 GB store: :func:`verify_quality` and friends
are exercised against a tmp store of tiny fake files with a hand-written table.
The one test that reads the real store is opt-in (``GT_STORE_TESTS=1``), and the
one that reads ``dist/GlassTranslate.exe`` skips when the exe has not been built.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packaging"))

import licenses  # noqa: E402
import portable_store  # noqa: E402
import verify_torch_free  # noqa: E402

import build_portable  # noqa: E402 - the drivers live at the repo root, already on sys.path
import build_renderer  # noqa: E402

APP_EXE = ROOT / "dist" / "GlassTranslate.exe"


# ------------------------------------------------------------------ verify_torch_free
def test_forbidden_entries_matches_module_names_and_submodules():
    names = ["torch", "torch.nn.functional", "transformers.models.bert", "glassrenderer.stages.lama"]
    assert verify_torch_free.forbidden_entries(names) == sorted(names)


def test_forbidden_entries_ignores_lookalike_module_names():
    names = [
        "glasstranslate.ocr.mangaocr",  # the app's own engine, not the manga_ocr package
        "torchgen",  # not torch / torch.*
        "safetensors_stub",
        "transformersuite",
    ]
    assert verify_torch_free.forbidden_entries(names) == []


def test_forbidden_entries_matches_binary_paths_and_basenames():
    names = [
        "torch\\lib\\torch_cpu.dll",
        "torch\\lib\\c10.dll",
        "cudnn64_9.dll",
        "cublas64_12.dll",
        "nvrtc64_120_0.dll",
        "safetensors\\_safetensors_rust.cp313-win_amd64.pyd",
    ]
    assert verify_torch_free.forbidden_entries(names) == sorted(names)


def test_forbidden_entries_matches_the_cuda_binary_families():
    names = [
        "cudart64_13.dll", "cufft64_12.dll", "cusparse64_12.dll", "cusolver64_12.dll",
        "curand64_10.dll", "nvJitLink_130_0.dll", "cupti64_2025.3.0.dll",
        "nvToolsExt64_1.dll", "libtorch.so", "c10.so", "nvidia.cublas.lib",
    ]
    assert verify_torch_free.forbidden_entries(names) == sorted(names)


def test_forbidden_entries_is_case_insensitive():
    names = ["Torch.nn", "TORCH\\lib\\Torch_cpu.dll", "NVIDIA.cudnn", "CuDNN64_9.DLL"]
    assert verify_torch_free.forbidden_entries(names) == sorted(names)
    assert verify_torch_free.forbidden_entries(["TorchGen.model"]) == []


def test_forbidden_entries_ignores_unrelated_binaries():
    names = [
        "python313.dll",
        "PySide6\\Qt6Core.dll",
        "PIL\\_imaging.cp313-win_amd64.pyd",
        "onnxruntime\\capi\\onnxruntime_providers_shared.dll",
        "VCRUNTIME140.dll",
    ]
    assert verify_torch_free.forbidden_entries(names) == []


def test_forbidden_entries_honours_a_custom_forbidden_tuple():
    assert verify_torch_free.forbidden_entries(["numpy.linalg"], forbidden=("numpy",)) == ["numpy.linalg"]
    assert verify_torch_free.forbidden_entries(["numpy.linalg"], forbidden=("scipy",)) == []


def test_frozen_top_level_modules_dedupes_sorts_and_drops_data_entries():
    names = [
        "glasstranslate.ui.app",
        "glasstranslate",
        "PIL\\_imaging.cp313-win_amd64.pyd",
        "PYZ.pyz",
        "python313.dll",
        "pyi-contents-directory _internal",
        "struct",
    ]
    assert verify_torch_free.frozen_top_level_modules(names) == ["PIL", "glasstranslate", "struct"]


def test_a_scan_that_found_nothing_fails_closed(tmp_path: Path):
    """No embedded PYZ and no _internal means nothing was scanned - never report 'clean'."""
    fake_exe = tmp_path / "nothing.exe"
    fake_exe.write_bytes(b"not a PyInstaller build")
    with pytest.raises(verify_torch_free.ArchiveScanError, match="nothing could be scanned"):
        verify_torch_free._onedir_entries(fake_exe, has_embedded_pyz=False)
    assert verify_torch_free._onedir_entries(fake_exe, has_embedded_pyz=True) == []


@pytest.mark.skipif(not APP_EXE.exists(), reason="dist/GlassTranslate.exe has not been built")
def test_the_built_app_exe_is_torch_free():
    names = verify_torch_free.frozen_toc(APP_EXE)
    assert len(names) > 500, "the archive reader returned an implausibly small TOC"
    assert verify_torch_free.forbidden_entries(names) == []
    assert verify_torch_free.verify_torch_free(APP_EXE) == []


# ------------------------------------------------------------------ portable_store fixtures
def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


_FAKE_QUALITY = (
    ("lama", "anime_manga_lama.pt", b"lama-weights"),
    ("lama", "LICENSE", b"MIT License\n"),
    ("sdxl", "Illustrious-XL-v1.0.safetensors", b"sdxl-weights"),
)


def _fake_quality_table() -> List[Any]:
    """Three rows of the quality table shape, tiny enough to hash in a test."""
    from glasstranslate.render.quality_models import QualityModel

    return [
        QualityModel(
            name=name,
            remote_path=f"{subdir}/{name}",
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            repo="example/repo",
            revision="0" * 40,
            subdir=subdir,
        )
        for subdir, name, payload in _FAKE_QUALITY
    ]


@pytest.fixture()
def fake_store(tmp_path: Path):
    """A models dir holding the three fake quality files and a fake manga-ocr pair."""
    from glasstranslate.ocr.models import ModelFile

    table = _fake_quality_table()
    for subdir, name, payload in _FAKE_QUALITY:
        _write(tmp_path / "quality" / subdir / name, payload)
    ocr_payloads = {"vocab.txt": b"vocab", "config.json": b"{}"}
    ocr_files = tuple(
        ModelFile(name, name, len(payload), hashlib.sha256(payload).hexdigest())
        for name, payload in ocr_payloads.items()
    )
    for spec in ocr_files:
        _write(tmp_path / "manga-ocr" / spec.name, ocr_payloads[spec.name])
    return tmp_path, table, ocr_files


# ------------------------------------------------------------------ portable_store: verification
def test_verify_quality_returns_one_row_per_table_entry(fake_store):
    models_dir, table, _ = fake_store
    rows = portable_store.verify_quality(models_dir, files=table)
    assert [r.path for r in rows] == [
        "quality/lama/LICENSE",
        "quality/lama/anime_manga_lama.pt",
        "quality/sdxl/Illustrious-XL-v1.0.safetensors",
    ]
    assert {r.kind for r in rows} == {"quality"}
    assert all(r.pair is None for r in rows)
    assert all(len(r.sha256) == 64 for r in rows)


def test_verify_quality_raises_naming_a_missing_file(fake_store):
    models_dir, table, _ = fake_store
    (models_dir / "quality" / "lama" / "anime_manga_lama.pt").unlink()
    with pytest.raises(portable_store.StoreError, match="anime_manga_lama.pt"):
        portable_store.verify_quality(models_dir, files=table)


def test_verify_quality_raises_on_a_size_mismatch(fake_store):
    models_dir, table, _ = fake_store
    (models_dir / "quality" / "lama" / "LICENSE").write_bytes(b"MIT License\nplus junk\n")
    with pytest.raises(portable_store.StoreError, match="size"):
        portable_store.verify_quality(models_dir, files=table)


def test_verify_quality_raises_on_a_digest_mismatch(fake_store):
    models_dir, table, _ = fake_store
    target = models_dir / "quality" / "lama" / "LICENSE"
    target.write_bytes(b"X" * target.stat().st_size)
    with pytest.raises(portable_store.StoreError, match="sha256"):
        portable_store.verify_quality(models_dir, files=table)


def test_verify_quality_raises_on_an_empty_table(tmp_path: Path):
    with pytest.raises(portable_store.StoreError, match="table"):
        portable_store.verify_quality(tmp_path, files=[])


def test_verify_manga_ocr_returns_rows_and_checks_digests(fake_store):
    models_dir, _, ocr_files = fake_store
    rows = portable_store.verify_manga_ocr(models_dir, files=ocr_files)
    assert [r.path for r in rows] == ["manga-ocr/config.json", "manga-ocr/vocab.txt"]
    assert {r.kind for r in rows} == {"manga-ocr"}
    (models_dir / "manga-ocr" / "vocab.txt").write_bytes(b"vocab-but-different")
    with pytest.raises(portable_store.StoreError, match="vocab.txt"):
        portable_store.verify_manga_ocr(models_dir, files=ocr_files)


# ------------------------------------------------------------------ portable_store: argos discovery
def _argos_dir(models_dir: Path, name: str, meta: Dict[str, Any]) -> Path:
    pack = models_dir / name
    (pack / "model").mkdir(parents=True, exist_ok=True)
    (pack / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (pack / "model" / "model.bin").write_bytes(b"ct2")
    (pack / "sentencepiece.model").write_bytes(b"spm")
    return pack


def _argos_archive(models_dir: Path, name: str, meta: Dict[str, Any]) -> Path:
    import zipfile

    archive = models_dir / name
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("pack/metadata.json", json.dumps(meta))
        zf.writestr("pack/sentencepiece.model", "spm")
    return archive


def test_discover_argos_reads_directories_and_archives(tmp_path: Path):
    _argos_dir(tmp_path, "translate-en_de-1_3", {"from_code": "en", "to_code": "de"})
    _argos_archive(tmp_path, "translate-fr_en-1_0.argosmodel", {"from_code": "fr", "to_code": "en"})
    packs = portable_store.discover_argos(tmp_path)
    assert sorted(p.pair for p in packs) == [("en", "de"), ("fr", "en")]
    by_pair = {p.pair: p for p in packs}
    assert by_pair[("en", "de")].directory is not None and by_pair[("en", "de")].archive is None
    assert by_pair[("fr", "en")].archive is not None and by_pair[("fr", "en")].directory is None


def test_discover_argos_skips_an_archive_whose_pair_has_a_directory(tmp_path: Path):
    _argos_dir(tmp_path, "ja_en", {"from_code": "ja", "to_code": "en"})
    _argos_archive(tmp_path, "translate-ja_en-1_1.argosmodel", {"from_code": "ja", "to_code": "en"})
    packs = portable_store.discover_argos(tmp_path)
    assert [p.pair for p in packs] == [("ja", "en")]
    assert packs[0].archive is None
    assert packs[0].directory is not None and packs[0].directory.name == "ja_en"


def _sugoi_dir(models_dir: Path, name: str = "sugoi-v4-ja-en") -> Path:
    pack = models_dir / name
    (pack / "spm").mkdir(parents=True)
    (pack / "model.bin").write_bytes(b"ct2")
    (pack / "spm" / "spm.ja.nopretok.model").write_bytes(b"spm")
    (pack / "spm" / "spm.en.nopretok.model").write_bytes(b"spm")
    return pack


def test_discover_argos_recognises_a_sugoi_directory(tmp_path: Path):
    pack = _sugoi_dir(tmp_path)
    (pack / "sugoi-v4-ja-en" / "spm").mkdir(parents=True)  # the stray empty nested dir
    packs = portable_store.discover_argos(tmp_path)
    assert [(p.pair, p.priority, p.kind) for p in packs] == [(("ja", "en"), 10, "sugoi")]


def test_the_sugoi_pack_is_marked_not_redistributable(tmp_path: Path):
    """NTT licences it for research use only, so the bundle must not ship it."""
    _sugoi_dir(tmp_path)
    _argos_dir(tmp_path, "translate-en_de-1_3", {"from_code": "en", "to_code": "de"})
    by_kind = {pack.kind: pack for pack in portable_store.discover_argos(tmp_path)}
    assert by_kind["sugoi"].redistributable is False
    assert by_kind["sugoi"].licence_note == portable_store.SUGOI_LICENCE
    assert by_kind["argos"].redistributable is True
    assert by_kind["argos"].licence_note == portable_store.ARGOS_LICENCE


def test_a_renamed_sugoi_directory_is_still_recognised(tmp_path: Path):
    """Renaming the folder must not smuggle the research pack into the zip."""
    _sugoi_dir(tmp_path, "ja-en-fast")
    packs = portable_store.discover_argos(tmp_path)
    assert [(p.kind, p.redistributable) for p in packs] == [("sugoi", False)]


def test_discover_argos_keeps_every_directory_serving_a_pair(tmp_path: Path):
    """``ja_en/`` still ships even though Sugoi (priority 10) wins ja->en at runtime."""
    _argos_dir(tmp_path, "ja_en", {"from_code": "ja", "to_code": "en"})
    _argos_dir(tmp_path, "sugoi-v4-ja-en", {"from_code": "ja", "to_code": "en", "priority": 10})
    packs = portable_store.discover_argos(tmp_path)
    assert [(p.directory.name, p.priority, p.kind) for p in packs] == [
        ("ja_en", 0, "argos"),
        ("sugoi-v4-ja-en", 10, "sugoi"),
    ]


def test_discover_argos_ignores_an_empty_directory(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    assert portable_store.discover_argos(tmp_path) == []


# ------------------------------------------------------------------ portable_store: pair coverage
def test_offered_pairs_are_every_language_against_english():
    pairs = portable_store.offered_pairs()
    assert ("ja", "en") in pairs and ("en", "ja") in pairs
    assert ("en", "en") not in pairs
    assert all("en" in pair for pair in pairs)
    assert len(pairs) == len(set(pairs)) and len(pairs) % 2 == 0


def test_uncovered_pairs_honours_the_english_pivot(tmp_path: Path):
    _argos_dir(tmp_path, "ja_en", {"from_code": "ja", "to_code": "en"})
    _argos_dir(tmp_path, "translate-en_de-1_3", {"from_code": "en", "to_code": "de"})
    packs = portable_store.discover_argos(tmp_path)
    uncovered = portable_store.uncovered_pairs(packs)
    assert ("ja", "en") not in uncovered
    assert ("en", "de") not in uncovered
    assert ("en", "ja") in uncovered  # no pack in that direction
    assert ("de", "en") in uncovered
    assert ("es", "en") in uncovered


# ------------------------------------------------------------------ portable_store: manifest + copy
def test_store_manifest_lists_every_shipped_file(fake_store):
    models_dir, table, ocr_files = fake_store
    _argos_dir(models_dir, "ja_en", {"from_code": "ja", "to_code": "en"})
    manifest = portable_store.store_manifest(
        models_dir, "2026-09-11", quality_files=table, manga_ocr_files=ocr_files
    )
    assert manifest["version"] == 1
    assert manifest["generated"] == "2026-09-11"
    paths = [row["path"] for row in manifest["files"]]
    assert paths == sorted(paths)
    assert "quality/lama/LICENSE" in paths
    assert "manga-ocr/vocab.txt" in paths
    assert "ja_en/metadata.json" in paths
    assert manifest["total_bytes"] == sum(row["size"] for row in manifest["files"])
    assert manifest["argos_packs"] == [
        {
            "pair": "ja->en", "priority": 0, "kind": "argos",
            "directory": "ja_en", "archive": None, "readme": None,
            "licence": portable_store.ARGOS_LICENCE,
            "source": portable_store.ARGOS_INDEX_URL,
            "redistributable": True,
        }
    ]
    assert "en->ja" in manifest["uncovered_pairs"]
    assert {row["kind"] for row in manifest["files"]} == {"quality", "manga-ocr", "argos"}


def test_pack_rows_and_entries_record_the_upstream_licence(tmp_path: Path):
    """Recorded verbatim from the upstream pages, never inferred (see the module note)."""
    _argos_dir(tmp_path, "ja_en", {"from_code": "ja", "to_code": "en"})
    (tmp_path / "ja_en" / "README.md").write_text("pack readme", encoding="utf-8")
    sugoi = tmp_path / "sugoi-v4-ja-en"
    (sugoi / "spm").mkdir(parents=True)
    (sugoi / "model.bin").write_bytes(b"ct2")
    (sugoi / "spm" / "spm.ja.nopretok.model").write_bytes(b"spm")
    (sugoi / "spm" / "spm.en.nopretok.model").write_bytes(b"spm")
    packs = portable_store.discover_argos(tmp_path)
    entries = [portable_store._pack_entry(pack, tmp_path) for pack in packs]
    by_kind = {entry["kind"]: entry for entry in entries}
    assert by_kind["argos"]["licence"].startswith("unknown - see https://github.com/argosopentech")
    assert by_kind["argos"]["readme"] == "ja_en/README.md"
    assert by_kind["sugoi"]["licence"].startswith("ntt-license")
    assert "no commercial use" in by_kind["sugoi"]["licence"]
    assert by_kind["sugoi"]["source"] == portable_store.SUGOI_MODEL_URL
    assert by_kind["sugoi"]["readme"] is None
    rows = portable_store._pack_files(tmp_path, packs[0])
    assert {row.licence for row in rows} == {portable_store.ARGOS_LICENCE}


def _store_with_sugoi(fake_store):
    """The fake store plus an Argos pack and the research-licensed Sugoi pack."""
    models_dir, table, ocr_files = fake_store
    _argos_dir(models_dir, "ja_en", {"from_code": "ja", "to_code": "en"})
    _sugoi_dir(models_dir)
    return models_dir, table, ocr_files


def test_store_manifest_ships_the_research_pack_by_default(fake_store):
    """The bundle is for personal use on the build machine, so Sugoi is included."""
    models_dir, table, ocr_files = _store_with_sugoi(fake_store)
    manifest = portable_store.store_manifest(
        models_dir, "2026-09-11", quality_files=table, manga_ocr_files=ocr_files
    )
    assert manifest["research_models_included"] is True
    assert manifest["excluded_packs"] == []
    assert any(row["path"].startswith("sugoi-v4-ja-en/") for row in manifest["files"])
    assert {pack["kind"] for pack in manifest["argos_packs"]} == {"argos", "sugoi"}
    sugoi = next(pack for pack in manifest["argos_packs"] if pack["kind"] == "sugoi")
    assert sugoi["redistributable"] is False
    assert sugoi["licence"] == portable_store.SUGOI_LICENCE
    assert "not for redistribution" in sugoi["licence"]
    assert sugoi["source"] == portable_store.SUGOI_MODEL_URL
    assert {row["licence"] for row in manifest["files"] if row["kind"] == "sugoi"} == {
        portable_store.SUGOI_LICENCE
    }


def test_store_manifest_excludes_the_research_pack_for_a_shareable_bundle(fake_store):
    models_dir, table, ocr_files = _store_with_sugoi(fake_store)
    manifest = portable_store.store_manifest(
        models_dir, "2026-09-11", quality_files=table, manga_ocr_files=ocr_files,
        include_research_models=False,
    )
    assert manifest["research_models_included"] is False
    assert not any("sugoi" in row["path"] for row in manifest["files"])
    assert {row["kind"] for row in manifest["files"]} == {"quality", "manga-ocr", "argos"}
    assert [pack["kind"] for pack in manifest["argos_packs"]] == ["argos"]
    assert manifest["excluded_packs"] == [{
        "directory": "sugoi-v4-ja-en", "pair": "ja->en", "priority": 10,
        "reason": portable_store.SUGOI_EXCLUSION_REASON,
        "source": portable_store.SUGOI_MODEL_URL,
    }]
    # ja->en is still served, by the Argos pack that stays.
    assert "ja->en" not in manifest["uncovered_pairs"]


def test_copy_store_follows_the_manifest_for_the_research_pack(fake_store, tmp_path: Path):
    """The zip file list is built from the same rows, so it cannot diverge from this."""
    models_dir, table, ocr_files = _store_with_sugoi(fake_store)
    for include, expect_sugoi in ((True, True), (False, False)):
        manifest = portable_store.store_manifest(
            models_dir, "2026-09-11", quality_files=table, manga_ocr_files=ocr_files,
            include_research_models=include,
        )
        dest = tmp_path / f"staged-{include}"
        portable_store.copy_store(models_dir, dest, manifest)
        copied = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
        assert copied == [row["path"] for row in manifest["files"]]
        assert any("sugoi" in path for path in copied) is expect_sugoi


def test_an_archive_survives_when_its_only_directory_is_left_out(tmp_path: Path):
    """Dropping Sugoi must not silently drop ja->en with it."""
    _sugoi_dir(tmp_path)
    _argos_archive(tmp_path, "translate-ja_en-1_1.argosmodel", {"from_code": "ja", "to_code": "en"})
    shipped_all = portable_store.discover_argos(tmp_path)
    assert [p.kind for p in shipped_all] == ["sugoi"]  # the directory wins, archive dropped
    without_research = portable_store.discover_argos(tmp_path, include_research_models=False)
    assert sorted((p.kind, p.archive is not None) for p in without_research) == [
        ("argos", True), ("sugoi", False),
    ]


@pytest.mark.parametrize("bad", ["/etc/passwd", "C:/windows/system32/x.dll", "../escape.bin",
                                 "quality/../../escape.bin", "\\\\server\\share\\x", ""])
def test_copy_store_refuses_a_manifest_path_that_escapes_the_destination(tmp_path: Path, bad: str):
    manifest = {"files": [{"path": bad, "size": 1, "sha256": "", "kind": "quality"}]}
    with pytest.raises(portable_store.StoreError, match="unsafe manifest path"):
        portable_store.copy_store(tmp_path, tmp_path / "dest", manifest)


def test_copy_store_copies_exactly_the_manifest_files(fake_store, tmp_path: Path):
    models_dir, table, ocr_files = fake_store
    _write(models_dir / "quality" / "lama" / "anime_manga_lama.pt.part", b"partial")
    _write(models_dir / "__pycache__" / "stray.pyc", b"junk")
    manifest = portable_store.store_manifest(
        models_dir, "2026-09-11", quality_files=table, manga_ocr_files=ocr_files
    )
    dest = tmp_path / "staged"
    copied = portable_store.copy_store(models_dir, dest, manifest)
    assert copied == len(manifest["files"])
    on_disk = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert on_disk == [row["path"] for row in manifest["files"]]


@pytest.mark.skipif(os.environ.get("GT_STORE_TESTS") != "1", reason="set GT_STORE_TESTS=1 to hash the real store")
def test_the_real_quality_store_verifies():
    rows = portable_store.verify_quality(ROOT / "models")
    assert len(rows) == 24


# ------------------------------------------------------------------ licenses
def _dist(name: str, **kw: Any) -> Dict[str, Any]:
    dist: Dict[str, Any] = {
        "name": name,
        "version": "1.0",
        "license_expression": "",
        "license_field": "",
        "classifiers": [],
        "license_files": [],
        "modules": [name.replace("-", "_")],
    }
    dist.update(kw)
    return dist


def test_classify_prefers_the_license_expression():
    dist = _dist("acme", license_expression="Apache-2.0", license_field="MIT",
                 classifiers=["License :: OSI Approved :: MIT License"])
    assert licenses.classify(dist) == "Apache-2.0"


def test_classify_falls_back_to_trove_classifiers():
    dist = _dist("acme", classifiers=["License :: OSI Approved :: MIT License", "Topic :: Utilities"])
    assert licenses.classify(dist) == "MIT"


def test_classify_joins_several_classifiers_with_or():
    dist = _dist("pyphen", classifiers=[
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "License :: OSI Approved :: Mozilla Public License 1.1 (MPL 1.1)",
    ])
    assert licenses.classify(dist) == "GPL-2.0-or-later OR LGPL-2.1-or-later OR MPL-1.1"


def test_classify_uses_a_short_license_field_when_there_are_no_classifiers():
    assert licenses.classify(_dist("acme", license_field="BSD 3-Clause")) == "BSD 3-Clause"


def test_classify_ignores_a_license_field_that_is_the_whole_licence_text():
    long_text = "Copyright (c) 2026 " + "the quick brown fox " * 40
    assert len(long_text) > licenses.MAX_LICENCE_FIELD_CHARS
    dist = _dist("scipy", license_field=long_text,
                 classifiers=["License :: OSI Approved :: BSD License"])
    assert licenses.classify(dist) == "BSD"


def test_classify_reports_unknown_when_nothing_is_declared():
    assert licenses.classify(_dist("acme")) == "UNKNOWN"


def test_gpl_violations_flags_a_gpl_only_distribution():
    dist = _dist("gpltool", license_expression="GPL-3.0-only")
    assert licenses.gpl_violations([dist]) == ["gpltool (GPL-3.0-only)"]


def test_gpl_violations_flags_agpl():
    dist = _dist("agpltool", license_expression="AGPL-3.0-or-later")
    assert licenses.gpl_violations([dist]) == ["agpltool (AGPL-3.0-or-later)"]


def test_gpl_violations_accepts_lgpl():
    assert licenses.gpl_violations([_dist("pynput", license_field="LGPLv3")]) == []


def test_gpl_violations_accepts_a_tri_licensed_distribution():
    dist = _dist("pyphen", classifiers=[
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "License :: OSI Approved :: Mozilla Public License 1.1 (MPL 1.1)",
    ])
    assert licenses.gpl_violations([dist]) == []
    assert "MPL" in licenses.ALLOWLIST["pyphen"]


def test_gpl_violations_ignores_a_long_license_field_that_merely_quotes_gpl():
    long_text = "This binary bundles libgfortran under the GNU General Public License " * 5
    dist = _dist("scipy", license_field=long_text,
                 classifiers=["License :: OSI Approved :: BSD License"])
    assert licenses.gpl_violations([dist]) == []


def test_bundled_distributions_maps_modules_and_drops_build_only_tools():
    dists = [
        _dist("pillow", modules=["PIL"], classifiers=["License :: OSI Approved :: MIT License"]),
        _dist("pyinstaller", modules=["PyInstaller"]),
        _dist("pytest", modules=["pytest", "_pytest"]),
        _dist("numpy", modules=["numpy"]),
        _dist("unused", modules=["unused"]),
    ]
    bundled = licenses.bundled_distributions(["PIL", "PyInstaller", "pytest", "numpy"], dists)
    assert [d["name"] for d in bundled] == ["numpy", "pillow"]


def test_write_license_index_writes_three_tables_and_copies_texts(tmp_path: Path):
    app = [_dist("pyphen", version="0.17.2", license_files=[{"name": "LICENSE", "text": "GPL/LGPL/MPL"}],
                 classifiers=["License :: OSI Approved :: Mozilla Public License 1.1 (MPL 1.1)"])]
    sidecar = [_dist("torch", version="2.14.0", license_files=[{"name": "LICENSE", "text": "BSD-3"}],
                     license_expression="BSD-3-Clause")]
    out = licenses.write_license_index(tmp_path, app, sidecar, ROOT / "renderer" / "MODELS.md")
    text = out.read_text(encoding="utf-8")
    assert out == tmp_path / "licenses" / "LICENSES.md"
    for heading in ("GlassTranslate.exe", "renderer", "Models", "PyInstaller"):
        assert heading in text
    assert "pyphen" in text and "torch" in text
    assert (tmp_path / "licenses" / "pyphen-0.17.2" / "LICENSE").read_text(encoding="utf-8") == "GPL/LGPL/MPL"
    assert (tmp_path / "licenses" / "torch-2.14.0" / "LICENSE").read_text(encoding="utf-8") == "BSD-3"


def _lgpl_dist(name: str, **kw: Any) -> Dict[str, Any]:
    return _dist(name, license_expression="LGPL-3.0-only", **kw)


LGPL_TEXT = "                   GNU LESSER GENERAL PUBLIC LICENSE\n                       Version 3\n"


def test_the_index_never_cites_an_absolute_developer_path(tmp_path: Path):
    app = [_dist("numpy", license_expression="BSD-3-Clause")]
    out = licenses.write_license_index(tmp_path, app, [], ROOT / "renderer" / "MODELS.md")
    text = out.read_text(encoding="utf-8")
    # A drive letter, but not the "s:/" inside "https://": the letter must stand alone.
    leaked = re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", text)
    assert leaked is None, f"a drive-letter path leaked into the index: {text[leaked.start():][:60]}"
    assert "`renderer/MODELS.md` in the GlassTranslate" in text


def test_qt_style_distributions_reference_the_canonical_lgpl_text(tmp_path: Path):
    """Qt ships only its commercial licence reference; the bundle carries the LGPL text once."""
    qt = _lgpl_dist("PySide6", version="6.11.2", modules=["PySide6"],
                    license_files=[{"name": "licenses/LicenseRef-Qt-Commercial.txt",
                                    "text": "Licensees holding valid commercial Qt licenses"}])
    pynput = _lgpl_dist("pynput", version="1.8.2",
                        license_files=[{"name": "licenses/COPYING.LGPL", "text": LGPL_TEXT}])
    out = licenses.write_license_index(tmp_path, [qt, pynput], [], ROOT / "renderer" / "MODELS.md")
    shared = tmp_path / "licenses" / licenses.LGPL_TEXT_NAME
    assert shared.read_text(encoding="utf-8") == LGPL_TEXT
    text = out.read_text(encoding="utf-8")
    qt_row = next(line for line in text.splitlines() if line.startswith("| `PySide6`"))
    assert licenses.LGPL_TEXT_NAME in qt_row
    pynput_row = next(line for line in text.splitlines() if line.startswith("| `pynput`"))
    assert licenses.LGPL_TEXT_NAME not in pynput_row  # it ships its own copy
    assert licenses.QT_SOURCE_URL in text and licenses.QT_LGPL_OBLIGATIONS_URL in text


def test_no_canonical_lgpl_file_when_nothing_needs_it(tmp_path: Path):
    licenses.write_license_index(tmp_path, [_dist("numpy", license_expression="BSD-3-Clause")],
                                 [], ROOT / "renderer" / "MODELS.md")
    assert not (tmp_path / "licenses" / licenses.LGPL_TEXT_NAME).exists()


def test_a_missing_lgpl_source_fails_the_build_with_a_clear_message(tmp_path: Path):
    qt = _lgpl_dist("PySide6", license_files=[{"name": "licenses/LicenseRef-Qt-Commercial.txt",
                                               "text": "commercial"}])
    with pytest.raises(licenses.LicenceError, match="COPYING.LGPL"):
        licenses.write_license_index(tmp_path, [qt], [], ROOT / "renderer" / "MODELS.md")


def test_the_index_lists_a_row_per_translation_pack(tmp_path: Path):
    packs = [
        {"pair": "ja->en", "priority": 0, "kind": "argos", "directory": "ja_en",
         "archive": None, "licence": "unknown - see https://example.invalid/index",
         "source": "https://example.invalid/index", "readme": "ja_en/README.md",
         "redistributable": True},
        {"pair": "ja->en", "priority": 10, "kind": "sugoi", "directory": "sugoi-v4-ja-en",
         "archive": None, "licence": "ntt-license (license: other) - no commercial use",
         "source": "https://example.invalid/sugoi", "readme": None, "redistributable": False},
    ]
    out = licenses.write_license_index(tmp_path, [_dist("numpy", license_expression="BSD")], [],
                                       ROOT / "renderer" / "MODELS.md", packs)
    text = out.read_text(encoding="utf-8")
    assert "### 3a. Translation packs" in text
    assert "`models/ja_en`" in text and "`models/ja_en/README.md`" in text
    assert "`models/sugoi-v4-ja-en`" in text and "ntt-license" in text
    assert "unknown - see https://example.invalid/index" in text
    # A shipped pack that may not be passed on carries the personal-use warning.
    assert licenses.PERSONAL_USE_NOTE in text
    assert "--no-research-models" in text


def test_the_index_omits_the_personal_use_note_when_every_pack_is_shareable(tmp_path: Path):
    packs = [{"pair": "en->de", "priority": 0, "kind": "argos", "directory": "translate-en_de-1_3",
              "archive": None, "licence": "unknown", "source": "https://example.invalid/index",
              "readme": None, "redistributable": True}]
    out = licenses.write_license_index(tmp_path, [_dist("numpy", license_expression="BSD")], [],
                                       ROOT / "renderer" / "MODELS.md", packs)
    assert licenses.PERSONAL_USE_NOTE not in out.read_text(encoding="utf-8")


def test_the_index_says_which_packs_are_not_shipped_and_why(tmp_path: Path):
    excluded = [{"directory": "sugoi-v4-ja-en", "pair": "ja->en", "priority": 10,
                 "reason": "NTT research licence: redistribution for research only, "
                           "no commercial use - run the app on a networked machine",
                 "source": "https://example.invalid/sugoi"}]
    out = licenses.write_license_index(tmp_path, [_dist("numpy", license_expression="BSD")], [],
                                       ROOT / "renderer" / "MODELS.md", (), excluded)
    text = out.read_text(encoding="utf-8")
    assert "Not shipped in this bundle" in text
    assert "`models/sugoi-v4-ja-en`" in text
    assert "NTT research licence" in text


def test_write_license_index_refuses_a_gpl_only_bundled_distribution(tmp_path: Path):
    app = [_dist("gpltool", license_expression="GPL-3.0-only")]
    with pytest.raises(licenses.LicenceError, match="gpltool"):
        licenses.write_license_index(tmp_path, app, [], ROOT / "renderer" / "MODELS.md")


# ------------------------------------------------------------------ build_portable / build_renderer
def _manifest(total_bytes: int, packs: List[Dict[str, Any]], research: bool) -> Dict[str, Any]:
    return {"total_bytes": total_bytes, "argos_packs": packs, "excluded_packs": [],
            "research_models_included": research, "files": [], "version": 1}


ARGOS_PACK = {"pair": "en->de", "priority": 0, "kind": "argos",
              "directory": "translate-en_de-1_3", "archive": None, "licence": "unknown",
              "source": "https://example.invalid", "readme": None, "redistributable": True}
SUGOI_PACK = {"pair": "ja->en", "priority": 10, "kind": "sugoi", "directory": "sugoi-v4-ja-en",
              "archive": None, "licence": "ntt-license", "source": "https://example.invalid",
              "readme": None, "redistributable": False}


def test_readme_names_the_shipped_packs_and_warns_about_the_research_one():
    text = build_portable.render_readme(
        _manifest(11_601_108_423, [ARGOS_PACK, SUGOI_PACK], True), "0.2.0", "2026-09-11"
    )
    assert "the models (about 12 GB)" in text  # rounded up from 11.6
    assert "en->de (translate-en_de-1_3)" in text and "ja->en (sugoi-v4-ja-en)" in text
    assert "personal-use licence" in text
    assert "config\\secrets.json stores your API key in plain text" in text
    assert "{" not in text and "}" not in text, "an unsubstituted placeholder was left behind"


def test_readme_omits_the_research_sentence_for_a_shareable_bundle():
    text = build_portable.render_readme(
        _manifest(10_494_824_244, [ARGOS_PACK], False), "0.2.0", "2026-09-11"
    )
    assert "the models (about 11 GB)" in text  # rounded up from 10.5
    assert "personal-use licence" not in text and "Sugoi" not in text
    assert "en->de (translate-en_de-1_3)" in text


@pytest.mark.parametrize("arcname, size, stored", [
    ("x/model.safetensors", 64 * 1024 * 1024, True),
    ("x/model.safetensors", 64 * 1024 * 1024 - 1, False),
    ("x/encoder.onnx", 200 * 1024 * 1024, True),
    ("x/model.bin", 100 * 1024 * 1024, True),
    ("x/lama.pt", 100 * 1024 * 1024, True),
    ("x/README.md", 100 * 1024 * 1024, False),
    ("x/vocab.json", 1024, False),
])
def test_compression_stores_only_big_already_compressed_payloads(arcname, size, stored):
    import zipfile

    expected = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    assert build_portable._compression(arcname, size) == expected


def test_zip_entries_share_one_sorted_top_level_folder(fake_store, tmp_path: Path):
    models_dir, table, ocr_files = fake_store
    manifest = portable_store.store_manifest(
        models_dir, "2026-09-11", quality_files=table, manga_ocr_files=ocr_files
    )
    stage = tmp_path / "GlassTranslate-0.2.0"
    (stage / "renderer").mkdir(parents=True)
    (stage / "GlassTranslate.exe").write_bytes(b"exe")
    (stage / "renderer" / "glassrenderer.exe").write_bytes(b"sidecar")
    app = build_portable._app_entries("0.2.0", stage)
    models = build_portable._models_entries("0.2.0", manifest, tmp_path / "MANIFEST.json")
    for entries in (app, models):
        assert entries == sorted(entries)
        assert {arc.split("/")[0] for arc, _ in entries} == {"GlassTranslate-0.2.0"}
    assert any(arc.endswith("/models/MANIFEST.json") for arc, _ in models)


def _fake_stage(tmp_path: Path) -> Path:
    stage = tmp_path / "GlassTranslate-0.2.0"
    (stage / "renderer").mkdir(parents=True)
    (stage / "GlassTranslate.exe").write_bytes(b"exe")
    (stage / "renderer" / "glassrenderer.exe").write_bytes(b"sidecar")
    (stage / "portable.txt").write_text("marker", encoding="utf-8")
    return stage


def test_zip_plan_single_is_one_full_archive_holding_both_halves(fake_store, tmp_path: Path):
    models_dir, table, ocr_files = fake_store
    manifest = portable_store.store_manifest(
        models_dir, "2026-09-11", quality_files=table, manga_ocr_files=ocr_files
    )
    stage = _fake_stage(tmp_path)
    manifest_path = tmp_path / "MANIFEST.json"
    plan = build_portable._zip_plan("0.2.0", stage, manifest, manifest_path, single=True)
    assert len(plan) == 1
    step, label, name, entries = plan[0]
    assert (step, label) == ("5/7", "full")
    assert name == "GlassTranslate-0.2.0-full-win64.zip" and name.endswith("-full-win64.zip")
    app = build_portable._app_entries("0.2.0", stage)
    models = build_portable._models_entries("0.2.0", manifest, manifest_path)
    assert entries == sorted(entries)
    assert entries == sorted(app + models)  # the union of the pair, nothing more, nothing less
    assert {arc.split("/")[0] for arc, _ in entries} == {"GlassTranslate-0.2.0"}
    names = {arc for arc, _ in entries}
    assert "GlassTranslate-0.2.0/models/MANIFEST.json" in names
    assert "GlassTranslate-0.2.0/GlassTranslate.exe" in names
    assert "GlassTranslate-0.2.0/portable.txt" in names
    assert "GlassTranslate-0.2.0/renderer/glassrenderer.exe" in names


def test_zip_plan_defaults_to_the_portable_and_models_pair(fake_store, tmp_path: Path):
    models_dir, table, ocr_files = fake_store
    manifest = portable_store.store_manifest(
        models_dir, "2026-09-11", quality_files=table, manga_ocr_files=ocr_files
    )
    stage = _fake_stage(tmp_path)
    plan = build_portable._zip_plan("0.2.0", stage, manifest, tmp_path / "MANIFEST.json")
    assert [(step, label, name) for step, label, name, _ in plan] == [
        ("5/7", "portable", "GlassTranslate-0.2.0-portable-win64.zip"),
        ("6/7", "models", "GlassTranslate-0.2.0-models-win64.zip"),
    ]
    assert plan[0][3] == build_portable._app_entries("0.2.0", stage)
    assert plan[1][3] == build_portable._models_entries("0.2.0", manifest, tmp_path / "MANIFEST.json")


def test_parse_args_accepts_single_and_rejects_it_with_stage_only():
    assert build_portable.parse_args(["--single"]).single is True
    assert build_portable.parse_args([]).single is False
    assert build_portable.parse_args(["--single", "--dry-run"]).dry_run is True
    with pytest.raises(SystemExit):
        build_portable.parse_args(["--single", "--stage-only"])


def test_write_sha256_uses_the_sha256sum_format(tmp_path: Path):
    target = tmp_path / "bundle.zip"
    target.write_bytes(b"payload")
    checksum = build_portable._write_sha256(target)
    assert checksum.name == "bundle.zip.sha256"
    assert checksum.read_text(encoding="utf-8") == (
        f"{hashlib.sha256(b'payload').hexdigest()}  bundle.zip\n"
    )


def test_write_zip_is_atomic_and_clears_a_stale_checksum(tmp_path: Path):
    target = tmp_path / "bundle.zip"
    target.write_bytes(b"stale zip")
    stale_checksum = tmp_path / "bundle.zip.sha256"
    stale_checksum.write_text("deadbeef  bundle.zip\n", encoding="utf-8")
    source = tmp_path / "payload.bin"
    source.write_bytes(b"real")
    with pytest.raises(build_portable.PortableBuildError):
        build_portable._write_zip(target, [("a/payload.bin", source), ("a/missing", None)])
    assert not stale_checksum.exists(), "the stale checksum must go before the zip is rewritten"
    assert not target.exists() and not (tmp_path / "bundle.zip.part").exists()


def test_prune_sizes_reads_the_specs_own_figures():
    lines = [
        "[prune] before: binaries 203 (3145.4 MB), datas 5522 (107.9 MB), total 3253.3 MB",
        "[prune]   -    379.9 MB  torch/lib",
        "[prune] after : binaries 193 (2731.8 MB), datas 5514 (107.9 MB), total 2839.6 MB "
        "(saved 413.6 MB)",
    ]
    assert build_renderer._prune_sizes(lines) == {
        "before_mb": 3253.3, "after_mb": 2839.6, "saved_mb": 413.6
    }
    assert build_renderer._prune_sizes(["nothing to see"]) == {}


def test_network_mentions_flags_only_real_network_lines():
    assert build_renderer._network_mentions(["stages loaded and warm\n", "READY 1234\n"]) == []
    hits = build_renderer._network_mentions(["fetching https://huggingface.co/x\n",
                                             "using proxy 127.0.0.1\n"])
    assert set(hits) == {"https://", "proxy", "huggingface.co"}


def test_collect_distributions_reads_the_current_interpreter() -> None:
    dists = licenses.collect_distributions(sys.executable)
    names = {d["name"].lower() for d in dists}
    assert "pytest" in names
    sample: Optional[Dict[str, Any]] = next((d for d in dists if d["name"].lower() == "pytest"), None)
    assert sample is not None
    assert sample["version"] and isinstance(sample["modules"], list)
