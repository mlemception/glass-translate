"""Package discovery in :mod:`glasstranslate.translate.argos`: ``models_dir``
itself as a package directory, and the metadata-less Sugoi CTranslate2
layout.  No models are ever loaded here (empty placeholder files, ``device=
"cpu"``) - only :meth:`ArgosCT2Translator.rescan` / ``installed_packages`` and
:func:`_read_metadata_dir` are exercised."""
from __future__ import annotations

import json
from pathlib import Path

from glasstranslate.translate.argos import ArgosCT2Translator, _read_metadata_dir


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_metadata(directory: Path, from_code: str, to_code: str) -> None:
    (directory / "metadata.json").write_text(
        json.dumps({"from_code": from_code, "to_code": to_code}), encoding="utf-8"
    )


def _sugoi_layout(directory: Path) -> None:
    _touch(directory / "config.json")
    _touch(directory / "model.bin")
    _touch(directory / "spm" / "spm.ja.nopretok.model")
    _touch(directory / "spm" / "spm.en.nopretok.model")


# ------------------------------------------------------- _read_metadata_dir
def test_read_metadata_dir_recognises_metadata_less_sugoi_layout(tmp_path: Path) -> None:
    _sugoi_layout(tmp_path)

    pkg = _read_metadata_dir(tmp_path)

    assert pkg is not None
    assert pkg.pair == ("ja", "en")
    assert pkg.directory == tmp_path
    assert pkg.model_subdir == "."
    assert pkg.source_spm == "spm/spm.ja.nopretok.model"
    assert pkg.target_spm == "spm/spm.en.nopretok.model"
    assert pkg.priority == 10


def test_read_metadata_dir_ignores_dir_with_only_model_bin(tmp_path: Path) -> None:
    _touch(tmp_path / "model.bin")

    assert _read_metadata_dir(tmp_path) is None


def test_read_metadata_dir_does_not_write_metadata_json(tmp_path: Path) -> None:
    _sugoi_layout(tmp_path)

    _read_metadata_dir(tmp_path)

    assert not (tmp_path / "metadata.json").exists()


# --------------------------------------------------------------- rescan()
def test_rescan_discovers_models_dir_itself_as_plain_ct2_package(tmp_path: Path) -> None:
    _touch(tmp_path / "model.bin")
    _touch(tmp_path / "sentencepiece.model")
    _write_metadata(tmp_path, "ja", "en")

    translator = ArgosCT2Translator(tmp_path, device="cpu")

    packages = translator.installed_packages()
    assert len(packages) == 1
    assert packages[0].pair == ("ja", "en")
    assert packages[0].directory == tmp_path


def test_rescan_discovers_models_dir_itself_as_sugoi_layout(tmp_path: Path) -> None:
    _sugoi_layout(tmp_path)

    translator = ArgosCT2Translator(tmp_path, device="cpu")

    packages = translator.installed_packages()
    assert len(packages) == 1
    pkg = packages[0]
    assert pkg.pair == ("ja", "en")
    assert pkg.directory == tmp_path
    assert pkg.source_spm == "spm/spm.ja.nopretok.model"
    assert pkg.target_spm == "spm/spm.en.nopretok.model"
    assert pkg.priority == 10


def test_rescan_still_discovers_argos_layout_subdir(tmp_path: Path) -> None:
    sub = tmp_path / "translate-ja_en-1_0"
    _touch(sub / "model" / "model.bin")
    _touch(sub / "sentencepiece.model")
    _write_metadata(sub, "ja", "en")

    translator = ArgosCT2Translator(tmp_path, device="cpu")

    packages = translator.installed_packages()
    assert len(packages) == 1
    assert packages[0].pair == ("ja", "en")
    assert packages[0].directory == sub


def test_read_metadata_dir_rejects_spm_paths_escaping_the_package(tmp_path: Path) -> None:
    """metadata.json is untrusted: ``..`` and absolute spm paths must not
    resolve outside the package directory."""
    outside = tmp_path / "outside.model"
    _touch(outside)
    package = tmp_path / "pkg"
    _touch(package / "model.bin")
    (package / "metadata.json").write_text(
        json.dumps(
            {
                "from_code": "ja",
                "to_code": "en",
                "source_spm": "../outside.model",
                "target_spm": str(outside),
            }
        ),
        encoding="utf-8",
    )

    assert _read_metadata_dir(package) is None
