"""Pinned model table and download store of the quality renderer.

Runs in the **main** venv (no torch): :mod:`glassrenderer.models` is the one
sidecar module the app also needs to reason about, so it imports nothing from
the generative stack.  Everything here uses a fake ``requests``-like session,
exactly as ``tests/test_ocr_models.py`` does for the manga-ocr store.

The table itself is data (``renderer/glassrenderer/models.json``); a copy lives
at ``glasstranslate/render/quality_models.json`` for the app-side downloader
and one test below keeps the two byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pytest

RENDERER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RENDERER_ROOT.parent
if str(RENDERER_ROOT) not in sys.path:
    sys.path.insert(0, str(RENDERER_ROOT))

from glassrenderer import models as M  # noqa: E402  (path set above)

APP_TABLE = PROJECT_ROOT / "glasstranslate" / "render" / "quality_models.json"
PERMISSIVE = re.compile(r"MIT|Apache-2\.0|BSD|OpenRAIL\+\+-M", re.IGNORECASE)
COPYLEFT = re.compile(r"\b(?:A?GPL|LGPL|Fair AI Public License|non-commercial)\b", re.IGNORECASE)


# ----------------------------------------------------------------- fakes
class FakeResponse:
    def __init__(self, body: bytes, status: int = 200, chunk: int = 7) -> None:
        self._body = body
        self.status_code = status
        self.headers = {"Content-Length": str(len(body))}
        self._chunk = chunk
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} for fake url")

    def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
        for i in range(0, len(self._body), self._chunk):
            yield self._body[i : i + self._chunk]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    """Maps URL -> body bytes (or an int HTTP status for an error)."""

    def __init__(self, routes: Dict[str, object]) -> None:
        self.routes = routes
        self.requests: List[Tuple[str, dict]] = []

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.requests.append((url, dict(kwargs)))
        if url not in self.routes:
            return FakeResponse(b"", status=404)
        body = self.routes[url]
        if isinstance(body, int):
            return FakeResponse(b"", status=body)
        assert isinstance(body, bytes)
        return FakeResponse(body)


def _file(name: str, body: bytes, *, subdir: str = "lama", digest: bool = True) -> M.ModelFile:
    return M.ModelFile(
        name=name,
        remote_path=name,
        size=len(body),
        sha256=hashlib.sha256(body).hexdigest() if digest else None,
        repo="acme/models",
        revision="0" * 40,
        licence="MIT",
        subdir=subdir,
    )


def _routes(*pairs: Tuple[M.ModelFile, bytes]) -> Dict[str, object]:
    return {M.download_url(f.remote_path, f.repo, f.revision): body for f, body in pairs}


def _place(models_dir: Path, spec: M.ModelFile, body: bytes) -> Path:
    """Write ``body`` where ``spec`` would live under ``models_dir``."""
    path = M.file_path(models_dir, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


# ------------------------------------------------------------ the table
def test_table_is_well_formed() -> None:
    assert len(M.QUALITY_FILES) >= 20
    for spec in M.QUALITY_FILES:
        assert spec.size > 0, spec.name
        assert spec.sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", spec.sha256), spec.name
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision), spec.name
        assert "/" not in spec.name and "\\" not in spec.name and ".." not in spec.name
        assert spec.subdir and not spec.subdir.startswith(("/", "\\"))
        assert ".." not in spec.subdir.split("/")
        assert spec.repo.count("/") == 1
        assert spec.remote_path


def test_every_file_records_a_permissive_licence() -> None:
    for spec in M.QUALITY_FILES:
        assert PERMISSIVE.search(spec.licence), f"{spec.name}: {spec.licence!r}"
        assert not COPYLEFT.search(spec.licence), f"{spec.name}: {spec.licence!r}"


def test_table_covers_the_four_model_groups() -> None:
    groups = {M.group_of(spec) for spec in M.QUALITY_FILES}
    assert groups == {M.LAMA, M.SDXL, M.CONTROLNET}
    names = {spec.name for spec in M.QUALITY_FILES}
    assert M.LAMA_WEIGHTS in names
    assert "Illustrious-XL-v1.0.safetensors" in names
    # The union ControlNet and the fp16-fix VAE land in the diffusers layout.
    assert (M.CONTROLNET_SUBDIR, "diffusion_pytorch_model.safetensors") in {
        (s.subdir, s.name) for s in M.QUALITY_FILES
    }
    assert (M.VAE_SUBDIR, "diffusion_pytorch_model.safetensors") in {
        (s.subdir, s.name) for s in M.QUALITY_FILES
    }


def test_single_file_loading_has_every_offline_config_it_needs() -> None:
    """``from_single_file`` must never reach the Hub: pin the SDXL base configs."""
    paths = {f"{s.subdir}/{s.name}" for s in M.QUALITY_FILES}
    base = M.SDXL_CONFIG_SUBDIR
    for needed in (
        f"{base}/model_index.json",
        f"{base}/scheduler/scheduler_config.json",
        f"{base}/text_encoder/config.json",
        f"{base}/text_encoder_2/config.json",
        f"{base}/unet/config.json",
        f"{base}/vae/config.json",
    ):
        assert needed in paths, needed
    for tok in (f"{base}/tokenizer", f"{base}/tokenizer_2"):
        for leaf in ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"):
            assert f"{tok}/{leaf}" in paths, f"{tok}/{leaf}"


def test_no_two_entries_claim_the_same_path() -> None:
    paths = [f"{s.subdir}/{s.name}" for s in M.QUALITY_FILES]
    assert len(paths) == len(set(paths))


def test_download_url_resolves_against_the_pinned_revision() -> None:
    spec = next(s for s in M.QUALITY_FILES if s.name == M.LAMA_WEIGHTS)
    assert M.download_url(spec.remote_path, spec.repo, spec.revision) == (
        f"https://huggingface.co/{spec.repo}/resolve/{spec.revision}/{spec.remote_path}"
    )


def test_total_download_size_is_reported() -> None:
    total = M.total_bytes()
    assert total == sum(s.size for s in M.QUALITY_FILES)
    assert 8 * 1024**3 < total < 14 * 1024**3  # ~10 GB of weights


# ----------------------------------------------------- the shared copy
def test_app_side_copy_is_byte_identical() -> None:
    assert APP_TABLE.is_file(), f"{APP_TABLE} missing"
    assert APP_TABLE.read_bytes() == M.MODELS_TABLE_PATH.read_bytes()


def test_the_json_round_trips_into_the_table() -> None:
    raw = json.loads(M.MODELS_TABLE_PATH.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert len(raw["files"]) == len(M.QUALITY_FILES)
    assert M.load_table(M.MODELS_TABLE_PATH) == M.QUALITY_FILES


# ----------------------------------------------------------- readiness
def test_missing_files_lists_everything_on_an_empty_dir(tmp_path: Path) -> None:
    assert M.missing_files(tmp_path) == list(M.QUALITY_FILES)
    assert M.models_ready(tmp_path) is False
    assert M.quality_dir(tmp_path) == tmp_path / M.QUALITY_SUBDIR


def test_models_ready_once_every_file_is_present(tmp_path: Path) -> None:
    files = [_file("a.pt", b"aaa"), _file("b.json", b"bb", subdir="sdxl")]
    _place(tmp_path, files[0], b"aaa")
    assert [s.name for s in M.missing_files(tmp_path, files)] == ["b.json"]
    _place(tmp_path, files[1], b"bb")
    assert M.models_ready(tmp_path, files) is True


def test_a_truncated_file_still_counts_as_missing(tmp_path: Path) -> None:
    """A half-written weight file must be re-fetched, not loaded."""
    spec = _file("a.pt", b"aaaaaaaa")
    _place(tmp_path, spec, b"aaa")
    assert M.missing_files(tmp_path, [spec]) == [spec]


def test_group_status_reports_missing_ready_per_group(tmp_path: Path) -> None:
    files = [
        _file("weights.pt", b"lama", subdir=M.LAMA_SUBDIR),
        _file("ckpt.safetensors", b"sdxl", subdir=M.SDXL_SUBDIR),
        _file("config.json", b"{}", subdir=M.VAE_SUBDIR),
        _file("cn.safetensors", b"controlnet", subdir=M.CONTROLNET_SUBDIR),
    ]
    assert M.group_status(tmp_path, files) == {
        M.LAMA: "missing",
        M.SDXL: "missing",
        M.CONTROLNET: "missing",
    }
    _place(tmp_path, files[0], b"lama")
    _place(tmp_path, files[1], b"sdxl")
    assert M.group_status(tmp_path, files)[M.LAMA] == "ready"
    # The VAE belongs to the SDXL group, so that group is not ready yet.
    assert M.group_status(tmp_path, files)[M.SDXL] == "missing"
    _place(tmp_path, files[2], b"{}")
    assert M.group_status(tmp_path, files)[M.SDXL] == "ready"
    assert M.group_status(tmp_path, files)[M.CONTROLNET] == "missing"


def test_quality_dir_accepts_the_store_itself(tmp_path: Path) -> None:
    """The app may pass ``<models>`` or ``<models>/quality`` as --models-dir."""
    assert M.quality_dir(tmp_path / "models") == tmp_path / "models" / M.QUALITY_SUBDIR
    store = tmp_path / "models" / M.QUALITY_SUBDIR
    assert M.quality_dir(store) == store


def test_missing_files_accepts_a_string_models_dir(tmp_path: Path) -> None:
    assert len(M.missing_files(str(tmp_path))) == len(M.QUALITY_FILES)


# ------------------------------------------------------------ download
def test_download_verifies_digest_and_renames_atomically(tmp_path: Path) -> None:
    body = b"lama-bytes" * 50
    spec = _file("anime_manga_lama.pt", body)
    session = FakeSession(_routes((spec, body)))

    out = M.ensure_models(tmp_path, files=[spec], session=session)

    assert out == M.quality_dir(tmp_path)
    assert M.file_path(tmp_path, spec).read_bytes() == body
    assert list(out.rglob("*.part")) == []
    assert session.requests[0][1].get("stream") is True


def test_subdirectories_are_created_for_nested_entries(tmp_path: Path) -> None:
    body = b'{"a": 1}'
    spec = _file("vocab.json", body, subdir="sdxl-base-config/tokenizer_2")
    M.ensure_models(tmp_path, files=[spec], session=FakeSession(_routes((spec, body))))
    assert (M.quality_dir(tmp_path) / "sdxl-base-config" / "tokenizer_2" / "vocab.json").read_bytes() == body


def test_present_files_are_not_downloaded_again(tmp_path: Path) -> None:
    body = b"abc"
    spec = _file("config.json", body)
    path = M.file_path(tmp_path, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    session = FakeSession({})

    M.ensure_models(tmp_path, files=[spec], session=session)

    assert session.requests == []


def test_corrupt_download_is_deleted_and_reported(tmp_path: Path) -> None:
    body = b"real weight bytes" * 10
    spec = _file("model.safetensors", body, subdir="controlnet")
    session = FakeSession(_routes((spec, body[:-1] + b"?")))  # same size, other bytes

    with pytest.raises(M.ModelDownloadError) as exc:
        M.ensure_models(tmp_path, files=[spec], session=session)

    assert "model.safetensors" in str(exc.value) and "sha256" in str(exc.value)
    assert not M.file_path(tmp_path, spec).exists()
    assert list(M.quality_dir(tmp_path).rglob("*.part")) == []


def test_oversized_stream_is_cut_off_and_reported(tmp_path: Path) -> None:
    body = b"weights" * 10
    spec = _file("model.safetensors", body)
    session = FakeSession(_routes((spec, body + b"and then some")))

    with pytest.raises(M.ModelDownloadError, match="more than the expected"):
        M.ensure_models(tmp_path, files=[spec], session=session)
    assert list(M.quality_dir(tmp_path).rglob("*.part")) == []


def test_short_download_is_reported_as_a_size_mismatch(tmp_path: Path) -> None:
    spec = M.ModelFile("gen.json", "gen.json", 10, None, "acme/models", "0" * 40, "MIT", "lama")
    with pytest.raises(M.ModelDownloadError, match="size"):
        M.ensure_models(tmp_path, files=[spec], session=FakeSession(_routes((spec, b"123"))))
    assert not M.file_path(tmp_path, spec).exists()


def test_http_error_is_reported_as_model_download_error(tmp_path: Path) -> None:
    spec = _file("x.pt", b"xyz")
    with pytest.raises(M.ModelDownloadError, match="x.pt"):
        M.ensure_models(tmp_path, files=[spec], session=FakeSession(_routes((spec, 503))))


@pytest.mark.parametrize("bad", ["../evil.pt", "sub/evil.pt", "sub\\evil.pt", "..", ""])
def test_path_traversal_in_a_model_name_is_rejected(tmp_path: Path, bad: str) -> None:
    spec = M.ModelFile(bad, "evil.pt", 1, None, "acme/models", "0" * 40, "MIT", "lama")
    session = FakeSession({})
    with pytest.raises(ValueError):
        M.ensure_models(tmp_path, files=[spec], session=session)
    assert session.requests == []


@pytest.mark.parametrize("bad", ["../evil", "/abs", "lama/../..", "C:\\windows", ""])
def test_path_traversal_in_a_subdir_is_rejected(tmp_path: Path, bad: str) -> None:
    spec = M.ModelFile("x.pt", "x.pt", 1, None, "acme/models", "0" * 40, "MIT", bad)
    session = FakeSession({})
    with pytest.raises(ValueError):
        M.ensure_models(tmp_path, files=[spec], session=session)
    assert session.requests == []


def test_progress_callback_reports_percent_per_file(tmp_path: Path) -> None:
    body = bytes(range(256)) * 4
    spec = _file("enc.pt", body)
    seen: List[Tuple[str, int]] = []

    M.ensure_models(
        tmp_path, files=[spec], session=FakeSession(_routes((spec, body))), progress=lambda n, p: seen.append((n, p))
    )

    assert seen and all(n == "enc.pt" for n, _ in seen)
    pcts = [p for _, p in seen]
    assert pcts == sorted(pcts) and pcts[-1] == 100


def test_cancel_event_aborts_and_removes_the_partial(tmp_path: Path) -> None:
    body = b"z" * 1000
    spec = _file("enc.pt", body)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(M.ModelDownloadError, match="cancel"):
        M.ensure_models(tmp_path, files=[spec], session=FakeSession(_routes((spec, body))), cancel=cancel)

    assert list(M.quality_dir(tmp_path).rglob("*.part")) == []


def test_default_session_is_created_and_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    body = b"vocab"
    spec = _file("vocab.json", body)
    created: List[FakeSession] = []

    class TrackingSession(FakeSession):
        def __init__(self) -> None:
            super().__init__(_routes((spec, body)))
            self.closed = False
            created.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(requests, "Session", TrackingSession)

    M.ensure_models(tmp_path, files=[spec])

    assert len(created) == 1 and created[0].closed


def test_concurrent_callers_download_once(tmp_path: Path) -> None:
    import time

    body = b"m" * 4096
    spec = _file("enc.pt", body)

    class SlowResponse(FakeResponse):
        def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
            for chunk in super().iter_content(chunk_size):
                time.sleep(0.002)
                yield chunk

    class SlowSession(FakeSession):
        def get(self, url: str, **kwargs: object) -> FakeResponse:
            self.requests.append((url, dict(kwargs)))
            return SlowResponse(body, chunk=256)

    session = SlowSession(_routes((spec, body)))
    errors: List[BaseException] = []

    def worker() -> None:
        try:
            M.ensure_models(tmp_path, files=[spec], session=session)
        except BaseException as exc:  # noqa: BLE001 - surfaced via ``errors``
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    assert len(session.requests) == 1


# ------------------------------------------------------- verify_file
def test_verify_file_accepts_a_good_file_and_rejects_a_tampered_one(tmp_path: Path) -> None:
    body = b"torchscript pickle bytes"
    spec = _file("anime_manga_lama.pt", body)
    path = M.file_path(tmp_path, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)

    M.verify_file(tmp_path, spec)  # no raise

    path.write_bytes(b"torchscript pickle byteS")
    with pytest.raises(M.ModelDownloadError, match="sha256"):
        M.verify_file(tmp_path, spec)


def test_verify_file_reports_a_missing_file(tmp_path: Path) -> None:
    spec = _file("anime_manga_lama.pt", b"abc")
    with pytest.raises(M.ModelDownloadError, match="missing"):
        M.verify_file(tmp_path, spec)


# ---------------------------------------------------- line-art builder
def test_lineart_control_image_is_white_lines_on_black() -> None:
    """The union ControlNet's thin-line channel wants inverted art (CPU only)."""
    import numpy as np

    from glassrenderer.stages import lineart

    panel = np.full((64, 64, 3), 255, np.uint8)
    panel[30:34, :] = 0  # a black ink stroke on white paper
    mask = np.zeros((64, 64), np.uint8)
    mask[8:20, 8:20] = 255

    control = lineart.build_control_image(panel, mask)

    assert control.shape == panel.shape and control.dtype == np.uint8
    assert control[32, 50, 0] > 200, "ink must become a white line"
    assert control[5, 5, 0] < 40, "paper must become black ground"
    assert control[8:20, 8:20].max() == 0, "the masked region must be blanked"
    assert (control[:, :, 0] == control[:, :, 1]).all() and (control[:, :, 1] == control[:, :, 2]).all()


def test_lineart_dilates_the_mask_before_blanking() -> None:
    import numpy as np

    from glassrenderer.stages import lineart

    panel = np.zeros((64, 64, 3), np.uint8)  # all ink -> all white lines
    mask = np.zeros((64, 64), np.uint8)
    mask[30:34, 30:34] = 255

    control = lineart.build_control_image(panel, mask, dilate_px=4)

    assert control[32, 32, 0] == 0
    assert control[26, 32, 0] == 0, "the blanked hole must be wider than the mask"
    assert control[10, 10, 0] > 200
