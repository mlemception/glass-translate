"""Model store for the manga-ocr ONNX files (``glasstranslate.ocr.models``).

Everything runs against a fake ``requests``-like session: no network, no real
model bytes.  Covers the missing-file report, digest / size verification with
the atomic ``.part`` -> final rename, corrupt-download cleanup, path-traversal
rejection, cancellation and the frozen-build directory.
"""
from __future__ import annotations

import hashlib
import sys
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pytest

from glasstranslate.config import settings as S
from glasstranslate.ocr import models as M


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


def _file(name: str, body: bytes, *, digest: bool = True, remote: Optional[str] = None) -> M.ModelFile:
    return M.ModelFile(
        name=name,
        remote_path=remote or f"onnx/{name}",
        size=len(body),
        sha256=hashlib.sha256(body).hexdigest() if digest else None,
    )


def _routes(*pairs: Tuple[M.ModelFile, bytes]) -> Dict[str, object]:
    return {M.download_url(f.remote_path, f.repo): body for f, body in pairs}


# ------------------------------------------------------------ constants
def test_manga_ocr_file_table_names_the_fp16_encoder_decoder_and_vocab() -> None:
    names = {f.name for f in M.MANGA_OCR_FILES}
    assert {M.ENCODER_FILE, M.DECODER_FILE, M.VOCAB_FILE, M.GENERATION_CONFIG_FILE, M.CONFIG_FILE} <= names
    assert "fp16" in M.ENCODER_FILE
    # The repo's decoder_model_fp16.onnx is a broken export that onnxruntime
    # cannot load (see docs/perf/2026-09-09-mangaocr-probe.md), so the decoder
    # is the int8 variant.  Pin it so a silent swap shows up here.
    assert M.DECODER_FILE == "decoder_model_int8.onnx"
    for f in M.MANGA_OCR_FILES:
        assert f.size > 0
        assert "/" not in f.name and "\\" not in f.name
        # Every shipped file is digest-pinned and fetched at a fixed commit.
        assert f.sha256 is not None and len(f.sha256) == 64
        assert len(f.revision) == 40
    by_repo = {f.repo for f in M.MANGA_OCR_FILES}
    assert by_repo == {M.MANGA_OCR_REPO, M.MANGA_OCR_TOKENIZER_REPO}


def test_download_url_resolves_against_the_pinned_revision() -> None:
    url = M.download_url("onnx/encoder_model_fp16.onnx")
    assert url == (
        f"https://huggingface.co/{M.MANGA_OCR_REPO}/resolve/{M.MANGA_OCR_REVISION}/onnx/encoder_model_fp16.onnx"
    )
    assert M.download_url("vocab.txt", "kha-white/manga-ocr-base", "abc123").endswith(
        "/kha-white/manga-ocr-base/resolve/abc123/vocab.txt"
    )
    vocab = next(f for f in M.MANGA_OCR_FILES if f.name == M.VOCAB_FILE)
    assert M.download_url(vocab.remote_path, vocab.repo, vocab.revision) == (
        f"https://huggingface.co/{M.MANGA_OCR_TOKENIZER_REPO}/resolve/{M.MANGA_OCR_TOKENIZER_REVISION}/vocab.txt"
    )


# ----------------------------------------------------------- readiness
def test_missing_files_lists_everything_on_an_empty_dir(tmp_path: Path) -> None:
    assert M.missing_files(tmp_path) == list(M.MANGA_OCR_FILES)
    assert M.models_ready(tmp_path) is False
    assert M.manga_ocr_dir(tmp_path) == tmp_path / M.MANGA_OCR_SUBDIR


def test_models_ready_once_every_file_is_present(tmp_path: Path) -> None:
    target = M.manga_ocr_dir(tmp_path)
    target.mkdir()
    for f in M.MANGA_OCR_FILES[:-1]:
        (target / f.name).write_bytes(b"x")
    assert [f.name for f in M.missing_files(tmp_path)] == [M.MANGA_OCR_FILES[-1].name]
    assert M.models_ready(tmp_path) is False
    (target / M.MANGA_OCR_FILES[-1].name).write_bytes(b"x")
    assert M.models_ready(tmp_path) is True
    assert M.missing_files(tmp_path) == []


def test_missing_files_accepts_a_string_models_dir(tmp_path: Path) -> None:
    assert len(M.missing_files(str(tmp_path))) == len(M.MANGA_OCR_FILES)


# ------------------------------------------------------------ download
def test_download_verifies_digest_and_renames_atomically(tmp_path: Path) -> None:
    body = b"encoder-bytes" * 50
    f = _file("encoder.onnx", body)
    session = FakeSession(_routes((f, body)))

    out = M.ensure_models(tmp_path, files=[f], session=session)

    assert out == M.manga_ocr_dir(tmp_path)
    assert (out / "encoder.onnx").read_bytes() == body
    assert list(out.glob("*.part")) == []
    assert session.requests[0][1].get("stream") is True
    assert M.models_ready(tmp_path, files=[f]) is True


def test_present_files_are_not_downloaded_again(tmp_path: Path) -> None:
    body = b"abc"
    f = _file("vocab.txt", body, remote="vocab.txt")
    target = M.manga_ocr_dir(tmp_path)
    target.mkdir()
    (target / "vocab.txt").write_bytes(body)
    session = FakeSession({})

    M.ensure_models(tmp_path, files=[f], session=session)

    assert session.requests == []


def test_size_only_verification_when_no_digest_is_known(tmp_path: Path) -> None:
    good = b"{}" * 20
    f = _file("config.json", good, digest=False, remote="config.json")
    M.ensure_models(tmp_path, files=[f], session=FakeSession(_routes((f, good))))
    assert (M.manga_ocr_dir(tmp_path) / "config.json").read_bytes() == good

    short = M.ModelFile("gen.json", "gen.json", size=10, sha256=None)
    with pytest.raises(M.ModelDownloadError, match="size"):
        M.ensure_models(tmp_path, files=[short], session=FakeSession(_routes((short, b"123"))))
    assert not (M.manga_ocr_dir(tmp_path) / "gen.json").exists()


def test_corrupt_download_is_deleted_and_reported(tmp_path: Path) -> None:
    body = b"real model bytes" * 10
    f = _file("decoder.onnx", body)
    session = FakeSession(_routes((f, body[:-1] + b"?")))  # same size, different bytes

    with pytest.raises(M.ModelDownloadError) as exc:
        M.ensure_models(tmp_path, files=[f], session=session)

    assert "decoder.onnx" in str(exc.value) and "sha256" in str(exc.value)
    target = M.manga_ocr_dir(tmp_path)
    assert not (target / "decoder.onnx").exists()
    assert list(target.glob("*.part")) == []


def test_oversized_stream_is_cut_off_and_reported(tmp_path: Path) -> None:
    """Security review M1: a lying Content-Length / endless stream must not fill the disk."""
    body = b"real model bytes" * 10
    f = _file("decoder.onnx", body)
    session = FakeSession(_routes((f, body + b"extra bytes beyond the pinned size")))

    with pytest.raises(M.ModelDownloadError, match="more than the expected"):
        M.ensure_models(tmp_path, files=[f], session=session)

    target = M.manga_ocr_dir(tmp_path)
    assert not (target / "decoder.onnx").exists()
    assert list(target.glob("*.part")) == []


def test_http_error_is_reported_as_model_download_error(tmp_path: Path) -> None:
    f = M.ModelFile("x.onnx", "onnx/x.onnx", size=3, sha256=None)
    with pytest.raises(M.ModelDownloadError, match="x.onnx"):
        M.ensure_models(tmp_path, files=[f], session=FakeSession(_routes((f, 503))))
    assert list(M.manga_ocr_dir(tmp_path).glob("*")) == []


@pytest.mark.parametrize("bad", ["../evil.onnx", "sub/evil.onnx", "sub\\evil.onnx", "..", ""])
def test_path_traversal_in_a_model_name_is_rejected(tmp_path: Path, bad: str) -> None:
    f = M.ModelFile(bad, "onnx/evil.onnx", size=1, sha256=None)
    session = FakeSession(_routes((f, b"x")))
    with pytest.raises(ValueError):
        M.ensure_models(tmp_path, files=[f], session=session)
    assert session.requests == []
    assert not (tmp_path / "evil.onnx").exists()
    assert not (M.manga_ocr_dir(tmp_path) / "evil.onnx").exists()


def test_progress_callback_reports_percent_per_file(tmp_path: Path) -> None:
    body = bytes(range(256)) * 4
    f = _file("enc.onnx", body)
    seen: List[Tuple[str, int]] = []

    M.ensure_models(
        tmp_path, files=[f], session=FakeSession(_routes((f, body))), progress=lambda n, p: seen.append((n, p))
    )

    assert seen and all(n == "enc.onnx" for n, _ in seen)
    pcts = [p for _, p in seen]
    assert pcts == sorted(pcts) and 0 <= pcts[0] and pcts[-1] == 100


def test_cancel_event_aborts_and_removes_the_partial(tmp_path: Path) -> None:
    body = b"z" * 1000
    f = _file("enc.onnx", body)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(M.ModelDownloadError, match="cancel"):
        M.ensure_models(tmp_path, files=[f], session=FakeSession(_routes((f, body))), cancel=cancel)

    target = M.manga_ocr_dir(tmp_path)
    assert not (target / "enc.onnx").exists()
    assert list(target.glob("*.part")) == []


def test_default_files_are_used_when_none_are_given(tmp_path: Path) -> None:
    session = FakeSession({})  # every URL 404s
    with pytest.raises(M.ModelDownloadError):
        M.ensure_models(tmp_path, session=session)
    first = M.MANGA_OCR_FILES[0]
    assert session.requests[0][0] == M.download_url(first.remote_path, first.repo, first.revision)


def test_default_session_is_created_and_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    body = b"vocab"
    f = _file("vocab.txt", body, remote="vocab.txt")
    created: List[FakeSession] = []

    class TrackingSession(FakeSession):
        def __init__(self) -> None:
            super().__init__(_routes((f, body)))
            self.closed = False
            created.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(requests, "Session", TrackingSession)

    M.ensure_models(tmp_path, files=[f])

    assert len(created) == 1 and created[0].closed
    assert (M.manga_ocr_dir(tmp_path) / "vocab.txt").read_bytes() == body


def test_unwritable_models_dir_raises_model_download_error(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x", encoding="utf-8")
    f = _file("vocab.txt", b"v", remote="vocab.txt")
    with pytest.raises(M.ModelDownloadError, match="cannot create"):
        M.ensure_models(not_a_dir, files=[f], session=FakeSession(_routes((f, b"v"))))


def test_concurrent_callers_download_once(tmp_path: Path) -> None:
    import time

    body = b"m" * 4096
    f = _file("enc.onnx", body)

    class SlowResponse(FakeResponse):
        def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
            for chunk in super().iter_content(chunk_size):
                time.sleep(0.002)
                yield chunk

    class SlowSession(FakeSession):
        def get(self, url: str, **kwargs: object) -> FakeResponse:
            self.requests.append((url, dict(kwargs)))
            return SlowResponse(body, chunk=256)

    session = SlowSession(_routes((f, body)))
    errors: List[BaseException] = []

    def worker() -> None:
        try:
            M.ensure_models(tmp_path, files=[f], session=session)
        except BaseException as exc:  # noqa: BLE001 - surfaced via ``errors``
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    assert len(session.requests) == 1  # the lock made the others find the file
    assert (M.manga_ocr_dir(tmp_path) / "enc.onnx").read_bytes() == body
    assert list(M.manga_ocr_dir(tmp_path).glob("*.part")) == []


# ------------------------------------------------------------- frozen
def test_frozen_build_uses_user_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    target = M.manga_ocr_dir(S.default_models_dir())
    assert target == S.user_data_dir() / "models" / M.MANGA_OCR_SUBDIR
    assert S.user_data_dir() in target.parents
    if sys.platform == "win32":
        assert target == tmp_path / "GlassTranslate" / "models" / M.MANGA_OCR_SUBDIR
