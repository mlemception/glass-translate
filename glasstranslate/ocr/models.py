"""Model store for the manga-ocr ONNX recogniser.

The recogniser (:mod:`glasstranslate.ocr.mangaocr`) needs the files listed in
:data:`MANGA_OCR_FILES` inside ``<models_dir>/manga-ocr``.  They are never
bundled with the app: :func:`ensure_models` downloads the missing ones from
Hugging Face on first use, streaming each into ``<name>.part`` and renaming it
into place only after the SHA-256 (or, for the small non-LFS files, the size)
matches the pinned value.

``models_dir`` is :func:`glasstranslate.config.settings.default_models_dir`
by default: ``<project>/models`` in a source checkout and
``%LOCALAPPDATA%/GlassTranslate/models`` in the frozen exe.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Sequence, Tuple, Union

log = logging.getLogger(__name__)

MANGA_OCR_REPO = "onnx-community/manga-ocr-base-ONNX"
# Commit of the repo the sizes / digests below were taken from (2026-04-18).
MANGA_OCR_REVISION = "f9023406bb2f6b17df67bc4a327c56ecd20611f0"
# The ONNX repo does not ship the tokenizer files; they come from the original
# PyTorch checkpoint (same author, same vocabulary, Apache-2.0).
MANGA_OCR_TOKENIZER_REPO = "kha-white/manga-ocr-base"
MANGA_OCR_TOKENIZER_REVISION = "aa6573bd10b0d446cbf622e29c3e084914df9741"
MANGA_OCR_SUBDIR = "manga-ocr"

HF_RESOLVE_URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"

ENCODER_FILE = "encoder_model_fp16.onnx"
# ``decoder_model_fp16.onnx`` in the repo is a broken export (an internal Cast
# emits float16 where the graph declares float32; onnxruntime refuses to load
# it on every provider - see docs/perf/2026-09-09-mangaocr-probe.md).  The
# int8 weight-quantised decoder loads, is 30 MB instead of 59 MB and decodes
# the same text on the probe pages, so it is the shipped decoder.
DECODER_FILE = "decoder_model_int8.onnx"
VOCAB_FILE = "vocab.txt"
CONFIG_FILE = "config.json"
GENERATION_CONFIG_FILE = "generation_config.json"
PREPROCESSOR_CONFIG_FILE = "preprocessor_config.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

CHUNK_SIZE = 1 << 20  # 1 MiB streaming chunks
REQUEST_TIMEOUT = (10.0, 60.0)  # connect, read (seconds)
PART_SUFFIX = ".part"

ProgressCallback = Callable[[str, int], None]


class ModelDownloadError(RuntimeError):
    """A model file could not be downloaded or failed verification."""


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


@dataclass(frozen=True)
class ModelFile:
    """One file of the manga-ocr bundle.

    ``sha256`` is the Git-LFS object id published by the Hugging Face API for
    the large blobs and the digest of the fetched bytes for the small
    plain-git files.  ``None`` means "verify by size only" (never used for
    the shipped table; kept for callers that fetch their own files).
    ``revision`` pins the repo commit so the URL content cannot drift.
    """

    name: str
    remote_path: str
    size: int
    sha256: Optional[str]
    repo: str = MANGA_OCR_REPO
    revision: str = MANGA_OCR_REVISION


MANGA_OCR_FILES: Tuple[ModelFile, ...] = (
    ModelFile(
        ENCODER_FILE,
        "onnx/encoder_model_fp16.onnx",
        171_851_891,
        "1a6a57bc3608195c4577b13ac3aadab810dce42fa22c5a3acf0570bffc013b60",
    ),
    ModelFile(
        DECODER_FILE,
        "onnx/decoder_model_int8.onnx",
        29_627_936,
        "2e7177d2b0a59f1c612b694ed70c13971bee765cc2b2bc7bc9376e4753652f27",
    ),
    ModelFile(
        CONFIG_FILE,
        "config.json",
        75_028,
        "0b45d5253cef67122d2d35b32408ccffa406a87d560e2c3b7ddec3a987863f2e",
    ),
    ModelFile(
        GENERATION_CONFIG_FILE,
        "generation_config.json",
        261,
        "faa07a187c72b147a39f660993bcbdbb85c5fc1a86f8a557d5487945285648c0",
    ),
    ModelFile(
        PREPROCESSOR_CONFIG_FILE,
        "preprocessor_config.json",
        351,
        "445c77049d082004aa07593344d5e1f521f1198228bf586196f70ce7ae021414",
    ),
    ModelFile(
        VOCAB_FILE,
        "vocab.txt",
        24_072,
        "344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d",
        MANGA_OCR_TOKENIZER_REPO,
        MANGA_OCR_TOKENIZER_REVISION,
    ),
    ModelFile(
        TOKENIZER_CONFIG_FILE,
        "tokenizer_config.json",
        486,
        "d775ad1deac162dc56b84e9b8638f95ed8a1f263d0f56f4f40834e26e205e266",
        MANGA_OCR_TOKENIZER_REPO,
        MANGA_OCR_TOKENIZER_REVISION,
    ),
)

# Serialises downloads so two callers (startup check + a UI "download now")
# never stream into the same ``.part`` file at once.
_DOWNLOAD_LOCK = threading.Lock()


def download_url(remote_path: str, repo: str = MANGA_OCR_REPO, revision: str = MANGA_OCR_REVISION) -> str:
    """Resolve URL of ``remote_path`` at commit ``revision`` of repo ``repo``."""
    return HF_RESOLVE_URL.format(repo=repo, revision=revision, path=remote_path)


def manga_ocr_dir(models_dir: Union[str, Path]) -> Path:
    """Directory holding the manga-ocr files under ``models_dir``."""
    return Path(models_dir) / MANGA_OCR_SUBDIR


def _is_present(target: Path, spec: ModelFile) -> bool:
    path = target / spec.name
    return path.is_file() and path.stat().st_size > 0


def missing_files(
    models_dir: Union[str, Path], files: Sequence[ModelFile] = MANGA_OCR_FILES
) -> List[ModelFile]:
    """The entries of ``files`` not yet present (non-empty) under ``models_dir``."""
    target = manga_ocr_dir(models_dir)
    return [spec for spec in files if not _is_present(target, spec)]


def models_ready(models_dir: Union[str, Path], files: Sequence[ModelFile] = MANGA_OCR_FILES) -> bool:
    """True when every manga-ocr file is present under ``models_dir``."""
    return not missing_files(models_dir, files)


def _validate_name(name: str) -> None:
    """Refuse names that could escape the target directory."""
    if not name or name in (".", "..") or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"invalid model file name {name!r}")
    if os.path.isabs(name) or name != os.path.basename(name):
        raise ValueError(f"invalid model file name {name!r}")


def _verify(path: Path, spec: ModelFile, digest: _Digest) -> None:
    """Raise :class:`ModelDownloadError` unless ``path`` matches ``spec``."""
    actual_size = path.stat().st_size
    if actual_size != spec.size:
        raise ModelDownloadError(
            f"{spec.name}: size mismatch (expected {spec.size} bytes, got {actual_size})"
        )
    if spec.sha256 is not None and digest.hexdigest() != spec.sha256.lower():
        raise ModelDownloadError(f"{spec.name}: sha256 mismatch (expected {spec.sha256})")


def _stream_to_part(
    response: object,
    part: Path,
    spec: ModelFile,
    progress: Optional[ProgressCallback],
    cancel: Optional[threading.Event],
) -> _Digest:
    digest = hashlib.sha256()
    received = 0
    last_pct = -1
    with open(part, "wb") as fh:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):  # type: ignore[attr-defined]
            if cancel is not None and cancel.is_set():
                raise ModelDownloadError(f"{spec.name}: download cancelled")
            if not chunk:
                continue
            fh.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            if spec.size and received > spec.size:
                # A lying Content-Length / endless stream must not fill the disk: the size is
                # pinned next to the sha256, so anything longer is wrong by definition.
                raise ModelDownloadError(f"{spec.name}: server sent more than the expected {spec.size} bytes")
            pct = min(100, int(100 * received / spec.size)) if spec.size else 100
            if progress is not None and pct != last_pct:
                progress(spec.name, pct)
                last_pct = pct
    if progress is not None and last_pct != 100:
        progress(spec.name, 100)
    return digest


def _download_one(
    session: object,
    target: Path,
    spec: ModelFile,
    progress: Optional[ProgressCallback],
    cancel: Optional[threading.Event],
) -> None:
    """Fetch ``spec`` into ``target``, verifying before the atomic rename."""
    import requests

    _validate_name(spec.name)
    final = target / spec.name
    part = target / (spec.name + PART_SUFFIX)
    url = download_url(spec.remote_path, spec.repo, spec.revision)
    log.info("downloading %s (%d bytes) from %s", spec.name, spec.size, url)
    try:
        response = session.get(url, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=True)  # type: ignore[attr-defined]
        try:
            response.raise_for_status()
            digest = _stream_to_part(response, part, spec, progress, cancel)
        finally:
            response.close()
        _verify(part, spec, digest)
        os.replace(part, final)
    except (requests.RequestException, OSError) as exc:
        _remove_quietly(part)
        raise ModelDownloadError(f"{spec.name}: download failed ({exc})") from exc
    except ModelDownloadError:
        _remove_quietly(part)
        raise


# Public names for the quality-renderer store (render/quality_models.py), which downloads its
# files with exactly this streaming / verifying / atomic-rename logic.
download_one = _download_one
validate_name = _validate_name


def _remove_quietly(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:  # pragma: no cover - best effort cleanup
        log.warning("could not remove partial download %s: %s", path, exc)


def ensure_models(
    models_dir: Union[str, Path],
    *,
    files: Sequence[ModelFile] = MANGA_OCR_FILES,
    progress: Optional[ProgressCallback] = None,
    session: Optional[object] = None,
    cancel: Optional[threading.Event] = None,
) -> Path:
    """Download every missing entry of ``files`` and return the manga-ocr dir.

    ``progress(filename, percent)`` is called as each file streams in.
    ``session`` is any object with a ``requests.Session``-like ``get``;
    the default is a fresh ``requests.Session``.  Setting ``cancel`` aborts
    between chunks with ``ModelDownloadError("... cancelled")``.  Concurrent
    calls are serialised by a process-wide lock (a second caller waits, then
    finds the files present); every failure surfaces as
    :class:`ModelDownloadError` (or ``ValueError`` for a bad file name).
    """
    for spec in files:
        _validate_name(spec.name)
    target = manga_ocr_dir(models_dir)
    if not missing_files(models_dir, files):
        return target
    with _DOWNLOAD_LOCK:
        todo = missing_files(models_dir, files)  # another caller may have finished
        if not todo:
            return target
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ModelDownloadError(f"cannot create {target} ({exc})") from exc
        owned: Optional[object] = None
        if session is None:
            import requests

            owned = session = requests.Session()
        try:
            for spec in todo:
                _download_one(session, target, spec, progress, cancel)
        finally:
            if owned is not None:
                owned.close()  # type: ignore[attr-defined]
    log.info("manga-ocr models ready in %s", target)
    return target
