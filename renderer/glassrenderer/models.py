"""Model store for the quality renderer's generative stages.

The sidecar never downloads anything itself (see ``renderer/PROTOCOL.md``): the
app fetches the files listed in :data:`QUALITY_FILES` through the Engines page
and the stages then load them from ``<models_dir>/quality`` with the network
off.  The table is data, not code — ``renderer/glassrenderer/models.json``,
copied byte for byte to ``glasstranslate/render/quality_models.json`` so both
sides agree on every digest.

:func:`ensure_models` mirrors :mod:`glasstranslate.ocr.models`: it streams each
file into ``<name>.part``, refuses a stream longer than the pinned size,
verifies the SHA-256 and only then renames it into place.  That verification is
load-bearing rather than cosmetic — ``torch.jit.load`` deserialises code, so the
LaMa checkpoint must be proven byte-exact before it is ever opened
(:func:`verify_file` re-checks it at load time).

This module imports nothing from the generative stack; the main (torch-free)
venv can import it to reason about the same table.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

log = logging.getLogger(__name__)

HF_RESOLVE_URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"

QUALITY_SUBDIR = "quality"
MODELS_TABLE_PATH = Path(__file__).with_name("models.json")

# Model groups, as reported by ``/health`` (see PROTOCOL.md).
LAMA = "lama"
SDXL = "sdxl"
CONTROLNET = "controlnet"

# Directories under ``<models_dir>/quality``.
LAMA_SUBDIR = "lama"
SDXL_SUBDIR = "sdxl"
SDXL_CONFIG_SUBDIR = "sdxl-base-config"
CONTROLNET_SUBDIR = "controlnet"
VAE_SUBDIR = "vae"

LAMA_WEIGHTS = "anime_manga_lama.pt"
SDXL_CHECKPOINT = "Illustrious-XL-v1.0.safetensors"

# The SDXL group is everything the img2img pipeline needs to build itself.
_GROUP_BY_ROOT: Dict[str, str] = {
    LAMA_SUBDIR: LAMA,
    SDXL_SUBDIR: SDXL,
    SDXL_CONFIG_SUBDIR: SDXL,
    VAE_SUBDIR: SDXL,
    CONTROLNET_SUBDIR: CONTROLNET,
}
GROUPS: Tuple[str, ...] = (LAMA, SDXL, CONTROLNET)

CHUNK_SIZE = 1 << 22  # 4 MiB streaming chunks (multi-GB files)
REQUEST_TIMEOUT = (10.0, 120.0)  # connect, read (seconds)
PART_SUFFIX = ".part"
HASH_CHUNK = 1 << 20

ProgressCallback = Callable[[str, int], None]


class ModelDownloadError(RuntimeError):
    """A model file could not be downloaded or failed verification."""


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


@dataclass(frozen=True)
class ModelFile:
    """One pinned file of the quality-renderer bundle.

    ``name`` is the local basename (it may differ from ``remote_path``: the
    ControlNet's ``diffusion_pytorch_model_promax.safetensors`` lands as the
    plain ``diffusion_pytorch_model.safetensors`` diffusers expects).
    ``subdir`` is the '/'-separated directory under ``<models_dir>/quality``.
    ``sha256`` is the Git-LFS object id for the big blobs and the digest of the
    fetched bytes for the small plain-git files; ``None`` means size only.
    ``licence`` records the grant this file ships under (see MODELS.md).
    """

    name: str
    remote_path: str
    size: int
    sha256: Optional[str]
    repo: str
    revision: str
    licence: str
    subdir: str


def load_table(path: Union[str, Path, None] = None) -> Tuple[ModelFile, ...]:
    """Parse ``models.json`` (default :data:`MODELS_TABLE_PATH`) into the table."""
    source = Path(path) if path is not None else MODELS_TABLE_PATH
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("version") != 1:
        raise ValueError(f"{source}: unsupported table version {raw.get('version')!r}")
    return tuple(
        ModelFile(
            name=entry["name"],
            remote_path=entry["remote_path"],
            size=int(entry["size"]),
            sha256=entry.get("sha256"),
            repo=entry["repo"],
            revision=entry["revision"],
            licence=entry["licence"],
            subdir=entry["subdir"],
        )
        for entry in raw["files"]
    )


QUALITY_FILES: Tuple[ModelFile, ...] = load_table()

# Serialises downloads so two callers never stream into the same ``.part``.
_DOWNLOAD_LOCK = threading.Lock()


def download_url(remote_path: str, repo: str, revision: str) -> str:
    """Resolve URL of ``remote_path`` at commit ``revision`` of repo ``repo``."""
    return HF_RESOLVE_URL.format(repo=repo, revision=revision, path=remote_path)


def quality_dir(models_dir: Union[str, Path]) -> Path:
    """Directory holding the quality-renderer files under ``models_dir``.

    The app owns ``<models_dir>/quality`` and may launch the sidecar with
    either the parent (``<project>/models``) or the store itself
    (``...\\models\\quality``, as ``renderer/PROTOCOL.md`` spells it), so a path
    that is already the store is accepted unchanged.
    """
    path = Path(models_dir)
    return path if path.name == QUALITY_SUBDIR else path / QUALITY_SUBDIR


def group_of(spec: ModelFile) -> str:
    """The ``/health`` group (:data:`LAMA` / :data:`SDXL` / :data:`CONTROLNET`)."""
    root = spec.subdir.split("/", 1)[0]
    try:
        return _GROUP_BY_ROOT[root]
    except KeyError:
        raise ValueError(f"{spec.name}: unknown subdir root {root!r}") from None


def total_bytes(files: Sequence[ModelFile] = QUALITY_FILES) -> int:
    """Sum of the pinned sizes — what a first run has to download."""
    return sum(spec.size for spec in files)


def _validate_name(name: str) -> None:
    """Refuse names that could escape the target directory."""
    if not name or name in (".", "..") or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"invalid model file name {name!r}")
    if os.path.isabs(name) or name != os.path.basename(name):
        raise ValueError(f"invalid model file name {name!r}")


def _validate_subdir(subdir: str) -> None:
    """Refuse subdirectories that are absolute or climb out of the store."""
    if not subdir or subdir.startswith(("/", "\\")) or "\\" in subdir or os.path.isabs(subdir):
        raise ValueError(f"invalid model subdir {subdir!r}")
    for part in subdir.split("/"):
        if not part or part in (".", ".."):
            raise ValueError(f"invalid model subdir {subdir!r}")


def _validate(spec: ModelFile) -> None:
    _validate_name(spec.name)
    _validate_subdir(spec.subdir)


def file_path(models_dir: Union[str, Path], spec: ModelFile) -> Path:
    """Absolute path of ``spec`` inside ``models_dir`` (validated)."""
    _validate(spec)
    return quality_dir(models_dir).joinpath(*spec.subdir.split("/")) / spec.name


def _is_present(models_dir: Union[str, Path], spec: ModelFile) -> bool:
    path = file_path(models_dir, spec)
    return path.is_file() and path.stat().st_size == spec.size


def missing_files(
    models_dir: Union[str, Path], files: Sequence[ModelFile] = QUALITY_FILES
) -> List[ModelFile]:
    """The entries of ``files`` not yet present at their pinned size."""
    return [spec for spec in files if not _is_present(models_dir, spec)]


def models_ready(models_dir: Union[str, Path], files: Sequence[ModelFile] = QUALITY_FILES) -> bool:
    """True when every entry of ``files`` is present under ``models_dir``."""
    return not missing_files(models_dir, files)


def group_files(group: str, files: Sequence[ModelFile] = QUALITY_FILES) -> List[ModelFile]:
    """Every entry of ``files`` belonging to ``group``."""
    return [spec for spec in files if group_of(spec) == group]


def group_status(
    models_dir: Union[str, Path], files: Sequence[ModelFile] = QUALITY_FILES
) -> Dict[str, str]:
    """``{"lama": "ready" | "missing", "sdxl": ..., "controlnet": ...}``."""
    return {
        group: "ready" if models_ready(models_dir, group_files(group, files)) else "missing"
        for group in GROUPS
    }


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(models_dir: Union[str, Path], spec: ModelFile) -> Path:
    """Re-hash ``spec`` on disk and return its path, or raise.

    Called before ``torch.jit.load`` (and before the safetensors loads) so a
    tampered file in the models dir cannot reach a deserialiser.
    """
    path = file_path(models_dir, spec)
    if not path.is_file():
        raise ModelDownloadError(f"{spec.name}: missing from {path.parent}")
    actual = path.stat().st_size
    if actual != spec.size:
        raise ModelDownloadError(f"{spec.name}: size mismatch (expected {spec.size}, got {actual})")
    if spec.sha256 is not None:
        digest = _sha256_of(path)
        if digest != spec.sha256.lower():
            raise ModelDownloadError(f"{spec.name}: sha256 mismatch (expected {spec.sha256}, got {digest})")
    return path


def _verify_part(part: Path, spec: ModelFile, digest: _Digest) -> None:
    """Raise :class:`ModelDownloadError` unless the streamed bytes match."""
    actual_size = part.stat().st_size
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
                # A lying Content-Length / endless stream must not fill the disk:
                # the size is pinned next to the sha256, so longer is wrong.
                raise ModelDownloadError(
                    f"{spec.name}: server sent more than the expected {spec.size} bytes"
                )
            pct = min(100, int(100 * received / spec.size)) if spec.size else 100
            if progress is not None and pct != last_pct:
                progress(spec.name, pct)
                last_pct = pct
    if progress is not None and last_pct != 100:
        progress(spec.name, 100)
    return digest


def _download_one(
    session: object,
    models_dir: Union[str, Path],
    spec: ModelFile,
    progress: Optional[ProgressCallback],
    cancel: Optional[threading.Event],
) -> None:
    """Fetch ``spec`` into the store, verifying before the atomic rename."""
    import requests

    final = file_path(models_dir, spec)
    part = final.with_name(final.name + PART_SUFFIX)
    url = download_url(spec.remote_path, spec.repo, spec.revision)
    log.info("downloading %s (%d bytes) from %s", spec.name, spec.size, url)
    try:
        final.parent.mkdir(parents=True, exist_ok=True)
        response = session.get(url, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=True)  # type: ignore[attr-defined]
        try:
            response.raise_for_status()
            digest = _stream_to_part(response, part, spec, progress, cancel)
        finally:
            response.close()
        _verify_part(part, spec, digest)
        os.replace(part, final)
    except (requests.RequestException, OSError) as exc:
        _remove_quietly(part)
        raise ModelDownloadError(f"{spec.name}: download failed ({exc})") from exc
    except ModelDownloadError:
        _remove_quietly(part)
        raise


def _remove_quietly(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:  # pragma: no cover - best effort cleanup
        log.warning("could not remove partial download %s: %s", path, exc)


def ensure_models(
    models_dir: Union[str, Path],
    *,
    files: Sequence[ModelFile] = QUALITY_FILES,
    progress: Optional[ProgressCallback] = None,
    session: Optional[object] = None,
    cancel: Optional[threading.Event] = None,
) -> Path:
    """Download every missing entry of ``files`` and return the quality dir.

    ``progress(filename, percent)`` is called as each file streams in.
    ``session`` is any object with a ``requests.Session``-like ``get``; the
    default is a fresh ``requests.Session``.  Setting ``cancel`` aborts between
    chunks with ``ModelDownloadError("... cancelled")``.  Concurrent calls are
    serialised by a process-wide lock (a second caller waits, then finds the
    files present); every failure surfaces as :class:`ModelDownloadError` (or
    ``ValueError`` for a bad name or subdirectory).
    """
    for spec in files:
        _validate(spec)
    target = quality_dir(models_dir)
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
                _download_one(session, models_dir, spec, progress, cancel)
        finally:
            if owned is not None:
                owned.close()  # type: ignore[attr-defined]
    log.info("quality-renderer models ready in %s", target)
    return target
