"""Offline translation with Argos Translate packages run directly on CTranslate2.

An Argos package (``*.argosmodel``) is a zip whose single top-level directory
holds ``metadata.json`` (``from_code``/``to_code``), a CTranslate2 ``model/``
directory and a ``sentencepiece.model``.  We load those two artefacts
ourselves, so the heavy ``argostranslate``/``stanza`` stack is never imported.

``glasstranslate.core.gpu.prepare_cuda_path()`` must run before ``import
ctranslate2`` for the pip-installed CUDA runtime to be found, which is why the
import is done lazily inside :func:`_ctranslate2`.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..core import gpu
from ..core.interfaces import Translator

log = logging.getLogger(__name__)

ARGOSPM_INDEX_URL = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"

# argos-net.com answers 403 to the default urllib agent; look like a browser.
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

# Inputs longer than this many SentencePiece tokens are split on sentence
# boundaries so every chunk stays well inside ``max_decoding_length``.
MAX_CHUNK_TOKENS = 160
_MAX_DECODING_LENGTH = 256
_BEAM_SIZE = 2
_DOWNLOAD_ATTEMPTS = 6
_DOWNLOAD_BACKOFF_S = 5.0

# Sentence boundaries: Western terminators followed by whitespace, or CJK
# full-width terminators (zero-width split), or line breaks.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|(?<=[。！？；])|\n+")
# Languages written without inter-word spaces; chunks are re-joined without one.
_NO_SPACE_LANGS = frozenset({"ja", "zh", "zt", "th"})

Pair = Tuple[str, str]
ProgressCallback = Callable[[int, Optional[int]], None]

# Sugoi v4 ja->en, CTranslate2 conversion (entai2965/sugoi-v4-ja-en-ctranslate2).
SUGOI_REPO_URL = "https://huggingface.co/entai2965/sugoi-v4-ja-en-ctranslate2/resolve/main"
SUGOI_DIR_NAME = "sugoi-v4-ja-en"
SUGOI_FILES = (
    "config.json",
    "model.bin",
    "shared_vocabulary.json",
    "spm/spm.ja.nopretok.model",
    "spm/spm.en.nopretok.model",
)


def _ctranslate2() -> Any:
    """Import ctranslate2 after the CUDA DLL directories are registered."""
    gpu.prepare_cuda_path()
    import ctranslate2  # noqa: WPS433 - deliberate lazy import

    return ctranslate2


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into non-empty sentence-like pieces (whitespace-stripped)."""
    return [piece.strip() for piece in _SENTENCE_SPLIT.split(text) if piece and piece.strip()]


@dataclass
class ArgosPackage:
    """One installed language pair, either extracted or still an archive.

    Two on-disk layouts are recognised:

    * **Argos package** - ``metadata.json`` + ``model/`` + ``sentencepiece.model``
      (what ``*.argosmodel`` archives extract to).
    * **Plain CTranslate2 directory** - ``model.bin`` + ``config.json`` next to a
      hand-written ``metadata.json`` naming ``from_code``/``to_code`` and,
      optionally, ``source_spm``/``target_spm`` (relative paths to the
      SentencePiece models; default ``sentencepiece.model`` for both).  This is
      how Hugging Face CTranslate2 conversions such as Sugoi are installed.
    """

    from_code: str
    to_code: str
    directory: Optional[Path]  # extracted package dir, once available
    archive: Optional[Path]  # the ``*.argosmodel`` zip, if any
    model_subdir: str = "model"
    source_spm: str = "sentencepiece.model"
    target_spm: str = "sentencepiece.model"
    priority: int = 0  # higher wins when several packages serve the same pair

    @property
    def pair(self) -> Pair:
        return (self.from_code, self.to_code)


@dataclass
class _LoadedPair:
    translator: Any  # ctranslate2.Translator
    source_tokenizer: Any  # sentencepiece.SentencePieceProcessor
    target_tokenizer: Any


def _read_sugoi_dir(directory: Path) -> Optional[ArgosPackage]:
    """Recognise the published Sugoi CTranslate2 conversion layout - a plain
    CTranslate2 directory with no ``metadata.json``, identified by
    ``model.bin`` plus both SentencePiece models.  Direction is fixed ja->en.
    (:func:`download_sugoi` writes a ``metadata.json`` for its own installs;
    this path exists for a manually placed Hugging Face checkout.)"""
    src_spm = directory / "spm" / "spm.ja.nopretok.model"
    tgt_spm = directory / "spm" / "spm.en.nopretok.model"
    if not ((directory / "model.bin").is_file() and src_spm.is_file() and tgt_spm.is_file()):
        return None
    return ArgosPackage("ja", "en", directory, None, ".", "spm/spm.ja.nopretok.model", "spm/spm.en.nopretok.model", 10)


def _spm_inside(directory: Path, rel: str) -> bool:
    """True when ``rel`` resolves to a file inside ``directory`` (metadata.json
    is untrusted: absolute paths and ``..`` must not escape the package)."""
    try:
        candidate = (directory / rel).resolve()
        return candidate.is_file() and candidate.is_relative_to(directory.resolve())
    except OSError:
        return False


def _read_metadata_dir(directory: Path) -> Optional[ArgosPackage]:
    meta_path = directory / "metadata.json"
    if not meta_path.is_file():
        return _read_sugoi_dir(directory)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        from_code, to_code = str(meta["from_code"]), str(meta["to_code"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("Ignoring package dir %s: %s", directory, exc)
        return None
    priority = int(meta.get("priority", 0)) if isinstance(meta.get("priority", 0), (int, float)) else 0
    if (directory / "model").is_dir() and (directory / "sentencepiece.model").is_file():
        return ArgosPackage(from_code, to_code, directory, None, priority=priority)
    if (directory / "model.bin").is_file():
        src_spm = str(meta.get("source_spm", "sentencepiece.model"))
        tgt_spm = str(meta.get("target_spm", src_spm))
        if not (_spm_inside(directory, src_spm) and _spm_inside(directory, tgt_spm)):
            log.warning("Ignoring %s: SentencePiece model(s) %s / %s missing or outside the package", directory, src_spm, tgt_spm)
            return None
        return ArgosPackage(from_code, to_code, directory, None, ".", src_spm, tgt_spm, priority)
    return None


def _zip_top_dir(zf: zipfile.ZipFile) -> Optional[str]:
    """Name of the single top-level directory inside an Argos archive."""
    tops = {name.split("/", 1)[0] for name in zf.namelist() if "/" in name}
    return tops.pop() if len(tops) == 1 else None


def _read_metadata_zip(archive: Path) -> Optional[ArgosPackage]:
    try:
        with zipfile.ZipFile(archive) as zf:
            top = _zip_top_dir(zf)
            if top is None:
                return None
            meta = json.loads(zf.read(f"{top}/metadata.json").decode("utf-8"))
        return ArgosPackage(str(meta["from_code"]), str(meta["to_code"]), None, archive)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        log.warning("Ignoring archive %s: %s", archive, exc)
        return None


def _extract_archive(archive: Path, dest_root: Path) -> Path:
    """Extract ``archive`` into ``dest_root`` and return the package directory.
    Refuses entries that would escape ``dest_root``."""
    dest_root = dest_root.resolve()
    with zipfile.ZipFile(archive) as zf:
        top = _zip_top_dir(zf)
        if top is None:
            raise ValueError(f"{archive.name} has no single top-level directory")
        target = dest_root / top
        for info in zf.infolist():
            if not (dest_root / info.filename).resolve().is_relative_to(dest_root):
                raise ValueError(f"{archive.name} contains an unsafe path: {info.filename}")
        if target.exists():
            shutil.rmtree(target)
        zf.extractall(dest_root)
    return target


class ArgosCT2Translator(Translator):
    """Translator backed by locally installed Argos packages.

    Packages are discovered in ``models_dir`` at construction (and after every
    :meth:`download_package`); models are loaded on first use of a pair.  When a
    pair has no package but both ``(src, "en")`` and ``("en", tgt)`` do, the
    text is pivoted through English.
    """

    name = "argos"
    offline = True

    def __init__(
        self,
        models_dir: Union[str, Path],
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        self.models_dir = Path(models_dir)
        self.compute_type = compute_type
        self.requested_device = device
        self.device = self._resolve_device(device)
        self._packages: Dict[Pair, ArgosPackage] = {}
        self._loaded: Dict[Pair, _LoadedPair] = {}
        self._index_cache: Optional[List[dict]] = None
        self._lock = threading.RLock()
        self.rescan()

    # --------------------------------------------------------------- setup
    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            return "cuda" if _ctranslate2().get_cuda_device_count() > 0 else "cpu"
        except Exception as exc:  # pragma: no cover - depends on local CUDA install
            log.info("CUDA probe failed (%s); using CPU", exc)
            return "cpu"

    def rescan(self) -> None:
        """Re-read ``models_dir``; extracted directories win over archives.

        ``models_dir`` itself is also tried as a package directory, so a user
        who points it straight at a single hand-assembled model (e.g. a Sugoi
        conversion with no parent folder) is still discovered."""
        found: Dict[Pair, ArgosPackage] = {}
        if self.models_dir.is_dir():
            for archive in sorted(self.models_dir.glob("*.argosmodel")):
                pkg = _read_metadata_zip(archive)
                if pkg is not None:
                    found[pkg.pair] = pkg
            for child in sorted(self.models_dir.iterdir()):
                if child.is_dir():
                    pkg = _read_metadata_dir(child)
                    if pkg is not None and (pkg.pair not in found or pkg.priority >= found[pkg.pair].priority):
                        found[pkg.pair] = pkg
            self_pkg = _read_metadata_dir(self.models_dir)
            if self_pkg is not None and (self_pkg.pair not in found or self_pkg.priority >= found[self_pkg.pair].priority):
                found[self_pkg.pair] = self_pkg
        with self._lock:
            self._packages = found
        log.info("Argos packages in %s: %s", self.models_dir, sorted(found) or "none")

    def installed_packages(self) -> List[ArgosPackage]:
        with self._lock:
            return list(self._packages.values())

    # -------------------------------------------------------- Translator API
    def supported_pairs(self) -> Iterable[Pair]:
        with self._lock:
            direct = set(self._packages)
        pivot = {
            (s, t)
            for (s, e) in direct
            if e == "en"
            for (e2, t) in direct
            if e2 == "en" and s != t
        }
        return direct | pivot

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str) -> List[str]:
        texts = list(texts)
        if src == tgt or not any(t.strip() for t in texts):
            return texts
        try:
            route = self._route(src, tgt)
            if route is None:
                log.warning("No Argos package for %s->%s (and no English pivot)", src, tgt)
                return texts
            out = texts
            for step_src, step_tgt in route:
                out = self._translate_direct(out, step_src, step_tgt)
            return out
        except Exception as exc:
            log.error("Argos translation %s->%s failed: %s", src, tgt, exc)
            return texts

    def close(self) -> None:
        with self._lock:
            for pair in self._loaded.values():
                unload = getattr(pair.translator, "unload_model", None)
                if callable(unload):
                    unload()
            self._loaded.clear()

    # -------------------------------------------------------------- routing
    def _route(self, src: str, tgt: str) -> Optional[List[Pair]]:
        with self._lock:
            if (src, tgt) in self._packages:
                return [(src, tgt)]
            if (src, "en") in self._packages and ("en", tgt) in self._packages:
                return [(src, "en"), ("en", tgt)]
        return None

    def _get_loaded(self, pair: Pair) -> _LoadedPair:
        with self._lock:
            loaded = self._loaded.get(pair)
            if loaded is not None:
                return loaded
            pkg = self._packages[pair]
            if pkg.directory is None:
                assert pkg.archive is not None
                pkg.directory = _extract_archive(pkg.archive, self.models_dir)
                log.info("Extracted %s -> %s", pkg.archive.name, pkg.directory)
            loaded = self._load_models(pkg)
            self._loaded[pair] = loaded
            return loaded

    def _load_models(self, pkg: ArgosPackage) -> _LoadedPair:
        import sentencepiece  # lazy: keep import cost off the UI thread's startup

        assert pkg.directory is not None
        directory = pkg.directory
        ct2 = _ctranslate2()
        model_dir = str((directory / pkg.model_subdir).resolve())
        try:
            translator = ct2.Translator(model_dir, device=self.device, compute_type=self.compute_type)
        except Exception as exc:
            if self.device == "cpu":
                raise
            log.warning("ctranslate2 on %s failed (%s); falling back to CPU", self.device, exc)
            self.device = "cpu"
            translator = ct2.Translator(model_dir, device="cpu", compute_type=self.compute_type)
        source_tok = sentencepiece.SentencePieceProcessor(model_file=str(directory / pkg.source_spm))
        if pkg.target_spm == pkg.source_spm:
            target_tok = source_tok
        else:
            target_tok = sentencepiece.SentencePieceProcessor(model_file=str(directory / pkg.target_spm))
        log.info("Loaded translation model %s on %s", directory.name, self.device)
        return _LoadedPair(translator, source_tok, target_tok)

    # ---------------------------------------------------------- translation
    def _chunk_tokens(self, tokenizer: Any, text: str) -> List[List[str]]:
        """Tokenise ``text`` into one or more token lists of at most
        :data:`MAX_CHUNK_TOKENS` each, cutting on sentence boundaries first and
        hard-splitting only sentences that are themselves too long."""
        tokens = tokenizer.encode(text, out_type=str)
        if len(tokens) <= MAX_CHUNK_TOKENS:
            return [tokens]
        chunks: List[List[str]] = []
        current: List[str] = []
        for sentence in split_sentences(text):
            sent_tokens = tokenizer.encode(sentence, out_type=str)
            if len(current) + len(sent_tokens) > MAX_CHUNK_TOKENS and current:
                chunks.append(current)
                current = []
            if len(sent_tokens) > MAX_CHUNK_TOKENS:
                chunks.extend(
                    sent_tokens[i : i + MAX_CHUNK_TOKENS] for i in range(0, len(sent_tokens), MAX_CHUNK_TOKENS)
                )
                continue
            current.extend(sent_tokens)
        if current:
            chunks.append(current)
        return chunks

    def _translate_direct(self, texts: List[str], src: str, tgt: str) -> List[str]:
        """Translate with a single installed package; whitespace-only inputs
        are passed through untouched."""
        loaded = self._get_loaded((src, tgt))
        plan: List[Tuple[int, int]] = []  # (text index, number of chunks)
        batch: List[List[str]] = []
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            chunks = self._chunk_tokens(loaded.source_tokenizer, text.strip())
            plan.append((i, len(chunks)))
            batch.extend(chunks)
        if not batch:
            return texts
        results = loaded.translator.translate_batch(
            batch, beam_size=_BEAM_SIZE, max_decoding_length=_MAX_DECODING_LENGTH
        )
        decoded = [loaded.target_tokenizer.decode(r.hypotheses[0]) for r in results]
        joiner = "" if tgt in _NO_SPACE_LANGS else " "
        out = list(texts)
        cursor = 0
        for i, count in plan:
            out[i] = joiner.join(part.strip() for part in decoded[cursor : cursor + count]).strip()
            cursor += count
        return out

    # ------------------------------------------------------- Sugoi (ja->en)
    def download_sugoi(self, progress_cb: Optional[ProgressCallback] = None) -> Path:
        """Install the Sugoi v4 Japanese->English model (a CTranslate2
        conversion published on Hugging Face) into ``models_dir``.

        Sugoi is trained on game/visual-novel dialogue and handles the short,
        colloquial lines found in manga far better than the Argos ja->en
        package, so it is registered with a higher priority for that pair.
        """
        target = self.models_dir / SUGOI_DIR_NAME
        target.mkdir(parents=True, exist_ok=True)
        (target / "spm").mkdir(exist_ok=True)
        done = 0
        for rel in SUGOI_FILES:
            dest = target / rel
            if dest.is_file() and dest.stat().st_size > 0:
                continue
            self._download_file(f"{SUGOI_REPO_URL}/{rel}", dest, progress_cb)
            done += 1
        meta = {
            "from_code": "ja",
            "to_code": "en",
            "from_name": "Japanese",
            "to_name": "English",
            "source_spm": "spm/spm.ja.nopretok.model",
            "target_spm": "spm/spm.en.nopretok.model",
            "priority": 10,
            "source": SUGOI_REPO_URL,
        }
        (target / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info("Sugoi v4 installed in %s (%d files downloaded)", target, done)
        self.rescan()
        return target

    # ------------------------------------------------------- argospm index
    def available_packages(self, refresh: bool = False) -> List[dict]:
        """Packages listed in the argospm index (``from_code``, ``to_code``,
        ``package_version``, ``links``, names...).  Cached after the first
        successful fetch; returns ``[]`` when offline."""
        with self._lock:
            if self._index_cache is not None and not refresh:
                return list(self._index_cache)
        try:
            req = urllib.request.Request(ARGOSPM_INDEX_URL, headers=_HTTP_HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                index = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            log.warning("Could not fetch argospm index: %s", exc)
            return []
        packages = [
            entry
            for entry in index
            if isinstance(entry, dict) and entry.get("from_code") and entry.get("to_code") and entry.get("links")
        ]
        with self._lock:
            self._index_cache = packages
        return list(packages)

    def download_package(
        self,
        from_code: str,
        to_code: str,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Path:
        """Download and extract the index package for ``(from_code, to_code)``
        into ``models_dir``; returns the extracted package directory.

        ``progress_cb(bytes_done, total_or_None)`` is invoked as data arrives.
        Raises ``LookupError`` if the index has no such pair and ``OSError`` if
        every mirror fails.
        """
        entry = next(
            (p for p in self.available_packages() if p["from_code"] == from_code and p["to_code"] == to_code),
            None,
        )
        if entry is None:
            raise LookupError(f"argospm index has no package for {from_code}->{to_code}")
        version = str(entry.get("package_version", "")).replace(".", "_")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        archive = self.models_dir / f"translate-{from_code}_{to_code}-{version}.argosmodel"
        errors: List[str] = []
        for url in entry["links"]:
            try:
                self._download_file(str(url), archive, progress_cb)
                break
            except (urllib.error.URLError, OSError) as exc:
                errors.append(f"{url}: {exc}")
                log.warning("Download failed from %s: %s", url, exc)
        else:
            raise OSError("all mirrors failed: " + "; ".join(errors))
        directory = _extract_archive(archive, self.models_dir)
        self.rescan()
        return directory

    @staticmethod
    def _download_file(url: str, dest: Path, progress_cb: Optional[ProgressCallback]) -> None:
        tmp = dest.with_name(dest.name + ".part")
        req = urllib.request.Request(url, headers=_HTTP_HEADERS)
        for attempt in range(_DOWNLOAD_ATTEMPTS):
            try:
                resp = urllib.request.urlopen(req, timeout=60)
                break
            except urllib.error.HTTPError as exc:
                # Hugging Face and argos-net rate-limit bursts; back off and retry.
                if exc.code not in (429, 502, 503) or attempt == _DOWNLOAD_ATTEMPTS - 1:
                    raise
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else _DOWNLOAD_BACKOFF_S * (2**attempt)
                log.warning("%s -> HTTP %d, retrying in %.0fs", url, exc.code, delay)
                time.sleep(delay)
        with resp, tmp.open("wb") as fh:
            length = resp.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            done = 0
            while True:
                block = resp.read(1 << 16)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if progress_cb is not None:
                    progress_cb(done, total)
        tmp.replace(dest)
