"""App side of the quality renderer: launch, protocol client and job scheduling.

The generative fill of free text over artwork runs in a **sidecar** process
(``renderer/``, package ``glassrenderer``, its own venv with torch + CUDA).
The main app stays torch-free: this module speaks the JSON-over-HTTP protocol
of ``renderer/PROTOCOL.md`` to it on ``127.0.0.1`` with a per-session token,
and nothing here imports Qt.

Three pieces:

* :class:`QualityClient` - spawns ``<python> -m glassrenderer serve`` (cwd =
  the ``renderer`` directory so the package is importable), waits for the
  ``READY <port>`` line, and offers :meth:`~QualityClient.health` /
  :meth:`~QualityClient.inpaint` / :meth:`~QualityClient.shutdown`.  Every
  response is re-composited against the mask, so a pixel the mask left at 0
  is byte-identical to the input whatever the sidecar returned.  Failures
  surface as :class:`QualityUnavailable` with a one-line reason.
* :func:`panel_jobs` - one job per *panel* holding free text over art: the
  panel crop, the union of the members' eraser masks (dilated by
  :data:`QUALITY_MASK_DILATE`) and a content key, so an unchanged panel is
  never resubmitted.  :func:`composite_block_patch` turns a result crop back
  into one block's ``clean_patch``.
* :class:`QualityScheduler` - a daemon worker running one job at a time,
  deduplicating by content key, dropping queued jobs a newer submission
  supersedes, and backing off for :data:`QUALITY_RETRY_S` after a failure.

The app never depends on any of it: with no sidecar the quick fill of
``render/erase.py`` is what the renderer shows.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import logging
import os
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

from ..config.settings import project_root, user_data_dir
from ..core.types import Rect, SegmentStyle

__all__ = [
    "Job",
    "MemberKey",
    "QUALITY_MASK_DILATE",
    "QUALITY_RETRY_S",
    "QualityClient",
    "QualityScheduler",
    "QualityUnavailable",
    "block_key",
    "composite_block_patch",
    "find_sidecar_python",
    "panel_jobs",
    "sidecar_dir",
]

log = logging.getLogger(__name__)

# --- tunables ---------------------------------------------------------------
QUALITY_MASK_DILATE = 2  # px the eraser's glyph mask is grown before inpainting
QUALITY_RETRY_S = 30.0  # back-off after a sidecar failure before it is tried again
FEATHER_PX = 2  # px of feathering at the mask edge when a result is composited
FALLBACK_PAD_EM = 1.0  # em grown around a block's clean_rect when it has no panel
# The sidecar's largest bucket (renderer/PROTOCOL.md): a panel bigger than this on its long side
# is cropped to a window of this size around its text so the generation keeps full resolution
# (a whole page sent as one panel would be downscaled first).  When the text of one panel does not
# fit such a window, the whole panel goes.
CONTEXT_MAX_PX = 1024
# ...and a panel smaller than this on a side is grown to it around its text (beyond the panel's
# borders when it must): SDXL needs a drawing to continue, not an 80 px crop.
CONTEXT_MIN_PX = 512
# Free text whose quick fill has no ink in the ring ART_RING_PX wide beyond the glyph mask's
# anti-aliased fringe (ART_FRINGE_PX) sits on plain paper: the exact quick fill stays and no job
# is sent.  Measured on the five reference pages: the generative fill wins where there is art to
# continue (before.jpg: LPIPS 0.36 -> 0.31 on the scaffolding block) and invents strokes on paper
# (1ja 記錄: the fringe alone read as 15.6 % "art", 4.1 % once excluded; 0.32 -> 0.33).
QUALITY_MIN_ART = 0.05
ART_RING_PX = 4
ART_FRINGE_PX = 2  # the glyph edge pixels the mask does not cover; never counted as art
ART_INK_DELTA = 60  # grey levels from the block's paper colour that count as ink in the patch
READY_TIMEOUT_FAKE_S = 30.0  # PROTOCOL.md: the READY line comes this fast in fake mode...
READY_TIMEOUT_REAL_S = 120.0  # ...and this slowly with the real stages
REQUEST_TIMEOUT_S = 300.0  # one /inpaint may run a full SDXL pass
HEALTH_TIMEOUT_S = 30.0
WARM_TIMEOUT_S = 900.0  # wait_warm: a cold page cache put the warm-up at 270 s; contention doubles it
WARM_POLL_S = 2.0
SHUTDOWN_TIMEOUT_S = 5.0
MAX_BODY_BYTES = 64 << 20  # PROTOCOL.md's body cap
DONE_KEYS_MAX = 256  # content keys remembered so an unchanged panel is not resubmitted

LOOPBACK = "127.0.0.1"
TOKEN_ENV = "GT_RENDERER_TOKEN"
SIDECAR_PACKAGE = "glassrenderer"
SIDECAR_DIRNAME = "renderer"
_LOG_DEFAULT = object()  # sentinel: "the standard renderer.log"; None means DEVNULL

MemberKey = Tuple[str, int, int, int, int]  # (source text, clean_rect x, y, w, h)


class QualityUnavailable(RuntimeError):
    """The quality renderer could not be used; the message is one line.

    ``transport`` is True when the sidecar process or its socket is gone (the scheduler
    then drops the client and starts a fresh one later) and False for a job the running
    sidecar refused or failed (the warm sidecar is kept).
    """

    def __init__(self, message: str, *, transport: bool = True) -> None:
        super().__init__(message)
        self.transport = transport


class QualityModelsMissing(QualityUnavailable):
    """``/health`` reported model files that are missing or failed to load."""

    def __init__(self, message: str) -> None:
        super().__init__(message, transport=False)


MODEL_OK_STATES = frozenset({"ready", "loading"})
MODELS_RETRY_S = 300.0  # a missing model store does not fix itself within the normal back-off


def _models_unusable(info: Dict[str, Any]) -> str:
    """One line naming the model groups ``/health`` reports as neither ready nor loading, else ''.

    PROTOCOL.md: a client that needs the models reads ``models.*`` before submitting jobs.  A
    body without a ``models`` map (older or test servers) passes."""
    models = info.get("models")
    if not isinstance(models, dict):
        return ""
    bad = sorted(str(key) for key, state in models.items() if str(state) not in MODEL_OK_STATES)
    if not bad:
        return ""
    error = info.get("load_error")
    return f"models not usable: {', '.join(bad)}" + (f" ({_one_line(error)})" if error else "")


def _one_line(value: object) -> str:
    """Collapse an exception / message to a single short line for the status strip."""
    text = " ".join(str(value).split())
    return text[:200] if text else "unknown error"


# --------------------------------------------------------------- interpreter lookup
def _venv_python(root: Path) -> Path:
    if sys.platform == "win32":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _sidecar_roots() -> List[Path]:
    """Candidate ``renderer`` directories: the checkout first, then the user profile."""
    roots: List[Path] = []
    if not getattr(sys, "frozen", False):
        roots.append(project_root() / SIDECAR_DIRNAME)
    roots.append(user_data_dir() / SIDECAR_DIRNAME)
    return roots


def find_sidecar_python(cfg: Any) -> Optional[Path]:
    """The interpreter that can run the sidecar, or None when it is not installed.

    ``cfg.quality_sidecar_python`` wins when it points at an existing file;
    otherwise ``<project>/renderer/.venv`` in a source checkout and
    ``%LOCALAPPDATA%/GlassTranslate/renderer/.venv`` in the frozen build (whose
    project root is a temporary extraction directory).
    """
    explicit = str(getattr(cfg, "quality_sidecar_python", "") or "").strip()
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
    for root in _sidecar_roots():
        candidate = _venv_python(root)
        if candidate.is_file():
            return candidate
    return None


def sidecar_dir(python: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Directory that holds the ``glassrenderer`` package (the sidecar's cwd)."""
    if python is not None:
        for parent in Path(python).resolve().parents:
            if (parent / SIDECAR_PACKAGE).is_dir():
                return parent
    for root in _sidecar_roots():
        if (root / SIDECAR_PACKAGE).is_dir():
            return root
        if root.is_dir():
            return root
    return None


# ------------------------------------------------------------------------ the client
def _png_b64(image: np.ndarray) -> str:
    """PNG-encode ``image`` (BGR or greyscale) and base64 it, as the protocol wants.

    ``cv2.imencode`` writes a BGR array as a correct RGB PNG, so no channel swap
    is needed here.
    """
    ok, buffer = cv2.imencode(".png", np.ascontiguousarray(image))
    if not ok:
        raise QualityUnavailable("could not PNG-encode the panel")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _decode_png(payload: str) -> Optional[np.ndarray]:
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError):
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


class QualityClient:
    """One sidecar process (or, in tests, one already-running server).

    Args:
        python: interpreter of the sidecar venv; None with an injected ``start``.
        models_dir: passed through as ``--models-dir``.
        fake: run the sidecar's numpy stand-ins (no torch, no GPU).
        token: the per-session ``GT_RENDERER_TOKEN``; a fresh one by default.
        start: test hook returning a port instead of spawning a process.
        cwd: the sidecar's working directory (default :func:`sidecar_dir`).
        log_path: file the sidecar's stderr is appended to; None = DEVNULL.
    """

    def __init__(
        self,
        python: Optional[Union[str, Path]],
        models_dir: Union[str, Path],
        *,
        fake: bool = False,
        token: Optional[str] = None,
        start: Optional[Callable[[], int]] = None,
        cwd: Optional[Union[str, Path]] = None,
        log_path: Any = _LOG_DEFAULT,
        ready_timeout: Optional[float] = None,
        timeout: float = REQUEST_TIMEOUT_S,
    ) -> None:
        self._python = None if python is None else Path(python)
        # The sidecar runs with cwd = renderer/, so a relative models path must be
        # anchored to our cwd before it crosses the process boundary.
        self._models_dir = Path(models_dir).absolute()
        self._fake = bool(fake)
        self._token = token or secrets.token_hex(32)
        self._start_hook = start
        self._cwd = None if cwd is None else Path(cwd)
        self._log_path = log_path
        self._ready_timeout = float(
            ready_timeout if ready_timeout is not None
            else (READY_TIMEOUT_FAKE_S if fake else READY_TIMEOUT_REAL_S)
        )
        self._timeout = float(timeout)
        self._port: Optional[int] = None
        self._proc: Optional[subprocess.Popen] = None
        self._log_file: Optional[Any] = None
        self._closed = False
        self._jobs = 0
        self._job_tag = secrets.token_hex(3)  # per-client log correlation id, unrelated to the token
        self.last_timings: Dict[str, float] = {}

    # ------------------------------------------------------------------ lifecycle
    @property
    def port(self) -> Optional[int]:
        return self._port

    @property
    def running(self) -> bool:
        return self._port is not None and not self._closed

    def start(self) -> int:
        """Launch the sidecar (or call the injected hook) and return its port."""
        if self._port is not None:
            return self._port
        if self._closed:
            raise QualityUnavailable("the quality sidecar was shut down")
        if self._start_hook is not None:
            self._port = int(self._start_hook())
            return self._port
        self._port = self._spawn()
        return self._port

    def _open_log(self) -> Any:
        if self._log_path is None:
            return subprocess.DEVNULL
        path = self._log_path
        if path is _LOG_DEFAULT:
            path = user_data_dir() / "logs" / "renderer.log"
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(path, "ab")
            return self._log_file
        except OSError as exc:  # pragma: no cover - unwritable profile
            log.warning("cannot open the renderer log (%s); sidecar stderr is discarded", exc)
            return subprocess.DEVNULL

    def _spawn(self) -> int:
        if self._python is None:
            raise QualityUnavailable("no sidecar interpreter (run renderer\\install.bat)")
        cmd = [str(self._python), "-m", SIDECAR_PACKAGE, "serve", "--models-dir", str(self._models_dir)]
        if self._fake:
            cmd.append("--fake")
        env = dict(os.environ)
        env[TOKEN_ENV] = self._token  # never on the command line: process lists are readable
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        # No data leaves the machine: the model files are pinned and verified locally, so the
        # Hub is never consulted and remote code is never fetched.
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["DIFFUSERS_DISABLE_REMOTE_CODE"] = "true"
        cwd = self._cwd if self._cwd is not None else sidecar_dir(self._python)
        kwargs: Dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                cmd,
                cwd=str(cwd) if cwd is not None and Path(cwd).is_dir() else None,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._open_log(),
                text=True,
                encoding="utf-8",
                errors="replace",
                **kwargs,
            )
        except OSError as exc:
            raise QualityUnavailable(f"cannot start the quality sidecar: {_one_line(exc)}") from exc
        self._proc = proc
        if self._closed:
            # shutdown() landed between the Popen and this line: it saw no process to kill.
            self._kill()
            self._close_streams()
            raise QualityUnavailable("the quality sidecar was shut down while starting")
        return self._read_ready(proc)

    def _read_ready(self, proc: subprocess.Popen) -> int:
        """Wait for the single ``READY <port>`` line the protocol promises on stdout."""
        result: "deque[str]" = deque(maxlen=8)
        done = threading.Event()

        def reader() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    result.append(line.strip())
                    if line.startswith("READY "):
                        break
            except Exception:  # pragma: no cover - pipe torn down under us
                pass
            finally:
                done.set()

        thread = threading.Thread(target=reader, name="glasstranslate-quality-ready", daemon=True)
        thread.start()
        done.wait(self._ready_timeout)
        lines = list(result)  # a snapshot: the reader may still be appending on a timeout
        for line in reversed(lines):
            if line.startswith("READY "):
                try:
                    return int(line.split(None, 1)[1])
                except (IndexError, ValueError):
                    break
        tail = result[-1] if result else ""
        self._kill()
        if proc.poll() is not None:
            raise QualityUnavailable(f"the quality sidecar exited with code {proc.returncode} ({_one_line(tail)})")
        raise QualityUnavailable(f"the quality sidecar did not report READY within {self._ready_timeout:.0f}s")

    def shutdown(self) -> None:
        """Ask the sidecar to exit, then make sure it did.  Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        if self._port is not None:
            try:
                self._request("POST", "/shutdown", {}, timeout=2.0)
            except QualityUnavailable:
                pass
        self._port = None
        proc = self._proc
        if proc is not None:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()  # stdin EOF is the protocol's "the parent died"
                except OSError:  # pragma: no cover
                    pass
            try:
                proc.wait(SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self._kill()
        self._close_streams()

    def _kill(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.kill()
            proc.wait(2.0)
        except Exception:  # pragma: no cover - already gone
            log.warning("could not kill the quality sidecar")

    def _close_streams(self) -> None:
        proc = self._proc
        for stream in (getattr(proc, "stdout", None), getattr(proc, "stdin", None), self._log_file):
            if stream is not None:
                try:
                    stream.close()
                except OSError:  # pragma: no cover
                    pass
        self._log_file = None

    # -------------------------------------------------------------------- protocol
    def _request(self, method: str, path: str, payload: Optional[dict] = None,
                 timeout: Optional[float] = None) -> dict:
        port = self._port
        if port is None:
            raise QualityUnavailable("the quality sidecar is not running")
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        if body is not None and len(body) > MAX_BODY_BYTES:
            raise QualityUnavailable(f"{path}: request body over the {MAX_BODY_BYTES} byte cap")
        headers = {
            "X-GT-Token": self._token,
            # Loopback only, and the Host the server's own check expects.
            "Host": f"{LOOPBACK}:{port}",
            "Accept": "application/json",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        conn = http.client.HTTPConnection(LOOPBACK, port, timeout=timeout or self._timeout)
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            status = response.status
            raw = response.read(MAX_BODY_BYTES + 1)
        except (OSError, http.client.HTTPException) as exc:
            raise QualityUnavailable(f"{path}: {type(exc).__name__}: {_one_line(exc)}") from exc
        finally:
            conn.close()
        if status != 200:
            # The server answered: the process is alive, only this job failed.
            raise QualityUnavailable(f"{path}: HTTP {status} ({_one_line(self._error_of(raw))})", transport=False)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise QualityUnavailable(f"{path}: malformed response ({_one_line(exc)})", transport=False) from exc
        if not isinstance(data, dict):
            raise QualityUnavailable(f"{path}: malformed response (not an object)", transport=False)
        return data

    @staticmethod
    def _error_of(raw: bytes) -> str:
        try:
            return str(json.loads(raw.decode("utf-8")).get("error", ""))
        except Exception:
            return raw[:120].decode("utf-8", "replace")

    def wait_warm(self, timeout: float = WARM_TIMEOUT_S,
                  should_stop: Optional[Callable[[], bool]] = None) -> dict:
        """Poll ``/health`` until the sidecar can take a job; return that body.

        Ready means ``warm`` (real stages loaded and warmed up), fake mode, or every
        ``models.*`` entry ``"ready"``.  A ``load_error`` or a model group that is neither
        ready nor loading raises :class:`QualityModelsMissing`; ``should_stop`` returning True
        or the timeout raises :class:`QualityUnavailable`.  Without this the first ``/inpaint``
        would sit behind the whole warm-up inside :data:`REQUEST_TIMEOUT_S` and time out on a
        slow start (measured: 166 – 270 s to warm, more under CPU contention).
        """
        deadline = time.monotonic() + timeout
        while True:
            info = self.health()
            error = info.get("load_error")
            if error:
                raise QualityModelsMissing(f"the quality sidecar failed to load: {_one_line(error)}")
            unusable = _models_unusable(info)
            if unusable:
                raise QualityModelsMissing(unusable)
            models = info.get("models")
            all_ready = isinstance(models, dict) and bool(models) and all(
                str(state) == "ready" for state in models.values()
            )
            if info.get("warm") or info.get("mode") == "fake" or all_ready:
                return info
            if should_stop is not None and should_stop():
                raise QualityUnavailable("the quality renderer was stopped", transport=False)
            if time.monotonic() >= deadline:
                raise QualityUnavailable(f"the quality sidecar was still loading after {timeout:.0f}s")
            time.sleep(WARM_POLL_S)

    def health(self) -> dict:
        """The sidecar's ``/health`` document (mode, device, model state, VRAM)."""
        self.start()
        return self._request("GET", "/health", timeout=HEALTH_TIMEOUT_S)

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray, params: Optional[dict] = None) -> np.ndarray:
        """Regenerate the masked pixels of ``image_bgr`` (BGR) and return the BGR result.

        ``mask`` is the panel-shaped fill mask (non-zero = regenerate).  The
        result is re-composited here as a guarantee: **every pixel where the
        mask is 0 is the input pixel, byte for byte**, whatever came back.
        """
        self.start()
        image_bgr = np.ascontiguousarray(image_bgr)
        mask8 = np.ascontiguousarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
        if mask8.shape[:2] != image_bgr.shape[:2]:
            raise QualityUnavailable("the inpainting mask does not match the panel size", transport=False)
        self._jobs += 1
        payload = {
            "job_id": f"j{self._jobs}-{self._job_tag}",  # a correlation id, never a token slice
            "image": _png_b64(image_bgr),
            "mask": _png_b64(mask8),
            "params": {"feather_px": FEATHER_PX, **(params or {})},
        }
        data = self._request("POST", "/inpaint", payload)
        result = _decode_png(str(data.get("image", "")))
        if result is None:
            raise QualityUnavailable("/inpaint: the response carried no decodable image", transport=False)
        if result.shape[:2] != image_bgr.shape[:2]:
            raise QualityUnavailable(
                f"/inpaint: size mismatch (asked for {image_bgr.shape[1]}x{image_bgr.shape[0]}, "
                f"got {result.shape[1]}x{result.shape[0]})",
                transport=False,
            )
        timings = data.get("timings_ms")
        self.last_timings = dict(timings) if isinstance(timings, dict) else {}
        out = np.ascontiguousarray(result)
        keep = mask8 == 0
        out[keep] = image_bgr[keep]
        return out


# ------------------------------------------------------------------------- planning
@dataclass(eq=False)
class Job:
    """One panel to regenerate: the crop, its fill mask and who it belongs to."""

    key: str  # sha1 of the crop + mask bytes: an unchanged panel is never resubmitted
    rect: Rect  # the panel in frame coordinates
    image: np.ndarray  # BGR crop of ``rect``
    mask: np.ndarray  # uint8 (0 / 255) of ``rect``'s shape
    members: Tuple[MemberKey, ...]


def block_key(text: str, style: SegmentStyle) -> MemberKey:
    """Stable identity of a block across pipeline passes (``id()`` is not)."""
    rect = style.clean_rect
    if rect is None:
        return (str(text), -1, -1, -1, -1)
    return (str(text), rect.x, rect.y, rect.w, rect.h)


def _member(item: Any) -> Tuple[str, SegmentStyle]:
    """``(source text, style)`` from a block, a translated segment, a pair or a bare style."""
    if isinstance(item, SegmentStyle):
        return "", item
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], SegmentStyle):
        return str(item[0]), item[1]
    style = getattr(item, "style", None)
    if not isinstance(style, SegmentStyle):
        raise TypeError(f"cannot read a block style from {type(item).__name__}")
    segment = getattr(item, "segment", None)
    text = getattr(segment, "text", None)
    if text is None:
        text = getattr(item, "source_text", "")
    return str(text), style


def _em_of(style: SegmentStyle) -> float:
    return float(style.max_font_px or style.text_height_px or 16.0)


def _disc(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def art_under_text(style: SegmentStyle) -> float:
    """Ink fraction of the ring (``ART_RING_PX`` wide, starting ``ART_FRINGE_PX`` beyond the
    glyph mask) in the quick-filled patch: the art the eraser kept next to and through the
    text, 0 on plain paper.  The fringe is skipped because the mask hugs the glyph cores and
    their anti-aliased edges would otherwise count as ink."""
    patch, mask = style.clean_patch, style.erase_mask
    if patch is None or mask is None or patch.shape[:2] != mask.shape[:2]:
        return 0.0
    glyphs = (mask > 0).astype(np.uint8)
    inner = cv2.dilate(glyphs, _disc(ART_FRINGE_PX)).astype(bool)
    ring = cv2.dilate(glyphs, _disc(ART_FRINGE_PX + ART_RING_PX)).astype(bool) & ~inner
    if not ring.any():
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.int16)
    r, g, b = style.bg
    paper = 0.299 * r + 0.587 * g + 0.114 * b
    ink = np.abs(gray - paper) > ART_INK_DELTA
    return float(ink[ring].mean())


def _is_free_text_over_art(style: SegmentStyle) -> bool:
    """Only free text on artwork is inpainted; bubbles are redrawn, flat paper is filled and
    a block whose surroundings hold no ink (:func:`art_under_text`) keeps the exact quick fill."""
    if not style.outline or style.in_bubble:
        return False
    rect, mask = style.clean_rect, style.erase_mask
    if rect is None or mask is None or rect.w <= 0 or rect.h <= 0:
        return False
    if mask.shape[:2] != (rect.h, rect.w) or not mask.any():
        return False
    return art_under_text(style) >= QUALITY_MIN_ART


def _job_rect(style: SegmentStyle, width: int, height: int) -> Rect:
    """The panel the block lies in, else its clean rect grown by one em."""
    if style.panel_box is not None:
        return style.panel_box.clamp(width, height)
    rect = style.clean_rect
    assert rect is not None
    pad = int(round(FALLBACK_PAD_EM * _em_of(style)))
    return Rect(rect.x - pad, rect.y - pad, rect.w + 2 * pad, rect.h + 2 * pad).clamp(width, height)


def _window_side(panel_side: int, frame_side: int) -> int:
    """A window side between :data:`CONTEXT_MIN_PX` and :data:`CONTEXT_MAX_PX`, the panel's
    own size in between, never larger than the frame."""
    return max(1, min(frame_side, max(min(panel_side, CONTEXT_MAX_PX), CONTEXT_MIN_PX)))


def _place(centre: float, size: int, lo: int, hi: int, frame: int) -> int:
    """Start of a ``size`` window centred on ``centre``, kept inside ``[lo, hi]`` (the panel)
    when it fits there, else inside the frame."""
    if hi - lo < size:
        lo, hi = 0, frame
    start = int(round(centre - size / 2.0))
    return max(lo, min(start, hi - size))


def _context_window(panel: Rect, members: Sequence[Tuple[str, SegmentStyle]], width: int, height: int) -> Rect:
    """The job window for ``panel``: the panel itself when its sides are within
    [:data:`CONTEXT_MIN_PX`, :data:`CONTEXT_MAX_PX`]; a larger panel is cropped to a window of
    that size around the members' clean rects (the whole panel when they do not fit); a smaller
    panel is grown around them, beyond its borders when it must."""
    union: Optional[Rect] = None
    for _text, style in members:
        assert style.clean_rect is not None
        union = style.clean_rect if union is None else union.union(style.clean_rect)
    if union is None:
        return panel
    w, h = _window_side(panel.w, width), _window_side(panel.h, height)
    if union.w > w or union.h > h:
        return panel
    x = _place(union.x + union.w / 2.0, w, panel.x, panel.x2, width)
    y = _place(union.y + union.h / 2.0, h, panel.y, panel.y2, height)
    return Rect(x, y, w, h)


def _paint_member(mask: np.ndarray, rect: Rect, style: SegmentStyle) -> None:
    """OR one member's eraser mask into the panel mask (both in frame coordinates)."""
    clean = style.clean_rect
    assert clean is not None and style.erase_mask is not None
    x0, y0 = max(clean.x, rect.x), max(clean.y, rect.y)
    x1, y1 = min(clean.x2, rect.x2), min(clean.y2, rect.y2)
    if x1 <= x0 or y1 <= y0:
        return
    src = style.erase_mask[y0 - clean.y : y1 - clean.y, x0 - clean.x : x1 - clean.x]
    dst = mask[y0 - rect.y : y1 - rect.y, x0 - rect.x : x1 - rect.x]
    np.putmask(dst, src, np.uint8(255))


def panel_jobs(frame_bgr: np.ndarray, blocks_or_styles: Sequence[Any]) -> List[Job]:
    """One :class:`Job` per cluster of free text over artwork within a panel.

    ``blocks_or_styles`` may hold ``layout.TextBlock`` objects, translated
    segments, ``(text, style)`` pairs or bare styles.  Members are grouped by
    ``style.panel_box``; a block without one gets its own job around its
    ``clean_rect``.  A panel larger than :data:`CONTEXT_MAX_PX` is cropped to a
    window of that size around its text (:func:`_context_window`); members that
    do not fit one window together are split into clusters that do
    (:func:`_cluster_members`), each rendered at native resolution.  The mask is
    the union of the members' eraser masks dilated by
    :data:`QUALITY_MASK_DILATE` px, and the key is the sha1 of the crop bytes
    plus the mask bytes.
    """
    height, width = frame_bgr.shape[:2]
    order: List[Rect] = []
    groups: Dict[Tuple[int, int, int, int], List[Tuple[str, SegmentStyle]]] = {}
    for item in blocks_or_styles:
        text, style = _member(item)
        if not _is_free_text_over_art(style):
            continue
        rect = _job_rect(style, width, height)
        if rect.w <= 0 or rect.h <= 0:
            continue
        key = (rect.x, rect.y, rect.w, rect.h)
        if key not in groups:
            groups[key] = []
            order.append(rect)
        groups[key].append((text, style))

    jobs: List[Job] = []
    for panel in order:
        members = groups[(panel.x, panel.y, panel.w, panel.h)]
        for cluster in _cluster_members(panel, members, width, height):
            rect = _context_window(panel, cluster, width, height)
            job = _make_job(frame_bgr, rect, cluster)
            if job is not None:
                jobs.append(job)
    return jobs


def _cluster_members(
    panel: Rect, members: Sequence[Tuple[str, SegmentStyle]], width: int, height: int
) -> List[List[Tuple[str, SegmentStyle]]]:
    """Split a panel's members into groups whose clean rects fit one context window each.

    A page-sized "panel" (no borders found) with text at the top and the bottom used to go
    to the sidecar whole, i.e. downscaled to the 1024 bucket and back; now each group is
    rendered at native resolution.  Members are swept in reading order and a new group
    starts when adding the next rect would push the union past the window size."""
    w, h = _window_side(panel.w, width), _window_side(panel.h, height)
    ordered = sorted(members, key=lambda m: (m[1].clean_rect.y, m[1].clean_rect.x))  # type: ignore[union-attr]
    clusters: List[List[Tuple[str, SegmentStyle]]] = []
    current: List[Tuple[str, SegmentStyle]] = []
    union: Optional[Rect] = None
    for member in ordered:
        rect = member[1].clean_rect
        assert rect is not None
        merged = rect if union is None else union.union(rect)
        if current and (merged.w > w or merged.h > h):
            clusters.append(current)
            current, merged = [], rect
        current.append(member)
        union = merged
    if current:
        clusters.append(current)
    return clusters


def _make_job(frame_bgr: np.ndarray, rect: Rect, members: Sequence[Tuple[str, SegmentStyle]]) -> Optional[Job]:
    """The job for ``members`` inside ``rect``; None when none of their masks lands in it."""
    # An owned copy: a full-width window would otherwise be a view pinning the whole frame
    # for as long as the job sits in the queue.
    crop = frame_bgr[rect.y : rect.y2, rect.x : rect.x2].copy()
    mask = np.zeros((rect.h, rect.w), np.uint8)
    for _text, style in members:
        _paint_member(mask, rect, style)
    if not mask.any():
        return None
    if QUALITY_MASK_DILATE > 0:
        mask = cv2.dilate(mask, _disc(QUALITY_MASK_DILATE))
    digest = hashlib.sha1(crop.tobytes())  # noqa: S324 - a cache key, not a signature
    digest.update(mask.tobytes())
    return Job(
        key=digest.hexdigest(),
        rect=rect,
        image=crop,
        mask=mask,
        members=tuple(block_key(text, style) for text, style in members),
    )


def _feather(mask: np.ndarray) -> np.ndarray:
    """Alpha for compositing: feathered inside the mask, exactly 0 outside it."""
    alpha = mask.astype(np.float32) / 255.0
    if FEATHER_PX > 0:
        k = 2 * FEATHER_PX + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha[mask == 0] = 0.0
    return alpha


def composite_block_patch(
    frame_bgr: Optional[np.ndarray], job: Job, result_crop: np.ndarray, style: SegmentStyle
) -> np.ndarray:
    """The new ``clean_patch`` of one member block of ``job``.

    The result crop is copied into the block's ``clean_rect`` through a
    feathered mask edge; every pixel the job's mask left at 0 keeps the
    current ``clean_patch`` byte for byte.  ``frame_bgr`` is only the source
    for a block that has no patch yet.
    """
    rect = style.clean_rect
    assert rect is not None
    base = style.clean_patch
    if base is None or base.shape[:2] != (rect.h, rect.w):
        if frame_bgr is None:
            raise ValueError("the block has no clean patch and no frame was given")
        base = frame_bgr[rect.y : rect.y2, rect.x : rect.x2]
    out = np.ascontiguousarray(base).copy()
    x0, y0 = max(rect.x, job.rect.x), max(rect.y, job.rect.y)
    x1, y1 = min(rect.x2, job.rect.x2), min(rect.y2, job.rect.y2)
    if x1 <= x0 or y1 <= y0:
        return out
    jm = job.mask[y0 - job.rect.y : y1 - job.rect.y, x0 - job.rect.x : x1 - job.rect.x]
    jr = result_crop[y0 - job.rect.y : y1 - job.rect.y, x0 - job.rect.x : x1 - job.rect.x]
    if jr.shape[:2] != jm.shape[:2]:
        raise ValueError("the result crop does not match the job")
    dst = out[y0 - rect.y : y1 - rect.y, x0 - rect.x : x1 - rect.x]
    alpha = _feather(jm)[:, :, None]
    blended = dst.astype(np.float32) * (1.0 - alpha) + jr[:, :, :3].astype(np.float32) * alpha
    dst[:] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------------ scheduling
ResultCallback = Callable[[Job, np.ndarray], None]


class QualityScheduler:
    """Runs the panel jobs on a daemon thread, one at a time.

    ``client_factory()`` builds a :class:`QualityClient` (or anything with the
    same three methods); it is called again after :data:`QUALITY_RETRY_S` when
    the sidecar failed.  ``on_result(job, crop)`` is called **from the worker
    thread**: the pipeline queues it and applies it on its own thread.
    """

    def __init__(
        self,
        client_factory: Callable[[], Any],
        *,
        on_result: ResultCallback,
        status: Optional[Callable[[str], None]] = None,
        retry_s: float = QUALITY_RETRY_S,
        name: str = "glasstranslate-quality",
    ) -> None:
        self._factory = client_factory
        self._on_result = on_result
        self._status = status
        self._retry_s = float(retry_s)
        self._cond = threading.Condition()
        self._queue: "deque[Job]" = deque()
        self._done: Set[str] = set()
        self._done_order: "deque[str]" = deque()
        self._in_flight: Optional[Job] = None
        self._stopping = False
        self._client: Optional[Any] = None
        self._available = False
        self._ready_reported = False
        self._reported_error: Optional[str] = None
        self._next_attempt = 0.0
        self._stats: Dict[str, Any] = {
            "done": 0, "failed": 0, "dropped": 0, "superseded": 0,
            "queued": 0, "in_flight": 0, "last_ms": 0.0,
        }
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    # --------------------------------------------------------------------- api
    @property
    def available(self) -> bool:
        """True once a job succeeded and no failure has happened since."""
        return self._available

    @property
    def stats(self) -> Dict[str, Any]:
        with self._cond:
            return dict(self._stats)

    def submit(self, jobs: Sequence[Job]) -> None:
        """Queue ``jobs``, dropping duplicates and superseded work."""
        fresh = [job for job in jobs if isinstance(job, Job)]
        if not fresh:
            return
        with self._cond:
            if self._stopping:
                return
            members = {key for job in fresh for key in job.members}
            kept: "deque[Job]" = deque()
            for queued in self._queue:
                if members.intersection(queued.members):
                    self._stats["superseded"] += 1
                    continue
                kept.append(queued)
            self._queue = kept
            known = {job.key for job in self._queue} | self._done
            if self._in_flight is not None:
                known.add(self._in_flight.key)
            for job in fresh:
                if job.key in known:
                    continue
                known.add(job.key)
                self._queue.append(job)
            self._stats["queued"] = len(self._queue)
            self._cond.notify()

    def stop(self) -> None:
        """Drop the queue, stop the worker and shut the sidecar down.  Idempotent."""
        with self._cond:
            already = self._stopping
            self._stopping = True
            self._queue.clear()
            self._stats["queued"] = 0
            self._cond.notify_all()
        if not already and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=SHUTDOWN_TIMEOUT_S)
        self._drop_client()

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        try:
            while True:
                with self._cond:
                    while not self._stopping and not self._queue:
                        self._cond.wait(0.2)
                    if self._stopping:
                        return
                    job = self._queue.popleft()
                    self._in_flight = job
                    self._stats["queued"] = len(self._queue)
                    self._stats["in_flight"] = 1
                try:
                    self._process(job)
                except Exception:  # pragma: no cover - the worker must never die
                    log.exception("quality job failed unexpectedly")
                finally:
                    with self._cond:
                        self._in_flight = None
                        self._stats["in_flight"] = 0
        finally:
            # The worker owns the client it created: the sidecar goes down with this
            # thread whether or not stop()'s bounded join won the race against a start.
            self._drop_client()

    def _process(self, job: Job) -> None:
        if self._client is None and time.monotonic() < self._next_attempt:
            with self._cond:
                self._stats["dropped"] += 1  # still backing off: the quick fill stands
            return
        try:
            client = self._ensure_client()
        except Exception as exc:  # noqa: BLE001 - any construction failure is a fallback
            self._fail(exc)
            return
        if self._stopping:
            self._drop_client()  # stopped while the sidecar was starting: no job, no result
            return
        started = time.perf_counter()
        try:
            crop = client.inpaint(job.image, job.mask)
        except Exception as exc:  # noqa: BLE001 - the app keeps the quick fill
            if getattr(exc, "transport", True):
                self._drop_client()  # the process or its socket is gone; a job error keeps a warm sidecar
            self._fail(exc)
            return
        if self._stopping:
            return  # never deliver into a pipeline that has released this scheduler
        with self._cond:
            self._stats["last_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            self._stats["done"] += 1
            self._remember(job.key)
        self._available = True
        self._reported_error = None
        try:
            self._on_result(job, crop)
        except Exception:  # pragma: no cover - a consumer bug must not kill the worker
            log.exception("quality result callback failed")

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        client = self._factory()
        if client is None:
            raise QualityUnavailable("no quality sidecar available")
        with self._cond:
            if self._stopping:
                raise QualityUnavailable("the quality renderer was stopped")
            self._client = client  # published before start(): _drop_client() can always reach it
        try:
            client.start()
            info: Dict[str, Any] = {}
            waiter = getattr(client, "wait_warm", None)
            if callable(waiter):
                # Block this worker (not the pipeline) until the stages are resident, so the
                # first job's request timeout covers the job and not the warm-up.
                info = waiter(should_stop=lambda: self._stopping) or {}
            else:
                try:
                    info = client.health() or {}
                except Exception as exc:  # noqa: BLE001 - health is informational
                    log.info("quality sidecar health check failed: %s", _one_line(exc))
            unusable = _models_unusable(info)
            if unusable:
                raise QualityModelsMissing(unusable)
        except BaseException:
            self._drop_client()
            raise
        if not self._ready_reported:
            self._ready_reported = True
            device = str(info.get("device") or "?").split(":")[0]
            self._report(f"Quality renderer: ready on {device}")
        return client

    def _drop_client(self) -> None:
        with self._cond:
            client, self._client = self._client, None
        if client is None:
            return
        try:
            client.shutdown()
        except Exception:  # pragma: no cover - best effort
            log.warning("shutting the quality sidecar down failed")

    def _fail(self, exc: BaseException) -> None:
        with self._cond:
            self._stats["failed"] += 1
        self._available = False
        retry_s = MODELS_RETRY_S if isinstance(exc, QualityModelsMissing) else self._retry_s
        self._next_attempt = time.monotonic() + retry_s
        reason = _one_line(exc)
        if reason == self._reported_error:
            return  # one line per distinct error, not one per job
        self._reported_error = reason
        log.warning("quality renderer unavailable: %s", reason)
        self._report(f"Quality renderer unavailable: {reason}; using the quick fill")

    def _remember(self, key: str) -> None:
        """Remember a finished content key (bounded, so a long session cannot grow)."""
        if key in self._done:
            return
        self._done.add(key)
        self._done_order.append(key)
        while len(self._done_order) > DONE_KEYS_MAX:
            self._done.discard(self._done_order.popleft())

    def _report(self, message: str) -> None:
        if self._status is None:
            return
        try:
            self._status(message)
        except Exception:  # pragma: no cover
            log.exception("quality status callback failed")
