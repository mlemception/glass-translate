"""The frozen sidecar half of the ``portable`` smoke run (checks (1) and (2)).

``packaging/smoke_portable.py`` drives the bundle as a whole; this module is everything that
speaks to ``renderer\\glassrenderer.exe``: spawning it, the loopback protocol of
``renderer/PROTOCOL.md``, the synthetic panel an ``/inpaint`` is asserted against, and the two
checks themselves.  It also owns the "did anything reach the network" scanner, which the app
half applies to the log files.

It imports from ``smoke_test`` (the result table) but never from ``smoke_portable``, so the two
can be imported in either order.
"""
from __future__ import annotations

import base64
import contextlib
import http.client
import json
import re
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from smoke_test import Results, Sandbox, _rc_text

RENDERER_DIRNAME = "renderer"
SIDECAR_EXE = "glassrenderer.exe"
TOKEN_ENV = "GT_RENDERER_TOKEN"
TOKEN_HEADER = "X-GT-Token"
LOOPBACK = "127.0.0.1"

READY_TIMEOUT_FAKE_S = 60.0
READY_TIMEOUT_REAL_S = 120.0
MODELS_READY_TIMEOUT_S = 900.0
MODELS_POLL_S = 2.0
HEALTH_TIMEOUT_S = 30.0
INPAINT_TIMEOUT_S = 300.0
STDIN_EXIT_S = 10.0  # fake mode has nothing to tear down: it must exit cleanly and fast
# After a real job the CUDA teardown under ``os._exit`` takes 14-19 s and can end in
# 0xC0000409 (measured on the frozen sidecar).  Not something this run can fix, so the GPU
# check allows the time and accepts any exit code, recording the one it saw.
GPU_EXIT_S = 45.0

# Any of these in a log or on the sidecar's stderr means the bundle reached for the network.
NOISE_RE = re.compile(
    r"(?i)(downloading|https?://|proxy|connection refused|urlopen|HTTPSConnectionPool|argospm|huggingface\.co)"
)


def read_text(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def noise_lines(text: str) -> List[str]:
    """The lines of ``text`` that mention a download, a URL or a proxy - the zero-network proof."""
    return [line.strip() for line in text.splitlines() if NOISE_RE.search(line)]


# ----------------------------------------------------------------------------- inpaint images
def synthetic_panel(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """A deterministic BGR panel with structure plus a centred fill mask (255 = regenerate)."""
    ys, xs = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width, 3), np.uint8)
    image[..., 0] = (xs * 255 // max(1, width - 1)).astype(np.uint8)
    image[..., 1] = (ys * 255 // max(1, height - 1)).astype(np.uint8)
    image[..., 2] = (((xs // 8 + ys // 8) % 2) * 200).astype(np.uint8)
    mask = np.zeros((height, width), np.uint8)
    mask[height // 4: height // 2, width // 4: width // 2] = 255
    return image, mask


def png_b64(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", np.ascontiguousarray(image))
    if not ok:
        raise ValueError("could not PNG-encode the panel")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_png(payload: str) -> Optional[np.ndarray]:
    try:
        raw = base64.b64decode(payload or "", validate=True)
    except (ValueError, TypeError):
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def same_outside_mask(source: np.ndarray, result: Optional[np.ndarray], mask: np.ndarray) -> bool:
    """PROTOCOL.md's guarantee: every pixel the mask left at 0 comes back unchanged."""
    if result is None or source.shape != result.shape:
        return False
    keep = mask == 0
    return bool(np.array_equal(source[keep], result[keep]))


# ----------------------------------------------------------------------------- the frozen sidecar
class Sidecar:
    """One ``renderer\\glassrenderer.exe serve`` process talked to over the loopback protocol."""

    def __init__(self, root: Path, env: Dict[str, str], stderr_path: Path, *,
                 fake: bool, models_dir: Optional[Path] = None) -> None:
        self.exe = Path(root) / RENDERER_DIRNAME / SIDECAR_EXE
        self.models_dir = Path(models_dir) if models_dir is not None else Path(root) / "models"
        self.stderr_path = Path(stderr_path)
        self.token = secrets.token_hex(32)
        self.port: Optional[int] = None
        self.proc: Optional[subprocess.Popen] = None
        self._fake = bool(fake)
        self._env = {**env, TOKEN_ENV: self.token}
        self._stderr: Any = None

    def start(self, timeout: float) -> float:
        """Spawn the sidecar and wait for its single ``READY <port>`` line; returns seconds."""
        cmd = [str(self.exe), "serve", "--models-dir", str(self.models_dir)]
        if self._fake:
            cmd.append("--fake")
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = open(self.stderr_path, "wb")
        kwargs: Dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        started = time.perf_counter()
        self.proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            cmd, cwd=str(self.exe.parent), env=self._env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._stderr, text=True, encoding="utf-8",
            errors="replace", **kwargs,
        )
        self.port = self._read_ready(timeout)
        return time.perf_counter() - started

    def _read_ready(self, timeout: float) -> int:
        found: List[int] = []

        def reader() -> None:
            stream = None if self.proc is None else self.proc.stdout
            if stream is None:
                return
            for line in stream:
                parts = line.strip().split()
                if parts and parts[0].upper() == "READY" and len(parts) >= 2 and parts[1].isdigit():
                    found.append(int(parts[1]))
                    return

        thread = threading.Thread(target=reader, name="gt-portable-ready", daemon=True)
        thread.start()
        thread.join(timeout)
        if not found:
            raise RuntimeError(f"no READY line within {timeout:.0f} s; stderr: {self.stderr_text()[-300:]!r}")
        return found[0]

    def request(self, method: str, path: str, payload: Optional[dict] = None,
                timeout: float = HEALTH_TIMEOUT_S) -> Tuple[int, dict]:
        """One protocol request.  ``http.client`` talks to the loopback directly, so the dead
        proxy variables in the environment cannot affect it - which is the point."""
        if self.port is None:
            raise RuntimeError("the sidecar is not running")
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {TOKEN_HEADER: self.token}
        if body is not None:
            headers["Content-Type"] = "application/json"
        conn = http.client.HTTPConnection(LOOPBACK, int(self.port), timeout=timeout)
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            status = response.status
        finally:
            conn.close()
        if not raw:
            return status, {}
        try:
            return status, json.loads(raw.decode("utf-8"))
        except ValueError:
            return status, {"error": raw[:200].decode("utf-8", "replace")}

    def close_stdin(self) -> None:
        if self.proc is not None and self.proc.stdin is not None:
            with contextlib.suppress(OSError, ValueError):
                self.proc.stdin.close()

    def wait(self, timeout: float) -> Optional[int]:
        if self.proc is None:
            return None
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            with contextlib.suppress(OSError):
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self._stderr is not None:
            with contextlib.suppress(OSError):
                self._stderr.close()
            self._stderr = None

    def stderr_text(self) -> str:
        if self._stderr is not None:
            with contextlib.suppress(OSError):
                self._stderr.flush()
        return read_text(self.stderr_path)


def _protocol_checks(car: Sidecar, res: Results, tag: str, size: Tuple[int, int], label: str) -> Optional[float]:
    """``/health`` plus one ``/inpaint``; returns the job's wall time in ms."""
    status, health = car.request("GET", "/health", timeout=HEALTH_TIMEOUT_S)
    res.add(tag, f"/health answers ok ({label})", status == 200 and bool(health.get("ok")),
            f"{status} {json.dumps(health)[:160]}")
    image, mask = synthetic_panel(*size)
    payload = {"job_id": f"{tag}-{secrets.token_hex(3)}", "image": png_b64(image), "mask": png_b64(mask)}
    started = time.perf_counter()
    status, body = car.request("POST", "/inpaint", payload, timeout=INPAINT_TIMEOUT_S)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    out = decode_png(str(body.get("image", ""))) if status == 200 else None
    res.add(tag, f"/inpaint answers with the input size ({label})",
            out is not None and out.shape == image.shape,
            f"{status} {None if out is None else out.shape} vs {image.shape} {str(body.get('error', ''))[:120]}")
    res.add(tag, f"/inpaint is byte-identical outside the mask ({label})",
            same_outside_mask(image, out, mask), f"{elapsed_ms:.0f} ms")
    return elapsed_ms


def check_sidecar_fake(sb: Sandbox, res: Results, tag: str = "S1") -> None:
    """Check (1): the frozen sidecar in fake mode, twice (cold and warm start)."""
    root = sb.dir
    exe = root / RENDERER_DIRNAME / SIDECAR_EXE
    if not res.add(tag, "frozen sidecar present", exe.is_file(), str(exe)):
        return
    cold: List[float] = []
    for attempt in (1, 2):
        car = Sidecar(root, sb.base_env, sb.temp / f"renderer-{tag}-{attempt}.err", fake=True)
        seconds, error = 0.0, ""
        try:
            seconds = car.start(READY_TIMEOUT_FAKE_S)
        except Exception as exc:  # noqa: BLE001 - the outcome is a check row
            error = f"{type(exc).__name__}: {exc}"
        started_ok = res.add(tag, f"fake sidecar printed READY (launch {attempt})", not error,
                             error or f"{seconds:.1f} s on port {car.port}")
        if started_ok:
            cold.append(round(seconds, 2))
            if attempt == 1:
                _protocol_checks(car, res, tag, (256, 256), "fake")
            car.close_stdin()
            code = car.wait(STDIN_EXIT_S)
            res.add(tag, f"sidecar exits 0 on stdin EOF (launch {attempt})", code == 0, _rc_text(code))
        noise = noise_lines(car.stderr_text())
        car.kill()
        res.add(tag, f"sidecar stderr has no download/URL/proxy line (launch {attempt})",
                not noise, "; ".join(noise[:3])[:200])
    res.info.setdefault("portable", {})[f"{res.prefix}{tag}_cold_start_s"] = cold


def health_ready(health: Dict[str, Any]) -> bool:
    """Whether a **real** sidecar has actually loaded its stages, per one ``/health`` body.

    ``models.*`` is only a file-presence check, so a broken build reports every model ``ready``
    straight away.  What proves the stages loaded and ran is ``warm`` (the server sets it after
    its warm-up) together with an empty ``load_error``; all three must hold.
    """
    if health.get("load_error"):
        return False
    models = health.get("models") or {}
    if not models or not all(str(state) == "ready" for state in models.values()):
        return False
    return bool(health.get("warm"))


def _wait_models_ready(car: Sidecar, timeout: float) -> Tuple[Optional[float], Dict[str, Any]]:
    """Poll ``/health`` until :func:`health_ready`; a ``load_error`` fails the run at once."""
    started = time.perf_counter()
    seen: Dict[str, Any] = {}
    while time.perf_counter() - started < timeout:
        _, health = car.request("GET", "/health", timeout=HEALTH_TIMEOUT_S)
        error = health.get("load_error")
        if error:
            raise RuntimeError(f"model load error: {str(error)[:200]}")
        seen = {"models": {str(k): str(v) for k, v in (health.get("models") or {}).items()},
                "warm": bool(health.get("warm"))}
        if health_ready(health):
            return round(time.perf_counter() - started, 1), seen
        time.sleep(MODELS_POLL_S)
    return None, seen


def check_sidecar_gpu(sb: Sandbox, res: Results, tag: str = "S2") -> None:
    """Check (2), ``GT_GPU_TESTS=1``: the real stages load from the bundled store and run a job."""
    root = sb.dir
    info = res.info.setdefault("portable", {})
    car = Sidecar(root, sb.base_env, sb.temp / f"renderer-{tag}.err", fake=False, models_dir=root / "models")
    try:
        seconds = car.start(READY_TIMEOUT_REAL_S)
        res.add(tag, "real sidecar printed READY", True, f"{seconds:.1f} s on port {car.port}")
        info[f"{tag}_ready_s"] = round(seconds, 1)
        ready_s, seen = _wait_models_ready(car, MODELS_READY_TIMEOUT_S)
        res.add(tag, "every model loads and the sidecar warms up (no download)", ready_s is not None,
                f"{seen} after {ready_s} s")
        info[f"{tag}_models_ready_s"] = ready_s
        elapsed_ms = _protocol_checks(car, res, tag, (512, 768), "real")
        info[f"{tag}_first_job_ms"] = None if elapsed_ms is None else round(elapsed_ms, 1)
        car.request("POST", "/shutdown", {}, timeout=HEALTH_TIMEOUT_S)
        code = car.wait(GPU_EXIT_S)
        # Any exit code: after a real job the CUDA teardown under os._exit often reports
        # 0xC0000409.  What matters here is that the process goes away at all.
        res.add(tag, f"sidecar exits within {GPU_EXIT_S:.0f} s after /shutdown (any exit code)",
                code is not None, f"{_rc_text(code)} (CUDA teardown may report 0xC0000409)")
    except Exception as exc:  # noqa: BLE001 - the outcome is a check row
        res.add(tag, "real sidecar served one job", False, f"{type(exc).__name__}: {exc}")
    finally:
        noise = noise_lines(car.stderr_text())
        car.kill()
        res.add(tag, "sidecar stderr has no download/URL/proxy line", not noise, "; ".join(noise[:3])[:200])
