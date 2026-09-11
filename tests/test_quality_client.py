"""``glasstranslate.render.quality`` client + job planning against a fake sidecar.

The sidecar protocol (``renderer/PROTOCOL.md``) is exercised by a tiny in-test HTTP server
that implements the token / ``Host`` checks, ``/health``, ``/inpaint`` (fills the whole crop
with a constant, so the client's re-composite is the only thing that can keep the pixels
outside the mask) and ``/shutdown``.  No process is spawned and no GPU is needed:
:class:`QualityClient` takes an injectable ``start`` hook that returns the port.

One opt-in subprocess test spawns the real sidecar in ``--fake`` mode when
``renderer/glassrenderer/__main__.py`` exists.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from glasstranslate.config.settings import AppConfig
from glasstranslate.core.types import Rect, SegmentStyle
from glasstranslate.render import quality as Q

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "a" * 64
FILL = 77  # the fake sidecar paints the whole crop this grey


# --------------------------------------------------------------------------- fake sidecar
class _FakeServer:
    """Threaded 127.0.0.1 server implementing the protocol's fake behaviour."""

    def __init__(self, *, fill: int = FILL, size_delta: int = 0, status: int = 200) -> None:
        self.fill = fill
        self.size_delta = size_delta  # non-zero: answer with a differently sized image
        self.status = status
        self.requests: List[Tuple[str, Dict[str, str]]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:  # noqa: D401 - silence the test log
                return

            def _json(self, code: int, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _checked(self) -> bool:
                outer.requests.append((self.path, dict(self.headers.items())))
                if self.headers.get("X-GT-Token") != TOKEN:
                    self._json(401, {"error": "unauthorized"})
                    return False
                host = (self.headers.get("Host") or "").split(":")[0]
                if host not in ("127.0.0.1", "localhost"):
                    self._json(403, {"error": "bad host"})
                    return False
                return True

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                if not self._checked():
                    return
                if self.path != "/health":
                    self._json(404, {"error": "not found"})
                    return
                self._json(200, {"ok": True, "version": "1", "mode": "fake", "device": "cpu",
                                 "models": {"lama": "ready", "sdxl": "ready"}, "warm": False,
                                 "uptime_s": 0.1})

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                if not self._checked():
                    return
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                if self.path == "/shutdown":
                    self._json(200, {"ok": True})
                    return
                if self.path != "/inpaint":
                    self._json(404, {"error": "not found"})
                    return
                if outer.status != 200:
                    self._json(outer.status, {"error": "boom: stage failed"})
                    return
                body = json.loads(raw.decode("utf-8"))
                png = base64.b64decode(body["image"])
                img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                out = np.full((h + outer.size_delta, w + outer.size_delta, 3), outer.fill, np.uint8)
                ok, buf = cv2.imencode(".png", out)
                assert ok
                self._json(200, {"job_id": body.get("job_id", ""),
                                 "image": base64.b64encode(buf.tobytes()).decode("ascii"),
                                 "timings_ms": {"total": 1.0}, "mode": "fake", "stages": ["lama"],
                                 "size": [out.shape[1], out.shape[0]]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture()
def server():
    srv = _FakeServer()
    yield srv
    srv.close()


def _client(srv: _FakeServer, *, token: str = TOKEN) -> Q.QualityClient:
    client = Q.QualityClient(None, "models", fake=True, token=token, start=lambda: srv.port)
    client.start()
    return client


def _panel() -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)
    mask = np.zeros((40, 60), np.uint8)
    mask[10:20, 15:35] = 255
    return image, mask


# ------------------------------------------------------------------------------- the client
def test_inpaint_recomposites_outside_the_mask_byte_for_byte(server: _FakeServer) -> None:
    image, mask = _panel()
    client = _client(server)
    out = client.inpaint(image, mask)
    assert out.shape == image.shape and out.dtype == np.uint8
    keep = mask == 0
    assert np.array_equal(out[keep], image[keep])  # byte for byte
    assert (out[mask > 0] == FILL).all()
    client.shutdown()


def test_request_carries_the_token_and_a_loopback_host(server: _FakeServer) -> None:
    image, mask = _panel()
    client = _client(server)
    client.inpaint(image, mask)
    path, headers = server.requests[-1]
    assert path == "/inpaint"
    assert headers["X-GT-Token"] == TOKEN
    assert headers["Host"] == f"127.0.0.1:{server.port}"
    client.shutdown()


def test_a_wrong_token_surfaces_as_quality_unavailable(server: _FakeServer) -> None:
    image, mask = _panel()
    client = _client(server, token="b" * 64)
    with pytest.raises(Q.QualityUnavailable) as excinfo:
        client.inpaint(image, mask)
    assert "401" in str(excinfo.value)
    assert "\n" not in str(excinfo.value)  # one line, no traceback


def test_a_server_error_and_a_wrong_size_both_raise() -> None:
    image, mask = _panel()
    broken = _FakeServer(status=500)
    try:
        with pytest.raises(Q.QualityUnavailable):
            _client(broken).inpaint(image, mask)
    finally:
        broken.close()
    resized = _FakeServer(size_delta=4)
    try:
        with pytest.raises(Q.QualityUnavailable) as excinfo:
            _client(resized).inpaint(image, mask)
        assert "size" in str(excinfo.value).lower()
    finally:
        resized.close()


def test_health_and_shutdown(server: _FakeServer) -> None:
    client = _client(server)
    info = client.health()
    assert info["ok"] is True and info["mode"] == "fake" and info["device"] == "cpu"
    client.shutdown()
    client.shutdown()  # idempotent
    assert [p for p, _ in server.requests if p == "/shutdown"]


# ------------------------------------------------------------------------- interpreter lookup
def test_find_sidecar_python_prefers_the_configured_interpreter(tmp_path: Path) -> None:
    exe = tmp_path / "python.exe"
    exe.write_text("", encoding="utf-8")
    cfg = AppConfig(quality_sidecar_python=str(exe))
    assert Q.find_sidecar_python(cfg) == exe
    # A configured path that does not exist falls through to the checkout / user-data lookup.
    missing = AppConfig(quality_sidecar_python=str(tmp_path / "nope.exe"))
    found = Q.find_sidecar_python(missing)
    assert found is None or found.exists()


# ----------------------------------------------------------------------------- panel jobs
def _style(clean: Rect, *, panel: Optional[Rect], outline: bool = True, in_bubble: bool = False,
           mask: Optional[np.ndarray] = None, art: bool = True) -> SegmentStyle:
    if mask is None:
        # Inset far enough that the QUALITY_MASK_DILATE growth still leaves a
        # kept margin inside clean_rect (the byte-identity assertions need one).
        mask = np.zeros((clean.h, clean.w), bool)
        mask[8:-8, 6:-6] = True
    patch = np.full((clean.h, clean.w, 3), 235, np.uint8)
    if art:  # hatching through the text: what the eraser keeps and the sidecar continues
        patch[::3, :, :] = 0
    return SegmentStyle(
        fg=(0, 0, 0), bg=(255, 255, 255), angle_deg=0.0, text_height_px=20.0, vertical=True,
        layout_box=Rect(clean.x, clean.y, clean.w, clean.h), clean_rect=clean,
        clean_patch=patch, outline=outline,
        in_bubble=in_bubble, erase_mask=mask, panel_box=panel, max_font_px=16.0,
    )


def test_wait_warm_polls_until_ready_and_raises_on_a_load_error(monkeypatch) -> None:
    monkeypatch.setattr(Q, "WARM_POLL_S", 0.01)
    client = Q.QualityClient(None, "models", start=lambda: 1)
    loading = {"ok": True, "mode": "real", "device": "cuda:0", "warm": False, "load_error": None,
               "models": {"lama": "loading", "sdxl": "loading", "controlnet": "loading"}}
    warm = dict(loading, warm=True, models={k: "ready" for k in loading["models"]})
    bodies = [loading, loading, warm]
    monkeypatch.setattr(client, "health", lambda: bodies.pop(0))
    assert client.wait_warm(timeout=5.0)["warm"] is True and bodies == []
    # Fake mode never says warm; it is ready as soon as it answers.
    monkeypatch.setattr(client, "health", lambda: {"ok": True, "mode": "fake", "device": "cpu", "warm": False})
    assert client.wait_warm(timeout=1.0)["mode"] == "fake"
    # A failed preload is reported, not waited out.
    monkeypatch.setattr(client, "health", lambda: dict(loading, load_error="RuntimeError: CUDA out of memory"))
    with pytest.raises(Q.QualityModelsMissing, match="CUDA out of memory"):
        client.wait_warm(timeout=5.0)
    # A stop request ends the wait; so does the timeout.
    monkeypatch.setattr(client, "health", lambda: loading)
    with pytest.raises(Q.QualityUnavailable, match="stopped"):
        client.wait_warm(timeout=5.0, should_stop=lambda: True)
    with pytest.raises(Q.QualityUnavailable, match="still loading"):
        client.wait_warm(timeout=0.0)


def test_free_text_on_plain_paper_is_not_sent_to_the_sidecar() -> None:
    frame = np.full((200, 300, 3), 180, np.uint8)
    plain = _style(Rect(10, 10, 30, 40), panel=Rect(0, 0, 150, 200), art=False)
    hatched = _style(Rect(60, 20, 30, 40), panel=Rect(0, 0, 150, 200))
    assert Q.art_under_text(plain) == 0.0 and Q.art_under_text(hatched) > Q.QUALITY_MIN_ART
    jobs = Q.panel_jobs(frame, [("plain", plain), ("hatched", hatched)])
    assert len(jobs) == 1 and [k[0] for k in jobs[0].members] == ["hatched"]


def test_small_panels_grow_to_the_minimum_context_window() -> None:
    frame = np.full((900, 900, 3), 180, np.uint8)
    small = Rect(400, 400, 80, 110)
    jobs = Q.panel_jobs(frame, [("tiny", _style(Rect(410, 410, 40, 60), panel=small))])
    rect = jobs[0].rect
    assert rect.w == Q.CONTEXT_MIN_PX and rect.h == Q.CONTEXT_MIN_PX
    assert rect.x <= 410 and rect.y <= 410 and rect.x2 >= 450 and rect.y2 >= 470
    assert rect.x >= 0 and rect.y >= 0 and rect.x2 <= 900 and rect.y2 <= 900
    # A block without a panel gets the same treatment around its clean rect.
    jobs = Q.panel_jobs(frame, [("solo", _style(Rect(20, 20, 40, 60), panel=None))])
    assert jobs[0].rect == Rect(0, 0, Q.CONTEXT_MIN_PX, Q.CONTEXT_MIN_PX)


def test_panel_jobs_group_by_panel_and_skip_bubbles_and_flat_paper() -> None:
    frame = np.full((200, 300, 3), 180, np.uint8)
    panel = Rect(0, 0, 150, 200)
    members = [
        ("one", _style(Rect(10, 10, 30, 40), panel=panel)),
        ("two", _style(Rect(60, 20, 30, 40), panel=panel)),
        ("bubble", _style(Rect(100, 20, 20, 20), panel=panel, in_bubble=True)),
        ("flat", _style(Rect(120, 20, 20, 20), panel=panel, outline=False)),
        ("other", _style(Rect(200, 20, 30, 40), panel=Rect(150, 0, 150, 200))),
    ]
    jobs = Q.panel_jobs(frame, members)
    assert len(jobs) == 2
    left = next(job for job in jobs if any(k[0] == "one" for k in job.members))
    assert sorted(k[0] for k in left.members) == ["one", "two"]
    # The panel is smaller than the minimum context window, so the job window is grown around
    # the text (here: the whole 300x200 frame); the crop and mask follow the window.
    assert left.rect == Rect(0, 0, 300, 200)
    assert left.image.shape == (left.rect.h, left.rect.w, 3)
    assert left.mask.shape == (left.rect.h, left.rect.w) and left.mask.dtype == np.uint8
    assert set(np.unique(left.mask)) <= {0, 255}
    # The mask sits where the members' erase masks are, dilated.
    assert left.mask[10 + 10, 10 + 10] == 255 and left.mask[100, 100] == 0
    assert all(len(job.key) == 40 for job in jobs)  # sha1 hex


def test_panel_jobs_without_a_panel_use_the_grown_clean_rect() -> None:
    frame = np.full((200, 300, 3), 180, np.uint8)
    jobs = Q.panel_jobs(frame, [("solo", _style(Rect(40, 50, 30, 40), panel=None))])
    assert len(jobs) == 1
    rect = jobs[0].rect
    assert rect.x < 40 and rect.y < 50 and rect.x2 > 70 and rect.y2 > 90


def test_panel_jobs_crop_a_large_panel_to_a_context_window_around_the_text() -> None:
    frame = np.full((1600, 1100, 3), 180, np.uint8)
    page = Rect(0, 0, 1100, 1600)  # a page-sized "panel" (no borders found)
    jobs = Q.panel_jobs(frame, [("one", _style(Rect(700, 1200, 60, 120), panel=page))])
    rect = jobs[0].rect
    assert max(rect.w, rect.h) <= Q.CONTEXT_MAX_PX
    assert rect.x <= 700 and rect.y <= 1200 and rect.x2 >= 760 and rect.y2 >= 1320  # covers the block
    assert rect.x >= 0 and rect.y >= 0 and rect.x2 <= 1100 and rect.y2 <= 1600
    assert abs((rect.x + rect.x2) / 2 - 730) < 300 and abs((rect.y + rect.y2) / 2 - 1260) < 300  # centred on it
    assert jobs[0].image.shape == (rect.h, rect.w, 3) and jobs[0].mask.shape == (rect.h, rect.w)
    # Two members far apart on one page share the window when both fit in it...
    jobs = Q.panel_jobs(frame, [("a", _style(Rect(100, 100, 60, 120), panel=page)),
                                ("b", _style(Rect(700, 300, 60, 120), panel=page))])
    assert len(jobs) == 1 and jobs[0].rect.x <= 100 and jobs[0].rect.x2 >= 760
    assert max(jobs[0].rect.w, jobs[0].rect.h) <= Q.CONTEXT_MAX_PX
    # ...and split into one native-resolution window each when they do not.
    jobs = Q.panel_jobs(frame, [("a", _style(Rect(100, 100, 60, 120), panel=page)),
                                ("b", _style(Rect(700, 1300, 60, 120), panel=page))])
    assert [job.members[0][0] for job in jobs] == ["a", "b"]
    assert all(max(job.rect.w, job.rect.h) <= Q.CONTEXT_MAX_PX for job in jobs)
    a, b = jobs[0].rect, jobs[1].rect
    assert a.x <= 100 and a.y <= 100 and a.x2 >= 160 and a.y2 >= 220
    assert b.x <= 700 and b.y <= 1300 and b.x2 >= 760 and b.y2 >= 1420
    assert all(0 <= job.rect.x and 0 <= job.rect.y and job.rect.x2 <= 1100 and job.rect.y2 <= 1600 for job in jobs)
    # A member wider than the window still sends the whole panel.
    jobs = Q.panel_jobs(frame, [("wide", _style(Rect(20, 100, 1060, 120), panel=page))])
    assert len(jobs) == 1 and jobs[0].rect == page


def test_panel_job_keys_change_with_the_pixels_and_not_with_the_text() -> None:
    frame = np.full((200, 300, 3), 180, np.uint8)
    members = [("one", _style(Rect(10, 10, 30, 40), panel=Rect(0, 0, 150, 200)))]
    first = Q.panel_jobs(frame, members)[0]
    again = Q.panel_jobs(frame, [("renamed", _style(Rect(10, 10, 30, 40), panel=Rect(0, 0, 150, 200)))])[0]
    assert first.key == again.key
    frame2 = frame.copy()
    frame2[0:20, 0:20] = 0
    assert Q.panel_jobs(frame2, members)[0].key != first.key


def test_composite_block_patch_only_changes_masked_pixels() -> None:
    frame = np.full((200, 300, 3), 180, np.uint8)
    clean = Rect(10, 10, 30, 40)
    style = _style(clean, panel=Rect(0, 0, 150, 200))
    job = Q.panel_jobs(frame, [("one", style)])[0]
    result = np.full((job.rect.h, job.rect.w, 3), FILL, np.uint8)
    patch = Q.composite_block_patch(frame, job, result, style)
    assert patch.shape == style.clean_patch.shape and patch.dtype == np.uint8
    sub = job.mask[clean.y - job.rect.y : clean.y2 - job.rect.y, clean.x - job.rect.x : clean.x2 - job.rect.x]
    keep = sub == 0
    assert np.array_equal(patch[keep], style.clean_patch[keep])
    core = np.zeros_like(sub, bool)
    core[10:-10, 10:-10] = sub[10:-10, 10:-10] > 0
    assert core.any() and (patch[core] == FILL).all()  # feathering only touches the mask edge


# ------------------------------------------------------------------------- real sidecar (opt in)
@pytest.mark.skipif(
    not (ROOT / "renderer" / "glassrenderer" / "__main__.py").is_file(),
    reason="the sidecar package is not delivered yet",
)
def test_real_sidecar_serves_the_protocol_in_fake_mode(tmp_path: Path) -> None:
    client = Q.QualityClient(Path(sys.executable), tmp_path, fake=True,
                             log_path=None, ready_timeout=60.0)
    try:
        client.start()
        info = client.health()
        assert info.get("ok") is True and info.get("mode") == "fake"
        image, mask = _panel()
        out = client.inpaint(image, mask)
        keep = mask == 0
        assert np.array_equal(out[keep], image[keep])
    except Q.QualityUnavailable as exc:  # pragma: no cover - the sidecar venv may be absent
        pytest.skip(f"sidecar unavailable: {exc}")
    finally:
        client.shutdown()
    assert not isinstance(client._proc, subprocess.Popen) or client._proc.poll() is not None
