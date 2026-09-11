"""The sidecar server end to end, spawned exactly as the app spawns it.

The child runs under the *main* venv (``sys.executable`` is the venv the tests
run in) in ``--fake`` mode, so the whole HTTP contract is covered without torch
and without a GPU.  ``PYTHONPATH`` carries ``renderer/`` because the package is
not installed into the main venv.
"""
from __future__ import annotations

import http.client
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pytest

from glassrenderer import protocol as P

RENDERER_DIR = Path(__file__).resolve().parents[1]
READY_TIMEOUT_S = 30.0
EXIT_TIMEOUT_S = 5.0


def _read_line(stream, timeout: float) -> str:
    """Read one line with a timeout (a blocked child must fail, not hang)."""
    box: list[str] = []
    reader = threading.Thread(target=lambda: box.append(stream.readline()), daemon=True)
    reader.start()
    reader.join(timeout)
    return box[0] if box else ""


def _spawn(models_dir: Path, token: str) -> Tuple[subprocess.Popen, int]:
    env = dict(os.environ)
    env["GT_RENDERER_TOKEN"] = token
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(RENDERER_DIR)
    proc = subprocess.Popen(
        [sys.executable, "-m", "glassrenderer", "serve", "--fake",
         "--models-dir", str(models_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    threading.Thread(target=lambda: proc.stderr.read(), daemon=True).start()
    line = _read_line(proc.stdout, READY_TIMEOUT_S).strip()
    if not line.startswith("READY "):
        proc.kill()
        raise AssertionError(f"child did not announce READY, got {line!r}")
    return proc, int(line.split()[1])


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    try:
        proc.communicate(timeout=EXIT_TIMEOUT_S)
    except Exception:  # pragma: no cover - best effort teardown
        pass


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Tuple[int, str]]:
    models = tmp_path_factory.mktemp("quality")
    token = secrets.token_hex(32)
    proc, port = _spawn(models, token)
    try:
        yield port, token
    finally:
        _kill(proc)


def _request(
    port: int,
    method: str,
    path: str,
    *,
    token: Optional[str] = None,
    body: Optional[str] = None,
    host: Optional[str] = None,
) -> Tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-GT-Token"] = token
    if host is not None:
        headers["Host"] = host
    try:
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {"raw": raw.decode("utf-8", "replace")}
        return response.status, payload
    finally:
        conn.close()


def _job(h: int = 48, w: int = 64) -> Tuple[dict, np.ndarray, np.ndarray]:
    image = np.full((h, w, 3), 235, np.uint8)
    image[16:32, 20:44] = 20
    mask = np.zeros((h, w), np.uint8)
    mask[16:32, 20:44] = 255
    body = {
        "job_id": "p3-1a2b3c",
        "image": P.encode_png_rgb(image),
        "mask": P.encode_png_mask(mask),
        "params": {"feather_px": 2},
    }
    return body, image, mask


# --- happy path --------------------------------------------------------------


def test_health(server: Tuple[int, str]) -> None:
    port, token = server
    status, payload = _request(port, "GET", "/health", token=token)
    assert status == 200
    assert payload["ok"] is True
    assert payload["version"] == P.VERSION
    assert payload["mode"] == "fake"
    assert payload["models"]["lama"] == "ready"
    assert payload["vram_total_mb"] is None
    assert payload["uptime_s"] >= 0.0


def test_inpaint_job(server: Tuple[int, str]) -> None:
    port, token = server
    body, image, mask = _job()
    status, payload = _request(
        port, "POST", "/inpaint", token=token, body=json.dumps(body)
    )
    assert status == 200, payload
    assert payload["job_id"] == "p3-1a2b3c"
    assert payload["mode"] == "fake"
    assert payload["stages"] == ["lama", "sdxl"]
    assert payload["size"] == [64, 48]
    assert set(payload["timings_ms"]) == {
        "decode", "lama", "sdxl", "composite", "total"
    }
    result = P.decode_png_rgb(payload["image"])
    assert result.shape == image.shape
    assert np.array_equal(result[mask == 0], image[mask == 0])  # the hard guarantee
    assert np.all(result[22:26, 28:36] == 235)  # the blob was filled


def test_inpaint_without_params_uses_the_defaults(server: Tuple[int, str]) -> None:
    port, token = server
    body, _, _ = _job()
    del body["params"]
    status, payload = _request(
        port, "POST", "/inpaint", token=token, body=json.dumps(body)
    )
    assert status == 200 and payload["stages"] == ["lama", "sdxl"]


# --- refusals ----------------------------------------------------------------


def test_a_wrong_token_is_401(server: Tuple[int, str]) -> None:
    port, _ = server
    status, payload = _request(port, "GET", "/health", token="0" * 64)
    assert status == 401 and payload == {"error": "unauthorized"}


def test_a_missing_token_is_401(server: Tuple[int, str]) -> None:
    port, _ = server
    status, payload = _request(port, "GET", "/health")
    assert status == 401 and payload == {"error": "unauthorized"}


def test_a_foreign_host_header_is_403(server: Tuple[int, str]) -> None:
    port, token = server
    status, payload = _request(
        port, "GET", "/health", token=token, host="evil.example.com"
    )
    assert status == 403 and "error" in payload


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_a_loopback_host_header_is_accepted(server: Tuple[int, str], host: str) -> None:
    port, token = server
    status, _ = _request(port, "GET", "/health", token=token, host=f"{host}:{port}")
    assert status == 200


def test_bad_json_is_400(server: Tuple[int, str]) -> None:
    port, token = server
    status, payload = _request(port, "POST", "/inpaint", token=token, body="{not json")
    assert status == 400 and "error" in payload
    assert "Traceback" not in json.dumps(payload)


def test_a_bad_schema_is_400(server: Tuple[int, str]) -> None:
    port, token = server
    body = json.dumps({"job_id": "x", "image": "zzz", "mask": "zzz"})
    status, payload = _request(port, "POST", "/inpaint", token=token, body=body)
    assert status == 400 and payload["error"].startswith("image:")


def test_an_unknown_path_is_404(server: Tuple[int, str]) -> None:
    port, token = server
    status, _ = _request(port, "GET", "/nope", token=token)
    assert status == 404


def test_an_oversized_body_is_413(server: Tuple[int, str]) -> None:
    port, token = server
    # Announce more than the cap without sending it: the server must refuse on
    # the Content-Length alone, never buffer 64 MiB.
    request = (
        f"POST /inpaint HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        f"X-GT-Token: {token}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {P.MAX_BODY_BYTES + 1}\r\n\r\n"
    ).encode()
    with socket.create_connection(("127.0.0.1", port), timeout=30) as sock:
        sock.sendall(request + b"{}")
        sock.settimeout(30)
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = sock.recv(4096)
            if not chunk:
                break
            head += chunk
    assert head.startswith(b"HTTP/1.1 413"), head[:80]


# --- lifecycle ---------------------------------------------------------------


def test_shutdown_endpoint_exits(tmp_path: Path) -> None:
    token = secrets.token_hex(32)
    proc, port = _spawn(tmp_path, token)
    try:
        status, payload = _request(port, "POST", "/shutdown", token=token, body="{}")
        assert status == 200 and payload == {"ok": True}
        assert proc.wait(timeout=EXIT_TIMEOUT_S) == 0
    finally:
        _kill(proc)


def test_closing_stdin_exits(tmp_path: Path) -> None:
    token = secrets.token_hex(32)
    proc, _ = _spawn(tmp_path, token)
    try:
        proc.stdin.close()
        assert proc.wait(timeout=EXIT_TIMEOUT_S) == 0
    finally:
        _kill(proc)


def test_serve_without_a_token_refuses_to_start(tmp_path: Path) -> None:
    env = dict(os.environ)
    env.pop("GT_RENDERER_TOKEN", None)
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(RENDERER_DIR)
    done = subprocess.run(
        [sys.executable, "-m", "glassrenderer", "serve", "--fake",
         "--models-dir", str(tmp_path)],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert done.returncode == 2
    assert "GT_RENDERER_TOKEN" in done.stderr
    assert "READY" not in done.stdout


def test_selftest_runs_a_fake_job(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(RENDERER_DIR)
    done = subprocess.run(
        [sys.executable, "-m", "glassrenderer", "selftest", "--fake"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert "total" in done.stdout
