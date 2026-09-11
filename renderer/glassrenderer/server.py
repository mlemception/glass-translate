"""The loopback HTTP front end of the sidecar (``renderer/PROTOCOL.md`` v1).

A threading ``http.server`` bound to ``127.0.0.1`` and nothing else.  The
defences are deliberately boring and all of them are tested:

* the per-session token comes from ``GT_RENDERER_TOKEN`` (never the command
  line, which is visible in a process list); without it the process refuses to
  start with exit code 2,
* every request is compared against it with :func:`hmac.compare_digest`,
* a ``Host`` header that is not ``127.0.0.1[:port]`` / ``localhost[:port]`` is
  refused with ``403`` (DNS rebinding),
* a ``Content-Length`` above 64 MiB is refused with ``413`` before a single
  byte of the body is read,
* no exception ever reaches the client as a traceback: it is logged to stderr
  and answered as ``{"error": "<type>: <message>"}``.

``/inpaint`` jobs are serialised by one lock - there is one GPU.  ``stdout``
carries exactly one line, ``READY <port>``; every log line goes to stderr, so
the parent can parse stdout without a filter.  The process exits on
``POST /shutdown``, on stdin EOF (the parent died) and on SIGTERM / SIGINT.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .pipeline import Renderer, RendererError
from .protocol import MAX_BODY_BYTES, InpaintRequest, InpaintResponse, ProtocolError

log = logging.getLogger(__name__)

TOKEN_ENV = "GT_RENDERER_TOKEN"
BIND_HOST = "127.0.0.1"
ALLOWED_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})
READY_PREFIX = "READY"


def configure_logging(level: int = logging.INFO) -> None:
    """Send the sidecar's logs to stderr; stdout is reserved for ``READY``."""
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class RendererServer(ThreadingHTTPServer):
    """A loopback-only threading server that can stop itself from a handler."""

    daemon_threads = True
    allow_reuse_address = False  # loopback: never let another process take over

    def server_bind(self) -> None:
        # On Windows ``allow_reuse_address = False`` alone does not stop another local
        # process from binding the same port with SO_REUSEADDR; exclusive use does.
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        super().server_bind()

    def trigger_shutdown(self) -> None:
        """Stop :meth:`serve_forever` from anywhere without deadlocking."""
        threading.Thread(target=self.shutdown, daemon=True).start()


class _Handler(BaseHTTPRequestHandler):
    """Base request handler; :func:`make_handler` binds the renderer to it."""

    protocol_version = "HTTP/1.1"
    server_version = "glassrenderer"
    sys_version = ""
    # Per-operation socket timeout: a connection that never sends its request cannot park a
    # handler thread forever.  Job time is not bounded by this (the request is fully read
    # before the job runs and the response is written after it).
    timeout = 30.0

    renderer: Renderer
    token: bytes
    job_lock: threading.Lock

    # --- plumbing ------------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        """Route the stdlib's access log to stderr through :mod:`logging`."""
        log.info("%s %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload: Dict[str, Any], *, close: bool = False) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if close:
            self.close_connection = True
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, *, close: bool = False) -> None:
        self._send(status, {"error": message}, close=close)

    # --- guards --------------------------------------------------------------

    def _host_allowed(self) -> bool:
        host = (self.headers.get("Host") or "").strip()
        if not host:
            return False
        if host.startswith("["):  # [::1]:port
            name = host[1:].split("]", 1)[0]
        else:
            name = host.rsplit(":", 1)[0] if ":" in host else host
        return name.lower() in ALLOWED_HOSTNAMES

    def _token_ok(self) -> bool:
        supplied = (self.headers.get("X-GT-Token") or "").encode("utf-8", "ignore")
        return hmac.compare_digest(supplied, self.token)

    def _guard(self) -> bool:
        """``True`` when the request may proceed; otherwise it is answered."""
        if not self._host_allowed():
            self._error(403, "forbidden host", close=True)
            return False
        if not self._token_ok():
            self._error(401, "unauthorized", close=True)
            return False
        return True

    def _read_body(self) -> Optional[bytes]:
        """The request body, or ``None`` when the request was already refused."""
        raw = self.headers.get("Content-Length")
        try:
            length = int(raw) if raw else 0
        except ValueError:
            self._error(400, "Content-Length: not an integer", close=True)
            return None
        if length < 0:
            self._error(400, "Content-Length: negative", close=True)
            return None
        if length > MAX_BODY_BYTES:
            self._error(413, f"body larger than {MAX_BODY_BYTES} bytes", close=True)
            return None
        return self.rfile.read(length) if length else b""

    # --- routes --------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        """``GET /health``."""
        if not self._guard():
            return
        if self.path.split("?", 1)[0] != "/health":
            self._error(404, "not found")
            return
        self._respond(self.renderer.health)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        """``POST /inpaint`` and ``POST /shutdown``."""
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        if path not in ("/inpaint", "/shutdown"):
            if self._read_body() is None:  # already refused (413 / bad length)
                return
            self._error(404, "not found")
            return
        body = self._read_body()
        if body is None:
            return
        if path == "/shutdown":
            log.info("shutdown requested")
            self._send(200, {"ok": True}, close=True)
            self.server.trigger_shutdown()  # type: ignore[attr-defined]
            return
        self._respond(lambda: self._inpaint(body))

    def _respond(self, work: Any) -> None:
        """Run ``work`` and map every failure onto the contract's error bodies."""
        try:
            self._send(200, work())
        except ProtocolError as exc:
            self._error(400, str(exc))
        except RendererError as exc:
            log.error("job failed: %s", exc)
            self._error(500, str(exc))
        except Exception as exc:  # never leak a traceback to the client
            log.exception("unhandled error while serving %s", self.path)
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _inpaint(self, body: bytes) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"body: invalid JSON ({exc})") from None
        request = InpaintRequest.from_payload(payload)
        decode_ms = (time.perf_counter() - started) * 1000.0
        with self.job_lock:  # one GPU: one job at a time
            image, timings, stages = self.renderer.inpaint(
                request.image, request.mask, request.params, decode_ms=decode_ms
            )
        log.info("job %s done in %.0f ms %s", request.job_id, timings["total"], stages)
        return InpaintResponse(
            job_id=request.job_id,
            image=image,
            timings_ms=timings,
            mode=self.renderer.mode,
            stages=stages,
        ).to_payload()


def make_handler(renderer: Renderer, token: str) -> Type[_Handler]:
    """A handler class bound to ``renderer`` and the session ``token``."""
    return type(
        "GlassRendererHandler",
        (_Handler,),
        {
            "renderer": renderer,
            "token": token.encode("utf-8"),
            "job_lock": threading.Lock(),
        },
    )


def create_server(renderer: Renderer, token: str, port: int = 0) -> RendererServer:
    """Bind a :class:`RendererServer` on ``127.0.0.1:port`` (0 = OS-chosen)."""
    return RendererServer((BIND_HOST, int(port)), make_handler(renderer, token))


STDIN_POLL_S = 0.25  # the stdin watcher polls; see _stdin_at_eof for why it never blocks
# PeekNamedPipe failures that mean the parent's end of the pipe is gone:
# ERROR_INVALID_HANDLE, ERROR_BROKEN_PIPE, ERROR_NO_DATA, ERROR_PIPE_NOT_CONNECTED.
_PIPE_GONE_ERRORS = frozenset({6, 109, 232, 233})
_peek_named_pipe: Any = None


def _win32_pipe_pending(fd: int) -> Optional[int]:
    """Bytes waiting on the pipe behind ``fd``; None once the pipe is broken.

    Raises OSError when ``fd`` is not a pipe (a console or NUL), which the caller treats
    as "never at EOF" - there is no parent to outlive in that case.
    """
    global _peek_named_pipe
    import ctypes
    import msvcrt
    from ctypes import wintypes

    if _peek_named_pipe is None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        peek = kernel32.PeekNamedPipe
        peek.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                         wintypes.LPDWORD, wintypes.LPDWORD, wintypes.LPDWORD]
        peek.restype = wintypes.BOOL
        _peek_named_pipe = peek
    available = wintypes.DWORD(0)
    handle = wintypes.HANDLE(msvcrt.get_osfhandle(fd))
    if _peek_named_pipe(handle, None, 0, None, ctypes.byref(available), None):
        return int(available.value)
    error = ctypes.get_last_error()
    if error in _PIPE_GONE_ERRORS:
        return None
    raise OSError(error, "PeekNamedPipe failed")


def _stdin_at_eof() -> bool:
    """True once the parent closed our stdin, decided without ever blocking.

    A thread parked in a blocking ``read()`` on the stdin pipe is not an option here:
    while torch and diffusers load in the preload thread, that synchronous pipe read
    stops every new thread from starting - the accept loop hangs in ``Thread.start`` and
    ``ExitProcess`` never completes - so the sidecar answered nothing for the whole
    warm-up.  Polling with ``PeekNamedPipe`` (Windows) or ``select`` keeps the contract
    of PROTOCOL.md (exit on stdin EOF) without a parked read.  Bytes the parent writes
    are read and discarded.  A console or NUL stdin is never at EOF.
    """
    stream = sys.stdin
    if stream is None or stream.closed:
        return True
    try:
        fd = stream.fileno()
    except (OSError, ValueError):
        return True
    if sys.platform == "win32":
        try:
            pending = _win32_pipe_pending(fd)
        except OSError:
            return False
        if pending is None:
            return True
        if pending == 0:
            return False
    else:
        import select

        readable, _, _ = select.select([fd], [], [], 0)
        if not readable:
            return False
    try:
        return os.read(fd, 4096) == b""
    except OSError:
        return True


def watch_stdin(httpd: RendererServer) -> threading.Thread:
    """Stop the server when stdin reaches EOF, i.e. when the parent died.

    Polls every :data:`STDIN_POLL_S` seconds through :func:`_stdin_at_eof`.
    """

    def _watch() -> None:
        try:
            while not _stdin_at_eof():
                time.sleep(STDIN_POLL_S)
        except Exception:  # pragma: no cover - a torn-down stdin is an EOF too
            pass
        log.info("stdin closed; shutting down")
        httpd.trigger_shutdown()

    thread = threading.Thread(target=_watch, name="stdin-watch", daemon=True)
    thread.start()
    return thread


def install_signal_handlers(httpd: RendererServer) -> None:
    """Shut down cleanly on SIGTERM / SIGINT where the platform allows it."""

    def _handle(signum: int, _frame: Any) -> None:
        log.info("signal %s; shutting down", signum)
        httpd.trigger_shutdown()

    for name in ("SIGTERM", "SIGINT"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            signal.signal(number, _handle)
        except (ValueError, OSError):  # pragma: no cover - not the main thread
            log.debug("could not install a handler for %s", name)


def read_token(env: Optional[Dict[str, str]] = None) -> str:
    """The session token from :data:`TOKEN_ENV`, or ``""`` when unset."""
    source = os.environ if env is None else env
    return (source.get(TOKEN_ENV) or "").strip()


def serve(
    models_dir: Path,
    *,
    fake: bool = False,
    device: str = "auto",
    port: int = 0,
    token: Optional[str] = None,
) -> int:
    """Run the sidecar until it is asked to stop.  Returns the exit code."""
    session_token = read_token() if token is None else token
    if not session_token:
        print(
            f"{TOKEN_ENV} is not set: the sidecar refuses to start without a "
            "per-session token.",
            file=sys.stderr,
            flush=True,
        )
        return 2
    renderer = Renderer(Path(models_dir), device=device, fake=fake)
    httpd = create_server(renderer, session_token, port)
    install_signal_handlers(httpd)
    watch_stdin(httpd)
    log.info(
        "listening on %s:%s (mode=%s device=%s models=%s)",
        BIND_HOST, httpd.server_address[1], renderer.mode, renderer.device, models_dir,
    )
    print(f"{READY_PREFIX} {httpd.server_address[1]}", flush=True)
    if not fake:
        # Weights onto the device and the one-off compiles run while the app already
        # sees READY; /health says "loading" until they are resident.
        threading.Thread(target=renderer.preload, name="preload", daemon=True).start()
    try:
        httpd.serve_forever(poll_interval=0.2)
    finally:
        httpd.server_close()
    log.info("stopped")
    return 0


__all__: List[str] = [
    "TOKEN_ENV",
    "BIND_HOST",
    "RendererServer",
    "configure_logging",
    "create_server",
    "install_signal_handlers",
    "make_handler",
    "read_token",
    "serve",
    "watch_stdin",
]
