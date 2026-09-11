"""Command line of the sidecar: ``serve`` and ``selftest``.

``serve`` is what the app spawns (see ``renderer/PROTOCOL.md``)::

    <python> -m glassrenderer serve --models-dir DIR [--fake]
             [--device cuda|cpu|auto] [--port N]

``selftest`` is for a human: it starts the server in-process on an OS-chosen
port, runs one job through the whole HTTP path and prints the timings, so the
integrator can tell "the sidecar works" from "the app cannot talk to it"
without wiring anything up::

    <python> -m glassrenderer selftest --fake
"""
from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import secrets
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import protocol
from .pipeline import Renderer
from .server import configure_logging, create_server, serve

DEVICES: Tuple[str, ...] = ("cuda", "cpu", "auto")
SELFTEST_SIZE = (512, 768)  # height, width - one bucket exactly


def build_parser() -> argparse.ArgumentParser:
    """The ``glassrenderer`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="glassrenderer", description="GlassTranslate quality renderer sidecar"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve_cmd = subcommands.add_parser("serve", help="run the loopback HTTP server")
    serve_cmd.add_argument("--models-dir", required=True, type=Path,
                           help="directory holding the downloaded model files")
    serve_cmd.add_argument("--fake", action="store_true",
                           help="numpy stand-ins instead of the GPU stages")
    serve_cmd.add_argument("--device", choices=DEVICES, default="auto")
    serve_cmd.add_argument("--port", type=int, default=0,
                           help="0 (the default) lets the OS choose")

    selftest_cmd = subcommands.add_parser(
        "selftest", help="run one job in-process and print the timings"
    )
    selftest_cmd.add_argument("--fake", action="store_true")
    selftest_cmd.add_argument("--models-dir", type=Path, default=None)
    selftest_cmd.add_argument("--device", choices=DEVICES, default="auto")
    return parser


def _sample_job() -> Dict[str, Any]:
    """A synthetic panel: grey paper, a dark text block, and its mask."""
    height, width = SELFTEST_SIZE
    image = np.full((height, width, 3), 232, np.uint8)
    image[::37, :] = 96  # a little structure for the stages to keep
    image[120:260, 180:420] = 24
    mask = np.zeros((height, width), np.uint8)
    mask[120:260, 180:420] = 255
    return {
        "job_id": "selftest-1",
        "image": protocol.encode_png_rgb(image),
        "mask": protocol.encode_png_mask(mask),
        "params": {},
    }


def _call(port: int, token: str, method: str, path: str, body: Optional[str]) -> Any:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    try:
        connection.request(
            method, path, body=body,
            headers={"X-GT-Token": token, "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        if response.status != 200:
            raise SystemExit(f"{path} answered {response.status}: {payload}")
        return payload
    finally:
        connection.close()


def _print_report(health: Dict[str, Any], result: Dict[str, Any], wall_ms: float) -> None:
    print(f"mode    : {health['mode']}  device: {health['device']}")
    print(f"models  : {health['models']}")
    print(f"stages  : {result['stages']}  size: {result['size']}")
    for name, value in result["timings_ms"].items():
        print(f"{name:>8}: {value:9.1f} ms")
    print(f"{'wall':>8}: {wall_ms:9.1f} ms")


def run_selftest(args: argparse.Namespace) -> int:
    """Start the server in-process, run one job, print what it cost."""
    configure_logging()
    models_dir = args.models_dir or Path(tempfile.gettempdir()) / "glassrenderer-selftest"
    models_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    renderer = Renderer(models_dir, device=args.device, fake=args.fake)
    httpd = create_server(renderer, token, 0)
    worker = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    worker.start()
    try:
        port = int(httpd.server_address[1])
        health = _call(port, token, "GET", "/health", None)
        started = time.perf_counter()
        result = _call(port, token, "POST", "/inpaint", json.dumps(_sample_job()))
        wall_ms = (time.perf_counter() - started) * 1000.0
        _print_report(health, result, wall_ms)
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=5.0)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of ``python -m glassrenderer``."""
    args = build_parser().parse_args(argv)
    if args.command == "selftest":
        return run_selftest(args)
    configure_logging()
    return serve(
        args.models_dir, fake=args.fake, device=args.device, port=args.port
    )


def _exit(code: int) -> None:
    """Leave the process without running the interpreter's finalisation.

    The daemon threads (the polling stdin watcher, the preload thread with
    torch's CUDA state) have nothing to tear down, and CPython's finalisation
    with torch resident is slow and has died with an access violation
    (0xC0000005) instead of the exit code the parent is waiting for.  Nothing
    here owns unflushed state beyond the two streams and the log handlers, so
    flushing them and calling :func:`os._exit` is both safe and exact.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # pragma: no cover - a closed pipe is not our problem
            pass
    logging.shutdown()
    os._exit(code)


__all__: List[str] = ["build_parser", "main", "run_selftest"]

if __name__ == "__main__":
    _exit(main())
