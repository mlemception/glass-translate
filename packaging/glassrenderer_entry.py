"""Frozen entry point of the quality-renderer sidecar (``dist/renderer/glassrenderer.exe``).

``glassrenderer/__main__.py`` cannot be handed to PyInstaller directly: its relative
imports (``from . import protocol``) need the package context, which a top-level
script does not have.  This module is that context-free shim - it does exactly what
``python -m glassrenderer`` does, and nothing else.

The offline environment is installed here rather than left to the caller because
``huggingface_hub`` reads every one of those variables at **import** time, long
before any job arrives: the app already sets them when it spawns the sidecar, and
doing it here as well makes a hand-run ``glassrenderer.exe serve ...`` offline by
construction.  ``setdefault`` throughout, so the parent always wins.

``glassrenderer.__main__`` is imported as a module and its ``main`` / ``_exit`` are
resolved at call time: a ``from ... import main`` would bind the function object at
import time, which is both untestable and one refactor away from calling the wrong
thing.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
from typing import Optional, Sequence, Tuple

# (name, value) pairs huggingface_hub / transformers / tokenizers read at import time.
OFFLINE_ENVIRONMENT: Tuple[Tuple[str, str], ...] = (
    ("HF_HUB_OFFLINE", "1"),
    ("TRANSFORMERS_OFFLINE", "1"),
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
    ("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1"),
    ("HF_HUB_DISABLE_PROGRESS_BARS", "1"),
    ("HF_HUB_DISABLE_XET", "1"),
    ("DO_NOT_TRACK", "1"),
    ("TOKENIZERS_PARALLELISM", "false"),  # Windows has no fork; silences the warning
)


def apply_offline_environment() -> None:
    """Set :data:`OFFLINE_ENVIRONMENT` without overriding what the parent chose."""
    for name, value in OFFLINE_ENVIRONMENT:
        os.environ.setdefault(name, value)


apply_offline_environment()  # must run before the first transformers / diffusers import

# Before the heavy import, not after: a re-spawned multiprocessing child re-runs this
# script from the top, and freeze_support() has to intercept it there - otherwise the
# child pays for importing torch and diffusers before being told it is a worker.
multiprocessing.freeze_support()

from glassrenderer import __main__ as cli  # noqa: E402 - after the environment is set


def run(argv: Optional[Sequence[str]] = None) -> None:
    """Run the CLI over ``argv`` (default ``sys.argv[1:]``) and leave the process.

    :func:`glassrenderer.__main__._exit` is ``os._exit`` after a flush: with torch
    resident, CPython's finalisation is slow and has died with an access violation
    instead of the exit code the parent waits for.
    """
    cli._exit(cli.main(sys.argv[1:] if argv is None else list(argv)))


if __name__ == "__main__":
    run()
