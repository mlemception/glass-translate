"""Overlay capture mode: let screenshot / screen-capture tools see the glass.

Normally the overlay carries ``WDA_EXCLUDEFROMCAPTURE`` so the pipeline's own grabs (and
every other capturer) look *through* it.  Capture mode reverses that for the overlay only,
and freezes the pipeline while it is on: with the lettering visible to capture, a running
pipeline would OCR its own output and loop.  The last painted segments stay on screen, the
GUI and any in-flight translation keep going, and nothing is ever persisted - a new session
always starts with capture mode off (the flag defeats a protection the user may forget).

``CaptureMode`` is deliberately independent of Qt so it is unit-testable with fakes; the
application wires it to the real overlay, pipeline and status strip.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Protocol

__all__ = ["CaptureMode", "CaptureTarget", "PausablePipeline"]

log = logging.getLogger(__name__)

ON_MESSAGE = "Capture mode: the glass is visible to screen capture and the pipeline is frozen until you turn it off"
OFF_MESSAGE = "Capture mode off: glass hidden from capture again"


class CaptureTarget(Protocol):
    def set_capture_excluded(self, excluded: bool) -> None: ...


class PausablePipeline(Protocol):
    def is_alive(self) -> bool: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...


class CaptureMode:
    """Runtime-only capture-mode state machine (see the module docstring).

    ``pipeline`` and ``wants_running`` are callables because both change over the session:
    the pipeline object is created lazily by the app, and ``wants_running`` is the user's
    Start/Stop intent (``AppConfig.running_on_start``), which decides whether leaving capture
    mode resumes anything.
    """

    def __init__(
        self,
        overlay: CaptureTarget,
        *,
        pipeline: Callable[[], Optional[PausablePipeline]],
        wants_running: Callable[[], bool],
        status: Callable[[str], None],
    ) -> None:
        self._overlay = overlay
        self._pipeline = pipeline
        self._wants_running = wants_running
        self._status = status
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def set_active(self, on: bool) -> None:
        """Enter or leave capture mode; idempotent."""
        on = bool(on)
        if on == self._active:
            return
        self._active = on
        if on:
            self._overlay.set_capture_excluded(False)
            self.hold()
            log.info("capture mode on: overlay capturable, pipeline paused")
            self._status(ON_MESSAGE)
            return
        self._overlay.set_capture_excluded(True)
        pipeline = self._pipeline()
        if pipeline is not None and pipeline.is_alive() and self._wants_running():
            pipeline.resume()
        log.info("capture mode off: overlay excluded from capture again")
        self._status(OFF_MESSAGE)

    def hold(self) -> None:
        """Pause the current pipeline while capture mode is on (call after starting one)."""
        if not self._active:
            return
        pipeline = self._pipeline()
        if pipeline is not None:
            pipeline.pause()

    def restore(self) -> None:
        """Teardown: re-exclude the overlay without resuming anything; idempotent."""
        if not self._active:
            return
        self._active = False
        self._overlay.set_capture_excluded(True)
        log.info("capture mode cleared at shutdown")
