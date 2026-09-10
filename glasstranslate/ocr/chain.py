"""``FallbackOCREngine``: a primary OCR engine with an always-available fallback.

The manga-ocr recogniser is the better reader of Japanese comic text but it
depends on downloaded models and a working DirectML stack; the PaddleOCR
engine (``rapidocr`` running PP-OCRv5) is always installed.  This wrapper
makes the pair look like one :class:`OCREngine` to the pipeline and keeps the
pipeline alive whatever the primary does:

* the primary is constructed **lazily** on the first ``warmup``/``recognize``
  so construction failures (missing models -> ``FileNotFoundError``, a DML
  provider error, an import error) are handled here, not by the caller;
* any exception from the primary (construction, ``warmup`` or a per-call
  failure) switches the chain to the fallback and reports that **once**
  through ``status``; the chain then stays on the fallback until
  ``retry_after_s`` has elapsed, retries the primary once, and so on; a
  successful retry reports "restored" once;
* if the fallback itself raises, ``recognize`` returns ``[]`` (reported once
  per outage) instead of propagating - the pipeline must never die on OCR.

Empty output is **not** a fallback trigger
------------------------------------------
``[]`` from the primary is a legitimate "no text in this crop" answer.
Running the fallback on every empty result would OCR every text-free frame
twice and halve the frame rate for nothing, so only exceptions and
unavailability switch engines.

``name``/``device`` report the engine that produced the **last** result, so
``PipelineStats.extra["ocr"]`` and the "OCR: <name> on <device>" status line
name the engine actually in use.  Not thread-safe: the pipeline calls it from
its single worker thread only.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional

import numpy as np

from glasstranslate.core.interfaces import OCREngine
from glasstranslate.core.types import Segment

log = logging.getLogger(__name__)

StatusCallback = Callable[[str], None]


class FallbackOCREngine(OCREngine):
    """Primary-with-fallback OCR engine (see the module docstring).

    Args:
        primary_factory: builds the primary engine; called lazily and again
            after each back-off while the primary is unavailable.
        fallback: the always-available engine used while the primary is
            unavailable.  Constructed by the caller (it doubles as the
            manga-ocr detector, so it exists anyway).
        primary_name: config name of the primary (``"mangaocr"``), used in
            status lines and as ``name`` while the primary is active.
        status: receives one human-readable line per state transition.
        retry_after_s: how long the chain stays on the fallback before it
            gives the primary another chance.
        clock: monotonic time source (injected by tests).
    """

    def __init__(
        self,
        primary_factory: Callable[[], OCREngine],
        fallback: OCREngine,
        *,
        primary_name: str,
        status: Optional[StatusCallback] = None,
        retry_after_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not primary_name:
            raise ValueError("primary_name must not be empty")
        if retry_after_s < 0:
            raise ValueError(f"retry_after_s must be >= 0, got {retry_after_s!r}")
        self._primary_factory = primary_factory
        self.fallback = fallback
        self.primary_name = str(primary_name)
        self._status = status
        self._retry_after_s = float(retry_after_s)
        self._clock = clock
        self._primary: Optional[OCREngine] = None
        self._failed_at: Optional[float] = None  # clock time of the last primary failure
        self._on_fallback = False  # a "failed; using <fallback>" line has been issued
        self._fallback_broken = False  # a "<fallback> failed" line has been issued
        # Until the primary has produced a result the fallback is the engine
        # the pipeline would get, so that is what ``name`` reports.
        self.active_engine: str = fallback.name

    # ---------------------------------------------------------------- identity
    @property
    def name(self) -> str:  # type: ignore[override]
        """Config name of the engine that produced the last result."""
        return self.active_engine

    @property
    def device(self) -> str:  # type: ignore[override]
        """Device of the engine that produced the last result."""
        return self._active().device

    @property
    def primary_available(self) -> bool:
        """True when the primary is built and not in a failure back-off."""
        return self._primary is not None and self._failed_at is None

    def _active(self) -> OCREngine:
        if self.active_engine == self.primary_name and self._primary is not None:
            return self._primary
        return self.fallback

    # ---------------------------------------------------------------- OCREngine
    def recognize(self, image_bgr: np.ndarray) -> List[Segment]:
        """OCR with the primary when it is usable, else with the fallback.
        Never raises (see the module docstring)."""
        primary = self._usable_primary()
        if primary is not None:
            try:
                result = primary.recognize(image_bgr)
            except Exception as exc:  # noqa: BLE001 - any failure means fall back
                self._primary_failed(exc)
            else:
                self._primary_recovered()
                return result
        return self._recognize_with_fallback(image_bgr)

    def warmup(self) -> None:
        """Warm the fallback (errors propagate: it is the baseline the
        pipeline relies on), then the primary (errors switch to the fallback)."""
        self.fallback.warmup()
        primary = self._usable_primary()
        if primary is None:
            return
        try:
            primary.warmup()
        except Exception as exc:  # noqa: BLE001
            self._primary_failed(exc)
        else:
            self._primary_recovered()

    def close(self) -> None:
        """Close both engines if they support it; never builds the primary."""
        for engine in (self._primary, self.fallback):
            close = getattr(engine, "close", None)
            if engine is None or not callable(close):
                continue
            try:
                close()
            except Exception:  # pragma: no cover - best effort
                log.exception("closing OCR engine %s failed", engine.name)

    # ---------------------------------------------------------------- primary state
    def _usable_primary(self) -> Optional[OCREngine]:
        """The primary engine when it may be used now: healthy, or past the
        back-off so it deserves another try (constructing it if needed)."""
        if self._failed_at is not None and self._clock() - self._failed_at < self._retry_after_s:
            return None
        if self._primary is None:
            try:
                self._primary = self._primary_factory()
            except Exception as exc:  # noqa: BLE001 - construction failures are the point
                self._primary_failed(exc)
                return None
        return self._primary

    def _primary_failed(self, exc: BaseException) -> None:
        self._failed_at = self._clock()
        self.active_engine = self.fallback.name
        log.warning("OCR engine %s failed (%s: %s); using %s", self.primary_name, type(exc).__name__, exc, self.fallback.name)
        if self._on_fallback:
            return
        self._on_fallback = True
        self._report(f"OCR: {self.primary_name} failed ({type(exc).__name__}); using {self.fallback.name}")

    def _primary_recovered(self) -> None:
        self._failed_at = None
        self.active_engine = self.primary_name
        if not self._on_fallback:
            return
        self._on_fallback = False
        self._report(f"OCR: {self.primary_name} restored")

    # ---------------------------------------------------------------- fallback
    def _recognize_with_fallback(self, image_bgr: np.ndarray) -> List[Segment]:
        self.active_engine = self.fallback.name
        try:
            result = self.fallback.recognize(image_bgr)
        except Exception as exc:  # noqa: BLE001 - the pipeline must not die
            log.exception("fallback OCR engine %s failed", self.fallback.name)
            if not self._fallback_broken:
                self._fallback_broken = True
                self._report(f"OCR: {self.fallback.name} failed ({type(exc).__name__}); no text recognised")
            return []
        self._fallback_broken = False
        return result

    def _report(self, message: str) -> None:
        if self._status is None:
            log.info(message)  # nobody else will surface it
            return
        try:
            self._status(message)
        except Exception:  # pragma: no cover - a UI callback must not break OCR
            log.exception("OCR status callback failed")
