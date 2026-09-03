"""The capture -> diff -> OCR -> style -> detect -> translate worker thread.

:class:`Pipeline` owns every engine and runs one pass per tick at
``cfg.refresh_hz``.  Each pass grabs the glass region, asks the
:class:`~glasstranslate.capture.ChangeDetector` which tiles changed, OCRs only
those (expanded) rectangles, measures the style of the new segments, detects
the dominant source language, translates the cache misses in one batch and
hands the complete set of *live* segments to ``on_result`` together with a
:class:`~glasstranslate.core.types.PipelineStats` latency breakdown.

Threading model
---------------
Everything except the constructor, :meth:`set_config`, :meth:`pause`,
:meth:`resume`, :meth:`invalidate`, :meth:`refresh_models` and :meth:`stop`
runs on the worker thread, including engine construction and hot-swapping.
``on_result`` and ``on_status`` are called **from the worker thread**; the UI
layer marshals them to Qt with signals.  The thread never dies on an
exception: errors are reported through ``on_status`` and the loop retries.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..capture.diff import ChangeDetector
from ..config.settings import AppConfig
from ..translate.cache import TranslationCache
from .interfaces import LanguageDetector, OCREngine, ScreenCapture, Translator
from .types import Frame, PipelineStats, Rect, Segment, StyledSegment, TranslatedSegment

__all__ = ["Pipeline", "ResultCallback", "StatusCallback"]

log = logging.getLogger(__name__)

ResultCallback = Callable[[List[TranslatedSegment], PipelineStats], None]
StatusCallback = Callable[[str], None]

# Dirty rectangles are grown by this many frame pixels before OCR so text that
# straddles a tile boundary is recognised whole.
_DIRTY_EXPAND_PX = 32
# ``ChangeDetector.fraction_changed`` above this means scroll/video: debounce.
_STORM_FRACTION = 0.4
# A debounce frame counts as "stable" below this changed fraction.
_STABLE_FRACTION = 0.002
# When the dirty crops cover more than this share of the frame (or there are
# many of them) a single whole-frame OCR is cheaper than many small ones.
_WHOLE_FRAME_AREA = 0.5
_MAX_CROPS = 6
# Back-off after an exception / failed engine construction.
_ERROR_SLEEP_S = 0.5
_ENGINE_RETRY_S = 5.0
# Languages whose script is unmistakable; a segment detected as one of these
# keeps its own language even when the page's dominant language differs.
_SCRIPT_LANGUAGES = frozenset({"ja", "ko", "zh", "ru", "ar", "el", "th", "he"})
# Source language used when nothing on the page is detectable.
_FALLBACK_SRC = "en"

# Config fields whose change requires rebuilding an engine.
_OCR_FIELDS = ("ocr_engine", "ocr_device", "min_confidence")
_TRANSLATOR_FIELDS = (
    "translation_backend",
    "translation_api_key",
    "translation_api_url",
    "translate_device",
    "models_dir",
)
_DETECTOR_FIELDS = ("tile_size", "change_threshold")
_LANGUAGE_FIELDS = ("source_lang", "target_lang")


def _default_capture_factory(cfg: AppConfig) -> ScreenCapture:
    from ..capture import create_capture

    return create_capture()


def _default_ocr_factory(cfg: AppConfig) -> OCREngine:
    from ..ocr import create_ocr

    return create_ocr(cfg.ocr_engine, cfg.ocr_device, cfg.min_confidence)


def _default_translator_factory(cfg: AppConfig) -> Translator:
    from ..translate import create_translator

    return create_translator(cfg)


def _default_detector_factory() -> LanguageDetector:
    from ..ocr import ScriptLanguageDetector

    return ScriptLanguageDetector()


_UNK_MARKERS = ("<unk>", "\u2047")  # sentencepiece / ctranslate2 unknown-token renderings


def _usable_translation(text: str) -> bool:
    """False when a model produced nothing but unknown tokens or punctuation."""
    stripped = text.strip()
    if not stripped:
        return False
    for marker in _UNK_MARKERS:
        stripped = stripped.replace(marker, "")
    return bool(stripped.strip(" .\u2026,!?"))


def _offset_segment(seg: TranslatedSegment, dx: int, dy: int) -> TranslatedSegment:
    inner = seg.styled.segment
    moved = Segment(
        inner.text, inner.quad + np.array([dx, dy], dtype=np.float32), inner.confidence, inner.lang_hint
    )
    return TranslatedSegment(
        styled=StyledSegment(moved, seg.styled.style, seg.styled.src_lang),
        translation=seg.translation,
        tgt_lang=seg.tgt_lang,
        from_cache=seg.from_cache,
    )


def _expand(rect: Rect, px: int, width: int, height: int) -> Rect:
    """Grow ``rect`` by ``px`` on every side and clamp to ``width x height``."""
    return Rect(rect.x - px, rect.y - px, rect.w + 2 * px, rect.h + 2 * px).clamp(width, height)


def _merge_rects(rects: Sequence[Rect]) -> List[Rect]:
    """Union intersecting rectangles until none intersect (order-independent)."""
    merged = [r for r in rects if r.w > 0 and r.h > 0]
    changed = True
    while changed:
        changed = False
        out: List[Rect] = []
        for r in merged:
            for i, o in enumerate(out):
                if r.intersects(o):
                    out[i] = o.union(r)
                    changed = True
                    break
            else:
                out.append(r)
        merged = out
    return merged


def _offset_segment(seg: Segment, dx: int, dy: int) -> Segment:
    quad = seg.quad.copy()
    quad[:, 0] += dx
    quad[:, 1] += dy
    return Segment(text=seg.text, quad=quad, confidence=seg.confidence, lang_hint=seg.lang_hint)


@dataclass
class _Timer:
    """Accumulates wall-clock durations for the per-stage stats."""

    start: float

    def lap(self) -> float:
        now = time.perf_counter()
        ms = (now - self.start) * 1000.0
        self.start = now
        return ms


class Pipeline(threading.Thread):
    """Background worker that keeps the glass region translated.

    Args:
        cfg: initial configuration (copied; use :meth:`set_config` to change).
        region_provider: returns the glass rectangle in virtual-screen physical
            pixels.  Called once per tick from the worker thread.
        on_result: receives *all* live segments plus the stats of the pass.
            Also called for cheap skipped passes with
            ``stats.skipped_unchanged=True`` and the unchanged segment list.
        on_status: receives human-readable progress / error messages.
        capture_factory, ocr_factory, translator_factory, detector_factory:
            engine constructors; the defaults build the real engines.  Tests
            inject fakes here.
    """

    def __init__(
        self,
        cfg: AppConfig,
        region_provider: Callable[[], Rect],
        on_result: ResultCallback,
        on_status: StatusCallback,
        *,
        capture_factory: Callable[[AppConfig], ScreenCapture] = _default_capture_factory,
        ocr_factory: Callable[[AppConfig], OCREngine] = _default_ocr_factory,
        translator_factory: Callable[[AppConfig], Translator] = _default_translator_factory,
        detector_factory: Callable[[], LanguageDetector] = _default_detector_factory,
    ) -> None:
        super().__init__(name="glasstranslate-pipeline", daemon=True)
        self._cfg = AppConfig.from_dict(cfg.to_dict())
        self._pending_cfg: Optional[AppConfig] = None
        self._region_provider = region_provider
        self._on_result = on_result
        self._on_status = on_status
        self._capture_factory = capture_factory
        self._ocr_factory = ocr_factory
        self._translator_factory = translator_factory
        self._detector_factory = detector_factory

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self._invalidate_requested = True
        self._rebuild_ocr = True
        self._rebuild_translator = True
        self._rebuild_change_detector = True
        self._next_engine_attempt = 0.0

        self._capture: Optional[ScreenCapture] = None
        self._ocr: Optional[OCREngine] = None
        self._translator: Optional[Translator] = None
        self._lang_detector: Optional[LanguageDetector] = None
        self._change_detector: Optional[ChangeDetector] = None
        self.cache = TranslationCache()

        self._live: Dict[int, TranslatedSegment] = {}
        self._next_id = 0
        self._last_region: Optional[Rect] = None
        # (frame.origin - region.origin): non-zero when the capture backend had
        # to clamp the glass rectangle to a monitor.  Applied in _emit so the
        # overlay always receives quads relative to its own top-left corner.
        self._frame_offset = (0, 0)
        self._last_src = _FALLBACK_SRC
        self._last_tick = 0.0
        self._fps = 0.0

    # ------------------------------------------------------------- control
    @property
    def config(self) -> AppConfig:
        """The configuration currently in effect (the pending one if any)."""
        with self._lock:
            return self._pending_cfg or self._cfg

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the worker to finish, wait for it and release the engines."""
        self._stop_event.set()
        self._running.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout)
        if self.is_alive() and threading.current_thread() is not self:
            # Still inside a pass (e.g. a slow model load).  Closing engines
            # under the worker would race; run() releases them when it exits.
            log.warning("pipeline worker did not stop within %.1fs; engines released on exit", timeout)
            return
        self._close_engines()

    def pause(self) -> None:
        """Suspend processing (also valid before :meth:`start`)."""
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    def set_config(self, cfg: AppConfig) -> None:
        """Apply ``cfg`` on the next pass; engines whose settings changed are
        rebuilt in the worker thread."""
        with self._lock:
            self._pending_cfg = AppConfig.from_dict(cfg.to_dict())

    def invalidate(self) -> None:
        """Force a full re-OCR on the next pass (e.g. after the glass moved)."""
        with self._lock:
            self._invalidate_requested = True

    def refresh_models(self) -> None:
        """Rebuild the translator so newly installed packages are picked up."""
        with self._lock:
            self._rebuild_translator = True

    # ---------------------------------------------------------------- loop
    def run(self) -> None:  # noqa: D401 - Thread API
        self._status("Pipeline started")
        while not self._stop_event.is_set():
            if not self._running.wait(timeout=0.1):
                continue
            if self._stop_event.is_set():
                break
            started = time.perf_counter()
            try:
                self.step()
            except Exception as exc:  # never let the worker die
                log.exception("pipeline pass failed")
                self._status(f"Error: {exc}")
                self._stop_event.wait(_ERROR_SLEEP_S)
                continue
            period = 1.0 / max(self._cfg.refresh_hz, 0.5)
            remaining = period - (time.perf_counter() - started)
            if remaining > 0:
                self._stop_event.wait(remaining)
        self._close_engines()
        self._status("Pipeline stopped")

    def step(self) -> Optional[PipelineStats]:
        """Run exactly one pass synchronously (the thread loop calls this).

        Returns the stats of the pass, or None when nothing could be done
        (no region, engines still unavailable, capture returned nothing).
        """
        self._apply_pending_config()
        if not self._ensure_engines():
            return None
        assert self._capture is not None and self._ocr is not None
        assert self._translator is not None and self._change_detector is not None

        stats = PipelineStats(ocr_device=self._ocr.device, translate_device=self._translator.device)
        stats.extra.update(
            capture=self._capture.name, ocr=self._ocr.name, translator=self._translator.name
        )
        timer = _Timer(time.perf_counter())
        now = timer.start
        if self._last_tick:
            inst = 1.0 / max(now - self._last_tick, 1e-6)
            self._fps = inst if not self._fps else 0.8 * self._fps + 0.2 * inst
        self._last_tick = now
        stats.fps = self._fps

        region = self._region_provider()
        if region.w <= 0 or region.h <= 0:
            return None
        with self._lock:
            invalidate = self._invalidate_requested
            self._invalidate_requested = False
        if region != self._last_region:
            self._last_region = region
            invalidate = True
        if invalidate:
            self._change_detector.reset()
            self._live.clear()

        frame = self._capture.grab(region)
        stats.capture_ms = timer.lap()
        if frame is None:
            return None
        offset = (frame.origin_x - region.x, frame.origin_y - region.y)
        if offset != self._frame_offset:
            # Clamping changed (e.g. the glass slid off a monitor edge); the
            # frame grid no longer lines up with the live segments.
            self._frame_offset = offset
            self._change_detector.reset()
            self._live.clear()
            invalidate = True

        dirty = self._change_detector.update(frame.image)
        if dirty and self._change_detector.fraction_changed > _STORM_FRACTION and not invalidate:
            frame, waited_ms = self._settle(region, frame)
            stats.extra["debounced_ms"] = round(waited_ms, 1)
            dirty = [Rect(0, 0, frame.width, frame.height)]
        stats.diff_ms = timer.lap()
        stats.dirty_regions = len(dirty)

        if not dirty:
            stats.skipped_unchanged = True
            stats.segments = len(self._live)
            stats.total_ms = stats.capture_ms + stats.diff_ms
            self._fill_common_stats(stats)
            self._emit(stats)
            return stats

        crops = self._plan_crops(dirty, frame.width, frame.height)
        self._drop_live_intersecting(crops)
        new_segments = self._ocr_crops(frame, crops)
        stats.ocr_ms = timer.lap()

        styled = self._style_and_detect(frame, new_segments)
        stats.style_ms = timer.lap()

        hits0, misses0 = self.cache.hits, self.cache.misses
        translated = self._translate(styled)
        stats.translate_ms = timer.lap()
        stats.cache_hits = self.cache.hits - hits0
        stats.cache_misses = self.cache.misses - misses0

        for seg in translated:
            self._live[self._next_id] = seg
            self._next_id += 1
        stats.segments = len(self._live)
        stats.extra["new_segments"] = len(translated)
        self._fill_common_stats(stats)
        stats.total_ms = (
            stats.capture_ms + stats.diff_ms + stats.ocr_ms + stats.style_ms + stats.translate_ms
        )
        self._emit(stats)
        return stats

    # ------------------------------------------------------------- engines
    def _apply_pending_config(self) -> None:
        with self._lock:
            new = self._pending_cfg
            self._pending_cfg = None
            if new is None:
                return
            old = self._cfg
            self._cfg = new
            if any(getattr(old, f) != getattr(new, f) for f in _OCR_FIELDS):
                self._rebuild_ocr = True
            if any(getattr(old, f) != getattr(new, f) for f in _TRANSLATOR_FIELDS):
                self._rebuild_translator = True
            if any(getattr(old, f) != getattr(new, f) for f in _DETECTOR_FIELDS):
                self._rebuild_change_detector = True
            if any(
                getattr(old, f) != getattr(new, f)
                for f in _OCR_FIELDS + _TRANSLATOR_FIELDS + _DETECTOR_FIELDS + _LANGUAGE_FIELDS
            ):
                self._invalidate_requested = True
            self._next_engine_attempt = 0.0

    def _ensure_engines(self) -> bool:
        """Build or rebuild engines as flagged.  Returns False (after reporting)
        when an engine could not be constructed; retried after a back-off."""
        if time.monotonic() < self._next_engine_attempt:
            return False
        cfg = self._cfg
        try:
            if self._capture is None:
                self._capture = self._capture_factory(cfg)
                self._status(f"Capture: {self._capture.name}")
            if self._lang_detector is None:
                self._lang_detector = self._detector_factory()
            if self._rebuild_change_detector or self._change_detector is None:
                self._change_detector = ChangeDetector(tile=cfg.tile_size, threshold=cfg.change_threshold)
                self._rebuild_change_detector = False
            if self._rebuild_ocr or self._ocr is None:
                self._ocr = None  # release the old engine before loading the new one
                self._status(f"Loading OCR engine {cfg.ocr_engine} ({cfg.ocr_device})...")
                ocr = self._ocr_factory(cfg)
                ocr.warmup()
                self._ocr = ocr
                self._rebuild_ocr = False
                self._status(f"OCR: {ocr.name} on {ocr.device}")
            if self._rebuild_translator or self._translator is None:
                old_tr, self._translator = self._translator, None
                self._status(f"Loading translator {cfg.translation_backend}...")
                self._translator = self._translator_factory(cfg)
                self._rebuild_translator = False
                self.cache.clear()  # entries from the old backend may be stale
                if old_tr is not None:
                    old_tr.close()
                self._status(f"Translator: {self._translator.name} on {self._translator.device}")
        except Exception as exc:
            log.exception("engine construction failed")
            self._status(f"Engine error: {exc}")
            self._next_engine_attempt = time.monotonic() + _ENGINE_RETRY_S
            return False
        return True

    def _close_engines(self) -> None:
        for engine in (self._capture, self._translator):
            if engine is not None:
                try:
                    engine.close()
                except Exception:  # pragma: no cover - best effort
                    log.exception("closing %s failed", engine)
        self._capture = None
        self._translator = None
        self._ocr = None

    # -------------------------------------------------------------- stages
    def _settle(self, region: Rect, frame: Frame) -> tuple[Frame, float]:
        """Scroll/video storm: keep grabbing until two consecutive frames are
        stable or ``debounce_ms`` elapsed.  Returns the last frame and the
        time spent waiting."""
        assert self._capture is not None and self._change_detector is not None
        start = time.perf_counter()
        deadline = start + self._cfg.debounce_ms / 1000.0
        interval = min(0.02, 1.0 / max(self._cfg.refresh_hz, 0.5))
        stable = 0
        while time.perf_counter() < deadline and not self._stop_event.is_set():
            self._stop_event.wait(interval)
            grabbed = self._capture.grab(region)
            if grabbed is None:
                continue
            frame = grabbed
            self._change_detector.update(frame.image)
            if self._change_detector.fraction_changed < _STABLE_FRACTION:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
        return frame, (time.perf_counter() - start) * 1000.0

    def _plan_crops(self, dirty: Sequence[Rect], width: int, height: int) -> List[Rect]:
        """Expanded, merged OCR crops for ``dirty``; grown to cover any live
        segment they touch so that segment is re-read whole."""
        crops = _merge_rects([_expand(r, _DIRTY_EXPAND_PX, width, height) for r in dirty])
        live_boxes = [seg.styled.segment.bbox for seg in self._live.values()]
        # Growing a crop can make it touch further live segments, so iterate
        # to a fixed point: every segment that gets dropped is then fully
        # inside some crop and is re-read whole.
        while True:
            grown: List[Rect] = []
            for crop in crops:
                for bbox in live_boxes:
                    if bbox.intersects(crop):
                        crop = crop.union(bbox).clamp(width, height)
                grown.append(crop)
            grown = _merge_rects(grown)
            if grown == crops:
                break
            crops = grown
        area = sum(c.w * c.h for c in crops)
        if len(crops) > _MAX_CROPS or area > _WHOLE_FRAME_AREA * width * height:
            return [Rect(0, 0, width, height)]
        return crops

    def _drop_live_intersecting(self, crops: Sequence[Rect]) -> None:
        doomed = [
            key
            for key, seg in self._live.items()
            if any(seg.styled.segment.bbox.intersects(c) for c in crops)
        ]
        for key in doomed:
            del self._live[key]

    def _ocr_crops(self, frame: Frame, crops: Sequence[Rect]) -> List[Segment]:
        assert self._ocr is not None
        found: List[Segment] = []
        for crop in crops:
            if crop.w < 2 or crop.h < 2:
                continue
            image = frame.image[crop.y : crop.y2, crop.x : crop.x2]
            for seg in self._ocr.recognize(image):
                found.append(_offset_segment(seg, crop.x, crop.y))
        return found

    def _style_and_detect(self, frame: Frame, segments: Sequence[Segment]) -> List[StyledSegment]:
        if not segments:
            return []
        from ..render.style import measure_style

        assert self._lang_detector is not None
        cfg = self._cfg
        if cfg.source_lang != "auto":
            dominant = cfg.source_lang
        else:
            texts = [s.text for s in segments] + [s.source_text for s in self._live.values()]
            detect_dominant = getattr(self._lang_detector, "detect_dominant", None)
            if callable(detect_dominant):
                dominant = detect_dominant(texts)
            else:
                dominant = self._lang_detector.detect(" ".join(texts))
            dominant = dominant or self._last_src
        self._last_src = dominant

        styled: List[StyledSegment] = []
        for seg in segments:
            lang = dominant
            if cfg.source_lang == "auto":
                own = self._lang_detector.detect(seg.text)
                if own in _SCRIPT_LANGUAGES and own != dominant:
                    lang = own
            styled.append(StyledSegment(segment=seg, style=measure_style(frame.image, seg), src_lang=lang))
        return styled

    def _translate(self, styled: Sequence[StyledSegment]) -> List[TranslatedSegment]:
        if not styled:
            return []
        assert self._translator is not None
        tgt = self._cfg.target_lang
        results: List[Optional[TranslatedSegment]] = [None] * len(styled)
        misses: Dict[str, List[int]] = {}
        for i, s in enumerate(styled):
            text = s.segment.text
            if s.src_lang == tgt or not text.strip():
                results[i] = TranslatedSegment(styled=s, translation=text, tgt_lang=tgt)
                continue
            cached = self.cache.get(text, s.src_lang, tgt)
            if cached is not None:
                results[i] = TranslatedSegment(styled=s, translation=cached, tgt_lang=tgt, from_cache=True)
            else:
                misses.setdefault(s.src_lang, []).append(i)
        for src, indices in misses.items():
            texts = [styled[i].segment.text for i in indices]
            translations = self._translator.translate_batch(texts, src, tgt)
            if len(translations) != len(texts):
                log.warning("translator returned %d results for %d inputs", len(translations), len(texts))
                translations = texts
            for i, translation in zip(indices, translations):
                source = styled[i].segment.text
                if not _usable_translation(translation):
                    # Nothing sensible came back (empty / all <unk>): show the
                    # original rather than blanking the text out.
                    translation = source
                if translation != source:
                    # Backends return the input unchanged when they cannot
                    # translate.  Never cache that, so a model installed later
                    # (refresh_models) takes effect without a restart.
                    self.cache.put(source, src, tgt, translation)
                results[i] = TranslatedSegment(styled=styled[i], translation=translation, tgt_lang=tgt)
        return [r for r in results if r is not None]

    def _fill_common_stats(self, stats: PipelineStats) -> None:
        stats.extra.update(
            src_lang=self._last_src,
            cache_size=len(self.cache),
            cache_hit_rate=round(self.cache.stats()["hit_rate"], 3),
        )

    # ------------------------------------------------------------ callbacks
    def _emit(self, stats: PipelineStats) -> None:
        segments = list(self._live.values())
        dx, dy = self._frame_offset
        if dx or dy:
            segments = [_offset_segment(s, dx, dy) for s in segments]
        try:
            self._on_result(segments, stats)
        except Exception:  # pragma: no cover - UI callback bug must not kill us
            log.exception("on_result callback failed")

    def _status(self, message: str) -> None:
        log.info(message)
        try:
            self._on_status(message)
        except Exception:  # pragma: no cover
            log.exception("on_status callback failed")
