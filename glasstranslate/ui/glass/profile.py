"""Opt-in event profiler for the glass control window (feature batch F2-0).

Set ``GLASSTRANSLATE_PROFILE=1`` before the process starts and every hook point in
``control.py`` / ``backdrop.py`` records a ``(perf_counter, kind, payload)`` event into a
bounded, thread-safe deque.  When the variable is unset (or ``"0"``) the module-level
``profiler`` is a :class:`NoopProfiler` whose methods do nothing, so a call site costs one
attribute lookup and one empty call - no timestamps, no allocations.

Event kinds recorded by the hooks (see ``docs/perf/2026-09-09-glass-baseline.md``):

* ``move``       window moved (physical rect ``x, y, w, h``)                  GUI thread
* ``poke``       the grabber was asked for an immediate grab                   GUI thread
* ``grab``       one ``_grab_once`` pass (``x, y, emitted, ms``)               grabber thread
* ``frame_gui``  a frame reached the GUI thread (``ts`` = grab time, ``fx, fy`` = the rect it
                 was grabbed for, ``cx, cy`` = the window rect *now*, ``dx, dy`` = stale offset)
* ``push_backdrop_ms``  duration of ``ControlBridge.push_backdrop`` (``ms``)
* ``swap``       ``QQuickWindow.frameSwapped``, queued to the GUI thread (a direct Python slot
                 on the render thread deadlocks against the GUI thread's GIL): exact counts,
                 GUI-dispatch timestamps                                        GUI thread
* ``tab``        ``tabBar.currentIndex`` changed (``index``)                   GUI thread

``dump(path)`` writes one JSON object per line; ``summary()`` counts events per kind and
computes latency percentiles for the paired kinds in :data:`PAIRS`.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from ...core.types import Rect

__all__ = [
    "MAX_EVENTS",
    "PAIRS",
    "PROFILE_ENV",
    "Event",
    "NoopProfiler",
    "Profiler",
    "enabled",
    "install",
    "mark",
    "pair_latencies_ms",
    "percentiles",
    "profiler",
    "stale_offset",
]

log = logging.getLogger(__name__)

PROFILE_ENV = "GLASSTRANSLATE_PROFILE"
MAX_EVENTS = 20000
# (first, second): for every `second` event, the time since the latest preceding `first`.
PAIRS: Tuple[Tuple[str, str], ...] = (("move", "poke"), ("poke", "grab"), ("move", "frame_gui"), ("grab", "frame_gui"))

Event = Tuple[float, str, Dict[str, Any]]


def enabled() -> bool:
    """True when ``GLASSTRANSLATE_PROFILE`` is set to anything but ``"0"`` / blank."""
    value = os.environ.get(PROFILE_ENV, "").strip()
    return bool(value) and value != "0"


def stale_offset(frame_rect: Rect, current_rect: Rect) -> Tuple[int, int]:
    """How far the window has moved (physical px) since ``frame_rect`` was grabbed for it."""
    return current_rect.x - frame_rect.x, current_rect.y - frame_rect.y


def percentiles(values: Sequence[float]) -> Dict[str, float]:
    """``n / p50 / p90 / max / mean`` of ``values`` (nearest-rank; all zero when empty)."""
    if not values:
        return {"n": 0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    ordered = sorted(float(v) for v in values)
    n = len(ordered)

    def pick(q: float) -> float:
        return ordered[max(0, min(n - 1, math.ceil(q * n) - 1))]

    return {"n": n, "p50": pick(0.5), "p90": pick(0.9), "max": ordered[-1], "mean": sum(ordered) / n}


def pair_latencies_ms(events: Iterable[Event], first: str, second: str) -> List[float]:
    """For each ``second`` event, milliseconds since the most recent earlier ``first`` event."""
    out: List[float] = []
    last: Optional[float] = None
    for t, kind, _payload in sorted(events, key=lambda e: e[0]):
        if kind == first:
            last = t
        elif kind == second and last is not None:
            out.append((t - last) * 1000.0)
    return out


class NoopProfiler:
    """Stand-in when profiling is off: every method is an empty call."""

    enabled = False

    def mark(self, kind: str, **payload: Any) -> None:
        return None

    def geometry(self, kind: str, rect: Optional[Rect]) -> None:
        return None

    def frame_gui(self, frame: Any, current_rect: Callable[[], Rect]) -> None:
        return None

    def timed_call(self, kind: str, fn: Callable[..., Any], *args: Any) -> Any:
        return fn(*args)

    def events(self) -> List[Event]:
        return []

    def clear(self) -> None:
        return None

    def dump(self, path: os.PathLike[str] | str) -> int:
        return 0

    def summary(self) -> Dict[str, Any]:
        return {}


class Profiler(NoopProfiler):
    """Thread-safe bounded event recorder (see the module docstring)."""

    enabled = True

    def __init__(self, maxlen: int = MAX_EVENTS) -> None:
        self._lock = threading.Lock()
        self._events: Deque[Event] = deque(maxlen=maxlen)

    def mark(self, kind: str, **payload: Any) -> None:
        t = time.perf_counter()
        with self._lock:
            self._events.append((t, kind, payload))

    def geometry(self, kind: str, rect: Optional[Rect]) -> None:
        """Record ``kind`` with a physical rect payload (``x, y, w, h``)."""
        if rect is None:
            self.mark(kind)
        else:
            self.mark(kind, x=rect.x, y=rect.y, w=rect.w, h=rect.h)

    def frame_gui(self, frame: Any, current_rect: Callable[[], Rect]) -> None:
        """Record a frame arriving on the GUI thread together with its stale offset."""
        now = current_rect()
        dx, dy = stale_offset(frame.window_rect, now)
        self.mark("frame_gui", ts=frame.timestamp, fx=frame.window_rect.x, fy=frame.window_rect.y,
                  cx=now.x, cy=now.y, dx=dx, dy=dy)

    def timed_call(self, kind: str, fn: Callable[..., Any], *args: Any) -> Any:
        """Call ``fn(*args)`` and record its duration as ``kind`` with ``ms``."""
        t0 = time.perf_counter()
        try:
            return fn(*args)
        finally:
            self.mark(kind, ms=(time.perf_counter() - t0) * 1000.0)

    def events(self) -> List[Event]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def dump(self, path: os.PathLike[str] | str) -> int:
        """Write every event as one JSON line; returns the number of lines written."""
        events = self.events()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for t, kind, payload in events:
                fh.write(json.dumps({"t": t, "kind": kind, **payload}, default=str) + "\n")
        log.info("profile: %d events written to %s", len(events), target)
        return len(events)

    def summary(self) -> Dict[str, Any]:
        """Counts per kind, ``ms`` percentiles per kind that carries one, paired latencies."""
        events = self.events()
        counts = Counter(kind for _t, kind, _p in events)
        ms_by_kind: Dict[str, List[float]] = {}
        for _t, kind, payload in events:
            if "ms" in payload:
                ms_by_kind.setdefault(kind, []).append(float(payload["ms"]))
        latency = {f"{a}->{b}": percentiles(pair_latencies_ms(events, a, b)) for a, b in PAIRS
                   if counts.get(a) and counts.get(b)}
        return {
            "counts": dict(counts),
            "ms": {kind: percentiles(v) for kind, v in ms_by_kind.items()},
            "latency_ms": latency,
        }


profiler: NoopProfiler = NoopProfiler()
mark: Callable[..., None] = profiler.mark


def install(force: Optional[bool] = None) -> NoopProfiler:
    """(Re)create the module singleton from the environment (or ``force``); returns it."""
    global profiler, mark
    on = enabled() if force is None else bool(force)
    profiler = Profiler() if on else NoopProfiler()
    mark = profiler.mark
    if on:
        log.info("glass profiling enabled (%s)", PROFILE_ENV)
    return profiler


install()
