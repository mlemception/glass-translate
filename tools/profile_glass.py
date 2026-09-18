"""Dev-only glass profiling driver (feature batch F2-0; lag baseline 2026-09-18): measure, do not fix.

A single run opens the real windows ON SCREEN with ``GLASSTRANSLATE_PROFILE=1`` and
``GLASSTRANSLATE_APPEARANCE=glass``, waits for exposure and the first backdrop frame, then
plays a script ``--repeat`` times:

1. a drag: ``window.setPosition`` in N steps (default 60 x 8 px over ~1 s via QTimer).  A
   real title-bar drag goes through ``startSystemMove`` (a modal Win32 move loop); the
   programmatic move is the closest scriptable equivalent and exercises the same
   ``moveEvent -> _refresh_geometry -> _poke`` path, just without the WM_MOVING cadence;
2. a tab-switch burst: ``tabBar.currentIndex`` 0 -> 1 -> 2 -> 3 -> 0, 400 ms apart;
3. an idle hold of 2 s;

Every phase boundary is recorded as a ``phase`` mark and a ``QTimer`` records a ``tick`` from
the GUI thread every ``--heartbeat-ms``, so a stall shows up as a gap between marks even when
nothing is being presented.  ``--mode`` picks what the run builds:

* ``control``  (default) - today's bare control window, no config, no pipeline: the baseline.
* ``session``  - the whole app (``GlassTranslateApp``: control window + glass overlay) on a
  **copy** of the user's ``config.json``, pipeline stopped: what the windows cost together.
* ``pipeline`` - as ``session``, plus two still pages fed to a running pipeline in turn, so
  the GUI is measured while OCR, translation and typesetting really run.

The session modes translate for real: when the effective ``translation_backend`` is not one of
``argos`` / ``identity`` the run calls that online service with the key stored on this machine
and spends quota, which the driver warns about - pass ``--translator argos`` to stay offline.

``--with-pipeline`` opens no window itself: it runs those three modes as child processes, one
after another, then judges them with ``tools/glass_report.py`` and prints the comparison.
``--analyse RUN.jsonl [--against BASELINE.jsonl]`` re-judges recorded events with no Qt at
all, and ``--assert-gate`` turns the verdict into the exit code.

Every run dumps its events to ``--jsonl`` (default
``demo/output/perf/glass-profile-<timestamp>.jsonl``) next to a ``.summary.json`` and a
``.qt-stderr.log``, and prints the summary that decides hypotheses (a)..(f) of
``docs/perf/2026-09-09-glass-baseline.md`` for repetition 1.  ``QSG_RENDER_TIMING=1`` is set
before the QApplication exists and the ``qt.scenegraph.time.*`` lines Qt prints to stderr are
captured for hypothesis (d) by redirecting fd 2 into a file (not ``qInstallMessageHandler``:
a Python handler is called on the render thread and needs the GIL while the GUI thread holds
it inside C++ ``exposeEvent`` -> deadlock on frame one).  Those lines carry no timestamps, so
(d) is reported over the whole run, CPU per phase.

    PYTHONUTF8=1 .venv/Scripts/python.exe tools/profile_glass.py [--steps 60] [--step-px 8]
    PYTHONUTF8=1 .venv/Scripts/python.exe tools/profile_glass.py --mode pipeline --repeat 3
    PYTHONUTF8=1 .venv/Scripts/python.exe tools/profile_glass.py --with-pipeline --assert-gate
    PYTHONUTF8=1 .venv/Scripts/python.exe tools/profile_glass.py --analyse run.jsonl --against base.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, List, Mapping, MutableMapping,
                    Optional, Sequence, Tuple)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glasstranslate.ui.glass.profile import pair_latencies_ms, percentiles  # noqa: E402
from tools import glass_report  # noqa: E402
from tools.glass_report import meta_payload  # noqa: E402,F401 - re-exported for the driver's tests
from tools.glass_session import (  # noqa: E402
    DEFAULT_PAGE_PERIOD_S,
    PAGES,
    AlternatingPageCapture,
    effective_backend,
    install_gc_marks,
    name_gc_threads,
    prepare_config,
    warn_if_online,
)

if TYPE_CHECKING:  # pragma: no cover - imported for the annotations only
    from glasstranslate.config.settings import AppConfig
    from glasstranslate.ui.app import GlassTranslateApp
    from glasstranslate.ui.control import ControlWindow

log = logging.getLogger("profile_glass")

OUT_DIR = ROOT / "demo" / "output" / "perf"
MODES: Tuple[str, ...] = ("control", "session", "pipeline")
RUN_NAMES: Dict[str, str] = {"control": "baseline", "session": "session_idle", "pipeline": "pipeline"}
WIN_POS = (600, 300)
TAB_GAP_MS = 400
TAB_WINDOW_MS = 200  # swaps counted this long after each tab change
IDLE_MS = 2000
STALE_THRESHOLD_PX = 2
WARMUP_TIMEOUT_S = 180.0  # the first pipeline pass loads the OCR and translation engines
HARD_STOP_BASE_S = 60
HARD_STOP_PER_REPEAT_S = 15
DEFAULT_MAX_EVENTS = 200_000
MAX_EVENTS_PER_REPEAT = 100_000
EVENTS_DROPPED_EXIT = 2
_RENDER_RE = re.compile(r"(\w+)=(\d+)")
_QT_ENV = {"GLASSTRANSLATE_PROFILE": "1", "GLASSTRANSLATE_APPEARANCE": "glass", "QSG_RENDER_TIMING": "1"}


def _prepare_qt_env(env: MutableMapping[str, str] = os.environ) -> None:
    """Set what Qt and the profiler read **before** the QApplication exists.

    Import time is deliberately free of this: ``--analyse`` and ``--with-pipeline`` open no
    window, and importing the module in a test must not turn profiling on for that process.
    """
    env["GLASSTRANSLATE_PROFILE"] = _QT_ENV["GLASSTRANSLATE_PROFILE"]
    env.setdefault("GLASSTRANSLATE_APPEARANCE", _QT_ENV["GLASSTRANSLATE_APPEARANCE"])
    env.setdefault("QSG_RENDER_TIMING", _QT_ENV["QSG_RENDER_TIMING"])


# ------------------------------------------------------------------------------ helpers
class Phase:
    """One measured phase: wall / CPU time plus the event index range it covers."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.t0 = self.t1 = 0.0
        self.cpu0 = self.cpu1 = 0.0

    def start(self) -> None:
        self.t0, self.cpu0 = time.perf_counter(), time.process_time()

    def stop(self) -> None:
        self.t1, self.cpu1 = time.perf_counter(), time.process_time()

    @property
    def seconds(self) -> float:
        return max(1e-9, self.t1 - self.t0)

    @property
    def cpu_pct(self) -> float:
        return 100.0 * (self.cpu1 - self.cpu0) / self.seconds

    def select(self, events: Sequence[Tuple[float, str, Dict[str, Any]]], kind: str) -> List[Dict[str, Any]]:
        return [p for t, k, p in events if k == kind and self.t0 <= t <= self.t1]

    def count(self, events: Sequence[Tuple[float, str, Dict[str, Any]]], kind: str) -> int:
        return sum(1 for t, k, _p in events if k == kind and self.t0 <= t <= self.t1)

    def times(self, events: Sequence[Tuple[float, str, Dict[str, Any]]], kind: str) -> List[float]:
        return [t for t, k, _p in events if k == kind and self.t0 <= t <= self.t1]


def pct_line(label: str, values: Sequence[float], unit: str = "ms") -> str:
    p = percentiles(values)
    return f"  {label:<44} n={p['n']:<5} p50={p['p50']:.2f} p90={p['p90']:.2f} max={p['max']:.2f} {unit}"


def deltas_ms(times: Sequence[float]) -> List[float]:
    return [(b - a) * 1000.0 for a, b in zip(times, times[1:])]


class RenderTiming:
    """Captures the ``qt.scenegraph.time.*`` lines Qt writes to stderr under ``QSG_RENDER_TIMING=1``.

    fd 2 is redirected into a file before the QApplication exists; :meth:`collect` parses it after
    the event loop ends.  Python logging keeps the original console through a dup of fd 2.
    """

    _LINE_RE = re.compile(r"qt\.scenegraph\.time\.(\w+):(.*)")

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: List[Tuple[str, Dict[str, int]]] = []
        self.other: List[str] = []
        self._console: Optional[int] = None

    def install(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr.flush()
        self._console = os.dup(2)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.dup2(fd, 2)
        os.close(fd)

    def console_stream(self) -> Any:
        """A text stream on the original stderr for the driver's own logging."""
        if self._console is None:
            return sys.stderr
        return os.fdopen(self._console, "w", encoding="utf-8", errors="replace", buffering=1)

    def collect(self) -> None:
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        for line in text.splitlines():
            m = self._LINE_RE.search(line)
            if m:
                self.records.append((m.group(1), {k: int(v) for k, v in _RENDER_RE.findall(m.group(2))}))
            elif line.strip():
                self.other.append(line.strip())

    def field(self, cat: str, key: str) -> List[float]:
        return [float(f[key]) for c, f in self.records if c == cat and key in f]


# ------------------------------------------------------------------------------ scenario
class Driver:
    """Builds what ``--mode`` asks for and plays the drag -> tabs -> idle script on it."""

    def __init__(self, args: argparse.Namespace, tmp: Path, cfg: Optional["AppConfig"] = None,
                 config_path: Optional[Path] = None) -> None:
        from PySide6.QtCore import QTimer
        from PySide6.QtQuick import QQuickItem
        from glasstranslate.ui.glass import profile

        self.args = args
        self.mode = str(args.mode)
        self.profile = profile
        self.tmp = tmp
        self.cfg: Optional["AppConfig"] = cfg
        self.session: Optional["GlassTranslateApp"] = None
        self.capture: Optional[AlternatingPageCapture] = None
        self.heartbeat: Any = None
        self.gc_events: List[Tuple[float, str, Dict[str, Any]]] = []
        self.stop_gc_marks: Callable[[], None] = lambda: None
        self.exit_code = 0
        self._warm_seen = False
        # The control window caches the recorder object inside its frameSwapped lambda, so the
        # bigger deque has to exist BEFORE any window is built - not after, as the hooks would
        # then keep marking into the 20k default.
        profile.install(force=True, maxlen=int(args.max_events))
        self.window = self._build(config_path)
        self.tab: Optional[QQuickItem] = self.window.rootObject().findChild(QQuickItem, "tabBar")
        self.phases: Dict[str, Phase] = {n: Phase(n) for n in ("drag", "tabs", "idle")}
        self.tab_marks: List[float] = []
        self.steps: Generator[int, None, None] = self._script()
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._advance)

    # -- construction
    def _build(self, config_path: Optional[Path]) -> "ControlWindow":
        from glasstranslate.config.settings import AppConfig
        from glasstranslate.ui.control import ControlWindow

        if self.mode == "control":
            self.cfg = AppConfig()
            return ControlWindow(self.cfg, self.tmp / "config.json")
        from glasstranslate.ui.app import GlassTranslateApp

        session = GlassTranslateApp(self.cfg, config_path)
        self.session = session
        if self.mode == "pipeline":
            self._build_page_source(session)
        return session.control

    def _build_page_source(self, session: "GlassTranslateApp") -> None:
        """Feed the pipeline two still pages in turn instead of the screen (the feedPage wiring)."""
        capture = AlternatingPageCapture([ROOT / name for name in PAGES],
                                         period_s=float(self.args.page_period), on_page=self._on_page)
        self.capture = capture
        session.capture_factory = lambda _cfg: capture
        # One extra listener; the session's own slots stay connected.
        session._result_ready.connect(self._on_result)

    def _on_page(self, index: int, name: str) -> None:
        """Called from the pipeline's worker thread; ``mark`` is the thread-safe part."""
        self.profile.profiler.mark("page", index=index, name=name)

    def _on_result(self, segments: object, stats: object) -> None:
        """GUI thread, through the queued ``_result_ready`` signal.

        Only a result for the page currently held counts: the first result after ``hold`` can
        still be the previous page's, which would end the warm-up one page too early.
        """
        capture = self.capture
        if capture is None or capture.serves_since_hold <= 0:
            return
        if isinstance(segments, list) and segments and not getattr(stats, "skipped_unchanged", False):
            self._warm_seen = True

    # -- scheduling: the generator yields the delay (ms) before its next step
    def start(self) -> None:
        if self.session is not None:
            self.session.show()
        else:
            self.window.show()
        self.window.setPosition(*WIN_POS)
        if self.mode == "pipeline":
            self.session.start_pipeline()
        self._advance()

    def _advance(self) -> None:
        from PySide6.QtWidgets import QApplication

        try:
            delay = next(self.steps)
        except StopIteration:
            QApplication.instance().quit()
            return
        except Exception:  # pragma: no cover - dev tool
            log.exception("scenario failed")
            self.exit_code = 1
            QApplication.instance().quit()
            return
        self.timer.start(max(0, int(delay)))

    def _script(self) -> Generator[int, None, None]:
        deadline = time.perf_counter() + 10.0
        while (not self.window.isExposed() or self.window.bridge.backdropSerial == 0) and time.perf_counter() < deadline:
            yield 50
        if self.window.bridge.backdropSerial == 0:
            raise RuntimeError("no backdrop frame arrived within 10 s (is glass allowed on this desktop?)")
        yield 1500  # let the first frames settle
        if self.mode == "pipeline":
            yield from self._warm_up()
        self.profile.profiler.clear()
        self._start_heartbeat()
        # Kept out of the profiler on purpose: see install_gc_marks (a collection can start inside
        # Profiler.mark, under its lock).  _drive appends these to the dump.
        self.gc_events, self.stop_gc_marks = install_gc_marks()
        for rep in range(1, int(self.args.repeat) + 1):
            yield from self._drag(rep)
            yield 800
            yield from self._tabs(rep)
            yield 800
            yield from self._idle(rep)

    def _warm_up(self) -> Generator[int, None, None]:
        """Hold each page until it really produced a result, so nothing is measured cold."""
        if self.capture is None:
            raise RuntimeError("pipeline mode without a page source")
        deadline = time.perf_counter() + WARMUP_TIMEOUT_S
        for index in (0, 1):
            self._warm_seen = False
            self.capture.hold(index)  # also resets the served counter _on_result waits for
            while not self._warm_seen:
                if time.perf_counter() > deadline:
                    raise RuntimeError(f"pipeline produced no result for page {index}: "
                                       "are the models installed?")
                yield 100
        self.capture.release()
        log.info("pipeline warm-up done; the two pages now alternate every %.1f s", self.args.page_period)

    def _start_heartbeat(self) -> None:
        interval = int(self.args.heartbeat_ms)
        if interval <= 0:
            return
        from PySide6.QtCore import Qt, QTimer

        module = self.profile  # looked up at call time: install() rebinds the singleton
        self.heartbeat = QTimer()
        self.heartbeat.setTimerType(Qt.TimerType.PreciseTimer)
        self.heartbeat.timeout.connect(lambda: module.profiler.mark("tick"))
        self.heartbeat.start(interval)

    # -- the script itself; only repetition 1 feeds the legacy hypothesis printout
    def _mark_phase(self, name: str, edge: str, rep: int) -> None:
        self.profile.profiler.mark("phase", name=name, edge=edge, rep=rep)

    def _drag(self, rep: int) -> Generator[int, None, None]:
        phase = self.phases["drag"]
        n, px = self.args.steps, self.args.step_px
        period_ms = max(1, int(round(1000.0 / n)))
        x0, y0 = WIN_POS
        if rep == 1:
            phase.start()
        self._mark_phase("drag", "start", rep)
        for i in range(1, n + 1):
            self.window.setPosition(x0 + i * px, y0)
            yield period_ms
        yield 150  # last frame in flight
        self._mark_phase("drag", "stop", rep)
        if rep == 1:
            phase.stop()
        self.window.setPosition(*WIN_POS)

    def _tabs(self, rep: int) -> Generator[int, None, None]:
        phase = self.phases["tabs"]
        if self.tab is None:
            raise RuntimeError("tabBar not found")
        if rep == 1:
            phase.start()
        self._mark_phase("tabs", "start", rep)
        for index in (1, 2, 3, 0):
            if rep == 1:
                self.tab_marks.append(time.perf_counter())
            self.tab.setProperty("currentIndex", index)
            yield TAB_GAP_MS
        self._mark_phase("tabs", "stop", rep)
        if rep == 1:
            phase.stop()

    def _idle(self, rep: int) -> Generator[int, None, None]:
        phase = self.phases["idle"]
        if rep == 1:
            phase.start()
        self._mark_phase("idle", "start", rep)
        yield IDLE_MS
        self._mark_phase("idle", "stop", rep)
        if rep == 1:
            phase.stop()

    # -- results
    def screen_info(self) -> Tuple[float, float]:
        """``(refresh_hz, frame_ms)`` of the screen the window ended up on."""
        screen = self.window.screen()
        refresh = float(screen.refreshRate()) if screen is not None else 0.0
        return refresh, (1000.0 / refresh if refresh > 0 else 0.0)

    def overlay_rect(self) -> Optional[List[int]]:
        if self.session is None:
            return None
        rect = self.session.overlay.physical_rect()
        return [rect.x, rect.y, rect.w, rect.h]

    def hard_stop(self) -> None:
        """The run overran its budget: say so loudly and leave a mark the report can read."""
        from PySide6.QtWidgets import QApplication

        self.profile.profiler.mark("hard_stop")
        log.error("hard stop after the scripted budget: the run never finished its script, so the "
                  "events are dumped but the record is incomplete and must not be judged")
        self.exit_code = 1
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def write_meta(self) -> None:
        """The run's parameters, written last so ``dropped`` is final."""
        prof = self.profile.profiler
        refresh, frame_ms = self.screen_info()
        backend = None if self.mode == "control" or self.cfg is None else str(self.cfg.translation_backend)
        prof.mark("meta", **meta_payload(
            mode=self.mode, refresh_hz=refresh, frame_ms=frame_ms, steps=self.args.steps,
            step_px=self.args.step_px, repeat=self.args.repeat, heartbeat_ms=self.args.heartbeat_ms,
            backend=backend, ocr_engine=None if self.cfg is None else str(self.cfg.ocr_engine),
            capacity=prof.capacity, dropped=prof.dropped, overlay=self.overlay_rect()))
        if prof.dropped:
            log.error("the profiler dropped %d of its events (capacity %d): this run measured less "
                      "than it recorded; re-run with a larger --max-events", prof.dropped, prof.capacity)
            self.exit_code = EVENTS_DROPPED_EXIT

    def finish(self) -> None:
        self.stop_gc_marks()
        if self.heartbeat is not None:
            self.heartbeat.stop()
        if self.session is not None:
            self.session.teardown()  # stops the pipeline and destroys both windows
        else:
            self.window.shutdown()
            self.window.close()


# ------------------------------------------------------------------------------ reporting
def _tab_transition_lines(driver: Driver, events: Sequence[Tuple[float, str, Dict[str, Any]]],
                          frame_ms: float) -> Tuple[List[str], List[int]]:
    """(e): swaps within 200 ms of each ``currentIndex`` change of repetition 1."""
    factor = float(driver.args.gap_factor)
    lines: List[str] = [f"(e) tab transitions: swaps within {TAB_WINDOW_MS} ms of each currentIndex "
                        f"change, inter-swap deltas (a gap is > {factor:.1f} frames)"]
    per_tab: List[int] = []
    for i, t_tab in enumerate(driver.tab_marks):
        window_swaps = [t for t, k, _p in events if k == "swap" and t_tab <= t <= t_tab + TAB_WINDOW_MS / 1000.0]
        per_tab.append(len(window_swaps))
        d = deltas_ms(window_swaps)
        dp = percentiles(d)
        over = {f: sum(1 for x in d if x > f * frame_ms) for f in (factor, 2.0, 2.5)} if frame_ms else {}
        lines.append(f"  switch {i + 1}: {len(window_swaps):3d} swaps / {TAB_WINDOW_MS} ms;"
                     f" delta p50 {dp['p50']:.2f} p90 {dp['p90']:.2f} max {dp['max']:.2f} ms;"
                     f" >{factor:.1f} frames: {over.get(factor, 0)}"
                     f" (>2.0: {over.get(2.0, 0)}, >2.5: {over.get(2.5, 0)})")
    return lines, per_tab


def report(driver: Driver, timing: RenderTiming, out: Path) -> Dict[str, Any]:
    events = driver.profile.profiler.events()
    screen = driver.window.screen()
    api = driver.window.rendererInterface().graphicsApi()
    refresh, frame_ms = driver.screen_info()
    drag, tabs, idle = (driver.phases[n] for n in ("drag", "tabs", "idle"))
    lines: List[str] = []
    if int(driver.args.repeat) > 1:
        lines.append(f"(a)..(f) below cover repetition 1 of {driver.args.repeat}; the whole run is "
                     f"judged by tools/glass_report.py from {out.name}")
    # The window can be off every screen (a disconnected monitor): report what is known.
    geometry = screen.geometry() if screen is not None else None
    size = f"{geometry.width()}x{geometry.height()}" if geometry is not None else "?x?"
    lines.append(f"screen {screen.name() if screen is not None else '?'} {size} @ {refresh:.2f} Hz"
                 f" ({frame_ms:.2f} ms/frame), dpr {screen.devicePixelRatio() if screen is not None else 1.0},"
                 f" RHI {api.name}")
    lines.append(f"mode {driver.mode}, events {len(events)} + {len(driver.gc_events)} gc (capacity {driver.profile.profiler.capacity},"
                 f" dropped {driver.profile.profiler.dropped}), jsonl {_relative(out)}")

    # (a) stale offset during the drag
    frames = drag.select(events, "frame_gui")
    offsets = [abs(p["dx"]) + abs(p["dy"]) for p in frames]
    over = sum(1 for o in offsets if o > STALE_THRESHOLD_PX)
    lines.append("(a) stale offset |P_now - P_frame| on GUI frames during the drag")
    lines.append(pct_line("offset px", offsets, "px"))
    lines.append(f"  frames with offset > {STALE_THRESHOLD_PX} px: {over}/{len(offsets)}"
                 f" (drag = {driver.args.steps} x {driver.args.step_px} px in {drag.seconds:.2f} s)")

    # (b) cadence
    def cadence(phase: Phase) -> Dict[str, float]:
        g, e, s = phase.count(events, "grab"), phase.count(events, "frame_gui"), phase.count(events, "swap")
        return {"grabs_s": g / phase.seconds, "frames_s": e / phase.seconds, "swaps_s": s / phase.seconds,
                "swaps_per_frame": (s / e) if e else 0.0, "grabs": g, "frames": e, "swaps": s}

    cad = {n: cadence(p) for n, p in driver.phases.items()}
    lines.append("(b) cadence: grabs/s, frames reaching GUI/s, swaps/s, swaps per backdrop frame")
    for n, c in cad.items():
        lines.append(f"  {n:<6} grabs {c['grabs_s']:6.1f}/s  frames {c['frames_s']:6.1f}/s  swaps {c['swaps_s']:7.1f}/s"
                     f"  swaps/frame {c['swaps_per_frame']:5.1f}   (n grabs {c['grabs']}, frames {c['frames']}, swaps {c['swaps']})")
    lines.append(pct_line("grab interval during drag", deltas_ms(drag.times(events, "grab"))))
    lines.append(pct_line("grab duration during drag (ms)", [p["ms"] for p in drag.select(events, "grab")]))

    # (c) latency
    drag_events = [(t, k, p) for t, k, p in events if drag.t0 <= t <= drag.t1]
    lines.append("(c) latency chain during the drag")
    for a, b in (("move", "poke"), ("poke", "grab"), ("grab", "frame_gui"), ("move", "frame_gui")):
        lines.append(pct_line(f"{a} -> {b}", pair_latencies_ms(drag_events, a, b)))
    lines.append(pct_line("push_backdrop slot (ms)", [p["ms"] for p in drag.select(events, "push_backdrop_ms")]))
    grab_age = [(t - p["ts"]) * 1000.0 for t, k, p in drag_events if k == "frame_gui"]
    lines.append(pct_line("frame age at GUI (grab ts -> slot)", grab_age))

    # (d) render timing
    lines.append("(d) render thread (QSG_RENDER_TIMING) and CPU")
    for n, p in driver.phases.items():
        lines.append(f"  {n:<6} cpu {p.cpu_pct:5.1f} %")
    if timing.records:
        for key in ("sync", "render", "swap"):
            lines.append(pct_line(f"renderloop {key} (whole run, {len(timing.records)} frames)", timing.field("renderloop", key)))
    else:
        lines.append("  QSG_RENDER_TIMING output not captured: (d) measured indirectly via swap cadence + CPU")
    lines.append(f"  idle swaps: {cad['idle']['swaps']} in {idle.seconds:.1f} s")

    # (e)/(f) tab transitions
    tab_lines, per_tab = _tab_transition_lines(driver, events, frame_ms)
    lines.extend(tab_lines)
    fade_frames_expected = 160.0 / frame_ms if frame_ms else 0.0
    lines.append(f"  expected at {refresh:.0f} Hz: {fade_frames_expected:.0f} frames per 160 ms fade; measured mean {statistics.fmean(per_tab) if per_tab else 0:.1f}")
    lines.append(pct_line("push_backdrop slot during tabs (ms)", [p["ms"] for p in tabs.select(events, "push_backdrop_ms")]))
    lines.append(f"  tabs phase cpu {tabs.cpu_pct:.1f} %  frames_gui {cad['tabs']['frames']}")
    # render tail: how long the scene keeps swapping after the last currentIndex change (the
    # contract says a page transition is <= 180 ms and nothing animates while idle)
    if driver.tab_marks:
        last = driver.tab_marks[-1]
        tail = [t for t, k, _p in events if k == "swap" and t > last]
        tail_ms = (tail[-1] - last) * 1000.0 if tail else 0.0
        lines.append(f"  render tail after the last switch: {len(tail)} swaps, last at +{tail_ms:.0f} ms")
    lines.append("(f) 160 ms fade is time-driven (NumberAnimation duration): frame count scales with refresh; see (e) count vs expected")
    if timing.other:
        lines.append(f"other Qt messages: {len(timing.other)} (first: {timing.other[0][:120]})")
    text = "\n".join(lines)
    print(text)
    return {"lines": lines, "cadence": cad, "offsets": percentiles(offsets), "over": over, "per_tab": per_tab,
            "refresh": refresh, "api": api.name, "mode": driver.mode,
            "dropped": driver.profile.profiler.dropped}


# --------------------------------------------------------------------------------- modes
def _relative(path: Path) -> str:
    """How a path is written into a report: inside the repo with forward slashes, else the name."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def _emit_report(args: argparse.Namespace, runs: Mapping[str, Dict[str, Any]], jsonl: Mapping[str, str],
                 gate: Optional[Dict[str, Any]], extra_path: Optional[Path] = None) -> int:
    document = glass_report.build_report(runs, jsonl, gate)
    problems = glass_report.validate_report(document)
    for problem in problems:
        log.error("report schema: %s", problem)
    print(glass_report.format_comparison(document))
    # No default=str and no NaN: a stray Path or a non-finite number must be a loud failure,
    # not a string that looks like a measurement.
    text = json.dumps(document, indent=2, allow_nan=False)
    targets = [p for p in (extra_path, Path(args.report) if args.report else None) if p is not None]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        log.info("report written to %s", target)
    if not args.assert_gate:
        return 0
    if problems:
        log.error("the report does not validate, so its gate cannot be trusted")
        return 1
    return 0 if (gate is not None and gate.get("pass")) else 1


def run_analyse(args: argparse.Namespace) -> int:
    """Judge recorded events; no Qt, no window, no display."""
    run = glass_report.analyse(glass_report.load_events(args.analyse), args.gap_factor)
    if args.against is None:
        return _emit_report(args, {"baseline": run}, {"baseline": _relative(Path(args.analyse))}, None)
    baseline = glass_report.analyse(glass_report.load_events(args.against), args.gap_factor)
    runs = {"baseline": baseline, "run": run}
    jsonl = {"baseline": _relative(Path(args.against)), "run": _relative(Path(args.analyse))}
    return _emit_report(args, runs, jsonl, glass_report.judge_gate(baseline, run))


def child_command(args: argparse.Namespace, mode: str, jsonl: Path) -> List[str]:
    """The exact command line one ``--with-pipeline`` child is started with."""
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode, "--jsonl", str(jsonl),
               "--out", str(args.out), "--steps", str(args.steps), "--step-px", str(args.step_px),
               "--repeat", str(args.repeat), "--heartbeat-ms", str(args.heartbeat_ms),
               "--max-events", str(args.max_events), "--page-period", str(args.page_period),
               "--gap-factor", str(args.gap_factor)]
    if args.translator:
        command += ["--translator", str(args.translator)]
    return command


def run_parent(args: argparse.Namespace) -> int:
    """Run the three modes as children, then judge the pipeline against the control baseline."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    warn_if_online(effective_backend(args.translator))  # once, before the children start
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    paths = {mode: out / f"glass-profile-{stamp}-{mode}.jsonl" for mode in MODES}
    for mode in MODES:
        command = child_command(args, mode, paths[mode])
        log.info("%s pass: %s", mode, " ".join(command))
        completed = subprocess.run(command, env=env)
        if completed.returncode != 0:
            log.error("the %s pass exited with %d (0x%X); nothing is judged", mode,
                      completed.returncode, completed.returncode & 0xFFFFFFFF)
            # A native crash reports a 32-bit NTSTATUS, which os._exit cannot take.
            return completed.returncode if 0 < completed.returncode < 256 else 1
    runs = {RUN_NAMES[mode]: glass_report.analyse(glass_report.load_events(paths[mode]), args.gap_factor)
            for mode in MODES}
    jsonl = {RUN_NAMES[mode]: _relative(paths[mode]) for mode in MODES}
    gate = glass_report.judge_gate(runs["baseline"], runs["pipeline"])
    return _emit_report(args, runs, jsonl, gate, extra_path=out / f"glass-profile-{stamp}-report.json")


def run_single(args: argparse.Namespace) -> int:
    """One in-process run of ``--mode``: real windows on screen, the script, the dump."""
    _prepare_qt_env()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.jsonl) if args.jsonl else Path(args.out) / f"glass-profile-{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    timing = RenderTiming(out.with_name(out.stem + ".qt-stderr.log"))
    timing.install()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
                        stream=timing.console_stream())
    try:
        return _drive(args, out, timing)
    except Exception:  # fd 2 is the Qt log by now, so report through the console logger
        log.exception("the %s run failed", args.mode)
        return 1


def _prepare_application(args: argparse.Namespace, tmp: Path) -> Tuple[Any, Optional["AppConfig"], Optional[Path]]:
    """The QApplication and, for the session modes, the temp copy of the user's config."""
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    cfg: Optional["AppConfig"] = None
    config_path: Optional[Path] = None
    if args.mode != "control":
        from glasstranslate.ui.app import apply_portable_qt_environment

        config_path, cfg = prepare_config(tmp, args.translator)
        warn_if_online(cfg.translation_backend)
        # Both Qt cache variables are read when the engines are built: before the application,
        # exactly as ``glasstranslate.ui.app.main`` does it.  The Python Qt message handler is
        # deliberately NOT installed (see the module docstring: render-thread GIL deadlock).
        apply_portable_qt_environment()
    fmt = QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("GlassTranslate")
    app.setQuitOnLastWindowClosed(False)
    return app, cfg, config_path


def _teardown(driver: Optional[Driver], tmp: Path) -> None:
    """Always: destroy the windows, then remove the temp directory.

    That directory holds a copy of the user's ``config.json``, which can carry
    ``translation_api_key``; leaving it in %TEMP% after every run is not acceptable.
    """
    if driver is not None:
        try:
            driver.finish()
        except Exception:  # noqa: BLE001 - a failing teardown must not hide the run's own result
            log.exception("tearing the windows down failed")
    shutil.rmtree(tmp, ignore_errors=True)


def _drive(args: argparse.Namespace, out: Path, timing: RenderTiming) -> int:
    """Build the windows, play the script, dump the events and print the legacy summary."""
    from PySide6.QtCore import QTimer

    tmp = Path(tempfile.mkdtemp(prefix="glass-profile-"))
    driver: Optional[Driver] = None
    try:
        app, cfg, config_path = _prepare_application(args, tmp)
        driver = Driver(args, tmp, cfg=cfg, config_path=config_path)
        hard_stop_s = HARD_STOP_BASE_S + HARD_STOP_PER_REPEAT_S * int(args.repeat)
        if args.mode == "pipeline":
            hard_stop_s += WARMUP_TIMEOUT_S
        QTimer.singleShot(int(hard_stop_s * 1000), driver.hard_stop)
        driver.start()
        app.exec()
        driver.write_meta()
        driver.stop_gc_marks()  # before its list is read: a collection could append mid-write
        driver.profile.profiler.dump(out)
        glass_report.append_events(out, name_gc_threads(driver.gc_events))
        timing.collect()
        summary = report(driver, timing, out)
        out.with_name(out.stem + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return driver.exit_code
    finally:
        _teardown(driver, tmp)


# ------------------------------------------------------------------------------------ main
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, default="control",
                    help="what a single run builds (default control)")
    ap.add_argument("--steps", type=int, default=60, help="drag steps (default 60)")
    ap.add_argument("--step-px", type=int, default=8, help="logical px per step (default 8)")
    ap.add_argument("--repeat", type=int, default=1, help="how often the script plays (default 1)")
    ap.add_argument("--heartbeat-ms", type=int, default=4,
                    help="GUI-thread heartbeat interval in ms, 0 = off (default 4)")
    ap.add_argument("--gap-factor", type=float, default=2.5,
                    help="a gap of more than this many frame intervals is a stall (default 2.5)")
    ap.add_argument("--max-events", type=int, default=None,
                    help="profiler capacity (default max(200000, 100000 x repeat))")
    ap.add_argument("--page-period", type=float, default=DEFAULT_PAGE_PERIOD_S,
                    help="seconds each page is served in pipeline mode (default 2.0)")
    # Imported here, not at module scope: it pulls the translation engines in and nothing else
    # in this module needs them (it is Qt-free, so --analyse stays display-free either way).
    from glasstranslate.translate.factory import available_backends

    ap.add_argument("--translator", default=None, choices=available_backends(),
                    help="translation_backend for the session modes (default: the config's)")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output dir (default demo/output/perf)")
    ap.add_argument("--jsonl", type=Path, default=None, help="event file of a single-mode run")
    ap.add_argument("--report", type=Path, default=None, help="write the JSON comparison here too")
    ap.add_argument("--with-pipeline", action="store_true",
                    help="parent mode: run control, session and pipeline as children and compare them")
    ap.add_argument("--analyse", type=Path, default=None, help="judge this JSONL instead of running")
    ap.add_argument("--against", type=Path, default=None, help="baseline JSONL for --analyse")
    ap.add_argument("--assert-gate", action="store_true", help="exit 1 when the gate did not pass")
    args = ap.parse_args(argv)
    if args.max_events is None:
        args.max_events = max(DEFAULT_MAX_EVENTS, MAX_EVENTS_PER_REPEAT * max(1, int(args.repeat)))
    _validate_args(ap, args)
    return args


def _validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject values the script cannot play and flags that would silently do nothing."""
    for name, minimum in (("steps", 1), ("step_px", 1), ("repeat", 1), ("heartbeat_ms", 0),
                          ("max_events", 1)):
        if int(getattr(args, name)) < minimum:
            ap.error(f"--{name.replace('_', '-')} must be >= {minimum}")
    for name in ("page_period", "gap_factor"):
        if not float(getattr(args, name)) > 0:
            ap.error(f"--{name.replace('_', '-')} must be > 0")
    judging = args.analyse is not None or args.with_pipeline
    if args.assert_gate and not judging:
        ap.error("--assert-gate needs --analyse or --with-pipeline: a single run judges nothing")
    if args.against is not None and args.analyse is None:
        ap.error("--against is the baseline for --analyse")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.analyse is not None or args.with_pipeline:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        return run_analyse(args) if args.analyse is not None else run_parent(args)
    return run_single(args)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # the Qt teardown with a translucent QQuickView occasionally hangs at interpreter exit
