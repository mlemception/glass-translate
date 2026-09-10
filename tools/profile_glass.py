"""Dev-only glass profiling driver (feature batch F2-0): measure, do not fix.

Runs the real control window ON SCREEN with ``GLASSTRANSLATE_PROFILE=1`` and
``GLASSTRANSLATE_APPEARANCE=glass`` (no pipeline, temp config), waits for exposure and the
first backdrop frame, then scripts

1. a drag: ``window.setPosition`` in N steps (default 60 x 8 px over ~1 s via QTimer).  A
   real title-bar drag goes through ``startSystemMove`` (a modal Win32 move loop); the
   programmatic move is the closest scriptable equivalent and exercises the same
   ``moveEvent -> _refresh_geometry -> _poke`` path, just without the WM_MOVING cadence;
2. a tab-switch burst: ``tabBar.currentIndex`` 0 -> 1 -> 2 -> 3 -> 0, 400 ms apart;
3. an idle hold of 2 s;

then dumps every event to ``demo/output/perf/glass-profile-<timestamp>.jsonl`` and prints
the summary that decides hypotheses (a)..(f) of ``docs/perf/2026-09-09-glass-baseline.md``.
``QSG_RENDER_TIMING=1`` is set before the QApplication exists and the ``qt.scenegraph.time.*``
lines Qt prints to stderr are captured for hypothesis (d) by redirecting fd 2 into a file
(not ``qInstallMessageHandler``: a Python handler is called on the render thread and needs
the GIL while the GUI thread holds it inside C++ ``exposeEvent`` -> deadlock on frame one).
Those lines carry no timestamps, so (d) is reported over the whole run, CPU per phase.

    PYTHONUTF8=1 .venv/Scripts/python.exe tools/profile_glass.py [--steps 60] [--step-px 8] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["GLASSTRANSLATE_PROFILE"] = "1"
os.environ.setdefault("GLASSTRANSLATE_APPEARANCE", "glass")
os.environ.setdefault("QSG_RENDER_TIMING", "1")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QSurfaceFormat  # noqa: E402
from PySide6.QtQuick import QQuickItem  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

log = logging.getLogger("profile_glass")

OUT_DIR = ROOT / "demo" / "output" / "perf"
WIN_POS = (600, 300)
TAB_GAP_MS = 400
TAB_WINDOW_MS = 200  # swaps counted this long after each tab change
IDLE_MS = 2000
STALE_THRESHOLD_PX = 2
_RENDER_RE = re.compile(r"(\w+)=(\d+)")


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
    from glasstranslate.ui.glass.profile import percentiles

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
    def __init__(self, args: argparse.Namespace) -> None:
        from glasstranslate.config.settings import AppConfig
        from glasstranslate.ui.control import ControlWindow
        from glasstranslate.ui.glass import profile

        self.args = args
        self.profile = profile
        self.tmp = Path(tempfile.mkdtemp(prefix="glass-profile-"))
        self.window = ControlWindow(AppConfig(), self.tmp / "config.json")
        self.tab: Optional[QQuickItem] = self.window.rootObject().findChild(QQuickItem, "tabBar")
        self.phases: Dict[str, Phase] = {n: Phase(n) for n in ("drag", "tabs", "idle")}
        self.tab_marks: List[float] = []
        self.steps: Generator[int, None, None] = self._script()
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._advance)
        self.exit_code = 0

    # -- scheduling: the generator yields the delay (ms) before its next step
    def start(self) -> None:
        self.window.show()
        self.window.setPosition(*WIN_POS)
        self._advance()

    def _advance(self) -> None:
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
        self.profile.profiler.clear()
        yield from self._drag()
        yield 800
        yield from self._tabs()
        yield 800
        yield from self._idle()

    def _drag(self) -> Generator[int, None, None]:
        phase = self.phases["drag"]
        n, px = self.args.steps, self.args.step_px
        period_ms = max(1, int(round(1000.0 / n)))
        x0, y0 = WIN_POS
        phase.start()
        for i in range(1, n + 1):
            self.window.setPosition(x0 + i * px, y0)
            yield period_ms
        yield 150  # last frame in flight
        phase.stop()
        self.window.setPosition(*WIN_POS)

    def _tabs(self) -> Generator[int, None, None]:
        phase = self.phases["tabs"]
        if self.tab is None:
            raise RuntimeError("tabBar not found")
        phase.start()
        for index in (1, 2, 3, 0):
            self.tab_marks.append(time.perf_counter())
            self.tab.setProperty("currentIndex", index)
            yield TAB_GAP_MS
        phase.stop()

    def _idle(self) -> Generator[int, None, None]:
        phase = self.phases["idle"]
        phase.start()
        yield IDLE_MS
        phase.stop()


# ------------------------------------------------------------------------------ reporting
def report(driver: Driver, timing: RenderTiming, out: Path) -> Dict[str, Any]:
    from glasstranslate.ui.glass.profile import pair_latencies_ms, percentiles

    events = driver.profile.profiler.events()
    screen = driver.window.screen()
    api = driver.window.rendererInterface().graphicsApi()
    refresh = float(screen.refreshRate()) if screen is not None else 0.0
    frame_ms = 1000.0 / refresh if refresh > 0 else 0.0
    drag, tabs, idle = (driver.phases[n] for n in ("drag", "tabs", "idle"))
    lines: List[str] = []
    lines.append(f"screen {screen.name()} {screen.geometry().width()}x{screen.geometry().height()} @ {refresh:.2f} Hz"
                 f" ({frame_ms:.2f} ms/frame), dpr {screen.devicePixelRatio()}, RHI {api.name}")
    lines.append(f"events {len(events)}, jsonl {out}")

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
    have_timing = bool(timing.records)
    for n, p in driver.phases.items():
        lines.append(f"  {n:<6} cpu {p.cpu_pct:5.1f} %")
    if have_timing:
        for key in ("sync", "render", "swap"):
            lines.append(pct_line(f"renderloop {key} (whole run, {len(timing.records)} frames)", timing.field("renderloop", key)))
    else:
        lines.append("  QSG_RENDER_TIMING output not captured: (d) measured indirectly via swap cadence + CPU")
    lines.append(f"  idle swaps: {cad['idle']['swaps']} in {idle.seconds:.1f} s")

    # (e)/(f) tab transitions
    lines.append("(e) tab transitions: swaps within 200 ms of each currentIndex change, inter-swap deltas")
    per_tab: List[int] = []
    all_deltas: List[float] = []
    fade_frames_expected = 160.0 / frame_ms if frame_ms else 0.0
    for i, t_tab in enumerate(driver.tab_marks):
        window_swaps = [t for t, k, _p in events if k == "swap" and t_tab <= t <= t_tab + TAB_WINDOW_MS / 1000.0]
        per_tab.append(len(window_swaps))
        d = deltas_ms(window_swaps)
        all_deltas.extend(d)
        dp = percentiles(d)
        lines.append(f"  switch {i + 1}: {len(window_swaps):3d} swaps / {TAB_WINDOW_MS} ms; delta p50 {dp['p50']:.2f} p90 {dp['p90']:.2f} max {dp['max']:.2f} ms;"
                     f" >2 frames gap: {sum(1 for x in d if x > 2.5 * frame_ms)}")
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
            "refresh": refresh, "api": api.name}


# ------------------------------------------------------------------------------ main
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=60, help="drag steps (default 60)")
    ap.add_argument("--step-px", type=int, default=8, help="logical px per step (default 8)")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="JSONL output dir (default demo/output/perf)")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    timing = RenderTiming(args.out / f"glass-profile-{stamp}.qt-stderr.log")
    timing.install()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=timing.console_stream())
    fmt = QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    driver = Driver(args)
    QTimer.singleShot(60000, app.quit)  # hard stop
    driver.start()
    app.exec()
    out = args.out / f"glass-profile-{stamp}.jsonl"
    driver.profile.profiler.dump(out)
    timing.collect()
    summary = report(driver, timing, out)
    (args.out / f"glass-profile-{stamp}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    driver.window.shutdown()
    driver.window.close()
    return driver.exit_code


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # the Qt teardown with a translucent QQuickView occasionally hangs at interpreter exit
