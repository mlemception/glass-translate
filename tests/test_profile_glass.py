"""Headless checks for the glass profiling driver (``tools/profile_glass.py``).

Nothing here opens a window: the argument surface, the alternating page source, the temp
config copy, the ``--analyse`` round trip and the parent mode with the child processes
replaced.  The on-screen part of the driver is exercised by running it, not by pytest.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from glasstranslate.config.settings import AppConfig
from glasstranslate.core.types import Rect
from tools import glass_report as G
from tools import glass_session as GS
from tools import profile_glass as PG

Event = Tuple[float, str, Dict[str, Any]]
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------- import rules
def test_importing_the_driver_leaves_the_environment_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--analyse`` and the parent mode must not need a display, so the module sets nothing."""
    for name in ("GLASSTRANSLATE_PROFILE", "GLASSTRANSLATE_APPEARANCE", "QSG_RENDER_TIMING"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "tools.profile_glass", raising=False)
    module = importlib.import_module("tools.profile_glass")
    assert "GLASSTRANSLATE_PROFILE" not in os.environ
    assert "GLASSTRANSLATE_APPEARANCE" not in os.environ and "QSG_RENDER_TIMING" not in os.environ
    env: Dict[str, str] = {}
    module._prepare_qt_env(env)
    assert env["GLASSTRANSLATE_PROFILE"] == "1" and env["GLASSTRANSLATE_APPEARANCE"] == "glass"
    assert env["QSG_RENDER_TIMING"] == "1"


# ------------------------------------------------------------------------------ arguments
def test_parse_args_defaults() -> None:
    args = PG.parse_args([])
    assert args.mode == "control" and args.steps == 60 and args.step_px == 8
    assert args.repeat == 1 and args.heartbeat_ms == 4 and args.gap_factor == 2.5
    assert args.page_period == PG.DEFAULT_PAGE_PERIOD_S
    assert args.jsonl is None and args.report is None and args.translator is None
    assert args.with_pipeline is False and args.assert_gate is False and args.analyse is None
    assert args.out == PG.OUT_DIR


def test_max_events_scales_with_repeat() -> None:
    assert PG.parse_args([]).max_events == 200_000
    assert PG.parse_args(["--repeat", "2"]).max_events == 200_000
    assert PG.parse_args(["--repeat", "5"]).max_events == 500_000
    assert PG.parse_args(["--repeat", "5", "--max-events", "1234"]).max_events == 1234


@pytest.mark.parametrize("argv", [
    ["--steps", "0"], ["--step-px", "0"], ["--repeat", "0"], ["--heartbeat-ms", "-1"],
    ["--max-events", "0"], ["--page-period", "0"], ["--gap-factor", "0"],
    ["--assert-gate"], ["--against", "base.jsonl"], ["--translator", "sugoi"],
])
def test_parse_args_rejects_nonsense(argv: List[str]) -> None:
    with pytest.raises(SystemExit):
        PG.parse_args(argv)


def test_parse_args_accepts_the_gate_flags_where_they_mean_something() -> None:
    assert PG.parse_args(["--with-pipeline", "--assert-gate"]).assert_gate is True
    args = PG.parse_args(["--analyse", "run.jsonl", "--against", "base.jsonl", "--assert-gate"])
    assert args.against == Path("base.jsonl")
    assert PG.parse_args(["--translator", "argos"]).translator == "argos"


def test_meta_payload_has_exactly_the_documented_keys() -> None:
    """The ``meta`` mark is the report's whole input: its key set is a contract."""
    args = PG.parse_args(["--repeat", "2", "--heartbeat-ms", "0"])
    payload = PG.meta_payload(mode="pipeline", refresh_hz=165.0, frame_ms=6.06, steps=args.steps,
                              step_px=args.step_px, repeat=args.repeat,
                              heartbeat_ms=args.heartbeat_ms, backend="argos",
                              ocr_engine="mangaocr", capacity=10, dropped=0, overlay=[0, 0, 8, 6])
    assert set(payload) == {"mode", "refresh_hz", "frame_ms", "steps", "step_px", "repeat",
                            "heartbeat_ms", "backend", "ocr_engine", "capacity", "dropped", "overlay"}
    assert set(payload) == set(G.META_KEYS)
    assert (payload["mode"], payload["repeat"], payload["heartbeat_ms"]) == ("pipeline", 2, 0)
    assert payload["backend"] == "argos" and payload["overlay"] == [0, 0, 8, 6]


# ------------------------------------------------------------------- online-translator warning
def test_only_an_online_backend_warns(caplog: pytest.LogCaptureFixture) -> None:
    from glasstranslate.translate.factory import available_backends

    assert set(GS.OFFLINE_BACKENDS) <= set(available_backends())
    for offline in GS.OFFLINE_BACKENDS:
        with caplog.at_level("WARNING", logger="profile_glass"):
            caplog.clear()
            assert PG.warn_if_online(offline) is False
        assert caplog.records == []
    for online in ("gemini", "libretranslate"):
        with caplog.at_level("WARNING", logger="profile_glass"):
            caplog.clear()
            assert PG.warn_if_online(online) is True
        text = caplog.text
        assert online in text and "--translator" in text
        assert "http" not in text and "key" in text  # names the risk, never a URL
    assert PG.warn_if_online(None) is False


# ------------------------------------------------------------------- alternating page source
def _png(path: Path, value: int) -> Path:
    import cv2

    assert cv2.imwrite(str(path), np.full((8, 12, 3), value, np.uint8))
    return path


def _blue(capture: GS.AlternatingPageCapture, region: Rect) -> int:
    frame = capture.grab(region)
    assert frame is not None and (frame.origin_x, frame.origin_y) == (region.x, region.y)
    return int(frame.image[0, 0, 0])


def test_alternating_capture_holds_then_alternates(tmp_path: Path) -> None:
    a, b = _png(tmp_path / "a.png", 10), _png(tmp_path / "b.png", 200)
    now = [0.0]
    marks: List[Tuple[int, str]] = []
    capture = GS.AlternatingPageCapture([a, b], period_s=2.0, clock=lambda: now[0],
                                        on_page=lambda i, n: marks.append((i, n)))
    region = Rect(5, 7, 100, 50)
    assert _blue(capture, region) == 10 and marks == [(0, "a.png")]
    now[0] = 100.0
    assert _blue(capture, region) == 10 and marks == [(0, "a.png")]  # held: no second mark
    capture.hold(1)
    assert _blue(capture, region) == 200 and marks[-1] == (1, "b.png")
    capture.release()  # released at t = 100.0
    assert _blue(capture, region) == 10 and marks[-1] == (0, "a.png")
    now[0] = 101.9
    assert _blue(capture, region) == 10 and len(marks) == 3
    now[0] = 102.1
    assert _blue(capture, region) == 200 and marks[-1] == (1, "b.png")
    now[0] = 104.1
    assert _blue(capture, region) == 10 and marks[-1] == (0, "a.png")
    assert [name for _i, name in marks] == ["a.png", "b.png", "a.png", "b.png", "a.png"]


def test_alternating_capture_publishes_its_state_as_one_tuple(tmp_path: Path) -> None:
    """The worker thread reads ``_state`` once, so it can never see a half-updated pair."""
    a, b = _png(tmp_path / "a.png", 10), _png(tmp_path / "b.png", 200)
    now = [0.0]
    capture = GS.AlternatingPageCapture([a, b], period_s=2.0, clock=lambda: now[0])
    assert capture._state == (0, 0.0)
    capture.hold(1)
    assert capture._state == (1, 0.0)
    now[0] = 5.0
    capture.release()
    assert capture._state == (None, 5.0)


def test_alternating_capture_counts_serves_of_the_held_page(tmp_path: Path) -> None:
    """The warm-up waits for a result *of the page it holds*, so it needs that count."""
    a, b = _png(tmp_path / "a.png", 10), _png(tmp_path / "b.png", 200)
    now = [0.0]
    capture = GS.AlternatingPageCapture([a, b], period_s=2.0, clock=lambda: now[0])
    region = Rect(0, 0, 10, 10)
    assert capture.serves_since_hold == 0
    capture.grab(region)
    capture.grab(region)
    assert capture.serves_since_hold == 2
    capture.hold(1)
    assert capture.serves_since_hold == 0  # a new page: nothing served for it yet
    capture.grab(region)
    assert capture.serves_since_hold == 1
    capture.release()
    capture.grab(region)
    assert capture.serves_since_hold == 1  # nothing is held any more


def test_alternating_capture_needs_two_pages(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="two pages"):
        GS.AlternatingPageCapture([_png(tmp_path / "a.png", 1)])


def test_the_two_shipped_pages_are_in_the_checkout() -> None:
    assert len(GS.PAGES) == 2
    for name in GS.PAGES:
        assert (ROOT / name).is_file()


# ---------------------------------------------------------------------- garbage-collection marks
def test_gc_marks_record_each_collection_and_uninstall_cleanly() -> None:
    import gc
    import threading

    before = list(gc.callbacks)
    events, uninstall = GS.install_gc_marks()
    try:
        gc.collect()  # a full (generation 2) collection
    finally:
        uninstall()
    assert gc.callbacks == before  # nothing of ours is left behind
    full = [p for _t, kind, p in events if kind == "gc" and p["gen"] == 2]
    assert full and full[-1]["ms"] >= 0.0 and full[-1]["thread"] == threading.get_ident()
    assert GS.name_gc_threads(events)[-1][2]["thread"] == threading.current_thread().name
    assert GS.name_gc_threads([(0.0, "gc", {"thread": -1})])[0][2]["thread"] == "-1"  # a thread that is gone
    assert all(isinstance(t, float) for t, _kind, _p in events)
    count = len(events)
    gc.collect()
    assert len(events) == count  # uninstalled: no further events
    uninstall()  # a second call is harmless


def test_gc_marks_survive_a_collection_while_the_profiler_lock_is_held() -> None:
    """The collector can run inside ``Profiler.mark``, i.e. under its non-reentrant lock: a
    callback that called the profiler there deadlocked the GUI thread on its first collection."""
    import gc

    from glasstranslate.ui.glass import profile as P

    prof = P.Profiler()
    events, uninstall = GS.install_gc_marks()
    try:
        with prof._lock:  # what mark() holds while it allocates its event tuple
            gc.collect()
    finally:
        uninstall()
    assert [kind for _t, kind, _p in events].count("gc") >= 1
    assert prof.events() == []  # nothing went through the profiler, so nothing could block on it


# ------------------------------------------------------------------------------- config copy
def test_prepare_config_copies_only_the_config_and_forces_the_run_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    source = home / "config.json"
    AppConfig(translation_backend="libretranslate", quality_renderer="auto", manga_mode=False,
              running_on_start=True).save(source)
    (home / "secrets.json").write_text('{"gemini_api_key": "NEVER-COPIED"}', encoding="utf-8")
    monkeypatch.setattr(GS.settings, "default_config_path", lambda: source)
    work = tmp_path / "work"
    work.mkdir()

    path, cfg = GS.prepare_config(work)
    assert path == work / "config.json" and path.is_file()
    assert [p.name for p in sorted(work.iterdir())] == ["config.json"]  # secrets.json stayed behind
    assert cfg.translation_backend == "libretranslate"  # the user's backend is kept by default
    assert cfg.quality_renderer == "off" and cfg.manga_mode is True and cfg.running_on_start is False
    _path, overridden = GS.prepare_config(work, translator="identity")
    assert overridden.translation_backend == "identity"
    # the user's own file is never written
    stored = json.loads(source.read_text(encoding="utf-8"))
    assert stored["translation_backend"] == "libretranslate" and stored["quality_renderer"] == "auto"
    assert stored["running_on_start"] is True


def test_prepare_config_without_a_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GS.settings, "default_config_path", lambda: tmp_path / "gone" / "config.json")
    work = tmp_path / "work"
    work.mkdir()
    path, cfg = GS.prepare_config(work, translator="identity")
    assert path == work / "config.json" and not path.exists()
    assert cfg.translation_backend == "identity" and cfg.quality_renderer == "off"


# ------------------------------------------------------------------------------- synthetic runs
def _events(mode: str, worst_gap_s: float = 0.01) -> List[Event]:
    meta = {"mode": mode, "refresh_hz": 100.0, "frame_ms": 10.0, "steps": 2, "step_px": 8,
            "repeat": 1, "heartbeat_ms": 4, "backend": None, "ocr_engine": "mangaocr",
            "capacity": 200_000, "dropped": 0, "overlay": None}
    out: List[Event] = [(0.0, "phase", {"name": "drag", "edge": "start", "rep": 1}),
                        (0.5, "phase", {"name": "drag", "edge": "stop", "rep": 1})]
    out += [(t, "swap", {}) for t in (0.0, 0.01, 0.02, 0.02 + worst_gap_s)]
    out += [(t, "tick", {}) for t in (0.0, 0.004, 0.008)]
    out += [(0.10, "overlay_apply_ms", {"ms": 2.0, "blocks": 3, "fresh": 3}),
            (0.11, "overlay_typeset_ms", {"ms": 9.0}),
            (0.12, "overlay_layer_ms", {"ms": 4.0}),
            (0.13, "paint_ms", {"ms": 1.5})]
    out.append((9.0, "meta", meta))
    return out


def _dump(path: Path, events: List[Event]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for t, kind, payload in events:
            fh.write(json.dumps({"t": t, "kind": kind, **payload}) + "\n")
    return path


# ---------------------------------------------------------------------------------- analyse
def test_analyse_mode_writes_a_valid_report(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    run = _dump(tmp_path / "run.jsonl", _events("control"))
    report = tmp_path / "report.json"
    assert PG.main(["--analyse", str(run), "--report", str(report)]) == 0
    printed = capsys.readouterr().out
    assert "baseline" in printed and "not evaluated" in printed
    data = json.loads(report.read_text(encoding="utf-8"))
    assert G.validate_report(data) == [] and data["gate"] is None
    assert set(data["runs"]) == {"baseline"}
    assert data["jsonl"]["baseline"].endswith("run.jsonl") and "\\" not in data["jsonl"]["baseline"]


def test_analyse_against_a_baseline_and_the_gate_exit_codes(tmp_path: Path) -> None:
    base = _dump(tmp_path / "base.jsonl", _events("control"))
    good = _dump(tmp_path / "good.jsonl", _events("pipeline"))
    bad = _dump(tmp_path / "bad.jsonl", _events("pipeline", worst_gap_s=0.05))
    assert PG.main(["--analyse", str(good), "--against", str(base), "--assert-gate"]) == 0
    assert PG.main(["--analyse", str(bad), "--against", str(base), "--assert-gate"]) == 1
    assert PG.main(["--analyse", str(good), "--assert-gate"]) == 1  # no baseline: cannot be judged
    assert PG.main(["--analyse", str(bad), "--against", str(base)]) == 0  # no flag: run errors only


def test_assert_gate_fails_on_a_schema_invalid_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A gate read off a report that does not validate cannot be trusted."""
    base = _dump(tmp_path / "base.jsonl", _events("control"))
    good = _dump(tmp_path / "good.jsonl", _events("pipeline"))
    monkeypatch.setattr(PG.glass_report, "validate_report", lambda _doc: ["runs.baseline.pages is missing"])
    assert PG.main(["--analyse", str(good), "--against", str(base), "--assert-gate"]) == 1
    assert PG.main(["--analyse", str(good), "--against", str(base)]) == 0  # without the flag: only a log


def test_report_writing_refuses_non_finite_numbers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A NaN or a stray Path in a record must be loud, not silently stringified."""
    run = _dump(tmp_path / "run.jsonl", _events("control"))
    real = PG.glass_report.analyse

    def poisoned(events: Any, gap_factor: float = 2.5) -> Dict[str, Any]:
        record = real(events, gap_factor)
        record["swap"]["all"]["p50_ms"] = float("nan")
        return record

    monkeypatch.setattr(PG.glass_report, "analyse", poisoned)
    with pytest.raises(ValueError):
        PG.main(["--analyse", str(run), "--report", str(tmp_path / "r.json")])


# ----------------------------------------------------------------------------- parent mode
class _FakeRuns:
    """Stands in for ``subprocess.run``: writes the child's JSONL and reports ``code``."""

    def __init__(self, code: int = 0, worst_gap_s: float = 0.01) -> None:
        self.code = code
        self.worst_gap_s = worst_gap_s
        self.commands: List[List[str]] = []
        self.envs: List[Dict[str, str]] = []

    def __call__(self, cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        self.commands.append(list(cmd))
        self.envs.append(dict(kwargs.get("env") or {}))
        if self.code == 0:
            mode = cmd[cmd.index("--mode") + 1]
            gap = self.worst_gap_s if mode == "pipeline" else 0.01
            _dump(Path(cmd[cmd.index("--jsonl") + 1]), _events(mode, gap))
        return subprocess.CompletedProcess(cmd, self.code)

    def value(self, index: int, flag: str) -> str:
        cmd = self.commands[index]
        return cmd[cmd.index(flag) + 1]


def test_with_pipeline_drives_three_children_and_judges_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    fake = _FakeRuns()
    monkeypatch.setattr(PG.subprocess, "run", fake)
    report = tmp_path / "report.json"
    code = PG.main(["--with-pipeline", "--out", str(tmp_path), "--report", str(report), "--repeat", "3"])
    assert code == 0
    assert [fake.value(i, "--mode") for i in range(3)] == ["control", "session", "pipeline"]
    for index, command in enumerate(fake.commands):
        assert command[0] == sys.executable and Path(command[1]).name == "profile_glass.py"
        assert fake.value(index, "--repeat") == "3" and fake.value(index, "--max-events") == "300000"
        assert fake.value(index, "--steps") == "60" and fake.value(index, "--step-px") == "8"
        assert fake.value(index, "--heartbeat-ms") == "4" and fake.value(index, "--page-period") == "2.0"
        assert fake.value(index, "--gap-factor") == "2.5"
        assert "--translator" not in command
        assert Path(fake.value(index, "--jsonl")).parent == tmp_path
        assert fake.envs[index]["PYTHONUTF8"] == "1"
    data = json.loads(report.read_text(encoding="utf-8"))
    assert set(data["runs"]) == {"baseline", "session_idle", "pipeline"}
    assert G.validate_report(data) == [] and data["gate"]["pass"] is True
    assert set(data["jsonl"]) == {"baseline", "session_idle", "pipeline"}
    assert "pipeline" in capsys.readouterr().out
    # the report is always written next to the JSONLs too
    assert [p for p in tmp_path.glob("*-report.json")]


def test_with_pipeline_passes_the_translator_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRuns()
    monkeypatch.setattr(PG.subprocess, "run", fake)
    assert PG.main(["--with-pipeline", "--out", str(tmp_path), "--translator", "identity"]) == 0
    assert [fake.value(i, "--translator") for i in range(3)] == ["identity"] * 3


def test_with_pipeline_warns_once_before_an_online_translator_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _FakeRuns()
    monkeypatch.setattr(PG.subprocess, "run", fake)
    with caplog.at_level("WARNING", logger="profile_glass"):
        assert PG.main(["--with-pipeline", "--out", str(tmp_path), "--translator", "gemini"]) == 0
    online = [r for r in caplog.records if "online service" in r.getMessage()]
    assert len(online) == 1 and "gemini" in online[0].getMessage()


def test_with_pipeline_stays_quiet_for_an_offline_translator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _FakeRuns()
    monkeypatch.setattr(PG.subprocess, "run", fake)
    with caplog.at_level("WARNING", logger="profile_glass"):
        assert PG.main(["--with-pipeline", "--out", str(tmp_path), "--translator", "argos"]) == 0
    assert not [r for r in caplog.records if "online service" in r.getMessage()]


def test_with_pipeline_aborts_on_a_failing_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRuns(code=7)
    monkeypatch.setattr(PG.subprocess, "run", fake)
    assert PG.main(["--with-pipeline", "--out", str(tmp_path)]) == 7
    assert len(fake.commands) == 1  # it stops at the first failure


def test_with_pipeline_maps_a_crashed_child_to_a_usable_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native crash reports a 32-bit NTSTATUS (0xC0000005), which ``os._exit`` cannot take."""
    fake = _FakeRuns(code=3221225477)
    monkeypatch.setattr(PG.subprocess, "run", fake)
    assert PG.main(["--with-pipeline", "--out", str(tmp_path)]) == 1


def test_with_pipeline_assert_gate_fails_on_a_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRuns(worst_gap_s=0.05)
    monkeypatch.setattr(PG.subprocess, "run", fake)
    assert PG.main(["--with-pipeline", "--out", str(tmp_path), "--assert-gate"]) == 1
    assert PG.main(["--with-pipeline", "--out", str(tmp_path)]) == 0  # without the flag: run errors only
