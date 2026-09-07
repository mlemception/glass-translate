"""External-folder smoke test for the frozen ``GlassTranslate.exe`` (docs/GLASS_DESIGN.md section 6, step 5).

The exe is copied *alone* into a fresh ``%TEMP%/gt_extest_*`` folder and launched from there the way
Explorer would: ``PATH`` reduced to System32 / Windows / Wbem / WindowsPowerShell, no ``PYTHON*`` /
``QT_*`` / ``VIRTUAL_ENV`` / ``QML*`` variables, **no inherited std handles** (so ``sys.stdout`` and
``sys.stderr`` are ``None`` inside the app), ``TEMP``/``TMP`` pointed at the test folder so the onefile
``_MEI*`` extraction is observable, ``GLASSTRANSLATE_CONFIG`` -> a file in the test folder and
``GLASSTRANSLATE_SMOKE_LOG`` -> the JSON report the app writes (section 5).

Runs (``--runs`` selects; default ``static,A,B,C``):

* ``static`` - pefile: GUI subsystem, RT_MANIFEST with PerMonitorV2, RT_VERSION, 7 RT_ICON sizes.
* ``A``      - defaults + ``GLASSTRANSLATE_APPEARANCE=glass``, AUTOEXIT 8000: report fields and
               screenshot statistics (numpy) exactly as section 6 lists them.
* ``B``      - config ``running_on_start=true``, ``translation_backend="identity"``, AUTOEXIT 20000:
               the OCR engine constructs inside the frozen tree (``status_history``).
* ``C``      - onefile lifecycle: launch with AUTOEXIT 60000, wait for the report, ``taskkill /F``, then
               launch run A again and assert the stale ``_MEI*`` directory was swept by ``run.py``.

``--mechanics`` (used by ``build.py --allow-incomplete``) checks only what does not need the new UI:
static resources, launch + clean exit from the bare folder, ``_MEI`` extraction/cleanup, the log
file under ``%LOCALAPPDATA%/GlassTranslate/logs``, the kill/sweep lifecycle, exe size and cold start.

Prints a table and exits non-zero when any selected check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXE = ROOT / "dist" / "GlassTranslate.exe"
LOG_FILE = Path(os.environ.get("LOCALAPPDATA", "")) / "GlassTranslate" / "logs" / "glasstranslate.log"
FORBIDDEN_ENV_PREFIXES = ("PYTHON", "QT_", "VIRTUAL_ENV", "QML", "QSG_")
AUTOEXIT_A_MS = 8000
AUTOEXIT_B_MS = 20000
AUTOEXIT_C_MS = 60000
LAUNCH_GRACE_S = 90.0  # extra wall time allowed beyond AUTOEXIT before a run is declared hung

# Slab geometry (logical px): window = slab + 72 wide / + 80 tall; slab inset left/right 36, top 28, bottom 52.
SLAB_LEFT, SLAB_TOP, SLAB_RIGHT, SLAB_BOTTOM = 36, 28, 36, 52
SLAB_INSET = 40
CONTENT_PAD, TITLE_H, GAP, TAB_H = 14, 40, 10, 36


# ------------------------------------------------------------------------------------------ results
@dataclass
class Check:
    run: str
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Launch:
    """Outcome of one exe launch from the sandbox."""

    returncode: Optional[int]
    wall_s: float
    mei_seen: Set[str]
    mei_left: List[str]
    report: Optional[Dict[str, Any]]
    report_path: Path
    killed: bool = False
    report_wait_s: Optional[float] = None
    error: str = ""
    log_before: Tuple[int, float] = (0, 0.0)
    log_after: Tuple[int, float] = (0, 0.0)


@dataclass
class Results:
    checks: List[Check] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def add(self, run: str, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append(Check(run, name, bool(ok), detail))
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


# ------------------------------------------------------------------------------------------- static
def _resource_entries(pe: Any, type_id: int) -> List[bytes]:
    """Raw data of every resource of *type_id* (RT_ICON = 3, RT_GROUP_ICON = 14, RT_VERSION = 16, RT_MANIFEST = 24)."""
    out: List[bytes] = []
    directory = getattr(pe, "DIRECTORY_ENTRY_RESOURCE", None)
    if directory is None:
        return out
    for top in directory.entries:
        if top.id != type_id:
            continue
        for entry in top.directory.entries:
            for lang in entry.directory.entries:
                out.append(pe.get_data(lang.data.struct.OffsetToData, lang.data.struct.Size))
    return out


def _rc_text(rc: Optional[int]) -> str:
    """``0`` -> ``"rc=0"``; NTSTATUS-style crashes also in hex (``rc=3221226505 (0xC0000409)``)."""
    if rc is None:
        return "rc=None"
    return f"rc={rc}" + (f" (0x{rc & 0xFFFFFFFF:08X})" if rc not in (0, 1) else "")


def static_checks(exe: Path, res: Results) -> None:
    """pefile checks on the exe's resources and headers (no launch)."""
    import pefile

    RT_ICON, RT_GROUP_ICON, RT_VERSION, RT_MANIFEST = 3, 14, 16, 24
    size_mb = exe.stat().st_size / 1e6
    res.info["exe_size_mb"] = round(size_mb, 1)
    res.add("static", "exe size recorded", True, f"{size_mb:.1f} MB")
    pe = pefile.PE(str(exe), fast_load=True)
    try:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]])
        res.add("static", "GUI subsystem (no console)", pe.OPTIONAL_HEADER.Subsystem == 2,
                f"Subsystem={pe.OPTIONAL_HEADER.Subsystem}")
        manifests = _resource_entries(pe, RT_MANIFEST)
        text = manifests[0].decode("utf-8", errors="replace") if manifests else ""
        res.add("static", "RT_MANIFEST present", bool(manifests), f"{len(manifests[0]) if manifests else 0} bytes")
        res.add("static", "manifest dpiAwareness PerMonitorV2", "PerMonitorV2" in text and "<dpiAwareness" in text)
        res.add("static", "manifest dpiAware true/pm", "true/pm" in text)
        res.add("static", "manifest asInvoker", 'level="asInvoker"' in text)
        versions = _resource_entries(pe, RT_VERSION)
        vtext = versions[0].decode("utf-16-le", errors="ignore") if versions else ""
        res.add("static", "RT_VERSION present", bool(versions), f"{len(versions[0]) if versions else 0} bytes")
        res.add("static", "version has ProductVersion + FileDescription",
                "ProductVersion" in vtext and "FileDescription" in vtext)
        icons = _resource_entries(pe, RT_ICON)
        res.add("static", "7 RT_ICON images", len(icons) == 7, f"{len(icons)} images")
        groups = _resource_entries(pe, RT_GROUP_ICON)
        sizes: List[int] = []
        if groups:
            count = struct.unpack("<H", groups[0][4:6])[0]
            sizes = [groups[0][6 + 14 * i] or 256 for i in range(count)]
        res.add("static", "RT_GROUP_ICON lists 16..256", sorted(sizes) == [16, 24, 32, 48, 64, 128, 256], f"{sizes}")
    finally:
        pe.close()


# ------------------------------------------------------------------------------------------ sandbox
class Sandbox:
    """A fresh ``%TEMP%/gt_extest_*`` folder holding one copy of the exe and its config/report files."""

    def __init__(self, exe: Path) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="gt_extest_"))
        self.exe = self.dir / exe.name
        shutil.copy2(exe, self.exe)
        self.config = self.dir / "config.json"
        self.base_env = self._make_env()
        self.last_killed_pids: List[int] = []

    def _make_env(self) -> Dict[str, str]:
        sysroot = os.environ.get("SystemRoot", r"C:\Windows")
        env = {
            "PATH": os.pathsep.join([sysroot + r"\System32", sysroot, sysroot + r"\System32\Wbem",
                                     sysroot + r"\System32\WindowsPowerShell\v1.0"]),
            "SystemRoot": sysroot,
            "SystemDrive": os.environ.get("SystemDrive", "C:"),
            "TEMP": str(self.dir),
            "TMP": str(self.dir),
            "USERPROFILE": os.environ.get("USERPROFILE", ""),
            "APPDATA": os.environ.get("APPDATA", ""),
            "LOCALAPPDATA": os.environ.get("LOCALAPPDATA", ""),
            "USERNAME": os.environ.get("USERNAME", ""),
            "COMSPEC": sysroot + r"\System32\cmd.exe",
            "PATHEXT": ".COM;.EXE;.BAT;.CMD",
            "NUMBER_OF_PROCESSORS": os.environ.get("NUMBER_OF_PROCESSORS", "4"),
            "GLASSTRANSLATE_CONFIG": str(self.config),
        }
        for key in env:
            if key.upper().startswith(FORBIDDEN_ENV_PREFIXES):
                raise RuntimeError(f"sandbox env leaks {key}")
        return env

    def write_config(self, overrides: Dict[str, Any]) -> None:
        self.config.write_text(json.dumps(overrides, indent=2), encoding="utf-8")

    def mei_dirs(self) -> List[str]:
        return sorted(d for d in os.listdir(self.dir) if d.upper().startswith("_MEI"))

    @staticmethod
    def _log_state() -> Tuple[int, float]:
        try:
            st = LOG_FILE.stat()
            return st.st_size, st.st_mtime
        except OSError:
            return 0, 0.0

    def launch(self, tag: str, autoexit_ms: int, extra_env: Optional[Dict[str, str]] = None, *,
               kill_when: str = "never", mechanics_kill_delay_s: float = 6.0) -> Launch:
        """Run the exe once.

        *kill_when*: ``"never"`` waits for the normal exit; ``"report"`` kills the process as soon as the
        smoke report appears (run C); ``"after_mei"`` kills it *mechanics_kill_delay_s* after the ``_MEI``
        dir shows up (run C without report support in the app).
        """
        report_path = self.dir / f"smoke_{tag}.json"
        if report_path.exists():
            report_path.unlink()
        env = dict(self.base_env)
        env["GLASSTRANSLATE_AUTOEXIT_MS"] = str(autoexit_ms)
        env["GLASSTRANSLATE_SMOKE_LOG"] = str(report_path)
        env.update(extra_env or {})
        for key in env:
            if key.upper().startswith(FORBIDDEN_ENV_PREFIXES):
                raise RuntimeError(f"sandbox env leaks {key}")
        log_before = self._log_state()
        t0 = time.perf_counter()
        # No stdin/stdout/stderr arguments on purpose: the GUI child then gets NO std handles, like Explorer.
        proc = subprocess.Popen([str(self.exe)], cwd=str(self.dir), env=env)
        seen: Set[str] = set()
        killed = False
        report_wait: Optional[float] = None
        mei_first_seen: Optional[float] = None
        deadline = t0 + autoexit_ms / 1000 + LAUNCH_GRACE_S
        error = ""
        while proc.poll() is None:
            now = time.perf_counter()
            current = self.mei_dirs()
            seen.update(current)
            if current and mei_first_seen is None:
                mei_first_seen = now
            if kill_when == "report" and report_path.exists() and report_wait is None:
                report_wait = now - t0
                time.sleep(0.3)  # let the writer close the file
                killed = self._kill(proc)
            elif kill_when == "after_mei" and mei_first_seen is not None and now - mei_first_seen > mechanics_kill_delay_s:
                killed = self._kill(proc)
            if now > deadline:
                error = f"timeout after {now - t0:.0f} s"
                self._kill(proc)
                killed = True
                break
            time.sleep(0.05)
        proc.wait(timeout=30)
        wall = time.perf_counter() - t0
        time.sleep(0.3)  # bootloader cleanup of _MEI happens right after the child exits
        report: Optional[Dict[str, Any]] = None
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                error = error or f"unreadable report: {exc}"
        return Launch(proc.returncode, wall, seen, self.mei_dirs(), report, report_path, killed, report_wait, error,
                      log_before, self._log_state())

    @staticmethod
    def _descendant_pids(pid: int) -> Set[int]:
        """PIDs of the live processes whose parent chain reaches *pid* (ToolHelp snapshot, no external tools).

        PyInstaller onefile is a two-process model: the bootloader parent extracts the bundle and spawns
        the Python child.  Killing only the parent orphans the child, which keeps running and keeps its
        ``_MEI`` directory locked - the opposite of the stale-dir condition run C wants to create.
        """
        import ctypes
        from ctypes import wintypes

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                        ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wintypes.DWORD),
                        ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                        ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                        ("szExeFile", ctypes.c_wchar * 260)]

        k32 = ctypes.windll.kernel32
        snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
        parents: Dict[int, int] = {}
        if snap == ctypes.c_void_p(-1).value or snap == 0:
            return set()
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            ok = k32.Process32FirstW(snap, ctypes.byref(entry))
            while ok:
                parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                ok = k32.Process32NextW(snap, ctypes.byref(entry))
        finally:
            k32.CloseHandle(snap)
        found: Set[int] = set()
        frontier = {pid}
        while frontier:
            nxt = {p for p, parent in parents.items() if parent in frontier and p not in found and p != pid}
            found |= nxt
            frontier = nxt
        return found

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        import ctypes

        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)

    def _kill(self, proc: subprocess.Popen) -> bool:
        """Hard kill the whole process tree like Task Manager's "End process tree" (``taskkill /F /T``).

        Returns True when the parent and every descendant are gone within 15 s.
        """
        pids = {proc.pid} | self._descendant_pids(proc.pid)
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            pass
        for pid in pids:  # belt and braces: taskkill /T can miss a child that re-parented
            if self._pid_alive(pid):
                try:
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=15)
                except (OSError, subprocess.SubprocessError):
                    pass
        if proc.poll() is None:
            proc.kill()
        deadline = time.perf_counter() + 15
        while time.perf_counter() < deadline:
            if not any(self._pid_alive(pid) for pid in pids):
                self.last_killed_pids = sorted(pids)
                return True
            time.sleep(0.1)
        self.last_killed_pids = sorted(pids)
        return False

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


# --------------------------------------------------------------------------------------- assertions
def _launch_mechanics(run: str, lau: Launch, res: Results, autoexit_ms: int, *, expect_exit: bool = True) -> None:
    """Checks that hold for any frozen build: exit, extraction, cleanup, log file, cold start."""
    if expect_exit:
        res.add(run, "exe exits 0 from the bare folder", lau.returncode == 0 and not lau.killed,
                f"{_rc_text(lau.returncode)} wall={lau.wall_s:.1f}s {lau.error}".strip())
    res.add(run, "_MEI extraction observed under TEMP", bool(lau.mei_seen), ", ".join(sorted(lau.mei_seen)) or "none")
    if expect_exit:
        res.add(run, "_MEI removed after normal exit", not lau.mei_left, ", ".join(lau.mei_left) or "clean")
        overhead = lau.wall_s - autoexit_ms / 1000
        res.info.setdefault("cold_start_s", {})[run] = round(overhead, 2)
        res.add(run, "cold start recorded", True, f"~{overhead:.1f} s (wall {lau.wall_s:.1f} s - autoexit {autoexit_ms / 1000:.0f} s)")
    size_b, mtime_b = lau.log_before
    size_a, mtime_a = lau.log_after
    written = LOG_FILE.exists() and (size_a != size_b or mtime_a > mtime_b)
    res.add(run, "log file written under %LOCALAPPDATA%/GlassTranslate/logs", written,
            f"{LOG_FILE} {size_b}->{size_a} bytes" if LOG_FILE.exists() else f"{LOG_FILE} missing")


def _screenshot_checks(run: str, report: Dict[str, Any], res: Results) -> None:
    """Section 6 run A pixel assertions on the ``grabWindow()`` PNG named in the report."""
    import numpy as np
    from PIL import Image

    path = report.get("screenshot")
    if not path or not Path(path).exists():
        res.add(run, "screenshot exists", False, f"{path!r}")
        return
    img = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32)
    h, w = img.shape[:2]
    dpr = float(report.get("dpr") or (report.get("window") or {}).get("dpr") or 1.0)
    res.add(run, "screenshot exists", True, f"{w}x{h} dpr={dpr}")
    corners = [img[0, 0, 3], img[0, w - 1, 3], img[h - 1, 0, 3], img[h - 1, w - 1, 3]]
    res.add(run, "alpha == 0 at the four window corners", all(a == 0 for a in corners), f"{[int(a) for a in corners]}")
    sx0, sy0 = int(SLAB_LEFT * dpr), int(SLAB_TOP * dpr)
    sx1, sy1 = int(w - SLAB_RIGHT * dpr), int(h - SLAB_BOTTOM * dpr)
    ins = int(SLAB_INSET * dpr)
    inner = img[sy0 + ins:sy1 - ins, sx0 + ins:sx1 - ins]
    if inner.size == 0:
        res.add(run, "slab region non-empty", False, f"slab {sx0},{sy0}-{sx1},{sy1}")
        return
    alpha_mean = float(inner[..., 3].mean())
    res.add(run, "alpha mean >= 250 inside slab (inset 40)", alpha_mean >= 250, f"{alpha_mean:.1f}")
    rgb_std = float(inner[..., :3].reshape(-1, 3).std(axis=0).mean())
    res.add(run, "RGB std >= 8 inside slab (glass not flat)", rgb_std >= 8, f"{rgb_std:.1f}")
    ty0 = sy0 + int((CONTENT_PAD + TITLE_H + GAP) * dpr)
    ty1 = ty0 + int(TAB_H * dpr)
    tab = img[ty0:ty1, sx0 + ins:sx1 - ins, :3]
    if tab.size == 0:
        res.add(run, "tab-bar row mean |diff| vs slab >= 6", False, "empty tab row")
        return
    diff = float(np.abs(tab.reshape(-1, 3).mean(axis=0) - inner[..., :3].reshape(-1, 3).mean(axis=0)).mean())
    res.add(run, "tab-bar row mean |diff| vs slab >= 6", diff >= 6, f"{diff:.1f}")


def _report_checks_a(run: str, lau: Launch, res: Results) -> None:
    rep = lau.report
    if rep is None:
        res.add(run, "smoke report written (GLASSTRANSLATE_SMOKE_LOG)", False,
                f"{lau.report_path.name} missing - app.py section 5 support absent? {lau.error}".strip())
        return
    res.add(run, "smoke report written (GLASSTRANSLATE_SMOKE_LOG)", True, lau.report_path.name)
    res.add(run, "frozen + meipass reported", bool(rep.get("frozen")) and bool(rep.get("meipass")),
            f"frozen={rep.get('frozen')} meipass={rep.get('meipass')}")
    res.add(run, 'qml_status == "Ready"', rep.get("qml_status") == "Ready", str(rep.get("qml_status")))
    res.add(run, "qml_errors == []", rep.get("qml_errors") == [], str(rep.get("qml_errors"))[:200])
    res.add(run, "qml_warnings == []", rep.get("qml_warnings") == [], str(rep.get("qml_warnings"))[:200])
    fx = rep.get("shader_effects") or {}
    res.add(run, "shader_effects.error == 0", fx.get("error") == 0,
            f"total={fx.get('total')} compiled={fx.get('compiled')} error={fx.get('error')}")
    res.add(run, 'rhi_backend == "D3D11"', rep.get("rhi_backend") == "D3D11", str(rep.get("rhi_backend")))
    app = rep.get("appearance") or {}
    res.add(run, 'appearance.mode == "glass"', app.get("mode") == "glass", str(app))
    res.add(run, "backdrop_serial >= 1", isinstance(rep.get("backdrop_serial"), int) and rep["backdrop_serial"] >= 1,
            str(rep.get("backdrop_serial")))
    res.add(run, "dpi_awareness == 2", rep.get("dpi_awareness") == 2, str(rep.get("dpi_awareness")))
    res.add(run, "control_exposed", bool(rep.get("control_exposed")))
    res.add(run, "overlay_shown", bool(rep.get("overlay_shown")))
    res.add(run, "pages_ok", bool(rep.get("pages_ok")))
    ro = rep.get("resources_ok") or {}
    res.add(run, "resources_ok all true", bool(ro) and all(bool(v) for v in ro.values()), str(ro))
    res.add(run, "fonts_ok (Anime Ace 2.0 BB)", bool(rep.get("fonts_ok")))
    _screenshot_checks(run, rep, res)


def run_a(sb: Sandbox, res: Results, mechanics: bool, tag: str = "A") -> Launch:
    sb.write_config({})
    lau = sb.launch(tag, AUTOEXIT_A_MS, {"GLASSTRANSLATE_APPEARANCE": "glass"})
    _launch_mechanics(tag, lau, res, AUTOEXIT_A_MS)
    if mechanics:
        res.info[f"report_{tag}"] = "present" if lau.report is not None else "absent"
    else:
        _report_checks_a(tag, lau, res)
    return lau


def run_b(sb: Sandbox, res: Results, mechanics: bool) -> Launch:
    sb.write_config({"running_on_start": True, "translation_backend": "identity"})
    lau = sb.launch("B", AUTOEXIT_B_MS)
    _launch_mechanics("B", lau, res, AUTOEXIT_B_MS)
    if mechanics:
        res.info["report_B"] = "present" if lau.report is not None else "absent"
        return lau
    rep = lau.report
    if rep is None:
        res.add("B", "smoke report written", False, f"{lau.report_path.name} missing {lau.error}".strip())
        return lau
    hist = [str(s) for s in rep.get("status_history") or []]
    res.add("B", 'status_history has "OCR: rapidocr on ..."', any(s.startswith("OCR: rapidocr on") for s in hist),
            next((s for s in hist if s.startswith("OCR: rapidocr on")), f"{hist[-3:]}"))
    bad = [s for s in hist if s.startswith("Engine error") or s.startswith("Error:")]
    res.add("B", 'no "Engine error"/"Error:" status', not bad, "; ".join(bad)[:200])
    return lau


def run_c(sb: Sandbox, res: Results, mechanics: bool) -> None:
    sb.write_config({})
    kill_when = "after_mei" if mechanics else "report"
    lau = sb.launch("C", AUTOEXIT_C_MS, {"GLASSTRANSLATE_APPEARANCE": "glass"}, kill_when=kill_when)
    if not mechanics:
        res.add("C", "report appeared before the kill", lau.report is not None and lau.report_wait is not None,
                f"after {lau.report_wait:.1f} s" if lau.report_wait is not None else lau.error or "no report")
    res.add("C", "process tree hard-killed while running", lau.killed and not lau.error,
            (lau.error or _rc_text(lau.returncode)) + f" pids={sb.last_killed_pids}")
    res.add("C", "_MEI left behind by the kill (precondition)", bool(lau.mei_left), ", ".join(lau.mei_left) or "none")
    stale = set(lau.mei_left)
    lau2 = run_a(sb, res, mechanics, tag="C2")
    res.add("C", "stale _MEI swept by the next launch (run.py)", stale and not (stale & set(lau2.mei_left)) and not lau2.mei_left,
            f"stale={sorted(stale)} left={lau2.mei_left}")


# ------------------------------------------------------------------------------------------- output
def print_table(res: Results) -> None:
    width = max([len(c.name) for c in res.checks] + [10])
    print(f"\n{'run':<7}{'check':<{width + 2}}{'result':<7}detail")
    print("-" * (width + 40))
    for c in res.checks:
        print(f"{c.run:<7}{c.name:<{width + 2}}{'PASS' if c.ok else 'FAIL':<7}{c.detail}")
    failed = [c for c in res.checks if not c.ok]
    print("-" * (width + 40))
    if res.info:
        print("info: " + json.dumps(res.info, sort_keys=True))
    print(f"RESULT: {'PASS' if res.ok else 'FAIL'} ({len(res.checks) - len(failed)}/{len(res.checks)} checks)")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test dist/GlassTranslate.exe from a bare external folder.")
    p.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    p.add_argument("--runs", default="static,A,B,C", help="comma list of static,A,B,C (default all)")
    p.add_argument("--mechanics", action="store_true",
                   help="only launch/exit/_MEI/log/lifecycle checks (no smoke-report assertions)")
    p.add_argument("--keep", action="store_true", help="keep the temp folder")
    p.add_argument("--json", type=Path, help="write the check list + info to this JSON file")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    args = parse_args(argv)
    exe = args.exe if args.exe.is_absolute() else ROOT / args.exe
    if not exe.exists():
        print(f"smoke_test: exe not found: {exe}", file=sys.stderr)
        return 2
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    res = Results()
    res.info["exe"] = str(exe)
    res.info["mode"] = "mechanics" if args.mechanics else "full"
    if "static" in runs:
        static_checks(exe, res)
    sb: Optional[Sandbox] = None
    if any(r in runs for r in ("A", "B", "C")):
        sb = Sandbox(exe)
        print(f"sandbox: {sb.dir}\nPATH: {sb.base_env['PATH']}")
        try:
            if "A" in runs:
                run_a(sb, res, args.mechanics)
            if "B" in runs:
                run_b(sb, res, args.mechanics)
            if "C" in runs:
                run_c(sb, res, args.mechanics)
        finally:
            if args.keep:
                print(f"kept: {sb.dir}")
            else:
                sb.cleanup()
    print_table(res)
    if args.json:
        args.json.write_text(json.dumps({"ok": res.ok, "info": res.info,
                                         "checks": [c.__dict__ for c in res.checks]}, indent=2), encoding="utf-8")
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
