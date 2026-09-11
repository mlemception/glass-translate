"""Freeze the quality-renderer sidecar into ``dist/renderer/`` and prove it runs.

    renderer\\.venv\\Scripts\\python is used for the freeze; run this with the APP venv:

    .venv\\Scripts\\python build_renderer.py           # build, then self-check
    .venv\\Scripts\\python build_renderer.py --test    # self-check an existing dist/renderer

Steps:
  (1) preflight   - renderer\\.venv with PyInstaller, pefile and an importable torch
  (2) PyInstaller - packaging/glassrenderer.spec (onedir, console, the measured prune list)
  (3) self-check  - ``glassrenderer.exe serve --fake`` from a stripped environment with dead
                    proxies: READY within 60 s, /health answers, stdin EOF ends the process
  (4) report      - sizes before/after the prune, elapsed time, build/renderer-build.json

Full PyInstaller output goes to ``build/build-renderer.log``; the console shows the
``[prune]`` lines, warnings and errors only.  Exit status is non-zero on any failure.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
PACKAGING = ROOT / "packaging"
DIST = Path(os.environ.get("GT_RENDERER_DIST") or (ROOT / "dist"))
BUILD = ROOT / "build"
WORK = Path(os.environ.get("GT_RENDERER_WORK") or (BUILD / "renderer"))
SPEC = PACKAGING / "glassrenderer.spec"
RENDERER_PYTHON = ROOT / "renderer" / ".venv" / "Scripts" / "python.exe"
DIST_DIR = DIST / "renderer"  # the spec's COLLECT(name="renderer")
SIDECAR_EXE = DIST_DIR / "glassrenderer.exe"
LOG_PATH = BUILD / "build-renderer.log"
REPORT_PATH = BUILD / "renderer-build.json"

READY_TIMEOUT_S = 60.0
# Research R1 section 6: after a *real* job, ExitProcess runs DLL_PROCESS_DETACH for ~250
# DLLs and takes up to 19 s; fake mode is 0.15-0.17 s.  60 s is generous for both.
EXIT_TIMEOUT_S = 60.0
DEAD_PROXY = "http://127.0.0.1:9"
_PRUNE_BEFORE = re.compile(r"\[prune\] before:.*?total ([\d.]+) MB")
_PRUNE_AFTER = re.compile(r"\[prune\] after\s*:.*?total ([\d.]+) MB")
_PRUNE_SAVED = re.compile(r"\(saved ([\d.]+) MB\)")


class RendererBuildError(RuntimeError):
    """A build step failed; the message is already user-readable."""


def _banner(step: str, title: str) -> None:
    print(f"\n=== [{step}] {title} " + "=" * max(0, 70 - len(step) - len(title)), flush=True)


def _mb(value: float) -> str:
    return f"{value:,.1f} MB"


def _folder_stats(folder: Path) -> Dict[str, float]:
    files = [p for p in folder.rglob("*") if p.is_file()] if folder.is_dir() else []
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


# ------------------------------------------------------------------------------------------ steps
def step_preflight() -> Dict[str, str]:
    """The sidecar venv must hold PyInstaller, pefile and a working torch."""
    _banner("1/4", "Preflight (renderer/.venv)")
    if sys.platform != "win32":
        raise RendererBuildError("build_renderer.py targets Windows")
    if not RENDERER_PYTHON.is_file():
        raise RendererBuildError(f"{RENDERER_PYTHON} not found - run renderer\\install.bat first")
    if not SPEC.is_file():
        raise RendererBuildError(f"missing {SPEC.relative_to(ROOT)}")
    probe = (
        "import json, PyInstaller, pefile, torch, torchvision; "
        "print(json.dumps({'pyinstaller': PyInstaller.__version__, 'pefile': pefile.__version__, "
        "'torch': torch.__version__, 'torchvision': torchvision.__version__}))"
    )
    proc = subprocess.run([str(RENDERER_PYTHON), "-c", probe], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RendererBuildError(
            f"the sidecar venv cannot import its build tools: {proc.stderr.strip()}"
        )
    versions = json.loads(proc.stdout.strip().splitlines()[-1])
    print("  " + "  ".join(f"{name} {value}" for name, value in versions.items()))
    return versions


def _interesting(line: str) -> bool:
    low = line.lower()
    return (
        line.startswith("[prune]")
        or line.startswith("[spec]")
        or "warning" in low
        or "error" in low
        or "completed successfully" in low
        or low.startswith("traceback")
    )


def _stream_build(cmd: Sequence[str], env: Dict[str, str]) -> List[str]:
    """Run ``cmd``, tee everything to the log, echo the interesting lines, return them."""
    BUILD.mkdir(parents=True, exist_ok=True)
    shown: List[str] = []
    with LOG_PATH.open("w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            if _interesting(line):
                shown.append(line.rstrip())
                print("  " + line.rstrip(), flush=True)
        rc = proc.wait()
    if rc != 0:
        tail = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        print("\n".join("  | " + t for t in tail))
        raise RendererBuildError(f"PyInstaller exited with {rc} (see {LOG_PATH})")
    return shown


def _prune_sizes(lines: Sequence[str]) -> Dict[str, float]:
    """The spec prints the before/after totals; the driver reads them back."""
    sizes: Dict[str, float] = {}
    for line in lines:
        before, after = _PRUNE_BEFORE.search(line), _PRUNE_AFTER.search(line)
        if before:
            sizes["before_mb"] = float(before.group(1))
        elif after:
            sizes["after_mb"] = float(after.group(1))
            # Take the spec's own figure rather than re-deriving it: recomputing the
            # difference from two rounded numbers drifts by a tenth of a MB.
            saved = _PRUNE_SAVED.search(line)
            if saved:
                sizes["saved_mb"] = float(saved.group(1))
    return sizes


def step_pyinstaller() -> Dict[str, float]:
    """Freeze the sidecar; return the prune sizes and the elapsed time."""
    _banner("2/4", f"PyInstaller ({SPEC.relative_to(ROOT)}, onedir)")
    env = dict(os.environ)
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    cmd = [RENDERER_PYTHON, "-m", "PyInstaller", "--noconfirm", "--clean", "--log-level", "INFO",
           "--distpath", str(DIST), "--workpath", str(WORK), str(SPEC)]
    print("$ " + " ".join(str(c) for c in cmd))
    print(f"(full log: {LOG_PATH.relative_to(ROOT)})")
    started = time.perf_counter()
    lines = _stream_build(cmd, env)
    elapsed = time.perf_counter() - started
    if not SIDECAR_EXE.is_file():
        raise RendererBuildError(f"PyInstaller reported success but {SIDECAR_EXE} is missing")
    sizes = _prune_sizes(lines)
    sizes["elapsed_s"] = round(elapsed, 1)
    stats = _folder_stats(DIST_DIR)
    print(f"\nbuilt {DIST_DIR.relative_to(ROOT)}: {int(stats['files'])} files, "
          f"{_mb(stats['bytes'] / 1e6)} in {elapsed:.0f} s")
    if "before_mb" in sizes:
        print(f"payload {_mb(sizes['before_mb'])} -> {_mb(sizes['after_mb'])} "
              f"(pruned {_mb(sizes['saved_mb'])})")
    return sizes


# ------------------------------------------------------------------------------------- self-check
def _stripped_environment(token: str) -> Dict[str, str]:
    """A minimal environment: System32 family on PATH, dead proxies, no Python anywhere."""
    windir = os.environ.get("SystemRoot", r"C:\Windows")
    path = os.pathsep.join([
        fr"{windir}\system32", windir, fr"{windir}\System32\Wbem",
        fr"{windir}\System32\WindowsPowerShell\v1.0",
    ])
    env = {
        "SystemRoot": windir, "windir": windir, "PATH": path,
        "PATHEXT": ".COM;.EXE;.BAT;.CMD", "COMSPEC": fr"{windir}\system32\cmd.exe",
        "SystemDrive": os.environ.get("SystemDrive", "C:"),
        "NUMBER_OF_PROCESSORS": os.environ.get("NUMBER_OF_PROCESSORS", "8"),
        "PROCESSOR_ARCHITECTURE": os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"),
        "TEMP": os.environ.get("TEMP", r"C:\Windows\Temp"),
        "TMP": os.environ.get("TMP", r"C:\Windows\Temp"),
        "GT_RENDERER_TOKEN": token,
        "HTTP_PROXY": DEAD_PROXY, "HTTPS_PROXY": DEAD_PROXY,
        "http_proxy": DEAD_PROXY, "https_proxy": DEAD_PROXY,
    }
    return env


def _drain(stream: Any, sink: List[str]) -> None:
    """Keep the sidecar's stderr moving: a full 64 KB pipe would block the process."""
    for line in stream:
        sink.append(line)


def _await_ready(proc: subprocess.Popen, timeout: float, stderr: List[str]) -> str:
    """The single ``READY <port>`` line, or a build error showing what came instead."""
    captured: List[str] = []

    def reader() -> None:
        assert proc.stdout is not None
        captured.append(proc.stdout.readline())

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    thread.join(timeout)
    if not captured or not captured[0].strip():
        proc.kill()
        raise RendererBuildError(
            f"no READY line within {timeout:.0f} s; sidecar stderr:\n" + "".join(stderr[-30:])
        )
    return captured[0].strip()


def _shutdown(proc: subprocess.Popen) -> Tuple[int, float]:
    """Close stdin (which the sidecar reads as "the parent died") and wait for the exit."""
    started = time.perf_counter()
    assert proc.stdin is not None
    proc.stdin.close()
    try:
        code = proc.wait(timeout=EXIT_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise RendererBuildError(
            f"the sidecar did not exit within {EXIT_TIMEOUT_S:.0f} s of stdin EOF"
        ) from exc
    return code, time.perf_counter() - started


def _health(port: int, token: str) -> Dict[str, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.request("GET", "/health", headers={"X-GT-Token": token})
        response = connection.getresponse()
        payload = json.loads(response.read())
        if response.status != 200:
            raise RendererBuildError(f"/health answered {response.status}: {payload}")
        return payload
    finally:
        connection.close()


def _network_mentions(stderr: Sequence[str]) -> List[str]:
    """Any stderr line hinting the sidecar tried to reach the network (it must not)."""
    text = "".join(stderr).lower()
    return [word for word in ("http://", "https://", "download", "proxy", "huggingface.co")
            if word in text]


def _spawn_sidecar(models_dir: str, token: str) -> subprocess.Popen:
    return subprocess.Popen(
        [str(SIDECAR_EXE), "serve", "--fake", "--models-dir", models_dir],
        cwd=str(DIST_DIR), env=_stripped_environment(token),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )


def step_self_check() -> Dict[str, Any]:
    """Run the frozen sidecar in fake mode from a stripped environment and shut it down."""
    _banner("3/4", "Self-check (serve --fake, stripped PATH, dead proxies)")
    if not SIDECAR_EXE.is_file():
        raise RendererBuildError(f"{SIDECAR_EXE} not found - build first (or drop --test)")
    token = secrets.token_hex(32)
    models_dir = tempfile.mkdtemp(prefix="glassrenderer-selfcheck-")
    started = time.perf_counter()
    proc = _spawn_sidecar(models_dir, token)
    stderr: List[str] = []
    threading.Thread(target=_drain, args=(proc.stderr, stderr), daemon=True).start()
    try:
        line = _await_ready(proc, READY_TIMEOUT_S, stderr)
        ready_s = time.perf_counter() - started
        if not line.startswith("READY "):
            raise RendererBuildError(f"first stdout line was {line!r}, not 'READY <port>'")
        print(f"  READY in {ready_s:.2f} s on port {line.split()[1]}")
        health = _health(int(line.split()[1]), token)
        print(f"  /health mode={health.get('mode')} device={health.get('device')} "
              f"load_error={health.get('load_error')}")
        if health.get("mode") != "fake" or health.get("load_error"):
            raise RendererBuildError(f"/health is not a healthy fake sidecar: {health}")
        code, exit_s = _shutdown(proc)
    finally:
        if proc.poll() is None:
            proc.kill()
        shutil.rmtree(models_dir, ignore_errors=True)
    print(f"  exited {code} in {exit_s:.2f} s after stdin EOF")
    if code != 0:
        raise RendererBuildError(f"fake-mode sidecar exited with {code}, expected 0")
    return {"ready_s": round(ready_s, 2), "exit_s": round(exit_s, 2), "exit_code": code,
            "health": health, "offline": not _network_mentions(stderr)}


def step_report(sizes: Dict[str, float], check: Dict[str, Any], versions: Dict[str, str]) -> None:
    _banner("4/4", "Summary")
    stats = _folder_stats(DIST_DIR)
    report = {
        "before_mb": sizes.get("before_mb"),
        "after_mb": sizes.get("after_mb"),
        "saved_mb": sizes.get("saved_mb"),
        "elapsed_s": sizes.get("elapsed_s"),
        "ready_s": check.get("ready_s"),
        "exit_s": check.get("exit_s"),
        "offline": check.get("offline"),
        "files": int(stats["files"]),
        "folder_bytes": int(stats["bytes"]),
        "versions": versions,
    }
    for key, value in report.items():
        if key != "versions":
            print(f"  {key:<14} {value}")
    BUILD.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {REPORT_PATH.relative_to(ROOT)}")


# ------------------------------------------------------------------------------------------- main
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--test", action="store_true",
                        help="self-check an existing dist/renderer (skip the build)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[union-attr]
    args = parse_args(argv)
    try:
        versions = step_preflight()
        sizes = {} if args.test else step_pyinstaller()
        check = step_self_check()
        step_report(sizes, check, versions)
    except RendererBuildError as exc:
        print(f"\nRENDERER BUILD FAILED: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nRENDERER BUILD INTERRUPTED", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
