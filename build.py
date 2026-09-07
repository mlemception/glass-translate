"""One-command Windows build of ``dist/GlassTranslate.exe`` (docs/GLASS_DESIGN.md section 6).

    .venv\\Scripts\\python build.py              # steps 1-5: build, then smoke-test the exe
    .venv\\Scripts\\python build.py --no-test    # steps 1-4 only
    .venv\\Scripts\\python build.py --test       # step 5 only, against the existing dist exe

Steps:
  (1) tools/build_resources.py       - compile shaders (.qsb) + Qt resources (resources_rc.py)
  (2) packaging/make_icon.py         - packaging/glasstranslate.ico (7 sizes) from glasstranslate/ui/icons/app.png
  (3) packaging/make_version_info.py - packaging/version_info.txt from glasstranslate.__version__
  (4) PyInstaller                    - packaging/GlassTranslate.spec (onefile, no console, PerMonitorV2 manifest)
  (5) packaging/smoke_test.py        - copy the exe alone into a bare %TEMP% folder, run it with no Python on
                                       PATH and assert the smoke report (runs A, B, C)

Exit status is non-zero on any failure.  Full PyInstaller output goes to ``build/build.log``; the
console shows the prune summary, warnings and errors only.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
PACKAGING = ROOT / "packaging"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
SPEC = PACKAGING / "GlassTranslate.spec"
RESOURCES_SCRIPT = ROOT / "tools" / "build_resources.py"
SMOKE_SCRIPT = PACKAGING / "smoke_test.py"
DEFAULT_ENTRY = ROOT / "run.py"
DEFAULT_NAME = "GlassTranslate"
FALLBACK_VERSION = "0.0.0"

sys.path.insert(0, str(PACKAGING))
import make_icon  # noqa: E402
import make_version_info  # noqa: E402


class BuildError(RuntimeError):
    """A build step failed; the message is already user-readable."""


def _child_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra:
        env.update(extra)
    return env


def _banner(step: str, title: str) -> None:
    print(f"\n=== [{step}] {title} " + "=" * max(0, 70 - len(step) - len(title)), flush=True)


def _run(cmd: Sequence[str], *, env: Optional[Dict[str, str]] = None, cwd: Path = ROOT) -> None:
    """Run *cmd* streaming its output; raise ``BuildError`` on a non-zero exit."""
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env or _child_env())
    if proc.returncode != 0:
        raise BuildError(f"{Path(cmd[0]).name} exited with {proc.returncode}")


def _preflight() -> None:
    if sys.platform != "win32":
        raise BuildError("build.py targets Windows (PyInstaller builds are not cross-platform)")
    missing = []
    for mod in ("PyInstaller", "pefile", "PIL"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise BuildError(
            f"missing build tools {missing}: run  {sys.executable} -m pip install -r requirements-build.txt"
        )
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        print(f"WARNING: not running from {venv_python}; the exe will freeze {sys.executable}'s site-packages")
    import PyInstaller

    print(f"python {platform.python_version()}  PyInstaller {PyInstaller.__version__}  root {ROOT}")


# ------------------------------------------------------------------------------------------ steps
def step_resources(allow_incomplete: bool) -> str:
    _banner("1/5", "Qt resources (tools/build_resources.py)")
    if not RESOURCES_SCRIPT.exists():
        msg = f"{RESOURCES_SCRIPT.relative_to(ROOT)} not found (owned by the python role, section 4)"
        if not allow_incomplete:
            raise BuildError(msg)
        print(f"WARNING: {msg}; skipped (--allow-incomplete)")
        return "skipped"
    _run([sys.executable, RESOURCES_SCRIPT])
    return "ok"


def step_icon(allow_incomplete: bool) -> str:
    _banner("2/5", "Icon (packaging/make_icon.py)")
    out = make_icon.make_icon(make_icon.DEFAULT_SOURCE, make_icon.DEFAULT_OUT, allow_placeholder=allow_incomplete)
    entries = make_icon.ico_entries(out)
    src_state = "app.png" if make_icon.DEFAULT_SOURCE.exists() else "PLACEHOLDER (app.png missing)"
    print(f"icon: {out.relative_to(ROOT)}  sizes={[w for w, _ in entries]}  source={src_state}")
    return "ok" if make_icon.DEFAULT_SOURCE.exists() else "placeholder"


def step_version(allow_incomplete: bool) -> str:
    _banner("3/5", "Version resource (packaging/make_version_info.py)")
    version = make_version_info.read_package_version()
    state = "ok"
    if version is None:
        msg = "glasstranslate/__init__.py has no __version__ (python role adds __version__ = \"0.2.0\", section 6)"
        if not allow_incomplete:
            raise BuildError(msg)
        print(f"WARNING: {msg}; using {FALLBACK_VERSION}")
        version, state = FALLBACK_VERSION, "fallback"
    out = make_version_info.write_version_info(version)
    print(f"version: {version} -> {out.relative_to(ROOT)}")
    return state


def _interesting(line: str) -> bool:
    low = line.lower()
    return (
        line.startswith("[prune]")
        or "warning" in low
        or "error" in low
        or "completed successfully" in low
        or low.startswith("traceback")
    )


def step_pyinstaller(entry: Path, name: str, onedir: bool) -> Path:
    _banner("4/5", f"PyInstaller ({SPEC.relative_to(ROOT)}, {'onedir' if onedir else 'onefile'})")
    for f in (PACKAGING / "glasstranslate.ico", PACKAGING / "version_info.txt", PACKAGING / "glasstranslate.manifest",
              PACKAGING / "gt_bundle.marker", SPEC):
        if not f.exists():
            raise BuildError(f"missing build input {f.relative_to(ROOT)}")
    if not entry.exists():
        raise BuildError(f"entry script not found: {entry}")
    BUILD.mkdir(exist_ok=True)
    # PyInstaller appends the spec name to the workpath; give `--name` variants (probe builds) their own tree so
    # two builds never share intermediates.  The product itself keeps the conventional build/GlassTranslate/.
    workpath = BUILD if name == DEFAULT_NAME else BUILD / name
    log_path = BUILD / f"build-{name}.log"
    env = _child_env({
        "GT_BUILD_ENTRY": str(entry),
        "GT_BUILD_NAME": name,
        "GT_BUILD_ONEDIR": "1" if onedir else "0",
    })
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--log-level", "INFO",
           "--distpath", str(DIST), "--workpath", str(workpath), str(SPEC)]
    print("$ " + " ".join(cmd))
    print(f"(full log: {log_path.relative_to(ROOT)})")
    t0 = time.perf_counter()
    shown = 0
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            if _interesting(line):
                print("  " + line.rstrip())
                shown += 1
        rc = proc.wait()
    elapsed = time.perf_counter() - t0
    if rc != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        print("\n".join("  | " + t for t in tail))
        raise BuildError(f"PyInstaller exited with {rc} after {elapsed:.0f} s (see {log_path})")
    exe = DIST / (f"{name}/{name}.exe" if onedir else f"{name}.exe")
    if not exe.exists():
        raise BuildError(f"PyInstaller reported success but {exe} is missing")
    size_mb = exe.stat().st_size / 1e6
    if onedir:
        total = sum(p.stat().st_size for p in exe.parent.rglob("*") if p.is_file()) / 1e6
        print(f"built {exe.relative_to(ROOT)}: {size_mb:.1f} MB launcher, {total:.1f} MB folder, {elapsed:.0f} s")
    else:
        print(f"built {exe.relative_to(ROOT)}: {size_mb:.1f} MB in {elapsed:.0f} s")
    return exe


def step_smoke(exe: Path, mechanics_only: bool, keep: bool) -> None:
    _banner("5/5", "Smoke test from an external folder (packaging/smoke_test.py)")
    if not exe.exists():
        raise BuildError(f"{exe} not found - build first (or drop --test)")
    cmd: List[str] = [sys.executable, str(SMOKE_SCRIPT), "--exe", str(exe)]
    if mechanics_only:
        cmd.append("--mechanics")
    if keep:
        cmd.append("--keep")
    _run(cmd)


# ------------------------------------------------------------------------------------------- main
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build dist/GlassTranslate.exe and smoke-test it.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--no-test", action="store_true", help="build only (skip step 5)")
    mode.add_argument("--test", action="store_true", help="smoke-test only (skip steps 1-4)")
    p.add_argument("--entry", type=Path, default=DEFAULT_ENTRY,
                   help="entry script to freeze (default run.py; e.g. a probe app to validate the machinery)")
    p.add_argument("--name", default=DEFAULT_NAME, help="exe base name (default GlassTranslate)")
    p.add_argument("--onedir", action="store_true",
                   help="build a folder (COLLECT) instead of a single exe: sub-second start, no per-launch extraction")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="dev: tolerate missing resources script / app.png / __version__ (placeholder icon, "
                        "version 0.0.0) and run the smoke test in --mechanics mode")
    p.add_argument("--keep-temp", action="store_true", help="keep the smoke test's temp folder for inspection")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        # line-buffered so progress shows up live even when the output is redirected to a file
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[union-attr]
    args = parse_args(argv)
    os.environ["PYTHONUTF8"] = "1"
    entry = args.entry if args.entry.is_absolute() else (ROOT / args.entry)
    exe = DIST / (f"{args.name}/{args.name}.exe" if args.onedir else f"{args.name}.exe")
    t_start = time.perf_counter()
    summary: List[str] = []
    try:
        _preflight()
        if not args.test:
            summary.append(f"resources: {step_resources(args.allow_incomplete)}")
            summary.append(f"icon: {step_icon(args.allow_incomplete)}")
            summary.append(f"version: {step_version(args.allow_incomplete)}")
            exe = step_pyinstaller(entry, args.name, args.onedir)
            summary.append(f"exe: {exe.relative_to(ROOT)} ({exe.stat().st_size / 1e6:.1f} MB)")
        if not args.no_test:
            step_smoke(exe, mechanics_only=args.allow_incomplete, keep=args.keep_temp)
            summary.append("smoke test: passed" + (" (mechanics only)" if args.allow_incomplete else ""))
    except BuildError as exc:
        print(f"\nBUILD FAILED: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nBUILD INTERRUPTED", file=sys.stderr)
        return 130
    print("\n=== build summary " + "=" * 56)
    for line in summary:
        print("  " + line)
    print(f"  total: {time.perf_counter() - t_start:.0f} s")
    if not args.test:
        print(f"  output folder: {DIST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
