"""Glass lab - dev harness for the Liquid Glass material (docs/GLASS_DESIGN.md §7, §8).

Loads ``demo/glass_lab/Lab.qml`` from **disk** in a frameless, per-pixel translucent
``QQuickView``.  ``Lab.qml`` imports the component directory
``glasstranslate/ui/qml`` with a relative directory import, so the very same
component files later load from ``qrc:/qml/``.  Fake ``bridge`` and ``appearance``
context objects stand in for the Python bridge and the appearance watcher.

Examples (Git Bash, always ``PYTHONUTF8=1``)::

    .venv/Scripts/python.exe demo/glass_lab.py                         # interactive, live pointer
    .venv/Scripts/python.exe demo/glass_lab.py --mode solid --dark
    .venv/Scripts/python.exe demo/glass_lab.py --pointer 300,200 --grab out.png --autoexit-ms 2500
    .venv/Scripts/python.exe demo/glass_lab.py --qa                    # measurement run (exit 4 on failure)
    .venv/Scripts/python.exe demo/glass_lab.py --qa --qa-set dpr --scale 1.5

Exit codes: 0 ok, 2 QML load error, 4 a QA check failed, 5 watchdog.

The controls agent owns ``glasstranslate/ui/qml/qmldir``.  Until it exists the
lab copies the component directory (and ``demo/glass_lab``) into
``demo/output/dev/glass_lab/stage/`` and adds a ``qmldir`` with the single line
``singleton Theme 1.0 Theme.qml`` there, so the relative import keeps working.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent  # demo/
PROJECT = HERE.parent
QML_DIR = PROJECT / "glasstranslate" / "ui" / "qml"
LAB_DIR = HERE / "glass_lab"
OUT_DIR = HERE / "output" / "dev" / "glass_lab"
STAGE_DIR = OUT_DIR / "stage"
QMLDIR_LINE = "singleton Theme 1.0 Theme.qml\n"

WINDOW_W, WINDOW_H = 832, 640
SLAB = (36, 28, WINDOW_W - 72, WINDOW_H - 80)  # x, y, w, h (logical px)
SLAB_RADIUS = 22.0
SLAB_SQUIRCLE = 4.5
SLAB_RIM = 34.0
BACKDROP_ORIGIN = (-32, -32)

log = logging.getLogger("glass_lab")


# ----------------------------------------------------------------------------- CLI
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("glass", "solid", "hc"), default="glass")
    p.add_argument("--reducemotion", action="store_true")
    theme = p.add_mutually_exclusive_group()
    theme.add_argument("--dark", action="store_true", help="appearance.darkMode = true")
    theme.add_argument("--light", action="store_true", help="appearance.darkMode = false (default)")
    p.add_argument("--luma", type=float, default=None,
                   help="backdrop luma 0..1; derives inkPolarity with the .62/.38 hysteresis rule")
    p.add_argument("--polarity", type=int, choices=(0, 1), default=None, help="force bridge.inkPolarity")
    p.add_argument("--textscale", type=float, default=1.0)
    p.add_argument("--backdrop", type=Path, default=None, help="static backdrop PNG (default: generated test image)")
    p.add_argument("--pointer", default=None, metavar="X,Y", help="fixed pointer position in window px")
    p.add_argument("--pointer-sweep", action="store_true", help="sweep the pointer across the slab")
    p.add_argument("--grab", type=Path, default=None, help="save a window grab (PNG with alpha) + _small.jpg")
    p.add_argument("--grab-ms", type=int, default=1200, help="when to grab (ms after show)")
    p.add_argument("--autoexit-ms", type=int, default=int(os.environ.get("GLASSTRANSLATE_AUTOEXIT_MS", "0") or 0))
    p.add_argument("--scale", type=float, default=None, help="sets QT_SCALE_FACTOR before Qt starts")
    p.add_argument("--no-context", action="store_true", help="expose neither bridge nor appearance (Theme fallbacks)")
    p.add_argument("--qa", action="store_true", help="run the measurement protocol and exit")
    p.add_argument("--qa-set", choices=("full", "dpr"), default="full")
    p.add_argument("--tag", default="", help="suffix for QA evidence files")
    p.add_argument("--position", default="200,120", metavar="X,Y", help="window position on screen")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- staging
def resolve_lab_qml() -> Path:
    """Return the Lab.qml to load: the source tree when the controls-owned qmldir
    exists, else a staged copy with a generated qmldir."""
    if (QML_DIR / "qmldir").exists():
        return LAB_DIR / "Lab.qml"
    log.warning("glasstranslate/ui/qml/qmldir is missing (controls role) - staging a copy with a qmldir under %s",
                STAGE_DIR)
    stage_qml = STAGE_DIR / "glasstranslate" / "ui" / "qml"
    stage_lab = STAGE_DIR / "demo" / "glass_lab"
    for d in (stage_qml, stage_lab):
        if d.exists():
            shutil.rmtree(d)
    shutil.copytree(QML_DIR, stage_qml)
    shutil.copytree(LAB_DIR, stage_lab)
    (stage_qml / "qmldir").write_text(QMLDIR_LINE, encoding="utf-8")
    return stage_lab / "Lab.qml"


# ----------------------------------------------------------------------------- backdrop images
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_test_backdrop(path: Path) -> Path:
    """1200x900 image with sharp text, thin lines, a photo-like gradient and
    high-contrast edges, so refraction and frost are visible everywhere the slab sits."""
    w, h = 1200, 900
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.clip((xs / w) * 0.65 + (ys / h) * 0.35, 0, 1)
    c0 = np.array([18, 34, 78], np.float32)     # deep blue
    c1 = np.array([28, 130, 140], np.float32)   # teal
    c2 = np.array([236, 150, 70], np.float32)   # warm
    seg = np.where(t < 0.55, t / 0.55, (t - 0.55) / 0.45)[..., None]
    grad = np.where(t[..., None] < 0.55, c0 + (c1 - c0) * seg, c1 + (c2 - c1) * seg)
    sun = np.exp(-(((xs - 900) / 220) ** 2 + ((ys - 200) / 180) ** 2))[..., None]
    grad = np.clip(grad + sun * np.array([120, 90, 40], np.float32), 0, 255)
    img = Image.fromarray(grad.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    # sharp text of several sizes (white and black)
    y = 70
    for size, colour in ((48, (255, 255, 255)), (28, (255, 255, 255)), (18, (10, 10, 20)), (13, (255, 255, 255)), (13, (0, 0, 0))):
        d.text((70, y), "Liquid glass refracts what is actually behind the window 0123456789", font=_font(size), fill=colour)
        y += size + 14
    # thin 1-px grid + diagonals
    for gx in range(80, 560, 40):
        d.line([(gx, 320), (gx, 560)], fill=(255, 255, 255), width=1)
    for gy in range(320, 560, 40):
        d.line([(80, gy), (560, gy)], fill=(255, 255, 255), width=1)
    for k in range(0, 200, 25):
        d.line([(600 + k, 320), (800 + k, 560)], fill=(0, 0, 0), width=1)
    # high-contrast edges: checkerboard, bars, a white and a black block
    for cy in range(600, 860, 32):
        for cx in range(70, 400, 32):
            if ((cx // 32) + (cy // 32)) % 2 == 0:
                d.rectangle([cx, cy, cx + 31, cy + 31], fill=(255, 255, 255))
            else:
                d.rectangle([cx, cy, cx + 31, cy + 31], fill=(0, 0, 0))
    for bx in range(430, 700, 12):
        d.rectangle([bx, 600, bx + 5, 860], fill=(255, 255, 255))
        d.rectangle([bx + 6, 600, bx + 11, 860], fill=(0, 0, 0))
    d.rectangle([760, 380, 1150, 560], fill=(255, 255, 255))   # white block (rim over white)
    d.text((780, 400), "White desktop window", font=_font(28), fill=(30, 30, 40))
    d.text((780, 450), "13 px text stays legible under the rim only", font=_font(13), fill=(30, 30, 40))
    d.rectangle([760, 600, 1150, 860], fill=(8, 8, 12))          # dark block
    d.text((780, 620), "Dark desktop window", font=_font(28), fill=(230, 230, 240))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def make_ramp(path: Path) -> Path:
    """Mid-grey with a continuous ramp over backdrop x in [56, 140): the ratio
    R/(R+B) encodes the backdrop x linearly (invariant under the lensing
    brightness), so a rendered row recovers the exact refraction mapping."""
    w, h = 1200, 900
    arr = np.full((h, w, 3), 128, np.uint8)
    xs = np.arange(56, 140)
    r = np.round(200.0 * (xs - 56) / 84.0).astype(np.uint8)
    arr[:, 56:140, 0] = r
    arr[:, 56:140, 1] = 100
    arr[:, 56:140, 2] = 200 - r
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGB").save(path)
    return path


def make_noise(path: Path) -> Path:
    """Gaussian-smoothed RGB noise: |rendered - backdrop| then measures displacement
    everywhere, independent of picture content."""
    from PIL import ImageFilter
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (900, 1200, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(1.5))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def make_flat(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1200, 900), (value, value, value)).save(path)
    return path


def make_white_text(path: Path) -> Path:
    """White page with black text; region x 500..800, y 300..500 (backdrop px) stays clear."""
    img = Image.new("RGB", (1200, 900), (255, 255, 255))
    d = ImageDraw.Draw(img)
    y = 80
    for _ in range(8):
        d.text((80, y), "The quick brown fox jumps over the lazy dog 0123456789", font=_font(13), fill=(0, 0, 0))
        y += 24
    y = 520
    for _ in range(4):
        d.text((80, y), "Menu  File  Edit  View  Window  Help", font=_font(20), fill=(0, 0, 0))
        y += 34
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ----------------------------------------------------------------------------- Qt part
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname).1s %(name)s: %(message)s", stream=sys.stdout)
    if args.scale is not None:
        os.environ["QT_SCALE_FACTOR"] = str(args.scale)
    os.environ.setdefault("QSG_INFO", "1")

    # Qt imports after the environment is final.
    from PySide6.QtCore import (QCoreApplication, QEventLoop, QObject, QPointF, Qt, QTimer, QUrl, Property, Signal,
                                qInstallMessageHandler)
    from PySide6.QtGui import QColor, QGuiApplication, QImage, QSurfaceFormat
    from PySide6.QtQuick import QQuickView

    qt_log: List[str] = []

    def handler(mode, ctx, msg):  # noqa: ANN001 - Qt callback
        line = f"[qt:{ctx.category}] {msg}"
        qt_log.append(line)
        log.info("%s", line)

    qInstallMessageHandler(handler)

    fmt = QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QGuiApplication(sys.argv[:1])

    # ---- fake context objects -------------------------------------------------
    def _prop(name: str, typ, notify):
        def getter(self):
            return self._v[name]

        def setter(self, value):
            if self._v[name] != value:
                self._v[name] = value
                self.changed.emit()

        return Property(typ, getter, setter, notify=notify)

    class LabAppearance(QObject):
        """Stand-in for glasstranslate.ui.glass.appearance.Appearance."""

        changed = Signal()

        def __init__(self, parent: Optional[QObject] = None, **values) -> None:
            super().__init__(parent)
            self._v: Dict[str, object] = {"transparency": True, "composition": True, "reduceMotion": False,
                                          "highContrast": False, "darkMode": False, "accent": QColor("#0078D4"),
                                          "textScale": 1.0}
            self._v.update(values)

        transparency = _prop("transparency", bool, changed)
        composition = _prop("composition", bool, changed)
        reduceMotion = _prop("reduceMotion", bool, changed)
        highContrast = _prop("highContrast", bool, changed)
        darkMode = _prop("darkMode", bool, changed)
        accent = _prop("accent", QColor, changed)
        textScale = _prop("textScale", float, changed)

        def _glass_allowed(self) -> bool:
            return bool(self._v["transparency"] and self._v["composition"] and not self._v["highContrast"])

        glassAllowed = Property(bool, _glass_allowed, notify=changed)

    class LabBridge(QObject):
        """Stand-in for the ControlBridge properties Theme reads."""

        changed = Signal()

        def __init__(self, parent: Optional[QObject] = None, **values) -> None:
            super().__init__(parent)
            self._v: Dict[str, object] = {"inkPolarity": 1, "backdropLuma": 0.5, "backdropSerial": 1,
                                          "backdropOrigin": QPointF(-32, -32), "statusMessage": "Ready",
                                          "running": False}
            self._v.update(values)

        inkPolarity = _prop("inkPolarity", int, changed)
        backdropLuma = _prop("backdropLuma", float, changed)
        backdropSerial = _prop("backdropSerial", int, changed)
        backdropOrigin = _prop("backdropOrigin", QPointF, changed)
        statusMessage = _prop("statusMessage", str, changed)
        running = _prop("running", bool, changed)

    class LabControl(QObject):
        """Everything the harness drives in Lab.qml."""

        changed = Signal()

        def __init__(self, parent: Optional[QObject] = None, **values) -> None:
            super().__init__(parent)
            self._v: Dict[str, object] = {"pointerDriven": False, "pointerX": -5000.0, "pointerY": -5000.0,
                                          "showContent": True, "showSlab": True, "showShadow": True,
                                          "showBackdrop": False, "rawSlab": False, "slabStrength": -1.0,
                                          "slabSpecular": -1.0, "backdropSource": "", "indicatorIndex": 1,
                                          "sliderValue": 0.62, "caption": ""}
            self._v.update(values)

        pointerDriven = _prop("pointerDriven", bool, changed)
        pointerX = _prop("pointerX", float, changed)
        pointerY = _prop("pointerY", float, changed)
        showContent = _prop("showContent", bool, changed)
        showSlab = _prop("showSlab", bool, changed)
        showShadow = _prop("showShadow", bool, changed)
        showBackdrop = _prop("showBackdrop", bool, changed)
        rawSlab = _prop("rawSlab", bool, changed)
        slabStrength = _prop("slabStrength", float, changed)
        slabSpecular = _prop("slabSpecular", float, changed)
        backdropSource = _prop("backdropSource", str, changed)
        indicatorIndex = _prop("indicatorIndex", int, changed)
        sliderValue = _prop("sliderValue", float, changed)
        caption = _prop("caption", str, changed)

    dark = bool(args.dark)
    appearance = LabAppearance(transparency=args.mode != "solid", highContrast=args.mode == "hc",
                               reduceMotion=args.reducemotion, darkMode=dark, textScale=args.textscale)
    polarity = 0 if dark else 1
    if args.luma is not None:
        polarity = 1 if args.luma > 0.62 else 0 if args.luma < 0.38 else polarity
    if args.polarity is not None:
        polarity = args.polarity
    bridge = LabBridge(inkPolarity=polarity, backdropLuma=args.luma if args.luma is not None else 0.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backdrop_png = args.backdrop.resolve() if args.backdrop else make_test_backdrop(OUT_DIR / "backdrop_test.png")
    lab = LabControl(backdropSource=QUrl.fromLocalFile(str(backdrop_png)).toString())
    if args.pointer:
        px, py = (float(v) for v in args.pointer.split(","))
        lab.pointerDriven = True
        lab.pointerX = px
        lab.pointerY = py

    # ---- view -----------------------------------------------------------------
    view = QQuickView()
    view.setFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
    view.setColor(Qt.GlobalColor.transparent)
    view.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
    ctx = view.rootContext()
    if not args.no_context:
        ctx.setContextProperty("appearance", appearance)
        ctx.setContextProperty("bridge", bridge)
    ctx.setContextProperty("lab", lab)
    view.resize(WINDOW_W, WINDOW_H)
    wx, wy = (int(v) for v in args.position.split(","))
    view.setPosition(wx, wy)
    view.setSource(QUrl.fromLocalFile(str(resolve_lab_qml())))
    if view.status() == QQuickView.Status.Error or view.errors():
        for e in view.errors():
            log.error("QML: %s", e.toString())
        return 2
    root = view.rootObject()

    frames = {"n": 0, "ready": None}

    def on_frame() -> None:
        frames["n"] += 1
        if frames["n"] == 3:  # scene up: start the steady-state CPU window here
            frames["ready"] = (time.monotonic(), time.process_time(), frames["n"])

    view.frameSwapped.connect(on_frame)
    view.show()

    def settle(min_frames: int = 3, timeout_ms: int = 2500, extra_ms: int = 0) -> None:
        """Pump the event loop until `min_frames` new frames were swapped (and the
        backdrop image is ready), then optionally wait `extra_ms` more (animations)."""
        target = frames["n"] + min_frames
        t0 = time.monotonic()
        while (frames["n"] < target or not root.property("backdropReady")) and (time.monotonic() - t0) * 1000 < timeout_ms:
            view.update()
            QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 15)
        if extra_ms:
            t1 = time.monotonic()
            while (time.monotonic() - t1) * 1000 < extra_ms:
                QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 15)
            settle(2, timeout_ms)

    def grab() -> np.ndarray:
        img = view.grabWindow().convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = img.width(), img.height()
        arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w, :]
        return arr.copy()

    def save_small(arr: np.ndarray, path: Path, max_side: int = 1000) -> Path:
        rgb = arr[..., :3] if arr.shape[2] >= 3 else arr
        im = Image.fromarray(np.ascontiguousarray(np.clip(rgb, 0, 255)).astype(np.uint8), "RGB")
        s = max_side / max(im.size)
        if s < 1:
            im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        im.save(path, quality=80)
        return path

    def save_zoom(arr: np.ndarray, box: Tuple[int, int, int, int], factor: int, path: Path) -> Path:
        x0, y0, x1, y1 = box
        crop = np.ascontiguousarray(arr[y0:y1, x0:x1, :3])
        im = Image.fromarray(crop, "RGB").resize((crop.shape[1] * factor, crop.shape[0] * factor), Image.NEAREST)
        im.save(path, quality=80)
        return path

    def do_grab(path: Path) -> None:
        arr = grab()
        Image.fromarray(arr, "RGBA").save(path)
        save_small(arr, path.with_name(path.stem + "_small.jpg"))
        log.info("grab saved: %s (%dx%d) Theme.mode=%s context=%s ink=%.2f", path, arr.shape[1], arr.shape[0],
                 root.property("themeMode"), root.property("themeHasContext"), root.property("themeInk"))

    # ---- pointer sweep ---------------------------------------------------------
    if args.pointer_sweep:
        lab.pointerDriven = True
        sweep = {"t0": time.monotonic()}

        def tick() -> None:
            t = (time.monotonic() - sweep["t0"]) % 4.0
            u = t / 2.0 if t < 2.0 else (4.0 - t) / 2.0
            lab.pointerX = SLAB[0] + u * SLAB[2]
            lab.pointerY = SLAB[1] + 0.5 * SLAB[3] * (0.5 + 0.5 * math.sin(u * math.pi * 2))

        sweep_timer = QTimer(view)
        sweep_timer.timeout.connect(tick)
        sweep_timer.start(16)

    if args.grab:
        QTimer.singleShot(args.grab_ms, lambda: do_grab(args.grab.resolve()))
    if args.autoexit_ms > 0:
        QTimer.singleShot(args.autoexit_ms, app.quit)

    # ---- QA protocol -------------------------------------------------------------
    results: List[Tuple[bool, str]] = []
    tag = args.tag or (f"dpr{args.scale}" if args.scale else "1x")

    def check(cond: bool, msg: str) -> None:
        results.append((bool(cond), msg))
        log.info("%s %s", "PASS" if cond else "FAIL", msg)

    def sd_slab(shape: Tuple[int, int], dpr: float) -> np.ndarray:
        """Squircle SDF of the slab in device px (negative inside), same formula as glass.frag."""
        ys, xs = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64) + 0.5
        x, y, w, h = (v * dpr for v in SLAB)
        r, n = SLAB_RADIUS * dpr, SLAB_SQUIRCLE
        cx, cy, hw, hh = x + w / 2, y + h / 2, w / 2, h / 2
        qx, qy = np.abs(xs - cx) - (hw - r), np.abs(ys - cy) - (hh - r)
        corner = (np.maximum(qx, 1e-4) ** n + np.maximum(qy, 1e-4) ** n) ** (1 / n) - r
        return corner + np.minimum(np.maximum(qx, qy), 0)

    def set_backdrop(path: Path) -> None:
        lab.backdropSource = QUrl.fromLocalFile(str(path)).toString()
        settle(3)

    def run_qa() -> None:
        dpr = float(view.devicePixelRatio())
        api = view.rendererInterface().graphicsApi()
        log.info("QA start: mode=%s dpr=%.2f rhi=%s context=%s", root.property("themeMode"), dpr, api,
                 root.property("themeHasContext"))
        check(root.property("themeMode") == args.mode, f"Theme.mode == {args.mode!r} (got {root.property('themeMode')!r})")
        settle(4)
        D = sd_slab(grab().shape[:2], dpr)
        ipx = lambda v: int(round(v * dpr))  # noqa: E731

        # -- hero shot + zooms (evidence)
        lab.pointerDriven, lab.pointerX, lab.pointerY = True, 300.0, 200.0
        settle(3)
        hero = grab()
        save_small(hero, OUT_DIR / f"hero_{tag}.jpg")
        save_zoom(hero, (ipx(SLAB[0] - 10), ipx(SLAB[1] - 10), ipx(SLAB[0] + 140), ipx(SLAB[1] + 110)), 3,
                  OUT_DIR / f"corner_zoom_{tag}.jpg")
        tr = root.property("trackRect")
        save_zoom(hero, (ipx(tr.x() - 8), ipx(tr.y() - 8), ipx(tr.x() + tr.width() + 8), ipx(tr.y() + tr.height() + 8)), 2,
                  OUT_DIR / f"capsule_zoom_{tag}.jpg")
        # corners transparent (the window is our own shape)
        a = hero[..., 3]
        corners = [int(a[0, 0]), int(a[0, -1]), int(a[-1, 0]), int(a[-1, -1])]
        check(max(corners) == 0, f"window corner alpha == 0: {corners}")
        # concentric capsule / pill: indicator radius = track radius - inset, both circular (squircle 2)
        tr_r, ind_r = float(root.property("trackRadius")), float(root.property("indicatorRadius"))
        check(abs(tr_r - tr.height() / 2) < 1e-6 and abs(ind_r - (tr_r - 3)) < 1e-6 and abs(ind_r - (tr.height() - 6) / 2) < 1e-6,
              f"concentric: track radius {tr_r:.1f} (= h/2 of {tr.height():.0f}), indicator radius {ind_r:.1f} = track - inset 3 = its own h/2")

        # -- refraction: rim vs centre, alignment (raw slab over smoothed noise, no content, no shadow)
        noise = make_noise(OUT_DIR / "backdrop_noise.png")
        set_backdrop(noise)
        lab.showContent, lab.showShadow = False, False
        lab.pointerX, lab.pointerY = -5000.0, -5000.0
        lab.showSlab, lab.showBackdrop = False, True
        settle(3)
        ref = grab()[..., :3].astype(np.int32)
        lab.showBackdrop, lab.showSlab, lab.rawSlab = False, True, True
        lab.slabSpecular = 0.0
        settle(3)
        raw = grab()[..., :3].astype(np.int32)
        diff = np.abs(raw - ref).mean(axis=2)
        centre = D < -60 * dpr
        rim = (D > -(SLAB_RIM - 3) * dpr) & (D < -3 * dpr)
        c_diff, r_diff = float(diff[centre].mean()), float(diff[rim].mean())
        check(c_diff < 1.5, f"refraction: centre mean|rendered-backdrop| = {c_diff:.3f} (n={int(centre.sum())})")
        check(r_diff > 8 and r_diff > 20 * max(c_diff, 0.2), f"refraction: rim band mean|rendered-backdrop| = {r_diff:.2f} (n={int(rim.sum())})")
        # displacement profile across the rim: mean|diff| in 4-px bins from the edge inward
        bins = []
        for k in range(0, 40, 4):
            m = (D > -(k + 4) * dpr) & (D <= -k * dpr)
            bins.append(float(diff[m].mean()))
        log.info("refraction profile (mean|diff| per 4-px band from the edge inward): %s", " ".join(f"{b:.1f}" for b in bins))
        non_increasing = all(bins[i + 1] <= bins[i] + 1.0 for i in range(len(bins) - 1))
        check(non_increasing and bins[0] > 12 and bins[-1] < 1.5,
              f"refraction falls off from {bins[0]:.1f} at the edge to {bins[-2]:.1f} at 32-36 px and {bins[-1]:.1f} beyond the rim width")
        if args.qa_set == "dpr":
            save_small(raw, OUT_DIR / f"raw_{tag}.jpg")

        lab.slabStrength = 0.0
        settle(3)
        raw0 = grab()[..., :3].astype(np.int32)
        x0, y0 = ipx(SLAB[0] + 120), ipx(SLAB[1] + 120)
        x1, y1 = ipx(SLAB[0] + SLAB[2] - 120), ipx(SLAB[1] + SLAB[3] - 120)
        best, zero = None, None
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                m = float(np.abs(raw0[y0:y1, x0:x1] - ref[y0 + dy:y1 + dy, x0 + dx:x1 + dx]).mean())
                if best is None or m < best[0]:
                    best = (m, dx, dy)
                if dx == 0 and dy == 0:
                    zero = m
        check(zero is not None and zero < 1.0 and best[1] == 0 and best[2] == 0,
              f"rect alignment: |diff| at shift 0 = {zero:.3f}, best shift = ({best[1]},{best[2]}) diff {best[0]:.3f} (device px)")
        lab.slabStrength = -1.0

        # -- edge line: 1 device px (flat backdrop, ambient only)
        flat64 = make_flat(OUT_DIR / "backdrop_flat64.png", 64)
        set_backdrop(flat64)
        lab.slabSpecular = 1.0
        settle(3)
        spec = grab()[..., :3].astype(np.float64).mean(axis=2)
        row = ipx(SLAB[1] + SLAB[3] / 2)
        ex = ipx(SLAB[0])
        prof = spec[row, ex - 4:ex + 14] - 64.0
        peak = prof.max()
        above = int((prof >= 0.5 * peak).sum())
        peak_at = int(np.argmax(prof)) - 4
        log.info("edge-line profile (left edge, row %d): %s", row, " ".join(f"{v:.0f}" for v in prof))
        check(peak > 25 and above == 1 and peak_at == 0,
              f"edge line: FWHM = {above} device px, peak +{peak:.0f} at offset {peak_at} px inside the edge (dpr {dpr:.2f})")
        col = ipx(SLAB[0] + SLAB[2] / 2)
        ey = ipx(SLAB[1])
        profv = spec[ey - 4:ey + 14, col] - 64.0
        abovev = int((profv >= 0.5 * profv.max()).sum())
        check(abovev == 1, f"edge line (top edge): FWHM = {abovev} device px, peak +{profv.max():.0f}")

        if args.qa_set == "full":
            # -- highlight follows the pointer, fades to ambient
            lab.pointerX, lab.pointerY = float(SLAB[0] + 80), float(SLAB[1] + 60)
            settle(3)
            near_img = grab()[..., :3].astype(np.float64).mean(axis=2)
            line_row = ipx(SLAB[1])
            nx = slice(ipx(SLAB[0] + 40), ipx(SLAB[0] + 120))
            fx = slice(ipx(SLAB[0] + 560), ipx(SLAB[0] + 700))
            near_line, far_line = near_img[line_row, nx].mean() - 64, near_img[line_row, fx].mean() - 64
            band = (D > -30 * dpr) & (D < -3 * dpr)
            band_near = band.copy(); band_near[:, :ipx(SLAB[0] + 30)] = False; band_near[:, ipx(SLAB[0] + 130):] = False; band_near[ipx(SLAB[1] + 60):, :] = False
            band_far = band.copy(); band_far[:, :ipx(SLAB[0] + 540)] = False; band_far[ipx(SLAB[1] + 60):, :] = False
            glow_near, glow_far = near_img[band_near].mean() - 64, near_img[band_far].mean() - 64
            lab.pointerX, lab.pointerY = -5000.0, -5000.0
            settle(3)
            rest_img = grab()[..., :3].astype(np.float64).mean(axis=2)
            rest_line = rest_img[line_row, nx].mean() - 64
            rest_glow = rest_img[band_near].mean() - 64
            check(near_line > far_line + 25 and near_line > rest_line + 25 and abs(rest_line - far_line) < 8,
                  f"specular follows pointer: top-edge line +{near_line:.0f} near the pointer vs +{far_line:.0f} far, "
                  f"+{rest_line:.0f} after the pointer left (ambient)")
            check(glow_near > glow_far + 3 and glow_near > rest_glow + 3,
                  f"rim glow near pointer +{glow_near:.1f} vs far +{glow_far:.1f} vs at rest +{rest_glow:.1f}")

            # -- exact rim mapping on a continuous ramp: recover the sampled backdrop x per
            #    rendered pixel along the left rim and compare with the analytic profile
            ramp = make_ramp(OUT_DIR / "backdrop_ramp.png")
            set_backdrop(ramp)
            lab.slabSpecular = 0.0

            def ramp_mapping(label: str, strength: float) -> None:
                img = grab()[..., :3].astype(np.float64)
                n = ipx(SLAB_RIM)
                seg = img[row, ex:ex + n]
                rho = seg[:, 0] / np.maximum(seg[:, 0] + seg[:, 2], 1.0)          # (bx - 56) / 84 in backdrop px
                measured = 56.0 + 84.0 * rho                                         # sampled backdrop x
                xin = (np.arange(n) + 0.5) / dpr                                     # px from the edge (logical)
                disp = strength * SLAB_RIM * np.clip(1.0 - xin / SLAB_RIM, 0, 1) ** 2.0
                analytic = SLAB[0] + xin - BACKDROP_ORIGIN[0] + disp
                err = np.abs(measured - analytic)
                drops = np.minimum(np.diff(measured), 0)
                worst = float(-drops.min()) if drops.size else 0.0
                slope = float((measured[ipx(2)] - measured[0]) / (xin[ipx(2)] - xin[0])) if n > 2 else 1.0
                log.info("rim mapping (%s): measured bx = %s", label, " ".join(f"{v:.1f}" for v in measured[:12]))
                check(err.max() < 1.0 and worst <= 0.15,
                      f"rim mapping ({label}): max |measured - analytic| = {err.max():.2f} px over {n} px, "
                      f"edge slope {slope:.2f} (analytic {1 - strength * 2:.2f}; < 0 would be a fold), "
                      f"worst order reversal {worst:.2f} px, peak displacement {disp[0]:.1f} px")

            settle(3)
            ramp_mapping("strength 0.38", float(root.property("slabStrengthEffective")))
            n_warn = len([m for m in qt_log if "folds the rim" in m])
            lab.slabStrength = 0.6
            settle(3)
            eff = float(root.property("slabStrengthEffective"))
            n_warn2 = len([m for m in qt_log if "folds the rim" in m])
            check(abs(eff - 0.45) < 1e-6 and n_warn2 > n_warn,
                  f"strength .6 * falloff 2 clamped to {eff:.3f} with console.warn ({n_warn2 - n_warn} warning(s))")
            ramp_mapping("strength .6 requested", eff)
            lab.slabStrength = -1.0

            # -- material lift adapts to polarity (flat mid grey, full material)
            flat128 = make_flat(OUT_DIR / "backdrop_flat128.png", 128)
            set_backdrop(flat128)
            lab.rawSlab, lab.slabSpecular = False, 0.0
            bridge.inkPolarity = 1
            settle(3, extra_ms=300)
            light = grab()[..., :3].astype(np.float64)
            bridge.inkPolarity = 0
            settle(3, extra_ms=300)
            darkm = grab()[..., :3].astype(np.float64)
            cm = D < -80 * dpr
            cm[ipx(SLAB[1] + SLAB[3] * 0.6):, :] = False  # stay above the inner-shadow zone
            lm, dm = light[cm].mean(axis=0), darkm[cm].mean(axis=0)
            check(lm.mean() > 140 and dm.mean() < 110 and lm[2] > lm[0] and dm[2] > dm[0],
                  f"material lift: mid-grey 128 renders {tuple(int(v) for v in lm)} in light polarity, "
                  f"{tuple(int(v) for v in dm)} in dark polarity (spec ~163 / ~97 before the cool lean)")
            bridge.inkPolarity = 0 if dark else 1

            # -- no white wash over a white backdrop
            white = make_white_text(OUT_DIR / "backdrop_white.png")
            set_backdrop(white)
            bridge.inkPolarity = 1
            settle(3, extra_ms=300)
            wimg = grab()[..., :3].astype(np.float64)
            clear = wimg[ipx(300 - 32):ipx(500 - 32), ipx(500 - 32):ipx(800 - 32)]
            wm = clear.mean(axis=(0, 1))
            textreg = wimg[ipx(90 - 32):ipx(260 - 32), ipx(100 - 32):ipx(600 - 32)]
            check(wm[0] >= 232 and wm[2] >= 245 and wm.max() <= 255 and textreg.std() > 2,
                  f"no white wash: white backdrop renders {tuple(int(v) for v in wm)} (>= 232/245); "
                  f"frosted text region std {textreg.std():.1f}")
            bridge.inkPolarity = 0 if dark else 1

            # -- shadows soft (default backdrop, slab + shadow, no content)
            set_backdrop(backdrop_png)
            lab.showShadow, lab.slabSpecular = True, -1.0
            settle(3, extra_ms=250)
            sh = grab()
            alpha = sh[..., 3].astype(np.int32)
            colx = ipx(SLAB[0] + SLAB[2] / 2)
            ybot = ipx(SLAB[1] + SLAB[3])
            prof_a = alpha[ybot:, colx]
            steps = np.diff(prof_a)
            log.info("shadow alpha below the slab (centre column): %s", " ".join(str(int(v)) for v in prof_a[::3]))
            check(30 <= prof_a[1] <= 100 and prof_a[-1] <= 4 and steps.max() <= 2 and (-steps).max() <= 6,
                  f"shadow soft: alpha {int(prof_a[1])} just below the slab -> {int(prof_a[-1])} at the window edge, "
                  f"max step {int((-steps).max())}/px over {len(prof_a)} px")
            check(int(alpha[2, colx]) <= 3 and int(alpha[ipx(SLAB[1] + SLAB[3] / 2), 2]) <= 3,
                  f"shadow stays inside the margins: alpha top {int(alpha[2, colx])}, left {int(alpha[ipx(SLAB[1] + SLAB[3] / 2), 2])}")

            # -- solid and HC modes (pixel checks with the content hidden, evidence shot with it)
            lab.showContent = False
            appearance.transparency = False
            settle(3, extra_ms=250)
            solid = grab()
            fill = root.property("themeFill")
            interior = D < -40 * dpr
            s_rgb = solid[..., :3][interior].astype(np.float64)
            exp_fill = np.array([fill.red(), fill.green(), fill.blue()], np.float64)
            check(np.abs(s_rgb.mean(axis=0) - exp_fill).max() <= 2 and s_rgb.std(axis=0).max() < 1.5
                  and solid[..., 3][interior].min() == 255 and int(solid[ybot + 4, colx, 3]) > 10,
                  f"solid mode: opaque flat fill {tuple(int(v) for v in s_rgb.mean(axis=0))} == Theme.fill "
                  f"{tuple(int(v) for v in exp_fill)}, std {s_rgb.std(axis=0).max():.2f}, "
                  f"shadow kept (alpha below {int(solid[ybot + 4, colx, 3])})")
            lab.showContent = True
            settle(3)
            solid_shot = grab()
            lab.showContent = False
            appearance.transparency = True
            appearance.highContrast = True
            settle(3, extra_ms=250)
            hc = grab()
            border = root.property("themeBorder")
            exp_border = np.array([border.red(), border.green(), border.blue()], np.float64)
            # the 2-px border is 1 - smoothstep(border - aa, border + aa, -d): pixels 1.5 px inside
            # the edge carry ~93 % border colour on flat edges (the outer row is the AA'd edge)
            ring = (D > -1.8 * dpr) & (D < -1.2 * dpr)
            h_ring = hc[..., :3][ring].astype(np.float64)
            hfill = root.property("themeFill")
            exp_hfill = np.array([hfill.red(), hfill.green(), hfill.blue()], np.float64)
            h_in = hc[..., :3][interior].astype(np.float64)
            ring_mean = h_ring.mean(axis=0)
            toward_border = 1.0 - np.abs(ring_mean - exp_border).max() / max(np.abs(exp_border - exp_hfill).max(), 1.0)
            outer = hc[..., :3][(D > -0.8 * dpr) & (D < -0.2 * dpr)].astype(np.float64).mean(axis=0)
            check(toward_border >= 0.85 and h_ring.std(axis=0).max() < 10
                  and int(hc[ybot + 4, colx, 3]) == 0 and np.abs(h_in.mean(axis=0) - exp_hfill).max() <= 2,
                  f"hc mode: border ring {tuple(int(v) for v in ring_mean)} is {100 * toward_border:.0f} % palette.windowText "
                  f"{tuple(int(v) for v in exp_border)} (outer AA row {tuple(int(v) for v in outer)}, no white specular line), "
                  f"no shadow (alpha {int(hc[ybot + 4, colx, 3])}), fill == palette.window {tuple(int(v) for v in exp_hfill)}")
            lab.showContent = True
            settle(3)
            hc_shot = grab()
            appearance.highContrast = args.mode == "hc"
            appearance.transparency = args.mode != "solid"
            side = np.concatenate([solid_shot[..., :3], hc_shot[..., :3]], axis=1)
            save_small(side, OUT_DIR / f"modes_{tag}.jpg")

        # restore + summary
        lab.rawSlab, lab.showContent, lab.showShadow, lab.slabSpecular = False, True, True, -1.0
        n_fail = sum(1 for ok, _ in results if not ok)
        summary = OUT_DIR / f"qa_{tag}.txt"
        with summary.open("w", encoding="utf-8") as fh:
            fh.write(f"glass_lab QA {tag}: mode={args.mode} dpr={dpr:.2f} rhi={api}\n")
            for ok, msg in results:
                fh.write(("PASS " if ok else "FAIL ") + msg + "\n")
        log.info("QA summary: %d pass, %d fail -> %s", len(results) - n_fail, n_fail, summary)
        app.exit(0 if n_fail == 0 else 4)

    if args.qa:
        QTimer.singleShot(700, run_qa)
        QTimer.singleShot(120000, lambda: (log.error("watchdog"), app.exit(5)))

    # Tear the scene down while the context objects are still alive (otherwise
    # every `lab.*` binding logs a TypeError during interpreter shutdown).
    t_start, cpu_start = time.monotonic(), time.process_time()

    def on_quit() -> None:
        wall = time.monotonic() - t_start
        cpu = time.process_time() - cpu_start
        log.info("run: %d frames in %.1f s (%.1f fps), process CPU %.1f %% of one core incl. startup%s", frames["n"], wall,
                 frames["n"] / max(wall, 1e-6), 100 * cpu / max(wall, 1e-6), " (pointer sweep)" if args.pointer_sweep else "")
        if frames["ready"] is not None:
            t_r, c_r, n_r = frames["ready"]
            w2, c2 = time.monotonic() - t_r, time.process_time() - c_r
            log.info("steady state (after frame 3): %d frames in %.1f s (%.1f fps), process CPU %.1f %% of one core",
                     frames["n"] - n_r, w2, (frames["n"] - n_r) / max(w2, 1e-6), 100 * c2 / max(w2, 1e-6))
        view.setSource(QUrl())

    app.aboutToQuit.connect(on_quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
