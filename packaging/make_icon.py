"""Derive the multi-size Windows icon ``packaging/glasstranslate.ico`` from the committed app icon.

Build step (2) of ``build.py``.  The source of truth is the 256 px
``glasstranslate/ui/icons/app.png`` (owned by the python role, also compiled into the Qt
resources as ``:/icons/app.png``); this script only *derives* the ``.ico`` and never writes into
the qrc inputs (docs/GLASS_DESIGN.md section 4).  Seven sizes are embedded (16, 24, 32, 48, 64,
128, 256) - PyInstaller copies every image of the ico into one RT_ICON each plus one RT_GROUP_ICON
(verified in demo/output/probes/packaging/app/make_ico.py).

Usage::

    python packaging/make_icon.py                       # app.png -> packaging/glasstranslate.ico
    python packaging/make_icon.py --placeholder         # draw a stand-in when app.png is missing
    python packaging/make_icon.py --source x.png --out y.ico
"""
from __future__ import annotations

import argparse
import logging
import struct
import sys
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

PACKAGING_DIR = Path(__file__).resolve().parent
ROOT = PACKAGING_DIR.parent
DEFAULT_SOURCE = ROOT / "glasstranslate" / "ui" / "icons" / "app.png"
DEFAULT_OUT = PACKAGING_DIR / "glasstranslate.ico"
ICO_SIZES: Tuple[Tuple[int, int], ...] = ((16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256))


def _placeholder_image(size: int = 256):
    """A cool-tinted glass squircle stand-in used only when the real ``app.png`` is unavailable."""
    from PIL import Image, ImageDraw, ImageFilter

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((20, 28, size - 20, size - 12), radius=56, fill=(0, 0, 0, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    canvas.alpha_composite(shadow)
    body = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(body)
    draw.rounded_rectangle((16, 16, size - 16, size - 16), radius=56, fill=(58, 134, 255, 235))
    draw.rounded_rectangle((30, 30, size - 30, size // 2), radius=44, fill=(160, 210, 255, 90))
    draw.ellipse((size * 0.30, size * 0.30, size * 0.70, size * 0.70), fill=(235, 246, 255, 255))
    draw.arc((size * 0.22, size * 0.22, size * 0.78, size * 0.78), start=200, end=330, fill=(255, 255, 255, 255), width=6)
    canvas.alpha_composite(body)
    return canvas


def ico_entries(path: Path) -> List[Tuple[int, int]]:
    """Return the ``(width, height)`` of every image in an ``.ico`` (0 in the header means 256)."""
    data = path.read_bytes()
    if len(data) < 6 or struct.unpack("<HH", data[:4]) != (0, 1):
        raise ValueError(f"{path} is not an ICO file")
    count = struct.unpack("<H", data[4:6])[0]
    out: List[Tuple[int, int]] = []
    for i in range(count):
        w, h = data[6 + 16 * i], data[7 + 16 * i]
        out.append((w or 256, h or 256))
    return out


def make_icon(source: Optional[Path], out: Path = DEFAULT_OUT, allow_placeholder: bool = False) -> Path:
    """Write the 7-size ``.ico`` at *out* from *source* (256 px PNG) and return *out*.

    Raises ``FileNotFoundError`` when *source* is missing unless *allow_placeholder* is true, in
    which case a stand-in is rendered in memory (nothing is written next to the source).
    """
    from PIL import Image

    if source is not None and source.exists():
        im = Image.open(source).convert("RGBA")
        if im.size != (256, 256):
            log.warning("icon source %s is %sx%s; resampling to 256x256", source, *im.size)
            im = im.resize((256, 256), Image.Resampling.LANCZOS)
    elif allow_placeholder:
        log.warning("icon source %s missing; using an in-memory placeholder icon", source)
        im = _placeholder_image()
    else:
        raise FileNotFoundError(f"icon source not found: {source}")
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, format="ICO", sizes=list(ICO_SIZES))
    entries = ico_entries(out)
    if len(entries) != len(ICO_SIZES):
        raise RuntimeError(f"{out}: expected {len(ICO_SIZES)} icon sizes, got {entries}")
    log.info("icon: %s -> %s (%d sizes)", source, out, len(entries))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="256 px RGBA PNG")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output .ico")
    parser.add_argument("--placeholder", action="store_true", help="render a stand-in if --source is missing")
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    try:
        out = make_icon(args.source, args.out, allow_placeholder=args.placeholder)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        log.error("%s", exc)
        return 1
    print(f"icon: {out} entries={ico_entries(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
