"""Generate synthetic "screenshots" for the GlassTranslate demo.

Four PNGs are written to ``demo/images`` (override with ``--out``):

* ``webpage.png``  - white-on-blue header bar, black-on-white body text in two
  sizes and a small grey caption.
* ``rotated.png``  - lines at 0, +20, -15 degrees and one 90-degree vertical
  line, in different colours on a light and a dark background.
* ``dark_ui.png``  - light text on a dark panel with a coloured accent button.
* ``manga.png``    - a manga panel: vertical Japanese in speech bubbles (one and
  two columns), a horizontal bubble, white and red text on a dark panel.

The first three are plain English so they OCR reliably and translate with any
``en -> xx`` model in ``models/``; the manga panel exercises ``ja -> en`` and
vertical-text layout.  Only real Windows TrueType fonts are used; the script
falls back to PIL's default font if one is missing.

Usage::

    .venv\\Scripts\\python demo\\make_images.py [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageFont

RGB = Tuple[int, int, int]

FONTS: Dict[str, str] = {
    "arial": "C:/Windows/Fonts/arial.ttf",
    "arial_bold": "C:/Windows/Fonts/arialbd.ttf",
    "segoe": "C:/Windows/Fonts/segoeui.ttf",
    "msgothic": "C:/Windows/Fonts/msgothic.ttc",  # Japanese, manga-like
    "yugothic_bold": "C:/Windows/Fonts/YuGothB.ttc",
}

DEFAULT_OUT = Path(__file__).resolve().parent / "images"


def font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a named TrueType font at ``size`` px, falling back to PIL's default."""
    try:
        return ImageFont.truetype(FONTS[name], size)
    except OSError:
        return ImageFont.load_default(size=size)


def draw_rotated_text(
    canvas: Image.Image,
    center: Tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: RGB,
    angle_deg: float,
) -> None:
    """Draw ``text`` rotated counter-clockwise (as seen on screen) by
    ``angle_deg`` and centred at ``center``.

    The text is rendered on a transparent layer, rotated with
    ``Image.rotate`` (which is CCW on screen as well) and alpha-composited.
    """
    left, top, right, bottom = fnt.getbbox(text)
    pad = 8
    layer = Image.new("RGBA", (right - left + 2 * pad, bottom - top + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((pad - left, pad - top), text, font=fnt, fill=(*fill, 255))
    rotated = layer.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
    x = center[0] - rotated.width // 2
    y = center[1] - rotated.height // 2
    canvas.alpha_composite(rotated, dest=(x, y))


def make_webpage() -> Image.Image:
    """White-on-blue header, black-on-white body in two sizes, grey caption."""
    img = Image.new("RGB", (1000, 700), (255, 255, 255))
    d = ImageDraw.Draw(img)

    header_bg: RGB = (24, 72, 176)
    d.rectangle((0, 0, 1000, 120), fill=header_bg)
    d.text((40, 30), "Welcome to the weather service", font=font("segoe", 46), fill=(255, 255, 255))

    d.text((40, 170), "Today will be sunny with a light breeze.", font=font("arial", 32), fill=(0, 0, 0))

    body = font("arial", 22)
    d.text((40, 250), "The temperature will reach twenty degrees this afternoon.", font=body, fill=(0, 0, 0))
    d.text((40, 295), "Please bring an umbrella tomorrow because rain is expected.", font=body, fill=(0, 0, 0))
    d.text((40, 340), "The wind will come from the west during the evening.", font=body, fill=(0, 0, 0))

    d.text((40, 640), "Last updated two hours ago", font=font("arial", 16), fill=(120, 120, 120))
    return img


def make_rotated() -> Image.Image:
    """Lines at 0, +20, -15 and 90 degrees on a light (top) and dark (bottom) half."""
    img = Image.new("RGBA", (1000, 700), (245, 245, 245, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 350, 1000, 700), fill=(28, 30, 36, 255))

    # Light half: dark red at 0 degrees, blue at +20 degrees.
    d.text((40, 40), "The door is open", font=font("arial_bold", 40), fill=(170, 30, 30, 255))
    draw_rotated_text(img, (520, 220), "Please close the window", font("arial", 36), (30, 80, 200), 20.0)

    # Dark half: yellow at -15 degrees, green vertical (90 degrees) on the right.
    draw_rotated_text(img, (400, 520), "The train leaves at noon", font("arial", 36), (255, 200, 40), -15.0)
    draw_rotated_text(img, (900, 525), "Open door", font("arial_bold", 34), (80, 220, 120), 90.0)
    return img.convert("RGB")


def make_dark_ui() -> Image.Image:
    """Light text on a dark panel plus an orange accent button with a white label."""
    img = Image.new("RGB", (900, 600), (18, 19, 24))
    d = ImageDraw.Draw(img)

    panel: RGB = (32, 34, 42)
    d.rounded_rectangle((60, 50, 840, 550), radius=18, fill=panel)

    d.text((100, 80), "Settings", font=font("segoe", 40), fill=(235, 235, 240))
    d.text((100, 170), "Enable notifications for new messages.", font=font("segoe", 26), fill=(220, 220, 225))
    d.text((100, 225), "Save your changes before you leave this page.", font=font("segoe", 22), fill=(200, 200, 205))
    d.text((100, 275), "Your account is connected.", font=font("segoe", 22), fill=(90, 210, 130))

    d.rounded_rectangle((100, 420, 420, 490), radius=12, fill=(240, 130, 20))
    d.text((140, 437), "Download now", font=font("segoe", 28), fill=(255, 255, 255))
    return img


def _vertical_text(d: ImageDraw.ImageDraw, x: int, y: int, text: str, fnt, fill: RGB) -> None:
    """Draw ``text`` top-to-bottom, one glyph per row, as manga lettering is."""
    for ch in text:
        d.text((x, y), ch, font=fnt, fill=fill)
        y += fnt.size + 2


def make_manga() -> Image.Image:
    """A manga panel: bordered frame, three speech bubbles with vertical and
    horizontal Japanese, and a dark lower panel with white and red lettering."""
    w, h = 900, 700
    img = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.rectangle((10, 10, w - 10, h - 10), outline=(0, 0, 0), width=4)
    d.rectangle((20, 400, 880, 690), fill=(40, 40, 40))

    def bubble(box: Tuple[int, int, int, int]) -> None:
        d.ellipse(box, fill=(255, 255, 255), outline=(0, 0, 0), width=3)

    vertical = font("msgothic", 34)
    horizontal = font("yugothic_bold", 30)
    black: RGB = (0, 0, 0)

    bubble((60, 40, 300, 360))  # two columns, read right to left
    _vertical_text(d, 215, 70, "ここは危険だ", vertical, black)
    _vertical_text(d, 140, 70, "早く逃げろ！", vertical, black)

    bubble((350, 60, 520, 330))  # single column
    _vertical_text(d, 418, 90, "お前は誰だ？", vertical, black)

    bubble((560, 120, 880, 260))  # horizontal
    d.text((600, 170), "ありがとう、助かった", font=horizontal, fill=black)

    _vertical_text(d, 820, 430, "まさか…", font("msgothic", 40), (255, 255, 255))
    d.text((80, 520), "うるさい！", font=font("yugothic_bold", 64), fill=(255, 80, 80))
    return img


GENERATORS = {
    "webpage": make_webpage,
    "rotated": make_rotated,
    "dark_ui": make_dark_ui,
    "manga": make_manga,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory (default: demo/images)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, make in GENERATORS.items():
        path = args.out / f"{name}.png"
        make().save(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
