"""Rendering helpers: style measurement, text fitting and PIL composition."""
from .compose import compose, default_font_path, pil_measurer
from .fit import FitResult, Measure, fit_text, wrap_lines
from .style import (
    extract_colors,
    is_vertical,
    measure_style,
    quad_angle_deg,
    quad_text_height,
    quad_text_width,
)

__all__ = [
    "FitResult",
    "Measure",
    "compose",
    "default_font_path",
    "extract_colors",
    "fit_text",
    "is_vertical",
    "measure_style",
    "pil_measurer",
    "quad_angle_deg",
    "quad_text_height",
    "quad_text_width",
    "wrap_lines",
]
from .layout import TextBlock, build_blocks, glyph_size, is_kana_only  # noqa: E402,F401
from .typeset import PlacedLine, Typeset, mask_spans, rect_spans, typeset  # noqa: E402,F401
from .compose import manga_font_path, typeset_block, block_spans  # noqa: E402,F401
