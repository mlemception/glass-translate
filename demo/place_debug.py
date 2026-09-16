"""Debug view for render/place.py: for every free-text block write a small
JPEG with  [ink map | blocked map | page crop with the chosen block]  of its
search box, plus the per-block numbers.  Uses the OCR cache only."""
from __future__ import annotations

import importlib
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glasstranslate.render import build_blocks  # noqa: E402
from glasstranslate.render.compose import block_font_path_for, default_font_path, pil_measurer  # noqa: E402

place = importlib.import_module("glasstranslate.render.place")


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "dbg"
    img = cv2.imread(str(ROOT / "Examples" / "before.jpg"))
    with (ROOT / "demo" / "cache" / "segments_all.pkl").open("rb") as fh:
        segs = pickle.load(fh)
    segs = [s for s in segs if s.confidence >= 0.5]
    blocks = build_blocks(img, segs)
    place.prepare(img, blocks)
    ref = {k: v for k, v in json.loads((ROOT / "demo" / "reference_text.json").read_text(encoding="utf-8")).items() if not k.startswith("_")}
    out = ROOT / "demo" / "output" / "dev"
    for i, b in enumerate(blocks):
        st = b.style
        if st.in_bubble or st.search_box is None:
            continue
        text = ref.get(b.segment.text)
        if not text:
            continue
        text = text.upper()
        bpath = block_font_path_for(st, text, default_font_path())
        ts = place.typeset_block(text, st, pil_measurer(bpath, condense=True))
        sb = st.search_box
        crop = img[sb.y : sb.y2, sb.x : sb.x2].copy()
        bb = ts.bbox
        if bb is not None:
            cv2.rectangle(crop, (bb.x - sb.x, bb.y - sb.y), (bb.x2 - sb.x, bb.y2 - sb.y), (0, 0, 255), 2)
        src = b.segment.bbox
        cv2.rectangle(crop, (src.x - sb.x, src.y - sb.y), (src.x2 - sb.x, src.y2 - sb.y), (0, 200, 0), 1)
        ink = cv2.cvtColor((255 - st.ink_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        blk = cv2.cvtColor((255 - st.blocked_map.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
        for view in (ink, blk):
            cv2.rectangle(view, (src.x - sb.x, src.y - sb.y), (src.x2 - sb.x, src.y2 - sb.y), (0, 200, 0), 1)
            if bb is not None:
                cv2.rectangle(view, (bb.x - sb.x, bb.y - sb.y), (bb.x2 - sb.x, bb.y2 - sb.y), (0, 0, 255), 2)
        sheet = np.concatenate([ink, np.full((sb.h, 6, 3), (0, 0, 220), np.uint8), blk, np.full((sb.h, 6, 3), (0, 0, 220), np.uint8), crop], axis=1)
        scale = min(1.0, 900 / sheet.shape[1])
        sheet = cv2.resize(sheet, (int(sheet.shape[1] * scale), int(sheet.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        path = out / f"{tag}_maps_{i}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        blocked_frac = float(st.blocked_map.mean())
        print(f"block {i}: search {sb.w}x{sb.h}@{sb.x},{sb.y} src {src.w}x{src.h}@{src.x},{src.y} blocked {blocked_frac:.2f} "
              f"ink {float(st.ink_map.mean()):.2f} -> size {ts.size:.1f} lines {len(ts.lines)} bbox {bb} -> {path.name}")
        for l in ts.lines:
            print(f"      {l.text!r}")


if __name__ == "__main__":
    main()
