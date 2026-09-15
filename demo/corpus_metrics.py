"""The corpus metrics ``demo/typeset_metrics.py`` does not already produce, and
the glue that turns one ``score_page`` result into the components
``demo/corpus_score.py`` weighs.

Six of the seven corpus metrics already exist in the per-page machinery
(containment and centring from ``typeset_metrics.score_geometry``, the erase
IoU and the kept art from ``score_erase``, the surviving ink from
``score_leftover``, the cap ratio from ``score_page``) and none of them is
reimplemented here - :func:`page_components` *consumes* those numbers.  What
this module adds is:

* :func:`text_iou` - did we letter the same *text area* the release lettered?
  Both masks are dilated by a stroke width first, because two correct renders
  of one sentence in different fonts share almost no glyph pixels;
* :func:`group_f1` - did we cut the page into the same *blocks*?  One lettered
  block per balloon, nothing lettered where the release left the art alone,
  nothing left unlettered;
* :func:`leftover_em2`, :func:`size_logratio_rms`, :func:`line_exact` - the
  resolution-free / symmetric / exact-match forms of numbers the per-page
  tables already carry per block;
* :func:`answered_fraction` - the anti-degenerate gate: abstaining on the hard
  blocks must strictly lose, never raise the remaining components;
* :func:`text_free_windows` and :func:`lpips_excess` - the arithmetic around
  LPIPS.  LPIPS itself needs torch and runs out of process in the sidecar venv;
  this module only picks the windows to measure the per-page floor on and
  subtracts it.

Everything above :func:`page_extras` is pure: numpy and cv2 (for the dilation)
only, no repo imports, no image I/O, no torch, no network - callers pass plain
masks and numbers in.  :func:`page_extras` is the one exception, and the only
function that reads files: it assembles a whole page's extras for
``demo/corpus_eval.py`` and imports the render stack lazily so that importing
this module stays cheap.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------- constants
TEXT_IOU_DILATE_EM = 0.15  # em: roughly one stroke width; both masks grow by this before the IoU
ASSIGN_EM = 2.0  # em: unassigned English lines closer than this are one missed balloon (typeset_reference.ASSIGN_EM)
REFERENCE_WEAK_THRESHOLD = 0.2  # share of English lines the ownership pass may drop before the page is untrustworthy
# ...but also at least this many lines.  On a two-line page a single dropped line is 50 %,
# which is not evidence of anything: 6 of the 11 pages the fraction alone flagged tripped it
# on 1-2 lines total, and several of those lines were rapidocr firing on screentone ("8889",
# "00000").  Requiring both a fraction and a count takes it from 11 pages to 4.
REFERENCE_WEAK_MIN_LINES = 3
# Largest believable official cap height, in source ems - see _plausible_ref.
REF_CAP_EM_MAX = 1.2
WINDOW_SIZE = 256  # px: side of an LPIPS floor window
WINDOW_COUNT = 6  # how many floor windows a page contributes at most
ENGLISH_BOX_PAD = 2  # px the English OCR boxes grow before masking their ink (typeset_reference.ENGLISH_BOX_PAD)

# The keys :func:`page_components` always returns, in the order the score reads
# them.  ``corpus_score`` maps them through its own ``c_*`` functions (erase_iou
# and art_kept share one component); the weights live only in that module.
COMPONENT_KEYS: Tuple[str, ...] = (
    "containment_mean",
    "erase_iou",
    "art_kept",
    "lpips_excess",
    "centre_offset_em_mean",
    "leftover_em2_mean",
    "size_logratio_rms",
    "group_f1",
    "line_exact",
    "text_iou_mean",
)
# Only bubble blocks have an interior, so a page without one has no value here
# at all - None, so the score renormalises, never 0, which would read as perfect.
BALLOON_ONLY_KEYS: Tuple[str, ...] = ("containment_mean", "centre_offset_em_mean", "leftover_em2_mean")
# What a weak English reference cannot be asked about (see :func:`reference_weak`).
# Metrics dropped when the English reference under-covers the page.  group_f1 is
# DELIBERATELY NOT among them: its FN term is the only thing in the whole score that
# penalises a balloon we never detected, and unassigned English lines are exactly what a
# missed balloon produces - so dropping it on a weak reference deleted the penalty in
# precisely the case it exists for.  A page with every block wrong in both directions
# (tp=0, fp=1, fn=2) scored 35.96 because group_f1 had been removed.
WEAK_REFERENCE_KEYS: Tuple[str, ...] = ("text_iou_mean", "line_exact")


# --------------------------------------------------------------- small helpers
def _kernel(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a bool mask by ``radius`` px (elliptical structuring element)."""
    grown = np.asarray(mask).astype(bool)
    if radius <= 0 or not grown.any():
        return grown
    return cv2.dilate(grown.astype(np.uint8), _kernel(radius)).astype(bool)


def _number(value: Any) -> Optional[float]:
    """``value`` as a finite float, or None (None, NaN and inf all drop out)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: Iterable[Any]) -> Optional[float]:
    """Mean of the finite values, or None when there is nothing to average."""
    nums = [v for v in (_number(x) for x in values) if v is not None]
    return float(sum(nums) / len(nums)) if nums else None


def _box_xywh(box: Any) -> Tuple[float, float, float, float]:
    """``(x, y, w, h)`` of a ``Rect``-like object, a 4-sequence, a ``Segment``
    (its ``bbox``) or the ``{"bbox": [x, y, w, h], ...}`` records ``blocks.json``
    stores its English lines as."""
    if isinstance(box, Mapping):
        box = box.get("bbox", box)
    box = getattr(box, "bbox", box)
    if hasattr(box, "x") and hasattr(box, "w"):
        return float(box.x), float(box.y), float(box.w), float(box.h)
    x, y, w, h = (float(v) for v in tuple(box)[:4])
    return x, y, w, h


def _overlaps(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


# --------------------------------------------------------------- text IoU
def text_iou(ours_ink: np.ndarray, ref_ink: np.ndarray, em_px: float) -> Optional[float]:
    """Overlap of our lettering with the release's, as *text areas* rather than
    glyphs: both bool masks are dilated by ``TEXT_IOU_DILATE_EM`` ems (about one
    stroke width, at least 1 px) before ``|A and B| / |A or B|``.

    The dilation is the whole point.  Two *correct* renders of the same sentence
    in different fonts - and the release's font is not ours - put their strokes
    a few pixels apart and draw them at different weights, so their raw glyph
    IoU is near zero while their inked *footprints* are nearly identical.  The
    undilated number would therefore punish every page for using the wrong
    typeface instead of measuring whether the right words sit in the right
    place; the dilated one measures the footprint.

    Returns None - never 0.0 - when either mask is empty: there is no text to
    compare, and a page with nothing to letter must not be scored as a total
    miss.
    """
    ours = np.asarray(ours_ink).astype(bool)
    ref = np.asarray(ref_ink).astype(bool)
    if ours.shape != ref.shape:
        raise ValueError(f"mask shapes differ: {ours.shape} vs {ref.shape}")
    if not ours.any() or not ref.any():
        return None
    radius = max(1, int(round(TEXT_IOU_DILATE_EM * max(_number(em_px) or 0.0, 0.0))))
    grown_ours, grown_ref = _dilate(ours, radius), _dilate(ref, radius)
    union = int((grown_ours | grown_ref).sum())
    return float(int((grown_ours & grown_ref).sum())) / union if union else None


# --------------------------------------------------------------- grouping
def _centre(segment: Any) -> Tuple[float, float]:
    """Centre of a segment's bounding box (see :func:`_box_xywh` for the shapes)."""
    x, y, w, h = _box_xywh(segment)
    return x + w / 2.0, y + h / 2.0


def _cluster_count(segments: Sequence[Any], em_px: float, radius_em: float = ASSIGN_EM) -> int:
    """How many balloons a set of unassigned English lines stands for: single-link
    clusters of their bbox centres within ``radius_em`` ems, the same reach
    ``typeset_reference.assign_english`` uses to give a line to a block."""
    count = len(segments)
    if count == 0:
        return 0
    limit = radius_em * max(_number(em_px) or 0.0, 1e-6)
    centres = [_centre(s) for s in segments]
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(count):
        for j in range(i + 1, count):
            if math.hypot(centres[i][0] - centres[j][0], centres[i][1] - centres[j][1]) <= limit:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_i] = root_j
    return len({find(i) for i in range(count)})


def group_f1(owned: Sequence[Sequence[Any]], unassigned: Sequence[Any], shared: Sequence[Optional[int]],
             em_px: float, untranslated: Optional[Set[int]] = None) -> Dict[str, Any]:
    """Did we cut the page into the blocks the release lettered?

    ``owned`` / ``unassigned`` come from ``typeset_reference.assign_english``
    (English lines per rendered block, and the lines no block claimed);
    ``shared`` from ``typeset_metrics.blocks_per_bubble`` (how many of our
    blocks letter into the same balloon interior; None for free text, which has
    no interior and cannot be shared).

    * **TP** - our block holds at least one English line and is alone in its
      balloon (``blocks_per_bubble`` 1, or None for free text);
    * **FP** - our block holds no English line (we lettered where the release
      did not) or shares its balloon (two of ours in one balloon);
    * **FN** - a cluster of unassigned English lines: a balloon the release
      lettered and we did not.

    ``f1 = 2*TP / (2*TP + FP + FN)``.  Blocks listed in ``untranslated`` are left
    out entirely: the release genuinely leaves logos, author names and
    watermarks alone, so neither lettering nor not lettering them is an error.

    Deviation from a literal reading of the definition: a block that is both
    empty *and* over-segmented counts as **one** FP, not two - every one of our
    blocks is classified exactly once, so the count cannot exceed the number of
    blocks.  Returns ``group_f1: None`` when there is nothing to score at all
    (no blocks of ours and no unassigned lines).
    """
    skip = set(untranslated or ())
    true_positive = false_positive = 0
    for index, lines in enumerate(owned):
        if index in skip:
            continue
        count = shared[index] if index < len(shared) else None
        over_segmented = count is not None and int(count) > 1
        if len(lines) >= 1 and not over_segmented:
            true_positive += 1
        else:
            false_positive += 1
    false_negative = _cluster_count(list(unassigned), em_px)
    denominator = 2 * true_positive + false_positive + false_negative
    score = (2.0 * true_positive / denominator) if denominator else None
    return {"group_f1": score, "tp": true_positive, "fp": false_positive, "fn": false_negative}


# --------------------------------------------------------------- scalar forms
def leftover_em2(leftover_px: Optional[float], em_px: float) -> Optional[float]:
    """Surviving Japanese ink in *em squares*: ``leftover_px / em_px**2``.

    ``typeset_metrics.score_leftover`` counts pixels, which a 2x scan doubles in
    each direction for the same page; dividing by the source em squared says
    "this much of an em-square of Japanese ink is still readable" and compares
    across volumes.  None when there is no count or no em (``em_px <= 0``)."""
    pixels = _number(leftover_px)
    em = _number(em_px)
    if pixels is None or em is None or em <= 0.0:
        return None
    return pixels / (em * em)


def size_logratio_rms(cap_ratios: Iterable[Any]) -> Optional[float]:
    """``sqrt(mean(ln(r)**2))`` over the cap ratios (ours / the release's) that
    are finite and positive.  0 is perfect, and the log makes it symmetric:
    lettering twice too big scores exactly as badly as half too small, which a
    plain ratio error does not.  None when no block is comparable."""
    logs = [math.log(r) for r in (_number(v) for v in cap_ratios) if r is not None and r > 0.0]
    return math.sqrt(sum(value * value for value in logs) / len(logs)) if logs else None


def line_exact(rows: Iterable[Mapping[str, Any]]) -> Optional[float]:
    """Share of the compared blocks whose rendered ``lines`` equals the
    reference's ``lines_ref`` (the rows ``typeset_metrics.score_lettering``
    produces).  A block with no ``lines_ref`` was not compared; a block flagged
    ``untranslated`` is left out, as in ``typeset_metrics.summarize``.  None when
    nothing is comparable."""
    compared = matched = 0
    for row in rows:
        if row.get("untranslated"):
            continue
        reference = row.get("lines_ref")
        if reference is None:
            continue
        compared += 1
        if row.get("lines") == reference:
            matched += 1
    return matched / float(compared) if compared else None


def answered_fraction(records: Sequence[Mapping[str, Any]],
                      paired_rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    """The anti-degenerate gate: of the ground-truth blocks the release actually
    lettered, what share did we put ink on?

    ``records`` are the ground-truth blocks (``demo/reference/<stem>/blocks.json``)
    and ``paired_rows`` the rows ``typeset_metrics.score_page`` produced for
    them, in the same order.  A record counts when its reference ``stats`` carry
    lettering ink, and is answered when our row carries ``ink_px > 0`` (a row
    flagged ``missed`` never does).  Untranslated blocks are excluded on both
    sides.

    This multiplies the page score in ``corpus_score.page_score``, so abstaining
    strictly loses: skipping the hard balloons would otherwise *raise* every
    remaining component (the blocks we did letter are the easy ones) and buy a
    better score for doing less.  Returns 0.0 when blocks needed answering and we
    produced nothing; None only when the page genuinely has no lettered ground
    truth, which leaves the gate open rather than zeroing an honest page."""
    needed = answered = 0
    for index, record in enumerate(records):
        row = paired_rows[index] if index < len(paired_rows) else {}
        if record.get("untranslated") or row.get("untranslated"):
            continue
        stats = record.get("stats") or {}
        if not (_number(stats.get("ink_px")) or 0.0) > 0.0:
            continue
        needed += 1
        if not row.get("missed") and (_number(row.get("ink_px")) or 0.0) > 0.0:
            answered += 1
    return answered / float(needed) if needed else None


# --------------------------------------------------------------- LPIPS glue
def text_free_windows(shape: Sequence[int], text_boxes: Sequence[Any], count: int = WINDOW_COUNT,
                      size: int = WINDOW_SIZE) -> List[Tuple[int, int, int, int]]:
    """Up to ``count`` square windows of ``size`` px that touch no text box,
    spread over the page: ``(x, y, w, h)`` each.

    These are where the per-page LPIPS *floor* is measured - art the letterer
    never touched, so any LPIPS there is scan difference, not our damage (see
    :func:`lpips_excess`).  Candidates are laid on a half-window grid, the ones
    meeting text are dropped, and the rest are taken farthest-point first from
    the page centre outwards, never overlapping each other, so the floor is not
    read off one corner.  Fewer than ``count`` (possibly none) come back on a
    page that is mostly text; the window never runs off the page."""
    height, width = int(shape[0]), int(shape[1])
    side = int(size)
    if side <= 0 or int(count) <= 0 or height < side or width < side:
        return []
    boxes = [_box_xywh(box) for box in text_boxes]
    step = max(1, side // 2)
    xs = list(range(0, width - side + 1, step))
    ys = list(range(0, height - side + 1, step))
    if xs[-1] != width - side:
        xs.append(width - side)
    if ys[-1] != height - side:
        ys.append(height - side)
    free = [(x, y) for y in ys for x in xs
            if not any(_overlaps((float(x), float(y), float(side), float(side)), box) for box in boxes)]
    if not free:
        return []

    page_centre = (width / 2.0, height / 2.0)

    def centre_of(origin: Tuple[int, int]) -> Tuple[float, float]:
        return origin[0] + side / 2.0, origin[1] + side / 2.0

    chosen: List[Tuple[int, int]] = []
    while free and len(chosen) < int(count):
        if not chosen:
            pick = min(free, key=lambda o: (math.hypot(centre_of(o)[0] - page_centre[0],
                                                       centre_of(o)[1] - page_centre[1]), o[1], o[0]))
        else:
            pick = max(free, key=lambda o: (min(math.hypot(centre_of(o)[0] - centre_of(c)[0],
                                                           centre_of(o)[1] - centre_of(c)[1]) for c in chosen),
                                            -o[1], -o[0]))
        chosen.append(pick)
        taken = (float(pick[0]), float(pick[1]), float(side), float(side))
        free = [o for o in free if not _overlaps((float(o[0]), float(o[1]), float(side), float(side)), taken)]
    return [(x, y, side, side) for x, y in chosen]


def lpips_excess(mean_art: Optional[float], floor: Optional[float]) -> Optional[float]:
    """``max(0.0, mean_art - floor)``: the perceptual distance over the art we
    regenerated, *above* what the two scans differ by anyway.

    The Japanese and English volumes are different scans: different JPEG
    quality, different screentone moire, different gamma - and LPIPS is
    sensitive to all three.  Raw LPIPS over an untouched panel is therefore
    already well above zero, by an amount that changes from volume to volume, so
    without subtracting a floor measured on this page's own text-free windows
    (:func:`text_free_windows`) the number cannot be compared across the corpus.
    Clamped at 0: beating the floor is noise, not credit.  None when either side
    is missing - a page whose floor could not be measured has no comparable art
    score, and the weight renormalises away instead of guessing."""
    art, base = _number(mean_art), _number(floor)
    if art is None or base is None:
        return None
    return max(0.0, art - base)


# --------------------------------------------------------------- page glue
def reference_weak(unassigned: Any, total_en_lines: int, threshold: float = REFERENCE_WEAK_THRESHOLD) -> bool:
    """True when more than ``threshold`` of the release's English lines were
    never assigned to one of our blocks (``unassigned`` is that list, or its
    length).

    The English reference comes from rapidocr, which misses stylised and SFX
    lettering - exactly the lettering that is hardest to place.  When a fifth of
    the lines went unclaimed the reference under-covers the page rather than the
    render failing it, so the metrics that compare against it
    (:data:`WEAK_REFERENCE_KEYS`: grouping, text IoU and line counts) must be
    dropped for that page instead of scored."""
    count = int(unassigned) if isinstance(unassigned, int) else len(unassigned)
    return (count >= REFERENCE_WEAK_MIN_LINES
            and count > float(threshold) * int(total_en_lines))


def _em_at(em_px: Any, index: int) -> Optional[float]:
    """The source em of block ``index``: from a per-block sequence, or one
    page-wide value, or None."""
    if em_px is None:
        return None
    if isinstance(em_px, (list, tuple, np.ndarray)):
        return _number(em_px[index]) if index < len(em_px) else None
    return _number(em_px)


def _answered(records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
              scored_indices: Sequence[int], is_sfx: Any, missed_clusters: int,
              ) -> Optional[float]:
    """Of the blocks the release lettered, how many did we actually ink?

    Two things this must get right that a naive pass over ``records`` does not:

    * **A balloon we never detected has no record.**  ``records`` comes from OUR
      segmentation, so a missed balloon is simply absent and a page that missed half its
      dialogue still reported ``answered = 1.0``.  ``missed_clusters`` - ``group_f1``'s FN
      term, the clusters of English the release lettered and no block of ours claimed -
      goes into the denominator, so an undetected balloon is an unanswered block.
    * **Echoing the source back is not abstention, it is a wrong answer.**  An
      ``untranslated`` block (translation == source) is excluded from every other metric,
      so emitting the Japanese for a block we cannot letter well used to DELETE the
      evidence rather than lose points.  Such a block still counts in the denominator here
      and never in the numerator.
    """
    def row_of(index: int) -> Mapping[str, Any]:
        return rows[index] if index < len(rows) else {}

    lettered = [i for i in range(len(records))
                if not is_sfx(i) and (records[i].get("english_lines") or [])]
    if not lettered and not missed_clusters:
        return None
    inked = sum(1 for i in lettered
                if not row_of(i).get("untranslated")
                and float(row_of(i).get("ink_px") or 0.0) > 0.0)
    denominator = len(lettered) + int(missed_clusters)
    return (inked / denominator) if denominator else None


def _plausible_ref(row: Mapping[str, Any]) -> bool:
    """Is this block's OFFICIAL cap-height measurement physically possible?

    ``cap_ref_em`` is the release's capital height in units of the source em, so it sits
    near 0.69 on a real block and can never approach 1.  Measured on the corpus, 17.7 % of
    blocks report more than 1.2 and one reports 6.08, and 40 of those 47 also report
    ``lines_ref == 1``: ``typeset_reference.line_clusters`` splits lines on ink-free rows,
    and VIZ's italic lettering, warped through the alignment homography, never leaves a
    blank row - so every line merges into one run and the "cap height" becomes the whole
    block.  Those blocks are a reference-side artefact, not a render defect, and they were
    half of the apparent "37 % of blocks too small".  They are excluded from the size and
    line metrics and counted, rather than silently dominating them.
    """
    cap_ref = _number(row.get("cap_ref_em"))
    return cap_ref is None or cap_ref <= REF_CAP_EM_MAX


def _prefer(extras: Mapping[str, Any], key: str, fallback: Optional[float]) -> Optional[float]:
    """``page_extras``' value when it supplied one, else the local computation.

    Only ``page_extras`` knows the in-scope block set, so its number wins whenever the key
    is PRESENT - including when it is ``None``, which is a real "nothing in scope to
    measure on this page" and must not silently fall back to a computation that would put
    the out-of-scope blocks back in.  Hence the ``in`` test, not a nullity test.
    """
    return _number(extras[key]) if key in extras else fallback


def _group_value(value: Any) -> Optional[float]:
    """The score out of :func:`group_f1`'s dict, or a plain number."""
    if isinstance(value, Mapping):
        return _number(value.get("group_f1"))
    return _number(value)


def page_components(scored: Mapping[str, Any], extras: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Turn one ``typeset_metrics.score_page`` result plus the extras this module
    computes into the component inputs of ``demo/corpus_score.py``.

    Every key of :data:`COMPONENT_KEYS` is always present and is a float or
    None, plus ``answered``.  Raw values only: ``corpus_score`` maps them through
    its own ``c_*`` functions and applies the weights, which live there and
    nowhere else.

    ``extras`` may carry:

    * ``em_px`` - the source em per block (aligned with ``scored["blocks"]``,
      i.e. the ground-truth records' ``em_px``) or one page-wide value; needed
      to turn leftover pixels into em squares;
    * ``lpips_art_mean`` / ``lpips_floor`` - the sidecar's numbers, subtracted
      by :func:`lpips_excess`;
    * ``group_f1`` - :func:`group_f1`'s dict or its score;
    * ``text_iou`` - :func:`text_iou` per block (or one value), meaned;
    * ``reference_weak`` - :func:`reference_weak`, which blanks
      :data:`WEAK_REFERENCE_KEYS`;
    * ``answered`` - :func:`answered_fraction`; absent or None means the page had
      nothing to answer and the gate stays open at 1.0.

    A page with no balloon blocks reports None - never 0 - for
    :data:`BALLOON_ONLY_KEYS`: the geometry summary simply has no value there,
    and ``page_score`` renormalises the remaining weights instead of reading a
    missing balloon as a perfectly centred one."""
    extras = extras or {}
    summary: Mapping[str, Any] = scored.get("summary") or {}
    blocks: Sequence[Mapping[str, Any]] = scored.get("blocks") or []
    rows = [row for row in blocks if not row.get("untranslated")]

    leftovers: List[Optional[float]] = []
    for index, row in enumerate(blocks):
        if row.get("untranslated") or row.get("leftover_px") is None:
            continue
        value = leftover_em2(row.get("leftover_px"), _em_at(extras.get("em_px"), index) or 0.0)
        if value is not None:
            leftovers.append(value)

    text_ious = extras.get("text_iou")
    if text_ious is None:
        text_iou_mean: Optional[float] = None
    elif isinstance(text_ious, (list, tuple, np.ndarray)):
        text_iou_mean = _mean(list(text_ious))
    else:
        text_iou_mean = _number(text_ious)

    out: Dict[str, Any] = {
        "containment_mean": _number(summary.get("containment")),
        "erase_iou": _number(summary.get("erase_iou")),
        "art_kept": _number(summary.get("art_kept")),
        "lpips_excess": lpips_excess(extras.get("lpips_art_mean"), extras.get("lpips_floor")),
        "centre_offset_em_mean": _number(summary.get("centre_offset_em")),
        # These three prefer the values page_extras already computed, because only
        # page_extras knows which blocks are IN SCOPE: it drops stylised onomatopoeia,
        # which this function cannot see.  Recomputing them from scored["blocks"] with
        # only the `untranslated` filter would silently put the SFX blocks back - and
        # `lines_ref` is 0 (not None) for an SFX block, so line_exact really does
        # compare them.  The local computation is the fallback for a caller with no
        # extras.
        "leftover_em2_mean": _prefer(extras, "leftover_em2_mean", _mean(leftovers)),
        "size_logratio_rms": _prefer(
            extras, "size_logratio_rms",
            size_logratio_rms([row.get("cap_ratio") for row in rows])),
        "group_f1": _group_value(extras.get("group_f1")),
        "line_exact": _prefer(extras, "line_exact", line_exact(rows)),
        "text_iou_mean": text_iou_mean,
    }
    if extras.get("reference_weak"):
        for key in WEAK_REFERENCE_KEYS:
            out[key] = None
    answered = _number(extras.get("answered"))
    out["answered"] = 1.0 if answered is None else max(0.0, min(1.0, answered))
    return out


# --------------------------------------------------------------- whole page
def _window_mask(shape: Sequence[int], window: Sequence[int]) -> np.ndarray:
    x, y, w, h = (int(v) for v in tuple(window)[:4])
    mask = np.zeros((int(shape[0]), int(shape[1])), bool)
    mask[max(0, y): y + h, max(0, x): x + w] = True
    return mask


def _boxes_mask(shape: Sequence[int], boxes: Iterable[Any], pad: int = 0) -> np.ndarray:
    """Rectangles rasterised into a bool mask, each grown by ``pad`` px: the
    rasterisation of ``typeset_reference.boxes_mask``, over any shape
    :func:`_box_xywh` accepts."""
    mask = np.zeros((int(shape[0]), int(shape[1])), bool)
    for box in boxes:
        x, y, w, h = _box_xywh(box)
        x1, y1 = max(0, int(x) - pad), max(0, int(y) - pad)
        mask[y1: int(y) + int(h) + pad, x1: int(x) + int(w) + pad] = True
    return mask


def _read_gray(path: Path) -> Optional[np.ndarray]:
    """Bytes through Python, never a ``str`` path into OpenCV - see corpus_index."""
    path = Path(path)
    if not path.exists():
        return None
    buf = np.frombuffer(path.read_bytes(), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf.size else None


def _load_lpips(path: Path) -> Tuple[Optional[float], Optional[float]]:
    """``(mean_art, floor)`` from the sidecar's result JSON, or ``(None, None)``
    when it is missing or unreadable."""
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # Distinguish "the sidecar wrote garbage" from "the sidecar did not run": both
        # renormalise c_art away, so without this the corruption is invisible.
        print(f"corpus_metrics: unreadable LPIPS result {path}: {exc}", file=sys.stderr)
        return None, None
    if not isinstance(data, Mapping):
        print(f"corpus_metrics: malformed LPIPS result {path}", file=sys.stderr)
        return None, None
    floor = next((v for v in (_number(data.get(key)) for key in ("floor", "mean_floor", "lpips_floor", "mean"))
                  if v is not None), None)
    return _number(data.get("mean_art")), floor


def page_extras(scored: Dict[str, Any], truth: Dict[str, Any], *,
                render_dir: Path, tag: str, ja_path: Path,
                unassigned: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """Everything one page needs beyond ``typeset_metrics.score_page``: the
    structural flags ``demo/corpus_eval.py`` gates on, and the ``extras`` mapping
    :func:`page_components` reads.

    ``scored`` is a ``score_page`` result and ``truth`` a
    ``typeset_metrics.load_ground_truth`` one (``gt``, ``eng_gray``,
    ``records``); ``render_dir`` holds ``<tag>_erased.png`` /
    ``<tag>_typeset.png`` and ``ja_path`` the materialised JA page, which fixes
    the shape the masks are built at.  ``score_page`` emits one row per
    ground-truth record, in order, so rows, records and the returned per-block
    lists all share one index.

    Returned, besides every key :func:`page_components` reads as its ``extras``:
    ``reference_weak``, ``bubble_heavy`` (most ground-truth blocks are in
    balloons), ``free_text_only`` (none is - which is what leaves
    :data:`BALLOON_ONLY_KEYS` None), and the page totals ``overflow_px`` /
    ``leftover_px`` - plain ints, 0 when clean, the regression gate's two hard
    invariants.

    Grouping is read off the pairing ``score_page`` already did.  The ground
    truth's records *are* blocks of this same pipeline, so each record's
    ``english_lines`` is that block's ``owned`` list: a record our render
    missed contributes its English lines to the unassigned pile (a balloon the
    release lettered and we did not), and each of ``summary["unpaired_blocks"]``
    - blocks we lettered that no record claims - contributes one empty block, so
    :func:`group_f1`'s own arithmetic turns both into FN and FP.  The unassigned
    lines of the ground truth are taken from ``truth["unassigned_english"]`` (or
    the ``unassigned`` argument); without them the page can only be judged on
    the records, and :func:`reference_weak` cannot fire.

    **LPIPS is never computed here** - it needs torch and runs in the sidecar
    venv.  The sidecar's numbers are read from ``<render_dir>/<tag>_lpips.json``
    (``{"mean", "mean_art", "blocks": [...]}``; the floor is its ``floor`` /
    ``mean_floor`` key when present, else ``mean``, the page-wide value measured
    on the text-free windows of :func:`text_free_windows`).  When that file is
    absent ``lpips_excess`` is None, so ``corpus_score`` renormalises the art
    component away rather than reading an unmeasured page as perfect."""
    records: List[Mapping[str, Any]] = list(truth.get("records") or [])
    rows: List[Mapping[str, Any]] = list(scored.get("blocks") or [])
    summary: Mapping[str, Any] = scored.get("summary") or {}

    def row_at(index: int) -> Mapping[str, Any]:
        return rows[index] if index < len(rows) else {}

    kinds = [str(record.get("kind") or "") for record in records]

    def is_sfx(index: int) -> bool:
        """Stylised onomatopoeia: free text over artwork the release did not re-letter.

        Out of scope by decision (2026-09-15), and excluded rather than penalised.  The
        release variously leaves Japanese SFX alone, glosses it, or redraws it in English,
        and which of those is right is an editorial call this harness cannot make.  Scoring
        these blocks would reward us on the pages where VIZ left the SFX and punish us on
        the pages where VIZ redrew it, for identical behaviour - so any conclusion drawn
        from them would be noise.
        """
        return kinds[index] == "art" and not (records[index].get("english_lines") or [])

    scored_indices = [i for i in range(len(records))
                      if not row_at(i).get("untranslated") and not is_sfx(i)]
    sfx_excluded = sum(1 for i in range(len(records)) if is_sfx(i))

    bubbles = sum(1 for i in scored_indices if kinds[i] == "bubble")
    ems = [_number(record.get("em_px")) for record in records]
    known_ems = [em for em in ems if em is not None and em > 0.0]
    page_em = float(np.median(known_ems)) if known_ems else 1.0

    # Grouping: one block per balloon, nothing extra, nothing missed.
    owned: List[List[Any]] = []
    shared: List[Optional[int]] = []
    missed_lines: List[Any] = []
    for index in scored_indices:
        lines = list(records[index].get("english_lines") or [])
        if row_at(index).get("missed"):
            missed_lines.extend(lines)
            continue
        owned.append(lines)
        shared.append(row_at(index).get("blocks_per_bubble"))
    for _ in range(int(summary.get("unpaired_blocks") or 0)):
        owned.append([])
        shared.append(None)
    loose = list(unassigned if unassigned is not None else (truth.get("unassigned_english") or []))
    grouping = group_f1(owned, loose + missed_lines, shared, page_em)

    # Text IoU: our lettering against the release's, block by block.
    ground: Any = truth.get("gt")
    english = np.asarray(ground.english).astype(bool) if ground is not None else None
    ja_gray = _read_gray(Path(ja_path))
    shape: Optional[Tuple[int, int]] = None
    if ja_gray is not None:
        shape = (int(ja_gray.shape[0]), int(ja_gray.shape[1]))
    elif english is not None:
        shape = (int(english.shape[0]), int(english.shape[1]))
    erased = _read_gray(Path(render_dir) / f"{tag}_erased.png")
    typeset = _read_gray(Path(render_dir) / f"{tag}_typeset.png")
    ious: List[Optional[float]] = []
    usable = (english is not None and shape is not None and erased is not None and typeset is not None
              and erased.shape[:2] == shape and typeset.shape[:2] == shape and english.shape[:2] == shape)
    if usable:
        # Local import: keeps the arithmetic above importable without the render stack.
        if str(Path(__file__).resolve().parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
        import typeset_metrics as TM

        for index in scored_indices:
            record = records[index]
            lines = list(record.get("english_lines") or [])
            if not lines or not record.get("window") or row_at(index).get("missed"):
                continue
            window = _window_mask(shape, record["window"])
            reference = english & window & _boxes_mask(shape, lines, ENGLISH_BOX_PAD)
            ours = TM.our_lettering(typeset, erased, window, bool(record.get("dark")))
            # None (no ink on one side) drops out: answered_fraction is what punishes lettering nothing.
            ious.append(text_iou(ours, reference, ems[index] or page_em))

    leftovers = [leftover_em2(row_at(i).get("leftover_px"), ems[i] or 0.0) for i in scored_indices
                 if row_at(i).get("leftover_px") is not None]
    art, floor = _load_lpips(Path(render_dir) / f"{tag}_lpips.json")
    total_lines = sum(len(record.get("english_lines") or []) for record in records) + len(loose)
    return {
        "reference_weak": reference_weak(loose, total_lines),
        "bubble_heavy": bubbles * 2 > len(scored_indices),
        "free_text_only": bubbles == 0,
        # Stylised onomatopoeia is out of scope: reported so the exclusion is visible in
        # the triage rather than silently shrinking the sample.
        "sfx_excluded": sfx_excluded,
        "scored_blocks": len(scored_indices),
        "overflow_px": int(sum(int(row_at(i).get("overflow_px") or 0) for i in scored_indices)),
        "leftover_px": int(sum(int(row_at(i).get("leftover_px") or 0) for i in scored_indices)),
        "em_px": ems,
        "text_iou": ious,
        "text_iou_mean": _mean(ious),
        "group_f1": grouping["group_f1"],
        "group_tp": grouping["tp"],
        "group_fp": grouping["fp"],
        "group_fn": grouping["fn"],
        "leftover_em2_mean": _mean(leftovers),
        "size_logratio_rms": size_logratio_rms(
            [row_at(i).get("cap_ratio") for i in scored_indices if _plausible_ref(row_at(i))]),
        "line_exact": line_exact(
            [row_at(i) for i in scored_indices if _plausible_ref(row_at(i))]),
        "implausible_ref_blocks": sum(1 for i in scored_indices
                                      if not _plausible_ref(row_at(i))),
        "lpips_art_mean": art,
        "lpips_floor": floor,
        "lpips_excess": lpips_excess(art, floor),
        "answered": _answered(records, rows, scored_indices, is_sfx, grouping["fn"]),
        "en_lines": total_lines,
        "unassigned_en_lines": len(loose),
    }
