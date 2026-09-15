"""Corpus-wide page score: one 0-100 number per page from a weighted
geometric mean of nine independent quality components, plus the
aggregation and regression-gate logic that turns scores into a verdict.

PURE module: stdlib + numpy only.  No cv2, no repo imports, no image I/O -
callers compute the raw per-block statistics and pass plain numbers in.
**The component weights and knee/scale constants live in this file and
nowhere else.**  ``page_score`` combines them::

    R_page = 100 * answered * prod(c_i ** (w_i / sum_of_present_w))

A component is ``None`` (absent) when the page has no blocks of the kind
it measures, and the remaining weights are renormalised over what is
present.  ``answered`` is not a component: it is the multiplicative,
anti-degenerate gate that stops abstaining on hard blocks from raising
every other component.  ``corpus_summary`` rolls per-page ``R`` into a
mean, a bad-tail p05 and a zero count, split by ``bubble_heavy`` /
``free_text_only``.  ``component_table_md`` / ``distribution_md`` render
plain markdown tables (numbers and page ids only).  ``compare_baseline``
is the regression gate's pure decision logic; the CLI lives elsewhere.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------- weights
CONTAIN_KNEE = 0.010    # em: c_contain decays to 1/e here
LPIPS_SPAN = 0.30       # LPIPS units: c_art reaches 0 here
# em: c_centre reaches 0 here.  Calibrated against the OFFICIAL release, not against an
# intuition: over 191 reference bubble blocks VIZ's own lettering sits a mean 0.425 em from
# the inscribed-circle centre, p90 1.075 em.  At the original 0.50 the ground truth itself
# scored c_centre = 0.149 - worse than our render - and 20 % of VIZ's own balloons scored
# exactly 0, because a letterer deliberately offsets toward the tail.  A component that
# fails its own reference one time in five cannot carry weight, so the span is now VIZ's
# p90: a render no worse placed than the release scores near 1.
CENTRE_SPAN = 1.10
LEFTOVER_SCALE = 0.25   # em^2: c_leftover decays to 1/e here
# log-ratio units: c_size reaches 0 here.  An RMS ln(cap_ratio) of 0.15 is roughly a
# letterer's own +-15 % variation and scores 0.81; 0.4 (type ~1.5x off) scores 0.5; 0.8
# (type half or double the right size, on average across the page) scores 0.
SIZE_SPAN = 0.80

# Keys match the ``components`` mapping and the ``c_*`` helpers below.  Sums to 1.00.
COMPONENT_WEIGHTS: dict[str, float] = {
    "c_contain": 0.20,
    "c_erase": 0.15,
    "c_art": 0.15,
    "c_centre": 0.12,
    "c_leftover": 0.10,
    "c_size": 0.10,
    "c_group": 0.08,
    "c_lines": 0.06,
    "c_textiou": 0.04,
}

# A headline mean_R drop of this much or less is run-to-run OCR/ORB jitter, not a regression.
# Smallest value an UNBOUNDED component may contribute to the product.
#
# Three components use an exponential kernel with no lower bound: c_contain, c_leftover and
# c_size.  Measured on the corpus, c_size reached 1.6e-29 on a real page, which made a
# nominally 0.10-weight component carry 92 % of the variance of ln(R) while c_contain,
# weighted 0.20, carried 6 % - and drove 20 % of pages to exactly 0.00, so the bad half of
# the distribution, the half the triage report exists to rank, could not be ordered at all.
#
# The floor applies ONLY to those three.  A bounded component reaching 0 is a fact about
# the page - nothing was erased, no block matched the release's - and must still take the
# page to zero; that is the whole point of a geometric mean and it is what the
# degenerate-output audit pins.  At 0.005 a blown exponential still costs a factor of ~0.6
# at weight 0.10 and ~0.34 at weight 0.20, so it hurts badly without annihilating.
COMPONENT_FLOOR = 0.005
# c_centre joined them, and it is a BOUNDED ramp, so the paragraph above needs qualifying:
# the audit's rule is that a component reaching 0 must zero the page, and that conflates two
# different zeros - "drew nothing" (degenerate) and "drew it badly" (ordinary variation).
# c_centre reaches 0 at CENTRE_SPAN, and CENTRE_SPAN is VIZ's OWN p90 (1.10 em): a tenth of
# the release's own balloons sit at or past the point where this component annihilates a
# page.  A component whose zero the ground truth itself reaches cannot be absorbing.  It
# held five of the eleven R = 0 pages down (v04_p093, v05_p026, v12_p140, v13_p004h1,
# v16_p143), every one of them with lettering actually drawn - badly placed, not missing.
# Nothing here weakens the real degenerate guards: lettering nothing is caught by
# `answered`, erasing nothing by c_erase, bleaching a balloon by c_erase's art_kept.
_FLOORED_COMPONENTS = ("c_contain", "c_leftover", "c_centre")

# Headline drop that still counts as noise rather than a regression.
#
# MEASURED.  This was 1.0, set from "two identical runs of `--sample 50 --seed 1` scored
# 45.72 and 46.48".  **That 0.76 was an artefact, not noise**:
# `demo/output/corpus/report-jitter.md` carries `| c_art | - | 0.15 | - |`, i.e. that run
# never had its LPIPS pass run, so `c_art` was None and the remaining eight weights
# renormalised - which drops the sample's weakest component out of the product and lifts
# R.  `c_erase` is identical to four decimals in both runs (0.697184), which is what a
# deterministic render looks like.
#
# The true render-to-render spread, measured three ways:
#
#   * folding `base`'s own `c_art` values back into the `jitter` run's components gives
#     45.6840 against base's 45.7175 - a spread of **0.0335 R**;
#   * re-scoring `base` from its own cache reproduces 45.72 exactly;
#   * rendering five pages twice in separate processes (including `v01_p020_36f9fe`,
#     17 blocks) gives 0 differing pixels of 8,696,332.
#
# The only nondeterminism located is the LPIPS pass itself - two identical passes over
# identical renders differ in 11 of 188 fields, by at most 0.0052 LPIPS, worth under 0.02
# in `c_art` on one page.  0.15 is ~4.5x the measured spread: enough headroom for that,
# small enough that a real slice is visible above it.  Evidence and the reproduction:
# docs/perf/2026-09-15-typeset-corpus.md.  Tightening this can only make the gate harder
# to pass; never raise it to let a regression through.
TOLERANCE = 0.15

# ...and for the same reason a single page tripping a hard invariant is weak evidence.
# The deliberate-defect demonstration moved 11 pages at once, so requiring two keeps all
# the signal and drops the noise.
INVARIANT_MIN_PAGES = 2


def _clamp01(value: float) -> float:
    """Clamp ``value`` into [0, 1]."""
    return max(0.0, min(1.0, value))


# --------------------------------------------------------------- components
def c_contain(containment_mean: float | None) -> float | None:
    """``exp(-containment_mean / CONTAIN_KNEE)``, clamped to [0, 1]."""
    if containment_mean is None:
        return None
    return _clamp01(math.exp(-containment_mean / CONTAIN_KNEE))


def c_erase(erase_iou: float | None, art_kept: float | None) -> float | None:
    """``sqrt(erase_iou * art_kept)``, clamped to [0, 1].

    ``art_kept`` is None when the ground truth has no kept-art pixels at all - text over
    plain paper, where there was no artwork to preserve.  That is not a reason to discard
    the erase axis: ``erase_iou`` was still measured, and dropping the whole component
    renormalised 0.15 away on some of the best pages in the corpus (three of the top four,
    all with ``erase_iou`` >= 0.996).  So a missing ``art_kept`` falls back to
    ``erase_iou`` alone rather than deleting the component.
    """
    if erase_iou is None:
        return None
    if art_kept is None:
        return _clamp01(erase_iou)
    return _clamp01(math.sqrt(max(0.0, erase_iou * art_kept)))


def c_art(lpips_excess: float | None) -> float | None:
    """``clamp(1 - lpips_excess / LPIPS_SPAN, 0, 1)``."""
    if lpips_excess is None:
        return None
    return _clamp01(1.0 - lpips_excess / LPIPS_SPAN)


def c_centre(centre_offset_em_mean: float | None) -> float | None:
    """``clamp(1 - centre_offset_em_mean / CENTRE_SPAN, 0, 1)``."""
    if centre_offset_em_mean is None:
        return None
    return _clamp01(1.0 - centre_offset_em_mean / CENTRE_SPAN)


def c_leftover(leftover_em2_mean: float | None) -> float | None:
    """``exp(-leftover_em2_mean / LEFTOVER_SCALE)``."""
    if leftover_em2_mean is None:
        return None
    return _clamp01(math.exp(-leftover_em2_mean / LEFTOVER_SCALE))


def c_size(size_logratio_rms: float | None) -> float | None:
    """``clamp(1 - size_logratio_rms / SIZE_SPAN, 0, 1)`` - a BOUNDED ramp.

    This was a Gaussian, ``exp(-rms**2 / (2 * 0.15**2))``, and that was wrong in a way the
    weights hid: with no lower bound it reached 1.6e-29 on a real page, so a nominally
    0.10-weight component carried 92 % of the variance of ln(R) - more than c_contain at
    0.20 - and pushed a fifth of the corpus onto exactly 0.00.  A ramp keeps the component
    inside its weight: a wildly wrong size still reaches a true 0 (and still zeroes the
    page, which the degenerate-output audit requires), but a merely poor one costs
    proportionally instead of exponentially.
    """
    if size_logratio_rms is None:
        return None
    return _clamp01(1.0 - abs(size_logratio_rms) / SIZE_SPAN)


def c_group(group_f1: float | None) -> float | None:
    """Identity pass-through of ``group_f1``, clamped to [0, 1]."""
    return None if group_f1 is None else _clamp01(group_f1)


def c_lines(line_closeness: float | None) -> float | None:
    """Identity pass-through of ``line_closeness``, clamped to [0, 1].

    Its input used to be ``line_exact`` - the share of blocks matching the
    release's line count *exactly* - which on a one-block page is binary.  See
    ``corpus_metrics.line_closeness`` for why an exact-match fraction is the
    wrong shape for a component of a geometric mean.
    """
    return None if line_closeness is None else _clamp01(line_closeness)


def c_textiou(text_iou_mean: float | None) -> float | None:
    """Identity pass-through of ``text_iou_mean``, clamped to [0, 1]."""
    return None if text_iou_mean is None else _clamp01(text_iou_mean)


# --------------------------------------------------------------- page score
def page_score(components: Mapping[str, float | None], answered: float) -> dict[str, Any]:
    """Combine the nine components into one 0-100 page score.  A missing key
    or ``None`` value in ``components`` means that component is absent for
    this page; ``answered`` in [0, 1] multiplies the geometric mean of the
    present components after their weights are renormalised to sum to 1.0."""
    present: dict[str, float] = {}
    missing: list[str] = []
    for key in COMPONENT_WEIGHTS:
        value = components.get(key)
        if value is None:
            missing.append(key)
        else:
            present[key] = _clamp01(value)

    if not present:
        return {
            "R": None,
            "answered": answered,
            "components": {},
            "weights_used": {},
            "missing": missing,
        }

    weight_total = sum(COMPONENT_WEIGHTS[key] for key in present)
    weights_used = {key: COMPONENT_WEIGHTS[key] / weight_total for key in present}

    # Only the unbounded exponential kernels are floored - see COMPONENT_FLOOR.  A bounded
    # component that is genuinely 0 still takes the page to 0, which is what the
    # degenerate-output audit requires.
    floored = {
        key: max(COMPONENT_FLOOR, value) if key in _FLOORED_COMPONENTS else value
        for key, value in present.items()
    }
    # 0 ** x == 0 for any x > 0: guard it explicitly so a zero component drives R to a
    # clean 0.0 rather than a math.log domain error.
    if any(value <= 0.0 for value in floored.values()):
        geometric_mean = 0.0
    else:
        log_sum = sum(weights_used[key] * math.log(value) for key, value in floored.items())
        geometric_mean = math.exp(log_sum)
    r_value = 100.0 * _clamp01(answered) * geometric_mean
    return {
        "R": r_value,
        "answered": answered,
        "components": present,
        "weights_used": weights_used,
        "missing": missing,
    }


# --------------------------------------------------------------- corpus summary
def _valid_r_values(pages: Sequence[Mapping[str, Any]]) -> list[float]:
    return [float(page["R"]) for page in pages if page.get("R") is not None]


def _page_stats(pages: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    """``mean_R``, ``p05_R`` (``verified_weak`` pages excluded) and
    ``zero_count`` over one population of pages."""
    r_values = _valid_r_values(pages)
    mean_r = float(np.mean(r_values)) if r_values else None
    zero_count = sum(1 for r in r_values if r == 0.0)
    tail_pages = [page for page in pages if not page.get("verified_weak", False)]
    tail_values = _valid_r_values(tail_pages)
    p05 = float(np.percentile(tail_values, 5)) if tail_values else None
    return {"mean_R": mean_r, "p05_R": p05, "zero_count": zero_count}


def _component_means(pages: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    sums = {key: 0.0 for key in COMPONENT_WEIGHTS}
    counts = {key: 0 for key in COMPONENT_WEIGHTS}
    for page in pages:
        for key, value in (page.get("components") or {}).items():
            if key in sums and value is not None:
                sums[key] += float(value)
                counts[key] += 1
    return {key: (sums[key] / counts[key] if counts[key] else None) for key in COMPONENT_WEIGHTS}


def corpus_summary(pages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Roll per-page results into ``mean_R`` / ``p05_R`` / ``zero_count``,
    overall and split by ``bubble_heavy`` / ``free_text_only``.  A page with
    ``R is None`` is excluded from every mean, never counted as 100."""
    bubble_heavy = [page for page in pages if page.get("bubble_heavy")]
    free_text_only = [page for page in pages if page.get("free_text_only")]
    summary = dict(_page_stats(pages))
    summary["n_pages"] = len(pages)
    summary["bubble_heavy"] = _page_stats(bubble_heavy)
    summary["free_text_only"] = _page_stats(free_text_only)
    summary["component_means"] = _component_means(pages)
    return summary


# --------------------------------------------------------------- markdown
def component_table_md(summary: Mapping[str, Any]) -> str:
    """Markdown table: key, mean, weight, weight*mean contribution."""
    means = summary.get("component_means", {})
    lines = ["| component | mean | weight | contribution |", "|---|---|---|---|"]
    for key, weight in COMPONENT_WEIGHTS.items():
        mean_value = means.get(key)
        mean_str = f"{mean_value:.3f}" if mean_value is not None else "-"
        contribution_str = f"{weight * mean_value:.3f}" if mean_value is not None else "-"
        lines.append(f"| {key} | {mean_str} | {weight:.2f} | {contribution_str} |")
    return "\n".join(lines) + "\n"


def distribution_md(pages: Sequence[Mapping[str, Any]]) -> str:
    """Markdown deciles of per-page ``R`` plus worst/best page id.  Numbers
    and page ids only - never image data."""
    scored = [(page.get("page_id", str(index)), float(page["R"]))
              for index, page in enumerate(pages) if page.get("R") is not None]
    if not scored:
        return "| decile | R |\n|---|---|\n\nNo scored pages.\n"
    values = sorted(r for _, r in scored)
    lines = ["| decile | R |", "|---|---|"]
    for decile in range(0, 101, 10):
        lines.append(f"| p{decile:02d} | {float(np.percentile(values, decile)):.2f} |")
    worst_id, worst_r = min(scored, key=lambda item: item[1])
    best_id, best_r = max(scored, key=lambda item: item[1])
    lines.append("")
    lines.append(f"worst: {worst_id} ({worst_r:.2f}); best: {best_id} ({best_r:.2f})")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- regression gate
# Counters that must not go from exactly zero to non-zero on a page.  `uncontained` and
# `collisions` are here because the other two were measured NOT to fire: a deliberately
# injected negative balloon margin moved total overflow 1,598 -> 2,343 px without taking a
# single page off zero, because it moved `containment` (measured against interior_core,
# which is inset by the margin) and not `overflow` (measured against the balloon outline).
# On that same defect `uncontained` fired on 10 pages, and every one of these five counters
# was bit-identical across a base/restored control pair, so none of them can fire on noise.
_HARD_INVARIANTS = (
    ("overflow", "overflow_px"),
    ("leftover", "leftover_px"),
    ("uncontained blocks", "uncontained"),
    ("panel-border collisions", "collisions"),
)


def _invariant_regressions(
    current_pages: Mapping[str, Mapping[str, Any]],
    baseline_pages: Mapping[str, Mapping[str, Any]],
    common_ids: Sequence[str],
) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []
    for page_id in common_ids:
        current_page = current_pages[page_id]
        baseline_page = baseline_pages[page_id]
        for label, field in _HARD_INVARIANTS:
            was = baseline_page.get(field)
            now = current_page.get(field)
            if was == 0 and now is not None and now > 0:
                regressions.append({"page_id": page_id, "invariant": label, "was": was, "now": now})
    return regressions


def _compare_message(
    headline_drop: bool,
    invariant_regressions: list[dict[str, Any]],
    current_mean: float | None,
    baseline_mean: float | None,
) -> str:
    if invariant_regressions:
        first = invariant_regressions[0]
        plural = "s" if len(invariant_regressions) != 1 else ""
        return (
            f"{len(invariant_regressions)} page{plural} regressed a hard invariant: "
            f"{first['page_id']} {first['invariant']} {first['was']} -> {first['now']}"
        )
    if headline_drop and current_mean is not None and baseline_mean is not None:
        drop = baseline_mean - current_mean
        return f"headline score {baseline_mean:.2f} -> {current_mean:.2f} (-{drop:.2f}, tolerance {TOLERANCE:.2f})"
    return "no regression"


def compare_baseline(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Pure regression-gate logic (the CLI and exit code live elsewhere).
    ``current`` / ``baseline`` are each a mapping with ``mean_R`` (float or
    ``None``) and ``pages`` (page id -> dict with ``overflow_px`` /
    ``leftover_px``)."""
    current_pages: Mapping[str, Mapping[str, Any]] = current.get("pages", {})
    baseline_pages: Mapping[str, Mapping[str, Any]] = baseline.get("pages", {})
    current_mean = current.get("mean_R")
    baseline_mean = baseline.get("mean_R")
    headline_drop = bool(
        current_mean is not None and baseline_mean is not None
        and current_mean < baseline_mean - TOLERANCE
    )
    common_ids = sorted(set(current_pages) & set(baseline_pages))
    only_current = sorted(set(current_pages) - set(baseline_pages))
    only_baseline = sorted(set(baseline_pages) - set(current_pages))
    regressions = _invariant_regressions(current_pages, baseline_pages, common_ids)

    return {
        # A lone page tripping an invariant is within measured run-to-run noise - see
        # INVARIANT_MIN_PAGES.  Regressions are still reported either way.
        "passed": not headline_drop and len({r["page_id"] for r in regressions}) < INVARIANT_MIN_PAGES,
        "headline_drop": headline_drop,
        "invariant_regressions": regressions,
        "only_current": only_current,
        "only_baseline": only_baseline,
        "message": _compare_message(headline_drop, regressions, current_mean, baseline_mean),
    }
