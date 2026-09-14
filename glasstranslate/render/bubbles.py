"""Which balloon a block's text sits in.

Artists join two speech balloons by erasing the arc they share, and a balloon
whose outline opens anywhere - a tail, a gap the pen left, a panel edge it
runs into - spills into the paper behind it.  The paper map of :mod:`.layout`
therefore hands back one component for two balloons, or for a balloon and the
whole page around it; the bubble gate, asked about that component, calls it
panel background and letters every block on it as free text over the artwork.

This module cuts such a component back into one interior per block:

1. **Rooms** - paper reachable only through a passage narrower than
   ``_NECK_MAX_EM`` ems of the block's *own* characters is not part of that
   block's room.  The passages are closed by keeping only the paper deeper
   than half that width (the distance transform), labelling what survives -
   each survivor is a room's *core* - and letting every core claim back the
   paper within the same distance of it.  A balloon that leaks into the page
   keeps its own room and the page keeps the rest.  The sealing never goes
   deeper than the paper the block's own text already has around it, so a
   balloon barely roomier than its text survives it.
2. **Co-tenants** - blocks that are in each other's rooms really do share one
   patch of paper: two speakers in one balloon, or balloons overlapping too
   far to seal apart.  Their room is divided at its waist - the paper deeper
   than *level* is taken for a rising series of levels, and the first level
   that leaves each block's text on a patch of its own is the cut; each block
   then claims the paper nearest its own patch.  When no level separates them
   they are in one balloon, and each keeps the paper nearest its own text.
   A block whose room merely swallows a neighbour's balloon - the page behind
   it does - is *not* a co-tenant: the neighbour's room does not hold this
   block's text.
3. **Disjointness** - wherever two rooms still overlap, and wherever a room
   reaches over another block's source footprint, each keeps only the paper
   nearer to its own text.

The result per block is an :class:`Interior`, which :mod:`.layout` puts
through the bubble gate.  Two blocks' interiors never overlap, and each is a
single connected patch of paper: the lettering is anchored on the region's
inscribed circle, and a mask in two pieces would aim at the wrong balloon.
The one thing no division can promise is a block's interior keeping clear of
a neighbour's *characters* when the two OCR boxes physically overlap - one
utterance the grouping read as two blocks (1ja 8 and 9, an ellipsis whose box
runs into the column beside it): the paper they share is at no distance from
either, and it goes to the first of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.types import Rect

# Paper reached only through a passage narrower than this many ems of the
# block's own characters belongs to some other room.  A balloon's tail and
# the gap an open outline leaves are about one character across on the
# reference pages, and balloons themselves run five ems and more; 1.5 em
# seals the first without touching the second.
_NECK_MAX_EM = 1.5
# Where to look for the waist between two balloons sharing a room, as a
# fraction of the paper the tightest of them gives its own text.  The first
# level that puts the blocks on patches of their own is the cut, so a short
# ladder is enough; past the top of it every patch would be gone and the
# blocks are in one balloon, not two.
_WAIST_LEVELS = (0.35, 0.5, 0.65, 0.8)
# How much of a block's source footprint has to lie in a room before that
# room counts as holding the block.
_COVER = 0.5
_RIM = np.ones((3, 3), np.uint8)  # the one-pixel collar :func:`_measure_walls` looks at


@dataclass
class Interior:
    """One block's own patch of paper: ``region`` is a boolean mask of shape
    ``(rect.h, rect.w)``, indexed ``[y - rect.y, x - rect.x]`` in page pixels,
    ``area`` its pixel count and ``walled`` the share of its edge that is a
    wall rather than paper the cut left behind (see :func:`_measure_walls`)."""

    rect: Rect
    region: np.ndarray
    area: int
    walled: float = 1.0

    def solidity(self) -> float:
        """Area of the region over the area of its convex hull: balloons are
        convex-ish, the paper around the artwork is not.  Tracing the hull of
        a page-sized region is not cheap, so this is measured on demand, once
        the size gates have had their say."""
        contours, _ = cv2.findContours(self.region.view(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        hull_area = float(cv2.contourArea(cv2.convexHull(np.concatenate(contours))))
        return self.area / hull_area if hull_area > 0 else 0.0


def interiors(
    region: np.ndarray, origin: Tuple[int, int], sources: Sequence[Rect], ems: Sequence[float]
) -> List[Optional[Interior]]:
    """One interior per source footprint, cut out of the paper component
    ``region`` (a 0/1 map whose top-left is ``origin`` in page pixels, with
    ``ems`` the blocks' source character ems); None where a block's text is
    in no room of its own.  See the module docstring."""
    paper = region.view(bool)
    dist = cv2.distanceTransform(region, cv2.DIST_L2, 3)
    deep = [_deepest(dist, source, origin) for source in sources]
    sealed: Dict[int, _Seal] = {}
    rooms = [_room(paper, dist, at, em, sealed) for at, em in zip(deep, ems)]
    for group in _cotenants(rooms, sources, origin):
        shared = min((rooms[i] for i in group), key=lambda room: int(room.sum()))
        for i, part in zip(group, _divide(shared, [sources[i] for i in group], origin)):
            rooms[i] = part
    _disjoin(rooms, sources, origin)
    out = [None if room is None else _piece(room, source, origin) for room, source in zip(rooms, sources)]
    _measure_walls(out, paper, origin)
    return out


def inset(region: np.ndarray, margin: int) -> np.ndarray:
    """``region`` pulled in by ``margin`` pixels on every side - the paper a
    balloon keeps between its outline and the lettering.

    This is an erosion by a Euclidean disc, done as a threshold on the
    distance transform rather than with a structuring element, and the
    difference matters.  The lettering is anchored on the region's inscribed
    circle, and by definition ``distance(inset) == distance(region) - margin``
    wherever the inset survives, so thresholding cannot move that circle: the
    block is anchored where the *balloon* is widest, not where the inset
    happens to be.  A ``MORPH_ELLIPSE`` erosion only approximates the disc,
    and on a balloon with two nearly equal inscribed circles - 1ja block 15,
    whose two candidates are 26.8 and 26.6 px - the approximation is enough to
    swap them and move the anchor 1.4 ems.

    The region is padded with background first: it always touches its own
    bounding box, and without the padding the bounding box's own edges (a
    caption box's outer sides) would not be pulled in at all."""
    padded = np.pad(region.view(np.uint8), 1)
    return (cv2.distanceTransform(padded, cv2.DIST_L2, 5) > margin)[1:-1, 1:-1]


# --------------------------------------------------------------- geometry
def _window(rect: Rect, origin: Tuple[int, int], shape: Tuple[int, ...]) -> Tuple[slice, slice]:
    """``rect`` in page pixels as a slice pair into an array at ``origin``."""
    ox, oy = origin
    y0, y1 = max(0, rect.y - oy), min(int(shape[0]), rect.y2 - oy)
    x0, x1 = max(0, rect.x - ox), min(int(shape[1]), rect.x2 - ox)
    return slice(y0, max(y0, y1)), slice(x0, max(x0, x1))


def _deepest(dist: np.ndarray, rect: Rect, origin: Tuple[int, int]) -> Tuple[float, int, int]:
    """The deepest paper under a source footprint as ``(depth, y, x)``: how
    much room the block's text has around it, and where.  Every patch of
    paper the block could be in holds this pixel, so reading a label there
    says which patch is the block's without a nearest-patch map."""
    win = _window(rect, origin, dist.shape)
    sub = dist[win]
    if not sub.size:
        return 0.0, 0, 0
    y, x = np.unravel_index(int(sub.argmax()), sub.shape)
    return float(sub[y, x]), int(y) + win[0].start, int(x) + win[1].start


def _closest(shape: Tuple[int, ...], sources: Sequence[Rect], origin: Tuple[int, int]) -> np.ndarray:
    """Index of the source footprint nearest to each pixel.

    One transform for all of them where that is exact: footprints that do not
    touch are separate patches in one seed image, and the nearest-patch
    labels that come back with the distances *are* the nearest-footprint
    map.  Footprints that do touch - one utterance the OCR read as two
    overlapping lines - would share a patch and a label, so those fall back
    to one transform each, where the earlier index keeps the ties."""
    if not any(a.intersects(b) for i, a in enumerate(sources) for b in sources[i + 1 :]):
        seed = np.full(shape[:2], 255, np.uint8)
        for source in sources:
            seed[_window(source, origin, shape)] = 0
        _, near = cv2.distanceTransformWithLabels(seed, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP)
        owner = np.zeros(shape[:2], np.uint8)
        for i, source in enumerate(sources):
            patch = near[_window(source, origin, shape)]
            if patch.size:
                owner[near == int(np.bincount(patch.ravel()).argmax())] = i
        return owner
    owner = np.zeros(shape[:2], np.uint8)
    best: Optional[np.ndarray] = None
    for i, source in enumerate(sources):
        seed = np.full(shape[:2], 255, np.uint8)
        seed[_window(source, origin, shape)] = 0
        apart = cv2.distanceTransform(seed, cv2.DIST_L2, 3)
        if best is None:
            best = apart
            continue
        owner[apart < best] = i
        np.minimum(best, apart, out=best)
    return owner


# ------------------------------------------------------------------ rooms
@dataclass
class _Seal:
    """The component sealed at one neck width.  ``single`` is the whole room
    when nothing came apart at that width; otherwise ``near`` labels every
    pixel with the room it is in and 0 where no room reaches."""

    single: Optional[np.ndarray]
    near: Optional[np.ndarray]


def _seal(paper: np.ndarray, dist: np.ndarray, neck: int) -> _Seal:
    """Close every passage narrower than ``2 * neck``: the paper deeper than
    ``neck`` is what survives, each surviving patch is a room's core, and
    every core claims back the paper within ``neck`` of it."""
    core = (dist > neck).view(np.uint8)
    count, _ = cv2.connectedComponents(core, connectivity=8)
    if count < 2:
        return _Seal(paper, None)
    if count == 2:
        # One core: the component opens into a single room, and the cheaper
        # transform (no labels) is enough to grow it back.
        reach = cv2.distanceTransform(1 - core, cv2.DIST_L2, 3)
        return _Seal((reach <= neck) & paper, None)
    reach, near = cv2.distanceTransformWithLabels(1 - core, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP)
    near[(reach > neck) | ~paper] = 0
    return _Seal(None, near)


def _room(
    paper: np.ndarray, dist: np.ndarray, at: Tuple[float, int, int], em: float, sealed: Dict[int, _Seal]
) -> Optional[np.ndarray]:
    """The block's own room: paper it can reach without going through a
    passage narrower than ``_NECK_MAX_EM`` of its own ems (step 1 of the
    module docstring).  Blocks that seal at the same width share the pass."""
    depth, y, x = at
    neck = int(min(_NECK_MAX_EM * em / 2.0, max(0.0, depth - 1.0)))
    if neck < 1:
        return paper
    seal = sealed.get(neck)
    if seal is None:
        seal = sealed[neck] = _seal(paper, dist, neck)
    if seal.near is None:
        return seal.single
    label = int(seal.near[y, x])
    return (seal.near == label) if label else None


def _covers(room: np.ndarray, source: Rect, origin: Tuple[int, int]) -> bool:
    """Is most of the block's source footprint in this room?"""
    inside = room[_window(source, origin, room.shape)]
    return bool(inside.size) and int(inside.sum()) >= _COVER * inside.size


# -------------------------------------------------------------- neighbours
def _cotenants(rooms: Sequence[Optional[np.ndarray]], sources: Sequence[Rect], origin: Tuple[int, int]) -> List[List[int]]:
    """Groups of blocks that are in each other's rooms; see step 2 of the
    module docstring.  Blocks with a room to themselves are not returned."""
    parent = {i: i for i, room in enumerate(rooms) if room is not None}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    live = sorted(parent)
    for pos, i in enumerate(live):
        for j in live[pos + 1 :]:
            mutual = _covers(rooms[i], sources[j], origin) and _covers(rooms[j], sources[i], origin)
            if mutual and find(i) != find(j):
                parent[find(j)] = find(i)
    groups: Dict[int, List[int]] = {}
    for i in live:
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def _divide(room: np.ndarray, sources: Sequence[Rect], origin: Tuple[int, int]) -> List[np.ndarray]:
    """Split one room between its co-tenants at its waist.  Whether a level
    separates them is read off the cheap connected components of the paper
    deeper than it; only a level that does pays for a nearest-patch map.  A
    cut that runs past somebody's text - handing one block the paper another
    block's characters are printed on - is no cut at all, and the ladder goes
    on without it."""
    dist = cv2.distanceTransform(room.view(np.uint8), cv2.DIST_L2, 3)
    deep = [_deepest(dist, source, origin) for source in sources]
    clearance = min(depth for depth, _, _ in deep)
    for level in _WAIST_LEVELS:
        core = (dist > level * clearance).view(np.uint8)
        count, patches = cv2.connectedComponents(core, connectivity=8)
        if count <= len(sources):
            continue
        owned = [int(patches[y, x]) for _, y, x in deep]
        if 0 in owned or len(set(owned)) != len(owned):
            continue
        _, near = cv2.distanceTransformWithLabels(1 - core, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP)
        parts = [(near == near[y, x]) & room for _, y, x in deep]
        if _apart(parts, sources, origin):
            return parts
    # Nothing separated them: one balloon, and each keeps what is nearest to
    # its own text - which never leaves a block's characters on another's
    # paper, since a block's own footprint is at no distance from itself.
    owner = _closest(room.shape, sources, origin)
    return [(owner == i) & room for i in range(len(sources))]


def _apart(parts: Sequence[np.ndarray], sources: Sequence[Rect], origin: Tuple[int, int]) -> bool:
    """Does every part keep clear of the other blocks' source footprints?"""
    for i, part in enumerate(parts):
        for j, source in enumerate(sources):
            if i != j and part[_window(source, origin, part.shape)].any():
                return False
    return True


def _disjoin(rooms: List[Optional[np.ndarray]], sources: Sequence[Rect], origin: Tuple[int, int]) -> None:
    """Step 3: where two rooms still overlap - a balloon inside the room the
    page behind it makes - and wherever a room reaches over another block's
    source footprint, each block keeps only the paper nearer to its own
    text."""
    live = [i for i, room in enumerate(rooms) if room is not None]
    if not live:
        return
    shape = rooms[live[0]].shape
    claimed = np.zeros(shape, np.uint8)
    for i in live:
        claimed += rooms[i].view(np.uint8)
    for j, source in enumerate(sources):
        win = _window(source, origin, shape)
        if any(i != j and rooms[i][win].any() for i in live):
            claimed[win] += 1
    contested = claimed > 1
    if not contested.any():
        return
    owner = _closest(shape, sources, origin)
    for i in live:
        rooms[i] = rooms[i] & (~contested | (owner == i))


def _measure_walls(out: Sequence[Optional[Interior]], paper: np.ndarray, origin: Tuple[int, int]) -> None:
    """Fill in each interior's ``walled``: the share of its edge that is a
    wall - its own outline, the artwork, or a neighbour's interior - rather
    than paper of the same component that the cut left to nobody.

    A balloon leaks into the page through a tail or a gap the pen left and is
    walled in everywhere else.  A room cut out of the paper a panel is drawn
    on - the white between the beams of a scaffold, a slice of a flat panel -
    is open on most of its sides, because nothing but the cut put it there.
    That is the one thing the size and convexity gates cannot see: both rooms
    are convex blobs of about the right size."""
    ox, oy = origin
    live = [i for i, it in enumerate(out) if it is not None]
    if not live:
        return
    claimed = np.zeros(paper.shape, bool)
    for i in live:
        rect = out[i].rect
        claimed[rect.y - oy : rect.y2 - oy, rect.x - ox : rect.x2 - ox] |= out[i].region
    for i in live:
        interior = out[i]
        rect = interior.rect
        y0, x0 = max(0, rect.y - oy - 1), max(0, rect.x - ox - 1)
        y1, x1 = min(paper.shape[0], rect.y2 - oy + 1), min(paper.shape[1], rect.x2 - ox + 1)
        own = np.zeros((y1 - y0, x1 - x0), bool)
        own[rect.y - oy - y0 : rect.y2 - oy - y0, rect.x - ox - x0 : rect.x2 - ox - x0] = interior.region
        rim = cv2.dilate(own.view(np.uint8), _RIM).astype(bool) & ~own
        edge = int(rim.sum())
        if edge:
            interior.walled = 1.0 - int((rim & paper[y0:y1, x0:x1] & ~claimed[y0:y1, x0:x1]).sum()) / edge


def _label_of(labels: np.ndarray, source: Rect, origin: Tuple[int, int]) -> int:
    """The labelled patch most of the block's source footprint is on, 0 when
    it is on none of them."""
    inside = labels[_window(source, origin, labels.shape)]
    inside = inside[inside > 0]
    return int(np.bincount(inside.ravel()).argmax()) if inside.size else 0


def one_piece(mask: np.ndarray, source: Rect, origin: Tuple[int, int]) -> Optional[np.ndarray]:
    """``mask`` cut down to the single connected patch the block's text is
    on, or to its largest patch when the text is on none of them; None when
    the mask is empty.  The lettering is anchored on the region's inscribed
    circle, so a mask in two pieces would aim at the wrong one."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.view(np.uint8), connectivity=4)
    if count < 2:
        return None
    label = _label_of(labels, source, origin)
    if label == 0:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def _piece(part: np.ndarray, source: Rect, origin: Tuple[int, int]) -> Optional[Interior]:
    """The one connected patch of ``part`` that holds the block's text."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(part.view(np.uint8), connectivity=4)
    if count < 2:
        return None
    best = _label_of(labels, source, origin)
    if best == 0:
        return None
    x, y, w, h, area = (int(v) for v in stats[best][:5])
    ox, oy = origin
    return Interior(Rect(x + ox, y + oy, w, h), labels[y : y + h, x : x + w] == best, area)
