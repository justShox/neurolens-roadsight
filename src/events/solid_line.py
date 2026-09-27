"""solid_line_crossing: a vehicle moves from one side of a solid marking to the other.

A vehicle driving next to a line can have its box straddle it in perspective, and a queued vehicle
jitters around it, so an event needs the box centre to move clearly from one side to the other.
The event spans the interval in which the box straddles the line, capped around the crossing moment.
"""
from __future__ import annotations

import numpy as np

from src.events.common import ASPECT, Context, bottom_corners
from src.segments import Segment, postprocess

SIDE_WINDOW = 0.7        # s on each side of the crossing moment used to measure the offset
MIN_OFFSET = 0.006       # mean distance from the line required on both sides (aspect-corrected units)
MAX_HALF_SPAN = 3.0      # s; the event never extends further than this from the crossing moment
LINE_MARGIN = 0.05       # fraction of the segment length allowed beyond its end points
MERGE_GAP = 2.0
MIN_SEGMENT = 0.5


def _offset_and_param(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance of points to the line through a-b, and their position along the segment."""
    s = np.array([ASPECT, 1.0])
    p, a, b = p * s, a * s, b * s
    d = b - a
    length = np.linalg.norm(d)
    offset = (d[0] * (p[:, 1] - a[1]) - d[1] * (p[:, 0] - a[0])) / length
    return offset, ((p - a) @ d) / length**2


def _crossings(t: np.ndarray, centre: np.ndarray, left: np.ndarray, right: np.ndarray,
               a: np.ndarray, b: np.ndarray, step: float) -> list[Segment]:
    off_c, u = _offset_and_param(centre, a, b)
    off_l, _ = _offset_and_param(left, a, b)
    off_r, _ = _offset_and_param(right, a, b)
    near = (u > -LINE_MARGIN) & (u < 1 + LINE_MARGIN)
    straddle = near & (np.sign(off_l) != np.sign(off_r))
    out = []
    for i in np.flatnonzero(near[1:] & near[:-1] & (np.sign(off_c[1:]) != np.sign(off_c[:-1]))) + 1:
        before = off_c[(t >= t[i] - SIDE_WINDOW) & (t < t[i]) & near]
        after = off_c[(t >= t[i]) & (t <= t[i] + SIDE_WINDOW) & near]
        if not len(before) or not len(after):
            continue
        mb, ma = before.mean(), after.mean()
        if abs(mb) < MIN_OFFSET or abs(ma) < MIN_OFFSET or np.sign(mb) == np.sign(ma):
            continue
        lo, hi = i - 1, i
        while lo > 0 and straddle[lo - 1] and t[i] - t[lo - 1] <= MAX_HALF_SPAN:
            lo -= 1
        while hi < len(t) - 1 and straddle[hi + 1] and t[hi + 1] - t[i] <= MAX_HALF_SPAN:
            hi += 1
        out.append((float(t[lo]), float(t[hi] + step)))
    return out


def detect(ctx: Context) -> list[Segment]:
    tr = ctx.tracks
    left, right = bottom_corners(tr.box)
    segs: list[Segment] = []
    lines = ctx.scene.of_type("solid_line")
    veh_rows = np.flatnonzero(ctx.vehicles)
    order = veh_rows[np.lexsort((tr.frame[veh_rows], tr.track_id[veh_rows]))]
    for rows in np.split(order, np.flatnonzero(np.diff(tr.track_id[order])) + 1):
        if len(rows) < 3:
            continue
        for line in lines:
            for a, b in zip(line.points[:-1], line.points[1:]):
                segs += _crossings(tr.t[rows], ctx.foot[rows], left[rows], right[rows], a, b, ctx.timeline.step)
    return postprocess(segs, max_gap=MERGE_GAP, min_len=MIN_SEGMENT, duration=ctx.duration)
