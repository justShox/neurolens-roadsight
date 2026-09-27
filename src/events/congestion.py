"""congestion: nearly every vehicle in the lanes of one direction stands still for a long time."""
from __future__ import annotations

import numpy as np

from src import signal
from src.events.common import Context
from src.scene import Shape
from src.segments import Segment, postprocess

STOP_SPEED = 0.01          # frame heights per second
MIN_VEHICLES = 4           # vehicles in the direction's lanes for a frame to count
MIN_STOPPED_SHARE = 0.8
SAME_DIRECTION = 0.7       # cosine between lane arrows grouped into one direction
MERGE_GAP = 3.0
MIN_SEGMENT = 10.0         # s
MIN_GREEN = 3.0            # s of green during which the jam must stay stopped


def directions(lanes: list[Shape]) -> list[list[Shape]]:
    """Group lanes whose arrows point the same way."""
    groups: list[list[Shape]] = []
    for lane in lanes:
        for g in groups:
            if g[0].unit_direction() @ lane.unit_direction() >= SAME_DIRECTION:
                g.append(lane)
                break
        else:
            groups.append([lane])
    return groups


def flagged(ctx: Context) -> np.ndarray:
    """Per-frame mask of jammed directions."""
    n = len(ctx.timeline.times)
    idx = ctx.timeline.index(ctx.tracks)
    mask = np.zeros(n, bool)
    for group in directions([s for s in ctx.scene.of_type("lane") if s.direction is not None]):
        rows = ctx.vehicles.copy()
        inside = np.zeros(rows.sum(), bool)
        for lane in group:
            inside |= lane.contains(ctx.foot[rows])
        rows[rows] = inside
        total = np.bincount(idx[rows], minlength=n)
        stopped = np.bincount(idx[rows & (ctx.speed < STOP_SPEED)], minlength=n)
        mask |= (total >= MIN_VEHICLES) & (stopped >= MIN_STOPPED_SHARE * total)
    return mask


def detect(ctx: Context) -> list[Segment]:
    """Jams that stay put through a green phase; a queue that clears on green is just a red light."""
    mask = flagged(ctx)
    green = (ctx.signal == signal.GREEN) & mask
    segs = postprocess(ctx.timeline.segments(mask), max_gap=MERGE_GAP, min_len=MIN_SEGMENT, duration=ctx.duration)
    t = ctx.timeline.times
    return [(s, e) for s, e in segs if green[(t >= s) & (t < e)].sum() * ctx.timeline.step >= MIN_GREEN]
