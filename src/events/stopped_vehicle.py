"""stopped_vehicle: a vehicle stands still on the road while traffic next to it keeps moving.

Traffic moving past is what separates it from a queue at the red light or a jam.
"""
from __future__ import annotations

import numpy as np

from src.events.common import Context, near_any, sustained
from src.segments import Segment, postprocess

STOP_SPEED = 0.01         # frame heights per second; below this the vehicle is standing
MOVE_SPEED = 0.03         # above this a neighbour counts as moving
MIN_STOPPED = 10.0        # s
MAX_STOPPED = 60.0        # s; longer is a parked car (or a static false detection), not an incident
PASSING_RADIUS = 0.2      # frame heights around the stopped vehicle searched for moving traffic
MIN_PASSING_SHARE = 0.3   # of the stopped time with moving traffic nearby
TRACK_GAP = 2.0
MERGE_GAP = 3.0
MIN_SEGMENT = 8.0


def flagged(ctx: Context) -> np.ndarray:
    """Track rows of vehicles standing on the road while others drive past."""
    stopped = ctx.vehicles & (ctx.speed < STOP_SPEED)
    foot = ctx.foot[stopped]
    stopped[stopped] = ctx.scene.in_any(foot, "lane") & ~ctx.scene.in_any(foot, "sidewalk")
    stopped = sustained(ctx.tracks, stopped, MIN_STOPPED, TRACK_GAP)
    passing = near_any(ctx, stopped, ctx.vehicles & (ctx.speed >= MOVE_SPEED), PASSING_RADIUS)

    out = np.zeros_like(stopped)
    ids, t = ctx.tracks.track_id, ctx.tracks.t
    for tid in np.unique(ids[stopped]):
        rows = stopped & (ids == tid)
        if t[rows].max() - t[rows].min() > MAX_STOPPED:
            continue
        if passing[rows].mean() >= MIN_PASSING_SHARE:
            out |= rows
    return out


def detect(ctx: Context) -> list[Segment]:
    mask = ctx.timeline.mask_of(ctx.tracks, flagged(ctx))
    return postprocess(ctx.timeline.segments(mask), max_gap=MERGE_GAP, min_len=MIN_SEGMENT, duration=ctx.duration)
