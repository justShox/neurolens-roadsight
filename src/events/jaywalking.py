"""jaywalking: a pedestrian on the carriageway outside a crossing."""
from __future__ import annotations

import numpy as np

from src.events.common import Context, sustained
from src.segments import Segment, postprocess

MIN_ON_ROAD = 0.5    # s the same person must stay on the road
ROAD_DEPTH = 0.012   # how far inside the road the foot point must be (people waiting at the kerb)
CLEARANCE = 0.015    # how far from any crossing / sidewalk (box bottoms overshoot the zebra edge)
TRACK_GAP = 1.0      # s of missed detections bridged inside one person's run
MERGE_GAP = 5.0      # s; labels mark whole periods of jaywalking activity, not single people
MIN_SEGMENT = 1.5


def flagged(ctx: Context) -> np.ndarray:
    """Track rows of pedestrians clearly on the road and clearly away from crossings and sidewalks."""
    rows = ctx.pedestrians.copy()
    foot = ctx.foot[rows]
    rows[rows] = ((ctx.scene.depth(foot, "road") >= ROAD_DEPTH)
                  & (ctx.scene.depth(foot, "crosswalk", "sidewalk") <= -CLEARANCE))
    return sustained(ctx.tracks, rows, MIN_ON_ROAD, TRACK_GAP)


def detect(ctx: Context) -> list[Segment]:
    segs = ctx.timeline.segments(ctx.timeline.mask_of(ctx.tracks, flagged(ctx)))
    return postprocess(segs, max_gap=MERGE_GAP, min_len=MIN_SEGMENT, duration=ctx.duration)
