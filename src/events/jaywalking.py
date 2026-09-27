"""jaywalking: a pedestrian on the carriageway outside a crossing."""
from __future__ import annotations

import numpy as np

from src.events.common import Context, sustained
from src.segments import Segment, postprocess, trim_sparse

MIN_ON_ROAD = 0.5    # s the same person must stay on the road
ROAD_DEPTH = 0.012   # how far inside the road the foot point must be (people waiting at the kerb)
CLEARANCE = 0.015    # how far from any crossing / sidewalk (box bottoms overshoot the zebra edge)
TRACK_GAP = 1.0      # s of missed detections bridged inside one person's run
MERGE_GAP = 8.0      # s; labels mark whole periods of jaywalking activity, not single people
MIN_SEGMENT = 2.0
PAD_START = 0.0
PAD_END = 0.0
TRIM_WINDOW = 2.0    # s; edges must stay this dense or they are trimmed
TRIM_DENSITY = 0.40  # fraction of flagged frames required inside TRIM_WINDOW


def flagged(ctx: Context) -> np.ndarray:
    """Track rows of pedestrians clearly on the road and clearly away from crossings and sidewalks."""
    rows = ctx.pedestrians.copy()
    foot = ctx.foot[rows]
    rows[rows] = ((ctx.scene.depth(foot, "road") >= ROAD_DEPTH)
                  & (ctx.scene.depth(foot, "crosswalk", "sidewalk") <= -CLEARANCE))
    return sustained(ctx.tracks, rows, MIN_ON_ROAD, TRACK_GAP)


def detect(ctx: Context) -> list[Segment]:
    mask = ctx.timeline.mask_of(ctx.tracks, flagged(ctx))
    segs = postprocess(
        ctx.timeline.segments(mask),
        max_gap=MERGE_GAP,
        min_len=MIN_SEGMENT,
        duration=ctx.duration,
        pad_start=PAD_START,
        pad_end=PAD_END,
    )
    return trim_sparse(ctx.timeline.times, mask, segs, min_density=TRIM_DENSITY, window=TRIM_WINDOW)
