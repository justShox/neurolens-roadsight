"""failure_to_yield: a vehicle drives across a crossing while a pedestrian is on the same crossing."""
from __future__ import annotations

import numpy as np

from src.events.common import Context, near_any, sustained
from src.segments import Segment, postprocess

MIN_PRESENCE = 0.3        # s a person / vehicle must stay on the crossing to count
MIN_VEHICLE_SPEED = 0.03  # aspect-corrected frame heights per second; below this the vehicle is waiting
MAX_DISTANCE = 0.08       # frame heights between the pedestrian's and the vehicle's foot points
TRACK_GAP = 1.0
MERGE_GAP = 2.0
MIN_SEGMENT = 0.5


def flagged(ctx: Context) -> tuple[np.ndarray, np.ndarray]:
    """(per-frame mask, contributing track rows) over all crossings."""
    mask = np.zeros(len(ctx.timeline.times), bool)
    rows = np.zeros(len(ctx.foot), bool)
    for crossing in ctx.scene.of_type("crosswalk"):
        peds = ctx.pedestrians.copy()
        peds[peds] = crossing.contains(ctx.foot[peds])
        vehs = ctx.vehicles & (ctx.speed >= MIN_VEHICLE_SPEED)
        vehs[vehs] = crossing.contains(ctx.foot[vehs])
        peds = sustained(ctx.tracks, peds, MIN_PRESENCE, TRACK_GAP)
        vehs = sustained(ctx.tracks, vehs, MIN_PRESENCE, TRACK_GAP)
        both = ctx.timeline.mask_of(ctx.tracks, near_any(ctx, peds, vehs, MAX_DISTANCE))
        mask |= both
        rows |= (peds | vehs) & both[ctx.timeline.index(ctx.tracks)]
    return mask, rows


def detect(ctx: Context) -> list[Segment]:
    mask, _ = flagged(ctx)
    return postprocess(ctx.timeline.segments(mask), max_gap=MERGE_GAP, min_len=MIN_SEGMENT, duration=ctx.duration)
