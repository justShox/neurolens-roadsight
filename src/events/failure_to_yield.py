"""failure_to_yield: a vehicle drives across a crossing while a pedestrian is on the same crossing."""
from __future__ import annotations

import numpy as np

from src import signal
from src.events.common import Context, near_any, sustained
from src.segments import Segment, postprocess

MIN_PRESENCE = 0.5        # s a person / vehicle must stay on the crossing to count
MIN_VEHICLE_SPEED = 0.0   # includes stopped vehicles blocking/cutting off pedestrians
MAX_DISTANCE = 0.16       # frame heights between pedestrian and vehicle foot points
TRACK_GAP = 1.0
MERGE_GAP = 0.5
MIN_SEGMENT = 0.8
PAD_START = 0.5
PAD_END = 0.0


def flagged(ctx: Context) -> tuple[np.ndarray, np.ndarray]:
    """(per-frame mask, contributing track rows) over all crossings."""
    mask = np.zeros(len(ctx.timeline.times), bool)
    rows = np.zeros(len(ctx.foot), bool)
    sig_rows = ctx.signal_at_rows()
    for crossing in ctx.scene.of_type("crosswalk"):
        peds = ctx.pedestrians.copy()
        peds[peds] = crossing.contains(ctx.foot[peds])
        vehs = ctx.vehicles.copy()
        if MIN_VEHICLE_SPEED > 0:
            vehs &= (ctx.speed >= MIN_VEHICLE_SPEED)
        if crossing.name in ("left_crosswalk", "right_crosswalk"):
            vehs &= (sig_rows == signal.RED)
        vehs[vehs] = crossing.contains(ctx.foot[vehs])
        peds = sustained(ctx.tracks, peds, MIN_PRESENCE, TRACK_GAP)
        vehs = sustained(ctx.tracks, vehs, MIN_PRESENCE, TRACK_GAP)
        both = ctx.timeline.mask_of(ctx.tracks, near_any(ctx, peds, vehs, MAX_DISTANCE))
        mask |= both
        rows |= (peds | vehs) & both[ctx.timeline.index(ctx.tracks)]
    return mask, rows


def detect(ctx: Context) -> list[Segment]:
    mask, _ = flagged(ctx)
    return postprocess(
        ctx.timeline.segments(mask),
        max_gap=MERGE_GAP,
        min_len=MIN_SEGMENT,
        duration=ctx.duration,
        pad_start=PAD_START,
        pad_end=PAD_END,
    )

