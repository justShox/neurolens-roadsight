"""stop_line: a vehicle waits on red with its front past the stop line (scene/camera.md).

The event runs from max(vehicle stops, signal turns red) until the vehicle leaves, i.e. the green.
"""
from __future__ import annotations

import numpy as np

from src import signal
from src.events.common import Context, stop_line_offset, sustained
from src.segments import Segment, postprocess

STOP_SPEED = 0.01    # frame heights per second
MAX_PAST = 0.06      # frame heights past the line; further on the vehicle is inside the intersection
MIN_WAIT = 2.0       # s
TRACK_GAP = 1.0
MERGE_GAP = 3.0
MIN_SEGMENT = 3.0
PAD_START = 0.0
PAD_END = 0.0


def flagged(ctx: Context) -> np.ndarray:
    offset, alongside = stop_line_offset(ctx)
    rows = (ctx.vehicles & alongside & (offset > 0) & (offset <= MAX_PAST)
            & (ctx.speed < STOP_SPEED) & (ctx.signal_at_rows() == signal.RED))
    return sustained(ctx.tracks, rows, MIN_WAIT, TRACK_GAP)


def detect(ctx: Context) -> list[Segment]:
    mask = ctx.timeline.mask_of(ctx.tracks, flagged(ctx))
    return postprocess(
        ctx.timeline.segments(mask),
        max_gap=MERGE_GAP,
        min_len=MIN_SEGMENT,
        duration=ctx.duration,
        pad_start=PAD_START,
        pad_end=PAD_END,
    )

