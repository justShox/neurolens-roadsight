"""red_light: a vehicle crosses the stop line into the intersection while the signal is red.

Labels span from the first violator until the end of that red phase (vehicles still clearing).
"""
from __future__ import annotations

import numpy as np

from src import signal
from src.events.common import Context, stop_line_offset
from src.segments import Segment, postprocess

RED_BEFORE = 3.0     # s the signal must already be red (skips amber runners at the phase start)
PRE = 1.0            # s of approach included before the crossing moment
POST = 5.0           # s after the red phase ends
MERGE_GAP = 5.0
MIN_SEGMENT = 2.0
PAD_START = 0.0
PAD_END = 10.0


def crossings(ctx: Context) -> list[float]:
    """Times at which a vehicle passes the stop line on a settled red."""
    offset, alongside = stop_line_offset(ctx)
    tracks, n_before = ctx.tracks, max(1, int(round(RED_BEFORE / ctx.timeline.step)))
    rows = np.flatnonzero(ctx.vehicles & alongside)
    rows = rows[np.lexsort((tracks.frame[rows], tracks.track_id[rows]))]
    same = tracks.track_id[rows[1:]] == tracks.track_id[rows[:-1]]
    passed = same & (offset[rows[:-1]] < 0) & (offset[rows[1:]] >= 0)
    out = []
    for r in rows[1:][passed]:
        i = int(ctx.timeline.index(tracks)[r])
        if i >= n_before and (ctx.signal[i - n_before:i + 1] == signal.RED).all():
            out.append(float(tracks.t[r]))
    return out


def _red_phase_end(ctx: Context, t: float) -> float:
    """End time of the red run that covers `t` (or `t` itself if the signal is not red)."""
    i = int(np.searchsorted(ctx.timeline.times, t, side="right") - 1)
    i = max(0, min(i, len(ctx.signal) - 1))
    if ctx.signal[i] != signal.RED:
        return t
    while i + 1 < len(ctx.signal) and ctx.signal[i + 1] == signal.RED:
        i += 1
    return float(ctx.timeline.times[i] + ctx.timeline.step)


def detect(ctx: Context) -> list[Segment]:
    segs = [(t - PRE, _red_phase_end(ctx, t) + POST) for t in crossings(ctx)]
    return postprocess(
        segs,
        max_gap=MERGE_GAP,
        min_len=MIN_SEGMENT,
        duration=ctx.duration,
        pad_start=PAD_START,
        pad_end=PAD_END,
    )
