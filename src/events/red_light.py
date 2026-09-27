"""red_light: a vehicle crosses the stop line into the intersection while the signal is red."""
from __future__ import annotations

import numpy as np

from src import signal
from src.events.common import Context, stop_line_offset
from src.segments import Segment, postprocess

RED_BEFORE = 1.0     # s the signal must already be red at the crossing (a yellow-to-red flip is not a violation)
PRE = 1.0            # s of approach included before the crossing moment
POST = 2.0           # s after it
MERGE_GAP = 10.0     # s; consecutive violators in one phase are labelled as one event
MIN_SEGMENT = 1.0


def crossings(ctx: Context) -> list[float]:
    """Times at which a vehicle passes the stop line on red."""
    offset, alongside = stop_line_offset(ctx)
    tracks, n_before = ctx.tracks, max(1, int(round(RED_BEFORE / ctx.timeline.step)))
    rows = np.flatnonzero(ctx.vehicles & alongside)
    rows = rows[np.lexsort((tracks.frame[rows], tracks.track_id[rows]))]
    same = tracks.track_id[rows[1:]] == tracks.track_id[rows[:-1]]
    passed = same & (offset[rows[:-1]] < 0) & (offset[rows[1:]] >= 0)
    out = []
    for r in rows[1:][passed]:
        i = ctx.timeline.index(tracks)[r]
        if i >= n_before and (ctx.signal[i - n_before:i + 1] == signal.RED).all():
            out.append(float(tracks.t[r]))
    return out


def detect(ctx: Context) -> list[Segment]:
    segs = [(t - PRE, t + POST) for t in crossings(ctx)]
    return postprocess(segs, max_gap=MERGE_GAP, min_len=MIN_SEGMENT, duration=ctx.duration)
