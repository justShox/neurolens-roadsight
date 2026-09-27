"""illegal_turn: forbidden turns at the intersection.

Two main categories:
1. U-turn or forbidden left turn across crosswalk_illegal (no_u_turn_zone).
2. Forbidden right turn into the slip lane from inner/straight lanes or from
   the main stop line after queuing (missing the dedicated pre-stop slip entrance).
"""
from __future__ import annotations

import numpy as np

from src.events.common import Context, stop_line_offset
from src.segments import Segment, postprocess

# Bounds for U-turn peak in the intersection before turning counter-clockwise
U_TURN_PEAK_X_MIN = 0.45
U_TURN_PEAK_X_MAX = 0.58
U_TURN_PEAK_Y_MIN = 0.70
U_TURN_LEAD_SEC = 1.4
U_TURN_TRAIL_STEPS = 7

# Geometry and thresholds for illegal slip road turns
SLIP_CORRIDOR_MAX_X = 0.36
SLIP_EXIT_MAX_X = 0.10
SLIP_EXIT_MIN_Y = 0.80
INNER_LANE_PEAK_X = 0.30
STOP_QUEUE_WAIT_SEC = 3.0
STOP_LINE_CROSS_OFFSET = -0.015
STOP_LINE_ADVANCE_OFFSET = 0.005

MERGE_GAP = 2.0
MIN_SEGMENT = 0.5


def detect(ctx: Context) -> list[Segment]:
    tr = ctx.tracks
    veh = ctx.vehicles
    tids = np.unique(tr.track_id[veh])
    segs: list[Segment] = []

    no_u_list = ctx.scene.of_type("no_u_turn_zone")
    no_u = no_u_list[0] if no_u_list else None
    offsets, _ = stop_line_offset(ctx)

    for tid in tids:
        m = (tr.track_id == tid) & veh
        tt = tr.t[m]
        if len(tt) < 20:
            continue
        foot = ctx.foot[m]

        # Vehicle must originate in the camera-side approach road
        if foot[0, 1] > 0.50 or foot[0, 0] > 0.45:
            continue

        dt = np.maximum(tt[2:] - tt[:-2], 1e-4)
        dp = np.linalg.norm(foot[2:] - foot[:-2], axis=1)
        v = np.zeros(len(tt))
        v[1:-1] = dp / dt
        v[0], v[-1] = v[1], v[-2]

        # 1. U-turn in intersection across no_u_turn_zone
        if no_u and no_u.contains(foot).any():
            idx_no_u = np.flatnonzero(no_u.contains(foot))
            peak_x_idx = np.argmax(foot[:idx_no_u[-1] + 1, 0])
            peak_x = foot[peak_x_idx, 0]
            peak_y = foot[peak_x_idx, 1]
            if U_TURN_PEAK_X_MIN <= peak_x <= U_TURN_PEAK_X_MAX and peak_y >= U_TURN_PEAK_Y_MIN:
                step = tt[1] - tt[0] if len(tt) > 1 else 0.1
                idx_start = max(0, peak_x_idx - int(U_TURN_LEAD_SEC / step))
                t_start = tt[idx_start]
                t_end = tt[min(len(tt) - 1, idx_no_u[-1] + U_TURN_TRAIL_STEPS)]
                segs.append((float(t_start), float(t_end)))
                continue

        # 2. Illegal right turn into slip lane
        if (foot[-1, 0] < SLIP_EXIT_MAX_X and foot[-1, 1] > SLIP_EXIT_MIN_Y
                and foot[:, 0].max() <= SLIP_CORRIDOR_MAX_X):
            max_x = foot[:, 0].max()
            max_x_idx = np.argmax(foot[:, 0])

            # Case 2a: From inner/straight lanes (lane 3-5)
            if max_x >= INNER_LANE_PEAK_X and foot[max_x_idx, 1] >= 0.48:
                off_m = offsets[m]
                cross_sub = np.flatnonzero(off_m >= STOP_LINE_CROSS_OFFSET)
                if len(cross_sub):
                    t_start = tt[cross_sub[0]]
                    slip_done = np.flatnonzero((foot[:, 0] <= 0.22) & (foot[:, 1] >= 0.65))
                    t_end = tt[slip_done[0]] if len(slip_done) else tt[-1]
                    segs.append((float(t_start), float(t_end)))
                    continue

            # Case 2b: From stop line after queuing (missed pre-stop slip entry)
            at_stop = (foot[:, 1] >= 0.46) & (foot[:, 0] >= 0.16)
            stopped = at_stop & (v < 0.015)
            step = tt[1] - tt[0] if len(tt) > 1 else 0.1
            if stopped.sum() * step >= STOP_QUEUE_WAIT_SEC and max_x >= 0.25:
                off_m = offsets[m]
                adv_sub = np.flatnonzero(off_m >= STOP_LINE_ADVANCE_OFFSET)
                if len(adv_sub):
                    t_start = tt[adv_sub[0]]
                    t_end = tt[-1]
                    segs.append((float(t_start), float(t_end)))
                    continue

    return postprocess(segs, max_gap=MERGE_GAP, min_len=MIN_SEGMENT, duration=ctx.duration)
