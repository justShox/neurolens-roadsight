"""illegal_turn: forbidden turns at the intersection.

Thresholds are derived from the aligned scene (no_u_turn_zone, exit_zone, stop_line),
so a camera shift between videos is handled by Scene.aligned — no frame-fixed boxes.
"""
from __future__ import annotations

import numpy as np

from src.events.common import Context, stop_line_offset
from src.scene import Scene
from src.segments import Segment, postprocess

U_TURN_LEAD_SEC = 1.4
U_TURN_TRAIL_STEPS = 7
STOP_QUEUE_WAIT_SEC = 3.0
STOP_LINE_CROSS_OFFSET = -0.015
STOP_LINE_ADVANCE_OFFSET = 0.005
STOP_SPEED = 0.015

MERGE_GAP = 2.0
MIN_SEGMENT = 0.5
PAD_START = 0.0
PAD_END = 0.0


def _named_exit(scene: Scene, key: str):
    for s in scene.of_type("exit_zone"):
        if key in s.name:
            return s
    return None


def _thresholds(scene: Scene) -> dict[str, float]:
    """Geometry-relative cutoffs, equivalent to the C3896-tuned constants after alignment."""
    no_u = (scene.of_type("no_u_turn_zone") or [None])[0]
    slip = _named_exit(scene, "slip")
    stop = (scene.of_type("stop_line") or [None])[0]

    stop_x = stop.points[:, 0] if stop is not None else np.array([0.12, 0.48])
    stop_y = stop.points[:, 1] if stop is not None else np.array([0.45, 0.50])
    nou_x = no_u.points[:, 0] if no_u is not None else np.array([0.26, 0.47])
    nou_y = no_u.points[:, 1] if no_u is not None else np.array([0.84, 1.00])
    slip_x = slip.points[:, 0] if slip is not None else np.array([0.00, 0.32])
    slip_y = slip.points[:, 1] if slip is not None else np.array([0.59, 0.90])

    return {
        "u_peak_x_min": float(nou_x.max() - 0.02),
        "u_peak_x_max": float(nou_x.max() + 0.11),
        "u_peak_y_min": float(nou_y.min() - 0.14),
        "slip_exit_max_x": float(slip_x.min() + 0.30 * (slip_x.max() - slip_x.min())),
        "slip_exit_min_y": float(slip_y.min() + 0.21),
        "slip_corridor_max_x": float(slip_x.max() + 0.04),
        "inner_peak_x": float(stop_x.min() + 0.55 * (stop_x.max() - stop_x.min())),
        "origin_max_x": float(stop_x.max() - 0.03),
        "origin_max_y": float(stop_y.max() + 0.02),
        "stop_y": float(np.median(stop_y)),
        "stop_x_min": float(stop_x.min()),
    }


def detect(ctx: Context) -> list[Segment]:
    tr = ctx.tracks
    veh = ctx.vehicles
    thr = _thresholds(ctx.scene)
    no_u = (ctx.scene.of_type("no_u_turn_zone") or [None])[0]
    offsets, alongside = stop_line_offset(ctx)

    segs: list[Segment] = []
    for tid in np.unique(tr.track_id[veh]):
        m = (tr.track_id == tid) & veh
        tt = tr.t[m]
        if len(tt) < 20:
            continue
        foot = ctx.foot[m]
        # Near-side approach origin (relative to the stop line after alignment).
        if foot[0, 1] > thr["origin_max_y"] or foot[0, 0] > thr["origin_max_x"]:
            continue

        dt = np.maximum(tt[2:] - tt[:-2], 1e-4)
        speed = np.zeros(len(tt))
        speed[1:-1] = np.linalg.norm(foot[2:] - foot[:-2], axis=1) / dt
        speed[0], speed[-1] = speed[1], speed[-2]
        step = float(tt[1] - tt[0]) if len(tt) > 1 else 0.1

        # 1. U-turn through the no-U zone
        if no_u is not None and no_u.contains(foot).any():
            idx_no_u = np.flatnonzero(no_u.contains(foot))
            peak_idx = int(np.argmax(foot[: idx_no_u[-1] + 1, 0]))
            peak_x, peak_y = foot[peak_idx]
            if (thr["u_peak_x_min"] <= peak_x <= thr["u_peak_x_max"]
                    and peak_y >= thr["u_peak_y_min"]):
                i0 = max(0, peak_idx - int(U_TURN_LEAD_SEC / step))
                i1 = min(len(tt) - 1, idx_no_u[-1] + U_TURN_TRAIL_STEPS)
                segs.append((float(tt[i0]), float(tt[i1])))
                continue

        # 2. Illegal right turn into the slip
        if not (foot[-1, 0] < thr["slip_exit_max_x"] and foot[-1, 1] > thr["slip_exit_min_y"]
                and foot[:, 0].max() <= thr["slip_corridor_max_x"]):
            continue

        max_x = float(foot[:, 0].max())
        max_x_idx = int(np.argmax(foot[:, 0]))
        off_m = offsets[m]

        # 2a: from inner/straight lanes
        if max_x >= thr["inner_peak_x"] and foot[max_x_idx, 1] >= thr["stop_y"] - 0.02:
            cross = np.flatnonzero(off_m >= STOP_LINE_CROSS_OFFSET)
            if len(cross):
                done = np.flatnonzero((foot[:, 0] <= thr["slip_corridor_max_x"] * 0.7)
                                      & (foot[:, 1] >= thr["slip_exit_min_y"] - 0.15))
                t_end = tt[done[0]] if len(done) else tt[-1]
                segs.append((float(tt[cross[0]]), float(t_end)))
                continue

        # 2b: queued at the stop line, then dived into the slip
        near_stop = alongside[m] & (off_m > -0.05) & (off_m < 0.04)
        if not near_stop.any():
            near_stop = (np.abs(foot[:, 1] - thr["stop_y"]) < 0.08) & (foot[:, 0] >= thr["stop_x_min"])
        stopped = near_stop & (speed < STOP_SPEED)
        if stopped.sum() * step >= STOP_QUEUE_WAIT_SEC and max_x >= thr["inner_peak_x"] - 0.05:
            adv = np.flatnonzero(off_m >= STOP_LINE_ADVANCE_OFFSET)
            if len(adv):
                segs.append((float(tt[adv[0]]), float(tt[-1])))
                continue

    return postprocess(
        segs,
        max_gap=MERGE_GAP,
        min_len=MIN_SEGMENT,
        duration=ctx.duration,
        pad_start=PAD_START,
        pad_end=PAD_END,
    )
