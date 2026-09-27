"""Shared state and helpers for the event rules."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config, signal
from src.scene import Scene
from src.segments import Segment, mask_to_segments
from src.tracking import Tracks

# A person is a rider if a two-wheeler box overlaps them this much (intersection / smaller area),
# and a passenger if this share of their box lies inside a car / bus / truck box.
RIDER_OVERLAP = 0.3
PASSENGER_OVERLAP = 0.6
ASPECT = config.ASPECT
# Speed is measured over +-SPEED_HALF_WINDOW samples of the same track to average out box jitter.
SPEED_HALF_WINDOW = 2
STOP_LINE_MARGIN = 0.05   # fraction of the stop line's length allowed beyond its end points


@dataclass(frozen=True)
class Timeline:
    """The analysed frames of a video (every STRIDE-th source frame)."""

    times: np.ndarray
    step: float

    @classmethod
    def of(cls, tracks: Tracks) -> Timeline:
        frames = np.arange(0, tracks.n_frames, config.STRIDE)
        return cls(frames / tracks.fps, config.STRIDE / tracks.fps)

    @staticmethod
    def index(tracks: Tracks) -> np.ndarray:
        """Analysed-frame index of every track row."""
        return tracks.frame // config.STRIDE

    def mask_of(self, tracks: Tracks, rows: np.ndarray) -> np.ndarray:
        """Boolean per analysed frame: True where any of the given track rows is present."""
        mask = np.zeros(len(self.times), bool)
        mask[self.index(tracks)[rows]] = True
        return mask

    def segments(self, mask: np.ndarray) -> list[Segment]:
        return mask_to_segments(self.times, mask, self.step)


@dataclass(frozen=True)
class Context:
    tracks: Tracks
    scene: Scene
    duration: float
    timeline: Timeline
    foot: np.ndarray          # (N, 2) bottom-centre of every box
    speed: np.ndarray         # (N,) foot-point speed in aspect-corrected normalised units per second
    pedestrians: np.ndarray   # (N,) rows of walking people (riders and passengers removed)
    vehicles: np.ndarray      # (N,) rows of cars, buses, trucks, motorcycles
    signal: np.ndarray        # (T,) signal state per analysed frame (src.signal constants)

    @classmethod
    def build(cls, tracks: Tracks, scene: Scene, duration: float, signal_state: np.ndarray | None = None,
              ) -> Context:
        persons = tracks.of_class("person")
        not_walking = per_track_mean(tracks.track_id, non_pedestrian_boxes(tracks)) >= 0.5
        foot = foot_points(tracks.box)
        timeline = Timeline.of(tracks)
        state = np.full(len(timeline.times), signal.UNKNOWN)
        if signal_state is not None:
            n = min(len(state), len(signal_state))
            state[:n] = signal_state[:n]
        return cls(
            tracks=tracks, scene=scene, duration=duration, timeline=timeline,
            foot=foot, speed=track_speed(tracks, foot),
            pedestrians=persons & ~not_walking,
            vehicles=tracks.of_class(*config.VEHICLE_CLASSES),
            signal=state,
        )

    def signal_at_rows(self) -> np.ndarray:
        return self.signal[self.timeline.index(self.tracks)]


def stop_line_offset(ctx: Context) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance of every foot point past the stop line (positive towards the intersection),
    and whether the point lies alongside the line rather than beyond its ends."""
    n = len(ctx.foot)
    lines = ctx.scene.of_type("stop_line")
    if not lines:
        return np.full(n, -np.inf), np.zeros(n, bool)
    scale = np.array([ASPECT, 1.0])
    a, b = lines[0].points[0] * scale, lines[0].points[-1] * scale
    u = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-u[1], u[0]])
    target = ctx.scene.of_type("intersection")
    if target and (target[0].points.mean(0) * scale - a) @ normal < 0:
        normal = -normal
    rel = ctx.foot * scale - a
    along = rel @ u / np.linalg.norm(b - a)
    return rel @ normal, (along >= -STOP_LINE_MARGIN) & (along <= 1 + STOP_LINE_MARGIN)


def foot_points(box: np.ndarray) -> np.ndarray:
    return np.stack([(box[:, 0] + box[:, 2]) / 2, box[:, 3]], axis=1)


def bottom_corners(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return box[:, [0, 3]], box[:, [2, 3]]


def track_speed(tracks: Tracks, points: np.ndarray) -> np.ndarray:
    """Central-difference speed of `points` along each track."""
    speed = np.zeros(len(points))
    p = points * [ASPECT, 1.0]
    order = np.lexsort((tracks.frame, tracks.track_id))
    k = SPEED_HALF_WINDOW
    for rows in np.split(order, np.flatnonzero(np.diff(tracks.track_id[order])) + 1):
        if len(rows) < 2:
            continue
        i = np.arange(len(rows))
        lo, hi = rows[np.maximum(i - k, 0)], rows[np.minimum(i + k, len(rows) - 1)]
        dt = np.maximum(tracks.t[hi] - tracks.t[lo], 1e-6)
        speed[rows] = np.linalg.norm(p[hi] - p[lo], axis=1) / dt
    return speed


def per_track_mean(track_id: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean of `values` over each track, broadcast back to every row."""
    _, inv = np.unique(track_id, return_inverse=True)
    return (np.bincount(inv, weights=values.astype(float)) / np.bincount(inv))[inv]


def _area(b: np.ndarray) -> np.ndarray:
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    h = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    return np.clip(w, 0, None) * np.clip(h, 0, None)


def non_pedestrian_boxes(tracks: Tracks) -> np.ndarray:
    """Person boxes that belong to a rider of a two-wheeler or a passenger inside a vehicle."""
    persons = tracks.of_class("person")
    two_wheel = tracks.of_class("bicycle", "motorcycle")
    big = tracks.of_class("car", "bus", "truck")
    out = np.zeros(len(tracks.frame), bool)
    order = np.argsort(tracks.frame, kind="stable")
    for rows in np.split(order, np.flatnonzero(np.diff(tracks.frame[order])) + 1):
        p = rows[persons[rows]]
        if not len(p):
            continue
        pb = tracks.box[p]
        if two_wheel[rows].any():
            wb = tracks.box[rows[two_wheel[rows]]]
            smaller = np.minimum(_area(pb)[:, None], _area(wb)[None, :])
            out[p[(_intersection(pb, wb) / (smaller + 1e-9) > RIDER_OVERLAP).any(1)]] = True
        if big[rows].any():
            vb = tracks.box[rows[big[rows]]]
            out[p[(_intersection(pb, vb) / (_area(pb)[:, None] + 1e-9) > PASSENGER_OVERLAP).any(1)]] = True
    return out


def near_any(ctx: Context, rows: np.ndarray, others: np.ndarray, radius: float) -> np.ndarray:
    """Rows of `rows` whose foot point is within `radius` of some `others` row in the same frame."""
    idx = ctx.timeline.index(ctx.tracks)
    out = np.zeros(len(rows), bool)
    p = ctx.foot * [ASPECT, 1.0]
    a_all, b_all = np.flatnonzero(rows), np.flatnonzero(others)
    a_all, b_all = a_all[np.argsort(idx[a_all], kind="stable")], b_all[np.argsort(idx[b_all], kind="stable")]
    a_idx, b_idx = idx[a_all], idx[b_all]
    for f in np.intersect1d(a_idx, b_idx):
        a = a_all[np.searchsorted(a_idx, f):np.searchsorted(a_idx, f, "right")]
        b = b_all[np.searchsorted(b_idx, f):np.searchsorted(b_idx, f, "right")]
        d = np.linalg.norm(p[a, None] - p[None, b], axis=-1)
        d[a[:, None] == b[None, :]] = np.inf
        out[a[(d <= radius).any(1)]] = True
    return out


def sustained(tracks: Tracks, flags: np.ndarray, min_duration: float, max_gap: float) -> np.ndarray:
    """Keep a flag only where the same track stays flagged for at least `min_duration` seconds.

    Gaps up to `max_gap` inside a run (missed detections, short occlusions) are bridged and filled.
    """
    out = np.zeros(len(flags), bool)
    order = np.lexsort((tracks.frame, tracks.track_id))
    t, f = tracks.t[order], flags[order]
    for local in np.split(np.arange(len(order)), np.flatnonzero(np.diff(tracks.track_id[order])) + 1):
        pos = local[f[local]]
        if not len(pos):
            continue
        for run in np.split(pos, np.flatnonzero(np.diff(t[pos]) > max_gap) + 1):
            if t[run[-1]] - t[run[0]] >= min_duration:
                out[order[run[0]:run[-1] + 1]] = True
    return out
