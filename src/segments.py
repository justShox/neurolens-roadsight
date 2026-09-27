"""Turn per-frame boolean flags into clean [start, end] segments."""
from __future__ import annotations

import numpy as np

Segment = tuple[float, float]


def mask_to_segments(times: np.ndarray, mask: np.ndarray, step: float) -> list[Segment]:
    """Contiguous runs of True. A run covering samples t_i..t_j spans [t_i, t_j + step]."""
    times, mask = np.asarray(times, float), np.asarray(mask, bool)
    if not mask.any():
        return []
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return [(float(times[s]), float(times[e] + step)) for s, e in zip(starts, ends)]


def merge_close(segs: list[Segment], max_gap: float) -> list[Segment]:
    """Merge segments that overlap or are separated by less than `max_gap` seconds."""
    out: list[list[float]] = []
    for s, e in sorted(segs):
        if out and s - out[-1][1] < max_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def postprocess(segs: list[Segment], *, max_gap: float, min_len: float, duration: float,
                pad_start: float = 0.0, pad_end: float = 0.0) -> list[Segment]:
    """Merge fragments, drop blips, apply boundary offsets and clip to the video."""
    merged = merge_close(segs, max_gap)
    padded = [(max(0.0, s - pad_start), min(duration, e + pad_end)) for s, e in merged if e - s >= min_len]
    return merge_close([(s, e) for s, e in padded if e > s], 0.0)


def smooth_mask(mask: np.ndarray, window: int) -> np.ndarray:
    """Majority vote over a centred window (odd length) to remove single-frame flicker."""
    if window <= 1:
        return np.asarray(mask, bool)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(mask, float), kernel, mode="same") > 0.5
