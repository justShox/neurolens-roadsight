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


def trim_sparse(times: np.ndarray, mask: np.ndarray, segs: list[Segment], *,
                min_density: float = 0.45, window: float = 1.5) -> list[Segment]:
    """Shrink each segment from both ends until a `window`-second edge is dense enough.

    Stops over-long events that only have sparse flags near the boundary (hurts tIoU@0.7).
    """
    times, mask = np.asarray(times, float), np.asarray(mask, bool)
    if not len(times):
        return segs
    step = float(times[1] - times[0]) if len(times) > 1 else 0.1
    half = max(1, int(round(window / step)))
    out: list[Segment] = []
    for s, e in segs:
        i0 = int(np.searchsorted(times, s, side="left"))
        i1 = int(np.searchsorted(times, e - 1e-9, side="right")) - 1
        i0, i1 = max(0, i0), min(len(times) - 1, i1)
        while i0 + half <= i1 and mask[i0:i0 + half].mean() < min_density:
            i0 += 1
        while i1 - half >= i0 and mask[i1 - half + 1:i1 + 1].mean() < min_density:
            i1 -= 1
        if i1 >= i0 and times[i1] + step - times[i0] > 0:
            out.append((float(times[i0]), float(times[i1] + step)))
    return out
