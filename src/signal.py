"""Traffic-signal state from the brightness of the three lamp sections of the opposite signal head.

Per the labelling convention (scene/camera.md) the opposite head drives the whole signal phase.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.scene import Scene

UNKNOWN, RED, YELLOW, GREEN = -1, 0, 1, 2
SECTIONS = ("signal_red", "signal_yellow", "signal_green")   # index == state
CROP_PAD = 0.04          # normalised margin around the lamps; covers the camera shift between videos
NOISE_MULT = 2.5        # multiplier on high-frequency noise for the detection threshold
MIN_CONTRAST = 3.0      # absolute floor on colour contrast
MIN_RUN = 5             # samples; shorter state runs take the previous state


def crop_box(scene: Scene) -> tuple[float, float, float, float]:
    """Normalised (x1, y1, x2, y2) region cut from every frame while tracking."""
    pts = np.concatenate([s.points for s in scene.of_type(*SECTIONS)])
    lo, hi = np.clip(pts.min(0) - CROP_PAD, 0, 1), np.clip(pts.max(0) + CROP_PAD, 0, 1)
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


class CropCollector:
    """Frame callback for tracking: keeps the signal region of each frame."""

    def __init__(self, box: tuple[float, float, float, float]):
        self.box = box
        self.crops: list[np.ndarray] = []

    def __call__(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.box
        self.crops.append(frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)].copy())


def _colour_scores(stack: np.ndarray) -> list[np.ndarray]:
    """Per-pixel redness, yellowness, greenness. In daylight lit lamps are barely brighter than
    unlit ones, but their colour still stands out."""
    b, g, r = (stack[..., i].astype(np.int16) for i in range(3))
    return [r - np.maximum(g, b), np.minimum(r, g) - b, g - r]


def levels(crops: list[np.ndarray], box: tuple[float, float, float, float], scene: Scene) -> np.ndarray:
    """(T, 3) colour strength of the red / yellow / green sections (scene already aligned to the video)."""
    stack = np.stack(crops)
    h, w = stack.shape[1:3]
    scores = _colour_scores(stack)
    x1, y1, x2, y2 = box
    out = np.zeros((len(stack), len(SECTIONS)))
    for j, name in enumerate(SECTIONS):
        shapes = scene.of_type(name)
        if not shapes:
            continue
        px = (shapes[0].points - [x1, y1]) / [x2 - x1, y2 - y1] * [w, h]
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 1)
        if mask.any():
            out[:, j] = np.percentile(scores[j][:, mask.astype(bool)], 90, axis=1)
    return out


def states(lv: np.ndarray) -> np.ndarray:
    """Per-sample state: section with largest normalised contrast above noise, with cycle continuity."""
    p10 = np.percentile(lv, 10, axis=0)
    contrast = lv - p10
    diffs = np.abs(np.diff(lv, axis=0))
    noise = np.maximum(np.percentile(diffs, 90, axis=0), 1.0)
    thr = np.maximum(NOISE_MULT * noise, MIN_CONTRAST)

    span = np.zeros(3)
    span[0] = max(np.percentile(lv[:, 0], 90) - p10[0], thr[0])
    span[2] = max(np.percentile(lv[:, 2], 90) - p10[2], thr[2])
    yellow_peak = np.percentile(lv[:, 1], 99.5) - p10[1]
    span[1] = max(yellow_peak, span[2] * 0.5, thr[1])

    norm_contrast = contrast / span
    valid = contrast >= thr
    norm_contrast[~valid] = -1.0

    state = np.full(len(lv), UNKNOWN)
    has_valid = valid.any(axis=1)
    state[has_valid] = norm_contrast[has_valid].argmax(axis=1)
    state = _drop_flicker(state)
    return _refine_cycles(state)


def _drop_flicker(state: np.ndarray) -> np.ndarray:
    out = state.copy()
    edges = np.flatnonzero(np.diff(out)) + 1
    starts, ends = np.r_[0, edges], np.r_[edges, len(out)]
    for s, e in zip(starts, ends):
        if e - s < MIN_RUN and s > 0:
            out[s:e] = out[s - 1]
    return out


def _refine_cycles(state: np.ndarray) -> np.ndarray:
    out = state.copy()
    for _ in range(5):
        edges = np.flatnonzero(np.diff(out)) + 1
        starts, ends = np.r_[0, edges], np.r_[edges, len(out)]
        for s, e in zip(starts, ends):
            prev_st = out[s - 1] if s > 0 else UNKNOWN
            next_st = out[e] if e < len(out) else UNKNOWN
            dur = (e - s) * 2 / 25.0

            if out[s] == YELLOW:
                if prev_st == RED or (prev_st == GREEN and next_st == GREEN):
                    out[s:e] = prev_st
            elif out[s] == UNKNOWN:
                if prev_st == next_st and prev_st != UNKNOWN and dur < 6.0:
                    out[s:e] = prev_st
                elif prev_st == GREEN and next_st == RED and dur < 4.5:
                    out[s:e] = YELLOW
                elif prev_st == YELLOW and next_st == RED and dur < 4.5:
                    out[s:e] = YELLOW
                elif prev_st != UNKNOWN and dur < 1.5:
                    out[s:e] = prev_st
                elif next_st != UNKNOWN and dur < 1.5:
                    out[s:e] = next_st
            elif dur < 2.0 and prev_st == next_st and prev_st in (RED, GREEN):
                out[s:e] = prev_st

    if out[0] == UNKNOWN:
        first = np.flatnonzero(out != UNKNOWN)
        if len(first) and first[0] * 2 / 25.0 < 5.0:
            out[:first[0]] = out[first[0]]
    if out[-1] == UNKNOWN:
        last = np.flatnonzero(out != UNKNOWN)
        if len(last) and (len(out) - 1 - last[-1]) * 2 / 25.0 < 5.0:
            out[last[-1] + 1:] = out[last[-1]]
    return out
