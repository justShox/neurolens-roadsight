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
LIT_CONTRAST = 15.0      # colour strength above the section's unlit level for it to count as on
MIN_RUN = 5              # samples; shorter state runs are flicker and take the previous state


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
    """Per-sample state: the section whose colour is furthest above its own unlit level, UNKNOWN if none is lit."""
    contrast = lv - np.percentile(lv, 10, axis=0)
    state = contrast.argmax(axis=1)
    state[contrast.max(axis=1) < LIT_CONTRAST] = UNKNOWN
    return _drop_flicker(state)


def _drop_flicker(state: np.ndarray) -> np.ndarray:
    out = state.copy()
    edges = np.flatnonzero(np.diff(out)) + 1
    starts, ends = np.r_[0, edges], np.r_[edges, len(out)]
    for s, e in zip(starts, ends):
        if e - s < MIN_RUN and s > 0:
            out[s:e] = out[s - 1]
    return out
