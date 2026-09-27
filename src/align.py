"""Compensate small camera shifts between recordings by registering each video to the reference frame
the scene was annotated on. Scene coordinates are normalised, so the transform is too."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src import config

REFERENCE_IMAGE = config.SCENE_DIR / "reference.jpg"
WORK_WIDTH = 960
MIN_INLIERS = 40


@dataclass(frozen=True)
class Alignment:
    """Similarity transform (2x3) mapping normalised reference coordinates to normalised video coordinates."""

    matrix: np.ndarray
    inliers: int

    @classmethod
    def identity(cls) -> Alignment:
        return cls(np.array([[1, 0, 0], [0, 1, 0]], np.float64), 0)

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, np.float64).reshape(-1, 2)
        return pts @ self.matrix[:, :2].T + self.matrix[:, 2]

    def shift_px(self, width: int, height: int) -> tuple[float, float]:
        centre = self.apply([[0.5, 0.5]])[0] - 0.5
        return float(centre[0] * width), float(centre[1] * height)


def background(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of a few frames spread over time removes most moving vehicles."""
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _prepare(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h = round(gray.shape[0] * WORK_WIDTH / gray.shape[1])
    gray = cv2.resize(gray, (WORK_WIDTH, h), interpolation=cv2.INTER_AREA)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def estimate(reference: np.ndarray, frame: np.ndarray) -> Alignment:
    """Register `frame` (ideally a median background) to `reference`. Falls back to identity."""
    ref, cur = _prepare(reference), _prepare(frame)
    sift = cv2.SIFT_create(nfeatures=4000)
    kr, dr = sift.detectAndCompute(ref, None)
    kc, dc = sift.detectAndCompute(cur, None)
    if dr is None or dc is None or len(kr) < MIN_INLIERS or len(kc) < MIN_INLIERS:
        return Alignment.identity()
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dr, dc, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return Alignment.identity()
    src = np.float32([kr[m.queryIdx].pt for m in good])
    dst = np.float32([kc[m.trainIdx].pt for m in good])
    m, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if m is None or inliers < MIN_INLIERS:
        return Alignment.identity()
    w, h = ref.shape[1], ref.shape[0]
    to_px = np.array([[w, 0, 0], [0, h, 0], [0, 0, 1]], np.float64)
    to_norm = np.linalg.inv(np.array([[cur.shape[1], 0, 0], [0, cur.shape[0], 0], [0, 0, 1]], np.float64))
    full = to_norm @ np.vstack([m, [0, 0, 1]]) @ to_px
    return Alignment(full[:2], inliers)


def load_reference() -> np.ndarray:
    img = cv2.imread(str(REFERENCE_IMAGE))
    if img is None:
        raise FileNotFoundError(REFERENCE_IMAGE)
    return img
