"""Drawing helpers shared by the debugging tools and the result videos for the website."""
from __future__ import annotations

import cv2
import numpy as np

from src.scene import Scene

SHAPE_COLORS = {  # BGR
    "road": (80, 200, 80), "crosswalk": (255, 255, 255), "sidewalk": (150, 150, 150),
    "lane": (255, 150, 50), "stop_line": (40, 40, 255), "solid_line": (0, 220, 255),
    "intersection": (255, 100, 200), "no_u_turn_zone": (0, 150, 255), "exit_zone": (200, 200, 0),
    "signal_red": (0, 0, 255), "signal_yellow": (0, 255, 255), "signal_green": (0, 255, 0),
}
POLYLINE_TYPES = {"stop_line", "solid_line"}


def draw_scene(img: np.ndarray, scene: Scene, alpha: float = 0.25) -> np.ndarray:
    h, w = img.shape[:2]
    overlay, out = img.copy(), img.copy()
    thickness = max(1, w // 640)
    for s in scene.shapes:
        color = SHAPE_COLORS.get(s.type, (255, 0, 255))
        pts = (s.points * [w, h]).astype(np.int32)
        closed = s.type not in POLYLINE_TYPES
        if closed:
            cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(out, [pts], closed, color, thickness * 2)
        if s.direction is not None:
            a, b = (s.direction * [w, h]).astype(int)
            cv2.arrowedLine(out, tuple(a), tuple(b), color, thickness * 3, tipLength=0.25)
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
