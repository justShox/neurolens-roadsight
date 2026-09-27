"""Part B: causal accident risk from pairwise time-to-collision of tracked road users."""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from src import config
from src.tracking import select_device, seed_everything

RISK_STRIDE = 5          # analyse every N-th frame; return last score in between
IMGSZ = 640              # lighter than Part A; risk only needs coarse boxes
CONF = 0.25
TTC_SOFT = 2.0           # s; risk ~= exp(-ttc / TTC_SOFT), so TTC=0 -> 1, TTC=2 -> ~0.37
HISTORY = 8              # analysed samples of history kept per track for velocity
MIN_SPEED = 0.02         # frame heights / s; ignore parked objects
COCO_KEEP = (0, 1, 2, 3, 5, 7)  # person, bicycle, car, motorcycle, bus, truck


class RiskEstimator:
    """Online TTC risk. Uses only frames seen so far (ByteTrack state is causal)."""

    def reset(self, meta: dict) -> None:
        seed_everything()
        self.meta = meta
        self.fps = float(meta["fps"])
        self.device = select_device()
        self.model = YOLO(str(config.DETECTOR_WEIGHTS))
        self.last_score = 0.0
        self.frame_i = 0
        self.history: dict[int, list[tuple[float, np.ndarray]]] = {}

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        if self.frame_i % RISK_STRIDE == 0:
            self.last_score = self._score(frame, t_sec)
        self.frame_i += 1
        return self.last_score

    def _score(self, frame: np.ndarray, t_sec: float) -> float:
        result = self.model.track(
            frame, persist=True, tracker=str(config.TRACKER_CONFIG), imgsz=IMGSZ,
            conf=CONF, classes=list(COCO_KEEP), device=self.device,
            quantize=16 if self.device.startswith("cuda") else None, verbose=False,
        )[0]
        if result.boxes is None or result.boxes.id is None:
            return self.last_score * 0.9

        h, w = frame.shape[:2]
        xyxy = result.boxes.xyxy.cpu().numpy() / np.array([w, h, w, h], np.float32)
        ids = result.boxes.id.int().tolist()
        centres = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2,
                            (xyxy[:, 1] + xyxy[:, 3]) / 2], axis=1)
        centres[:, 0] *= config.ASPECT

        live = set(ids)
        for tid, c in zip(ids, centres):
            hist = self.history.setdefault(tid, [])
            hist.append((t_sec, c.copy()))
            del hist[:-HISTORY]
        for tid in list(self.history):
            if tid not in live:
                del self.history[tid]

        states = []
        for tid, hist in self.history.items():
            if len(hist) < 2:
                continue
            (t0, p0), (t1, p1) = hist[0], hist[-1]
            dt = max(t1 - t0, 1e-3)
            v = (p1 - p0) / dt
            if np.linalg.norm(v) < MIN_SPEED:
                continue
            states.append((p1, v))
        if len(states) < 2:
            return 0.0

        risk = 0.0
        for i in range(len(states)):
            pi, vi = states[i]
            for j in range(i + 1, len(states)):
                pj, vj = states[j]
                ttc = _ttc(pi, vi, pj, vj)
                if ttc is not None and 0.0 <= ttc <= RISK_HORIZON:
                    risk = max(risk, float(np.exp(-ttc / TTC_SOFT)))
        return float(np.clip(risk, 0.0, 1.0))


RISK_HORIZON = 5.0


def _ttc(p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray) -> float | None:
    """Time until closest approach if relative motion closes the gap; None if diverging."""
    r, v = p2 - p1, v2 - v1
    vv = float(v @ v)
    if vv < 1e-12:
        return 0.0 if float(r @ r) < 1e-4 else None
    t_star = -float(r @ v) / vv
    if t_star < 0:
        return None
    closest = r + v * t_star
    if float(closest @ closest) > 0.04 ** 2:  # miss by more than 0.04 frame heights
        return None
    return t_star
