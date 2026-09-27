"""Detection + multi-object tracking over a video, stored as a flat table of boxes."""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from src import config
from src.video import VideoInfo, iter_frames

TRACK_CLASSES = list(config.COCO_CLASSES.values())
_COCO_TO_IDX = {coco: i for i, coco in enumerate(config.COCO_CLASSES)}


@dataclass
class Tracks:
    """One row per (analysed frame, tracked object). Boxes are normalised to [0, 1]."""

    frame: np.ndarray      # int32, source frame index
    t: np.ndarray          # float32, seconds
    track_id: np.ndarray   # int32
    cls: np.ndarray        # int8, index into TRACK_CLASSES
    conf: np.ndarray       # float32
    box: np.ndarray        # float32 (N, 4): x1, y1, x2, y2
    fps: float
    n_frames: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, frame=self.frame, t=self.t, track_id=self.track_id, cls=self.cls,
                            conf=self.conf, box=self.box, fps=self.fps, n_frames=self.n_frames)

    @classmethod
    def load(cls, path: Path) -> Tracks:
        d = np.load(path)
        return cls(d["frame"], d["t"], d["track_id"], d["cls"], d["conf"], d["box"],
                   float(d["fps"]), int(d["n_frames"]))

    def of_class(self, *names: str) -> np.ndarray:
        return np.isin(self.cls, [TRACK_CLASSES.index(n) for n in names])


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def track_video(info: VideoInfo, stride: int = config.STRIDE, width: int = config.DECODE_WIDTH,
                n_keyframes: int = 0, on_frame: Callable[[np.ndarray], None] | None = None,
                ) -> tuple[Tracks, list[np.ndarray]]:
    """Track all road users. Also returns `n_keyframes` frames spread over the video (for alignment).

    `on_frame` sees every analysed frame, so other per-frame measurements share the single decode pass.
    """
    seed_everything()
    device = select_device()
    model = YOLO(str(config.DETECTOR_WEIGHTS))
    rows: list[tuple] = []
    boxes: list[np.ndarray] = []
    keyframes: list[np.ndarray] = []
    keep_every = max(1, info.n_frames // stride // n_keyframes) if n_keyframes else 0
    for k, (idx, frame) in enumerate(iter_frames(info, stride, width)):
        if keep_every and k % keep_every == 0 and len(keyframes) < n_keyframes:
            keyframes.append(frame[::2, ::2].copy())
        if on_frame is not None:
            on_frame(frame)
        result = model.track(
            frame, persist=True, tracker=str(config.TRACKER_CONFIG), imgsz=config.DETECTOR_IMGSZ,
            conf=config.DETECTOR_CONF, classes=list(config.COCO_CLASSES), device=device,
            quantize=16 if device.startswith("cuda") else None, verbose=False,
        )[0]
        if result.boxes is None or result.boxes.id is None:
            continue
        b = result.boxes
        h, w = frame.shape[:2]
        xyxy = b.xyxy.cpu().numpy() / np.array([w, h, w, h], dtype=np.float32)
        for tid, c, p, box in zip(b.id.int().tolist(), b.cls.int().tolist(), b.conf.tolist(), xyxy):
            rows.append((idx, idx / info.fps, tid, _COCO_TO_IDX[c], p))
            boxes.append(box)
    cols = list(zip(*rows)) if rows else [[]] * 5
    tracks = Tracks(
        frame=np.asarray(cols[0], np.int32),
        t=np.asarray(cols[1], np.float32),
        track_id=np.asarray(cols[2], np.int32),
        cls=np.asarray(cols[3], np.int8),
        conf=np.asarray(cols[4], np.float32),
        box=np.asarray(boxes, np.float32).reshape(-1, 4),
        fps=info.fps,
        n_frames=info.n_frames,
    )
    return tracks, keyframes
