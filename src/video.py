"""Video probing and fast strided frame reading through an ffmpeg pipe."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import imageio_ffmpeg
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoInfo:
    """Same metadata source as the official harness (OpenCV), so timestamps agree."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    info = VideoInfo(
        path=Path(path),
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def sample_frames(path: str | Path, n: int) -> list[np.ndarray]:
    """`n` frames evenly spread over the video via seeking (for tools on small proxy files)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def ffmpeg_exe() -> str:
    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()


def iter_frames(info: VideoInfo, stride: int, width: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame scaled to `width`) for every `stride`-th frame.

    Selection and scaling run inside ffmpeg, so only the analysed frames cross the pipe.
    """
    height = round(info.height * width / info.width / 2) * 2
    cmd = [
        ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(info.path),
        "-an", "-sn", "-dn",
        "-vf", f"select=not(mod(n\\,{stride})),scale={width}:{height}:flags=area",
        "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
    try:
        k = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield k * stride, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            k += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
