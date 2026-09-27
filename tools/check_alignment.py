#!/usr/bin/env python3
"""Build the reference frame and check that the scene lands on every video after alignment.

    python tools/check_alignment.py --make-reference proxy/C3896.mp4
    python tools/check_alignment.py proxy/*.mp4 --scene scene/scene.json   # -> reports/alignment/*.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import align, config  # noqa: E402
from src.render import draw_scene  # noqa: E402
from src.scene import SCENE_FILE, Scene  # noqa: E402
from src.video import sample_frames  # noqa: E402

N_BACKGROUND_FRAMES = 25


def median_background(path: Path) -> np.ndarray:
    return align.background(sample_frames(path, N_BACKGROUND_FRAMES))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", type=Path)
    ap.add_argument("--make-reference", type=Path, help="video the scene was annotated on")
    ap.add_argument("--scene", type=Path, default=SCENE_FILE)
    ap.add_argument("--out", type=Path, default=config.ROOT / "reports" / "alignment")
    args = ap.parse_args()

    if args.make_reference:
        cv2.imwrite(str(align.REFERENCE_IMAGE), median_background(args.make_reference), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"wrote {align.REFERENCE_IMAGE}")

    if not args.videos:
        return 0
    reference = align.load_reference()
    scene = Scene.load(args.scene)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.videos:
        bg = median_background(path)
        a = align.estimate(reference, bg)
        dx, dy = a.shift_px(3840, 2160)
        rot = np.degrees(np.arctan2(a.matrix[1, 0], a.matrix[0, 0]))
        print(f"[{path.name}] inliers={a.inliers:4d}  shift=({dx:+.0f}, {dy:+.0f}) px @4K  rotation={rot:+.2f} deg")
        both = np.hstack([draw_scene(bg, scene), draw_scene(bg, scene.aligned(a))])
        cv2.imwrite(str(args.out / f"{path.stem}.jpg"), both)
    print(f"left = raw scene, right = aligned; images in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
