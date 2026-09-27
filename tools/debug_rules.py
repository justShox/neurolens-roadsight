#!/usr/bin/env python3
"""Show where the rules fire: flagged foot points over the aligned scene, plus the worst offending tracks.

    python tools/debug_rules.py          # -> reports/debug/<video>.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import align, config  # noqa: E402
from src.events import failure_to_yield, jaywalking  # noqa: E402
from src.events.common import Context  # noqa: E402
from src.render import draw_scene  # noqa: E402
from src.scene import Scene  # noqa: E402
from src.tracking import Tracks  # noqa: E402
from src.video import probe, sample_frames  # noqa: E402

TOP_TRACKS = 8
COLORS = {"jaywalking": (0, 0, 255), "fty_person": (0, 255, 255), "fty_vehicle": (255, 255, 0)}  # BGR


def draw_points(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    for x, y in (pts * [w, h]).astype(int):
        cv2.circle(img, (int(x), int(y)), 2, color, -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=sorted((config.ROOT / "samples").glob("*.MP4")), type=Path)
    ap.add_argument("--out", type=Path, default=config.ROOT / "reports" / "debug")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reference, scene = align.load_reference(), Scene.load()

    for path in args.videos:
        stem = path.stem
        tracks = Tracks.load(config.CACHE_DIR / "tracks" / f"{stem}.npz")
        bg = align.background(sample_frames(config.ROOT / "proxy" / f"{stem}.mp4", 25))
        aligned = scene.aligned(align.estimate(reference, bg))
        ctx = Context.build(tracks, aligned, probe(path).duration)

        jw = jaywalking.flagged(ctx)
        _, fty = failure_to_yield.flagged(ctx)
        img = draw_scene(bg, aligned, alpha=0.15)
        draw_points(img, ctx.foot[jw], COLORS["jaywalking"])
        draw_points(img, ctx.foot[fty & ctx.pedestrians], COLORS["fty_person"])
        draw_points(img, ctx.foot[fty & ctx.vehicles], COLORS["fty_vehicle"])
        cv2.imwrite(str(args.out / f"{stem}.jpg"), img)

        print(f"\n[{path.name}] jaywalking rows={jw.sum()}  failure_to_yield rows={fty.sum()}")
        print("  longest jaywalking tracks: id, seconds flagged, mean foot (x, y) @1280x720, mean speed")
        ids, counts = np.unique(tracks.track_id[jw], return_counts=True)
        for tid in ids[np.argsort(-counts)][:TOP_TRACKS]:
            m = jw & (tracks.track_id == tid)
            x, y = ctx.foot[m].mean(0) * [1280, 720]
            print(f"  #{tid:<5} {m.sum() * ctx.timeline.step:6.1f}s  ({x:4.0f}, {y:4.0f})  "
                  f"speed={ctx.speed[m].mean():.3f}  t={tracks.t[m].min():.0f}-{tracks.t[m].max():.0f}s")
    print(f"\nimages in {args.out}  (red = jaywalking, yellow = pedestrian on crossing, cyan = vehicle on crossing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
