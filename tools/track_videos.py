#!/usr/bin/env python3
"""Run detection + tracking on videos and cache the tracks for rule development and EDA.

    python tools/track_videos.py samples/*.MP4            # -> cache/tracks/<name>.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.tracking import TRACK_CLASSES, select_device, track_video  # noqa: E402
from src.video import probe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "tracks")
    ap.add_argument("--force", action="store_true", help="recompute even if the cache exists")
    args = ap.parse_args()

    print(f"device: {select_device()}")
    for path in args.videos:
        out = args.out / f"{path.stem}.npz"
        if out.exists() and not args.force:
            print(f"[{path.name}] cached -> {out}")
            continue
        info = probe(path)
        t0 = time.perf_counter()
        tracks, _ = track_video(info)
        dt = time.perf_counter() - t0
        tracks.save(out)
        per_class = {c: len(set(tracks.track_id[tracks.cls == i].tolist())) for i, c in enumerate(TRACK_CLASSES)}
        print(f"[{path.name}] {info.duration:.0f}s video in {dt:.0f}s ({dt / info.duration:.2f}x realtime), "
              f"{len(tracks.t)} boxes, tracks per class: {per_class}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
