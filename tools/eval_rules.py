#!/usr/bin/env python3
"""Score the event rules on the dev labels from cached tracks (no detector run, takes seconds).

    python tools/eval_rules.py                       # all classes in the labels
    python tools/eval_rules.py --only-predicted      # restrict the labels to classes we already predict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import evaluate, print_report, validate  # noqa: E402
from src import align, config, events  # noqa: E402
from src.scene import Scene  # noqa: E402
from src.tracking import Tracks  # noqa: E402
from src.video import sample_frames  # noqa: E402

N_BACKGROUND_FRAMES = 25


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=config.ROOT / "labels" / "dev_labels.json")
    ap.add_argument("--tracks", type=Path, default=config.CACHE_DIR / "tracks")
    ap.add_argument("--signal", type=Path, default=config.CACHE_DIR / "signal",
                    help="signal states from tools/read_signal.py; rules needing them stay silent without")
    ap.add_argument("--out", type=Path, default=config.ROOT / "reports" / "predictions_dev.json")
    ap.add_argument("--only-predicted", action="store_true")
    args = ap.parse_args()

    gt = json.loads(args.labels.read_text())
    reference, scene = align.load_reference(), Scene.load()
    pred = {"team": "neurolens", "videos": {}}
    for name, entry in gt.items():
        stem = Path(name).stem
        tracks = Tracks.load(args.tracks / f"{stem}.npz")
        bg = align.background(sample_frames(config.ROOT / "proxy" / f"{stem}.mp4", N_BACKGROUND_FRAMES))
        aligned = scene.aligned(align.estimate(reference, bg))
        signal_file = args.signal / f"{stem}.npz"
        state = np.load(signal_file)["state"] if signal_file.exists() else None
        pred["videos"][name] = {"events": events.detect_all(tracks, aligned, entry["duration"], state), "risk": []}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pred, indent=1))
    errors, _ = validate(pred, gt)
    if errors:
        print("\n".join(errors))
        return 1

    if args.only_predicted:
        predicted = set(events.RULES)
        gt = {v: {**e, "events": [ev for ev in e["events"] if ev[2] in predicted]} for v, e in gt.items()}
    print_report(evaluate(gt, pred, per_video=True))
    print(f"\npredictions written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
