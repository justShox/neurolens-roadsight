#!/usr/bin/env python3
"""Hyperparameter tuning for event rules with leave-one-out scoring.

    python tools/tune.py                # report only
    python tools/tune.py --apply        # write improving constants back into src/events/*.py
    python tools/tune.py --class jaywalking
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import evaluate  # noqa: E402
from src import align, config, events  # noqa: E402
from src.events.common import Context  # noqa: E402
from src.scene import Scene  # noqa: E402
from src.tracking import Tracks  # noqa: E402
from src.video import sample_frames  # noqa: E402

N_BG_FRAMES = 25

MODULE_MAP = {
    "jaywalking": "src.events.jaywalking",
    "failure_to_yield": "src.events.failure_to_yield",
    "stop_line": "src.events.stop_line",
    "solid_line_crossing": "src.events.solid_line",
    "stopped_vehicle": "src.events.stopped_vehicle",
    "red_light": "src.events.red_light",
    "illegal_turn": "src.events.illegal_turn",
}

# Keep each class under ~36 combinations; LOO reuses per-video scores.
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "jaywalking": {
        "ROAD_DEPTH": [0.010, 0.012, 0.015],
        "CLEARANCE": [0.012, 0.015, 0.020],
        "MERGE_GAP": [5.0, 7.0],
        "PAD_START": [0.0, 1.0],
        "PAD_END": [0.0, 1.0],
    },
    "failure_to_yield": {
        "MAX_DISTANCE": [0.12, 0.16, 0.20],
        "MERGE_GAP": [2.0, 3.0, 5.0],
        "PAD_START": [0.0, 1.0],
        "PAD_END": [0.0, 1.0],
    },
    "stop_line": {
        "MAX_PAST": [0.04, 0.06, 0.08],
        "MIN_WAIT": [1.5, 2.0, 3.0],
        "PAD_START": [0.0, 1.0],
        "PAD_END": [0.0, 1.0],
    },
    "solid_line_crossing": {
        "SIDE_WINDOW": [0.5, 0.7, 1.0],
        "MIN_OFFSET": [0.004, 0.006, 0.008],
        "PAD_START": [0.0, 0.5],
        "PAD_END": [0.0, 0.5],
    },
    "stopped_vehicle": {
        "MIN_STOPPED": [8.0, 10.0, 15.0],
        "MIN_PASSING_SHARE": [0.2, 0.3, 0.5],
        "PAD_START": [0.0, 2.0],
        "PAD_END": [0.0, 2.0],
    },
    "red_light": {
        "RED_BEFORE": [1.5, 2.0, 3.0],
        "POST": [1.0, 2.0, 5.0],
        "PAD_END": [0.0, 5.0, 10.0],
    },
}


def load_contexts(gt: dict[str, Any]) -> dict[str, Context]:
    reference, scene = align.load_reference(), Scene.load()
    contexts = {}
    for name, entry in gt.items():
        stem = Path(name).stem
        tracks = Tracks.load(config.CACHE_DIR / "tracks" / f"{stem}.npz")
        bg = align.background(sample_frames(config.ROOT / "proxy" / f"{stem}.mp4", N_BG_FRAMES))
        aligned = scene.aligned(align.estimate(reference, bg))
        sig_file = config.CACHE_DIR / "signal" / f"{stem}.npz"
        state = np.load(sig_file)["state"] if sig_file.exists() else None
        contexts[name] = Context.build(tracks, aligned, entry["duration"], state)
    return contexts


def score_videos(event_class: str, mod: Any, contexts: dict[str, Context], gt: dict[str, Any],
                 video_keys: list[str]) -> float:
    preds = {"team": "neurolens", "videos": {}}
    sub_gt = {}
    for k in video_keys:
        segs = mod.detect(contexts[k])
        preds["videos"][k] = {"events": [[s, e, event_class] for s, e in segs], "risk": []}
        sub_gt[k] = {
            "duration": gt[k]["duration"], "fps": gt[k]["fps"],
            "events": [ev for ev in gt[k]["events"] if ev[2] == event_class],
        }
    pc = evaluate(sub_gt, preds, per_video=False)["part_a"]["per_class"]
    return pc[event_class]["f1_mean"] if event_class in pc else 0.0


def per_video_scores(event_class: str, mod: Any, contexts: dict[str, Context],
                     gt: dict[str, Any]) -> dict[str, float]:
    return {k: score_videos(event_class, mod, contexts, gt, [k]) for k in contexts}


def update_source_file(mod_path: str, params: dict[str, Any]) -> None:
    path = config.ROOT / (mod_path.replace(".", "/") + ".py")
    text = path.read_text()
    for name, val in params.items():
        pattern = rf"^({name}\s*=\s*)([0-9\.]+)(.*)$"
        if isinstance(val, float):
            val_str = f"{val:.4f}".rstrip("0").rstrip(".")
            if "." not in val_str:
                val_str += ".0"
        else:
            val_str = str(val)
        text = re.sub(pattern, rf"\g<1>{val_str}\g<3>", text, flags=re.MULTILINE)
    path.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    gt = json.loads((config.ROOT / "labels" / "dev_labels.json").read_text())
    print("Precomputing contexts...", flush=True)
    contexts = load_contexts(gt)
    videos = sorted(contexts.keys())
    print(f"Loaded {len(videos)} videos.", flush=True)

    classes = [args.cls] if args.cls else list(PARAM_GRIDS)
    print("\n" + "=" * 88, flush=True)
    print(f"{'Class':<22} | {'Old':>8} | {'Holdout':>8} | Parameters", flush=True)
    print("=" * 88, flush=True)

    for cls in classes:
        if cls not in PARAM_GRIDS:
            print(f"Skipping {cls}: no grid", flush=True)
            continue
        mod = importlib.import_module(MODULE_MAP[cls])
        grid = PARAM_GRIDS[cls]
        keys = list(grid)
        combos = [dict(zip(keys, prod)) for prod in itertools.product(*(grid[k] for k in keys))]
        orig = {k: getattr(mod, k) for k in keys}
        for k, v in orig.items():
            setattr(mod, k, v)
        old = score_videos(cls, mod, contexts, gt, videos)
        print(f"\nTuning {cls}: {len(combos)} combos...", flush=True)

        # One detect pass per (combo, video); LOO reuses the matrix.
        matrix: list[dict[str, float]] = []
        for combo in combos:
            for k, v in combo.items():
                setattr(mod, k, v)
            matrix.append(per_video_scores(cls, mod, contexts, gt))

        holdout_preds: dict[str, list] = {}
        fold_params: list[dict] = []
        for val in videos:
            train = [v for v in videos if v != val]
            best_i, best_s = 0, -1.0
            for i, scores in enumerate(matrix):
                s = float(np.mean([scores[v] for v in train]))
                if s > best_s:
                    best_i, best_s = i, s
            fold_params.append(combos[best_i])
            for k, v in combos[best_i].items():
                setattr(mod, k, v)
            holdout_preds[val] = [[s, e, cls] for s, e in mod.detect(contexts[val])]

        pooled = {"team": "neurolens", "videos": {v: {"events": holdout_preds[v], "risk": []} for v in videos}}
        sub_gt = {v: {"duration": gt[v]["duration"], "fps": gt[v]["fps"],
                      "events": [e for e in gt[v]["events"] if e[2] == cls]} for v in videos}
        pc = evaluate(sub_gt, pooled, per_video=False)["part_a"]["per_class"]
        holdout = pc[cls]["f1_mean"] if cls in pc else 0.0

        # Params to apply: best mean over all videos among combos (report); only write if holdout >= old.
        best_all_i = int(np.argmax([np.mean(list(s.values())) for s in matrix]))
        best_params = combos[best_all_i]
        for k, v in orig.items():
            setattr(mod, k, v)

        params_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
        print(f"{cls:<22} | {old:8.4f} | {holdout:8.4f} | {params_str}", flush=True)

        if args.apply and holdout >= old - 1e-9:
            update_source_file(MODULE_MAP[cls], best_params)
            print(f"  -> applied", flush=True)
        elif args.apply:
            print(f"  -> skipped (holdout did not improve)", flush=True)

    print("=" * 88, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
