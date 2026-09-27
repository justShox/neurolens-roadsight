#!/usr/bin/env python3
"""Merge per-video *_labels.json files from tools/annotator.html into one ground-truth file.

    python tools/merge_labels.py labels/raw/*_labels.json --out labels/dev_labels.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OFFICIAL_CLASSES = {
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
}


def check(video: str, entry: dict) -> list[str]:
    problems = []
    by_class: dict[str, list[tuple[float, float]]] = {}
    for s, e, label in entry["events"]:
        if label not in OFFICIAL_CLASSES:
            problems.append(f"{video}: unknown label {label!r}")
        if not 0 <= s < e <= entry["duration"] + 0.5:
            problems.append(f"{video}: bad times [{s}, {e}]")
        by_class.setdefault(label, []).append((s, e))
    for label, segs in by_class.items():
        segs.sort()
        for (s1, e1), (s2, e2) in zip(segs, segs[1:]):
            if s2 < e1:
                problems.append(f"{video}: overlapping {label} [{s1}, {e1}] and [{s2}, {e2}]")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("labels/dev_labels.json"))
    args = ap.parse_args()

    merged, problems = {}, []
    for path in args.files:
        for video, entry in json.loads(path.read_text()).items():
            entry = {k: entry[k] for k in ("duration", "fps", "events")}
            problems += check(video, entry)
            merged[video] = entry

    for p in problems:
        print("!", p)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=1))
    counts: dict[str, int] = {}
    for entry in merged.values():
        for *_, label in entry["events"]:
            counts[label] = counts.get(label, 0) + 1
    print(f"wrote {args.out}: {len(merged)} videos, {sum(counts.values())} events")
    for label, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<20} {n}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
