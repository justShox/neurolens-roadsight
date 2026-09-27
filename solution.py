"""
solution.py — entry point imported by the organizers' harness (run_submission.py).

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

The implementation lives in src/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Jury machines are offline; skip Ultralytics connectivity / update checks.
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_VERBOSE", "False")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import pipeline  # noqa: E402
from src.risk import RiskEstimator  # noqa: E402

CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A: see src/pipeline.py."""
    return pipeline.detect_events(video_path)


__all__ = ["CLASSES", "RISK_HORIZON_SEC", "detect_events", "RiskEstimator"]
