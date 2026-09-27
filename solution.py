"""
solution.py — entry point imported by the organizers' harness (run_submission.py).

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

The implementation lives in src/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import pipeline  # noqa: E402

CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A: see src/pipeline.py."""
    return pipeline.detect_events(video_path)


class RiskEstimator:
    """Part B: causal accident anticipation. Not implemented yet (constant zero risk)."""

    def reset(self, meta: dict) -> None:
        self.meta = meta
        self.last_score = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self.last_score
