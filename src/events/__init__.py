"""Event rules: each module turns tracks + scene (+ signal state) into segments for one class."""
from __future__ import annotations

import numpy as np

from src.events import (failure_to_yield, illegal_turn, jaywalking, red_light, solid_line, stop_line,
                        stopped_vehicle)
from src.events.common import Context
from src.scene import Scene
from src.tracking import Tracks

RULES = {
    "jaywalking": jaywalking.detect,
    "failure_to_yield": failure_to_yield.detect,
    "solid_line_crossing": solid_line.detect,
    "stopped_vehicle": stopped_vehicle.detect,
    # congestion intentionally not predicted (dev TP=0; extra pred-only class hurts Score A).
    "red_light": red_light.detect,
    "stop_line": stop_line.detect,
    "illegal_turn": illegal_turn.detect,
}


def detect_all(tracks: Tracks, scene: Scene, duration: float, signal_state: np.ndarray | None = None,
               ) -> list[list]:
    """Return [[start_sec, end_sec, label], ...] from all implemented rules."""
    ctx = Context.build(tracks, scene, duration, signal_state)
    return [[round(s, 3), round(e, 3), label] for label, rule in RULES.items() for s, e in rule(ctx)]
