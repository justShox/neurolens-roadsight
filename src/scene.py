"""Static scene layout (road, lanes, crossings, lines, signal) annotated with tools/scene_editor.html."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

from src import config
from src.align import Alignment

SCENE_FILE = config.SCENE_DIR / "scene.json"


@dataclass(frozen=True)
class Shape:
    type: str
    name: str
    points: np.ndarray                 # (K, 2) normalised
    direction: np.ndarray | None = None  # (2, 2) tail -> head, lanes only

    def contains(self, xy: np.ndarray) -> np.ndarray:
        """Vectorised point-in-polygon for (N, 2) normalised points."""
        return MplPath(self.points).contains_points(np.asarray(xy).reshape(-1, 2))

    def signed_distance(self, xy: np.ndarray) -> np.ndarray:
        """Distance to the polygon boundary (frame heights), positive inside and negative outside."""
        xy = np.asarray(xy).reshape(-1, 2)
        s = np.array([config.ASPECT, 1.0])
        p, a = xy * s, self.points * s
        ab = np.roll(a, -1, axis=0) - a
        ap = p[:, None, :] - a[None, :, :]
        t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
        d = np.linalg.norm(ap - t[..., None] * ab, axis=-1).min(axis=1)
        return np.where(self.contains(xy), d, -d)

    def unit_direction(self) -> np.ndarray:
        v = self.direction[1] - self.direction[0]
        return v / (np.linalg.norm(v) + 1e-9)


@dataclass(frozen=True)
class Scene:
    shapes: tuple[Shape, ...] = field(default_factory=tuple)

    @classmethod
    def load(cls, path: Path = SCENE_FILE) -> Scene:
        data = json.loads(Path(path).read_text())
        shapes = tuple(
            Shape(s["type"], s.get("name", ""), np.asarray(s["points"], np.float64),
                  np.asarray(s["direction"], np.float64) if s.get("direction") else None)
            for s in data["shapes"]
        )
        return cls(shapes)

    def of_type(self, *types: str) -> list[Shape]:
        return [s for s in self.shapes if s.type in types]

    def aligned(self, alignment: Alignment) -> Scene:
        return Scene(tuple(
            replace(s, points=alignment.apply(s.points),
                    direction=None if s.direction is None else alignment.apply(s.direction))
            for s in self.shapes
        ))

    def depth(self, xy: np.ndarray, *types: str) -> np.ndarray:
        """Largest signed distance to any shape of the given types (-inf if there are none)."""
        xy = np.asarray(xy).reshape(-1, 2)
        out = np.full(len(xy), -np.inf)
        for s in self.of_type(*types):
            out = np.maximum(out, s.signed_distance(xy))
        return out

    def in_any(self, xy: np.ndarray, *types: str) -> np.ndarray:
        xy = np.asarray(xy).reshape(-1, 2)
        mask = np.zeros(len(xy), bool)
        for s in self.of_type(*types):
            mask |= s.contains(xy)
        return mask
