#!/usr/bin/env python3
"""EDA of the sample videos from cached tracks (run tools/track_videos.py first).

    python tools/eda.py samples/*.MP4        # -> reports/eda/<video>/*.png + summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import align, config  # noqa: E402
from src.tracking import TRACK_CLASSES, Tracks  # noqa: E402
from src.video import probe, sample_frames  # noqa: E402

MIN_TRACK_POINTS = 5
CLASS_COLORS = {"person": "tab:green", "bicycle": "tab:olive", "car": "tab:blue",
                "motorcycle": "tab:purple", "bus": "tab:orange", "truck": "tab:red"}


def foot_points(tracks: Tracks) -> np.ndarray:
    b = tracks.box
    return np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], axis=1)


def long_track_mask(tracks: Tracks) -> np.ndarray:
    ids, counts = np.unique(tracks.track_id, return_counts=True)
    return np.isin(tracks.track_id, ids[counts >= MIN_TRACK_POINTS])


def plot_counts(tracks: Tracks, duration: float, out: Path) -> dict:
    """Distinct objects visible per second, per class."""
    bins = np.arange(0, np.ceil(duration) + 1)
    series = {}
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, name in enumerate(TRACK_CLASSES):
        m = tracks.cls == i
        if not m.any():
            continue
        sec = np.floor(tracks.t[m]).astype(int)
        pairs = np.unique(np.stack([sec, tracks.track_id[m]], axis=1), axis=0)
        counts = np.bincount(pairs[:, 0], minlength=len(bins) - 1)[: len(bins) - 1]
        series[name] = counts.tolist()
        ax.plot(bins[:-1], counts, label=name, color=CLASS_COLORS[name], lw=1.2)
    ax.set_xlabel("time, s")
    ax.set_ylabel("objects visible")
    ax.legend(ncol=6, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return series


def plot_heatmap(bg: np.ndarray, pts: np.ndarray, out: Path, title: str) -> None:
    h, w = bg.shape[:2]
    heat, _, _ = np.histogram2d(pts[:, 1] * h, pts[:, 0] * w, bins=(h // 8, w // 8), range=[[0, h], [0, w]])
    heat = cv2.GaussianBlur(np.log1p(heat).astype(np.float32), (0, 0), 2)
    heat = cv2.resize(heat / (heat.max() + 1e-9), (w, h))
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (heat[..., None] * 0.8).clip(0, 0.8)
    img = (bg * (1 - alpha) + color * alpha).astype(np.uint8)
    cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(out), img)


def plot_trajectories(bg: np.ndarray, tracks: Tracks, mask: np.ndarray, out: Path) -> None:
    """Vehicle trajectories coloured by overall heading (hue = direction of travel)."""
    h, w = bg.shape[:2]
    img = (bg * 0.6).astype(np.uint8)
    pts = foot_points(tracks)
    for tid in np.unique(tracks.track_id[mask]):
        m = mask & (tracks.track_id == tid)
        p = pts[m] * [w, h]
        if len(p) < MIN_TRACK_POINTS or np.linalg.norm(p[-1] - p[0]) < 0.03 * w:
            continue
        angle = (np.degrees(np.arctan2(*(p[-1] - p[0])[::-1])) + 360) % 360
        hsv = np.uint8([[[angle / 2, 255, 255]]])
        color = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
        cv2.polylines(img, [p.astype(np.int32)], False, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(out), img)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--tracks", type=Path, default=config.CACHE_DIR / "tracks")
    ap.add_argument("--out", type=Path, default=config.ROOT / "reports" / "eda")
    args = ap.parse_args()

    for path in args.videos:
        tracks = Tracks.load(args.tracks / f"{path.stem}.npz")
        info = probe(path)
        out = args.out / path.stem
        out.mkdir(parents=True, exist_ok=True)
        proxy = config.ROOT / "proxy" / f"{path.stem}.mp4"
        bg = align.background(sample_frames(proxy if proxy.exists() else path, 25))

        keep = long_track_mask(tracks)
        vehicles = keep & tracks.of_class(*config.VEHICLE_CLASSES)
        persons = keep & tracks.of_class("person")
        series = plot_counts(tracks, info.duration, out / "counts.png")
        plot_heatmap(bg, foot_points(tracks)[vehicles], out / "heatmap_vehicles.png", "vehicles")
        plot_heatmap(bg, foot_points(tracks)[persons], out / "heatmap_persons.png", "pedestrians")
        plot_trajectories(bg, tracks, vehicles, out / "trajectories.png")
        cv2.imwrite(str(out / "background.jpg"), bg)

        n_tracks = {c: int(len(np.unique(tracks.track_id[keep & (tracks.cls == i)])))
                    for i, c in enumerate(TRACK_CLASSES)}
        summary = {
            "video": path.name, "width": info.width, "height": info.height, "fps": round(info.fps, 3),
            "duration_sec": round(info.duration, 2), "mean_brightness": round(float(bg.mean()), 1),
            "tracks_per_class": n_tracks,
            "mean_visible_per_class": {c: round(float(np.mean(v)), 2) for c, v in series.items()},
            "peak_visible_per_class": {c: int(np.max(v)) for c, v in series.items()},
            "visible_per_second": series,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"[{path.name}] brightness={summary['mean_brightness']}  tracks={n_tracks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
