#!/usr/bin/env python3
"""Read the signal state of every analysed frame from the proxies and cache it for tools/eval_rules.py.

    python tools/read_signal.py            # -> cache/signal/<stem>.npz, reports/signal/<stem>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import align, config, signal  # noqa: E402
from src.scene import Scene  # noqa: E402
from src.video import iter_frames, probe, sample_frames  # noqa: E402

N_BACKGROUND_FRAMES = 25
PROXY_WIDTH = 1280
COLORS = {signal.RED: "red", signal.YELLOW: "orange", signal.GREEN: "green", signal.UNKNOWN: "lightgray"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxies", type=Path, default=config.ROOT / "proxy")
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "signal")
    ap.add_argument("--plots", type=Path, default=config.ROOT / "reports" / "signal")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.plots.mkdir(parents=True, exist_ok=True)

    reference, scene = align.load_reference(), Scene.load()
    for proxy in sorted(args.proxies.glob("*.mp4")):
        info = probe(proxy)
        crops = signal.CropCollector(signal.crop_box(scene))
        for _, frame in iter_frames(info, config.STRIDE, PROXY_WIDTH):
            crops(frame)
        aligned = scene.aligned(align.estimate(reference, align.background(sample_frames(proxy, N_BACKGROUND_FRAMES))))
        lv = signal.levels(crops.crops, crops.box, aligned)
        state = signal.states(lv)
        np.savez_compressed(args.out / f"{proxy.stem}.npz", levels=lv, state=state)

        t = np.arange(len(state)) * config.STRIDE / info.fps
        fig, (a, b) = plt.subplots(2, 1, figsize=(16, 5), sharex=True, height_ratios=[3, 1])
        for j, c in enumerate(("red", "orange", "green")):
            a.plot(t, lv[:, j], color=c, lw=0.8)
        a.set_ylabel("colour strength")
        b.scatter(t, np.zeros_like(t), c=[COLORS[s] for s in state], marker="|", s=400)
        b.set_yticks([])
        b.set_xlabel("s")
        fig.suptitle(proxy.stem)
        fig.tight_layout()
        fig.savefig(args.plots / f"{proxy.stem}.png", dpi=80)
        plt.close(fig)
        share = {n: round(float((state == v).mean()), 2) for n, v in
                 (("red", signal.RED), ("yellow", signal.YELLOW), ("green", signal.GREEN), ("unknown", signal.UNKNOWN))}
        print(f"[{proxy.stem}] {len(state)} samples, share {share}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
