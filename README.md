# NeuroLens — WIUT Hackathon 2026 CV Track

Offline intersection video analytics for the elimination round.

* **Part A** — temporal event segments `[start_sec, end_sec, label]` (macro F1 @ tIoU 0.3 / 0.5 / 0.7).
* **Part B** — causal online risk estimate in `[0, 1]` from pairwise time-to-collision.

Do **not** modify `run_submission.py` or `evaluate.py`.

## Package layout

```
.
├── solution.py              # harness entry (detect_events, RiskEstimator)
├── run_submission.py        # organizers' harness (unchanged)
├── evaluate.py              # organizers' scorer (unchanged)
├── requirements.txt
├── predictions_samples.json # frozen sample-run output (reproducibility)
├── README.md
├── weights/
│   └── yolo11s.pt           # YOLO11s COCO (~19 MB, ≤ 5 GB limit)
├── scene/
│   ├── scene.json           # intersection geometry (lanes, crossings, signal boxes)
│   ├── reference.jpg        # alignment reference frame
│   └── camera.md            # scene / labelling notes
├── src/                     # pipeline, tracking, rules, risk
├── labels/
│   ├── dev_labels.json      # merged GT for the 4 sample videos
│   └── raw/                 # per-video annotation sources
└── tools/                   # local eval / annotation helpers (not used by the harness)
```

Sample videos (`samples/`) are **not** shipped; place the four `.MP4` files locally when validating.

## Install

Python 3.11+ recommended. On the jury node (CUDA T4 16 GB):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Weights are already under `weights/yolo11s.pt` (no download step). The solution sets `YOLO_OFFLINE=true` so Ultralytics does not touch the network.

## Run

Official-style run (Part A + Part B, time budget = 3× video duration):

```bash
python run_submission.py --videos /path/to/videos --out predictions.json --team neurolens
```

Local checks on the four samples:

```bash
python run_submission.py --videos samples --out predictions_samples.json --team neurolens
python evaluate.py --gt labels/dev_labels.json --pred predictions_samples.json
```

Part A only (faster debug):

```bash
python run_submission.py --videos samples --out predictions.json --team neurolens --no-risk
```

## Approach

### Part A — detect_events

1. **Decode** every `STRIDE=3` frame at 1920 px width (`src/video.py`).
2. **Detect + track** with YOLO11s + ByteTrack (`src/tracking.py`). COCO classes kept: person, bicycle, car, motorcycle, bus, truck.
3. **Align** the live background to `scene/reference.jpg` with SIFT + RANSAC (`src/align.py`) and warp the annotated geometry (`src/scene.py`).
4. **Signal state** from colour contrast on the opposite-approach heads (`src/signal.py`); sample timing matches the decode stride.
5. **Rule-based events** on aligned tracks (`src/events/`):
   - `jaywalking`, `failure_to_yield`, `solid_line_crossing`, `stopped_vehicle`
   - `red_light`, `stop_line`, `illegal_turn`
   - Classes we do not fire on this scene (`accident`, `near_miss`, `wrong_way`, `illegal_u_turn`, `congestion`, `road_obstacle`, `fire_smoke`) stay in `CLASSES` for harness compatibility but are never predicted (avoids adding empty predicted-only classes to the score union when they also appear only as FN in GT).

**Learned vs rule-based:** detection/tracking weights are learned (COCO-pretrained YOLO11s + ByteTrack). Alignment, signal reading, and all event classes are hand-written geometric/temporal rules on tracks + `scene/scene.json`. Part B is a rule-based TTC risk (no accident-trained model).

### Part B — RiskEstimator

Causal online TTC risk (`src/risk.py`): ByteTrack every 5th frame at imgsz 640; pairwise closing TTC → `exp(-ttc / 2.0)` clipped to `[0, 1]`. Uses only past frames.

## Weights

| File | Source | Size | License |
|------|--------|------|---------|
| `weights/yolo11s.pt` | Ultralytics YOLO11s, COCO pretrained | ~19 MB | AGPL-3.0 (Ultralytics) |

Shipped in-repo (no `weights/download.sh`). No fine-tuning. Total weights ≪ 5 GB.

## Datasets

| Dataset | Use | License |
|---------|-----|---------|
| MS COCO (via Ultralytics YOLO11s checkpoint) | Pretrained detector only | COCO terms / Ultralytics distribution |
| Own annotations of the 4 sample videos (`labels/`) | Dev eval / rule tuning | Team NeuroLens |

No other external video datasets were used for training.

## Reproducibility

| Item | Value |
|------|--------|
| Seed | `0` (`src/config.py` → `seed_everything`) |
| Detector | YOLO11s, imgsz 1280 (Part A) / 640 (Part B), conf 0.2 / 0.25 |
| Tracker | ByteTrack (`src/bytetrack.yaml`) |
| Decode | stride 3, width 1920 |
| Frozen sample output | `predictions_samples.json` |

Non-determinism: GPU/MPS kernels and OpenCV SIFT can differ slightly across devices; on the **same** machine two runs should match within floating-point noise. Official sample freeze: `predictions_samples.json`.

## Licenses / attribution

- Solution code: team NeuroLens (WIUT Hackathon 2026).
- Ultralytics YOLO11 + bundled ByteTrack: [AGPL-3.0](https://github.com/ultralytics/ultralytics).
- OpenCV, NumPy, PyTorch: BSD / BSD-3 / BSD-style.

## Team

| Name | Contribution |
|------|----------------|
| Moldagulov Arman | Part A pipeline, event rules, tuning |
| Maxkamov Shoxrux | Part B `RiskEstimator`, packaging / harness |
| Vakhobov Abdulaziz | Website, scene geometry, sample labelling (shared) |

Team id used in predictions: **neurolens**.
