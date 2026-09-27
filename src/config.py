"""Global settings shared by the pipeline and the tools."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
SCENE_DIR = ROOT / "scene"
CACHE_DIR = ROOT / "cache"

SEED = 0

# Normalised coordinates are stretched by this in x so distances are isotropic on a 16:9 frame;
# all geometric thresholds are in these units (fractions of the frame height).
ASPECT = 16 / 9

# Frames are decoded at full rate (inter-frame codec) but only every STRIDE-th one is analysed.
# 29.97 fps / 3 ~= 10 analysed frames per second.
STRIDE = 3
# Width the decoder scales frames to before detection (source is 3840x2160).
DECODE_WIDTH = 1920

DETECTOR_WEIGHTS = WEIGHTS_DIR / "yolo11s.pt"
DETECTOR_IMGSZ = 1280
DETECTOR_CONF = 0.2
TRACKER_CONFIG = ROOT / "src" / "bytetrack.yaml"

# COCO class id -> our name. Everything else the detector outputs is ignored.
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLASSES = ("car", "bus", "truck", "motorcycle")
