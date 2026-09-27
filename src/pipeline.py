"""Part A end to end: video -> tracks + signal state -> aligned scene -> rule-based events."""
from __future__ import annotations

from src import align, events, signal
from src.scene import Scene
from src.tracking import track_video
from src.video import probe

N_KEYFRAMES = 25


def detect_events(video_path: str) -> list[list]:
    info = probe(video_path)
    reference_scene = Scene.load()
    crops = signal.CropCollector(signal.crop_box(reference_scene))
    tracks, keyframes = track_video(info, n_keyframes=N_KEYFRAMES, on_frame=crops)
    alignment = align.estimate(align.load_reference(), align.background(keyframes))
    scene = reference_scene.aligned(alignment)
    state = signal.states(signal.levels(crops.crops, crops.box, scene))
    return events.detect_all(tracks, scene, info.duration, state)
