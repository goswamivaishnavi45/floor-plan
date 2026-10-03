"""One command per capture.

    python run.py <capture folder or video file> [--out outputs] [--rotate cw]

Detects the input tier from what is given, runs the pipeline and writes, in
outputs/<capture name>/:
    result.json   measurements in the published format (schema/result.schema.json)
    plan.png      the floor plan
plus debug pictures and the point cloud for inspection.

Tiers:
    lidar   Stray Scanner export: depth/, confidence/, odometry.csv, rgb.mp4
    video   a video file (.mp4 / .mov), or a folder holding one. Rooms are
            measured per video segment (see src/video.py). Use --rotate
            cw|ccw|180 for videos stored sideways, e.g. Stray Scanner's
            rgb.mp4; iPhone Camera videos are upright already.
    photo   a folder of sub-folders, one per room, holding 2-8 photos; a doorway
            photo saved in two room folders links those rooms. Rooms are
            measured only if they pass a quality check (see src/photo.py)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.layout import build_layout
from src.plan import draw_plan
from src.pointcloud import build_pointcloud
from src.damage import DamageDetector, lidar_damage, picture_damage, to_outputs
from src.photo import IMAGE_TYPES as PHOTO_TYPES, load_photos, measure_photo_folders
from src.result import build_result, validate
from src.video import measure_pieces, video_pieces

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".heic"}
VIDEO_TYPES = {".mp4", ".mov", ".m4v"}


VIDEO_SCALE_SIGMA = 0.10   # 1-sigma size uncertainty of the video tier: piece sizes were
                           # -6% to +18% off on c00a170fe1 (scripts/check_pieces.py)
ROTATIONS = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}


def detect_tier(folder):
    if folder.is_file():
        return "video" if folder.suffix.lower() in VIDEO_TYPES else None
    if (folder / "depth").is_dir() and (folder / "odometry.csv").exists():
        return "lidar"
    files = [f for f in folder.iterdir() if f.is_file()]
    if any(f.suffix.lower() in VIDEO_TYPES for f in files):
        return "video"
    subfolders = [d for d in folder.iterdir() if d.is_dir()]
    if subfolders and all(any(f.suffix.lower() in IMAGE_TYPES for f in d.iterdir()) for d in subfolders):
        return "photo"
    return None


def add_damage(result, detections):
    result["damage"], result["concealed_damage_flags"], result["scope"] = to_outputs(detections, result)
    result["warnings"] = [w for w in result["warnings"] if not w.startswith("Damage detection is not")]
    result["warnings"].append(f"Damage: {len(result['damage'])} region(s) found by an open-vocabulary detector "
                              "(OWLv2) on sharp frames; not validated on staged damage.")


def run_lidar(folder, out_dir, step, drift_correction=True, damage=True):
    started = time.time()
    step(1, "building 3D points" + ("" if drift_correction else " (drift correction off)"))
    points, colors, capture, stride = build_pointcloud(folder, out_dir, drift_correction=drift_correction)
    step(2, "finding walls and rooms")
    layout, grid, room_labels = build_layout(points, colors, out_dir)
    step(3, "writing results")
    info = {"id": folder.name, "tier": "lidar", "source": "Stray Scanner",
            "frames": capture.num_frames, "frames_used": len(range(0, capture.num_frames, stride)),
            "processing_seconds": 0.0}
    result = build_result(layout, grid, room_labels, info)
    result["capture"]["processing_seconds"] = round(time.time() - started, 1)
    result["warnings"].append("Drift: " + ("frames fused in 20 s chunks re-anchored to the floor where "
                                           "chunks overlap (src/drift.py)" if drift_correction
                                           else "poses used as recorded (--no-drift-correction)") + ".")
    if damage:
        print("      searching for damage")
        add_damage(result, lidar_damage(folder, layout, result, DamageDetector()))
    result["capture"]["processing_seconds"] = round(time.time() - started, 1)
    return result


def run_video(video, capture_id, out_dir, step, rotate, damage=True):
    started = time.time()
    step(1, "reconstructing video segments")
    pieces, images, info = video_pieces(video, out_dir, rotate)
    step(2, "finding walls and rooms in each segment")
    capture = {"id": capture_id, "tier": "video", "source": video.name,
               "frames": info["frames"], "frames_used": info["frames_placed"], "processing_seconds": 0.0}
    result, segments = measure_pieces(pieces, images, out_dir, capture, VIDEO_SCALE_SIGMA)
    (out_dir / "video_segments.json").write_text(json.dumps(segments, indent=2))
    if result is None:
        sys.exit("error: no part of the video could be reconstructed")
    if damage:
        print("      searching for damage")
        frames = sorted(images.items())
        frames = [frames[k] for k in np.linspace(0, len(frames) - 1, min(20, len(frames))).astype(int)]
        add_damage(result, picture_damage([(f"frame {i}", bgr, None) for i, bgr in frames], DamageDetector()))
    step(3, "writing results")
    result["capture"]["processing_seconds"] = round(time.time() - started, 1)
    return result


def run_photo(folder, out_dir, step, damage=True):
    started = time.time()
    step(1, "reconstructing each room from its photos")
    capture = {"id": folder.name, "tier": "photo", "source": "photo folders", "processing_seconds": 0.0}
    result = measure_photo_folders(folder, out_dir, capture)
    step(2, "linking rooms through shared doorway photos")
    if damage:
        print("      searching for damage")
        room_ids = {r["name"]: r["id"] for r in result["rooms"]}
        pictures = [(f"{d.name}/{p['name']}", p["bgr"], room_ids.get(d.name))
                    for d in sorted(folder.iterdir()) if d.is_dir() and d.name in room_ids
                    for p in load_photos(d)]
        add_damage(result, picture_damage(pictures, DamageDetector()))
    step(3, "writing results")
    result["capture"]["processing_seconds"] = round(time.time() - started, 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", help="capture folder or video file")
    parser.add_argument("--out", default="outputs", help="output root folder (default: outputs)")
    parser.add_argument("--rotate", choices=sorted(ROTATIONS), help="turn sideways video frames upright")
    parser.add_argument("--no-damage", action="store_true", help="skip damage detection (faster)")
    parser.add_argument("--no-drift-correction", action="store_true",
                        help="LiDAR tier: use the recorded poses as they are (for the on/off comparison)")
    args = parser.parse_args()

    folder = Path(args.capture)
    if not folder.exists():
        sys.exit(f"error: {folder} does not exist")
    tier = detect_tier(folder)
    if tier is None:
        sys.exit(f"error: could not tell what kind of capture {folder} is (see `python run.py --help`)")
    video = None
    capture_id = folder.name
    if tier == "video":
        video = folder if folder.is_file() else next(
            f for f in sorted(folder.iterdir()) if f.suffix.lower() in VIDEO_TYPES)
        if folder.is_file():   # data/x/rgb.mp4 -> x_video
            capture_id = f"{video.parent.name}_video"
    out_dir = Path(args.out) / capture_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Detected: {tier} capture in {folder}")
    clock = {"t": time.time()}

    def step(n, label):
        now = time.time()
        if n > 1:
            print(f"      done in {now - clock['t']:.0f} s")
        clock["t"] = now
        print(f"[{n}/3] {label} ...")

    if tier == "video":
        result = run_video(video, capture_id, out_dir, step, ROTATIONS.get(args.rotate), not args.no_damage)
    elif tier == "photo":
        result = run_photo(folder, out_dir, step, not args.no_damage)
    else:
        result = run_lidar(folder, out_dir, step, drift_correction=not args.no_drift_correction,
                           damage=not args.no_damage)
    print(f"      done in {time.time() - clock['t']:.0f} s")
    validate(result)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    draw_plan(result, out_dir / "plan.png")

    total = result["property"]["total_floor_area"]
    area = f"{total['value']:.1f} +- {total['pm95']:.1f} m2" if total["value"] is not None else "not measured"
    print(f"{result['property']['room_count']} rooms, {len(result['openings'])} doors, "
          f"{len(result['adjacency'])} connections, total floor area {area}, "
          f"{result['capture']['processing_seconds']:.0f} s")
    print(f"-> {out_dir / 'result.json'}")
    print(f"-> {out_dir / 'plan.png'}")
    for warning in result["warnings"]:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
