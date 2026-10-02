"""One command per capture.

    python run.py <capture folder> [--out outputs]

Detects the input tier from what is in the folder, runs the pipeline and
writes, in outputs/<capture name>/:
    result.json   measurements in the published format (schema/result.schema.json)
    plan.png      the floor plan
plus debug pictures and the point cloud for inspection.

Tiers:
    lidar   Stray Scanner export: depth/, confidence/, odometry.csv, rgb.mp4
    video   a folder with one video file (.mp4 / .mov)            [not yet supported]
    photo   a folder of sub-folders, one per room, holding images [not yet supported]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from src.layout import build_layout
from src.plan import draw_plan
from src.pointcloud import build_pointcloud
from src.result import build_result, validate

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".heic"}
VIDEO_TYPES = {".mp4", ".mov", ".m4v"}


def detect_tier(folder):
    if (folder / "depth").is_dir() and (folder / "odometry.csv").exists():
        return "lidar"
    files = [f for f in folder.iterdir() if f.is_file()]
    if any(f.suffix.lower() in VIDEO_TYPES for f in files):
        return "video"
    subfolders = [d for d in folder.iterdir() if d.is_dir()]
    if subfolders and all(any(f.suffix.lower() in IMAGE_TYPES for f in d.iterdir()) for d in subfolders):
        return "photo"
    return None


def run_lidar(folder, out_dir, step):
    started = time.time()
    step(1, "building 3D points")
    points, colors, capture, stride = build_pointcloud(folder, out_dir)
    step(2, "finding walls and rooms")
    layout, grid, room_labels = build_layout(points, colors, out_dir)
    step(3, "writing results")
    info = {"id": folder.name, "tier": "lidar", "source": "Stray Scanner",
            "frames": capture.num_frames, "frames_used": len(range(0, capture.num_frames, stride)),
            "processing_seconds": 0.0}
    result = build_result(layout, grid, room_labels, info)
    result["capture"]["processing_seconds"] = round(time.time() - started, 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", help="capture folder")
    parser.add_argument("--out", default="outputs", help="output root folder (default: outputs)")
    args = parser.parse_args()

    folder = Path(args.capture)
    if not folder.is_dir():
        sys.exit(f"error: {folder} is not a folder")
    tier = detect_tier(folder)
    if tier is None:
        sys.exit(f"error: could not tell what kind of capture {folder} is (see `python run.py --help`)")
    if tier != "lidar":
        sys.exit(f"error: {tier} captures are not supported yet")

    out_dir = Path(args.out) / folder.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Detected: {tier} capture in {folder}")
    clock = {"t": time.time()}

    def step(n, label):
        now = time.time()
        if n > 1:
            print(f"      done in {now - clock['t']:.0f} s")
        clock["t"] = now
        print(f"[{n}/3] {label} ...")

    result = run_lidar(folder, out_dir, step)
    print(f"      done in {time.time() - clock['t']:.0f} s")
    validate(result)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    draw_plan(result, out_dir / "plan.png")

    total = result["property"]["total_floor_area"]
    print(f"{result['property']['room_count']} rooms, {len(result['openings'])} doors, "
          f"total floor area {total['value']:.1f} +- {total['pm95']:.1f} m2, "
          f"{result['capture']['processing_seconds']:.0f} s")
    print(f"-> {out_dir / 'result.json'}")
    print(f"-> {out_dir / 'plan.png'}")
    for warning in result["warnings"]:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
