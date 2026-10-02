"""How good is the AI depth model on our data? Compare it with LiDAR.

    python scripts/compare_depth.py data/c00a170fe1 --frames 30

For sharp frames spread over a LiDAR capture, run the depth model on the
colour frame only, and compare with the LiDAR depth of the same frame
(high-confidence pixels). Reports, per orientation of the input picture:

    scale     median of model / LiDAR over all pixels: 1.00 = right size,
              0.94 = the model makes everything 6% too close
    spread    how much that ratio varies from frame to frame (std), which
              limits how well one scale correction can fix every frame
    AbsRel    mean |model - LiDAR| / LiDAR per pixel, before and after
              dividing by the scale (the remaining error is shape error)

Stray Scanner stores frames sideways (the phone is held upright, the sensor
reads landscape). Both the stored and the upright orientation are tested,
since the model was trained on upright photos.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.capture import DEPTH_HEIGHT, DEPTH_WIDTH, load_capture  # noqa: E402
from src.mono_depth import DepthModel  # noqa: E402
from src.video import pick_frames  # noqa: E402


def upright_rotation(pose):
    """cv2 rotation that turns the stored frame upright, from the camera pose:
    the image axis pointing most downward in the world should become the
    image's bottom."""
    right, down = pose[:3, 0], pose[:3, 1]
    world_down = np.array([0.0, -1.0, 0.0])
    scores = {"none": down @ world_down, "cw": right @ world_down,
              "ccw": -right @ world_down, "180": -down @ world_down}
    best = max(scores, key=scores.get)
    return {"none": None, "cw": cv2.ROTATE_90_CLOCKWISE,
            "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}[best]


UNDO = {cv2.ROTATE_90_CLOCKWISE: cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE: cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_180: cv2.ROTATE_180}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture")
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    capture = load_capture(args.capture)
    model = DepthModel()
    frames = pick_frames(capture.root / "rgb.mp4", args.frames)

    stats = {"stored": [], "upright": []}
    pixels = {"stored": ([], []), "upright": ([], [])}
    seconds = []
    for index, bgr in frames:
        if not capture.has_depth[index]:
            continue
        lidar = capture.depth(index)
        valid = (capture.confidence(index) >= 2) & (lidar > 0.2) & (lidar < 5.0)
        rotation = upright_rotation(capture.poses[index])
        for name in stats:
            started = time.time()
            if name == "upright" and rotation is not None:
                pred = model.predict(cv2.rotate(bgr, rotation))
                pred = cv2.rotate(pred, UNDO[rotation])
            else:
                pred = model.predict(bgr)
            seconds.append(time.time() - started)
            pred = cv2.resize(pred, (DEPTH_WIDTH, DEPTH_HEIGHT), interpolation=cv2.INTER_AREA)
            p, l = pred[valid], lidar[valid]
            stats[name].append(float(np.median(p / l)))
            pixels[name][0].append(p)
            pixels[name][1].append(l)

    report = {"capture": capture.root.name, "frames": len(stats["stored"]),
              "seconds_per_frame": round(float(np.mean(seconds)), 2)}
    for name, ratios in stats.items():
        p, l = np.concatenate(pixels[name][0]), np.concatenate(pixels[name][1])
        scale = float(np.median(p / l))
        report[name] = {
            "scale": round(scale, 4),
            "spread_between_frames": round(float(np.std(ratios)), 4),
            "absrel_raw": round(float(np.mean(np.abs(p - l) / l)), 4),
            "absrel_after_scale": round(float(np.mean(np.abs(p / scale - l) / l)), 4),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
