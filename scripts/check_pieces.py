"""Check part 6c against LiDAR: is each video piece the right size and the
right way up?

    python scripts/check_pieces.py data/c00a170fe1 --frames 300

Runs the video tier on the colour video of a LiDAR capture (frames turned
upright, COLMAP pieces, AI-depth scale, up direction) and, per piece,
compares with the ARKit poses of the same frames:

    path_m        how far the camera moved within the piece (true metres);
                  short pieces give unreliable checks
    scale_error   estimated metres per COLMAP unit / true - 1
    up_error_deg  angle between estimated up and true up, before and after
                  levelling the floor
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.compare_depth import upright_rotation  # noqa: E402
from scripts.compare_path import umeyama  # noqa: E402
from src.capture import load_capture  # noqa: E402
from src.mono_depth import DepthModel  # noqa: E402
from src.pointcloud import save_ply  # noqa: E402
from src.video import camera_up, pick_frames, piece_points, reconstruct_pieces, scale_piece  # noqa: E402


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    capture = load_capture(args.capture)
    out = Path(args.out) / capture.root.name
    started = time.time()
    frames = pick_frames(capture.root / "rgb.mp4", args.frames)
    rotation = upright_rotation(capture.poses[frames[len(frames) // 2][0]])
    if rotation is not None:
        frames = [(i, cv2.rotate(f, rotation)) for i, f in frames]
    images = dict(frames)
    pieces = reconstruct_pieces(frames, out / "colmap")
    model = DepthModel()

    report = []
    for n, piece in enumerate(pieces):
        scale_piece(piece, images, model)
        placed = sorted(piece["cameras"])
        row = {"piece": n, "frames": f"{placed[0]}-{placed[-1]}", "placed": len(placed)}
        centres = np.array([piece["cameras"][i]["center"] for i in placed])
        truth = capture.poses[placed, :3, 3]
        row["path_m"] = round(float(np.linalg.norm(np.diff(truth, axis=0), axis=1).sum()), 2)
        if piece["metres_per_unit"] is None or len(placed) < 3:
            row["note"] = "too few spots to scale"
            report.append(row)
            continue
        s_true, R, _ = umeyama(centres, truth)
        row["scale_error"] = round(piece["metres_per_unit"] / s_true - 1, 3)
        true_up = R.T @ np.array([0.0, 1.0, 0.0])          # true up in COLMAP's frame
        row["up_error_deg_camera"] = round(angle(camera_up(piece), true_up), 1)
        points, colours = piece_points(piece, images)
        if points is None:
            row["rejected"] = piece["rejected"]
            report.append(row)
            continue
        levelled_up = piece["to_level"].T @ np.array([0.0, 1.0, 0.0])
        row["up_error_deg_floor"] = round(angle(levelled_up, true_up), 1)
        save_ply(out / f"video_piece_{n}.ply", points, colours)
        report.append(row)
    print(json.dumps({"capture": capture.root.name, "pieces": len(pieces),
                      "seconds": round(time.time() - started), "per_piece": report}, indent=2))


if __name__ == "__main__":
    main()
