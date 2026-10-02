"""Did COLMAP find the camera path? Compare it with the LiDAR (ARKit) path.

    python scripts/compare_path.py data/c00a170fe1 --frames 150

Uses only the colour video of a LiDAR capture: picks sharp frames, turns
them upright, runs COLMAP, then lines COLMAP's camera centres up with the
ARKit centres of the same frames (best rotation + shift + one size factor,
Umeyama's method) and reports how far apart they are.

    placed        frames COLMAP could place, out of those given
    path_m        length of the true (ARKit) path over the placed frames
    rms_error_m   distance between the two paths after lining them up
    error_pct     rms_error_m as a share of path_m
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
from src.capture import load_capture  # noqa: E402
from src.video import pick_frames, reconstruct_cameras  # noqa: E402


def umeyama(source, target):
    """Scale s, rotation R, shift t minimising |s R source + t - target|."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    a, b = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(b.T @ a / len(source))
    flip = np.eye(3)
    flip[2, 2] = np.sign(np.linalg.det(u @ vt))
    R = u @ flip @ vt
    s = np.trace(np.diag(d) @ flip) / a.var(axis=0).sum()
    return s, R, mu_t - s * R @ mu_s


def draw_paths(true_xz, colmap_xz, path):
    both = np.vstack([true_xz, colmap_xz])
    low, high = both.min(axis=0) - 0.3, both.max(axis=0) + 0.3
    scale = 600 / max(high - low)
    image = np.full((int((high - low)[1] * scale) + 40, int((high - low)[0] * scale) + 40, 3), 255, np.uint8)

    def px(p):
        return int((p[0] - low[0]) * scale) + 20, int((p[1] - low[1]) * scale) + 20

    for xz, colour in ((true_xz, (60, 160, 60)), (colmap_xz, (40, 40, 220))):
        for a, b in zip(xz[:-1], xz[1:]):
            cv2.line(image, px(a), px(b), colour, 2, cv2.LINE_AA)
    cv2.putText(image, "green = LiDAR/ARKit path, red = COLMAP from video only", (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(path), image)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    capture = load_capture(args.capture)
    out = Path(args.out) / capture.root.name
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frames = pick_frames(capture.root / "rgb.mp4", args.frames)
    rotation = upright_rotation(capture.poses[frames[len(frames) // 2][0]])
    if rotation is not None:
        frames = [(i, cv2.rotate(f, rotation)) for i, f in frames]
    picked = time.time()
    cameras, _ = reconstruct_cameras(frames, out / "colmap")
    finished = time.time()

    placed = sorted(cameras)
    report = {"capture": capture.root.name, "frames": len(frames), "placed": len(placed),
              "seconds_picking": round(picked - started), "seconds_colmap": round(finished - picked)}
    if len(placed) >= 3:
        colmap_centres = np.array([cameras[i]["center"] for i in placed])
        true_centres = capture.poses[placed, :3, 3]
        s, R, t = umeyama(colmap_centres, true_centres)
        aligned = (s * (R @ colmap_centres.T)).T + t
        error = np.linalg.norm(aligned - true_centres, axis=1)
        path_m = np.linalg.norm(np.diff(true_centres, axis=0), axis=1).sum()
        report.update({"path_m": round(float(path_m), 2),
                       "rms_error_m": round(float(np.sqrt((error ** 2).mean())), 3),
                       "max_error_m": round(float(error.max()), 3),
                       "error_pct": round(float(100 * np.sqrt((error ** 2).mean()) / path_m), 2),
                       "colmap_units_per_metre": round(float(1 / s), 4),
                       "focal_px_fullres": round(float(cameras[placed[0]]["K"][0, 0]), 1),
                       "true_focal_px": round(float(capture.K_depth[placed[0]][0, 0] * 1920 / 256), 1)})
        draw_paths(true_centres[:, [0, 2]], aligned[:, [0, 2]], out / "path_check.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
