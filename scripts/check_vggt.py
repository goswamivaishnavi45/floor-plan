"""Feasibility test: can VGGT run on this laptop, and is it accurate?

    python scripts/check_vggt.py data/c00a170fe1 --frames 8

VGGT (Wang et al., CVPR 2025; weights facebook/VGGT-1B, CC-BY-NC-4.0) takes
several pictures at once and predicts, for all of them together, camera
positions, lens focal length and per-pixel depth that agree with each other.
Its units are arbitrary (one overall size factor is unknown), like COLMAP's.

Uses only the colour video of a LiDAR capture (frames spread over the whole
video, turned upright) and compares with the capture's LiDAR:

    seconds, peak_ram_gb   cost on this machine
    path_error_pct         camera centres vs ARKit after best rotation, shift
                           and size (Umeyama), as % of the path length
    focal_error_pct        predicted focal length vs the true one
    frame_scale_spread     std over frames of median(VGGT depth / LiDAR depth),
                           relative to its mean: how consistent the depth
                           maps are with each other (Depth Anything: 0.22)
    absrel_after_scale     per-pixel depth error after one overall size factor
                           (Depth Anything per frame: 0.15)
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.compare_depth import UNDO, upright_rotation  # noqa: E402
from scripts.compare_path import umeyama  # noqa: E402
from src.capture import load_capture  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "vggt-1b"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture")
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    capture = load_capture(args.capture)
    total = capture.num_frames
    indices = [int(i) for i in np.linspace(10, total - 10, args.frames)]
    rotation = upright_rotation(capture.poses[indices[len(indices) // 2]])

    # Read the chosen frames (sequentially, see src/capture.py) and save them upright.
    wanted, frames = set(indices), {}
    video = cv2.VideoCapture(str(capture.root / "rgb.mp4"))
    i = 0
    while i <= max(indices) and video.grab():
        if i in wanted:
            frames[i] = cv2.rotate(video.retrieve()[1], rotation)
        i += 1
    tmp = Path(tempfile.mkdtemp())
    paths = []
    for i in indices:
        path = tmp / f"{i:06d}.jpg"
        cv2.imwrite(str(path), frames[i])
        paths.append(str(path))

    process = psutil.Process()
    started = time.time()
    model = VGGT.from_pretrained(str(MODEL_DIR)).eval()
    loaded = time.time()
    # "pad" keeps every pixel: the long side becomes 518 px and the short side
    # is padded with white to a square. (The default "crop" cuts the top and
    # bottom off, which would misalign the comparison with LiDAR.)
    images = load_and_preprocess_images(paths, mode="pad")   # (S, 3, 518, 518)
    full_h, full_w = frames[indices[0]].shape[:2]
    content_w = round(full_w * (518 / full_h) / 14) * 14 if full_h >= full_w else 518
    content_h = 518 if full_h >= full_w else round(full_h * (518 / full_w) / 14) * 14
    left, top = (518 - content_w) // 2, (518 - content_h) // 2
    with torch.no_grad():
        pred = model(images)
    finished = time.time()
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    extrinsic, intrinsic = extrinsic[0].numpy(), intrinsic[0].numpy()
    depth = pred["depth"][0, ..., 0].numpy()[:, top:top + content_h, left:left + content_w]

    report = {"frames": len(indices), "input_size": list(images.shape[-2:]),
              "seconds_load": round(loaded - started, 1), "seconds_run": round(finished - loaded, 1),
              "peak_ram_gb": round(process.memory_info().peak_wset / 1e9, 2)
              if hasattr(process.memory_info(), "peak_wset") else None}

    # Camera path vs ARKit.
    centres = np.array([-e[:, :3].T @ e[:, 3] for e in extrinsic])
    truth = capture.poses[indices, :3, 3]
    s, R, t = umeyama(centres, truth)
    aligned = (s * (R @ centres.T)).T + t
    path_m = np.linalg.norm(np.diff(truth, axis=0), axis=1).sum()
    report["path_m"] = round(float(path_m), 2)
    report["path_error_pct"] = round(float(100 * np.sqrt(((aligned - truth) ** 2).sum(1).mean()) / path_m), 2)

    # Focal length vs the true one (VGGT works on a resized image).
    true_f = capture.K_depth[indices[0]][0, 0] * 1920 / 256
    report["focal_error_pct"] = round(float(100 * (intrinsic[0, 0, 0] * full_w / content_w / true_f - 1)), 2)

    # Depth vs LiDAR, in the LiDAR's 256x192 grid.
    ratios, p_all, l_all = [], [], []
    for k, i in enumerate(indices):
        lidar = capture.depth(i)
        valid = (capture.confidence(i) >= 2) & (lidar > 0.2) & (lidar < 5.0)
        d = cv2.rotate(depth[k], UNDO[rotation]) if rotation is not None else depth[k]
        d = cv2.resize(d, (lidar.shape[1], lidar.shape[0]), interpolation=cv2.INTER_AREA)
        ratios.append(float(np.median(d[valid] / lidar[valid])))
        p_all.append(d[valid])
        l_all.append(lidar[valid])
    p, l = np.concatenate(p_all), np.concatenate(l_all)
    scale = float(np.median(p / l))
    report["frame_scale_spread"] = round(float(np.std(ratios) / np.mean(ratios)), 4)
    report["absrel_after_scale"] = round(float(np.mean(np.abs(p / scale - l) / l)), 4)
    report["scale_from_depth_vs_path"] = round(float((1 / scale) / s - 1), 4)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
