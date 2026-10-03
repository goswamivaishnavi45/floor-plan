"""Measure one room from its photo folder and compare with LiDAR (part 7b).

    python scripts/check_photo_room.py data/photos_c7d28f72c6 R5

Runs the photo tier on <folder>/<room>/ (pictures only), with the focal
length from EXIF and, for comparison, from VGGT's own estimate, and prints
the measured room next to the LiDAR measurements stored in manifest.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.photo  # noqa: E402
from scripts.compare_path import umeyama  # noqa: E402
from src.capture import load_capture  # noqa: E402
from src.layout import build_layout  # noqa: E402
from src.mono_depth import DepthModel  # noqa: E402
from src.photo import load_photos, room_points, run_vggt  # noqa: E402
from src.pointcloud import horizontal_levels, save_ply  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("room")
    parser.add_argument("--size", type=int, default=None, help="VGGT input size (default: 350 on CPU)")
    args = parser.parse_args()
    if args.size:
        src.photo.VGGT_SIZE_CPU = args.size

    folder = Path(args.folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    truth = manifest["rooms"][args.room]["lidar"]
    photos = load_photos(folder / args.room)
    started = time.time()
    recon = run_vggt([p["bgr"] for p in photos])
    vggt_seconds = time.time() - started

    # Camera check: the photo_NN files are the manifest's frames, in order.
    capture = load_capture(Path("data") / manifest["capture"])
    frames = manifest["rooms"][args.room]["frames"]
    keep = [k for k, p in enumerate(photos) if p["name"].startswith("photo")]
    centres, true_centres = recon["centre"][keep], capture.poses[frames, :3, 3]
    s, R, t = umeyama(centres, true_centres)
    aligned = (s * (R @ centres.T)).T + t
    view_errors = [np.degrees(np.arccos(np.clip((R @ recon["R"][k][:, 2]) @ capture.poses[i, :3, 2], -1, 1)))
                   for k, i in zip(keep, frames)]
    print(f"cameras: position error {np.sqrt(((aligned - true_centres) ** 2).sum(1).mean()):.2f} m rms, "
          f"viewing direction error {np.round(view_errors, 1).tolist()} deg")
    depth_model = DepthModel()

    print(f"LiDAR {args.room}: width {truth['width']['value']} length {truth['length']['value']} "
          f"area {truth['floor_area']['value']} ceiling {truth['ceiling_height']['value']}")
    for use_exif in (True, False):
        points, colours, info = room_points(photos, recon, depth_model, use_exif_focal=use_exif)
        floor_y, _ = horizontal_levels(points)
        points[:, 1] -= floor_y
        size = src.photo.VGGT_SIZE_CPU
        out = Path("outputs") / f"photo_{folder.name}_{args.room}_{'exif' if use_exif else 'vggt'}_{size}"
        out.mkdir(parents=True, exist_ok=True)
        save_ply(out / "pointcloud.ply", points, colours)
        layout, _, _ = build_layout(points, colours, out)
        rooms = sorted(layout["rooms"], key=lambda r: -r["area_m2_rough"])
        print(f"focal from {'EXIF' if use_exif else 'VGGT'}: {info}")
        for r in rooms[:3]:
            print(f"   room: width {r['width']['value']} length {r['length']['value']} "
                  f"area {r['floor_area']['value']} ceiling {r['ceiling']['value']} "
                  f"missing sides {r['missing_sides']}")
    print(f"VGGT {vggt_seconds:.0f} s, total {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
