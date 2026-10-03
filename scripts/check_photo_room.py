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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.layout import build_layout  # noqa: E402
from src.mono_depth import DepthModel  # noqa: E402
from src.photo import load_photos, room_points, run_vggt  # noqa: E402
from src.pointcloud import horizontal_levels, save_ply  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("room")
    args = parser.parse_args()

    folder = Path(args.folder)
    truth = json.loads((folder / "manifest.json").read_text())["rooms"][args.room]["lidar"]
    photos = load_photos(folder / args.room)
    started = time.time()
    recon = run_vggt([p["bgr"] for p in photos])
    vggt_seconds = time.time() - started
    depth_model = DepthModel()

    print(f"LiDAR {args.room}: width {truth['width']['value']} length {truth['length']['value']} "
          f"area {truth['floor_area']['value']} ceiling {truth['ceiling_height']['value']}")
    for use_exif in (True, False):
        points, colours, info = room_points(photos, recon, depth_model, use_exif_focal=use_exif)
        floor_y, _ = horizontal_levels(points)
        points[:, 1] -= floor_y
        out = Path("outputs") / f"photo_{folder.name}_{args.room}_{'exif' if use_exif else 'vggt'}"
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
