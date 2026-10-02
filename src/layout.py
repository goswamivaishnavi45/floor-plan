"""Room layout from a fused LiDAR point cloud.

Usage:
    python -m src.layout outputs/c00a170fe1

Part 4a, straightening: the x and z axes point wherever the phone faced when
the recording started, so walls come out at an arbitrary angle. We turn the
cloud about the vertical axis until walls run along x and z.

Writes to the same folder:
    walls_straight.png   wall view after straightening
    layout.json          the rotation angle (later parts add measurements)
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.pointcloud import horizontal_levels, read_ply, save_topdown_views, wall_band

BIN = 0.02   # 2 cm strips, the same resolution as the voxel grid


def rotate_about_vertical(points, angle_deg):
    """Turn points about the y (up) axis by angle_deg. Heights are unchanged."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    out = points.copy()
    out[:, 0] = c * points[:, 0] + s * points[:, 2]
    out[:, 2] = -s * points[:, 0] + c * points[:, 2]
    return out


def spikiness(xz, angle_deg):
    """How strongly the points pile into a few 2 cm strips along x and along z.

    When walls are aligned with the axes, each wall's points fall into one
    strip, giving tall spikes. Summing the squared strip counts rewards spikes:
    one strip of 100 points scores 10,000, ten strips of 10 score only 1,000.
    Divided by N^2 so the score does not depend on how many points there are.
    """
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    x = c * xz[:, 0] + s * xz[:, 1]
    z = -s * xz[:, 0] + c * xz[:, 1]
    score = 0.0
    for v in (x, z):
        counts = np.bincount(np.floor((v - v.min()) / BIN).astype(int))
        score += np.sum(counts.astype(np.float64) ** 2)
    return score / len(xz) ** 2


def find_rotation(points, floor_y, ceiling_y):
    """Angle in degrees that makes the walls run along x and z.

    Only the wall band is used, so floor and ceiling cannot dominate. Walls in
    homes meet at right angles, so turning by 90 degrees gives the same
    picture and 0-90 covers every case. A coarse search every 0.5 degrees
    finds the right neighbourhood, then a fine search every 0.05 degrees
    around it: 0.5 degrees of leftover tilt would skew a 4 m wall by 3.5 cm.
    """
    xz = points[wall_band(points, floor_y, ceiling_y)][:, [0, 2]]
    coarse = np.arange(0.0, 90.0, 0.5)
    best = coarse[np.argmax([spikiness(xz, a) for a in coarse])]
    fine = np.arange(best - 0.5, best + 0.5001, 0.05)
    scores = [spikiness(xz, a) for a in fine]
    return float(fine[np.argmax(scores)]), coarse, fine, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="output folder containing pointcloud.ply")
    args = parser.parse_args()

    folder = Path(args.folder)
    points, colors = read_ply(folder / "pointcloud.ply")
    floor_y, ceiling_y = horizontal_levels(points)

    angle, *_ = find_rotation(points, floor_y, ceiling_y)
    straight = rotate_about_vertical(points, angle)
    save_topdown_views(straight, colors, folder, floor_y, ceiling_y, suffix="_straight")

    layout_path = folder / "layout.json"
    layout = json.loads(layout_path.read_text()) if layout_path.exists() else {}
    layout["rotation_deg"] = round(angle, 2)
    layout_path.write_text(json.dumps(layout, indent=2))
    print(f"{folder.name}: turned by {angle:.2f} degrees -> {folder / 'walls_straight.png'}")


if __name__ == "__main__":
    main()
