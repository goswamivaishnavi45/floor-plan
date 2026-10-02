"""Room layout from a fused LiDAR point cloud.

Usage:
    python -m src.layout outputs/c00a170fe1

Part 4a, straightening: the x and z axes point wherever the phone faced when
the recording started, so walls come out at an arbitrary angle. We turn the
cloud about the vertical axis until walls run along x and z.

Part 4b, walls: in the straightened cloud each wall sits at one x (or one z)
value, so wall-band points pile into a spike in the histogram of x (or z).
Each spike is followed along its length to find where the wall starts, stops,
and has gaps; short or low runs (furniture) are dropped.

Writes to the same folder:
    walls_straight.png   wall view after straightening
    walls_found.png      detected walls drawn over the wall view
    layout.json          rotation angle and wall list (later parts add rooms)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.pointcloud import horizontal_levels, read_ply, save_topdown_views, wall_band

BIN = 0.02   # 2 cm strips, the same resolution as the voxel grid

# Wall detection settings (part 4b)
PEAK_HALF_WIDTH = 0.03   # a wall's points lie within +-3 cm of its centre line;
                         # spikes closer than this are treated as one wall
MIN_PEAK_POINTS = 500    # ~40 cm of wall at half the band height; weaker spikes are clutter
ALONG_CELL = 0.05        # follow each wall in 5 cm steps along its length
MIN_CELL_COVER = 0.5     # a 5 cm step counts as wall only if its points reach half
                         # the band height; flat things like a sofa seat do not
MAX_GAP = 0.30           # a hole longer than 30 cm ends the wall (doorway, window, room change)
MIN_LENGTH = 0.40        # shorter runs are furniture sides or clutter
MIN_HEIGHT_COVER = 0.6   # a wall must have points over 60% of the band height;
                         # sofas and low cupboards reach only part of it
HEIGHT_CELL = 0.10       # height coverage is measured in 10 cm slices
MIN_PROMINENCE = 1.5     # a wall is thin: its +-2 cm core must hold at least 1.5x the
                         # points per cm of the strips 4-7 cm to either side. Thick blocks
                         # (wardrobes, pillars, door frames) are evenly dense and fail.


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


def find_spikes(position):
    """Centres of the strips where points pile up (step 1).

    A strip is a spike if it holds at least MIN_PEAK_POINTS and is the
    largest strip within +-PEAK_HALF_WIDTH, so one wall gives one spike.
    """
    origin = position.min()
    counts = np.bincount(np.floor((position - origin) / BIN).astype(int))
    reach = int(round(PEAK_HALF_WIDTH / BIN))
    padded = np.pad(counts, reach)
    neighbourhood_max = np.max([padded[k:k + len(counts)] for k in range(2 * reach + 1)], axis=0)
    spikes = np.where((counts >= MIN_PEAK_POINTS) & (counts == neighbourhood_max))[0]
    return origin + (spikes + 0.5) * BIN


def runs_along(along, heights, band_bottom, band_top):
    """Split one spike's points into wall pieces (steps 3 and 4).

    Walk along the wall in 5 cm steps. A step is wall only if its points
    stack up over at least half of the band height; otherwise it is a hole.
    Checking each step (not just the whole piece) stops a long sofa, whose
    seat, back and cushions sit at different heights, from passing as a wall.
    Holes up to MAX_GAP are bridged, longer ones end the piece. Each piece
    must be long enough and cover enough of the band height to be a wall.
    Returns a list of (start, end, mask of the piece's points, cover).
    """
    start = along.min()
    cells = np.floor((along - start) / ALONG_CELL).astype(int)
    n_height_slices = int(np.ceil((band_top - band_bottom) / HEIGHT_CELL))
    all_slices = np.clip(np.floor((heights - band_bottom) / HEIGHT_CELL).astype(int), 0, n_height_slices - 1)
    distinct = np.unique(cells * n_height_slices + all_slices)   # (cell, slice) pairs present
    slices_per_cell = np.bincount(distinct // n_height_slices, minlength=cells.max() + 1)
    occupied = slices_per_cell >= MIN_CELL_COVER * n_height_slices
    max_gap_cells = int(round(MAX_GAP / ALONG_CELL))

    pieces, first, last, gap = [], None, None, 0
    for k, filled in enumerate(np.append(occupied, False)):
        if filled:
            first = k if first is None else first
            last, gap = k, 0
        elif first is not None:
            gap += 1
            if gap > max_gap_cells or k == len(occupied):
                pieces.append((first, last))
                first, gap = None, 0

    walls = []
    for first, last in pieces:
        in_piece = (cells >= first) & (cells <= last)
        slices = np.floor((heights[in_piece] - band_bottom) / HEIGHT_CELL).astype(int)
        cover = len(np.unique(slices[(slices >= 0) & (slices < n_height_slices)])) / n_height_slices
        lo = start + first * ALONG_CELL
        hi = start + (last + 1) * ALONG_CELL
        if hi - lo >= MIN_LENGTH and cover >= MIN_HEIGHT_COVER:
            walls.append((lo, hi, in_piece, cover))
    return walls


def prominence(offset):
    """Points per cm in the wall's +-2 cm core divided by the denser of the two
    side strips 4-7 cm away. `offset` is each nearby point's distance from the
    wall's centre line, measured across the wall."""
    core = np.sum(np.abs(offset) <= 0.02) / 4.0
    left = np.sum((offset >= -0.07) & (offset <= -0.04)) / 3.0
    right = np.sum((offset >= 0.04) & (offset <= 0.07)) / 3.0
    return core / max(left, right, 1.0)


def find_walls(points, floor_y, ceiling_y):
    """List of straight wall pieces in a straightened cloud.

    'axis' says which coordinate is constant along the wall: a wall with
    axis 'x' runs up/down the plan at position x, from start to end in z.
    The position is the median of the wall's points (step 2), which is far
    more precise than the 2 cm strip it was found in.
    """
    band = wall_band(points, floor_y, ceiling_y)
    band_bottom = floor_y + 0.3
    band_top = points[band, 1].max()
    p = points[band]
    walls = []
    for axis, pos_col, along_col in (("x", 0, 2), ("z", 2, 0)):
        for centre in find_spikes(p[:, pos_col]):
            near = np.abs(p[:, pos_col] - centre) <= PEAK_HALF_WIDTH
            q = p[near]
            for lo, hi, in_piece, cover in runs_along(q[:, along_col], q[:, 1], band_bottom, band_top):
                position = float(np.median(q[in_piece, pos_col]))
                beside = (p[:, along_col] >= lo) & (p[:, along_col] <= hi)
                sharpness = prominence(p[beside, pos_col] - position)
                if sharpness < MIN_PROMINENCE:
                    continue
                walls.append({
                    "axis": axis,
                    "position": round(position, 4),
                    "start": round(float(lo), 3),
                    "end": round(float(hi), 3),
                    "length": round(float(hi - lo), 3),
                    "points": int(in_piece.sum()),
                    "height_cover": round(float(cover), 2),
                    "prominence": round(float(sharpness), 1),
                })
    return walls


def draw_walls(points, floor_y, ceiling_y, walls, path, cell=0.02):
    """Grey wall-density background with each detected wall as a coloured line."""
    xz = points[wall_band(points, floor_y, ceiling_y)][:, [0, 2]]
    origin = xz.min(axis=0) - 0.2
    size = np.ceil((xz.max(axis=0) + 0.2 - origin) / cell).astype(int)
    counts = np.zeros((size[1], size[0]))
    cols, rows = ((xz - origin) / cell).astype(int).T
    np.add.at(counts, (rows, cols), 1)
    shade = np.clip(counts / np.percentile(counts[counts > 0], 95), 0, 1)
    image = cv2.cvtColor((255 - 120 * shade).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def pixel(x, z):
        return int((x - origin[0]) / cell), int((z - origin[1]) / cell)

    palette = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
               (70, 240, 240), (240, 50, 230), (0, 128, 128), (170, 110, 40), (128, 0, 0)]
    for n, w in enumerate(walls):
        if w["axis"] == "x":
            a, b = pixel(w["position"], w["start"]), pixel(w["position"], w["end"])
        else:
            a, b = pixel(w["start"], w["position"]), pixel(w["end"], w["position"])
        colour = palette[n % len(palette)]
        cv2.line(image, a, b, colour, 3)
        label_at = ((a[0] + b[0]) // 2 + 4, (a[1] + b[1]) // 2 - 4)
        cv2.putText(image, f"{n}:{w['length']:.2f}", label_at, cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)
    length = int(round(1.0 / cell))
    h = image.shape[0]
    cv2.line(image, (20, h - 20), (20 + length, h - 20), (0, 0, 0), 3)
    cv2.putText(image, "1 m", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(str(path), image)


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

    walls = find_walls(straight, floor_y, ceiling_y)
    draw_walls(straight, floor_y, ceiling_y, walls, folder / "walls_found.png")

    layout_path = folder / "layout.json"
    layout = json.loads(layout_path.read_text()) if layout_path.exists() else {}
    layout["rotation_deg"] = round(angle, 2)
    layout["walls"] = walls
    layout_path.write_text(json.dumps(layout, indent=2))
    print(f"{folder.name}: turned by {angle:.2f} degrees, found {len(walls)} walls "
          f"-> {folder / 'walls_found.png'}")


if __name__ == "__main__":
    main()
