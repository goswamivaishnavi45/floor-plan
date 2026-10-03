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

# Room settings (part 4c)
GRID = 0.05              # the paint-bucket grid: one square = 5 cm x 5 cm of floor
FLOOR_TOLERANCE = 0.08   # floor points lie within +-8 cm of the floor height
                         # (the floor varies by ~6 cm across rooms, see Stage 8)
FLOOR_FILL = 0.30        # fill holes in the seen floor up to 30 cm (chair legs, bins)
WALL_EXTEND = 0.15       # stretch wall ends by 15 cm so corners close
ALIGN_TOLERANCE = 0.15   # two walls this close sideways count as one line
CLOSE_AS_WALL = 0.60     # gaps shorter than this are holes in a wall: close them
DOOR_MIN, DOOR_MAX = 0.60, 1.20   # gaps this wide are doorways: close and record them
MIN_ROOM_AREA = 1.5      # m^2; smaller painted areas are corners or clutter
DOOR_SPLIT = 0.45        # m: shrinking the floor by this closes passages under 0.9 m
                         # (doorways are 0.7-0.9 m) so each room becomes its own island

# Height settings (part 4d)
MIN_CEILING_ABOVE_FLOOR = 1.8   # no room has a lower ceiling; below this is furniture
SHELF = 0.02             # 2 cm height shelves when searching for the ceiling
SHELF_REACH = 0.02       # precise level = median of points within +-2 cm of the best shelf
MIN_CEILING_SEEN = 0.25  # report a ceiling only if it was seen over 25% of the room
MIN_SECOND_LEVEL_SEEN = 0.10    # a second ceiling level must cover 10% of the room

# Measurement and uncertainty settings (part 4e). The +- ranges are 95% ranges
# (2 sigma). The sensor term is an assumption from published iPhone LiDAR room
# tests (centimetre level per point, averaging over thousands of points leaves
# a bias of a few mm per surface); it has not been checked against tape
# measurements of these rooms yet.
SURFACE_SIGMA = 0.005    # m, sensor bias left in one fitted wall/floor/ceiling surface
MIN_DRIFT_SIGMA = 0.003  # m, drift floor even when room floors agree perfectly
SIDE_SEARCH = 0.30       # a room side's wall must be within 30 cm of the painted edge
MISSING_SIDE_SIGMA = 0.05   # m, when no wall was found for a side we use the painted
                            # edge, which is only good to a few 5 cm squares
LOW_COVER_SIGMA = 0.01   # m, extra for a ceiling seen over less than half the room
RECTANGLE_FILL = 0.8     # painted area must fill 80% of the wall rectangle to call
                         # the room rectangular; otherwise the area comes from the paint


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


def wall_point(wall, along):
    """(x, z) of the point at `along` on a wall's centre line."""
    return (wall["position"], along) if wall["axis"] == "x" else (along, wall["position"])


def find_gaps(walls):
    """Gaps between wall ends that must be closed before painting.

    From each wall end, look straight ahead along the wall for the nearest
    wall within DOOR_MAX: either the next piece on the same line, or a wall
    crossing the line. Shorter than CLOSE_AS_WALL is a hole in the wall
    (glass, missing points, a corner that does not quite meet); DOOR_MIN to
    DOOR_MAX is a doorway. Returns dicts with the two gap ends, width and kind.
    """
    gaps = {}
    for w in walls:
        for end, direction in ((w["end"], 1), (w["start"], -1)):
            best = None
            for v in walls:
                if v is w:
                    continue
                if v["axis"] == w["axis"]:
                    if abs(v["position"] - w["position"]) > ALIGN_TOLERANCE:
                        continue
                    facing = v["start"] if direction > 0 else v["end"]
                    target_along, target = facing, wall_point(v, facing)
                else:
                    if not v["start"] - ALIGN_TOLERANCE <= w["position"] <= v["end"] + ALIGN_TOLERANCE:
                        continue
                    target_along = v["position"]
                    target = wall_point(w, target_along)
                distance = (target_along - end) * direction
                if 0 <= distance <= DOOR_MAX and (best is None or distance < best[0]):
                    best = (distance, target)
            if best is None:
                continue
            width, target = best
            a = wall_point(w, end)
            key = tuple(sorted([tuple(np.round(a, 1)), tuple(np.round(target, 1))]))
            if key not in gaps:
                gaps[key] = {"a": [round(float(c), 3) for c in a],
                             "b": [round(float(c), 3) for c in target],
                             "width": round(float(width), 3),
                             "kind": "door" if width >= DOOR_MIN else "wall"}
    return list(gaps.values())


class Grid:
    """Maps plan coordinates (x, z) to cells of the 5 cm paint-bucket grid."""

    def __init__(self, points):
        xz = points[:, [0, 2]]
        self.origin = xz.min(axis=0) - 0.2
        self.shape = tuple(np.ceil((xz.max(axis=0) + 0.2 - self.origin) / GRID).astype(int)[::-1])

    def cell(self, x, z):
        return int((x - self.origin[0]) / GRID), int((z - self.origin[1]) / GRID)

    def cells(self, xz):
        return ((xz - self.origin) / GRID).astype(int).T   # (columns, rows)


def grow(seeds, free):
    """Grow labelled seeds over the free cells, one cell per step through
    up/down/left/right neighbours, until nothing changes: each free cell ends
    up with the label of the seed it is closest to through free space (walls
    block the growth)."""
    labels = seeds.copy()
    free = free.astype(bool)
    while True:
        changed = False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = np.zeros_like(labels)
            neighbour[max(dr, 0):labels.shape[0] + min(dr, 0), max(dc, 0):labels.shape[1] + min(dc, 0)] =                 labels[max(-dr, 0):labels.shape[0] + min(-dr, 0), max(-dc, 0):labels.shape[1] + min(-dc, 0)]
            take = (labels == 0) & free & (neighbour > 0)
            if take.any():
                labels[take] = neighbour[take]
                changed = True
        if not changed:
            return labels


def find_rooms(points, floor_y, walls):
    """Rooms: the free floor split at narrow passages.

    1. Mark every grid square where anything was seen, at any height; fill
       small holes. LiDAR cannot see through walls, so any point is inside
       the home. Using only floor points left rooms patchy: the floor is
       hidden under furniture and the phone rarely looked straight down.
    2. Draw walls (stretched at the ends) and small wall holes as barriers.
    3. Shrink the free floor by DOOR_SPLIT: every passage narrower than
       twice that (doorways are 70-90 cm) closes, so each room becomes a
       separate island, its seed.
    4. Grow the seeds back over the free floor (grow()), so neighbouring
       rooms meet at their doorways.
    Fix loop (fixloop/declaration.md): rooms used to be split only where
    both wall ends of a doorway were detected and joined; a missed wall end
    let rooms merge differently in two captures of the same home (17% of
    room dimensions repeatable). Splitting at narrow passages depends only
    on the shape of the floor, which repeats (walls within 0.5 cm). Doorway
    gaps are still recorded as openings.
    Returns (grid, label image, rooms, gaps).
    """
    grid = Grid(points)
    above_floor = points[:, 1] >= floor_y - FLOOR_TOLERANCE
    cols, rows = grid.cells(points[above_floor][:, [0, 2]])
    floor = np.zeros(grid.shape, np.uint8)
    floor[rows, cols] = 1
    size = int(round(FLOOR_FILL / GRID)) | 1
    floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))

    gaps = find_gaps(walls)
    barrier = np.zeros(grid.shape, np.uint8)
    for w in walls:
        a = grid.cell(*wall_point(w, w["start"] - WALL_EXTEND))
        b = grid.cell(*wall_point(w, w["end"] + WALL_EXTEND))
        cv2.line(barrier, a, b, 1, 2)
    for g in gaps:
        if g["kind"] == "wall":
            cv2.line(barrier, grid.cell(*g["a"]), grid.cell(*g["b"]), 1, 2)

    free = (floor & (1 - barrier)).astype(np.uint8)
    k = 2 * int(round(DOOR_SPLIT / GRID)) + 1
    islands = cv2.erode(free, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    count, seeds = cv2.connectedComponents(islands, connectivity=4)
    labels = grow(seeds.astype(np.int32), free)

    rooms, room_labels = [], np.zeros_like(labels)
    for label in range(1, count):
        patch = (labels == label).astype(np.uint8)
        # Fill holes inside the room (furniture the floor was hidden under).
        contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(patch)
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
        filled &= 1 - barrier
        area = filled.sum() * GRID * GRID
        if area < MIN_ROOM_AREA:
            continue
        room_id = len(rooms) + 1
        room_labels[filled.astype(bool) & (room_labels == 0)] = room_id
        r, c = np.nonzero(filled)
        rooms.append({"id": room_id, "area_m2_rough": round(float(area), 2),
                      "centre": [round(float(grid.origin[0] + (c.mean() + 0.5) * GRID), 2),
                                 round(float(grid.origin[1] + (r.mean() + 0.5) * GRID), 2)]})

    def rooms_beside(x, z, across):
        """Room ids found 12 cm either side of a point, across = unit (dx, dz)."""
        found = set()
        for side in (-1, 1):
            c, r = grid.cell(x + side * 0.12 * across[0], z + side * 0.12 * across[1])
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and room_labels[r, c]:
                found.add(int(room_labels[r, c]))
        return found

    for n, w in enumerate(walls):
        across = (1, 0) if w["axis"] == "x" else (0, 1)
        touching = set()
        for along in np.linspace(w["start"], w["end"], 7)[1:-1]:
            touching |= rooms_beside(*wall_point(w, along), across)
        for room_id in touching:
            rooms[room_id - 1].setdefault("walls", []).append(n)
    for n, g in enumerate(gaps):
        if g["kind"] != "door":
            continue
        dx, dz = g["b"][0] - g["a"][0], g["b"][1] - g["a"][1]
        across = (abs(dz) > abs(dx), abs(dx) >= abs(dz))   # perpendicular to the doorway
        g["rooms"] = sorted(rooms_beside((g["a"][0] + g["b"][0]) / 2, (g["a"][1] + g["b"][1]) / 2, across))
    return grid, room_labels, rooms, gaps


def measure_heights(points, floor_y, grid, room_labels, rooms):
    """Floor height, ceiling height and ceiling-to-floor distance per room.

    Floor: the floor is always the lowest big surface and Stage 8 already
    found it roughly, so take the room's points within +-8 cm of it and use
    their median. Ceiling: its height differs between rooms (lowered ceilings)
    and other things sit up high (wardrobe tops, lamps), so search: count the
    room's points at least 1.8 m above its floor in 2 cm shelves, take the
    busiest shelf, and use the median of the points within +-2 cm of it.
    The ceiling is only reported when it was seen over 25% of the room.
    """
    cols, rows = grid.cells(points[:, [0, 2]])
    inside = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    label = np.zeros(len(points), int)
    label[inside] = room_labels[rows[inside], cols[inside]]

    def seen_fraction(mask, room_cells):
        """Share of the room's grid squares that hold at least one masked point."""
        seen = np.unique(rows[mask] * grid.shape[1] + cols[mask])
        return len(seen) / room_cells

    for room in rooms:
        in_room = label == room["id"]
        room_cells = int((room_labels == room["id"]).sum())
        y = points[:, 1]

        near_floor = in_room & (np.abs(y - floor_y) <= FLOOR_TOLERANCE)
        room_floor = float(np.median(y[near_floor])) if near_floor.sum() >= 100 else float(floor_y)
        room["floor_y"] = round(room_floor, 4)
        room["floor_points"] = int(near_floor.sum())
        room["ceiling_y"] = room["ceiling_height"] = None
        room["ceiling_seen"] = 0.0

        high = in_room & (y >= room_floor + MIN_CEILING_ABOVE_FLOOR)
        if high.sum() < 100:
            room["ceiling_note"] = "ceiling not seen"
            continue
        edges = np.arange(y[high].min(), y[high].max() + SHELF, SHELF)
        counts, edges = np.histogram(y[high], bins=edges)
        order = np.argsort(counts)[::-1]

        levels = []
        for k in order[:10]:
            centre = (edges[k] + edges[k + 1]) / 2
            if any(abs(centre - level) < 0.10 for level, _ in levels):
                continue   # part of a level we already have
            on_shelf = high & (np.abs(y - centre) <= SHELF / 2 + SHELF_REACH)
            levels.append((float(np.median(y[on_shelf])), seen_fraction(on_shelf, room_cells)))

        main_level, main_seen = levels[0]
        room["ceiling_seen"] = round(main_seen, 2)
        if main_seen < MIN_CEILING_SEEN:
            room["ceiling_note"] = f"ceiling seen over only {main_seen:.0%} of the room; not reported"
            continue
        room["ceiling_y"] = round(main_level, 4)
        room["ceiling_height"] = round(main_level - room_floor, 4)
        others = [(lv, s) for lv, s in levels[1:] if s >= MIN_SECOND_LEVEL_SEEN]
        if others:
            lv, s = others[0]
            room["other_ceiling_level"] = {"height": round(lv - room_floor, 4), "seen": round(s, 2)}


def wall_sigma(wall):
    """One sigma of a wall's position: sensor bias plus the scatter of its
    points around the centre line divided by sqrt(points), which is tiny for
    thousands of points but large for a wall seen only briefly."""
    scatter = PEAK_HALF_WIDTH / np.sqrt(3)   # points spread over +-3 cm
    return float(np.hypot(SURFACE_SIGMA, scatter / np.sqrt(wall["points"])))


def measure_rooms(grid, room_labels, rooms, walls):
    """Width, length, area and ceiling height per room, each with a 95% range.

    Sizes come from the walls, which are located to millimetres, not from the
    painted squares. For each side of the room (left, right, front, back) we
    take the room's wall closest to the painted edge. Drift is estimated from
    how much the room floors disagree within this recording: floors in one
    home are level, so their spread is a lower bound on the tracking drift.
    """
    floors = [r["floor_y"] for r in rooms if r["floor_points"] >= 100]
    drift = max(MIN_DRIFT_SIGMA, (max(floors) - min(floors)) / 2) if len(floors) > 1 else MIN_DRIFT_SIGMA

    for room in rooms:
        rows, cols = np.nonzero(room_labels == room["id"])
        x = grid.origin[0] + (cols + 0.5) * GRID
        z = grid.origin[1] + (rows + 0.5) * GRID
        edges = {"left": ("x", x.min()), "right": ("x", x.max()),
                 "front": ("z", z.min()), "back": ("z", z.max())}

        sides = {}
        for side, (axis, edge) in edges.items():
            candidates = [n for n in room.get("walls", []) if walls[n]["axis"] == axis
                          and abs(walls[n]["position"] - edge) <= SIDE_SEARCH]
            if candidates:
                n = min(candidates, key=lambda n: abs(walls[n]["position"] - edge))
                sides[side] = {"wall": n, "position": walls[n]["position"], "sigma": wall_sigma(walls[n])}
            else:
                sides[side] = {"wall": None, "position": float(edge), "sigma": MISSING_SIDE_SIGMA}

        def span(a, b):
            value = sides[b]["position"] - sides[a]["position"]
            sigma = np.sqrt(sides[a]["sigma"] ** 2 + sides[b]["sigma"] ** 2 + drift ** 2)
            return value, sigma

        width, w_sigma = span("left", "right")      # along x
        length, l_sigma = span("front", "back")     # along z
        rect_area = width * length
        area_sigma = np.hypot(length * w_sigma, width * l_sigma)
        rectangular = room["area_m2_rough"] >= RECTANGLE_FILL * rect_area
        if rectangular:
            area = rect_area
        else:
            # L-shaped or merged room: the wall rectangle overstates the area,
            # so use the painted squares, which are only good to about one
            # square all the way round the outline.
            area = room["area_m2_rough"]
            perimeter = 2 * (width + length)
            area_sigma = np.hypot(area_sigma, perimeter * GRID / 2)

        room["shape"] = "rectangular" if rectangular else "not rectangular (area from painted floor)"
        room["sides"] = {s: v["wall"] for s, v in sides.items()}
        room["side_positions"] = {s: round(float(v["position"]), 4) for s, v in sides.items()}
        room["missing_sides"] = [s for s, v in sides.items() if v["wall"] is None]
        room["width"] = {"value": round(width, 3), "pm95": round(2 * w_sigma, 3)}
        room["length"] = {"value": round(length, 3), "pm95": round(2 * l_sigma, 3)}
        room["floor_area"] = {"value": round(area, 2), "pm95": round(2 * area_sigma, 2)}
        if room["ceiling_height"] is not None:
            sigma = np.sqrt(2 * SURFACE_SIGMA ** 2 + drift ** 2
                            + (LOW_COVER_SIGMA ** 2 if room["ceiling_seen"] < 0.5 else 0))
            room["ceiling"] = {"value": round(room["ceiling_height"], 3), "pm95": round(2 * sigma, 3)}
        else:
            room["ceiling"] = {"value": None, "pm95": None, "note": room.get("ceiling_note")}
    return drift


def draw_rooms(grid, room_labels, rooms, walls, gaps, path):
    """Each room in its own colour, walls black, wall-gaps grey, doors red."""
    scale = 2   # draw at 2.5 cm per pixel so labels are readable
    image = np.full(grid.shape + (3,), 255, np.uint8)
    palette = [(255, 205, 210), (200, 230, 201), (187, 222, 251), (255, 236, 179), (225, 190, 231),
               (178, 235, 242), (255, 204, 188), (220, 237, 200), (209, 196, 233), (240, 244, 195)]
    for room in rooms:
        image[room_labels == room["id"]] = palette[(room["id"] - 1) % len(palette)]
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def px(x, z):
        c, r = grid.cell(x, z)
        return c * scale, r * scale

    for w in walls:
        cv2.line(image, px(*wall_point(w, w["start"])), px(*wall_point(w, w["end"])), (40, 40, 40), 3)
    for g in gaps:
        colour, width = ((0, 0, 220), 3) if g["kind"] == "door" else ((150, 150, 150), 2)
        cv2.line(image, px(*g["a"]), px(*g["b"]), colour, width)
    for room in rooms:
        x, y = px(*room["centre"])
        lines = [f"R{room['id']}"]
        if "width" in room:
            lines.append(f"{room['width']['value']:.2f} x {room['length']['value']:.2f} m")
            lines.append(f"{room['floor_area']['value']:.1f} +-{room['floor_area']['pm95']:.1f} m2")
            ceiling = room["ceiling"]
            lines.append(f"h {ceiling['value']:.2f} +-{100 * ceiling['pm95']:.0f}cm" if ceiling["value"] else "h not seen")
        for k, text in enumerate(lines):
            cv2.putText(image, text, (x - 40, y + 16 * k), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 if k == 0 else 0.42, (0, 0, 0), 2 if k == 0 else 1)
    length = int(round(1.0 / GRID)) * scale
    h = image.shape[0]
    cv2.line(image, (20, h - 20), (20 + length, h - 20), (0, 0, 0), 3)
    cv2.putText(image, "1 m", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, "red = door  grey = closed hole in wall", (20, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(path), image)


def build_layout(points, colors, folder):
    """Run parts 4a-4e on a fused cloud. Writes the debug pictures and
    layout.json to `folder` and returns the layout plus the room grid."""
    floor_y, ceiling_y = horizontal_levels(points)

    angle, *_ = find_rotation(points, floor_y, ceiling_y)
    straight = rotate_about_vertical(points, angle)
    save_topdown_views(straight, colors, folder, floor_y, ceiling_y, suffix="_straight")

    walls = find_walls(straight, floor_y, ceiling_y)
    draw_walls(straight, floor_y, ceiling_y, walls, folder / "walls_found.png")

    grid, room_labels, rooms, gaps = find_rooms(straight, floor_y, walls)
    measure_heights(straight, floor_y, grid, room_labels, rooms)
    drift = measure_rooms(grid, room_labels, rooms, walls)
    draw_rooms(grid, room_labels, rooms, walls, gaps, folder / "rooms.png")

    layout = {"rotation_deg": round(angle, 2), "drift_sigma_m": round(drift, 4),
              "walls": walls, "rooms": rooms, "gaps": gaps}
    (folder / "layout.json").write_text(json.dumps(layout, indent=2))
    return layout, grid, room_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="output folder containing pointcloud.ply")
    args = parser.parse_args()

    folder = Path(args.folder)
    points, colors = read_ply(folder / "pointcloud.ply")
    layout, _, _ = build_layout(points, colors, folder)
    doors = [g for g in layout["gaps"] if g["kind"] == "door"]
    print(f"{folder.name}: turned by {layout['rotation_deg']:.2f} degrees, {len(layout['walls'])} walls, "
          f"{len(layout['rooms'])} rooms, {len(doors)} doors -> {folder / 'rooms.png'}")


if __name__ == "__main__":
    main()
