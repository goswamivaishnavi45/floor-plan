"""Drift correction for LiDAR captures: plane-anchored chunk alignment.

ARKit tracks the phone by adding up small movements, so small errors add up
over a long walk ("drift"). We saw it as walls drawn twice a few cm apart
and as room floors at different heights (3 cm spread on 1a8384c3f6).

The capture is fused in chunks of CHUNK_SECONDS, short enough that drift
inside a chunk is negligible. Chunks are then added to the plan one by one:

1. Floor anchor: where the chunk's floor overlaps floor already in the plan
   (the same spot seen twice), the chunk is moved up or down by the median
   height difference there. Comparing only overlapping spots leaves stairs
   and other floor levels alone: anchoring every chunk to one floor height
   wrongly lifted half of c7d28f72c6 (a home with stairs) by 2.1 cm. A chunk
   with no overlapping floor keeps the previous chunk's shift.
2. Wall anchor (off by default, see WALL_ANCHOR): seen from above, the chunk's wall points are slid (up to
   MAX_SHIFT) and turned (up to MAX_TURN) to where they overlap the walls
   already in the plan the most. Drift changes slowly, so the search starts
   from the previous chunk's correction. A chunk that does not overlap known
   walls enough (a new area) keeps the previous correction.

Corrections are rigid (shift and turn about the vertical axis); ARKit's
gravity direction is reliable, so tilt is not corrected.
"""

import numpy as np

from src.pointcloud import VOXEL, horizontal_levels, voxel_downsample

CHUNK_SECONDS = 20.0
WALL_ANCHOR = False       # off by default: on c7d28f72c6 its small turns accumulated over
                          # chunks (to -4 deg on 1a8384c3f6) and broke 2 of 5 rooms, while
                          # wall sharpness did not improve (scripts/drift_ablation.py)
MAX_SHIFT = 0.10          # m, largest horizontal correction searched per chunk
SHIFT_STEP = 0.02         # m, one 2 cm cell of the wall map
MAX_TURN = 1.0            # degrees, largest turn searched per chunk
TURN_STEP = 0.25          # degrees
FLOOR_CELL = 0.10         # m, floor height map resolution
FLOOR_TOLERANCE = 0.05    # m, floor points lie within this of a floor cell's height
MIN_FLOOR_CELLS = 50      # overlapping floor cells (0.5 m^2) needed to trust a height correction
MAX_FLOOR_GAP = 0.15      # m: a chunk's floor layer must be this close to the capture's floor
MAX_LIFT_STEP = 0.05      # m: drift is centimetres; a bigger floor offset means the chunk's
                          # "floor" is something else (on c7d28f72c6, chunks looking at the
                          # ceiling gave -2.1 m), so it is ignored
WALL_CELL = 0.02          # m, wall map resolution
MIN_WALL_POINTS = 8       # points stacked in a cell for it to count as wall
MIN_OVERLAP = 150         # wall cells (~3 m of wall) shared with the plan to trust a correction
MIN_GAIN = 1.05           # the best correction must overlap 5% more than keeping the previous one


class FloorMap:
    """Floor height per 10 cm cell of the plan, from the floor points of
    chunks already placed. Floor points are those within 5 cm of the lowest
    strong horizontal layer of their chunk."""

    def __init__(self):
        self.sums, self.counts = {}, {}

    @staticmethod
    def floor_points(points):
        floor_y, _ = horizontal_levels(points)
        return points[np.abs(points[:, 1] - floor_y) <= FLOOR_TOLERANCE]

    @staticmethod
    def keys(points):
        cells = np.floor(points[:, [0, 2]] / FLOOR_CELL).astype(np.int64)
        return cells[:, 0] * 1_000_003 + cells[:, 1]

    @classmethod
    def cell_means(cls, floor):
        """(cell keys, mean height, point count) of floor points per cell."""
        keys, inverse, counts = np.unique(cls.keys(floor), return_inverse=True, return_counts=True)
        return keys, np.bincount(inverse.ravel(), weights=floor[:, 1]) / counts, counts

    def height_offset(self, floor):
        """Median of (plan floor height - chunk floor height) over cells both
        saw, or None if they share fewer than MIN_FLOOR_CELLS cells."""
        keys, means, _ = self.cell_means(floor)
        diffs = [self.sums[k] / self.counts[k] - m for k, m in zip(keys, means) if k in self.counts]
        return float(np.median(diffs)) if len(diffs) >= MIN_FLOOR_CELLS else None

    def add(self, floor):
        keys, means, counts = self.cell_means(floor)
        for k, m, c in zip(keys, means, counts):
            self.sums[k] = self.sums.get(k, 0.0) + m * c
            self.counts[k] = self.counts.get(k, 0) + c


def turn_and_shift(xz, pivot, turn_deg, shift):
    a = np.radians(turn_deg)
    c, s = np.cos(a), np.sin(a)
    d = xz - pivot
    return np.column_stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]]) + pivot + shift


class WallMap:
    """Top view of wall points on a fixed 2 cm grid covering all chunks."""

    def __init__(self, all_xz):
        self.origin = all_xz.min(axis=0) - 0.5
        size = np.ceil((all_xz.max(axis=0) + 0.5 - self.origin) / WALL_CELL).astype(int)
        self.counts = np.zeros((size[1], size[0]))

    def cells(self, xz):
        cols, rows = np.floor((xz - self.origin) / WALL_CELL).astype(int).T
        ok = (rows >= 0) & (rows < self.counts.shape[0]) & (cols >= 0) & (cols < self.counts.shape[1])
        return rows[ok], cols[ok]

    def mask(self, xz):
        counts = np.zeros_like(self.counts)
        np.add.at(counts, self.cells(xz), 1)
        return counts >= MIN_WALL_POINTS

    def walls(self):
        return self.counts >= MIN_WALL_POINTS

    def add(self, xz):
        np.add.at(self.counts, self.cells(xz), 1)


def best_alignment(wall_map, wall_xz, pivot, start_turn, start_shift):
    """Turn and shift (searched around the starting ones) that put the
    chunk's walls on the plan's walls the most. Returns (turn, shift,
    overlap, overlap at the start)."""
    known = wall_map.walls()
    steps = int(round(MAX_SHIFT / SHIFT_STEP))
    best = (start_turn, np.array(start_shift), -1)
    start_overlap = 0
    for turn in np.arange(start_turn - MAX_TURN, start_turn + MAX_TURN + 1e-9, TURN_STEP):
        chunk = wall_map.mask(turn_and_shift(wall_xz, pivot, turn, start_shift))
        rows, cols = np.nonzero(chunk)
        if len(rows) == 0:
            continue
        for dr in range(-steps, steps + 1):
            r = rows + dr
            for dc in range(-steps, steps + 1):
                c = cols + dc
                ok = (r >= 0) & (r < known.shape[0]) & (c >= 0) & (c < known.shape[1])
                overlap = int(known[r[ok], c[ok]].sum())
                if abs(turn - start_turn) < 1e-9 and dr == 0 and dc == 0:
                    start_overlap = overlap
                if overlap > best[2]:
                    best = (float(turn), np.array(start_shift) + np.array([dc, dr]) * WALL_CELL, overlap)
    return best[0], best[1], best[2], start_overlap


def correct_drift(chunks, log=print, wall_anchor=WALL_ANCHOR):
    """Align fused chunks to each other (see module doc). Returns (points,
    colours, report) with one report row per chunk."""
    all_points = np.concatenate([c["points"] for c in chunks])
    reference, _ = horizontal_levels(all_points)
    wall_map = WallMap(all_points[:, [0, 2]])
    floor_map = FloorMap()
    pivot = all_points[:, [0, 2]].mean(axis=0)

    lift, turn, shift = 0.0, 0.0, np.zeros(2)
    merged_points, merged_colours, merged_weights, report = [], [], [], []
    for n, chunk in enumerate(chunks):
        points = chunk["points"].copy()
        points[:, 1] += lift
        floor = FloorMap.floor_points(points)
        # Only a layer near the capture's overall floor can be floor (a chunk
        # looking at the ceiling finds the ceiling as its lowest layer).
        floor = floor[np.abs(floor[:, 1] - reference) <= MAX_FLOOR_GAP]
        offset = floor_map.height_offset(floor) if floor_map.counts else None
        if offset is not None and abs(offset) > MAX_LIFT_STEP:
            offset = None
            floor = floor[:0]   # implausible: do not trust or record this "floor"
        if offset is not None:
            lift += offset
            points[:, 1] += offset
            floor[:, 1] += offset

        band = (points[:, 1] > reference + 0.3) & (points[:, 1] < reference + 2.0)
        wall_xz = points[band][:, [0, 2]]
        overlap = start_overlap = 0
        if wall_anchor and wall_map.walls().any() and len(wall_xz):
            new_turn, new_shift, overlap, start_overlap = best_alignment(wall_map, wall_xz, pivot, turn, shift)
            if overlap >= MIN_OVERLAP and overlap >= MIN_GAIN * start_overlap:
                turn, shift = new_turn, new_shift
        points[:, [0, 2]] = turn_and_shift(points[:, [0, 2]], pivot, turn, shift)
        floor[:, [0, 2]] = turn_and_shift(floor[:, [0, 2]], pivot, turn, shift)
        wall_map.add(points[band][:, [0, 2]])
        floor_map.add(floor)

        merged_points.append(points)
        merged_colours.append(chunk["colors"])
        merged_weights.append(chunk["weights"])
        report.append({"chunk": n, "frames": f"{chunk['first']}-{chunk['last']}",
                       "floor_overlap": offset is not None, "lift_cm": round(100 * lift, 1),
                       "turn_deg": round(turn, 2), "shift_cm": [round(100 * v, 1) for v in shift],
                       "wall_overlap": overlap, "overlap_before": start_overlap})
        log(f"      chunk {n}: lift {100 * lift:+.1f} cm, turn {turn:+.2f} deg, "
            f"shift ({100 * shift[0]:+.0f}, {100 * shift[1]:+.0f}) cm, walls shared {overlap}")
    points, colours, _ = voxel_downsample(np.concatenate(merged_points), np.concatenate(merged_colours),
                                          np.concatenate(merged_weights), VOXEL)
    return points, colours, report
