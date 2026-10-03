"""Fuse the depth frames of a LiDAR capture into one coloured point cloud,
and render top-down views of it.

Usage:
    python -m src.pointcloud data/c00a170fe1

Writes to outputs/<capture name>/:
    pointcloud.ply   fused cloud, viewable in MeshLab / CloudCompare
    topdown.png      top view coloured by height
    walls.png        top view of wall points only (the rough floor plan)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.capture import load_capture

MIN_CONFIDENCE = 2         # keep only high-confidence LiDAR readings
MIN_DEPTH, MAX_DEPTH = 0.2, 5.0   # metres; iPhone LiDAR is unreliable beyond ~5 m
EDGE_THRESHOLD = 0.05      # drop pixels whose depth differs >5% from a neighbour
VOXEL = 0.02               # 2 cm: one point kept per voxel in the fused cloud


def backproject(depth, K, mask):
    """Pixels -> 3D points in the camera frame (OpenCV axes)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    z = depth[mask]
    x = (u[mask] - K[0, 2]) / K[0, 0] * z
    y = (v[mask] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1)


def valid_mask(depth, confidence):
    """Pixels we trust: high confidence, in range, and not on a depth edge.

    Depth edges produce "flying pixels" that float between a foreground object
    and the wall behind it; they would blur wall positions, so we drop them.
    Mirrors and glass usually come back with low confidence and are removed
    by the confidence test.
    """
    mask = (confidence >= MIN_CONFIDENCE) & (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
    padded = np.pad(depth, 1, mode="edge")
    max_neighbour_jump = np.zeros_like(depth)
    for dy, dx in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        neighbour = padded[dy:dy + depth.shape[0], dx:dx + depth.shape[1]]
        max_neighbour_jump = np.maximum(max_neighbour_jump, np.abs(neighbour - depth))
    mask &= max_neighbour_jump < EDGE_THRESHOLD * np.maximum(depth, 1e-6)
    return mask


def voxel_downsample(points, colors, weights, voxel):
    """Keep one point per voxel: the weighted mean of the points inside it.

    `weights` is how many raw measurements each input point stands for (1 for
    a raw point, more for a point that is already an average). Carrying the
    weights makes chunked merging give exactly the same result as merging
    everything at once.
    """
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n = inverse.max() + 1
    total_weight = np.bincount(inverse, weights=weights, minlength=n)
    mean_points = np.zeros((n, 3))
    mean_colors = np.zeros((n, 3))
    np.add.at(mean_points, inverse, points * weights[:, None])
    np.add.at(mean_colors, inverse, colors * weights[:, None])
    return (mean_points / total_weight[:, None], mean_colors / total_weight[:, None],
            total_weight)


def fuse_chunks(capture, stride, chunk_seconds=None):
    """Fuse frames into clouds, one per time chunk of `chunk_seconds` (one
    chunk for the whole capture if None). Each chunk is a dict with points,
    colors, weights (raw points per voxel) and its first/last frame. The
    video is read once, front to back."""
    frames = [i for i in range(0, capture.num_frames, stride) if capture.has_depth[i]]
    t0 = capture.timestamps[0]

    def chunk_of(i):
        return 0 if chunk_seconds is None else int((capture.timestamps[i] - t0) // chunk_seconds)

    chunks = {}
    for n, (i, bgr) in enumerate(capture.rgb_frames(frames)):
        depth = capture.depth(i)
        mask = valid_mask(depth, capture.confidence(i))
        cam = backproject(depth, capture.K_depth[i], mask)
        pose = capture.poses[i]
        c = chunks.setdefault(chunk_of(i), {"points": [], "colors": [], "weights": [], "first": i, "last": i})
        c["points"].append(cam @ pose[:3, :3].T + pose[:3, 3])
        c["colors"].append(bgr[mask][:, ::-1].astype(np.float64))   # BGR -> RGB
        c["weights"].append(np.ones(len(cam)))
        c["last"] = i
        # Downsample as we go so memory stays bounded on long captures.
        if len(c["points"]) >= 50:
            merged = voxel_downsample(np.concatenate(c["points"]), np.concatenate(c["colors"]),
                                      np.concatenate(c["weights"]), VOXEL)
            c["points"], c["colors"], c["weights"] = [merged[0]], [merged[1]], [merged[2]]
        if n % 100 == 0:
            print(f"  frame {i}/{capture.num_frames}")
    out = []
    for key in sorted(chunks):
        c = chunks[key]
        p, col, w = voxel_downsample(np.concatenate(c["points"]), np.concatenate(c["colors"]),
                                     np.concatenate(c["weights"]), VOXEL)
        out.append({"points": p, "colors": col, "weights": w, "first": c["first"], "last": c["last"]})
    return out


def fuse(capture, stride):
    """All frames of a capture in one cloud, poses used as recorded."""
    chunk = fuse_chunks(capture, stride)[0]
    return chunk["points"], chunk["colors"]


def horizontal_levels(points, bin_size=0.02):
    """Rough floor and ceiling heights from the histogram of world-y.

    Horizontal surfaces pile many points into a single height bin. The floor
    is the strongest peak in the lower part of the cloud, the ceiling the
    strongest in the upper part. Returns (floor_y, ceiling_y or None). This is
    only a first look; proper plane fitting comes in the layout step.
    """
    y = points[:, 1]
    edges = np.arange(y.min(), y.max() + bin_size, bin_size)
    hist, edges = np.histogram(y, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    middle = np.median(y)
    low, high = centres < middle, centres > middle
    floor_y = centres[low][np.argmax(hist[low])]
    ceiling_y = centres[high][np.argmax(hist[high])]
    # A real ceiling forms a peak comparable to the floor. If the strongest
    # upper bin is weak the ceiling was not scanned.
    if hist[high].max() < 0.2 * hist[low].max() or ceiling_y - floor_y < 1.8:
        ceiling_y = None
    return floor_y, ceiling_y


def render_topdown(points, values, cell, image_fn):
    """Rasterise points onto the x-z plane (looking down along -y)."""
    xz = points[:, [0, 2]]
    origin = xz.min(axis=0) - 0.2
    size = np.ceil((xz.max(axis=0) + 0.2 - origin) / cell).astype(int)
    cols, rows = ((xz - origin) / cell).astype(int).T
    return image_fn(rows, cols, values, (size[1], size[0])), origin


def draw_scale_bar(image, cell):
    length = int(round(1.0 / cell))
    h = image.shape[0]
    cv2.line(image, (20, h - 20), (20 + length, h - 20), (0, 0, 0), 3)
    cv2.putText(image, "1 m", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return image


def wall_band(points, floor_y, ceiling_y):
    """Mask of points well above the floor and below the ceiling.

    Removing floor and ceiling leaves mostly vertical surfaces, which seen from
    above collapse onto lines. Without a ceiling we stop at door height.
    """
    top = (ceiling_y - 0.3) if ceiling_y is not None else floor_y + 2.0
    return (points[:, 1] > floor_y + 0.3) & (points[:, 1] < top)


def save_topdown_views(points, colors, out_dir, floor_y, ceiling_y, cell=0.02, suffix=""):
    # Height-coloured top view: for each cell keep the colour of the highest point.
    def highest_colour(rows, cols, _, shape):
        image = np.full(shape + (3,), 255, np.uint8)
        order = np.argsort(points[:, 1])   # world y is up: draw low first
        image[rows[order], cols[order]] = colors[order][:, ::-1].astype(np.uint8)
        return image
    image, _ = render_topdown(points, None, cell, highest_colour)
    cv2.imwrite(str(out_dir / f"topdown{suffix}.png"), draw_scale_bar(image, cell))

    # Wall view: count wall-band points per cell; vertical surfaces stack
    # many points into one cell and show up as dark lines.
    band = wall_band(points, floor_y, ceiling_y)

    def density(rows, cols, _, shape):
        counts = np.zeros(shape, np.float64)
        np.add.at(counts, (rows, cols), 1)
        counts = np.clip(counts / np.percentile(counts[counts > 0], 95), 0, 1)
        return (255 * (1 - counts)).astype(np.uint8)
    walls, _ = render_topdown(points[band], None, cell, density)
    walls = cv2.cvtColor(walls, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_dir / f"walls{suffix}.png"), draw_scale_bar(walls, cell))


PLY_RECORD = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])


def read_ply(path):
    """Read a point cloud written by save_ply. Returns (points, colors)."""
    raw = Path(path).read_bytes()
    start = raw.index(b"end_header\n") + len(b"end_header\n")
    data = np.frombuffer(raw[start:], dtype=PLY_RECORD)
    points = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    colors = np.stack([data["r"], data["g"], data["b"]], axis=1)
    return points, colors


def save_ply(path, points, colors):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    record = np.zeros(len(points), dtype=PLY_RECORD)
    record["x"], record["y"], record["z"] = points.T
    record["r"], record["g"], record["b"] = np.clip(colors, 0, 255).astype(np.uint8).T
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(record.tobytes())


TARGET_FRAMES = 1200   # about one frame every 0.1-0.2 s of walking; more adds time, not detail


def auto_stride(num_frames):
    """Every 5th frame for short captures, sparser for long ones so the
    number of fused frames stays near TARGET_FRAMES."""
    return max(5, round(num_frames / TARGET_FRAMES))


def build_pointcloud(capture_root, out_dir, stride=None, drift_correction=True):
    """Fuse a capture and write pointcloud.ply, topdown.png and walls.png.
    With drift_correction, frames are fused in chunks and re-anchored to each
    other (src/drift.py); the corrections go to out_dir/drift.json.
    Returns (points, colors, capture, stride)."""
    import json

    from src.drift import CHUNK_SECONDS, correct_drift

    capture = load_capture(capture_root)
    stride = stride or auto_stride(capture.num_frames)
    if drift_correction:
        chunks = fuse_chunks(capture, stride, CHUNK_SECONDS)
        points, colors, report = correct_drift(chunks, log=lambda message: None)
        (Path(out_dir) / "drift.json").write_text(json.dumps(report, indent=2))
    else:
        points, colors = fuse(capture, stride)
    floor_y, ceiling_y = horizontal_levels(points)
    save_ply(out_dir / "pointcloud.ply", points, colors)
    save_topdown_views(points, colors, out_dir, floor_y, ceiling_y)
    return points, colors, capture, stride


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", help="path to a Stray Scanner capture folder")
    parser.add_argument("--stride", type=int, default=None, help="use every Nth frame (default: automatic)")
    parser.add_argument("--out", default="outputs", help="output root folder")
    args = parser.parse_args()

    out_dir = Path(args.out) / Path(args.capture).name
    out_dir.mkdir(parents=True, exist_ok=True)
    points, colors, capture, stride = build_pointcloud(args.capture, out_dir, args.stride)
    print(f"{capture.root.name}: {capture.num_frames} frames, used every {stride}th")
    floor_y, ceiling_y = horizontal_levels(points)

    extent = points.max(axis=0) - points.min(axis=0)
    print(f"points after 2 cm voxel filter: {len(points):,}")
    print(f"bounding box (x, y, z): {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")
    print(f"floor at y = {floor_y:.2f} m")
    if ceiling_y is None:
        print("ceiling: not enough ceiling points to locate it")
    else:
        print(f"ceiling at y = {ceiling_y:.2f} m  (rough height {ceiling_y - floor_y:.2f} m)")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
