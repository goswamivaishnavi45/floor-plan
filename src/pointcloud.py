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


def voxel_downsample(points, colors, voxel):
    """Keep one point (the mean) per voxel."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    mean_points = np.zeros((len(counts), 3))
    mean_colors = np.zeros((len(counts), 3))
    np.add.at(mean_points, inverse, points)
    np.add.at(mean_colors, inverse, colors)
    return mean_points / counts[:, None], mean_colors / counts[:, None]


def fuse(capture, stride):
    frames = [i for i in range(0, capture.num_frames, stride) if capture.has_depth[i]]
    points, colors = [], []
    for n, (i, bgr) in enumerate(capture.rgb_frames(frames)):
        depth = capture.depth(i)
        mask = valid_mask(depth, capture.confidence(i))
        cam = backproject(depth, capture.K_depth, mask)
        pose = capture.poses[i]
        points.append(cam @ pose[:3, :3].T + pose[:3, 3])
        colors.append(bgr[mask][:, ::-1].astype(np.float64))   # BGR -> RGB
        # Downsample in chunks so memory stays bounded on long captures.
        if len(points) >= 50:
            p, c = voxel_downsample(np.concatenate(points), np.concatenate(colors), VOXEL)
            points, colors = [p], [c]
        if n % 100 == 0:
            print(f"  frame {i}/{capture.num_frames}")
    return voxel_downsample(np.concatenate(points), np.concatenate(colors), VOXEL)


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


def save_topdown_views(points, colors, out_dir, floor_y, ceiling_y, cell=0.02):
    # Height-coloured top view: for each cell keep the colour of the highest point.
    def highest_colour(rows, cols, _, shape):
        image = np.full(shape + (3,), 255, np.uint8)
        order = np.argsort(points[:, 1])   # world y is up: draw low first
        image[rows[order], cols[order]] = colors[order][:, ::-1].astype(np.uint8)
        return image
    image, _ = render_topdown(points, None, cell, highest_colour)
    cv2.imwrite(str(out_dir / "topdown.png"), draw_scale_bar(image, cell))

    # Wall view: only points well above the floor and below the ceiling, so
    # floor and ceiling are removed and vertical surfaces stand out as lines.
    top = (ceiling_y - 0.3) if ceiling_y is not None else floor_y + 2.0
    band = (points[:, 1] > floor_y + 0.3) & (points[:, 1] < top)

    def density(rows, cols, _, shape):
        counts = np.zeros(shape, np.float64)
        np.add.at(counts, (rows, cols), 1)
        counts = np.clip(counts / np.percentile(counts[counts > 0], 95), 0, 1)
        return (255 * (1 - counts)).astype(np.uint8)
    walls, _ = render_topdown(points[band], None, cell, density)
    walls = cv2.cvtColor(walls, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_dir / "walls.png"), draw_scale_bar(walls, cell))


def save_ply(path, points, colors):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    record = np.zeros(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                           ("r", "u1"), ("g", "u1"), ("b", "u1")])
    record["x"], record["y"], record["z"] = points.T
    record["r"], record["g"], record["b"] = np.clip(colors, 0, 255).astype(np.uint8).T
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(record.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", help="path to a Stray Scanner capture folder")
    parser.add_argument("--stride", type=int, default=5, help="use every Nth frame (default 5)")
    parser.add_argument("--out", default="outputs", help="output root folder")
    args = parser.parse_args()

    capture = load_capture(args.capture)
    out_dir = Path(args.out) / capture.root.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{capture.root.name}: {capture.num_frames} frames, using every {args.stride}th")

    points, colors = fuse(capture, args.stride)
    floor_y, ceiling_y = horizontal_levels(points)
    save_ply(out_dir / "pointcloud.ply", points, colors)
    save_topdown_views(points, colors, out_dir, floor_y, ceiling_y)

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
