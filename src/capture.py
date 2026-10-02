"""Reading a Stray Scanner LiDAR capture from disk.

A capture folder contains:
  camera_matrix.csv   3x3 intrinsics for the 1920x1440 RGB image (fallback)
  odometry.csv        per frame: camera pose (position + quaternion) and intrinsics
  depth/NNNNNN.png    256x192 uint16 depth in millimetres
  confidence/NNNNNN.png  256x192 uint8, 0 = low, 1 = medium, 2 = high
  rgb.mp4             colour video, one video frame per odometry row

Pose convention: the quaternion and position give the camera-to-world
transform with the camera axes in the OpenCV convention (x right, y down,
z forward). We verified this empirically: with this convention frames from
different viewpoints overlap (41k occupied 5 cm voxels on c00a170fe1), while
the OpenGL/ARKit convention scatters them (112k voxels). World y is
gravity-aligned (ARKit), so horizontal surfaces are planes of constant y.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

RGB_WIDTH = 1920
DEPTH_WIDTH, DEPTH_HEIGHT = 256, 192


def quaternion_to_matrix(q):
    """Rotation matrix from a quaternion given as (x, y, z, w)."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class Capture:
    root: Path
    timestamps: np.ndarray   # (N,) seconds
    poses: np.ndarray        # (N, 4, 4) camera-to-world transforms
    K_depth: np.ndarray      # (N, 3, 3) per-frame intrinsics scaled to the depth image
    has_depth: np.ndarray    # (N,) bool, False where the depth PNG is missing

    @property
    def num_frames(self):
        return len(self.poses)

    def depth(self, i):
        """Depth of frame i in metres, (192, 256) float32. 0 means no reading."""
        d = cv2.imread(str(self.root / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        return d.astype(np.float32) / 1000.0

    def confidence(self, i):
        return cv2.imread(str(self.root / "confidence" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)

    def rgb_frames(self, indices, size=(DEPTH_WIDTH, DEPTH_HEIGHT)):
        """Yield (index, BGR image resized to the depth resolution) for the
        requested frame indices. Decodes the video sequentially because
        seeking in long H.264 files is unreliable."""
        wanted = set(int(i) for i in indices)
        last = max(wanted)
        video = cv2.VideoCapture(str(self.root / "rgb.mp4"))
        i = 0
        while i <= last and video.grab():
            if i in wanted:
                _, frame = video.retrieve()
                yield i, cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            i += 1
        video.release()


def load_capture(root):
    root = Path(root)
    odometry = np.genfromtxt(root / "odometry.csv", delimiter=",", skip_header=1, usecols=range(13))

    # Intrinsics change from frame to frame because autofocus moves the lens:
    # fx varies by up to 1.2% within one capture (1581 to 1615 px on
    # c7d28f72c6), worth ~2 cm at the image edge 3 m away. So we use the
    # per-frame fx, fy, cx, cy from odometry.csv rather than the single
    # camera_matrix.csv, falling back to it if a row has no intrinsics.
    K = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
    K_frames = np.tile(K, (len(odometry), 1, 1))
    per_frame = ~np.isnan(odometry[:, 9:13]).any(axis=1)
    fx, fy, cx, cy = odometry[per_frame, 9:13].T
    K_frames[per_frame, 0, 0], K_frames[per_frame, 1, 1] = fx, fy
    K_frames[per_frame, 0, 2], K_frames[per_frame, 1, 2] = cx, cy
    K_depth = K_frames.copy()
    K_depth[:, :2] *= DEPTH_WIDTH / RGB_WIDTH
    poses = np.tile(np.eye(4), (len(odometry), 1, 1))
    for n, row in enumerate(odometry):
        poses[n, :3, :3] = quaternion_to_matrix(row[5:9])
        poses[n, :3, 3] = row[2:5]

    # Keep only frames whose depth map exists on disk.
    have_depth = np.array([(root / "depth" / f"{i:06d}.png").exists() for i in range(len(poses))])
    if not have_depth.all():
        print(f"warning: {(~have_depth).sum()} frames have no depth map and will be skipped")
    return Capture(root=root, timestamps=odometry[:, 0], poses=poses, K_depth=K_depth,
                   has_depth=have_depth)
