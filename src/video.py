"""Video tier: frames from a handheld walkthrough video.

Part 6a: pick frames. A walkthrough has tens of frames per second, almost all
repeating their neighbours, and some blurred by hand motion. We cut the video
into equal time windows and keep the sharpest frame of each window.
Sharpness = variance of the Laplacian (how much the image changes from pixel
to pixel); a blurred frame has soft edges and a low score.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np


def sharpness(bgr):
    gray = cv2.cvtColor(cv2.resize(bgr, (480, 360)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_frames(video_path, count):
    """Return [(frame index, BGR image)] with one sharp frame per window.

    Reads the video once from start to end (seeking is unreliable, see
    src/capture.py) and keeps only the current best frame of each window
    in memory.
    """
    video = cv2.VideoCapture(str(video_path))
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    windows = np.linspace(0, total, count + 1).astype(int)
    picked, best, window = [], None, 0
    index = 0
    while video.grab():
        while window < count and index >= windows[window + 1]:
            if best is not None:
                picked.append(best[1:])
            best, window = None, window + 1
        if window >= count:
            break
        _, frame = video.retrieve()
        score = sharpness(frame)
        if best is None or score > best[0]:
            best = (score, index, frame)
        index += 1
    if best is not None and window < count:
        picked.append(best[1:])
    video.release()
    return picked


# Part 6b: camera positions with COLMAP (structure from motion).
COLMAP_SIZE = 960            # frames are shrunk to this long side for matching: SIFT
                             # spots are found just as well and it runs 4x faster
FOCAL_GUESS = 0.85           # starting focal length as a share of the long side
                             # (iPhone main camera); COLMAP refines it
NEIGHBOURS = 10              # match each frame with the next 10 (and 2^k further)


def reconstruct_cameras(frames, workdir):
    """Run COLMAP on [(frame index, upright BGR)] and return the camera of
    every frame it could place.

    1. Write each frame as a small JPEG.
    2. Find SIFT spots in every frame (corners, edges, patterns).
    3. Match spots between each frame and its next NEIGHBOURS frames (the
       video moves smoothly, so far-apart frames rarely share spots).
    4. Incremental mapping: start from two well-matched frames, then add
       frames one by one, solving where each camera was and where the
       spots are, with bundle adjustment polishing everything together.
    One shared camera (the same phone lens throughout); focal length is
    refined. Result is in COLMAP's own units: shape right, size unknown.

    Returns (cameras, reconstruction) where cameras maps frame index to
    {"R": world-from-camera rotation, "center": camera centre, "K": 3x3
    intrinsics at the frames' full resolution}.
    """
    import pycolmap

    workdir = Path(workdir)
    images = workdir / "images"
    if workdir.exists():
        shutil.rmtree(workdir)
    images.mkdir(parents=True)
    h, w = frames[0][1].shape[:2]
    shrink = COLMAP_SIZE / max(h, w)
    for index, bgr in frames:
        small = cv2.resize(bgr, (round(w * shrink), round(h * shrink)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(images / f"{index:06d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 95])

    sw, sh = round(w * shrink), round(h * shrink)
    reader = pycolmap.ImageReaderOptions(
        camera_model="SIMPLE_RADIAL",
        camera_params=f"{FOCAL_GUESS * max(sw, sh)},{sw / 2},{sh / 2},0")
    database = workdir / "database.db"
    pycolmap.extract_features(database, images, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader)
    pycolmap.match_sequential(database, pairing_options=pycolmap.SequentialPairingOptions(
        overlap=NEIGHBOURS, quadratic_overlap=True))
    maps = pycolmap.incremental_mapping(database, images, workdir / "sparse")
    if not maps:
        return {}, None
    reconstruction = max(maps.values(), key=lambda r: r.num_reg_images())

    cameras = {}
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        cam = reconstruction.cameras[image.camera_id]
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]) / shrink
        K[2, 2] = 1.0
        cameras[int(Path(image.name).stem)] = {
            "R": pose.rotation.matrix().T,          # world-from-camera
            "center": np.asarray(image.projection_center()),
            "K": K,
        }
    return cameras, reconstruction
