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
MIN_PIECE = 8                # pieces with fewer placed frames are dropped


def reconstruct_pieces(frames, workdir):
    """Run COLMAP on [(frame index, upright BGR)] and return every piece of
    the video it could reconstruct, in time order.

    1. Write each frame as a small JPEG.
    2. Find SIFT spots in every frame (corners, edges, patterns).
    3. Match spots between each frame and its next NEIGHBOURS frames (the
       video moves smoothly, so far-apart frames rarely share spots).
    4. Incremental mapping: start from two well-matched frames, then add
       frames one by one, solving where each camera was and where the
       spots are, with bundle adjustment polishing everything together.
       When it loses track (plain walls, blur) it starts a new piece.
    Thresholds are looser than COLMAP's defaults (fewer matches and inliers
    needed), which placed ~20% more frames on our captures.
    One shared camera (the same phone lens throughout); focal length is
    refined. Each piece is in its own units and position: shape right,
    size and placement unknown.

    Each piece: {"cameras": {frame index: {"R": world-from-camera rotation,
    "center", "K": 3x3 at full frame resolution}}, "spots": {frame index:
    (pixel xy at full resolution, 3D spot positions)}}.
    """
    import pycolmap

    pycolmap.set_random_seed(0)   # same video in, same pieces out
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
    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = 10
    options.mapper.init_min_num_inliers = 50
    options.mapper.abs_pose_min_num_inliers = 15
    options.mapper.abs_pose_min_inlier_ratio = 0.15
    maps = pycolmap.incremental_mapping(database, images, workdir / "sparse", options=options)

    pieces = []
    for reconstruction in maps.values():
        cameras, spots = {}, {}
        for image in reconstruction.images.values():
            if not image.has_pose:
                continue
            pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
            cam = reconstruction.cameras[image.camera_id]
            f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
            K = np.array([[f / shrink, 0, cx / shrink], [0, f / shrink, cy / shrink], [0, 0, 1]])
            index = int(Path(image.name).stem)
            cameras[index] = {"R": pose.rotation.matrix().T,          # world-from-camera
                              "center": np.asarray(image.projection_center()), "K": K}
            xy, xyz = [], []
            for p in image.points2D:
                if p.has_point3D():
                    xy.append(np.asarray(p.xy) / shrink)
                    xyz.append(reconstruction.points3D[p.point3D_id].xyz)
            spots[index] = (np.array(xy), np.array(xyz))
        if len(cameras) >= MIN_PIECE:
            pieces.append({"cameras": cameras, "spots": spots})
    return sorted(pieces, key=lambda piece: min(piece["cameras"]))


def reconstruct_cameras(frames, workdir):
    """Cameras of the largest piece only (used by scripts/compare_path.py)."""
    pieces = reconstruct_pieces(frames, workdir)
    if not pieces:
        return {}, None
    largest = max(pieces, key=lambda piece: len(piece["cameras"]))
    return largest["cameras"], largest


# Part 6c: real size, up direction and 3D points for each piece.
AI_BIAS = 1.35               # the depth model reads this many times too far on our
                             # captures (scripts/compare_depth.py: 1.36 and 1.34 on two
                             # recordings). One home only: the video ranges must cover
                             # the chance that another home differs.
DEPTH_FRAMES = 20            # AI depth is slow on CPU (~1.7 s), so use up to 20 frames a piece
DEPTH_BUDGET = 240           # and at most this many in total, shared by piece size (~7 min)
MIN_SPOTS = 20               # a frame needs this many COLMAP spots to measure its ratio
FUSE_SIZE = (192, 256)       # (w, h) of the depth maps turned into points, like LiDAR's grid
BOX_TOLERANCE = 10.0         # degrees: a surface faces a box direction if within this
MIN_WALK = 0.5               # m: the camera must move this far sideways to tell up from its path
MAX_TILT = 60.0              # degrees: nobody films with the phone tilted further from upright;
                             # a piece whose up disagrees this much is a broken reconstruction


def spot_depths(camera, spots):
    """Distance of each COLMAP spot in front of the camera, in COLMAP units."""
    return (spots - camera["center"]) @ camera["R"][:, 2]


def scale_piece(piece, frame_images, model, count=DEPTH_FRAMES):
    """Give a piece its real size (part 6c, step 1).

    For up to DEPTH_FRAMES frames of the piece, run the AI depth model and
    compare, at each COLMAP spot, the AI's distance with COLMAP's distance:
    their ratio is how many AI-metres one COLMAP unit is in that frame. The
    median over the frame is the frame's ratio (its own size wobble); the
    median over frames, divided by AI_BIAS, is the piece's metres per unit.
    Stores per frame the AI depth and ratio, so each depth map can later be
    rescaled to agree with COLMAP exactly.
    """
    indices = sorted(piece["cameras"])
    chosen = [indices[k] for k in np.linspace(0, len(indices) - 1, min(count, len(indices))).astype(int)]
    piece["depth"], piece["ratio"] = {}, {}
    for index in dict.fromkeys(chosen):
        xy, xyz = piece["spots"][index]
        if len(xy) < MIN_SPOTS:
            continue
        bgr = frame_images[index]
        depth = model.predict(bgr, size=FUSE_SIZE)
        to_small = FUSE_SIZE[0] / bgr.shape[1]
        cols = np.clip((xy[:, 0] * to_small).astype(int), 0, FUSE_SIZE[0] - 1)
        rows = np.clip((xy[:, 1] * to_small).astype(int), 0, FUSE_SIZE[1] - 1)
        colmap = spot_depths(piece["cameras"][index], xyz)
        ok = colmap > 0
        if ok.sum() < MIN_SPOTS:
            continue
        piece["depth"][index] = depth
        piece["ratio"][index] = float(np.median(depth[rows[ok], cols[ok]] / colmap[ok]))
    if not piece["ratio"]:
        piece["metres_per_unit"] = None
        return
    piece["metres_per_unit"] = float(np.median(list(piece["ratio"].values()))) / AI_BIAS


def rotation_between(a, b):
    """Rotation matrix turning unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / np.linalg.norm(v) ** 2)


def camera_up(piece):
    """Average 'up' of the camera over the piece: a rough first guess. When
    the phone is tilted down, as it usually is, this leans forward by the
    tilt (25-60 degrees off on c00a170fe1), so it only picks which of the
    room's own directions is up (see box_directions)."""
    ups = np.array([cam["R"] @ np.array([0.0, -1.0, 0.0]) for cam in piece["cameras"].values()])
    up = ups.mean(axis=0)
    return up / np.linalg.norm(up)


def surface_normals(points_grid):
    """Facing direction of each small surface patch in an (h, w, 3) grid of
    points, from the cross product of its right and down neighbours."""
    right = points_grid[1:-1, 2:] - points_grid[1:-1, :-2]
    down = points_grid[2:, 1:-1] - points_grid[:-2, 1:-1]
    n = np.cross(right, down).reshape(-1, 3)
    length = np.linalg.norm(n, axis=1)
    ok = length > 1e-9
    return n[ok] / length[ok, None]


def box_directions(normals, rng=np.random.default_rng(0), trials=300):
    """The 3 right-angled directions most surfaces face (rooms are boxes).

    Try pairs of surface directions that are at right angles; each pair plus
    their cross product is a candidate box. Keep the box that the most
    surfaces agree with (within BOX_TOLERANCE of one of its directions,
    either sign), then refine each direction as the average of its surfaces.
    Returns a 3x3 matrix whose rows are the directions.
    """
    cos_tol = np.cos(np.radians(BOX_TOLERANCE))
    sample = normals[rng.choice(len(normals), min(len(normals), 20000), replace=False)]
    best, best_score = None, -1
    for _ in range(trials):
        a = sample[rng.integers(len(sample))]
        right_angle = np.abs(sample @ a) < np.sin(np.radians(5))
        if not right_angle.any():
            continue
        b = sample[rng.choice(np.nonzero(right_angle)[0])]
        b = b - (b @ a) * a
        b /= np.linalg.norm(b)
        axes = np.array([a, b, np.cross(a, b)])
        score = (np.abs(sample @ axes.T).max(axis=1) > cos_tol).sum()
        if score > best_score:
            best, best_score = axes, score
    refined = []
    for axis in best:
        dots = sample @ axis
        near = np.abs(dots) > cos_tol
        mean = (sample[near] * np.sign(dots[near])[:, None]).mean(axis=0)
        refined.append(mean / np.linalg.norm(mean))
    return np.array(refined)


def piece_points(piece, frame_images):
    """Coloured 3D points of a piece in metres, with y pointing up (part 6c,
    step 3).

    Each AI depth map is divided by its frame's ratio (so it agrees with
    COLMAP in that frame) and multiplied by the piece's metres per unit, then
    placed with the COLMAP camera, as Step 3 did with LiDAR depth. The piece
    is turned so the room's own up direction (box_directions) points along +y.
    Stores piece["to_level"] (rotation) so cameras can be moved into the
    same frame: position_m = to_level @ (metres_per_unit * centre).
    """
    from src.pointcloud import valid_mask, voxel_downsample, VOXEL

    s = piece["metres_per_unit"]
    points, colours, normals = [], [], []
    for index, ai in piece["depth"].items():
        cam = piece["cameras"][index]
        metric = ai / piece["ratio"][index] * s
        K = cam["K"] * (FUSE_SIZE[0] / frame_images[index].shape[1])
        K[2, 2] = 1.0
        mask = valid_mask(metric, np.full(metric.shape, 2, np.uint8))
        h, w = metric.shape
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        z = metric[mask]
        rays = np.stack([(u[mask] - K[0, 2]) / K[0, 0] * z, (v[mask] - K[1, 2]) / K[1, 1] * z, z], axis=1)
        points.append(rays @ cam["R"].T + s * cam["center"])
        grid = np.stack([(u - K[0, 2]) / K[0, 0] * metric, (v - K[1, 2]) / K[1, 1] * metric, metric], axis=-1)
        normals.append(surface_normals(grid[::2, ::2]) @ cam["R"].T)
        small = cv2.resize(frame_images[index], FUSE_SIZE, interpolation=cv2.INTER_AREA)
        colours.append(small[mask][:, ::-1].astype(np.float64))
    points, colours = np.concatenate(points), np.concatenate(colours)

    # Up is one of the room's 3 directions. People walk on a level floor, so
    # the camera path spreads least along up: use that when the path is long
    # enough to tell, otherwise the direction closest to the camera's rough
    # up. The camera's rough up only sets the sign (which way is up, not down).
    axes = box_directions(np.concatenate(normals))
    rough = camera_up(piece)
    centres = np.array([cam["center"] for cam in piece["cameras"].values()]) * s
    spread = (centres - centres.mean(axis=0)) @ axes.T
    extent = spread.max(axis=0) - spread.min(axis=0)
    if np.sort(extent)[1] > MIN_WALK:
        up = axes[np.argmin(extent)]
    else:
        up = axes[np.argmax(np.abs(axes @ rough))]
    up = up if up @ rough > 0 else -up
    if np.degrees(np.arccos(np.clip(up @ rough, -1, 1))) > MAX_TILT:
        piece["rejected"] = "camera tilt implausible; reconstruction likely broken"
        return None, None
    level = rotation_between(up, np.array([0.0, 1.0, 0.0]))
    points = points @ level.T
    piece["to_level"] = level
    points, colours, _ = voxel_downsample(points, colours, np.ones(len(points)), VOXEL)
    return points, colours


# Part 6d: measure rooms per video segment (pieces are not joined).
# Joining pieces end to end was tried and failed (git history, commit
# "chain pieces end to end"): walkers double back, so pieces overlap rather
# than follow each other, and per-piece size errors of 5-40% do not fit.
VIDEO_FRAMES = 300           # frames handed to COLMAP (~4-8 per second of a short video)
MIN_SIDES_FOUND = 3          # a room is reported only if walls were found on 3 of its 4 sides


def video_pieces(video_path, workdir, rotate=None, log=print):
    """Video file -> COLMAP pieces with real size (parts 6a-6c).
    `rotate` is a cv2 rotation for videos stored sideways (Stray Scanner's
    rgb.mp4); videos from the iPhone Camera app carry an orientation flag
    that OpenCV applies when reading, so they need none.
    Returns (pieces, frame images, info)."""
    from src.mono_depth import DepthModel

    frames = pick_frames(video_path, VIDEO_FRAMES)
    if rotate is not None:
        frames = [(i, cv2.rotate(f, rotate)) for i, f in frames]
    images = dict(frames)
    log(f"      {len(frames)} sharp frames picked; matching them with COLMAP")
    pieces = reconstruct_pieces(frames, Path(workdir) / "colmap")
    placed = sum(len(p["cameras"]) for p in pieces)
    log(f"      COLMAP placed {placed} frames in {len(pieces)} pieces; measuring depth")

    model = DepthModel()
    for piece in pieces:
        share = round(DEPTH_BUDGET * len(piece["cameras"]) / max(placed, 1))
        scale_piece(piece, images, model, count=max(3, min(DEPTH_FRAMES, share)))
    return pieces, images, {"frames": len(frames), "frames_placed": placed, "pieces": len(pieces)}


def measure_pieces(pieces, images, out_dir, capture_info, scale_sigma, log=print):
    """Rooms measured inside each piece separately, merged into one result.

    Each usable piece gets its own point cloud (floor at height 0) and runs
    the LiDAR-tier layout (walls, rooms, sizes) on its own, with debug
    pictures in out_dir/segment_N/. Only rooms with walls found on at least
    MIN_SIDES_FOUND sides are kept: a half-seen room's size would be a guess.
    Rooms keep the number of the segment they came from; they are not placed
    relative to rooms of other segments, and the same real room can appear
    in two segments.
    """
    from src.layout import build_layout
    from src.pointcloud import horizontal_levels, save_ply
    from src.result import build_result

    merged = None
    segments = []
    for n, piece in enumerate(pieces):
        frames = sorted(piece["cameras"])
        row = {"segment": n + 1, "frames": f"{frames[0]}-{frames[-1]}", "placed": len(frames)}
        segments.append(row)
        if piece.get("metres_per_unit") is None:
            row["skipped"] = "too few COLMAP spots to give it a size"
            continue
        points, colours = piece_points(piece, images)
        if points is None:
            row["skipped"] = piece["rejected"]
            continue
        floor_y, _ = horizontal_levels(points)
        points[:, 1] -= floor_y
        segment_dir = Path(out_dir) / f"segment_{n + 1}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        save_ply(segment_dir / "pointcloud.ply", points, colours)
        layout, grid, room_labels = build_layout(points, colours, segment_dir)
        complete = {room["id"] for room in layout["rooms"]
                    if 4 - len(room["missing_sides"]) >= MIN_SIDES_FOUND}
        layout["rooms"] = [room for room in layout["rooms"] if room["id"] in complete]
        part = build_result(layout, grid, room_labels, dict(capture_info), scale_sigma=scale_sigma)
        row["rooms"] = len(part["rooms"])
        log(f"      segment {n + 1}: frames {row['frames']}, {row['rooms']} complete rooms")

        if merged is None:
            merged = part
            merged["rooms"], merged["openings"], merged["adjacency"] = [], [], []
        # Renumber rooms and openings so ids are unique across segments.
        rename = {}
        for room in part["rooms"]:
            new_id = f"R{len(merged['rooms']) + 1}"
            rename[room["id"]] = new_id
            room["id"], room["name"], room["segment"] = new_id, f"Room {new_id[1:]} (segment {n + 1})", n + 1
            for k, wall in enumerate(room["walls"], start=1):
                wall["id"] = f"{new_id}-W{k}"
            merged["rooms"].append(room)
        for opening in part["openings"]:
            rooms = [rename[r] for r in opening["rooms"] if r in rename]
            if not rooms:
                continue
            opening["id"], opening["rooms"] = f"O{len(merged['openings']) + 1}", rooms
            merged["openings"].append(opening)
            if len(rooms) == 2:
                merged["adjacency"].append({"rooms": rooms, "via": opening["id"]})

    if merged is None:
        return None, segments
    areas = [(r["floor_area"]["value"], r["floor_area"]["pm95"]) for r in merged["rooms"]]
    merged["property"] = {"room_count": len(merged["rooms"]), "total_floor_area": {
        "value": round(sum(a for a, _ in areas), 3),
        "pm95": round(float(np.sqrt(sum(p ** 2 for _, p in areas))), 3)}}
    # Room-specific warnings refer to per-segment ids; keep only the general ones.
    general = [w for w in merged["warnings"] if not w.startswith("R")]
    used = sum(1 for s in segments if "skipped" not in s)
    merged["warnings"] = [
        f"Video: rooms are measured separately in {used} of {len(segments)} video segments and are NOT "
        "stitched into one plan; the same real room may be listed once per segment that saw it, so the "
        "total floor area can double count.",
        f"Only rooms with walls found on at least {MIN_SIDES_FOUND} of 4 sides are reported.",
    ] + general
    return merged, segments
