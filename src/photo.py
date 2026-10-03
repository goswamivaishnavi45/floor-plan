"""Photo tier: measure a room from a few photos (part 7b).

For one room folder (2-8 photos):
  1. VGGT on all photos together -> camera of each photo and depth maps that
     agree with each other, in VGGT's own units.
  2. Real size: Depth Anything's metric depth vs VGGT's depth on the same
     pixels -> one metres-per-unit factor for the room (divided by the 1.35
     bias measured against LiDAR, see scripts/compare_depth.py).
  3. Up: the room's three box directions (src/video.py box_directions); the
     one closest to the cameras' rough up (photos are taken upright) is up.
  4. 3D points -> the LiDAR-tier layout (walls, rooms, sizes).
Focal length: from the photo's EXIF (FocalLengthIn35mmFilm) when present,
otherwise VGGT's estimate.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

VGGT_DIR = Path(__file__).resolve().parent.parent / "models" / "vggt-1b"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

try:   # HEIC is the default photo format on recent iPhones and Samsung phones
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    IMAGE_TYPES -= {".heic", ".heif"}
EXIF_IFD, FOCAL_35MM = 0x8769, 0xA405
FULL_FRAME_DIAGONAL = 43.27     # mm, the 36 x 24 mm frame 35 mm-equivalent focal lengths refer to
VGGT_SIZE_CPU, VGGT_SIZE_GPU = 350, 518   # input size: 518 is what VGGT was trained at;
                                          # 350 is 2.8x faster on CPU for ~1 point more depth error
GRID = (192, 256)               # (w, h) depth grid turned into points, like the LiDAR tier
MIN_CONFIDENCE_SHARE = 0.7      # keep the 70% most confident VGGT depth pixels of each photo


def load_photos(folder):
    """Photos of a room folder, upright (EXIF orientation applied), with
    their 35 mm-equivalent focal length from EXIF or None."""
    photos = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in IMAGE_TYPES:
            continue
        image = Image.open(path)
        f35 = image.getexif().get_ifd(EXIF_IFD).get(FOCAL_35MM)
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        photos.append({"name": path.name, "bgr": cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                       "f35": float(f35) if f35 else None})
    return photos


def pad_to_square(bgr_frames, size):
    """VGGT input: each picture scaled so its long side is `size` px (a
    multiple of 14, VGGT's patch size) and padded with white to a square, as
    RGB in 0-1. Padding keeps every pixel (VGGT's default crop cuts the top
    and bottom off). Returns (tensor S x 3 x size x size, (top, left, h, w))."""
    h, w = bgr_frames[0].shape[:2]
    scale = size / max(h, w)
    ch, cw = round(h * scale / 14) * 14, round(w * scale / 14) * 14
    top, left = (size - ch) // 2, (size - cw) // 2
    batch = np.ones((len(bgr_frames), size, size, 3), np.float32)
    for k, bgr in enumerate(bgr_frames):
        rgb = cv2.cvtColor(cv2.resize(bgr, (cw, ch), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        batch[k, top:top + ch, left:left + cw] = rgb / 255.0
    return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous(), (top, left, ch, cw)


_vggt = None


def run_vggt(bgr_frames):
    """Cameras, focal and depth for a set of pictures of one place.
    Returns dict with R (S,3,3 world-from-camera), centre (S,3), focal_px
    and depth/confidence (S, h, w) at the content size (padding removed)."""
    global _vggt
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if _vggt is None:
        if not VGGT_DIR.exists():
            raise FileNotFoundError(f"{VGGT_DIR} not found; run `python scripts/fetch_models.py` first")
        _vggt = VGGT.from_pretrained(str(VGGT_DIR)).eval()
        if torch.cuda.is_available():
            _vggt = _vggt.cuda()
    size = VGGT_SIZE_GPU if torch.cuda.is_available() else VGGT_SIZE_CPU
    images, (top, left, ch, cw) = pad_to_square(bgr_frames, size)
    with torch.no_grad():
        pred = _vggt(images.to(next(_vggt.parameters()).device))
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    extrinsic, intrinsic = extrinsic[0].cpu().numpy(), intrinsic[0].cpu().numpy()
    R_cw = extrinsic[:, :, :3]
    return {
        "R": np.transpose(R_cw, (0, 2, 1)),
        "centre": np.array([-r.T @ t for r, t in zip(R_cw, extrinsic[:, :, 3])]),
        "focal_px": intrinsic[:, 0, 0],                      # at content size
        "content": (ch, cw),
        "depth": pred["depth"][0, ..., 0].cpu().numpy()[:, top:top + ch, left:left + cw],
        "confidence": pred["depth_conf"][0].cpu().numpy()[:, top:top + ch, left:left + cw],
    }


def room_points(photos, recon, depth_model, use_exif_focal=True, log=print):
    """Coloured points in metres with y up, for one room's photos.
    Returns (points, colours, info)."""
    from src.pointcloud import valid_mask, voxel_downsample, VOXEL
    from src.video import AI_BIAS, box_directions, rotation_between, surface_normals

    gw, gh = GRID
    ch, cw = recon["content"]
    ratios, per_photo = [], []
    for k, photo in enumerate(photos):
        depth = cv2.resize(recon["depth"][k], (gw, gh), interpolation=cv2.INTER_AREA)
        conf = cv2.resize(recon["confidence"][k], (gw, gh), interpolation=cv2.INTER_AREA)
        keep = conf >= np.quantile(conf, 1 - MIN_CONFIDENCE_SHARE)
        metric = depth_model.predict(photo["bgr"], size=(gw, gh))
        ratios.append(float(np.median(metric[keep] / np.maximum(depth[keep], 1e-6))))
        per_photo.append((depth, keep))
    metres_per_unit = float(np.median(ratios)) / AI_BIAS

    points, colours, normals, focal_used = [], [], [], []
    for k, (photo, (depth, keep)) in enumerate(zip(photos, per_photo)):
        h, w = photo["bgr"].shape[:2]
        if use_exif_focal and photo["f35"]:
            f_full = photo["f35"] * np.hypot(w, h) / FULL_FRAME_DIAGONAL
            focal = f_full * gw / w
        else:
            focal = recon["focal_px"][k] * gw / cw
        focal_used.append(focal * w / gw)
        z = depth * metres_per_unit
        mask = keep & valid_mask(z, np.full(z.shape, 2, np.uint8))
        u, v = np.meshgrid(np.arange(gw) + 0.5, np.arange(gh) + 0.5)
        cam = np.stack([(u - gw / 2) / focal * z, (v - gh / 2) / focal * z, z], axis=-1)
        world = cam[mask] @ recon["R"][k].T + metres_per_unit * recon["centre"][k]
        points.append(world)
        small = cv2.resize(photo["bgr"], (gw, gh), interpolation=cv2.INTER_AREA)
        colours.append(small[mask][:, ::-1].astype(np.float64))
        normals.append(surface_normals(cam[::2, ::2]) @ recon["R"][k].T)
    points, colours = np.concatenate(points), np.concatenate(colours)

    # Up: of the room's three box directions, the one closest to the cameras'
    # rough up (minus image-y; photos are taken holding the phone upright).
    axes = box_directions(np.concatenate(normals))
    rough = np.mean([r @ np.array([0.0, -1.0, 0.0]) for r in recon["R"]], axis=0)
    up = axes[np.argmax(np.abs(axes @ rough))]
    up = up if up @ rough > 0 else -up
    level = rotation_between(up, np.array([0.0, 1.0, 0.0]))
    points = points @ level.T
    points, colours, _ = voxel_downsample(points, colours, np.ones(len(points)), VOXEL)
    info = {"photos": len(photos), "metres_per_unit": round(metres_per_unit, 4),
            "ratio_spread": round(float(np.std(ratios) / np.mean(ratios)), 3),
            "focal_px_used": [round(f, 1) for f in focal_used]}
    return points, colours, info


# Part 7d: the photo tier for a whole property, with a quality gate.
PHOTO_SCALE_SIGMA = 0.15     # 1-sigma size uncertainty when a room passes the gate; not
                             # calibrated (no room passed on our test photos), so wide
MAX_RATIO_SPREAD = 0.10      # photos must agree on size within 10% (Depth Anything / VGGT)
MIN_SIDES = 3                # walls must be found on at least 3 of the room's 4 sides


def shared_photos(room_folders):
    """Rooms connected by a doorway photo: the same picture saved in two
    room folders (the capture protocol asks for this). Compared by content
    hash, so file names do not matter. Returns [(room a, room b, file name)]."""
    import hashlib

    seen = {}
    for name, folder in room_folders.items():
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_TYPES:
                digest = hashlib.sha1(path.read_bytes()).hexdigest()
                seen.setdefault(digest, []).append((name, path.name))
    links = []
    for copies in seen.values():
        rooms = sorted({room for room, _ in copies})
        for a in range(len(rooms)):
            for b in range(a + 1, len(rooms)):
                links.append((rooms[a], rooms[b], copies[0][1]))
    return links


def measure_photo_folders(folder, out_dir, capture_info, log=print):
    """Photo tier: one result for a folder of room folders.

    Each room is reconstructed from its own photos (room_points) and run
    through the LiDAR layout. Its numbers are reported only if it passes the
    quality gate: walls on >= MIN_SIDES sides and photos agreeing on size
    within MAX_RATIO_SPREAD. Otherwise the room is listed as not measured,
    with the reason. Adjacency comes from doorway photos shared between
    room folders, which needs no reconstruction at all.
    """
    from src.layout import build_layout
    from src.mono_depth import DepthModel
    from src.pointcloud import horizontal_levels, save_ply
    from src.result import build_result

    folder = Path(folder)
    room_folders = {d.name: d for d in sorted(folder.iterdir())
                    if d.is_dir() and any(f.suffix.lower() in IMAGE_TYPES for f in d.iterdir())}
    depth_model = DepthModel()
    result = {
        "schema_version": "1.0", "capture": capture_info, "units": "metres",
        "rooms": [], "openings": [], "adjacency": [],
        "damage": [], "concealed_damage_flags": [], "scope": [],
        "warnings": [
            "Photo tier: rooms are reconstructed from their photos with VGGT and given a size with an AI "
            "depth model; a room's numbers are reported only if it passes a quality check, otherwise it is "
            "listed as not measured. Rooms are not placed relative to each other; plan.png is a schematic.",
            "Damage detection is not implemented in this version; damage, flags and scope are empty.",
        ],
    }
    names = {}
    for n, (name, room_folder) in enumerate(room_folders.items(), start=1):
        room_id = f"R{n}"
        names[name] = room_id
        photos = load_photos(room_folder)
        entry = {"id": room_id, "name": name, "photos": len(photos), "shape": "not measured",
                 "width": {"value": None, "pm95": None}, "length": {"value": None, "pm95": None},
                 "floor_area": {"value": None, "pm95": None},
                 "ceiling_height": {"value": None, "pm95": None}, "walls": []}
        reason = None
        if len(photos) < 2:
            reason = "fewer than 2 photos"
        else:
            log(f"      {name}: {len(photos)} photos, reconstructing")
            recon = run_vggt([p["bgr"] for p in photos])
            points, colours, info = room_points(photos, recon, depth_model)
            floor_y, _ = horizontal_levels(points)
            points[:, 1] -= floor_y
            room_dir = Path(out_dir) / f"room_{name}"
            room_dir.mkdir(parents=True, exist_ok=True)
            save_ply(room_dir / "pointcloud.ply", points, colours)
            layout, grid, labels = build_layout(points, colours, room_dir)
            if not layout["rooms"]:
                reason = "no room outline found in the reconstruction"
            else:
                best = max(layout["rooms"], key=lambda r: r["area_m2_rough"])
                sides = 4 - len(best["missing_sides"])
                if sides < MIN_SIDES:
                    reason = f"walls found on only {sides} of 4 sides"
                elif info["ratio_spread"] > MAX_RATIO_SPREAD:
                    reason = f"photos disagree on size by {100 * info['ratio_spread']:.0f}%"
                else:
                    layout["rooms"] = [best]
                    measured = build_result(layout, grid, labels, dict(capture_info),
                                            scale_sigma=PHOTO_SCALE_SIGMA)["rooms"][0]
                    measured.update({"id": room_id, "name": name, "photos": len(photos)})
                    for k, wall in enumerate(measured["walls"], start=1):
                        wall["id"] = f"{room_id}-W{k}"
                    entry = measured
            log(f"      {name}: " + (f"measured {entry['width']['value']} x {entry['length']['value']} m"
                                     if reason is None else f"not measured ({reason})"))
        if reason:
            note = f"not measured: {reason}"
            for key in ("width", "length", "floor_area", "ceiling_height"):
                entry[key]["note"] = note
            result["warnings"].append(f"{room_id} ({name}): {note}.")
        result["rooms"].append(entry)

    for a, b, photo in shared_photos(room_folders):
        result["adjacency"].append({"rooms": [names[a], names[b]], "via": f"shared photo {photo}"})

    measured = [r["floor_area"] for r in result["rooms"] if r["floor_area"]["value"] is not None]
    total = {"value": round(sum(m["value"] for m in measured), 3) if measured else None,
             "pm95": round(float(np.sqrt(sum(m["pm95"] ** 2 for m in measured))), 3) if measured else None}
    if len(measured) < len(result["rooms"]):
        total["note"] = f"sum over {len(measured)} of {len(result['rooms'])} rooms (the rest not measured)"
    result["property"] = {"room_count": len(result["rooms"]), "total_floor_area": total}
    return result
