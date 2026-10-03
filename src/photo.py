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
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}
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
