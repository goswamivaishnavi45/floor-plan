"""Damage: detection, surface and size, concealed-damage flags, repair scope.

1. Detection: OWLv2 (Minderer et al. 2023, google/owlv2-base-patch16-ensemble,
   Apache-2.0), an open-vocabulary detector: given text such as "crack in
   wall" it returns boxes around matching regions with a score. Run on
   sharp frames spread over the capture (or on the photos).
2. Surface and size (LiDAR tier): the LiDAR depth inside each box places the
   damage in 3D, on the nearest floor, ceiling or wall of the plan; its
   width and height are box size x distance / focal length. Other tiers
   report the room only, without a size.
3. The same damage seen in several frames is merged (same class, within
   MERGE_DISTANCE).
4. Concealed-damage rules (RULES) flag what is likely behind what is seen;
   each flag names the rule that fired.
5. Scope: one repair line item per damaged surface, quantity from the
   surface or the damage size.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

OWL_DIR = Path(__file__).resolve().parent.parent / "models" / "owlv2-base"

# Each damage class is searched with a few phrasings; the best-scoring one counts.
PROMPTS = {
    "water stain": ["a water stain on a wall", "a water stain on a ceiling", "a damp patch"],
    "crack": ["a crack in a wall", "a crack in plaster", "a cracked ceiling"],
    "mould": ["black mould", "mold on a wall"],
    "peeling paint": ["peeling paint", "flaking paint"],
    "hole": ["a hole in a wall"],
}
SCORE_THRESHOLD = 0.30     # OWLv2 scores; lower ones were mostly shadows and textures
MAX_BOX_SHARE = 0.5        # boxes covering over half the picture are the whole wall, not damage
DAMAGE_FRAMES = 20         # frames searched per capture (~5 s each on CPU)
MERGE_DISTANCE = 0.5       # m: detections of one class closer than this are the same damage
SIZE_SIGMA = 0.15          # relative 1-sigma of a box-based size (box fit, viewing angle)

RULES = [
    ("C1", "water stain on a ceiling",
     lambda d: d["class"] == "water stain" and d["surface_type"] == "ceiling",
     "possible leak above the ceiling (plumbing, roof or a wet room upstairs)"),
    ("C2", "water stain or mould on a wall within 0.3 m of the floor",
     lambda d: d["class"] in ("water stain", "mould") and d["surface_type"] == "wall"
     and d.get("height_above_floor") is not None and d["height_above_floor"] < 0.3,
     "possible pipe leak or rising damp behind the wall"),
    ("C3", "mould anywhere",
     lambda d: d["class"] == "mould",
     "hidden moisture likely behind or around the surface"),
    ("C4", "crack longer than 1 m",
     lambda d: d["class"] == "crack" and d.get("length_m") is not None and d["length_m"] > 1.0,
     "possible structural movement; needs a structural inspection"),
]

SCOPE = {
    "water stain": ("Stain-block primer and repaint", "surface"),
    "mould": ("Mould treatment, stain-block and repaint", "damage+margin"),
    "peeling paint": ("Scrape, prime and repaint", "surface"),
    "crack": ("Rake out, fill and repaint crack", "length"),
    "hole": ("Patch hole and repaint", "count"),
}


class DamageDetector:
    def __init__(self, model_dir=OWL_DIR):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        if not Path(model_dir).exists():
            raise FileNotFoundError(f"{model_dir} not found; run `python scripts/fetch_models.py` first")
        self.processor = Owlv2Processor.from_pretrained(model_dir)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_dir).eval()
        self.labels = [(cls, phrase) for cls, phrases in PROMPTS.items() for phrase in phrases]

    @torch.no_grad()
    def detect(self, bgr, threshold=SCORE_THRESHOLD):
        """Damage boxes in a BGR picture: [{class, score, box (x0, y0, x1, y1)}],
        overlapping boxes reduced to the best one."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(text=[[p for _, p in self.labels]], images=rgb, return_tensors="pt")
        outputs = self.model(**inputs)
        # OWLv2 pads the picture to a square; boxes come back in that square.
        side = max(h, w)
        found = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=torch.tensor([[side, side]]))[0]
        boxes = []
        for score, label, box in zip(found["scores"], found["labels"], found["boxes"]):
            x0, y0, x1, y1 = [float(v) for v in box]
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) > MAX_BOX_SHARE * w * h:
                continue
            boxes.append({"class": self.labels[int(label)][0], "score": float(score),
                          "box": (x0, y0, x1, y1)})
        boxes.sort(key=lambda b: -b["score"])
        kept = []
        for b in boxes:
            if all(iou(b["box"], k["box"]) < 0.5 for k in kept):
                kept.append(b)
        return kept


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def place_on_plan(point, result):
    """Room, surface id and surface type for a plan point (x, y up, z)."""
    x, y, z = point
    for room in result["rooms"]:
        if "outline" not in room:
            continue
        outline = np.asarray(room["outline"], np.float32)
        if cv2.pointPolygonTest(outline, (float(x), float(z)), True) < -0.3:
            continue
        ceiling = room["ceiling_height"]["value"]
        if y < 0.15:
            return room["id"], f"{room['id']}-floor", "floor"
        if ceiling is not None and y > ceiling - 0.25:
            return room["id"], f"{room['id']}-ceiling", "ceiling"
        best, best_d = None, None
        for wall in room["walls"]:
            a, b = np.asarray(wall["start"]), np.asarray(wall["end"])
            t = np.clip(np.dot([x, z] - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
            d = np.linalg.norm([x, z] - (a + t * (b - a)))
            if best_d is None or d < best_d:
                best, best_d = wall, d
        if best is not None and best_d < 0.35:
            return room["id"], best["id"], "wall"
        return room["id"], f"{room['id']}-unknown", "unknown"
    return None, None, "unknown"


def merge(detections):
    """Same class, same surface and centres within MERGE_DISTANCE: one damage,
    the best-scoring sighting, with the median size of all sightings."""
    groups = []
    for d in sorted(detections, key=lambda d: -d["score"]):
        for g in groups:
            if (g[0]["class"] == d["class"] and g[0]["surface"] == d["surface"]
                    and (d.get("centre") is None or g[0].get("centre") is None
                         or np.linalg.norm(np.subtract(d["centre"], g[0]["centre"])) < MERGE_DISTANCE)):
                g.append(d)
                break
        else:
            groups.append([d])
    merged = []
    for g in groups:
        best = dict(g[0])
        best["sightings"] = len(g)
        for key in ("width_m", "height_m"):
            values = [d[key] for d in g if d.get(key) is not None]
            best[key] = float(np.median(values)) if values else None
        merged.append(best)
    return merged


def to_outputs(detections, result):
    """Damage list, concealed-damage flags and scope line items in the
    published format."""
    damage, flags, scope = [], [], []
    surfaces = {}
    for room in result["rooms"]:
        height = room["ceiling_height"]["value"]
        for wall in room["walls"]:
            surfaces[wall["id"]] = wall["length"]["value"] * height if height else None
        if room["floor_area"]["value"] is not None:
            surfaces[f"{room['id']}-ceiling"] = surfaces[f"{room['id']}-floor"] = room["floor_area"]["value"]

    def m(value, rel=SIZE_SIGMA):
        return {"value": None, "pm95": None} if value is None else \
            {"value": round(value, 3), "pm95": round(2 * rel * value, 3)}

    for n, d in enumerate(detections, start=1):
        w, h = d.get("width_m"), d.get("height_m")
        d["length_m"] = max(w, h) if w is not None and h is not None else None
        item = {"id": f"D{n}", "class": d["class"], "score": round(d["score"], 2),
                "room": d.get("room"), "surface": d.get("surface"), "surface_type": d["surface_type"],
                "seen_in": d["seen_in"], "sightings": d.get("sightings", 1),
                "width": m(w), "height": m(h), "area": m(w * h if w and h else None, 2 * SIZE_SIGMA)}
        if d.get("height_above_floor") is not None:
            item["height_above_floor"] = round(d["height_above_floor"], 2)
        damage.append(item)
        for rule_id, condition, fires, meaning in RULES:
            if fires(d):
                flags.append({"damage": item["id"], "rule": rule_id, "condition": condition,
                              "flag": meaning, "surface": item["surface"]})
        action, basis = SCOPE[d["class"]]
        surface_area = surfaces.get(d.get("surface"))
        if basis == "surface":
            quantity, unit = (surface_area, "m2") if surface_area else (item["area"]["value"], "m2")
        elif basis == "damage+margin":
            quantity, unit = ((w + 0.6) * (h + 0.6), "m2") if w and h else (None, "m2")
        elif basis == "length":
            quantity, unit = d["length_m"], "m"
        else:
            quantity, unit = 1, "each"
        scope.append({"id": f"S{n}", "surface": item["surface"], "damage": item["id"], "action": action,
                      "quantity": None if quantity is None else round(float(quantity), 2), "unit": unit})
    return damage, flags, scope


def lidar_damage(capture_root, layout, result, detector, log=print):
    """Detect damage on sharp frames of a LiDAR capture and place it on the plan.

    Frames are turned upright for the detector (it expects upright photos).
    The stored-image coordinates of every upright pixel are found by turning
    a coordinate grid the same way, so each box pixel gets its LiDAR depth
    and is placed in 3D with that frame's pose, then on the plan.
    """
    from scripts.compare_depth import upright_rotation
    from src.capture import DEPTH_HEIGHT, DEPTH_WIDTH, RGB_WIDTH, load_capture
    from src.layout import rotate_about_vertical
    from src.video import pick_frames

    capture = load_capture(capture_root)
    floors = [r["floor_y"] for r in layout["rooms"] if r.get("floor_points", 0) >= 100]
    floor_y = float(np.median(floors)) if floors else 0.0
    cols, rows = np.meshgrid(np.arange(DEPTH_WIDTH, dtype=np.float32), np.arange(DEPTH_HEIGHT, dtype=np.float32))
    detections = []
    for index, bgr in pick_frames(capture.root / "rgb.mp4", DAMAGE_FRAMES):
        if not capture.has_depth[index]:
            continue
        rotation = upright_rotation(capture.poses[index])
        turn = (lambda a: a) if rotation is None else (lambda a: cv2.rotate(a, rotation))
        upright = turn(bgr)
        found = detector.detect(upright)
        if not found:
            continue
        depth = turn(capture.depth(index))
        confident = turn((capture.confidence(index) >= 2).astype(np.uint8)).astype(bool)
        stored_u, stored_v = turn(cols), turn(rows)
        to_grid = depth.shape[1] / upright.shape[1]
        K, pose = capture.K_depth[index], capture.poses[index]
        focal_full = K[0, 0] * RGB_WIDTH / DEPTH_WIDTH
        for f in found:
            x0, y0, x1, y1 = [int(round(v * to_grid)) for v in f["box"]]
            region = (slice(y0, max(y1, y0 + 1)), slice(x0, max(x1, x0 + 1)))
            ok = confident[region] & (depth[region] > 0.2)
            if ok.sum() < 5:
                continue
            z = depth[region][ok]
            u, v = stored_u[region][ok] + 0.5, stored_v[region][ok] + 0.5
            cam = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
            world = cam @ pose[:3, :3].T + pose[:3, 3]
            centre = rotate_about_vertical(np.median(world, axis=0)[None], layout["rotation_deg"])[0]
            centre[1] -= floor_y
            room, surface, kind = place_on_plan(centre, result)
            distance = float(np.median(z))
            bw, bh = f["box"][2] - f["box"][0], f["box"][3] - f["box"][1]
            detections.append({**f, "seen_in": f"frame {index}", "centre": centre.round(3).tolist(),
                               "room": room, "surface": surface, "surface_type": kind,
                               "height_above_floor": float(centre[1]),
                               "width_m": bw * distance / focal_full, "height_m": bh * distance / focal_full})
    log(f"      {len(detections)} damage sightings")
    return merge(detections)


def picture_damage(pictures, detector, log=print):
    """Detect damage in photos or video frames without depth: room only, no
    size. pictures: [(name, BGR, room id or None)]."""
    detections = []
    for name, bgr, room in pictures:
        for f in detector.detect(bgr):
            detections.append({**f, "seen_in": name, "room": room,
                               "surface": f"{room}-unknown" if room else None, "surface_type": "unknown"})
    log(f"      {len(detections)} damage sightings")
    return merge(detections)
