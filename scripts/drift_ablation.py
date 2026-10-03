"""Drift correction on vs off, on the same capture (the brief's ablation).

    python scripts/drift_ablation.py data/c7d28f72c6

Fuses the capture once in 20 s chunks, then builds the plan twice from the
same chunks: poses as recorded (off) and with plane-anchored correction (on,
src/drift.py). Writes outputs/<capture>_drift_{off,on}/ (full layout
pictures), drift_ablation.json and drift_ablation.png (plans side by side).

Measures:
    floor_spread_cm     max - min of room floor heights (floors are level)
    wall_spread_cm      for every wall over 1 m: how spread out its points are
                        across the wall (std of their distance from the wall
                        line, within 5 cm); median over walls. Drift records a
                        wall a little shifted on each visit, so it smears the
                        wall: lower means better aligned
    double_walls        pairs of parallel walls 2-8 cm apart overlapping by
                        >= 0.5 m: the same wall recorded twice (real walls are
                        thicker than 8 cm, so their two faces are not counted)
    walls, rooms, footprint_m2   the stitched plan
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.capture import load_capture  # noqa: E402
from src.drift import CHUNK_SECONDS, correct_drift  # noqa: E402
from src.layout import build_layout, rotate_about_vertical  # noqa: E402
from src.pointcloud import horizontal_levels, wall_band  # noqa: E402
from src.pointcloud import VOXEL, auto_stride, fuse_chunks, save_ply, voxel_downsample  # noqa: E402


def double_walls(walls):
    count = 0
    for i, a in enumerate(walls):
        for b in walls[i + 1:]:
            if a["axis"] != b["axis"] or not 0.02 <= abs(a["position"] - b["position"]) <= 0.08:
                continue
            if min(a["end"], b["end"]) - max(a["start"], b["start"]) >= 0.5:
                count += 1
    return count


def wall_spread(layout, points):
    floor_y, ceiling_y = horizontal_levels(points)
    straight = rotate_about_vertical(points[wall_band(points, floor_y, ceiling_y)], layout["rotation_deg"])
    spreads = []
    for w in layout["walls"]:
        if w["length"] < 1.0:
            continue
        pos, along = (straight[:, 0], straight[:, 2]) if w["axis"] == "x" else (straight[:, 2], straight[:, 0])
        offset = pos - w["position"]
        near = (np.abs(offset) <= 0.05) & (along >= w["start"]) & (along <= w["end"])
        if near.sum() > 100:
            spreads.append(float(np.std(offset[near])))
    return round(100 * float(np.median(spreads)), 2) if spreads else None


def measure(layout, points):
    floors = [r["floor_y"] for r in layout["rooms"] if r["floor_points"] >= 100]
    return {"rooms": len(layout["rooms"]), "walls": len(layout["walls"]),
            "wall_spread_cm": wall_spread(layout, points),
            "walls_over_1m": sum(1 for w in layout["walls"] if w["length"] >= 1.0),
            "double_walls": double_walls(layout["walls"]),
            "floor_spread_cm": round(100 * (max(floors) - min(floors)), 1) if len(floors) > 1 else None,
            "footprint_m2": round(sum(r["floor_area"]["value"] for r in layout["rooms"]), 2),
            "room_floor_heights": [round(f, 4) for f in floors]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture")
    args = parser.parse_args()

    capture = load_capture(args.capture)
    chunks = fuse_chunks(capture, auto_stride(capture.num_frames), CHUNK_SECONDS)
    report = {"capture": capture.root.name, "chunks": len(chunks)}
    pictures = []
    for mode in ("off", "on"):
        out = Path("outputs") / f"{capture.root.name}_drift_{mode}"
        out.mkdir(parents=True, exist_ok=True)
        if mode == "off":
            points, colours, _ = voxel_downsample(np.concatenate([c["points"] for c in chunks]),
                                                  np.concatenate([c["colors"] for c in chunks]),
                                                  np.concatenate([c["weights"] for c in chunks]), VOXEL)
        else:
            points, colours, report["corrections"] = correct_drift(chunks)
        save_ply(out / "pointcloud.ply", points, colours)
        layout, _, _ = build_layout(points, colours, out)
        report[mode] = measure(layout, points)
        image = cv2.imread(str(out / "walls_found.png"))
        cv2.putText(image, f"drift correction {mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
        pictures.append(image)

    h = max(p.shape[0] for p in pictures)
    pictures = [np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 255, np.uint8)]) for p in pictures]
    cv2.imwrite(str(Path("outputs") / f"{capture.root.name}_drift_ablation.png"), np.hstack(pictures))
    (Path("outputs") / f"{capture.root.name}_drift_ablation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "corrections"}, indent=2))


if __name__ == "__main__":
    main()
