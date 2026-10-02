"""Turn the layout measurements into result.json, in the published format
(schema/result.schema.json), and check the output against that schema.

The layout keeps internal bookkeeping (wall indices, painted squares, debug
numbers); the result keeps only what a reader of the floor plan needs, each
measurement as {"value", "pm95"}.
"""

import json
from pathlib import Path

import cv2
import jsonschema
import numpy as np

from src.layout import ALONG_CELL, GRID, MISSING_SIDE_SIGMA

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "result.schema.json"

# A doorway's width is the distance between two wall ends. Each end is found
# on the 5 cm steps used to follow walls, so it is uncertain by about one
# step / sqrt(12) (uniform rounding); the width combines two ends.
DOOR_WIDTH_SIGMA = np.sqrt(2) * ALONG_CELL / np.sqrt(12)


def measure(value, pm95, note=None):
    out = {"value": None if value is None else round(float(value), 3),
           "pm95": None if pm95 is None else round(float(pm95), 3)}
    if note:
        out["note"] = note
    return out


def point(x, z):
    return [round(float(x), 3), round(float(z), 3)]


def rectangle_outline(room):
    s = room["side_positions"]
    return [point(s["left"], s["front"]), point(s["right"], s["front"]),
            point(s["right"], s["back"]), point(s["left"], s["back"])]


SPIKE = 0.30        # painted spikes thinner than 30 cm are leaks, not room
CORNER = 0.15       # border bends smaller than 15 cm are not corners
SNAP = 0.15         # an outline edge within 15 cm of a detected wall moves onto it


def manhattan_outline(room, grid, room_labels, walls):
    """Clean right-angled outline of a non-rectangular room.

    1. Smooth the painted area: remove spikes thinner than 30 cm.
    2. Trace its border and keep only bends larger than 15 cm.
    3. Force each edge to run along x or along z (walls meet at right
       angles), merging neighbouring edges that run the same way.
    4. Snap each edge onto the room's detected wall if one is within 15 cm,
       which makes it exact; otherwise it stays at the painted edge.
    Returns (corner points, detected flag per edge) or None if too small.
    """
    patch = (room_labels == int(room["id"])).astype(np.uint8)
    k = int(round(SPIKE / GRID)) | 1
    patch = cv2.morphologyEx(patch, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), CORNER / GRID, True)[:, 0, :]
    xz = np.column_stack([grid.origin[0] + (contour[:, 0] + 0.5) * GRID,
                          grid.origin[1] + (contour[:, 1] + 0.5) * GRID])

    # Each edge: orientation ('z' = runs along x at constant z, 'x' = constant x) and level.
    edges = []
    for a, b in zip(xz, np.roll(xz, -1, axis=0)):
        horizontal = abs(b[0] - a[0]) >= abs(b[1] - a[1])
        axis, level, weight = ("z", (a[1] + b[1]) / 2, abs(b[0] - a[0])) if horizontal \
            else ("x", (a[0] + b[0]) / 2, abs(b[1] - a[1]))
        if edges and edges[-1][0] == axis:
            prev_axis, prev_level, prev_weight = edges[-1]
            total = prev_weight + weight
            edges[-1] = (axis, (prev_level * prev_weight + level * weight) / max(total, 1e-9), total)
        else:
            edges.append((axis, level, weight))
    if len(edges) > 1 and edges[0][0] == edges[-1][0]:
        axis, level, weight = edges.pop()
        first_axis, first_level, first_weight = edges[0]
        total = first_weight + weight
        edges[0] = (axis, (first_level * first_weight + level * weight) / max(total, 1e-9), total)
    if len(edges) < 4:
        return None

    room_walls = [walls[n] for n in room.get("walls", [])]
    snapped, detected = [], []
    for axis, level, _ in edges:
        near = [w for w in room_walls if w["axis"] == axis and abs(w["position"] - level) <= SNAP]
        if near:
            level = min(near, key=lambda w: abs(w["position"] - level))["position"]
        snapped.append((axis, level))
        detected.append(bool(near))

    corners = []
    for (axis_a, level_a), (axis_b, level_b) in zip(snapped, snapped[1:] + snapped[:1]):
        x, z = (level_b, level_a) if axis_a == "z" else (level_a, level_b)
        corners.append(point(x, z))
    # Corner k joins edge k and edge k+1, so the side from corner k to corner
    # k+1 lies on edge k+1: shift the flags by one to match.
    detected = detected[1:] + detected[:1]
    return corners, detected


def polygon_area(outline):
    x, z = np.array(outline).T
    return 0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))


def outline_walls(room_name, outline, room, detected_edges=None):
    """One wall entry per edge of the outline, with length and its range."""
    entries = []
    for k, start in enumerate(outline):
        end = outline[(k + 1) % len(outline)]
        length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        if detected_edges is None:
            # rectangle: front/back edges run along x (the room width), left/right along z
            source = room["width"] if k % 2 == 0 else room["length"]
            detected = room["sides"][["front", "right", "back", "left"][k]] is not None
            pm95 = source["pm95"]
        else:
            detected = detected_edges[k]
            # Two ends, each on a snapped wall (5 mm) or a painted edge (5 cm).
            end_sigma = 0.005 if detected else MISSING_SIDE_SIGMA
            pm95 = 2 * np.hypot(end_sigma, end_sigma)
        entries.append({"id": f"{room_name}-W{k + 1}", "start": start, "end": end,
                        "length": measure(length, pm95), "detected": bool(detected)})
    return entries


def widen(m, relative_sigma, power=1):
    """Add a size uncertainty to a measurement's 95% range. Lengths scale with
    the size factor (power 1), areas with its square (power 2, so twice the
    relative error)."""
    if m["value"] is None:
        return
    extra = 2 * power * relative_sigma * abs(m["value"])
    m["pm95"] = round(float(np.hypot(m["pm95"] or 0.0, extra)), 3)


def build_result(layout, grid, room_labels, capture_info, scale_sigma=0.0):
    """scale_sigma: relative 1-sigma uncertainty of the overall size, for tiers
    whose metres come from an AI depth model (video, photo); 0 for LiDAR."""
    walls = layout["walls"]
    result = {
        "schema_version": "1.0",
        "capture": capture_info,
        "units": "metres",
        "rooms": [], "openings": [], "adjacency": [],
        "damage": [], "concealed_damage_flags": [], "scope": [],
        "warnings": [
            "Ranges assume a 5 mm sensor bias per surface plus measured scatter and drift; "
            "they have not yet been checked against tape measurements.",
            "Damage detection is not implemented in this version; damage, flags and scope are empty.",
        ],
    }
    result["capture"]["drift_sigma_m"] = layout["drift_sigma_m"]
    result["capture"]["plan_rotation_deg"] = layout["rotation_deg"]

    areas = []
    for room in layout["rooms"]:
        name = f"R{room['id']}"
        rectangular = room["shape"] == "rectangular"
        traced = None if rectangular else manhattan_outline(room, grid, room_labels, walls)
        if traced is None:
            outline, detected_edges = rectangle_outline(room), None
            area = room["floor_area"]
        else:
            outline, detected_edges = traced
            # Area of the clean outline; its range comes from the edges that
            # were not snapped to a wall, each uncertain by MISSING_SIDE_SIGMA.
            loose = sum(np.hypot(b[0] - a[0], b[1] - a[1])
                        for a, b, d in zip(outline, outline[1:] + outline[:1], detected_edges) if not d)
            area = {"value": polygon_area(outline),
                    "pm95": 2 * float(np.hypot(loose * MISSING_SIDE_SIGMA, room["floor_area"]["pm95"] / 4))}
        ceiling = room["ceiling"]
        result["rooms"].append({
            "id": name,
            "name": f"Room {room['id']}",
            "outline": outline,
            "shape": "rectangular" if rectangular else "not rectangular",
            "width": measure(room["width"]["value"], room["width"]["pm95"]),
            "length": measure(room["length"]["value"], room["length"]["pm95"]),
            "floor_area": measure(area["value"], area["pm95"]),
            "ceiling_height": measure(ceiling["value"], ceiling["pm95"], ceiling.get("note")),
            "walls": outline_walls(name, outline, room, detected_edges),
        })
        areas.append((area["value"], area["pm95"]))
        if not rectangular:
            note = " It may be two spaces merged (no wall or door found between them)." \
                if room["floor_area"]["value"] > 15 else ""
            result["warnings"].append(f"{name} is not rectangular; its outline is traced from the seen floor "
                                      f"and snapped to walls where found.{note}")
        if room["missing_sides"]:
            result["warnings"].append(f"{name}: no wall found on side(s) {', '.join(room['missing_sides'])}; "
                                      "those sides use the edge of the seen floor.")
        if ceiling["value"] is None:
            result["warnings"].append(f"{name}: {ceiling.get('note') or 'ceiling not measured'}.")

    doors = [g for g in layout["gaps"] if g["kind"] == "door"]
    for n, gap in enumerate(doors, start=1):
        opening_id = f"O{n}"
        rooms = [f"R{r}" for r in gap.get("rooms", [])]
        result["openings"].append({
            "id": opening_id, "type": "door",
            "start": point(*gap["a"]), "end": point(*gap["b"]),
            "width": measure(gap["width"], 2 * DOOR_WIDTH_SIGMA),
            "rooms": rooms,
        })
        if len(rooms) == 2:
            result["adjacency"].append({"rooms": rooms, "via": opening_id})

    total = sum(a for a, _ in areas)
    total_pm95 = float(np.sqrt(sum(p ** 2 for _, p in areas)))
    result["property"] = {"room_count": len(result["rooms"]),
                          "total_floor_area": measure(total, total_pm95)}

    if scale_sigma:
        for room in result["rooms"]:
            for key in ("width", "length", "ceiling_height"):
                widen(room[key], scale_sigma)
            widen(room["floor_area"], scale_sigma, power=2)
            for wall in room["walls"]:
                widen(wall["length"], scale_sigma)
        for opening in result["openings"]:
            widen(opening["width"], scale_sigma)
        widen(result["property"]["total_floor_area"], scale_sigma, power=2)
        result["warnings"].insert(0, f"Sizes come from an AI depth model; every range includes a "
                                     f"{100 * scale_sigma:.0f}% (1 sigma) size uncertainty.")
    return result


def validate(result):
    """Raise jsonschema.ValidationError if the result breaks the published format."""
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(result, schema)
