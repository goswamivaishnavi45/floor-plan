"""Draw plan.png, the floor plan a homeowner sees, from result.json.

White background, each room as a clean outline with its wall lengths written
along the walls, doors as red gaps with their width, and a label with the
room's name, size, area and ceiling height. Video-tier results, whose rooms
are measured per video segment and not placed relative to each other, get
one panel per segment instead of one plan.
"""

import cv2
import numpy as np

PIXELS_PER_METRE = 80
MARGIN = 90   # px around the plan for dimension labels
HEADER = 60   # px for the title lines
WALL = (30, 30, 30)
ROOM_FILL = (246, 243, 238)
DOOR = (40, 40, 220)
TEXT = (20, 20, 20)
DIM = (110, 80, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def centred_text(image, text, centre, scale, colour, thickness=1):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.putText(image, text, (int(centre[0] - w / 2), int(centre[1] + h / 2)), FONT, scale, colour, thickness,
                cv2.LINE_AA)


def draw_panel(rooms, openings, header_lines):
    """One plan drawing of the given rooms and openings, in their coordinates."""
    corners = [p for room in rooms for p in room["outline"]]
    corners += [p for o in openings for p in (o["start"], o["end"])]
    if not corners:
        image = np.full((300, 600, 3), 255, np.uint8)
        centred_text(image, "No rooms found", (300, 150), 0.8, TEXT, 2)
        return image
    corners = np.array(corners)
    low, high = corners.min(axis=0), corners.max(axis=0)
    size = ((high - low) * PIXELS_PER_METRE + 2 * MARGIN).astype(int)
    width = max(size[0], 12 * max(len(line) for line, _ in header_lines))
    image = np.full((size[1] + HEADER, width, 3), 255, np.uint8)

    def px(p):
        return (int(round((p[0] - low[0]) * PIXELS_PER_METRE + MARGIN)),
                int(round((p[1] - low[1]) * PIXELS_PER_METRE + MARGIN + HEADER)))

    # Room fills first, then walls on top, so shared walls stay crisp.
    for room in rooms:
        cv2.fillPoly(image, [np.array([px(p) for p in room["outline"]], np.int32)], ROOM_FILL)
    for room in rooms:
        outline = np.array([px(p) for p in room["outline"]], np.int32)
        cv2.polylines(image, [outline], True, WALL, 4, cv2.LINE_AA)

    # Doors: cut the wall and mark the opening.
    for opening in openings:
        a, b = px(opening["start"]), px(opening["end"])
        cv2.line(image, a, b, (255, 255, 255), 7)
        cv2.line(image, a, b, DOOR, 2, cv2.LINE_AA)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        centred_text(image, f"{opening['width']['value']:.2f}", (mid[0], mid[1] - 12), 0.4, DOOR)

    # Wall lengths, written just inside each wall, near its middle.
    for room in rooms:
        centre = np.mean([px(p) for p in room["outline"]], axis=0)
        for wall in room["walls"]:
            if wall["length"]["value"] < 0.5:
                continue
            a, b = np.array(px(wall["start"])), np.array(px(wall["end"]))
            mid = (a + b) / 2
            inward = centre - mid
            inward = inward / (np.linalg.norm(inward) + 1e-9)
            label = f"{wall['length']['value']:.2f}" + ("" if wall["detected"] else "?")
            centred_text(image, label, mid + 16 * inward, 0.42, DIM)

    # Room labels.
    for room in rooms:
        cx, cy = np.mean([px(p) for p in room["outline"]], axis=0)
        ceiling = room["ceiling_height"]
        lines = [(room["name"].split(" (")[0], 0.55, 2),
                 (f"{room['width']['value']:.2f} x {room['length']['value']:.2f} m", 0.42, 1),
                 (f"{room['floor_area']['value']:.1f} m2 +-{room['floor_area']['pm95']:.1f}", 0.42, 1),
                 (f"ceiling {ceiling['value']:.2f} m" if ceiling["value"] is not None else "ceiling not seen", 0.42, 1)]
        for k, (text, scale, thickness) in enumerate(lines):
            centred_text(image, text, (cx, cy - 24 + 17 * k), scale, TEXT, thickness)

    for k, (line, scale) in enumerate(header_lines):
        cv2.putText(image, line, (20, 30 + 22 * k), FONT, scale, TEXT if k == 0 else DIM,
                    2 if k == 0 else 1, cv2.LINE_AA)
    h = image.shape[0]
    cv2.line(image, (20, h - 20), (20 + PIXELS_PER_METRE, h - 20), TEXT, 3)
    cv2.putText(image, "1 m", (20, h - 28), FONT, 0.5, TEXT, 1, cv2.LINE_AA)
    return image


def draw_schematic(result, path):
    """Photo tier: rooms are not placed relative to each other, so draw a
    connection diagram instead of a floor plan: one box per room on a grid,
    a line for each doorway shared between two rooms."""
    rooms = result["rooms"]
    cols = int(np.ceil(np.sqrt(len(rooms)))) or 1
    box_w, box_h, gap = 260, 120, 90
    rows = int(np.ceil(len(rooms) / cols)) or 1
    image = np.full((HEADER + 40 + rows * (box_h + gap), 40 + cols * (box_w + gap), 3), 255, np.uint8)
    centres = {}
    for k, room in enumerate(rooms):
        x = 40 + (k % cols) * (box_w + gap)
        y = HEADER + 40 + (k // cols) * (box_h + gap)
        centres[room["id"]] = (x + box_w // 2, y + box_h // 2)
    for link in result["adjacency"]:
        a, b = link["rooms"]
        cv2.line(image, centres[a], centres[b], DOOR, 3, cv2.LINE_AA)
    for k, room in enumerate(rooms):
        cx, cy = centres[room["id"]]
        measured = room["width"]["value"] is not None
        cv2.rectangle(image, (cx - box_w // 2, cy - box_h // 2), (cx + box_w // 2, cy + box_h // 2),
                      ROOM_FILL if measured else (235, 235, 235), -1)
        cv2.rectangle(image, (cx - box_w // 2, cy - box_h // 2), (cx + box_w // 2, cy + box_h // 2), WALL, 3)
        lines = [(f"{room['id']}  {room['name']}", 0.55, 2)]
        if measured:
            lines.append((f"{room['width']['value']:.2f} x {room['length']['value']:.2f} m", 0.45, 1))
            lines.append((f"{room['floor_area']['value']:.1f} +-{room['floor_area']['pm95']:.1f} m2", 0.45, 1))
        else:
            lines.append(("size not measured", 0.45, 1))
        lines.append((f"{room.get('photos', 0)} photos", 0.4, 1))
        for n, (text, scale, thickness) in enumerate(lines):
            centred_text(image, text, (cx, cy - 30 + 22 * n), scale, TEXT, thickness)
    cv2.putText(image, f"{result['capture']['id']}  |  photo tier  |  {len(rooms)} rooms, "
                       f"{len(result['adjacency'])} connections", (20, 30), FONT, 0.6, TEXT, 2, cv2.LINE_AA)
    cv2.putText(image, "schematic, not to scale: boxes are rooms, red lines are doorways seen in shared photos",
                (20, 52), FONT, 0.42, DIM, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def draw_plan(result, path):
    if result["capture"]["tier"] == "photo":
        draw_schematic(result, path)
        return
    total = result["property"]["total_floor_area"]
    title = (f"{result['capture']['id']}  |  {result['capture']['tier']} tier  |  "
             f"{result['property']['room_count']} rooms, {total['value']:.1f} +-{total['pm95']:.1f} m2")
    legend = "lengths in metres; ? = side not seen as a wall; red = door width"
    segments = sorted({room["segment"] for room in result["rooms"] if "segment" in room})
    if not segments:
        image = draw_panel(result["rooms"], result["openings"], [(title, 0.6), (legend, 0.42)])
        cv2.imwrite(str(path), image)
        return

    panels = []
    for segment in segments:
        rooms = [r for r in result["rooms"] if r.get("segment") == segment]
        ids = {r["id"] for r in rooms}
        openings = [o for o in result["openings"] if set(o["rooms"]) & ids]
        panels.append(draw_panel(rooms, openings, [(f"video segment {segment}", 0.6),
                                                    ("not placed relative to other segments", 0.42)]))
    height = max(p.shape[0] for p in panels)
    padded = [np.vstack([p, np.full((height - p.shape[0], p.shape[1], 3), 255, np.uint8)]) for p in panels]
    gap = np.full((height, 12, 3), 200, np.uint8)
    row = padded[0]
    for panel in padded[1:]:
        row = np.hstack([row, gap, panel])
    top = np.full((70, row.shape[1], 3), 255, np.uint8)
    cv2.putText(top, title, (20, 30), FONT, 0.6, TEXT, 2, cv2.LINE_AA)
    cv2.putText(top, legend + "; each panel is one video segment, rooms may repeat across panels",
                (20, 55), FONT, 0.42, DIM, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([top, row]))
