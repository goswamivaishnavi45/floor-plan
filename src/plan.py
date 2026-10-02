"""Draw plan.png, the floor plan a homeowner sees, from result.json.

White background, each room as a clean outline with its wall lengths written
along the walls, doors as red gaps with their width, and a label with the
room's name, size, area and ceiling height.
"""

import cv2
import numpy as np

PIXELS_PER_METRE = 80
MARGIN = 90   # px around the plan for dimension labels
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


def draw_plan(result, path):
    corners = [p for room in result["rooms"] for p in room["outline"]]
    corners += [p for o in result["openings"] for p in (o["start"], o["end"])]
    if not corners:
        image = np.full((300, 600, 3), 255, np.uint8)
        centred_text(image, "No rooms found", (300, 150), 0.8, TEXT, 2)
        cv2.imwrite(str(path), image)
        return
    corners = np.array(corners)
    low, high = corners.min(axis=0), corners.max(axis=0)
    size = ((high - low) * PIXELS_PER_METRE + 2 * MARGIN).astype(int)
    image = np.full((size[1] + 60, size[0], 3), 255, np.uint8)

    def px(p):
        return (int(round((p[0] - low[0]) * PIXELS_PER_METRE + MARGIN)),
                int(round((p[1] - low[1]) * PIXELS_PER_METRE + MARGIN + 60)))

    # Room fills first, then walls on top, so shared walls stay crisp.
    for room in result["rooms"]:
        cv2.fillPoly(image, [np.array([px(p) for p in room["outline"]], np.int32)], ROOM_FILL)
    for room in result["rooms"]:
        outline = np.array([px(p) for p in room["outline"]], np.int32)
        cv2.polylines(image, [outline], True, WALL, 4, cv2.LINE_AA)

    # Doors: cut the wall and mark the opening.
    for opening in result["openings"]:
        a, b = px(opening["start"]), px(opening["end"])
        cv2.line(image, a, b, (255, 255, 255), 7)
        cv2.line(image, a, b, DOOR, 2, cv2.LINE_AA)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        centred_text(image, f"{opening['width']['value']:.2f}", (mid[0], mid[1] - 12), 0.4, DOOR)

    # Wall lengths, written just inside each wall, near its middle.
    for room in result["rooms"]:
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
    for room in result["rooms"]:
        cx, cy = np.mean([px(p) for p in room["outline"]], axis=0)
        ceiling = room["ceiling_height"]
        lines = [(room["name"], 0.55, 2),
                 (f"{room['width']['value']:.2f} x {room['length']['value']:.2f} m", 0.42, 1),
                 (f"{room['floor_area']['value']:.1f} m2 +-{room['floor_area']['pm95']:.1f}", 0.42, 1),
                 (f"ceiling {ceiling['value']:.2f} m" if ceiling["value"] is not None else "ceiling not seen", 0.42, 1)]
        for k, (text, scale, thickness) in enumerate(lines):
            centred_text(image, text, (cx, cy - 24 + 17 * k), scale, TEXT, thickness)

    total = result["property"]["total_floor_area"]
    title = (f"{result['capture']['id']}  |  {result['capture']['tier']} tier  |  "
             f"{result['property']['room_count']} rooms, {total['value']:.1f} +-{total['pm95']:.1f} m2")
    cv2.putText(image, title, (20, 30), FONT, 0.6, TEXT, 2, cv2.LINE_AA)
    cv2.putText(image, "lengths in metres; ? = side not seen as a wall; red = door width",
                (20, 52), FONT, 0.42, DIM, 1, cv2.LINE_AA)
    h = image.shape[0]
    cv2.line(image, (20, h - 20), (20 + PIXELS_PER_METRE, h - 20), TEXT, 3)
    cv2.putText(image, "1 m", (20, h - 28), FONT, 0.5, TEXT, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)
