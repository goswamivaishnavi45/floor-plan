"""Make photo-tier test input from a LiDAR capture (part 7a).

    python scripts/make_photo_folders.py c7d28f72c6

We have no real per-room photo folders, but every frame of a capture's
rgb.mp4 is a photo, and row N of its odometry.csv says where the phone was
when frame N was taken. With the LiDAR result's room outlines (run.py on the
same capture first) each frame can be filed under the room it was taken in:

  1. turn the phone position by the plan's straightening angle and look it
     up in the room outlines (frames within 20 cm of a wall are skipped)
  2. per room, keep PER_ROOM sharp photos facing different directions, the
     way a person turns around a room: split the full turn into PER_ROOM
     sectors and take the sharpest frame facing into each
  3. wherever the walk crosses from one room into another, the sharpest
     frame within ~1 s of the crossing goes into both rooms' folders (a
     shared view through the doorway, for stitching)

Writes data/photos_<capture>/<room id>/photo_NN.jpg (upright) and
manifest.json (frame numbers and the LiDAR measurements per room, used only
to check the photo tier, never as its input).
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.compare_depth import upright_rotation  # noqa: E402
from src.capture import load_capture  # noqa: E402
from src.layout import rotate_about_vertical  # noqa: E402
from src.video import sharpness  # noqa: E402

PER_ROOM = 6
WALL_MARGIN = 0.20     # m: skip frames taken right against a wall
CROSSING_FRAMES = 60   # frames (~1 s) either side of a room-to-room crossing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", help="capture name, e.g. c7d28f72c6")
    args = parser.parse_args()

    capture = load_capture(Path("data") / args.capture)
    result = json.loads((Path("outputs") / args.capture / "result.json").read_text())
    angle = result["capture"]["plan_rotation_deg"]

    # Phone position and viewing direction of every frame, in plan coordinates.
    positions = rotate_about_vertical(capture.poses[:, :3, 3], angle)[:, [0, 2]]
    forwards = rotate_about_vertical(capture.poses[:, :3, 2], angle)[:, [0, 2]]
    headings = np.degrees(np.arctan2(forwards[:, 0], forwards[:, 1])) % 360

    def room_of(p):
        for room in result["rooms"]:
            outline = np.array(room["outline"], np.float32)
            if cv2.pointPolygonTest(outline, (float(p[0]), float(p[1])), True) >= WALL_MARGIN:
                return room["id"]
        return None

    rooms_of_frame = [room_of(p) for p in positions]

    # Sharpness of every frame (one pass through the video).
    scores = np.zeros(capture.num_frames)
    video = cv2.VideoCapture(str(capture.root / "rgb.mp4"))
    i = 0
    while video.grab():
        if i < capture.num_frames and (rooms_of_frame[i] or i % 5 == 0):
            scores[i] = sharpness(video.retrieve()[1])
        i += 1

    chosen = {}   # frame -> list of (room id, file name)
    manifest = {"capture": args.capture, "rooms": {}, "doorways": []}
    for room in result["rooms"]:
        inside = [i for i in range(capture.num_frames) if rooms_of_frame[i] == room["id"]]
        picks = []
        for k in range(PER_ROOM):
            sector = [i for i in inside if int(headings[i] // (360 / PER_ROOM)) == k]
            if sector:
                picks.append(max(sector, key=lambda i: scores[i]))
        # Too few directions seen: fill with the sharpest remaining frames, spread in time.
        for i in sorted(set(inside) - set(picks), key=lambda i: -scores[i]):
            if len(picks) >= PER_ROOM:
                break
            if all(abs(i - j) > 60 for j in picks):
                picks.append(i)
        picks.sort()
        for n, i in enumerate(picks, start=1):
            chosen.setdefault(i, []).append((room["id"], f"photo_{n:02d}.jpg"))
        manifest["rooms"][room["id"]] = {
            "frames": picks, "frames_inside": len(inside),
            "lidar": {k: room[k] for k in ("width", "length", "floor_area", "ceiling_height")},
        }

    # Doorway photos: wherever the walk crosses from one room into another,
    # the frames around the crossing see both rooms. Keep the sharpest frame
    # within CROSSING_FRAMES of each crossing, one per pair of rooms.
    best_per_pair, last_room, last_frame = {}, None, None
    for i, room_id in enumerate(rooms_of_frame):
        if room_id is None:
            continue
        if last_room is not None and room_id != last_room:
            window = range(max(0, last_frame - CROSSING_FRAMES), min(capture.num_frames, i + CROSSING_FRAMES))
            frame = max(window, key=lambda j: scores[j])
            pair = tuple(sorted((last_room, room_id)))
            if pair not in best_per_pair or scores[frame] > scores[best_per_pair[pair]]:
                best_per_pair[pair] = frame
        last_room, last_frame = room_id, i
    for pair, frame in sorted(best_per_pair.items()):
        for room_id in pair:
            chosen.setdefault(frame, []).append((room_id, f"door_{pair[0]}_{pair[1]}.jpg"))
        manifest["doorways"].append({"rooms": list(pair), "frame": int(frame)})

    out = Path("data") / f"photos_{args.capture}"
    video = cv2.VideoCapture(str(capture.root / "rgb.mp4"))
    i = 0
    while i <= max(chosen) and video.grab():
        if i in chosen:
            frame = video.retrieve()[1]
            rotation = upright_rotation(capture.poses[i])
            if rotation is not None:
                frame = cv2.rotate(frame, rotation)
            for room_id, name in chosen[i]:
                (out / room_id).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / room_id / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        i += 1
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for room_id, info in manifest["rooms"].items():
        print(f"{room_id}: {len(info['frames'])} photos from {info['frames_inside']} frames inside")
    print(f"{len(manifest['doorways'])} doorway photos; -> {out}")


if __name__ == "__main__":
    main()
