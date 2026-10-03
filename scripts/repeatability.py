"""Repeatability: do two captures of the same rooms give the same plan?

    python scripts/repeatability.py outputs/1a8384c3f6 outputs/c7d28f72c6

Reads both result.json files (run.py on each capture first) and:
  1. lines plan B up with plan A: both plans are straightened, so only the
     4 quarter turns are tried; for each, the shift is found where the two
     plans' room areas overlap most (cross-correlation of 5 cm room masks)
  2. matches rooms: a room of A and a room of B are the same room when
     their areas overlap by at least MIN_IOU (intersection over union)
  3. compares width, length, floor area and ceiling height of each pair
     against the brief's gate: within 1 cm or 0.5% (per wall dimension)

  4. compares walls directly, ignoring rooms: for each wall of A over 1 m,
     the nearest parallel wall of B (same direction after the turn, within
     10 cm, overlapping along its length) and their position difference.
     This separates "are walls measured the same" from "are rooms divided
     the same"

Writes benchmark/repeatability.md, benchmark/repeatability.json and
benchmark/repeatability.png (plan A in blue, aligned plan B in red).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CELL = 0.05        # m, room mask resolution for lining the plans up
MIN_IOU = 0.3      # rooms overlapping this much (intersection / union) are the same room
GATE_ABS = 0.01    # m
GATE_REL = 0.005   # 0.5%


def turn(points, quarter_turns):
    """Turn plan points by quarter_turns x 90 degrees about the origin."""
    x, z = np.asarray(points, float).T
    for _ in range(quarter_turns % 4):
        x, z = -z, x
    return np.column_stack([x, z])


def masks(rooms, origin, shape):
    """One filled mask per room (rooms with an outline only)."""
    out = {}
    for room in rooms:
        if "outline" not in room:
            continue
        m = np.zeros(shape, np.uint8)
        poly = np.round((np.asarray(room["outline"]) - origin) / CELL).astype(np.int32)
        cv2.fillPoly(m, [poly], 1)
        out[room["id"]] = m
    return out


def best_shift(a, b):
    """Shift (rows, cols) of mask b that overlaps mask a the most, by FFT
    cross-correlation."""
    fa, fb = np.fft.rfft2(a), np.fft.rfft2(b)
    corr = np.fft.irfft2(fa * np.conj(fb), s=a.shape)
    r, c = np.unravel_index(np.argmax(corr), corr.shape)
    if r > a.shape[0] // 2:
        r -= a.shape[0]
    if c > a.shape[1] // 2:
        c -= a.shape[1]
    return int(r), int(c), float(corr.max())


def align(rooms_a, rooms_b):
    """Quarter turn and shift (metres) that line plan B up with plan A."""
    pts_a = np.vstack([r["outline"] for r in rooms_a if "outline" in r])
    best = None
    for k in range(4):
        turned = [dict(r, outline=turn(r["outline"], k).tolist()) for r in rooms_b if "outline" in r]
        pts_b = np.vstack([r["outline"] for r in turned])
        low = np.minimum(pts_a.min(0), pts_b.min(0)) - 1.0
        high = np.maximum(pts_a.max(0), pts_b.max(0)) + 1.0
        span = high - low
        shape = (int(np.ceil(2 * span[1] / CELL)), int(np.ceil(2 * span[0] / CELL)))
        mask_a = sum(masks(rooms_a, low, shape).values()).clip(0, 1).astype(float)
        mask_b = sum(masks(turned, low, shape).values()).clip(0, 1).astype(float)
        r, c, score = best_shift(mask_a, mask_b)
        if best is None or score > best[0]:
            best = (score, k, np.array([c, r]) * CELL)
    return best[1], best[2]


def wall_lines(result_walls, k, shift):
    """Room walls as (axis, position, start, end) after turning by k quarter
    turns and shifting (plan B into plan A's frame). Duplicates (a wall shared
    by two rooms) are merged by rounding."""
    lines = set()
    for room in result_walls:
        for w in room.get("walls", []):
            if not w["detected"] or w["length"]["value"] < 1.0:
                continue
            a, b = turn([w["start"], w["end"]], k) + shift
            axis = "x" if abs(a[0] - b[0]) < 1e-6 else ("z" if abs(a[1] - b[1]) < 1e-6 else None)
            if axis is None:
                continue
            pos = a[0] if axis == "x" else a[1]
            lo, hi = sorted((a[1], b[1]) if axis == "x" else (a[0], b[0]))
            lines.add((axis, round(pos, 3), round(lo, 2), round(hi, 2)))
    return sorted(lines)


def within_gate(a, b):
    diff = abs(a - b)
    return diff <= GATE_ABS or diff <= GATE_REL * (a + b) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args()

    result_a = json.loads((Path(args.a) / "result.json").read_text())
    result_b = json.loads((Path(args.b) / "result.json").read_text())
    rooms_a = [r for r in result_a["rooms"] if "outline" in r]
    k, shift = align(rooms_a, [r for r in result_b["rooms"] if "outline" in r])
    rooms_b = [dict(r, outline=(turn(r["outline"], k) + shift).tolist())
               for r in result_b["rooms"] if "outline" in r]

    all_pts = np.vstack([r["outline"] for r in rooms_a + rooms_b])
    origin = all_pts.min(0) - 0.5
    shape = tuple((np.ceil((all_pts.max(0) + 0.5 - origin) / CELL)).astype(int)[::-1])
    ma, mb = masks(rooms_a, origin, shape), masks(rooms_b, origin, shape)

    pairs, used = [], set()
    for ra in rooms_a:
        scores = [((ma[ra["id"]] & mb[rb["id"]]).sum() / max((ma[ra["id"]] | mb[rb["id"]]).sum(), 1), rb)
                  for rb in rooms_b if rb["id"] not in used]
        if not scores:
            continue
        iou, rb = max(scores, key=lambda s: s[0])
        if iou >= MIN_IOU:
            used.add(rb["id"])
            pairs.append((ra, rb, float(iou)))

    rows, checks = [], []
    for ra, rb, iou in pairs:
        # A quarter turn swaps which side is "width" (along x) and "length".
        wb, lb = (rb["length"], rb["width"]) if k % 2 else (rb["width"], rb["length"])
        for name, va, vb in (("width", ra["width"], wb), ("length", ra["length"], lb),
                             ("floor area", ra["floor_area"], rb["floor_area"]),
                             ("ceiling", ra["ceiling_height"], rb["ceiling_height"])):
            if va["value"] is None or vb["value"] is None:
                rows.append((ra["id"], rb["id"], iou, name, va["value"], vb["value"], None, None, None))
                continue
            diff = vb["value"] - va["value"]
            ok = within_gate(va["value"], vb["value"]) if name != "floor area" else None
            if ok is not None:
                checks.append(ok)
            rows.append((ra["id"], rb["id"], iou, name, va["value"], vb["value"], diff,
                         100 * diff / va["value"], ok))

    only_a = [r["id"] for r in rooms_a if r["id"] not in {p[0]["id"] for p in pairs}]
    only_b = [r["id"] for r in rooms_b if r["id"] not in used]
    summary = {"a": result_a["capture"]["id"], "b": result_b["capture"]["id"],
               "quarter_turns": k, "shift_m": shift.round(2).tolist(),
               "rooms_a": len(rooms_a), "rooms_b": len(rooms_b), "matched": len(pairs),
               "only_in_a": only_a, "only_in_b": only_b,
               "dimensions_checked": len(checks), "dimensions_within_gate": int(sum(checks)),
               "pass_rate": round(sum(checks) / len(checks), 3) if checks else None}

    walls_a, walls_b = wall_lines(rooms_a, 0, np.zeros(2)), wall_lines(result_b["rooms"], k, shift)
    wall_diffs = {"x": [], "z": []}
    for axis, pos, lo, hi in walls_a:
        near = [p for ax, p, l2, h2 in walls_b
                if ax == axis and abs(p - pos) <= 0.10 and min(hi, h2) - max(lo, l2) >= 0.5]
        if near:
            wall_diffs[axis].append(min(near, key=lambda p: abs(p - pos)) - pos)
    # The plans were lined up on a 5 cm grid; remove the leftover offset per
    # direction (the median difference) before comparing walls.
    d = np.concatenate([np.array(v) - np.median(v) for v in wall_diffs.values() if v] or [np.zeros(0)])
    if len(d):
        summary["walls_compared"] = len(d)
        summary["wall_position_diff_cm_median"] = round(100 * float(np.median(np.abs(d))), 2)
        summary["walls_within_1cm"] = int((np.abs(d) <= 0.01).sum())

    out = Path("benchmark")
    out.mkdir(exist_ok=True)
    lines = [f"# Repeatability: {summary['a']} vs {summary['b']} (LiDAR tier)", "",
             f"Plans lined up with {k} quarter turn(s) and a shift of {summary['shift_m']} m. "
             f"Rooms: {len(rooms_a)} vs {len(rooms_b)}, matched {len(pairs)}; "
             f"only in {summary['a']}: {only_a or 'none'}; only in {summary['b']}: {only_b or 'none'}.", "",
             f"Gate (brief): two captures agree within 1 cm or 0.5% per wall dimension. "
             f"**{sum(checks)} of {len(checks)} dimensions pass ({100 * (summary['pass_rate'] or 0):.0f}%).**", "",
             f"Walls compared directly (ignoring rooms): {summary.get('walls_compared', 0)} walls over 1 m "
             f"found in both; median position difference {summary.get('wall_position_diff_cm_median')} cm; "
             f"{summary.get('walls_within_1cm', 0)} within 1 cm.", "",
             "| Room A | Room B | Overlap | Dimension | A (m) | B (m) | B - A | % | Within gate |",
             "|---|---|---|---|---|---|---|---|---|"]
    for ida, idb, iou, name, va, vb, diff, pct, ok in rows:
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        lines.append(f"| {ida} | {idb} | {iou:.2f} | {name} | {fmt(va)} | {fmt(vb)} | "
                     f"{'n/a' if diff is None else f'{diff:+.3f}'} | {'n/a' if pct is None else f'{pct:+.1f}'} | "
                     f"{'n/a' if ok is None else ('yes' if ok else 'no')} |")
    (out / "repeatability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "repeatability.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    image = np.full(shape + (3,), 255, np.uint8)
    for room_masks, colour in ((ma, (200, 120, 40)), (mb, (40, 40, 220))):
        for m in room_masks.values():
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, colour, 2)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.putText(image, f"blue: {summary['a']}   red: {summary['b']} (aligned)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(str(out / "repeatability.png"), image)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
