# Technical report

Phone capture to dimensioned floor plan, damage and repair scope. Vaishnavi Goswami, October 2026.

## 1. Architecture

One command per capture (`python run.py <capture>`) detects the tier from the input and writes `result.json` (validated against `schema/result.schema.json`) and `plan.png`.

```
LiDAR   Stray Scanner export ──► fuse depth frames (20 s chunks) ──► drift: floor anchor ──┐
Video   video file ──► sharp frames ──► COLMAP pieces ──► AI-depth size ──► per piece ────┤
Photo   room folders ──► VGGT per room ──► AI-depth size ──► quality gate ───────────────┤
                                                                                        ▼
        straighten ─► walls ─► rooms (split at narrow passages) ─► heights ─► sizes ± ─► result.json
        damage (OWLv2) ─► surface + size (LiDAR) ─► concealed-damage rules ─► scope ────► plan.png
```

**Point cloud (LiDAR).** Each depth pixel with high LiDAR confidence, depth 0.2-5 m and no depth jump of more than 5% to a neighbour (flying pixels at object edges) is back-projected with that frame's intrinsics and pose. Stray Scanner's poses are already in OpenCV camera axes; we verified this because frames overlap far better that way (41k vs 112k occupied 5 cm voxels). Intrinsics are taken per frame, because autofocus moves the focal length by up to 1.2% within a capture (about 2 cm at the image edge 3 m away). Points are merged on a 2 cm voxel grid as count-weighted means. Video frames are read sequentially: seeking in these H.264 files returned the wrong frame in 4 of 5 tests, up to 15 frames off.

**Layout.** All steps work on the top view.
- *Straighten:* turn the cloud about the vertical until wall points pile into the fewest 2 cm strips along x and z (0.5 degree search, then 0.05 degree). Leftover tilt of 0.5 degrees would skew a 4 m wall by 3.5 cm.
- *Walls:* a wall at one x (or z) is a spike in the strip histogram. Each spike is followed in 5 cm steps; a step counts only if its points reach half the band height (a sofa's seat and back do not). Holes up to 30 cm are bridged. A piece must be at least 40 cm long and thin: its ±2 cm core at least 1.5x denser than the strips 4-7 cm beside it, which rejects wardrobes and door frames. Its position is the median of its points, much finer than the 2 cm strip.
- *Rooms:* everything seen at any height marks inside space (LiDAR cannot see through walls). Walls are drawn as barriers, then the free floor is shrunk by 45 cm so every doorway (0.7-0.9 m) closes, each island becomes a room seed, and the seeds are grown back so rooms meet at doorways (the fix-loop change, section 6). Gaps of 0.6-1.2 m between wall ends are recorded as doors.
- *Heights:* per room, the floor is the median height of the room's points within ±8 cm of the rough floor. The ceiling is searched in 2 cm shelves at least 1.8 m above it. A ceiling is reported only if it was seen over 25% of the room. On `c7d28f72c6` ceilings range 2.29-3.08 m between rooms; one whole-home value would have been up to 72 cm wrong.
- *Sizes:* each room side is the wall that bounds the grown room along most of that side, with room cells on one side of it only. A side with no such wall is reported as missing, with a 5 cm sigma; the edge of the seen floor is used only to draw it. Non-rectangular rooms get a right-angled outline snapped to walls within 15 cm.

## 2. Tier design and device matrix

| Tier | Capture | Geometry | Metric size | On our data |
|---|---|---|---|---|
| LiDAR | Stray Scanner, iPhone Pro | LiDAR depth + ARKit poses | LiDAR | full plan |
| Video | Camera app, any iPhone | COLMAP structure from motion, in pieces | Depth Anything V2 / 1.35 bias | 0 complete rooms |
| Photo | Camera app, any iPhone | VGGT per room folder | Depth Anything V2 / 1.35 bias; focal from EXIF | rooms linked, 0 of 5 measured |

The full device matrix and per-tier protocol are in `CAPTURE_PROTOCOL.md`.

**Video.** COLMAP placed frames accurately where it worked: path error 2.5 cm (1.05%) and focal 0.7% off, without help. But it broke the single-room video into 6 pieces and the whole-apartment video into dozens, because plain white walls, glossy tiles and quick turns leave too few matchable spots. Each piece gets a size from Depth Anything (median depth ratio at COLMAP's spots, divided by the 1.35 bias measured against LiDAR) and an up direction from the room's box directions. Chaining pieces end to end produced a 4.7 x 11.6 m strip for a 9.4 x 7 m home, because the walker doubles back. The tier therefore measures rooms per piece and does not stitch them.

**Photo.** VGGT (one network for all photos of a room) gave 3-4% depth error on 8 overlapping frames from one walk, against 15% for single-picture depth. On photos taken all around a room it misplaced cameras by 0.6-0.7 m and turned two by 55-135 degrees, at both 350 and 518 px input. Rooms are therefore reported only past a quality gate (walls on at least 3 sides, photos agreeing on size within 10%). Rooms are linked by doorway photos stored in two folders, compared by content hash. The plan is a schematic.

## 3. Drift handling

ARKit integrates motion, so error accumulates. The capture is fused in 20 s chunks. Where a chunk's floor overlaps floor already placed (10 cm cells), the chunk is moved up or down by the median height difference. Comparing only overlapping spots leaves stairs alone: anchoring to one global floor lifted half of `c7d28f72c6` by 2.1 cm. Each correction is bounded to 5 cm, and only layers within 15 cm of the capture's floor can be floor: chunks looking at the ceiling had produced a -2.1 m "correction". A wall anchor, which slides and turns chunks onto walls already placed, is implemented but off by default: its small turns accumulated to 4 degrees and broke 2 of 5 rooms.

Ablation (`scripts/drift_ablation.py`, same chunks with recorded poses vs corrected): floor spread between rooms 2.4 → 2.1 cm and 0.4 → 1.3 cm; wall spread 2.10 → 2.11 cm and 2.13 → 2.17 cm; rooms 5 → 5 on both. ARKit's drift here is about 1-2 cm and the correction works at that noise level: it keeps the plan intact without clearly improving it.

## 4. Error budget

Every measurement is reported as `value ± pm95`, a 95% range (2 sigma).

| Source | Size (1 sigma) | Applies to |
|---|---|---|
| LiDAR bias left in a fitted surface | 5 mm (assumed, from published iPhone LiDAR tests) | each wall, floor, ceiling |
| Point scatter of a wall / sqrt(points) | under 1 mm for thousands of points | each wall |
| Drift | half the spread of room floor heights in the capture, at least 3 mm | each dimension |
| Side with no detected wall | 5 cm | that side |
| Ceiling seen over less than half the room | +1 cm | ceiling height |
| Doorway width (wall ends on 5 cm steps) | 2 cm | openings |
| Overall size from AI depth | 10% (video), 15% (photo) | every length; area twice that |
| Damage box to surface size | 15% | damage width and height |

A room width combines two sides and drift: for a room with both walls found that is about ±1.5 cm. A side without a wall raises it to about ±10 cm.

## 5. Calibration analysis

The ranges are built from error sources, not fitted to outcomes. Without tape or laser measurements of these rooms we cannot check that 95% ranges contain the truth 95% of the time. What we can check:

- **Repeatability vs ranges.** Walls found in both captures differ by 0.55 cm (median), inside the ±1.5 cm stated for a two-wall dimension. Room dimensions differ by 10-40%, far outside their stated ranges. The ranges are consistent for walls but **overconfident for room sizes**, because they model measurement noise and not the risk of a room side being taken from a different wall.
- **Video and photo size terms** come from measured errors: per-piece size errors of -6% to +18% (video, pieces over 1 m of path) and photo-tier size disagreement of 10-14%. The photo gate refuses rather than widening to an uninformative ±90%.
- **Damage threshold** is set from data: the detector's highest score on clean frames was 0.26, so detections need 0.30.

Calibrating properly needs a tape-measured benchmark: at least 20 dimensions per tier, then scale each error term until 95% of truths fall inside the ranges.

## 6. Fix loop

Declaration first (`fixloop/declaration.md`, commit `32c0ecf`), then the fix (`6cab0ca`), after results, and a post-mortem.

- **Worst measurable gate:** repeatability. Two captures of the same apartment matched 3 of 5 rooms and only 1 of 6 room dimensions within 1 cm / 0.5% (17%).
- **Hypothesis:** room division was unstable. The paint-bucket fill stopped at a doorway only if both its wall ends were detected; a missed end merged rooms in one capture only. Evidence: walls agreed within 0.5 cm, while matched rooms overlapped only 43-79%, and R2 included the corridor in one capture only.
- **Fix:** split rooms at narrow passages (erode 45 cm, seed, grow back), which depends only on the floor's shape.
- **Prediction:** at least 4 rooms matched; about 50% of dimensions within the gate.
- **Result:** rooms matched 3 → 7, so the cause was real and fixed. Dimensions 17% → 7%, so the prediction was badly wrong.
- **Post-mortem:** matching rooms exposed the next cause. Room sides fall back to the edge of the seen floor when no wall is detected (9 of 28 sides in the floor-only capture vs 4 of 28), and the nearest-wall rule can pick different walls (R4 differs by 1.09 m with walls found on almost every side). I predicted from the wall-to-wall agreement without checking how sides are chosen.
- **Follow-up (after the fix loop):** sides are now taken from walls that bound the grown room along most of the side. This raised repeatable dimensions from 1 to 2 of 14. The rest of the gap is rooms still divided differently in the two captures (R6/R8 overlap 0.33); R4's width still differs by 1.48 m.

## 7. Known failure modes

1. **Room sizes are not repeatable** (section 6: 2 of 14 dimensions within the gate), though wall detection is.
2. **Glass, mirrors, windows:** low-confidence LiDAR returns are dropped, so glass walls go missing and rooms can leak through them; curtains make wavy walls.
3. **Open plan and wide openings** (over 0.9 m) stay one room. Corridors under 0.9 m and cupboards are absorbed into neighbours.
4. **Floor-only captures** cannot give ceiling heights; the output says "not seen".
5. **Video:** plain walls fragment COLMAP. No complete room on our video; segments are not stitched.
6. **Photo:** VGGT misplaces photos spread around a plain-walled room; rooms come out "not measured". Real overlapping photos taken per the protocol may do better; untested.
7. **AI-depth size bias** (1.35x) was measured on one apartment only.
8. **Damage:** on 3 phone photos of real damage the detector found 3 areas, missed 1 and raised 2 false alarms (a door frame as a crack, a clean panel as peeling paint); not validated on staged damage in a capture. Sizes are box-based (±15%) and assume a frontal view; video and photo tiers give no size.
9. **Multi-storey homes:** one plan for all floors; stairs are not modelled.
10. **Speed on CPU:** photo tier about 3 min per room, damage about 10 s per frame.

## Models and licences

Depth Anything V2 Metric Indoor Small (Apache-2.0), VGGT-1B (CC-BY-NC-4.0, non-commercial; a product would need Meta's commercial variant), OWLv2 base patch16 ensemble (Apache-2.0), COLMAP via pycolmap (BSD). All run locally after `python scripts/fetch_models.py`.
