# Benchmark report

All numbers are regenerable from the raw captures with the commands given; see `README.md` for setup.

## Benchmark set and its gaps

The brief asks for a benchmark we build ourselves. What we have is the provided sample data: three Stray Scanner LiDAR captures of one two-storey apartment.

| Capture | Content | Used as |
|---|---|---|
| `c00a170fe1` | 37 s, part of the apartment | LiDAR tier, video tier (its `rgb.mp4`) |
| `1a8384c3f6` | 87 s, whole apartment, camera pointed down (floor only) | LiDAR tier, repeatability capture A |
| `c7d28f72c6` | 162 s, whole apartment, ceiling covered | LiDAR tier, repeatability capture B, photo tier (photos cut from its video) |

Required by the brief and **not available**, so the matching gates are reported as not measured rather than estimated:

- **Tape or laser ground truth.** None was provided and we had no access to the apartment. Absolute accuracy gates (opening widths, ceiling height, wall lengths against truth) cannot be scored.
- **A furnished room with staged damage.** The apartment shows no visible damage.
- **Native video and photo captures.** The video tier was run on the LiDAR captures' `rgb.mp4` (sideways, `--rotate cw`). Photo folders were cut from `c7d28f72c6`'s video with `scripts/make_photo_folders.py`: frames filed by room using the LiDAR plan, iPhone-style EXIF focal length written in. LiDAR was used only to sort frames and to check results, never as input.
- **Head-to-head with a consumer app.** It needs magicplan or Polycam run on the same rooms, i.e. access to the apartment and a LiDAR iPhone. Not done.

## Gates

| Gate | Tier | Result | Status |
|---|---|---|---|
| Opening widths <= 2 cm on >= 85% | all | No ground truth. Doorway widths are reported with a +-4 cm range (wall ends found on 5 cm steps) | not measured |
| Ceiling height <= 1.5 cm per room | LiDAR | No ground truth. Per-room ceilings found where the ceiling was seen (c7d28f72c6: 2.29-3.08 m, +-2 cm reported) | not measured |
| Ceiling spread across captures <= 1 cm | LiDAR | Only one capture saw the ceiling (`1a8384c3f6` reports "not seen" in every room, by design) | not measurable |
| Repeatability within 1 cm or 0.5% per wall | LiDAR | Walls: 0.55 cm median difference, 4 of 8 within 1 cm. Room dimensions: 1 of 14 (7%) | **fail** (fix loop) |
| Drift accountability, on/off ablation | LiDAR | Implemented (floor-anchored chunks); ablation below | done |
| Photo-tier whole-property stitch, +-8% | photo | Rooms linked correctly (4 of 4 doorways); no room passed the quality gate, so no stitched dimensions | **fail** |
| Wall lengths +-8% (photo) / +-3% (video) | photo, video | No room measured on either tier | **fail** |
| Head-to-head, beat or tie >= 70% | LiDAR | Not run | not done |

## Repeatability (LiDAR)

`python scripts/repeatability.py outputs/1a8384c3f6 outputs/c7d28f72c6` after `run.py` on both captures. Full table: `benchmark/repeatability.md` (current pipeline); before and after the fix loop: `fixloop/before/`, `fixloop/after/`.

| | Before fix | After fix (current) |
|---|---|---|
| Rooms found (A / B) | 5 / 5 | 9 / 8 |
| Rooms matched | 3 | 7 |
| Room dimensions within gate | 1 of 6 (17%) | 1 of 14 (7%) |
| Walls in both, median position difference | 0.5 cm (13 walls) | 0.55 cm (8 walls) |

Walls are measured consistently; room sizes are not, because a room side without a detected wall falls back to the edge of the seen floor (9 of 28 sides in the floor-only capture). See `fixloop/postmortem.md`.

## Drift ablation (LiDAR)

`python scripts/drift_ablation.py data/<capture>`: the same 20 s chunks merged with the recorded poses (off) and with correction (on). Pictures: `outputs/<capture>_drift_ablation.png`.

| | 1a8384c3f6 off | on | c7d28f72c6 off | on |
|---|---|---|---|---|
| Floor spread between rooms | 2.4 cm | 2.1 cm | 0.4 cm | 1.3 cm |
| Wall spread (median std across a wall) | 2.10 cm | 2.11 cm | 2.13 cm | 2.17 cm |
| Walls over 1 m | 31 | 32 | 31 | 34 |
| Rooms | 5 | 5 | 5 | 5 |

ARKit's own drift on these captures is about 1-2 cm. The floor anchor keeps the plan intact but does not clearly improve it. The wall anchor (also implemented, `WALL_ANCHOR`) made things worse in an earlier ablation (walls over 1 m 31 -> 22, rooms 5 -> 3 on `c7d28f72c6`) because its small turns accumulated, so it is off by default.

## Component checks against LiDAR

| Check | Script | Result |
|---|---|---|
| Single-picture AI depth (Depth Anything V2) vs LiDAR | `scripts/compare_depth.py` | Reads 1.35x too far on this home; per-frame size wobble 22-32%; shape error 15-19% after size correction. Sideways input 2-3x worse |
| COLMAP camera path vs ARKit | `scripts/compare_path.py` | Where it works: 2.5 cm path error (1.05%), focal 0.7% off. Breaks into 6 pieces (single room) to dozens (whole apartment) on plain walls |
| Video pieces: size and up direction | `scripts/check_pieces.py` | Size error -6% to +44% per piece; up within 2-6 degrees on 5 of 6 pieces |
| VGGT, 8 overlapping frames of one room | `scripts/check_vggt.py` | Depth error 3.1-4.3%, size wobble 2.9%, camera path 1.2-3.9%, focal 8-16% off |
| VGGT, photos around a room | `scripts/check_photo_room.py` | Camera error 0.58-0.72 m; two photos off by 55-135 degrees; room shape distorted |
| Damage detector on clean frames | `src/damage.py` | Highest score 0.26 on undamaged frames, so threshold 0.30 |

## Timing (laptop, Intel i5-1235U, CPU only, 16 GB)

| Run | Time |
|---|---|
| LiDAR `c00a170fe1` (37 s capture) | 38 s, plus ~3.5 min damage search |
| LiDAR `1a8384c3f6` (87 s) | 81 s, plus damage |
| LiDAR `c7d28f72c6` (162 s) | 126 s, plus damage |
| Video `c00a170fe1/rgb.mp4` | 5.3 min, plus damage |
| Photo, 5 rooms x 9-11 photos | 20 min (VGGT ~3 min per room), plus damage |
| Damage search | ~10 s per frame, 20 frames |
