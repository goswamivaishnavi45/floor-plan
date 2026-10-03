# Fix declaration

Written and committed before the fix (code at commit `59656e1`).

## 1. Worst-performing gate

**Repeatability** (two captures of the same rooms at the same tier agree within 1 cm or 0.5% per wall).

LiDAR tier, `1a8384c3f6` vs `c7d28f72c6` (the same apartment captured twice), drift correction on:

- rooms matched between the two plans: **3 of 5**
- room dimensions within the gate: **1 of 6 (17%)**
- example: room R2 is 10.32 m wide in one capture and 6.37 m in the other

The other gates cannot be scored without tape or laser ground truth, which this sample data does not include. Repeatability is the worst gate we can measure.

## 2. Root cause and evidence

**Hypothesis:** the room division is unstable, not the wall measurement.

Rooms come from a paint-bucket fill that only stops at a doorway when both wall ends around it were detected and joined by a line (`find_gaps` in `src/layout.py`). If one wall end is missed in a capture (occluded, door leaf open, short stub), paint leaks through and the two rooms merge in that capture but not the other. Room dimensions are then measured between different walls.

**Evidence** (`fixloop/before/repeatability.md`):

- Walls compared directly, ignoring rooms: 13 walls over 1 m found in both captures, **median position difference 0.5 cm**, 7 within 1 cm. Walls repeat.
- Rooms that do match overlap only **43-79%** (intersection over union), and 2 rooms per capture have no partner.
- R2's width differs by 3.95 m because one capture's R2 includes the corridor: a merge, not a measurement error.

## 3. Fix

Split rooms at **narrow passages** instead of at detected doorways:

1. Shrink the free floor area inwards by 45 cm (morphological erosion). Doorways are 70-90 cm wide, so every doorway closes and each room becomes a separate island.
2. Each island is a room seed.
3. Grow the seeds back over the full floor area, each floor cell going to the nearest seed through floor (geodesic growth), so neighbouring rooms meet at the doorways.

This depends only on the shape of the floor, which the evidence shows is repeatable, and not on finding both ends of every doorway. Small wall holes (< 0.6 m) are still closed before growing; doorway gaps are still recorded as openings but no longer used to divide rooms.

## Prediction

- rooms matched: **at least 4 of 5**
- room dimensions within the gate: from **17% to about 50%**

Not 100%: some room sides will still be taken from different walls in the two captures, and walls themselves differ by up to 1-3 cm.

## Regenerating

```
git checkout <commit>           # 59656e1 for before; the fix commit for after
python run.py data/1a8384c3f6
python run.py data/c7d28f72c6
python scripts/repeatability.py outputs/1a8384c3f6 outputs/c7d28f72c6
```

The before results are copied into `fixloop/before/`; the after results go to `fixloop/after/`.
