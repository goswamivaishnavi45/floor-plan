# Fix loop post-mortem

Declaration: `fixloop/declaration.md` (committed before the fix, `32c0ecf`).
Fix: split rooms at narrow passages instead of detected doorways (`6cab0ca`, `find_rooms` in `src/layout.py`).
Before / after: `fixloop/before/`, `fixloop/after/` (regenerate with the commands in the declaration).

## Result against the prediction

| | Before | After | Predicted |
|---|---|---|---|
| Rooms matched between the two captures | 3 of 5 | 7 (of 9 vs 8) | at least 4 |
| Room dimensions within 1 cm / 0.5% | 1 of 6 (17%) | 1 of 14 (7%) | about 50% |
| Walls compared directly, median difference | 0.5 cm | 0.55 cm | (unchanged) |

The gate did not pass, and the dimension prediction was badly wrong.

## What the fix did

The declared root cause was real and the fix addressed it. Room division now agrees between captures: 7 rooms match instead of 3, and the overlay (`fixloop/after/repeatability.png`) shows the top rooms, the middle room and the right-hand room divided the same way in both. Before the fix, R2 included the corridor in one capture only.

## Why the dimensions did not follow

Matching rooms only exposes the next problem: how a room's size is measured once it is found. Each room's width and length come from its four sides, and each side is the room's wall nearest the edge of its painted floor (within 30 cm), or the painted edge itself when no wall qualifies. Two effects make this unstable:

1. **Fallback to the floor edge.** In the matched rooms, 9 of 28 sides in the floor-only capture `1a8384c3f6` have no detected wall, against 4 of 28 in `c7d28f72c6`. The floor-only capture was filmed pointing down and sees less of each room, so its painted floor stops short of the walls and its rooms come out smaller. The overlay shows its outlines (blue) inside the other capture's (red).
2. **A different wall picked as the side.** R4 matches R4 with walls found on almost every side in both captures and still differs by 1.09 m in width. The "nearest wall to the painted edge" rule can pick a wardrobe front in one capture and the room wall in the other.

The prediction assumed room sides would come from the same walls once rooms matched, so the remaining error would be the wall-to-wall difference (0.5 cm median). That assumption was not checked before predicting; checking how many sides fall back to the painted edge would have shown it.

## Next fix (not shipped)

Take room sides only from walls that bound the grown room region (walls the room's cells actually touch), prefer the longest such wall per side, and report a side as unmeasured instead of falling back to the painted edge. That removes both effects; walls themselves repeat within about 0.5 cm.
