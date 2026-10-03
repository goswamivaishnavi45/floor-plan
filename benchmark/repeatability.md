# Repeatability: 1a8384c3f6 vs c7d28f72c6 (LiDAR tier)

Plans lined up with 3 quarter turn(s) and a shift of [-0.05, 6.05] m. Rooms: 5 vs 5, matched 3; only in 1a8384c3f6: ['R3', 'R5']; only in c7d28f72c6: ['R1', 'R5'].

Gate (brief): two captures agree within 1 cm or 0.5% per wall dimension. **1 of 6 dimensions pass (17%).**

Walls compared directly (ignoring rooms): 13 walls over 1 m found in both; median position difference 0.5 cm; 7 within 1 cm.

| Room A | Room B | Overlap | Dimension | A (m) | B (m) | B - A | % | Within gate |
|---|---|---|---|---|---|---|---|---|
| R1 | R4 | 0.69 | width | 5.595 | 6.481 | +0.886 | +15.8 | no |
| R1 | R4 | 0.69 | length | 3.342 | 3.719 | +0.377 | +11.3 | no |
| R1 | R4 | 0.69 | floor area | 8.266 | 11.776 | +3.510 | +42.5 | n/a |
| R1 | R4 | 0.69 | ceiling | n/a | 3.081 | n/a | n/a | n/a |
| R2 | R2 | 0.43 | width | 10.324 | 6.373 | -3.951 | -38.3 | no |
| R2 | R2 | 0.43 | length | 6.075 | 7.099 | +1.024 | +16.9 | no |
| R2 | R2 | 0.43 | floor area | 25.477 | 18.514 | -6.963 | -27.3 | n/a |
| R2 | R2 | 0.43 | ceiling | n/a | 2.456 | n/a | n/a | n/a |
| R4 | R3 | 0.79 | width | 5.405 | 2.957 | -2.448 | -45.3 | no |
| R4 | R3 | 0.79 | length | 3.004 | 3.009 | +0.005 | +0.2 | yes |
| R4 | R3 | 0.79 | floor area | 11.298 | 8.900 | -2.398 | -21.2 | n/a |
| R4 | R3 | 0.79 | ceiling | n/a | 2.976 | n/a | n/a | n/a |
