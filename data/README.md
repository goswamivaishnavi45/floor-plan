# Sample data

Three LiDAR recordings made with the Stray Scanner iOS app, all of the same apartment.

| Folder | Original zip | Notes |
|---|---|---|
| `c00a170fe1` | single_room.zip | Short single-room scan (~37 s) |
| `1a8384c3f6` | single_scan_floor_only.zip | Camera aimed mostly down; ceiling barely covered |
| `c7d28f72c6` | single_scan_with_ceiling.zip | Camera also aimed up; ceiling covered |

Each folder contains:

- `rgb.mp4`: colour video, 1920x1440
- `depth/`: one 16-bit PNG per frame, distance in millimetres
- `confidence/`: LiDAR confidence per depth pixel (0 = low, 1 = medium, 2 = high)
- `odometry.csv`: camera position and rotation per frame, from ARKit
- `camera_matrix.csv`: camera intrinsics (focal length, image centre)
- `imu.csv`: accelerometer and gyroscope readings

The recordings are not stored in git because of their size. Place them in this folder before running the pipeline.
