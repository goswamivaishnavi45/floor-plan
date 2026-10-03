# Example outputs

What `python run.py <capture>` produced on the sample data with the code in this repository, copied here so the outputs can be read without running anything. Each folder holds `result.json` (validated against `schema/result.schema.json`) and `plan.png`.

| Folder | Command | Tier |
|---|---|---|
| `lidar_c00a170fe1/` | `python run.py data/c00a170fe1` | LiDAR, 37 s capture |
| `lidar_1a8384c3f6/` | `python run.py data/1a8384c3f6` | LiDAR, whole apartment, ceiling not filmed |
| `lidar_c7d28f72c6/` | `python run.py data/c7d28f72c6` | LiDAR, whole apartment, ceiling filmed |
| `video_c00a170fe1/` | `python run.py data/c00a170fe1/rgb.mp4 --rotate cw` | video (the capture's colour video only) |
| `photo_c7d28f72c6/` | `python run.py data/photos_c7d28f72c6` | photo (folders made by `scripts/make_photo_folders.py c7d28f72c6`) |

Read the `warnings` list in each `result.json`: it says what was not measured and why. Sizes are `{"value", "pm95"}`, a 95% range. See `BENCHMARK.md` for how these numbers compare between captures.
