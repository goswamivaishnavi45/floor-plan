# Floor plan and damage pipeline

Turns a phone capture of a home (LiDAR scan, video, or per-room photos) into a dimensioned floor plan, damage regions, concealed-damage flags and a repair scope. Every measurement carries a 95% range.

| Document | What it is |
|---|---|
| `TECHNICAL_REPORT.md` | Architecture, tiers, drift, error budget, calibration, fix loop, failure modes |
| `BENCHMARK.md` | Every measured number, gates, repeatability, drift ablation, timing |
| `CAPTURE_PROTOCOL.md` | How to capture each tier (Route 2: stock apps) and the device matrix |
| `COMPLIANCE.md` | Brief requirement → file → artifact → status |
| `fixloop/` | Fix declaration, before/after results, post-mortem |

## Setup (about 10 minutes on a clean machine)

Needs Python 3.10-3.12, git and curl. Runs on CPU; uses a GPU when one is present.

```
git clone https://github.com/goswamivaishnavi45/floor-plan.git
cd floor-plan
python -m venv .venv
.venv\Scripts\activate             # Windows; macOS/Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/fetch_models.py --lidar   # LiDAR tier only: ~0.6 GB (damage detector)
python scripts/fetch_models.py           # all tiers: ~5.8 GB (adds VGGT and Depth Anything)
```

## Run on a capture (one command)

```
python run.py <capture>
```

The tier is detected from what `<capture>` is:

| Input | Tier | Example |
|---|---|---|
| Stray Scanner export folder (`depth/`, `odometry.csv`, `rgb.mp4`) | LiDAR | `python run.py data/c7d28f72c6` |
| Video file, or folder holding one | video | `python run.py data/c00a170fe1/rgb.mp4 --rotate cw` |
| Folder of room folders with photos | photo | `python run.py data/photos_c7d28f72c6` |

Options: `--no-damage` skips damage detection (faster); `--no-drift-correction` uses LiDAR poses as recorded; `--rotate cw|ccw|180` turns sideways video upright (Stray Scanner's `rgb.mp4`; iPhone Camera videos are already upright).

Results in `outputs/<capture>/`:

- `result.json`: rooms (outline, walls, width, length, floor area, ceiling height), openings, adjacency, total area, damage, concealed-damage flags, scope, warnings. Each measurement is `{"value", "pm95"}`. The file is validated against `schema/result.schema.json` before it is written.
- `plan.png`: the floor plan (schematic for the photo tier, one panel per segment for video).
- Debug: `pointcloud.ply`, `topdown.png`, `walls_straight.png`, `walls_found.png`, `rooms.png`, `layout.json`, `drift.json`.

## Data

The three sample captures (Stray Scanner, ~1.5 GB) are not in git. Put the folders in `data/` (see `data/README.md`):

```
data/c00a170fe1/   data/1a8384c3f6/   data/c7d28f72c6/
```

## Regenerating every reported number

All commands run from the repository root after setup.

| Numbers | Command |
|---|---|
| LiDAR results per capture | `python run.py data/<capture>` |
| Repeatability (`BENCHMARK.md`, `benchmark/`) | `python run.py data/1a8384c3f6`, `python run.py data/c7d28f72c6`, then `python scripts/repeatability.py outputs/1a8384c3f6 outputs/c7d28f72c6` |
| Fix loop before / after | the same three commands at commits `59656e1` (before) and `6cab0ca` (after); see `fixloop/declaration.md` |
| Drift ablation | `python scripts/drift_ablation.py data/1a8384c3f6` and `data/c7d28f72c6` |
| AI depth vs LiDAR | `python scripts/compare_depth.py data/c00a170fe1 --frames 30` |
| COLMAP path vs ARKit | `python scripts/compare_path.py data/c00a170fe1 --frames 150` |
| Video pieces: size and up | `python scripts/check_pieces.py data/c00a170fe1 --frames 300` |
| VGGT checks | `python scripts/check_vggt.py data/c00a170fe1 --frames 8 --start 270 --end 560 --size 350` |
| Photo test folders | `python scripts/make_photo_folders.py c7d28f72c6` (after `run.py` on that capture) |
| Photo room checks | `python scripts/check_photo_room.py data/photos_c7d28f72c6 R3 --size 518` |
| Video tier | `python run.py data/c00a170fe1/rgb.mp4 --rotate cw` |
| Photo tier | `python run.py data/photos_c7d28f72c6` |
| Damage detector on photos | `python scripts/check_damage_photos.py <folder of photos>` |
| Fix-loop code change | `fixloop/fix.diff` (or `git diff 32c0ecf 6cab0ca -- src/layout.py`) |

COLMAP runs with a fixed random seed, so the video tier gives the same pieces on a re-run.

## Code map

| File | Role |
|---|---|
| `run.py` | One command: tier detection, pipeline, result, plan |
| `src/capture.py` | Reads a Stray Scanner export (per-frame poses and intrinsics) |
| `src/pointcloud.py` | Depth frames → filtered, fused point cloud; top views; PLY |
| `src/drift.py` | Chunked fusion with floor-anchored drift correction |
| `src/layout.py` | Straighten, walls, rooms, heights, sizes with ranges |
| `src/result.py` | `result.json` in the published schema, validation |
| `src/plan.py` | `plan.png` |
| `src/video.py` | Video tier: frames, COLMAP pieces, AI-depth size, up direction |
| `src/photo.py` | Photo tier: VGGT per room, quality gate, doorway links |
| `src/mono_depth.py` | Depth Anything V2 wrapper |
| `src/damage.py` | OWLv2 damage detection, surfaces, rules, scope |
| `src/viewer.py` | `python -m src.viewer outputs/<capture>/pointcloud.ply` → browser 3D view |
