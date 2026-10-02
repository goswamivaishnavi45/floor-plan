# Floor plan and damage pipeline

Turns a phone capture of a home (photos, video, or LiDAR) into a dimensioned floor plan, a damage report, and a repair scope. Every measurement comes with a 95% range.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run on a capture (one command)

```
python run.py data/c7d28f72c6
```

The tier is detected from the folder contents (currently LiDAR / Stray Scanner exports; video and photo tiers are in progress). Results go to `outputs/<capture>/`:

- `result.json`: rooms, walls, openings, adjacency and total area, each measurement as `{"value", "pm95"}`, in the format published in `schema/result.schema.json` (the output is validated against it before it is written)
- `plan.png`: the floor plan

Debug outputs in the same folder: `pointcloud.ply`, `topdown.png`, `walls.png`, `walls_straight.png`, `walls_found.png`, `rooms.png`, `layout.json`.

## Individual stages

### Point cloud from a LiDAR capture

```
python -m src.pointcloud data/c00a170fe1
```

Writes `outputs/<capture>/pointcloud.ply` (open in MeshLab or CloudCompare), `topdown.png` and `walls.png`.

### Look at a point cloud in 3D

```
python -m src.viewer outputs/c00a170fe1/pointcloud.ply
```

Writes `outputs/c00a170fe1/viewer.html`. Open it in a browser: drag to rotate, scroll to zoom, and use the slider to cut away the ceiling.

### Room layout

```
python -m src.layout outputs/c00a170fe1
```

Straightens the cloud so walls run along the x and z axes (`walls_straight.png`) and records the angle in `layout.json`.
