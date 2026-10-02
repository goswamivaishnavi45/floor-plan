# Floor plan and damage pipeline

Turns a phone capture of a home (photos, video, or LiDAR) into a dimensioned floor plan, a damage report, and a repair scope. Every measurement comes with a confidence interval.

Work in progress.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Put the capture folders in `data/` (see `data/README.md`).

## Point cloud from a LiDAR capture

```
python -m src.pointcloud data/c00a170fe1
```

Writes `outputs/<capture>/pointcloud.ply` (open in MeshLab or CloudCompare), `topdown.png` and `walls.png`.
