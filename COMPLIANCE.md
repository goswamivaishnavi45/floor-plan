# Compliance matrix

Requirement (from the brief) → where it lives → what to look at → status.
Status: **done**, **partial** (exists but falls short, with the reason), **missing**.

## Part 1: capture

| Requirement | File | Artifact | Status |
|---|---|---|---|
| Capture route (Route 2: stock app + one-page protocol) | `CAPTURE_PROTOCOL.md` | Stray Scanner (LiDAR), Camera app (video, photo); walk, avoid, hand-over steps | done |
| Three input tiers, same output contract | `run.py` (`detect_tier`) | One command takes a LiDAR folder, a video file or folder, or a folder of room photo folders | done |
| Photo tier: per-room folders produce a stitched plan | `src/photo.py` | Rooms linked through shared doorway photos; sizes only past a quality gate; plan is a schematic | partial: no room passed the gate on our test photos, so no dimensioned stitched plan |
| Video tier | `src/video.py` | COLMAP pieces, AI-depth size, rooms per segment | partial: segments are not stitched; no complete room on our video |
| LiDAR tier | `src/capture.py`, `src/pointcloud.py`, `src/layout.py` | Fused cloud, walls, rooms, sizes | done |
| Intervals widen as sensor data thins | `src/result.py` (`widen`), `run.py` | LiDAR: sensor + scatter + drift; video: +10% size sigma; photo: +15% or not measured | done |
| Device matrix | `CAPTURE_PROTOCOL.md` | Tier → device → what it delivers | done |

## Part 2: output contract and gates

| Requirement | File | Artifact | Status |
|---|---|---|---|
| Dimensioned per-room plan: walls, ceiling height, floor area, openings | `src/layout.py`, `src/result.py`, `src/plan.py` | `outputs/<capture>/result.json`, `plan.png` | done (LiDAR) |
| Stitched multi-room plan with adjacency | `src/layout.py` (`find_rooms`), `src/result.py` | `rooms`, `openings`, `adjacency` in `result.json` | done (LiDAR); partial (photo: adjacency only) |
| Per-surface damage regions with class and metric extent | `src/damage.py` | `damage` in `result.json` (surface id, class, width/height/area +- range) | partial: metric extent on LiDAR only; not validated on staged damage |
| Concealed-damage flags with the rule that fired | `src/damage.py` (`RULES`) | `concealed_damage_flags` (rule id, condition, flag) | partial: rules implemented, none fired on our data |
| Scope line items keyed to surfaces | `src/damage.py` (`SCOPE`) | `scope` (surface id, action, quantity, unit) | partial: as above |
| Confidence interval on every measurement | `src/result.py` | every measure is `{"value", "pm95"}` (95% range) | done; not calibrated against ground truth |
| One command per capture | `run.py` | `python run.py <capture>` | done |
| JSON to a published schema | `schema/result.schema.json`, `src/result.py` (`validate`) | output validated before it is written | done |
| Rendered plan | `src/plan.py` | `plan.png` (schematic for the photo tier, per-segment panels for video) | done |
| Benchmark set (multi-room, staged damage, all tiers, repeat capture, ground truth) | `BENCHMARK.md` | what exists and what is missing | partial: no ground truth, no staged damage, no native video/photo captures |
| Gates: openings, ceiling, repeatability, drift, photo stitch | `BENCHMARK.md` | gates table | partial: repeatability fails, drift done, others not measurable without ground truth |
| Drift accountability with on/off ablation | `src/drift.py`, `scripts/drift_ablation.py` | `BENCHMARK.md`, `outputs/<capture>_drift_ablation.png` | done |

## Parts 3-5

| Requirement | File | Artifact | Status |
|---|---|---|---|
| Head-to-head vs a consumer app on 2 rooms | `BENCHMARK.md` | | missing: needs the apartment and a LiDAR iPhone |
| Fix loop: declaration, before, after, diff | `fixloop/` | `declaration.md` (before the fix), `before/`, `after/`, `fix.diff` (the code change), `postmortem.md` (with the follow-up) | done (gate not passed; prediction missed, post-mortem explains) |
| Process evidence: commit as you work | git history | 35+ commits over two days, each with its reasoning and measured results | done |

## Deliverables

| Deliverable | File | Status |
|---|---|---|
| 1. Compliance matrix | `COMPLIANCE.md` | done |
| 2. Capture route + device matrix | `CAPTURE_PROTOCOL.md` | done |
| 3. Repo, README, one command per capture | `README.md`, `run.py` | done |
| 4. Reproduction bundle | `README.md` (setup, `scripts/fetch_models.py`, commands per number) | done; raw captures fetched separately (too large for git) |
| 5. Benchmark report | `BENCHMARK.md`, `benchmark/` | partial (see gaps) |
| 6. Fix loop bundle | `fixloop/` | done |
| 7. Technical report (max 6 pages) | `TECHNICAL_REPORT.md` | done |
| 8. Raw benchmark data | provided sample captures; app exports and ground truth do not exist | partial |

## Constraints

| Constraint | Where | Status |
|---|---|---|
| Handheld consumer capture only | `CAPTURE_PROTOCOL.md` | done |
| Pretrained models disclosed | `scripts/fetch_models.py`, `TECHNICAL_REPORT.md` | done (Depth Anything V2, VGGT-1B (non-commercial licence), OWLv2) |
| Runs without our infrastructure | all | done: runs offline on a laptop CPU after `fetch_models.py` |
| Weights fetched by script | `scripts/fetch_models.py` | done |
| Mirrors, glass, wet-look surfaces, low light covered | `src/pointcloud.py` (`valid_mask`), `TECHNICAL_REPORT.md` section 7 (items 2-4), `CAPTURE_PROTOCOL.md` (lights on, avoid mirrors and glass) | partial: low-confidence LiDAR returns dropped; glass walls go missing; glossy surfaces broke video tracking; LiDAR works in low light, the other tiers do not |
