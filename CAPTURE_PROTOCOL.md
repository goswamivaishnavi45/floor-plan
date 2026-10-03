# Capture protocol (Route 2: stock apps)

One page per person capturing a home. Follow it literally.

## Before you start (all tiers)

- Turn on every light and open every internal door fully.
- Clear the floor of things you can move; furniture can stay.
- Note the home's name; it becomes the folder name, e.g. `flat12`.

## Tier 3: LiDAR (iPhone 12 Pro or newer Pro / Pro Max)

**Install:** **Stray Scanner** (free, App Store, by Kenneth Blomqvist). No account, no settings to change.

**Walk:**
1. Open Stray Scanner, tap **Record**. Stand in the entrance, phone **upright** at chest height.
2. Walk **slowly** (half a normal pace) into each room in turn. In every room:
   - point the phone at the floor-to-wall line and walk once along every wall, 1-2 m away;
   - then **tilt up and sweep the whole ceiling** once (ceiling height needs it);
   - pause 2 seconds in each doorway, pointing through it at the next room.
3. Visit every room once, return to the entrance, tap **Stop**.
4. Aim for **1-4 minutes**. Over 5 minutes: split the home into two recordings.

**Avoid:** turning fast; pointing at mirrors, windows or glass doors for more than a second; walking in the dark; covering the camera with fingers.

**Hand over:** in Stray Scanner, open the recording → **Export** → AirDrop or Files → copy the **whole folder** (it contains `rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`, `imu.csv`) to the laptop as `data/flat12/`. Then run:

```
python run.py data/flat12
```

## Tier 2: video (any iPhone 15 or newer)

**Install:** nothing. Use the **Camera** app, **Video**, default settings.

**Walk:** same route and pace as the LiDAR walk above, including the ceiling sweep in every room. Keep the phone upright; do not zoom.

**Hand over:** AirDrop or cable the `.MOV` file to the laptop as `data/flat12_video/walk.MOV`. Then `python run.py data/flat12_video`.

## Tier 1: photos (any iPhone 15 or newer)

**Install:** nothing. Use the **Camera** app, **Photo**, 1x lens (not 0.5x), default settings.

**Per room, 4-8 photos that overlap:**
1. Stand in a corner. Take a photo, then **turn a little** so the next photo shows about **half of the previous one** again. Continue round the room.
2. Move to the opposite corner and repeat for the walls you have not covered.
3. Every photo should show some floor-to-wall line; no close-ups of a blank wall.

**Doorways:** standing in each doorway, take **one photo looking into the next room**. Put a copy of that photo in **both** rooms' folders (this is how rooms are linked).

**Hand over:** one folder per room, named after the room, inside one folder for the home:

```
data/flat12_photos/
  kitchen/   IMG_0101.JPG ... IMG_0107.JPG  door_kitchen_hall.JPG
  hall/      IMG_0110.JPG ... IMG_0115.JPG  door_kitchen_hall.JPG
```

Copy with a cable, AirDrop or iCloud **as originals** (not WhatsApp, which strips the focal length). Then `python run.py data/flat12_photos`.

## Device matrix

| Tier | Runs on | Capture app | What it delivers on our benchmark (honest) |
|---|---|---|---|
| LiDAR | iPhone 12 Pro and newer Pro / Pro Max (LiDAR sensor) | Stray Scanner | Stitched plan with rooms, walls, ceiling heights, doorways. Wall detection repeats between two captures; room dimensions repeat within 1 cm / 0.5% for 2 of 14 dimensions (14%), because the two captures still divide some rooms differently. Ranges assume 5 mm per surface; not checked against tape. |
| Video | Any iPhone 15 or newer (any phone camera works the same way) | Camera app | Runs end to end. Classical reconstruction breaks into pieces on plain walls; on our video no complete room was found. Sizes would carry about +-20% (95%). |
| Photo | Any iPhone 15 or newer (any phone camera; focal length read from EXIF) | Camera app | Runs end to end; links rooms through shared doorway photos (4 of 4 on our test set). Room sizes are reported only if a quality check passes; on our test photos (cut from video) none passed. |
| All | — | — | Damage detection (OWLv2) on every tier; sizes in metres only on the LiDAR tier. |
