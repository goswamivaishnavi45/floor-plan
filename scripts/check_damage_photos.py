"""Run the damage detector on a folder of photos and draw what it finds.

    python scripts/check_damage_photos.py data/damage_photos

For each photo (JPEG, PNG or HEIC), prints every detection with a score of
at least 0.10 and saves the photo with boxes to outputs/damage_check/:
red boxes are reported (score >= SCORE_THRESHOLD, 0.30), yellow ones are
below the threshold. Used for the real-damage check in BENCHMARK.md (three
Samsung A35 photos of damaged walls; the photos are not in git).
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.damage import SCORE_THRESHOLD, DamageDetector  # noqa: E402
from src.photo import load_photos  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("--out", default="outputs/damage_check")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detector = DamageDetector()
    for photo in load_photos(args.folder):
        found = detector.detect(photo["bgr"], threshold=0.10)
        print(f"{photo['name']}: EXIF 35 mm focal {photo['f35']}")
        image = photo["bgr"].copy()
        for f in found:
            reported = f["score"] >= SCORE_THRESHOLD
            print(f"   {'REPORTED' if reported else 'below   '} {f['class']:<14} {f['score']:.2f} "
                  f"box {[int(v) for v in f['box']]}")
            colour = (0, 0, 255) if reported else (0, 200, 255)
            x0, y0, x1, y1 = (int(v) for v in f["box"])
            cv2.rectangle(image, (x0, y0), (x1, y1), colour, 8)
            cv2.putText(image, f"{f['class']} {f['score']:.2f}", (x0 + 10, y0 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, colour, 6)
        h, w = image.shape[:2]
        cv2.imwrite(str(out / f"{Path(photo['name']).stem}.jpg"), cv2.resize(image, (int(w * 720 / h), 720)))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
