"""Video tier: frames from a handheld walkthrough video.

Part 6a: pick frames. A walkthrough has tens of frames per second, almost all
repeating their neighbours, and some blurred by hand motion. We cut the video
into equal time windows and keep the sharpest frame of each window.
Sharpness = variance of the Laplacian (how much the image changes from pixel
to pixel); a blurred frame has soft edges and a low score.
"""

import cv2
import numpy as np


def sharpness(bgr):
    gray = cv2.cvtColor(cv2.resize(bgr, (480, 360)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_frames(video_path, count):
    """Return [(frame index, BGR image)] with one sharp frame per window.

    Reads the video once from start to end (seeking is unreliable, see
    src/capture.py) and keeps only the current best frame of each window
    in memory.
    """
    video = cv2.VideoCapture(str(video_path))
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    windows = np.linspace(0, total, count + 1).astype(int)
    picked, best, window = [], None, 0
    index = 0
    while video.grab():
        while window < count and index >= windows[window + 1]:
            if best is not None:
                picked.append(best[1:])
            best, window = None, window + 1
        if window >= count:
            break
        _, frame = video.retrieve()
        score = sharpness(frame)
        if best is None or score > best[0]:
            best = (score, index, frame)
        index += 1
    if best is not None and window < count:
        picked.append(best[1:])
    video.release()
    return picked
