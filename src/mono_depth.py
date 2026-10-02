"""Metric depth from a single colour image with Depth Anything V2 (Metric
Indoor, Small). Fetch the weights first with `python scripts/fetch_models.py`.

The model guesses how far each pixel is, in metres, from one picture, the way
a person with one eye closed still judges distances from experience. It is a
guess: overall scale can be off by several percent, which the video tier
corrects and reports (see scripts/compare_depth.py).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "depth-anything-v2-metric-indoor-small"


class DepthModel:
    def __init__(self, model_dir=MODEL_DIR):
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"{model_dir} not found; run `python scripts/fetch_models.py` first")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_dir).to(self.device).eval()

    @torch.no_grad()
    def predict(self, bgr, size=None):
        """Depth in metres for a BGR image, resized to `size` (w, h) or the
        image's own size."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        depth = self.model(**inputs).predicted_depth[0].cpu().numpy()
        w, h = size or (bgr.shape[1], bgr.shape[0])
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
