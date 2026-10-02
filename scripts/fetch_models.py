"""Download the pretrained model weights the pipeline uses.

    python scripts/fetch_models.py

Weights are not stored in git; this script fetches them into models/.
Model used (disclosed as required by the brief):
    Depth Anything V2, Metric Indoor, Small (Yang et al., 2024), Apache-2.0,
    https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
    Monocular metric depth for indoor scenes, trained on Hypersim; ~25M parameters.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = Path(__file__).resolve().parent.parent / "models"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def main():
    target = MODELS / "depth-anything-v2-metric-indoor-small"
    snapshot_download(DEPTH_MODEL, local_dir=target)
    print(f"depth model -> {target}")


if __name__ == "__main__":
    main()
