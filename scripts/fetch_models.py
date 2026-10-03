"""Download the pretrained model weights the pipeline uses into models/.

    python scripts/fetch_models.py            # all models (~5.8 GB)
    python scripts/fetch_models.py --lidar    # only what the LiDAR tier needs (~0.6 GB)

Weights are not stored in git. Downloads use plain HTTPS with resume and
retries, because large Hugging Face downloads stalled on our connection.
Models (disclosed as required by the brief):

  Depth Anything V2, Metric Indoor, Small (Yang et al., 2024), Apache-2.0
    depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf, ~95 MB
    metric depth from one picture: overall size for the video and photo tiers
  VGGT-1B (Wang et al., CVPR 2025), CC-BY-NC-4.0 (non-commercial)
    facebook/VGGT-1B, ~5 GB
    cameras and depth for several pictures at once: video and photo tiers
  OWLv2 base, patch16 ensemble (Minderer et al., 2023), Apache-2.0
    google/owlv2-base-patch16-ensemble, ~620 MB
    open-vocabulary detection: damage, all tiers
"""

import argparse
import subprocess
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
HUB = "https://huggingface.co/{repo}/resolve/main/{file}"

DOWNLOADS = {
    "depth-anything-v2-metric-indoor-small": (
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        ["config.json", "preprocessor_config.json", "model.safetensors"], False),
    "owlv2-base": (
        "google/owlv2-base-patch16-ensemble",
        ["added_tokens.json", "config.json", "merges.txt", "preprocessor_config.json",
         "special_tokens_map.json", "tokenizer_config.json", "vocab.json", "model.safetensors"], True),
    "vggt-1b": ("facebook/VGGT-1B", ["config.json", "model.safetensors"], False),
}


def remote_size(url):
    """Size in bytes reported by the server (after redirects), or None."""
    head = subprocess.run(["curl", "-sSIL", url], capture_output=True, text=True).stdout
    sizes = [int(line.split(":")[1]) for line in head.splitlines() if line.lower().startswith("content-length:")]
    return sizes[-1] if sizes else None


def fetch(repo, name, target):
    """Download one file with curl, resuming after dropped connections."""
    url = HUB.format(repo=repo, file=name)
    size = remote_size(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(30):
        if target.exists() and size is not None and target.stat().st_size == size:
            return
        subprocess.run(["curl", "-sS", "-L", "-C", "-", "-o", str(target), url])
        if size is None:
            return
        print(f"  {name}: {target.stat().st_size if target.exists() else 0} of {size} bytes, resuming")
    sys.exit(f"could not download {repo}/{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lidar", action="store_true", help="only the models the LiDAR tier needs")
    args = parser.parse_args()
    for folder, (repo, files, lidar) in DOWNLOADS.items():
        if args.lidar and not lidar:
            continue
        for name in files:
            target = MODELS / folder / name
            print(f"{repo}/{name}")
            fetch(repo, name, target)
    print(f"models in {MODELS}")


if __name__ == "__main__":
    main()
