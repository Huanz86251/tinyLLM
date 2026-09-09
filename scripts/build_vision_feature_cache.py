"""Build the InternViT feature cache for the original VLM stage."""

from _run_module import run


if __name__ == "__main__":
    run("train/vlm/cache_vision_features.py")

