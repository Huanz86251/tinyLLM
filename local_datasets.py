"""Local dataset access; missing inputs fail explicitly instead of downloading."""
import json
from datasets import Dataset
from project_paths import path

def load_local_dataset(name, split, config=None):
    if split not in {"train", "test", "validation"}:
        raise ValueError(f"Unsupported split: {split}")
    if name == "gsm8k":
        folder = "gsm8k"
    elif name == "arc":
        folder = {"ARC-Easy": "arc_easy", "ARC-Challenge": "arc_challenge"}.get(config)
        if folder is None:
            raise ValueError(f"Unknown ARC config: {config}")
    else:
        raise ValueError(f"Unknown dataset: {name}")
    source = path(f"datasets/{folder}/{split}.jsonl")
    if not source.is_file():
        raise FileNotFoundError(f"Dataset is not present locally: {source}. No network fallback.")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    return Dataset.from_list(rows)
