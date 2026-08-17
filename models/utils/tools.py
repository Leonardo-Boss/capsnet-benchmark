from pathlib import Path
import re

import yaml


def read_yaml(fname: str | Path) -> dict:
    """Reads a YAML file and returns the data as dictionary."""
    fname = Path(fname)
    with fname.open("r", encoding="utf8") as file:
        return yaml.safe_load(file)

def write_yaml(data: dict, fname: str | Path) -> None:
    """Writes a dictionary to a YAML file."""
    fname = Path(fname)
    with fname.open("w", encoding="utf8") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)

def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the path to the highest-epoch checkpoint in `checkpoint_dir`.

    Looks for files named `ep<N>.pth` (as written by BaseTrainer) and
    returns the one with the largest N.

    Args:
        checkpoint_dir (str | Path): Directory to search.

    Returns:
        Path | None: Path to the latest checkpoint, or None if the
            directory doesn't exist or contains no matching checkpoints.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    candidates = []
    for f in checkpoint_dir.glob("ep*.pth"):
        match = re.fullmatch(r"ep(\d+)\.pth", f.name)
        if match:
            candidates.append((int(match.group(1)), f))

    if not candidates:
        return None

    return max(candidates, key=lambda pair: pair[0])[1]
