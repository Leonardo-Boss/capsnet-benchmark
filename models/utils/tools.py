from pathlib import Path

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
