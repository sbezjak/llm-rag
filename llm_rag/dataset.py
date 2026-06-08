from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict | list:
    with open(path) as f:
        return yaml.safe_load(f)
