from __future__ import annotations
import yaml

def load_preprocessing_cfg(path: str) -> dict:
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data or {}