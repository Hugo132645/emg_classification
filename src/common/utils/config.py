from __future__ import annotations
import yaml

def load_yaml(path: str) -> dict:
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data or {}