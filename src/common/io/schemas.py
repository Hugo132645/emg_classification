# Imports
from __future__ import annotations # -> definying the instance of the output
from types import SimpleNamespace # access dictionary keys with dot notation like cfg.sample_rate_hz instead of cfg["sample_rate_hz"]
import os
from src.common.utils.config import load_preprocessing_cfg

# Default configuration values
defaults = {
    "sample_rate_hz": 1000,
    "num_channels": 3,

    "bandpass_hz": [20.0, 450.0],
    "notch_hz": 50.0,

    "window_size_ms": 200,
    "hop_size_ms": 100,

    "gestures": ["rest", "fist", "open", "pinch"],
    "label_map": {"rest": 0, "fist": 1, "open": 2, "pinch": 3},

    "protocol": {
        "rest_duration_s": 5,
        "cue_duration_s": 0,
        "hold_duration_s": 3,
        "relax_duration_s": 2,
        "repetitions_per_gesture": 10,
        "sessions_per_day": 2,
        "days": 3,
    },

    "raw_file_template": "data/raw/{subject_id}/{session_date}/session_{session_id}.parquet",
    "features_file_template": "data/processed/{subject_id}/{session_date}/features_session_{session_id}_{timestamp}.parquet",
    "model_file_template": "exports/{track}/{subject_id}/{session_date}/model_session_{session_id}_{timestamp}.{model_artifact_ext}",
    "mirror_csv": True,
}


# Load YAML and merge it with defaults
def load_cfg(path: str = "configs/preprocessing.yaml") -> dict:
    try:
        user = load_preprocessing_cfg(path)
        if not isinstance(user, dict):
            raise TypeError("YAML file is not a dictionary.")
    except FileNotFoundError:
        user = {}
    cfg = defaults.copy()
    for key, val in user.items():
        if key in cfg:
            cfg[key] = val
    cfg = SimpleNamespace(**cfg)
    return cfg

# Helper functions
def ms_to_samples(ms: int, fs: int) -> int:
    return int(round(ms * fs / 1000))

def samples_to_ms(n: int, fs: int) -> int:
    return int(round(n * 1000 / fs))

def expand_template(template: str, **kwargs) -> str:
    return template.format(**kwargs)

def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Tests
if __name__ == "__main__":
    cfg = load_cfg()
    for k, v in cfg.items():
        print(f"{k}: {v}")