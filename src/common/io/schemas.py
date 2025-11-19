# Imports
from __future__ import annotations # -> definying the instance of the output
from types import SimpleNamespace # access dictionary keys with dot notation like cfg.sample_rate_hz instead of cfg["sample_rate_hz"]
from time import time
import os
from ..utils.config import load_preprocessing_cfg

# Default configuration values
defaults = {
    "sample_rate_hz": 1000,
    "num_channels": 3,

    "bandpass_hz": None,
    "notch_hz": None,

    "window_ms": 200,
    "hop_ms": 100,

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
def load_cfg(path: str = "configs/preprocessing.yaml") -> SimpleNamespace:
    try:
        user = load_preprocessing_cfg(path)
        if not isinstance(user, dict):
            raise TypeError("YAML file is not a dictionary.")
    except FileNotFoundError:
        user = {}
    cfg = defaults.copy()
    for key, val in user.items():
            cfg[key] = val
    cfg = SimpleNamespace(**cfg)
    return cfg

# Helper functions
def ms_to_samples(ms: int, fs: int) -> int: #Transform ms to samples
    return int(round(ms * fs / 1000))

def samples_to_ms(n: int, fs: int) -> int: #Transform samples to ms
    return int(round(n * 1000 / fs))

def window_len_samples(cfg) -> int: #Transform window length to samples
    return ms_to_samples(cfg.window_size_ms, cfg.sample_rate_hz)

def hop_len_samples(cfg) -> int: #Transform hop length to samples
    return ms_to_samples(cfg.hop_size_ms, cfg.sample_rate_hz)

def id_map(cfg) -> dict: #Returns the IDs of the gestures
    return cfg.label_map

def gesture_to_id(cfg, gesture:str ) -> int: #Returns ID of given gesture
    mapping = id_map(cfg)
    if gesture not in mapping:
        raise KeyError(f"'{gesture}' not found.")
    return mapping[gesture]

def inverse_id_map(cfg) -> dict: #Inverse of the previous function
    m = cfg.label_map
    return {v: k for k, v in m.items()}

def id_to_gesture(cfg, gesture_id: int) -> str: #Returns gesture of given ID
    mapping =inverse_id_map(cfg)
    if gesture_id not in mapping:
        raise KeyError(f"'{gesture_id}' not found.")
    return mapping[gesture_id]

def expand_template(template: str, **kwargs) -> str: #Simpler way to create directory paths
    return template.format(**kwargs)

def ensure_parent_dir(path: str) -> None: #Checks if a path exists and if not, creates it
    os.makedirs(os.path.dirname(path), exist_ok=True)

def sanity_check_cfg(cfg) -> None: #Verifies if there is anything wrong in the cfg
    if cfg.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0.")
    if cfg.window_size_ms <= 0 or cfg.hop_size_ms <= 0:
        raise ValueError("window_size_ms and hop_size_ms must be > 0.")
    if not hasattr(cfg, "gestures") or not hasattr(cfg, "label_map"):
        raise ValueError("gestures and label_map must be defined in YAML.")
    if set(cfg.gestures) != set(cfg.label_map.keys()):
        raise ValueError("label_map keys must match gestures exactly.")

def is_raw_mode(cfg) -> bool: #Checks if data is raw
    if cfg.bandpass_hz is not None and not isinstance(cfg.bandpass_hz, (tuple)):
        raise ValueError("bandpass_hz must be a tuple or None.")
    if cfg.notch_hz is not None and not isinstance(cfg.notch_hz, (int, float)):
        raise ValueError("notch_hz must be a number or None.")
    return cfg.bandpass_hz is not None or cfg.notch_hz is not None

# Test
if __name__ == "__main__":
    cfg = load_cfg()
    sanity_check_cfg(cfg)
    print("Sanity check: OK")
    print("is_raw_mode:", is_raw_mode(cfg))
    print("sample_rate:", cfg.sample_rate_hz)
    print("window/hop (ms):", cfg.window_size_ms, cfg.hop_size_ms)
    print("window/hop (samples):",
          ms_to_samples(cfg.window_size_ms, cfg.sample_rate_hz),
          ms_to_samples(cfg.hop_size_ms, cfg.sample_rate_hz))
    path = expand_template(cfg.features_file_template,
                           subject_id="S01", session_date="2025-11-06",
                           session_id=1, timestamp=int(time()))
    ensure_parent_dir(path)
    print("Expanded features path:", path)
    print("All main helpers work correctly.")