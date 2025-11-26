# Imports

from __future__ import annotations
from typing import Literal, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from ..features import time_domain as td
from ..features import freq_domain as fd

# Choose between time features, freq features and both

Mode = Literal["time", "freq", "time+freq"]

# Features extraction

def _extract_features(windows: np.ndarray,  fs: float, mode: Mode) -> np.ndarray:
    if mode == "time":
        X = td.extract_td_features_per_window(windows)
    elif mode == "freq":
        X = fd.extract_fd_features_per_window(windows, fs)
    elif mode == "time+freq":
        X_time = td.extract_td_features_per_window(windows)
        X_freq = fd.extract_fd_features_per_window(windows, fs)
        X = np.hstack([X_time, X_freq])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return X

# Data set builder

def build_classic_ml_dataset(windows: np.ndarray, window_labels, fs: float, mode: Mode = "time+freq", scale: bool = True) -> Tuple[np.ndarray, np.ndarray, StandardScaler | None]:
    win_arr = np.asarray(windows)
    labels_arr = np.asarray(window_labels, dtype=object)
    # Drop windows with no label (None)
    mask = labels_arr != None
    if not np.any(mask):
        raise ValueError("No labeled windows found.")
    win_arr = win_arr[mask]
    y = labels_arr[mask]
    # Extract features
    X = _extract_features(win_arr, fs=fs, mode=mode)
    # Optionally scale
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None
    return X, y, scaler

# Test

if __name__ == "__main__":
    fs = 1000.0
    N, W = 10, 200
    t = np.linspace(0, 0.2, W, endpoint=False)
    wins = []
    labels = []
    for i in range(N):
        x = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.randn(W)
        wins.append(x.astype(np.float32))
        labels.append("fist" if i % 2 == 0 else "rest")
    wins = np.stack(wins, axis=0)
    X, y, scaler = build_classic_ml_dataset(
        windows=wins,
        window_labels=labels,
        fs=fs,
        mode="time+freq",
        scale=True,
    )
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("First row of X:", X[0])
    print("First label:", y[0])