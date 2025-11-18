import numpy as np
from typing import List, Tuple

EPS = 1e-7

def _basic_stats_(windows: np.ndarray) -> Tuple[np.ndarray, list[str]]:
    mean = windows.mean(axis=1)
    std = windows.std(axis=1) + EPS
    min_ = windows.min(axis=1)
    max_ = windows.max(axis=1)
    range_ = max_ - min_
    w0 = windows - windows.mean(axis=1, keepdims=True)
    zero_crossings = (w0[:, 1:] * w0[:,-1] < 0 ).mean(axis=1)
    feat = np.stack([mean, std, min_, max_, range_, zero_crossings], axis=1)
    names = ['mean', 'std', 'min', 'max', 'range', 'zero_crossing_rate']
    return feat, names

def _shape_feat_(windows: np.ndarray, fs: float) -> Tuple[np.ndarray, list[str]]:
    time_st = windows.shape[1]

    #Least Mean-Squares
    t = (np.arange(time_st) / max(1.0, fs))[None, :]
    t = (t-t.mean() / (t.std() + EPS))
    x = windows - windows.mean(axis=1, keepdims=True)
    slope = (x*t).sum(axis=1) / ((t**2).sum(axis=1) + EPS)
    s = x.std(axis=1) + EPS

    #Moment skew and Excess kurtosis
    skew = np.mean((x/s[:, None])**3,axis=1)
    kurt = np.mean((x/s[:, None])**4, axis=1) - 3

    feat = np.stack([slope, skew, kurt], axis=1)
    names = ['slope', 'moment_skew', 'kurt']
    return feat, names

def compute_seq_features(window: np.ndarray, include_feat1: bool = False, include_feat2: bool = False) -> Tuple[np.darray, list[str]]:
    return None







