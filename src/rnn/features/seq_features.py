import numpy as np
from typing import List, Tuple

EPS = 1e-7

def _basic_stats_(windows: np.ndarray) -> Tuple[np.ndarray, list[str]]:
    mean = windows.mean(axis=1)
    std = windows.std(axis=1) + EPS
    min_ = windows.min(axis=1)
    max_ = windows.max(axis=1)
    range_ = max_ - min_
    feat = np.stack([mean, std, min_, max_, range_], axis=1)
    names = ['mean', 'std', 'min', 'max', 'range']
    return feat, names

def _shape_feat_(windows: np.ndarray) -> Tuple[np.ndarray, list[str]]:
    return None

def compute_seq_features(window: np.ndarray, include_feat1: bool = False, include_feat2: bool = False) -> Tuple[np.darray, list[str]]:
    return None
