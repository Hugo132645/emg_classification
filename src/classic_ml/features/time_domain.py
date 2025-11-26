# Imports

from __future__ import annotations
import numpy as np

# Constants

Delta = 1e-6    #used to avoid counting noise
EPS   = 1e-12   # to avoid divide-by-zero

# If you want Hjorth/AR in the feature vector, set these True

use_hjorth = True
use_ar4 = True   # requires W >= 4

# Time features

def _mav(x: np.ndarray) -> float:   #Mean absolute value
    return float(np.mean(np.abs(x)))

def _rms(x: np.ndarray) -> float:   #Root mean square
    return float(np.sqrt(np.mean(x * x)))

def _var(x: np.ndarray) -> float: #Variance
    return float(np.var(x))

def _sd(x: np.ndarray) -> float:    #Standard deviation
    return float(np.std(x))

def _wl(x: np.ndarray) -> float:    #Wave length
    dx = np.diff(x, axis=-1)
    return float(np.sum(np.abs(dx)))

def _zc(x: np.ndarray, delta: float = Delta) -> int:    #Zero count
    # Count sign changes with minimum jump delta
    x1 = x[..., :-1]
    x2 = x[..., 1:]
    cond = (np.signbit(x1) != np.signbit(x2)) & (np.abs(x2 - x1) > delta)
    return int(np.sum(cond))

def _ssc(x: np.ndarray, delta: float = Delta) -> int:   #Slope sign  change
    d1 = np.diff(x, n=1)
    # sign change between d1[i] and d1[i+1] is (d1[:-1] * d1[1:]) < 0
    cond_mag  = (np.abs(d1[:-1]) > delta) & (np.abs(d1[1:]) > delta)
    cond_sign = (d1[:-1] * d1[1:]) < 0
    return int(np.sum(cond_mag & cond_sign))

def _wamp(x: np.ndarray, theta: float) -> int:          #Willison amptitude
    dx = np.diff(x)
    return int(np.sum(np.abs(dx) > theta))

# Optional extras

def _hjorth(x: np.ndarray) -> tuple[float, float, float]: #Hjorth
    x = x.astype(np.float64, copy=False)
    var0 = np.var(x) + EPS
    dx   = np.diff(x)
    var1 = np.var(dx) + EPS
    ddx  = np.diff(dx)
    var2 = np.var(ddx) + EPS

    activity   = var0
    mobility   = np.sqrt(var1 / var0)
    complexity = np.sqrt((var2 / var1) / (var1 / var0))
    return float(activity), float(mobility), float(complexity)

def _ar_coeffs_order4(x: np.ndarray) -> np.ndarray:        #AR4
    x = x.astype(np.float64, copy=False)
    W = x.shape[-1]
    if W < 8:
        return np.zeros(4, dtype=np.float32)
    x = x - np.mean(x)
    r = np.correlate(x, x, mode="full")
    mid = len(r) // 2
    ac = r[mid:mid+5] / (W - np.arange(5))
    R = np.array([
        [ac[0], ac[1], ac[2], ac[3]],
        [ac[1], ac[0], ac[1], ac[2]],
        [ac[2], ac[1], ac[0], ac[1]],
        [ac[3], ac[2], ac[1], ac[0]],
    ], dtype=np.float64)
    b = ac[1:5]
    try:
        a = np.linalg.solve(R, b)
    except np.linalg.LinAlgError:
        a = np.zeros(4, dtype=np.float64)
    return a.astype(np.float32)

# Global functions

def td_feature_names(single_channel: bool = True) -> list[str]:
    base = ["MAV", "RMS", "VAR", "SD", "WL", "ZC", "SSC", "WAMP"]
    if use_hjorth:
        base += ["Hj_Activity", "Hj_Mobility", "Hj_Complexity"]
    if use_ar4:
        base += ["AR1", "AR2", "AR3", "AR4"]
    if single_channel:
        return base
    return base

def td_features_one_window(win: np.ndarray) -> np.ndarray: #Extracts the features for only one window, used in the next function
    if np.isnan(win).any():
        win = np.nan_to_num(win, copy=False)
    rms = _rms(win)
    theta = 0.01 * max(rms, EPS)   # Used for WAMP
    feats = [
        _mav(win),
        rms,
        _var(win),
        _sd(win),
        _wl(win),
        _zc(win),
        _ssc(win),
        _wamp(win, theta),
    ]
    if use_hjorth:
        feats.extend(list(_hjorth(win)))
    if use_ar4:
        feats.extend(list(_ar_coeffs_order4(win)))
    return np.asarray(feats, dtype=np.float32)

def extract_td_features_per_window(windows: np.ndarray) -> np.ndarray: #Extracts the features for a given number of windows and channels
    win_arr = np.asarray(windows)
    if win_arr.ndim == 2:      # (N, W) number of windows and window length
        feats = [td_features_one_window(win_arr[i]) for i in range(len(win_arr))]
        X = np.vstack(feats)
    elif win_arr.ndim == 3:    # (N, C, W) +channels
        N, C, W = win_arr.shape
        rows = []
        for i in range(N):
            f_ch = [td_features_one_window(win_arr[i, c]) for c in range(C)]
            rows.append(np.hstack(f_ch))
        X = np.vstack(rows)
    else:
        raise ValueError("windows must be (N, W) or (N, C, W)")
    return X

# Test

if __name__ == "__main__":
    N, W = 6, 200
    t = np.linspace(0, 1, W, endpoint=False)
    wins = []
    for i in range(N):
        x = 0.5*np.sin(2*np.pi*5*t) + 0.05*np.random.randn(W)
        wins.append(x.astype(np.float32))
    wins = np.stack(wins, axis=0)
    X = extract_td_features_per_window(wins)
    print("X shape:", X.shape)
    print("features:", td_feature_names())
    print("first row:", X[0])