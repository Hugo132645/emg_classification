# Imports

from __future__ import annotations
import numpy as np
from scipy.signal import welch

#Constant

EPS = 1e-12  # to avoid divide-by-zero and log(0)

# Frequency bands for band-power features (in Hz)
Bands = [
    (20.0, 150.0),
    (150.0, 350.0),
]

# Welch PSD: transform time-domain window into frequency domain

def _welch_psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Welch PSD expects a 1D window.")
    nperseg = min(256, len(x))
    if nperseg < 8:
        # too short to estimate a meaningful spectrum
        freqs = np.array([0.0], dtype=np.float64)
        psd = np.array([0.0], dtype=np.float64)
        return freqs, psd
    freqs, psd = welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
    )
    return freqs, psd

# Frequency features

def _band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:  #Band power
    f_low, f_high = band
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    f_band = freqs[mask]
    p_band = psd[mask]
    df = np.mean(np.diff(freqs)) if len(freqs) > 1 else 1.0
    return float(np.sum(p_band) * df)

def _mean_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:   #Mean frequency
    total_power = np.sum(psd)
    if total_power <= EPS:
        return 0.0
    return float(np.sum(freqs * psd) / total_power)

def _median_frequency(freqs: np.ndarray, psd: np.ndarray) -> float: #Median frequency
    df = np.mean(np.diff(freqs)) if len(freqs) > 1 else 1.0
    power = psd * df
    cumsum = np.cumsum(power)
    total = cumsum[-1] if len(cumsum) > 0 else 0.0
    if total <= EPS:
        return 0.0
    half = 0.5 * total
    return float(np.interp(half, cumsum, freqs))

def _spectral_entropy(psd: np.ndarray) -> float:       #Spectral entropy
    power = psd.clip(min=0.0)
    total = np.sum(power)
    if total <= EPS:
        return 0.0
    p = power / total
    # avoid log(0) with EPS
    ent = -np.sum(p * np.log(p + EPS))
    # normalize by log(N)
    n = len(p)
    if n <= 1:
        return 0.0
    return float(ent / np.log(n))

# Global functions

def feature_names_freq(single_channel: bool = True) -> list[str]:
    base = [
        "BP_20_150",
        "BP_150_350",
        "MeanFreq",
        "MedianFreq",
        "SpecEntropy",
    ]
    return base

def fd_features_one_window(win: np.ndarray, fs: float) -> np.ndarray:   #Extracts the features for only one window, used in the next function
    if np.isnan(win).any():
        win = np.nan_to_num(win, copy=False)
    freqs, psd = _welch_psd(win, fs=fs)
    bp_feats = [_band_power(freqs, psd, band) for band in Bands]
    mf = _mean_frequency(freqs, psd)
    medf = _median_frequency(freqs, psd)
    sent = _spectral_entropy(psd)
    feats = [
        *bp_feats,
        mf,
        medf,
        sent,
    ]
    return np.asarray(feats, dtype=np.float32)


def extract_fd_features_per_window(windows: np.ndarray, fs: float) -> np.ndarray:   #Extracts the features for a given number of windows and channels
    win_arr = np.asarray(windows)
    if win_arr.ndim == 2:  # (N, W) number of windows and window length
        feats = [fd_features_one_window(win_arr[i], fs) for i in range(len(win_arr))]
        X = np.vstack(feats)
    elif win_arr.ndim == 3:  # (N, C, W) +channels
        N, C, W = win_arr.shape
        rows = []
        for i in range(N):
            f_ch = [fd_features_one_window(win_arr[i, c], fs) for c in range(C)]
            rows.append(np.hstack(f_ch))
        X = np.vstack(rows)
    else:
        raise ValueError("windows must be (N, W) or (N, C, W)")

    return X

# Test

if __name__ == "__main__":
    fs = 1000.0
    N, W = 6, 200
    t = np.linspace(0, 0.2, W, endpoint=False)
    wins = []
    for i in range(N):
        x = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 120 * t)
        x += 0.05 * np.random.randn(W)
        wins.append(x.astype(np.float32))
    wins = np.stack(wins, axis=0)
    X = extract_fd_features_per_window(wins, fs=fs)
    print("X shape:", X.shape)
    print("feature names:", feature_names_freq())
    print("first row:", X[0])