import numpy as np
from typing import List, Tuple

EPS = 1e-7


def _windows_correct_multichannel(windows: np.ndarray) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float64)

    if windows.ndim == 1:
        return windows[None, :, None]   # (1, time, 1)

    if windows.ndim == 2:
        return windows[None, :, :]      # (1, time, channels)

    if windows.ndim == 3:
        return windows                  # (n_windows, time, channels)

    raise ValueError("Windows should be 1D, 2D, or 3D.")


def _basic_stats_(windows: np.ndarray) -> Tuple[np.ndarray, list[str]]:
    windows = _windows_correct_multichannel(windows)   # (W, T, C)
    mean = windows.mean(axis=1)
    std = windows.std(axis=1) + EPS
    min_ = windows.min(axis=1)
    max_ = windows.max(axis=1)
    range_ = max_ - min_

    w0 = windows - windows.mean(axis=1, keepdims=True)
    zero_crossings = (w0[:, 1:, :] * w0[:, :-1, :] < 0).mean(axis=1)

    feat = np.concatenate(
        [mean, std, min_, max_, range_, zero_crossings],
        axis=1
    )

    n_channels = windows.shape[2]
    names = (
        [f'ch{c+1}_mean' for c in range(n_channels)] +
        [f'ch{c+1}_std' for c in range(n_channels)] +
        [f'ch{c+1}_min' for c in range(n_channels)] +
        [f'ch{c+1}_max' for c in range(n_channels)] +
        [f'ch{c+1}_range' for c in range(n_channels)] +
        [f'ch{c+1}_zero_crossing_rate' for c in range(n_channels)]
    )

    return feat.astype(np.float64), names


def _shape_feat_(windows: np.ndarray, fs: float) -> Tuple[np.ndarray, list[str]]:
    windows = _windows_correct_multichannel(windows)   # (W, T, C)
    _, time_st, n_channels = windows.shape

    t = np.arange(time_st, dtype=np.float64) / max(fs, 1.0)   # (T,)
    t = (t - t.mean()) / (t.std() + EPS)                      # normalized time
    t = t[None, :, None]                                      # (1, T, 1)

    x = windows - windows.mean(axis=1, keepdims=True)         # (W, T, C)

    denom = (t ** 2).sum() + EPS
    slope = (x * t).sum(axis=1) / denom                       # (W, C)

    s = x.std(axis=1) + EPS                                   # (W, C)

    skew = np.mean((x / s[:, None, :]) ** 3, axis=1)          # (W, C)
    kurt = np.mean((x / s[:, None, :]) ** 4, axis=1) - 3.0    # (W, C)

    feat = np.concatenate([slope, skew, kurt], axis=1)

    names = (
        [f'ch{c+1}_slope' for c in range(n_channels)] +
        [f'ch{c+1}_moment_skew' for c in range(n_channels)] +
        [f'ch{c+1}_excess_kurtosis' for c in range(n_channels)]
    )

    return feat.astype(np.float64), names


def _spectral_light_(windows: np.ndarray, fs: float) -> Tuple[np.ndarray, list[str]]:
    windows = _windows_correct_multichannel(windows)   # (W, T, C)
    _, time_st, n_channels = windows.shape

    x = windows - windows.mean(axis=1, keepdims=True)          # (W, T, C)

    fft = np.fft.rfft(x, axis=1)                               # (W, F, C)
    pow_spectrum = (np.abs(fft) ** 2) / time_st                # (W, F, C)

    freqs = np.fft.rfftfreq(time_st, d=1.0 / fs)               # (F,)
    freqs_reshaped = freqs[None, :, None]                      # (1, F, 1)

    pow_sum = pow_spectrum.sum(axis=1) + EPS                   # (W, C)

    centroid = (pow_spectrum * freqs_reshaped).sum(axis=1) / pow_sum   # (W, C)

    bandwidth = np.sqrt(
        (pow_spectrum * (freqs_reshaped - centroid[:, None, :]) ** 2).sum(axis=1) / pow_sum
    )                                                          # (W, C)

    spec_bb_power = pow_sum                                    # (W, C)

    feat = np.concatenate([centroid, bandwidth, spec_bb_power], axis=1)

    names = (
        [f'ch{c+1}_spec_centroid_hz' for c in range(n_channels)] +
        [f'ch{c+1}_spec_bandwidth' for c in range(n_channels)] +
        [f'ch{c+1}_spec_bb_power' for c in range(n_channels)]
    )

    return feat.astype(np.float64), names


def _add_delta_batch_(F: np.ndarray, names: list[str]) -> Tuple[np.ndarray, list[str]]:
    """
    Delta across consecutive windows, not across time samples.
    Input shape: (n_windows, n_features)
    """
    F = np.asarray(F)
    diff = np.vstack([np.zeros((1, F.shape[1]), dtype=F.dtype), np.diff(F, axis=0)])
    features_out = np.concatenate([F, diff], axis=1)
    names_out = names + [f"{n}_delta" for n in names]
    return features_out, names_out


def compute_seq_features(
    windows: np.ndarray,
    fs: float,
    basic_feat: bool = True,
    shape_feat: bool = True,
    spectral_feat: bool = False,
    deltas: bool = False
) -> Tuple[np.ndarray, list[str]]:
    """
    Input:
    - one window: (time, channels)
    - batch: (n_windows, time, channels)

    Output:
    - features: (n_windows, n_features)
    - names: list of feature names
    """
    windows = _windows_correct_multichannel(windows)

    blocks: List[np.ndarray] = []
    names: List[str] = []

    if basic_feat:
        f, n = _basic_stats_(windows)
        blocks.append(f)
        names += n

    if shape_feat:
        f, n = _shape_feat_(windows, fs)
        blocks.append(f)
        names += n

    if spectral_feat:
        f, n = _spectral_light_(windows, fs)
        blocks.append(f)
        names += n

    if not blocks:
        raise ValueError("Select at least one feature block.")

    feature_vector_all = np.hstack(blocks).astype(np.float32)

    if deltas:
        feature_vector_all, names = _add_delta_batch_(feature_vector_all, names)

    return feature_vector_all, names