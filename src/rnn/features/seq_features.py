import numpy as np
from typing import List, Tuple
from src.common.io.dummy_data import generate_dummy_emg
from src.common.io.schemas import load_cfg
from src.common.preprocessing.windowing import window_signal

EPS = 1e-7

def _windows_correct(windows: np.ndarray) -> np.array:
    windows = np.asarray(windows)
    if windows.ndim == 1:
        return windows[None, :]
    if windows.ndim == 2:
        return windows
    raise ValueError("Windows should be 1D, or 2D stack of Windows")

def _basic_stats_(windows: np.ndarray) -> Tuple[np.array, list[str]]:
    windows = _windows_correct(windows).astype(np.float64)
    mean = windows.mean(axis=1)
    std = windows.std(axis=1) + EPS
    min_ = windows.min(axis=1)
    max_ = windows.max(axis=1)
    range_ = max_ - min_
    w0 = windows - windows.mean(axis=1, keepdims=True)
    zero_crossings = (w0[:, 1:] * w0[:,-1:] < 0 ).mean(axis=1)
    feat = np.stack([mean, std, min_, max_, range_, zero_crossings], axis=1)
    names = ['mean', 'std', 'min', 'max', 'range', 'zero_crossing_rate']
    return feat, names

def _shape_feat_(windows: np.ndarray, fs: float) -> Tuple[np.array, list[str]]:
    windows = _windows_correct(windows).astype(np.float64)
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
    names = ['slope', 'moment_skew', 'excess_kurtosis']
    return feat.astype(np.float64), names

def _spectral_light_(windows: np.ndarray, fs: float) -> Tuple[np.array, list[str]]:
    windows = _windows_correct(windows).astype(np.float64)
    x = windows - windows.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(x,axis=1)
    pow_spectrum = (fft.real**2 + fft.imag**2) / windows.shape[1]
    freqs = np.fft.rfftfreq(windows.shape[1], d = 1.0 / fs)

    pow_sum = pow_spectrum.sum(axis=1) + EPS
    centroid = (pow_spectrum * freqs[None, :]).sum(axis=1) / pow_sum
    bandwidth = np.sqrt((pow_spectrum * (freqs[None, :] - centroid[:, None])**2).sum(axis=1) / pow_sum)
    spec_bb_pow = pow_sum

    feat = np.stack([centroid, bandwidth, spec_bb_pow], axis=1)
    names = ['spec_centroid_hz', 'spec_bandwidth', 'spec_bb_power']
    return feat, names

def _add_delta_batch_(F: np.ndarray, names: list[str]) -> tuple[np.array, list[str]]:
    diff = np.vstack([np.zeros((1,F.shape[1]), dtype=F.dtype), np.diff(F, axis=0)])
    features_out = np.concatenate([F, diff], axis=1)
    names_out = names + [f"{n}_delta" for n in names]
    return features_out, names


def compute_seq_features(windows: np.ndarray, fs: float,
                         basic_feat: bool = True, shape_feat: bool = True, spectral_feat: bool = False, deltas: bool = False)\
        -> Tuple[np.array, list[str]]:

    blocks: List[np.ndarray] = []
    names: List[str] = []

    if basic_feat:
        feature_vector, n = _basic_stats_(windows);
        blocks.append(feature_vector);
        names += n
    if shape_feat:
        feature_vector, n = _shape_feat_(windows, fs);
        blocks.append(feature_vector);
        names += n
    if spectral_feat:
        feature_vector, n = _spectral_light_(windows, fs);
        blocks.append(feature_vector);
        names += n

    if not blocks:
        raise ValueError("Select at least one feature block (basic/shape/spectral).")

    # horizontally stack feature blocks: (N, D_total)
    feature_vector_all = np.hstack(blocks).astype(np.float32)
    if deltas:
        feature_vector_all, names = _add_delta_batch_(feature_vector_all, names)
    return feature_vector_all, names



if __name__ == '__main__':
    num = np.array([1,2,3])
    Gestures = load_cfg().gestures
    dat, int_labels, time_stamps = generate_dummy_emg(10, 1000, Gestures)
    window, window_labels, times_ms = window_signal(dat.signal, 1000, labels = int_labels, timestamps_ms=time_stamps)
    Fvector1, names1 = _shape_feat_(window[0], 1000)
    print(Fvector1, names1)
    Fvector2, names2 = compute_seq_features(window[0], 1000, basic_feat=True, shape_feat=True, spectral_feat=True)
    print(Fvector2, names2)
    print(Fvector2.shape, names1)






