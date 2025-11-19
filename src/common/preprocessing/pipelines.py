# src/preprocessing/pipelines.py
from __future__ import annotations

from typing import Dict, Tuple, Literal, Any
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

NormMethod = Literal["zscore", "minmax"]


# ---------- helpers ----------
def _as_float(x: np.ndarray) -> np.ndarray:
    """Operate in float64 internally; keep original shape."""
    return np.asarray(x, dtype=np.float64)


def _butter_bandpass(low: float, high: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    if not (0 < low < high < nyq):
        raise ValueError(
            f"Invalid bandpass: low={low}, high={high}, Nyquist={nyq}. "
            "Ensure 0 < low < high < fs/2."
        )
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b, a


def _butter_lowpass(cutoff: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    if not (0 < cutoff < nyq):
        raise ValueError(
            f"Invalid low-pass cutoff: {cutoff} (Nyquist={nyq}). "
            "Ensure 0 < cutoff < fs/2."
        )
    b, a = butter(order, cutoff / nyq, btype="low")
    return b, a


def _apply_filtfilt(b: np.ndarray, a: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable filtfilt with automatic short-signal padlen guard."""
    padlen = 3 * max(len(a), len(b))
    n = x.shape[axis]
    if n <= 1:
        return x
    if n <= padlen:
        return filtfilt(b, a, x, axis=axis, padlen=n - 1)
    return filtfilt(b, a, x, axis=axis)


def _normalize(x: np.ndarray, method: NormMethod, axis: int = -1) -> Tuple[np.ndarray, Dict[str, Any]]:
    if method == "zscore":
        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, ddof=0, keepdims=True)
        std_safe = np.where(std == 0, 1.0, std)
        y = (x - mean) / std_safe
        stats = {"method": "zscore", "mean": np.squeeze(mean), "std": np.squeeze(std_safe)}
        return y, stats

    if method == "minmax":
        x_min = np.min(x, axis=axis, keepdims=True)
        x_max = np.max(x, axis=axis, keepdims=True)
        rng = np.where((x_max - x_min) == 0, 1.0, (x_max - x_min))
        y = (x - x_min) / rng
        stats = {"method": "minmax", "min": np.squeeze(x_min), "max": np.squeeze(x_max)}
        return y, stats

    raise ValueError(f"Unknown normalization method: {method}")


def _restore_dtype(x_norm: np.ndarray, orig_dtype: np.dtype, method: NormMethod) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Return array cast back to original dtype with a sensible mapping recorded in metadata.
    - If original dtype is float: cast back directly.
    - If original dtype is integer:
        * minmax: map [0,1] -> full int range
        * zscore: clip [-4,4], map to full int range (robust spread)
    """
    meta: Dict[str, Any] = {"dtype_out": str(orig_dtype)}
    if np.issubdtype(orig_dtype, np.floating):
        meta["output_scale"] = {"strategy": "float-cast"}
        return x_norm.astype(orig_dtype, copy=False), meta

    info = np.iinfo(orig_dtype)
    lo, hi = info.min, info.max

    if method == "minmax":
        s = np.clip(x_norm, 0.0, 1.0)
        mapped = lo + s * (hi - lo)
        meta["output_scale"] = {"strategy": "minmax_to_full_int_range", "range": [int(lo), int(hi)]}
    else:
        z = np.clip(x_norm, -4.0, 4.0)
        s = (z + 4.0) / 8.0
        mapped = lo + s * (hi - lo)
        meta["output_scale"] = {
            "strategy": "zscore_clipped_to_full_int_range",
            "clip": [-4.0, 4.0],
            "range": [int(lo), int(hi)],
        }

    return np.rint(mapped).astype(orig_dtype, copy=False), meta


# --------------------
def preprocess_raw(
    x: np.ndarray,
    fs: float,
    *,
    notch_50hz: bool = False,
    band: Tuple[float, float] = (20.0, 450.0),
    lp_cut: float = 10.0,
    band_order: int = 4,
    lp_order: int = 4,
    notch_q: float = 30.0,
    norm: NormMethod = "zscore",
    axis: int = -1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Raw EMG pipeline:
        bandpass(20–450 Hz) -> optional notch(50 Hz) -> rectify -> low-pass(10 Hz) -> normalize.
    Returns same SHAPE and same DTYPE as the input, plus a metadata dict.

    Parameters
    ----------
    x : np.ndarray
        Input signal. Any shape. Processing runs along `axis`.
    fs : float
        Sampling rate in Hz.
    notch_50hz : bool
        Apply 50 Hz notch (mains) if True.
    band : (low, high)
        Bandpass in Hz (default (20, 450)).
    lp_cut : float
        Low-pass cutoff after rectification (default 10 Hz).
    band_order, lp_order : int
        Filter orders (Butterworth).
    notch_q : float
        Q-factor for notch.
    norm : {"zscore","minmax"}
        Normalization method.
    axis : int
        Processing axis (default last).

    Returns
    -------
    y : np.ndarray
        Processed signal, same dtype and shape as `x`.
    meta : dict
        Details of filters and normalization scales.
    """
    orig_dtype = np.asarray(x).dtype
    xf = _as_float(x)

    # Bandpass
    bp_b, bp_a = _butter_bandpass(band[0], band[1], fs, order=band_order)
    y = _apply_filtfilt(bp_b, bp_a, xf, axis=axis)

    # Optional 50 Hz notch
    notch_params = None
    if notch_50hz:
        w0 = 50.0 / (fs / 2.0)
        b_notch, a_notch = iirnotch(w0, Q=notch_q)
        y = _apply_filtfilt(b_notch, a_notch, y, axis=axis)
        notch_params = {"freq_hz": 50.0, "Q": notch_q}

    # Rectify then smooth envelope
    y = np.abs(y)
    lp_b, lp_a = _butter_lowpass(lp_cut, fs, order=lp_order)
    y = _apply_filtfilt(lp_b, lp_a, y, axis=axis)

    # Normalize in float
    y_norm, norm_stats = _normalize(y, method=norm, axis=axis)

    # Cast back to original dtype
    y_out, cast_meta = _restore_dtype(y_norm, orig_dtype, method=norm)

    meta: Dict[str, Any] = {
        "pipeline": "preprocess_raw",
        "fs": fs,
        "axis": axis,
        "filters": {
            "bandpass": {"low_hz": band[0], "high_hz": band[1], "order": band_order, "type": "butter"},
            "notch_50hz": notch_params,
            "lowpass_after_rectify": {"cutoff_hz": lp_cut, "order": lp_order, "type": "butter"},
        },
        "normalization": norm_stats,
        **cast_meta,
    }
    return y_out, meta


def preprocess_envelope(
    x: np.ndarray,
    fs: float,
    *,
    lp_cut: float = 10.0,
    lp_order: int = 2,
    norm: NormMethod = "zscore",
    axis: int = -1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Envelope pipeline:
        light low-pass(10 Hz) -> normalize.
    Returns same SHAPE and same DTYPE as the input, plus a metadata dict.

    Parameters
    ----------
    x : np.ndarray
        Input envelope-like signal. Any shape. Processing runs along `axis`.
    fs : float
        Sampling rate in Hz.
    lp_cut : float
        Low-pass cutoff (default 10 Hz).
    lp_order : int
        Low-pass order (default 2).
    norm : {"zscore","minmax"}
        Normalization method.
    axis : int
        Processing axis (default last).
    """
    orig_dtype = np.asarray(x).dtype
    xf = _as_float(x)

    # Light low-pass
    lp_b, lp_a = _butter_lowpass(lp_cut, fs, order=lp_order)
    y = _apply_filtfilt(lp_b, lp_a, xf, axis=axis)

    # Normalize in float
    y_norm, norm_stats = _normalize(y, method=norm, axis=axis)

    # Cast back to original dtype
    y_out, cast_meta = _restore_dtype(y_norm, orig_dtype, method=norm)

    meta: Dict[str, Any] = {
        "pipeline": "preprocess_envelope",
        "fs": fs,
        "axis": axis,
        "filters": {"lowpass": {"cutoff_hz": lp_cut, "order": lp_order, "type": "butter"}},
        "normalization": norm_stats,
        **cast_meta,
    }
    return y_out, meta
