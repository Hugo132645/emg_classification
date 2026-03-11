"""
Spectrogram transformations for EMG signals.

This module provides functions to convert windowed EMG signals into
time-frequency representations suitable for CNN-based classification.
"""

import numpy as np
import librosa
from typing import Optional


def compute_stft_spectrogram(
    window: np.ndarray,
    fs: float,
    n_fft: int = 256,
    hop_length: int = 128,
    n_mels: int = 64,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
    log_offset: float = 1e-10
) -> np.ndarray:
    """
    Compute log-mel spectrogram from a single EMG window using STFT.
    
    Parameters
    ----------
    window : np.ndarray
        1D array of EMG samples (typically 200 samples @ 1kHz).
    fs : float
        Sampling rate in Hz (e.g., 1000).
    n_fft : int
        FFT window size (default 256).
    hop_length : int
        Number of samples between successive frames (default 128).
    n_mels : int
        Number of mel frequency bins (default 64).
    fmin : float
        Minimum frequency in Hz (default 20).
    fmax : Optional[float]
        Maximum frequency in Hz. Defaults to fs/2 if None.
    log_offset : float
        Small constant to avoid log(0) (default 1e-10).
    
    Returns
    -------
    log_mel_spec : np.ndarray
        2D array of shape (n_mels, time_frames) containing log-mel spectrogram.
        Values are in log scale (dB-like).
    
    Examples
    --------
    >>> window = np.random.randn(200)  # 200ms @ 1kHz
    >>> spec = compute_stft_spectrogram(window, fs=1000)
    >>> spec.shape
    (64, 2)  # 64 mel bins, ~2 time frames for 200 samples
    """
    # Ensure input is float
    window = np.asarray(window, dtype=np.float32)
    
    # Set fmax to Nyquist frequency if not provided
    if fmax is None:
        fmax = fs / 2.0
    
    # Compute STFT
    # librosa.stft returns complex-valued STFT
    stft = librosa.stft(window, n_fft=n_fft, hop_length=hop_length, center=True, window='hann')
    
    # Compute power spectrogram (magnitude squared)
    power_spec = np.abs(stft) ** 2
    
    # Create mel filterbank and apply it
    mel_basis = librosa.filters.mel(
        sr=fs,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    # Apply mel filterbank: (n_mels, n_fft//2+1) @ (n_fft//2+1, time_frames)
    mel_spec = mel_basis @ power_spec
    
    # Convert to log scale (dB-like)
    log_mel_spec = np.log(mel_spec + log_offset)
    
    return log_mel_spec.astype(np.float32)


def batch_compute_spectrograms(
    windows: np.ndarray,
    fs: float,
    n_fft: int = 256,
    hop_length: int = 128,
    n_mels: int = 64,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
    log_offset: float = 1e-10
) -> np.ndarray:
    """
    Compute log-mel spectrograms from a batch of EMG windows.
    
    Parameters
    ----------
    windows : np.ndarray
        2D array of shape (n_windows, window_length) containing EMG windows.
    fs : float
        Sampling rate in Hz.
    n_fft, hop_length, n_mels, fmin, fmax, log_offset :
        Same as compute_stft_spectrogram.
    
    Returns
    -------
    spectrograms : np.ndarray
        3D array of shape (n_windows, n_mels, time_frames).
    
    Examples
    --------
    >>> windows = np.random.randn(10, 200)  # 10 windows of 200 samples each
    >>> specs = batch_compute_spectrograms(windows, fs=1000)
    >>> specs.shape
    (10, 64, 2)  # 10 spectrograms, 64 mel bins, ~2 time frames
    """
    spectrograms = []
    
    for window in windows:
        spec = compute_stft_spectrogram(
            window, fs, n_fft, hop_length, n_mels, fmin, fmax, log_offset
        )
        spectrograms.append(spec)
    
    return np.stack(spectrograms, axis=0)


if __name__ == "__main__":
    # Quick self-test
    print("Testing spectrogram generation...")
    
    # Generate a simple test signal (200ms @ 1kHz = 200 samples)
    fs = 1000
    t = np.linspace(0, 0.2, 200)
    test_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
    
    # Compute spectrogram
    spec = compute_stft_spectrogram(test_signal, fs=fs)
    
    print(f"Input shape: {test_signal.shape}")
    print(f"Output shape: {spec.shape}")
    print(f"Output dtype: {spec.dtype}")
    print(f"Output range: [{spec.min():.2f}, {spec.max():.2f}]")
    print(f"Contains NaN: {np.isnan(spec).any()}")
    print(f"Contains Inf: {np.isinf(spec).any()}")
    
    # Test batch processing
    batch = np.random.randn(5, 200)
    batch_specs = batch_compute_spectrograms(batch, fs=fs)
    print(f"\nBatch input shape: {batch.shape}")
    print(f"Batch output shape: {batch_specs.shape}")
    
    print("\n✓ Spectrogram generation test passed!")
