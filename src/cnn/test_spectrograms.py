"""
Simple test script to verify spectrogram generation works correctly.
This can be run independently to test the spectrograms module.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cnn.transforms.spectrograms import compute_stft_spectrogram, batch_compute_spectrograms


def test_single_spectrogram():
    """Test single spectrogram generation."""
    print("Test 1: Single spectrogram generation")
    print("-" * 50)
    
    # Create a test signal: 200ms @ 1kHz = 200 samples
    fs = 1000
    t = np.linspace(0, 0.2, 200)
    
    # Simulated EMG: mix of different frequencies
    signal = (
        np.sin(2 * np.pi * 50 * t) +      # 50 Hz component
        0.5 * np.sin(2 * np.pi * 100 * t) # 100 Hz component
    )
    
    # Add noise
    signal += np.random.randn(200) * 0.1
    
    # Compute spectrogram
    spec = compute_stft_spectrogram(signal, fs=fs)
    
    # Verify
    assert spec.shape[0] == 64, f"Expected 64 mel bins, got {spec.shape[0]}"
    assert spec.dtype == np.float32, f"Expected float32, got {spec.dtype}"
    assert not np.isnan(spec).any(), "Spectrogram contains NaN values"
    assert not np.isinf(spec).any(), "Spectrogram contains Inf values"
    
    print(f"✓ Input shape: {signal.shape}")
    print(f"✓ Output shape: {spec.shape}")
    print(f"✓ Output dtype: {spec.dtype}")
    print(f"✓ Value range: [{spec.min():.2f}, {spec.max():.2f}]")
    print(f"✓ No NaN or Inf values")
    print()


def test_batch_spectrograms():
    """Test batch spectrogram generation."""
    print("Test 2: Batch spectrogram generation")
    print("-" * 50)
    
    # Create batch of windows
    n_windows = 10
    window_length = 200
    fs = 1000
    
    # Random EMG-like signals
    windows = np.random.randn(n_windows, window_length).astype(np.float32)
    
    # Compute batch
    specs = batch_compute_spectrograms(windows, fs=fs)
    
    # Verify
    assert specs.shape == (n_windows, 64, 2), f"Expected shape (10, 64, 2), got {specs.shape}"
    assert specs.dtype == np.float32, f"Expected float32, got {specs.dtype}"
    assert not np.isnan(specs).any(), "Batch spectrograms contain NaN"
    assert not np.isinf(specs).any(), "Batch spectrograms contain Inf"
    
    print(f"✓ Input shape: {windows.shape}")
    print(f"✓ Output shape: {specs.shape}")
    print(f"✓ All spectrograms valid")
    print()


def test_different_parameters():
    """Test with different parameters."""
    print("Test 3: Different parameters")
    print("-" * 50)
    
    signal = np.random.randn(200)
    fs = 1000
    
    # Test with different n_mels
    spec_32 = compute_stft_spectrogram(signal, fs=fs, n_mels=32)
    spec_128 = compute_stft_spectrogram(signal, fs=fs, n_mels=128)
    
    assert spec_32.shape[0] == 32, "Failed with n_mels=32"
    assert spec_128.shape[0] == 128, "Failed with n_mels=128"
    
    print(f"✓ n_mels=32: shape {spec_32.shape}")
    print(f"✓ n_mels=128: shape {spec_128.shape}")
    
    # Test with different n_fft
    spec_128_fft = compute_stft_spectrogram(signal, fs=fs, n_fft=128)
    spec_512_fft = compute_stft_spectrogram(signal, fs=fs, n_fft=512)
    
    print(f"✓ n_fft=128: shape {spec_128_fft.shape}")
    print(f"✓ n_fft=512: shape {spec_512_fft.shape}")
    print()


def main():
    """Run all tests."""
    print("=" * 50)
    print("Spectrogram Generation Test Suite")
    print("=" * 50)
    print()
    
    try:
        test_single_spectrogram()
        test_batch_spectrograms()
        test_different_parameters()
        
        print("=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("=" * 50)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
