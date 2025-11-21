"""
Demo script to generate and visualize spectrograms from dummy EMG data.

This script demonstrates the full pipeline:
1. Generate dummy EMG data
2. Preprocess the signal
3. Window the signal
4. Compute spectrograms
5. Visualize and save results
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.io.dummy_data import generate_dummy_emg
from src.common.io.schemas import load_cfg
from src.common.preprocessing.pipelines import preprocess_raw, preprocess_envelope
from src.common.preprocessing.windowing import window_signal
from src.cnn.transforms.spectrograms import batch_compute_spectrograms


def main():
    """Run the spectrogram generation demo."""
    
    # Load configuration
    cfg = load_cfg("configs/preprocessing.yaml")
    
    # Parameters
    fs = cfg.sample_rate_hz  # 1000 Hz
    duration_s = 30  # 30 seconds of data
    classes = ['fist', 'open', 'pinch']  # Using classes from config (excluding rest)
    
    print("=" * 70)
    print("EMG Spectrogram Generation Demo")
    print("=" * 70)
    
    # Step 1: Generate dummy EMG data
    print("\n[1/5] Generating dummy EMG data...")
    df, interval_labels, timestamps_ms = generate_dummy_emg(
        seconds=duration_s,
        fs=fs,
        classes=classes,
        block_s=5,
        mode="raw",  # Generate raw EMG
        seed=42
    )
    
    signal = df['signal'].values
    labels = df['label'].values
    print(f"  ✓ Generated {len(signal)} samples ({duration_s}s @ {fs}Hz)")
    print(f"  ✓ Gestures: {set(interval_labels)}")
    
    # Step 2: Preprocess the signal
    print("\n[2/5] Preprocessing signal...")
    processed_signal, meta = preprocess_raw(
        signal,
        fs=fs,
        band=(20.0, 450.0),  # EMG typical range
        lp_cut=10.0,  # Envelope smoothing
        norm="zscore"
    )
    print(f"  ✓ Applied: bandpass → rectify → lowpass → zscore normalization")
    print(f"  ✓ Signal shape: {processed_signal.shape}")
    
    # Step 3: Window the signal
    print("\n[3/5] Windowing signal...")
    windows, window_labels, times_ms = window_signal(
        processed_signal,
        fs=fs,
        window_ms=cfg.window_ms,
        hop_ms=cfg.hop_ms,
        labels=interval_labels,
        timestamps_ms=timestamps_ms
    )
    print(f"  ✓ Created {len(windows)} windows")
    print(f"  ✓ Window size: {cfg.window_ms}ms ({windows.shape[1]} samples)")
    print(f"  ✓ Hop size: {cfg.hop_ms}ms")
    
    # Step 4: Compute spectrograms
    print("\n[4/5] Computing spectrograms...")
    spectrograms = batch_compute_spectrograms(
        windows,
        fs=fs,
        n_fft=256,
        hop_length=128,
        n_mels=64,
        fmin=20.0,
        fmax=500.0  # Up to 500 Hz (Nyquist is 500 Hz)
    )
    print(f"  ✓ Generated {len(spectrograms)} spectrograms")
    print(f"  ✓ Spectrogram shape: {spectrograms.shape} (N, mels, time)")
    print(f"  ✓ Value range: [{spectrograms.min():.2f}, {spectrograms.max():.2f}]")
    
    # Step 5: Visualize and save results
    print("\n[5/5] Visualizing spectrograms...")
    
    # Create output directory
    output_dir = Path("outputs/spectrograms")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select one example per gesture class
    gesture_examples = {}
    for i, label in enumerate(window_labels):
        if label and label != 'rest' and label not in gesture_examples:
            gesture_examples[label] = i
        if len(gesture_examples) == len(classes):
            break
    
    # Create visualization
    n_examples = len(gesture_examples)
    fig, axes = plt.subplots(2, n_examples, figsize=(4 * n_examples, 8))
    if n_examples == 1:
        axes = axes.reshape(2, 1)
    
    for col, (gesture, idx) in enumerate(sorted(gesture_examples.items())):
        # Plot raw window
        axes[0, col].plot(windows[idx], linewidth=0.8)
        axes[0, col].set_title(f'{gesture.capitalize()} - EMG Window', fontweight='bold')
        axes[0, col].set_xlabel('Sample')
        axes[0, col].set_ylabel('Amplitude (normalized)')
        axes[0, col].grid(True, alpha=0.3)
        
        # Plot spectrogram
        spec = spectrograms[idx]
        im = axes[1, col].imshow(
            spec,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        axes[1, col].set_title(f'{gesture.capitalize()} - Log-Mel Spectrogram', fontweight='bold')
        axes[1, col].set_xlabel('Time Frame')
        axes[1, col].set_ylabel('Mel Bin')
        plt.colorbar(im, ax=axes[1, col], label='Log Magnitude')
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "spectrogram_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization: {output_path}")
    
    # Also create a grid showing multiple examples of one gesture
    print("\n[Bonus] Creating detailed view of 'fist' gesture...")
    fist_indices = [i for i, label in enumerate(window_labels) if label == 'fist'][:6]
    
    if len(fist_indices) >= 6:
        fig2, axes2 = plt.subplots(2, 3, figsize=(12, 6))
        axes2 = axes2.flatten()
        
        for i, idx in enumerate(fist_indices):
            spec = spectrograms[idx]
            im = axes2[i].imshow(
                spec,
                aspect='auto',
                origin='lower',
                cmap='viridis',
                interpolation='nearest'
            )
            axes2[i].set_title(f'Fist #{i+1}')
            axes2[i].set_xlabel('Time Frame')
            axes2[i].set_ylabel('Mel Bin')
            plt.colorbar(im, ax=axes2[i])
        
        plt.suptitle('Multiple "Fist" Gesture Spectrograms', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path2 = output_dir / "fist_spectrograms_grid.png"
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved grid: {output_path2}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Total windows: {len(windows)}")
    print(f"Total spectrograms: {len(spectrograms)}")
    print(f"Spectrogram shape: {spectrograms[0].shape} (mels × time)")
    print(f"Memory per spectrogram: {spectrograms[0].nbytes / 1024:.2f} KB")
    print(f"Total memory: {spectrograms.nbytes / (1024**2):.2f} MB")
    
    # Count windows per class
    print("\nWindows per gesture:")
    from collections import Counter
    label_counts = Counter([l for l in window_labels if l])
    for gesture, count in sorted(label_counts.items()):
        print(f"  {gesture:>8}: {count:4d}")
    
    print("\n" + "=" * 70)
    print("✓ Demo completed successfully!")
    print(f"✓ Output saved to: {output_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
