"""
Simple example: Generate spectrograms from dummy EMG data.

This script shows the basic usage of the spectrogram module.
Run this to see spectrograms generated from the project's dummy data.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from the common modules (already in the project)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.common.io.dummy_data import generate_dummy_emg
from src.common.preprocessing.windowing import window_signal

# Import the new spectrogram module
from src.cnn.transforms.spectrograms import batch_compute_spectrograms


def main():
    """Simple example of generating spectrograms from dummy EMG data."""
    
    print("Generating spectrograms from dummy EMG data...\n")
    
    # Step 1: Generate dummy EMG data (using existing function from the project)
    print("1. Generating dummy EMG data...")
    df, labels, timestamps = generate_dummy_emg(
        seconds=20,              # 20 seconds of data
        fs=1000,                 # 1000 Hz sampling rate
        classes=['fist', 'open', 'pinch'],  # Gesture classes
        block_s=5,               # 5 second blocks
        mode="raw",              # Raw EMG signal
        seed=42                  # For reproducibility
    )
    signal = df['signal'].values
    print(f"   Generated {len(signal)} samples\n")
    
    # Step 2: Window the signal (using existing windowing function)
    print("2. Windowing the signal...")
    windows, window_labels, times = window_signal(
        signal, 
        fs=1000,
        window_ms=200,           # 200ms windows
        hop_ms=100,              # 100ms hop (50% overlap)
        labels=labels,
        timestamps_ms=timestamps
    )
    print(f"   Created {len(windows)} windows of {windows.shape[1]} samples each\n")
    
    # Step 3: Generate spectrograms (NEW - this is what we added!)
    print("3. Computing spectrograms...")
    spectrograms = batch_compute_spectrograms(
        windows,
        fs=1000,
        n_fft=256,               # FFT size
        hop_length=128,          # Hop between frames
        n_mels=64                # Number of mel frequency bins
    )
    print(f"   Generated {len(spectrograms)} spectrograms")
    print(f"   Spectrogram shape: {spectrograms[0].shape} (mel_bins, time_frames)\n")
    
    # Step 4: Visualize a few examples
    print("4. Visualizing spectrograms...")
    
    # Find one example of each gesture
    gesture_indices = {}
    for i, label in enumerate(window_labels):
        if label and label not in gesture_indices:
            gesture_indices[label] = i
    
    # Create a simple plot
    fig, axes = plt.subplots(1, len(gesture_indices), figsize=(12, 3))
    
    for ax, (gesture, idx) in zip(axes, sorted(gesture_indices.items())):
        spec = spectrograms[idx]
        
        # Display the spectrogram
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'{gesture.upper()}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency (mel)')
        plt.colorbar(im, ax=ax, label='Magnitude (dB)')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "example_spectrograms.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"   Saved visualization to: {output_path}\n")
    print("✓ Done! Check the outputs folder to see your spectrograms.")


if __name__ == "__main__":
    main()
