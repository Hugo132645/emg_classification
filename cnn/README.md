# CNN Spectrograms for EMG Classification

This adds spectrogram generation for the CNN track of the EMG classification project.

## What This Does

Converts EMG signals into spectrograms that can be used with CNN models for gesture recognition.

## Quick Start

```bash
# Run the simple example
python src/cnn/example_simple.py
```

This will:
1. Generate dummy EMG data (using the existing `dummy_data.py`)
2. Window the signal (using the existing `windowing.py`)
3. **Create spectrograms** (new functionality)
4. Save a visualization to `outputs/example_spectrograms.png`

## What's New

### Main File: `src/cnn/transforms/spectrograms.py`

Two functions:

**`compute_stft_spectrogram(window, fs)`** - Convert one EMG window to a spectrogram  
**`batch_compute_spectrograms(windows, fs)`** - Convert multiple windows at once

### Example Usage

```python
from src.common.io.dummy_data import generate_dummy_emg
from src.common.preprocessing.windowing import window_signal
from src.cnn.transforms.spectrograms import batch_compute_spectrograms

# 1. Get dummy data (already exists in project)
df, labels, timestamps = generate_dummy_emg(
    seconds=20, fs=1000, classes=['fist', 'open', 'pinch']
)

# 2. Window it (already exists in project)
windows, _, _ = window_signal(df['signal'].values, fs=1000, window_ms=200)

# 3. Create spectrograms (NEW!)
spectrograms = batch_compute_spectrograms(windows, fs=1000)

# Now you have spectrograms ready for a CNN!
# Shape: (num_windows, 64, 2) - 64 frequency bins, 2 time frames
```

### Title
```
Add STFT log-mel spectrogram generation for CNN track
```

### Description
```
Implements time-frequency spectrogram generation for CNN-based EMG gesture classification.

- Adds spectrograms.py with STFT log-mel implementation
- Integrates with existing dummy_data and windowing modules
- Includes simple example, comprehensive demo, and unit tests
- Follows project structure (src/cnn/transforms/)

Usage:
python src/cnn/example_simple.py
```



## Testing

```bash
# Run the simple example
python src/cnn/example_simple.py

# Run unit tests
python src/cnn/test_spectrograms.py

# Run full demo (with detailed plots)
python -m src.cnn.demo_spectrograms
```
