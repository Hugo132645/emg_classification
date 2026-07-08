# Prosthetic Arm — EEG/EMG Intent Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#project-status)
[![Made with PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-red.svg)](https://pytorch.org/)
[![Signal Processing](https://img.shields.io/badge/domain-biosignal%20processing-purple.svg)](#system-pipeline)

End-to-end research prototype for classifying **human motor intent from EEG/EMG biosignals** and translating those predictions into a control layer for a prosthetic robotic arm.

The project focuses primarily on **surface electromyography (sEMG)** classification, while also following a broader **EEG/EMG human-machine interface** direction for assistive robotics. It combines biosignal acquisition, preprocessing, windowing, feature extraction, machine learning, deep learning, and prosthetic control concepts.

The repository is organized around three complementary modelling tracks:

- **Classic Machine Learning** — interpretable feature-based baselines.
- **CNN Models** — time-frequency image classification using spectrograms.
- **RNN / BRNN Models** — temporal sequence models for rolling biosignal windows.

<p align="center">
  <img src="assets/presentations/20260213_114047.jpg" alt="Project presentation booth" width="850">
</p>

<p align="center">
  <em>Project presentation and live demonstration booth at WAICF, Cannes.</em>
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [System Pipeline](#system-pipeline)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Biosignal Data Format](#biosignal-data-format)
- [Signal Preprocessing](#signal-preprocessing)
- [Windowing](#windowing)
- [Modelling Tracks](#modelling-tracks)
- [Model Comparison: Classic ML vs RNN](#model-comparison-classic-ml-vs-rnn)
- [Evaluation Protocol](#evaluation-protocol)
- [Metrics](#metrics)
- [Project Demo](#project-demo)
- [Reproducibility](#reproducibility)
- [Roadmap](#roadmap)
- [Project Status](#project-status)
- [Team](#team)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Project Overview

This project investigates how **EEG and EMG biosignals** can be used to infer user intent for prosthetic arm control.

In the current implementation, the repository focuses mainly on **surface EMG classification**. EMG signals are collected from muscle activity, processed into usable windows, and classified using machine learning and deep learning models. The resulting class predictions can then be used as a high-level control signal for a prosthetic robotic arm.

The broader research direction is to combine:

- **EMG**, which captures muscle activation.
- **EEG**, which captures neural activity.
- **Robotic control**, which translates predicted intent into prosthetic movement.

This project is a **research prototype**, not a finished medical device. It is intended for experimentation in biosignal processing, machine learning, human-machine interfaces, and assistive robotics.

---

## Motivation

Modern prosthetic systems require reliable and intuitive control methods. Traditional mechanical or button-based interfaces can be limiting because they do not directly capture the user's intended motion.

Biosignals offer a more natural control path:

- **EMG signals** can indicate muscle activation patterns associated with gestures or intended movements.
- **EEG signals** can provide an additional neural-control layer for future intent recognition.
- **Machine learning models** can map these signals to discrete commands or movement classes.
- **Robotic prosthetic systems** can use those commands to trigger functional hand or arm actions.

The purpose of this project is to explore how these components can be combined into a practical experimental pipeline.

---

## System Pipeline

The system follows a modular biosignal-processing pipeline:

```text
EEG / EMG Signal Acquisition
            ↓
Signal Cleaning and Preprocessing
            ↓
Windowing / Segmentation
            ↓
Feature Extraction or Spectrogram Generation
            ↓
Machine Learning / Deep Learning Classification
            ↓
Predicted Motor Intent
            ↓
Prosthetic Arm Control Layer
```

The pipeline is designed to support both offline experimentation and future real-time inference.

---

## Repository Structure

```text
emg_classification/
├── configs/
│   └── preprocessing.yaml              # Sampling, filtering, windowing and normalization settings
│
├── data/
│   └── input_data/
│       └── ninapro_db1/                # Local EMG dataset workspace
│
├── exports/
│   ├── classic_ml/
│   ├── classic_ml_best/
│   ├── classic_ml_online/
│   └── classic_ml_online_best/         # Saved model outputs and experiment artifacts
│
├── reports/
│   └── <date>/
│       ├── best/
│       ├── online/
│       └── rest/                       # Generated plots and evaluation visuals
│
├── src/
│   ├── common/
│   │   ├── io/
│   │   │   ├── dummy_data.py           # Synthetic EMG generation for testing
│   │   │   ├── emg_loader.py           # Data loading helpers
│   │   │   └── schemas.py              # Shared constants and file patterns
│   │   ├── preprocessing/
│   │   │   ├── pipelines.py            # Raw/envelope preprocessing functions
│   │   │   └── windowing.py            # Signal windowing and label aggregation
│   │   └── utils/
│   │       └── config.py               # YAML configuration loader
│   │
│   ├── classic_ml/
│   │   ├── datasets/
│   │   │   └── classic_ml_dataset.py   # Feature-table dataset preparation
│   │   ├── features/
│   │   │   ├── freq_domain.py          # Frequency-domain EMG features
│   │   │   └── time_domain.py          # Time-domain EMG features
│   │   ├── models/
│   │   │   ├── regularization_classic.py
│   │   │   └── train_classic.py        # Classic ML training script
│   │   └── utils/
│   │       └── plots.py                # Classic ML visualization helpers
│   │
│   ├── cnn/
│   │   ├── datasets/
│   │   │   └── cnn_dataset.py          # Dataset preparation for CNN models
│   │   ├── models/
│   │   │   ├── model_cnn.py            # CNN architecture
│   │   │   └── train_cnn_dummy.py      # CNN training entrypoint
│   │   ├── transforms/
│   │   │   └── spectrograms.py         # Spectrogram generation
│   │   ├── demo_spectrograms.py
│   │   ├── example_simple.py
│   │   └── test_spectrograms.py
│   │
│   └── rnn/
│       ├── datasets/
│       │   └── sequence_dataset.py     # Dataset preparation for sequence models
│       ├── features/
│       │   └── seq_features.py         # Sequence feature extraction
│       └── models/
│           ├── model_rnn.py            # RNN / LSTM model definition
│           └── train_rnn.py            # RNN training script
│
├── assets/
│   ├── classic_ml/                     # README figures for classic ML results
│   ├── cnn/                            # README figures for CNN results
│   └── presentations/                  # Presentation and project media
│
├── streamlit_app.py                    # Streamlit interface for demos and model interaction
├── requirements.txt
├── LICENSE
└── README.md
```

The repository now includes source code, a Streamlit app entrypoint, local dataset workspace folders, generated experiment outputs, and tracked visual assets used throughout the documentation.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hugo132645/emg_classification.git
cd emg_classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate
```

```bash
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project is designed for Python 3.10+.

---

## Quick Start

After installation, the typical workflow is:

```text
1. Prepare or load biosignal data
2. Apply preprocessing and windowing
3. Train one of the model tracks
4. Evaluate model performance
5. Use predictions as prosthetic control commands
```

Example training entry points may include:

```bash
python src/classic_ml/models/train_classic.py
```

```bash
python src/cnn/models/train_cnn.py
```

```bash
python src/rnn/models/train_rnn.py
```

Depending on your local implementation, configuration files in `configs/` should be adjusted before running experiments.

---

## Biosignal Data Format

The main data format is intended to be **Parquet**, with optional CSV exports for inspection.

Suggested raw-data path:

```text
data/raw/{subject_id}/{session_date}/session_{session_id}.parquet
```

Suggested processed-data path:

```text
data/processed/
```

Recommended columns:

| Column | Type | Description |
|---|---:|---|
| `timestamp_ms` | `int64` | Timestamp in milliseconds |
| `emg_ch1_raw` | `int16` / `float` | Raw EMG channel signal |
| `emg_ch1_env` | `float32` | EMG envelope signal, if available |
| `eeg_ch*_raw` | `float32` | Optional EEG channel values |
| `label` | `string` | Gesture or intent class |
| `subject_id` | `string` | Participant identifier |
| `session_id` | `string` | Recording session identifier |
| `trial_id` | `int` | Trial index |
| `marker` | `int` | Optional cue or event marker |
| `notes` | `string` | Optional recording notes |

The current pipeline is EMG-centered, but the schema is written so that EEG channels can be added later.

---

## Signal Preprocessing

The shared preprocessing layer is responsible for preparing raw biosignals before modelling.

For EMG, the expected preprocessing steps are:

```text
Raw EMG
  ↓
Band-pass filtering
  ↓
Notch filtering
  ↓
Rectification
  ↓
Envelope extraction
  ↓
Normalization
  ↓
Windowing
```

Typical EMG preprocessing operations include:

- Band-pass filtering to isolate the useful EMG frequency range.
- Optional notch filtering to reduce power-line noise.
- Rectification to convert the signal into absolute amplitude.
- Low-pass filtering to extract the muscle activation envelope.
- Session-level normalization.
- Sliding-window segmentation.

The shared preprocessing functions are located in:

```text
src/common/preprocessing/
```

---

## Windowing

The windowing module converts continuous biosignals into fixed-length samples for model training.

Typical configuration:

```text
Window length: 200 ms
Hop length:    100 ms
```

For each window, the pipeline returns:

```text
windows   -> signal windows
labels    -> majority label per window
times_ms  -> start and end timestamp of each window
```

This allows all model tracks to use the same segmentation strategy.

---

## Modelling Tracks

### 1. Classic Machine Learning

The classic ML track treats each EMG window as a structured feature vector and trains lightweight supervised models on top of those descriptors. It is currently the most stable baseline on `main` and serves as the clearest interpretable reference point for the rest of the project.

```text
Windowed EMG signal
        ↓
Time-domain + frequency-domain feature extraction
        ↓
Feature scaling / preprocessing
        ↓
Classical classifier
        ↓
Gesture prediction
```

This modelling path is useful early in the project because it helps us validate data quality, feature separability, and class behavior before relying on heavier neural architectures.

Typical feature families used in this pipeline:

| Feature Domain | Example Features | Why They Help |
| --- | --- | --- |
| Time-domain | RMS, MAV, standard deviation, variance | Captures amplitude and signal energy changes |
| Signal-shape | Waveform length, zero crossings, slope sign changes, Willison amplitude | Describes local movement patterns and activation changes |
| Descriptive / parametric | Hjorth parameters, autoregressive coefficients | Summarizes signal complexity and compact temporal behavior |
| Frequency-domain | Welch PSD, mean frequency, median frequency, band powers, spectral entropy | Captures spectral distribution and muscle activation characteristics |

Candidate models in the classic ML track:

| Model | Strength | Typical Role in This Project |
| --- | --- | --- |
| Logistic Regression | Simple and interpretable | Quick linear baseline |
| Support Vector Machine | Strong performance on structured feature spaces | Mid-complexity classifier baseline |
| Random Forest | Robust and easy to inspect | Current strong classical baseline |
| Gradient Boosting / XGBoost | Handles nonlinear decision boundaries well | Higher-capacity classic model for comparison |

Why this track matters:

| Advantage | Value to the Project |
| --- | --- |
| Interpretability | Makes it easier to understand which signal characteristics drive predictions |
| Fast iteration | Lets us test preprocessing and labeling assumptions quickly |
| Reliable baseline | Gives the CNN and RNN/LSTM tracks a grounded benchmark |
| Debugging utility | Helps surface data quality or feature-separation problems early |

Current implementation status:

- `main` contains the version adapted to the shared repository structure and currently works on dummy data for development, debugging, and visualization.
- The branch `classicML_testing` contains an adapted version used to test on an online dataset that does **not** match the intended EMG data format of this repository.
- That branch is intentionally kept separate from `main` so the core pipeline can stay aligned with the project's target data structure while the external dataset experiments remain isolated.

Example outputs from the current Classic ML workflow:

<p align="center">
  <img src="assets/classic_ml/confusion_matrix_rf_1777802744.png" alt="Random Forest confusion matrix" width="48%">
  <img src="assets/classic_ml/tsne_2d_1777802744.png" alt="t-SNE feature space 2D" width="48%">
</p>
<p align="center">
  <em>Random Forest confusion matrix</em> &nbsp;&nbsp;&nbsp;&nbsp; <em>t-SNE feature space (2D)</em>
</p>

<p align="center">
  <img src="assets/classic_ml/tsne_3d_1777802744.png" alt="t-SNE feature space 3D" width="48%">
  <img src="assets/classic_ml/umap_2d_1777802744.png" alt="UMAP feature space 2D" width="48%">
</p>
<p align="center">
  <em>t-SNE feature space (3D)</em> &nbsp;&nbsp;&nbsp;&nbsp; <em>UMAP feature space (2D)</em>
</p>

These visualizations show that the feature engineering pipeline already produces usable class structure, making the classic ML track a practical work-in-progress baseline while the neural tracks continue to mature.

---

### 2. CNN — Time-Frequency Classification

The CNN track converts each windowed EMG segment into a log-mel spectrogram and classifies gestures with a lightweight 2D convolutional network. It is implemented across three files:

```text
src/cnn/transforms/spectrograms.py   # Log-mel spectrogram computation (STFT + mel filterbank)
src/cnn/datasets/cnn_dataset.py      # PyTorch Dataset wrapper around spectrogram arrays
src/cnn/models/model_cnn.py          # EMGConvNet architecture
src/cnn/models/train_cnn_dummy.py    # End-to-end training/evaluation pipeline
```

```text
EMG Window (200 ms)
        ↓
Log-Mel Spectrogram (STFT → mel filterbank → log)
        ↓
SpectrogramDataset  (numpy → tensor, label encoding)
        ↓
EMGConvNet  (3 conv blocks → global average pool → FC)
        ↓
Intent Class
```

This track lets the model learn directly from a time-frequency image of the signal instead of hand-crafted features, at the cost of needing more data and compute than the classic ML baselines.

---

#### Spectrogram Generation

The file `src/cnn/transforms/spectrograms.py` turns a 1D EMG window into a 2D log-mel spectrogram using `librosa`.

```python
compute_stft_spectrogram(window, fs, n_fft=256, hop_length=128,
                          n_mels=64, fmin=20.0, fmax=None, log_offset=1e-10)
```

Processing steps:

1. **STFT** — `librosa.stft` with a Hann window, `n_fft=256`, `hop_length=128`.
2. **Power spectrum** — squared magnitude of the complex STFT.
3. **Mel filterbank** — `librosa.filters.mel(sr=fs, n_fft=256, n_mels=64, fmin=20.0, fmax=fs/2)` projects the linear-frequency power spectrum onto 64 mel bands.
4. **Log compression** — `log(mel_power + 1e-10)` to keep dynamic range CNN-friendly.

`batch_compute_spectrograms(windows, fs, ...)` loops this over an array of windows and stacks the result into a single `(N, n_mels, time_frames)` array — this is the function the training script calls once per dataset.

| Parameter | Value used in `train_cnn_dummy.py` | Meaning |
|---|---:|---|
| `n_fft` | 256 | FFT size per STFT frame |
| `hop_length` | 128 | Samples between STFT frames (spectrogram time resolution) |
| `n_mels` | 64 | Number of mel frequency bins |
| `fmin` / `fmax` | 20 Hz / 500 Hz | Mel filterbank frequency range |

> **Requires `fs ≈ 1000 Hz`.** With a 200-sample window (200 ms @ 1 kHz) and `hop_length=128`, this configuration produces spectrograms of shape `(64, 2)` — 64 mel bins by 2 time frames. See [Reproducibility Notes](#reproducibility-notes-cnn) below for why the sample rate matters and what happens if it isn't 1 kHz.

---

#### CNN Dataset

`src/cnn/datasets/cnn_dataset.py` defines `SpectrogramDataset`, a thin `torch.utils.data.Dataset` wrapper:

```python
dataset = SpectrogramDataset(spectrograms, labels, cfg)   # spectrograms: (N, n_mels, T) float array
spec_tensor, label_idx = dataset[0]                        # (1, n_mels, T) tensor, int label
weights = dataset.compute_class_weights()                  # inverse-frequency weights, for nn.CrossEntropyLoss(weight=...)
train_ds, val_ds = train_val_split(dataset, train_ratio=0.8, seed=42)
```

| Output | Meaning |
|---|---|
| `spec_tensor` | `(1, n_mels, time_frames)` — a channel dimension is added so 2D convolutions treat the spectrogram like a single-channel image |
| `label_idx` | Integer class index, taken from `cfg.label_map` (e.g. `{'rest': 0, 'fist': 1, 'open': 2, 'pinch': 3}`) |

`compute_class_weights()` returns `N / (num_classes * count_i)` per class, for use with a weighted cross-entropy loss on imbalanced gesture distributions. `train_val_split()` shuffles indices with a fixed seed and splits by ratio, rebuilding two `SpectrogramDataset` instances from the resulting subsets.

---

#### Model Architecture — `EMGConvNet`

`src/cnn/models/model_cnn.py` defines a lightweight ConvNet sized for small spectrograms and CPU training:

```text
Input: (B, 1, n_mels, time_frames)
├─ ConvBlock1:  1 →  32 channels, 3×3 conv, BatchNorm, ReLU, MaxPool 2×2
├─ ConvBlock2: 32 →  64 channels, 3×3 conv, BatchNorm, ReLU, NO pooling
├─ ConvBlock3: 64 → 128 channels, 3×3 conv, BatchNorm, ReLU, NO pooling
├─ Global Average Pooling → (B, 128)
├─ Dropout (p=0.3)
└─ Fully Connected: 128 → num_classes
```

Design notes (from the source):

- **Only the first block pools.** Spectrograms this small (as narrow as 2 time frames) collapse to zero spatial size if every block pools; blocks 2 and 3 use `pool_size=1` (no-op pooling) to preserve spatial dimensions.
- **Global average pooling** before the classifier makes the network agnostic to the exact `(n_mels, time_frames)` shape, so it tolerates windows of slightly different length.
- **BatchNorm + ReLU** in every block; conv layers use `bias=False` since BatchNorm makes the bias redundant.

`model.extract_features(x)` returns the 128-dim pooled embedding before the classifier head, for embedding inspection (e.g. t-SNE) if needed later.

For `num_classes=4` (rest / fist / open / pinch), the model measures at **93,412 trainable parameters** (~0.36 MB in float32) — see [Reproduced Results](#reproduced-results-cnn) for how this was measured.

---

#### Training Pipeline

`src/cnn/models/train_cnn_dummy.py` runs the full pipeline end to end, on synthetic ("dummy") EMG data generated by `src/common/io/dummy_data.py`:

```text
Generate dummy EMG (60 s, 3 active gesture classes + rest)
        ↓
preprocess_raw(): bandpass(20–450 Hz) → rectify → lowpass(10 Hz) → z-score
        ↓
window_signal(): 200 ms windows, 100 ms hop, majority-vote labels
        ↓
batch_compute_spectrograms(): (N, 64, time_frames) log-mel spectrograms
        ↓
SpectrogramDataset + train_val_split(train_ratio=0.8)
        ↓
EMGConvNet training (CrossEntropyLoss, Adam)
        ↓
Validation metrics + confusion matrix
        ↓
Save best checkpoint + training curve / confusion matrix plots
```

| Setting | Value |
|---|---:|
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 0.001 (Adam) |
| Train / val split | 80% / 20% |
| Seed | 42 |
| Device | CPU |
| Dummy data duration | 60 s, 5 s gesture blocks |

---

#### Reproduced Results <a name="reproduced-results-cnn"></a>

The numbers and figures below come from an actual execution of the pipeline above (`src/cnn/transforms/spectrograms.py`, `src/cnn/datasets/cnn_dataset.py`, `src/cnn/models/model_cnn.py`, `src/cnn/models/train_cnn_dummy.py`, imported and run unmodified), on synthetic EMG — **not** on real biosignal recordings. Treat these as a pipeline sanity-check, not a gesture-recognition accuracy claim.

Run configuration: `seed=42`, `sample_rate_hz=1000` (see [Reproducibility Notes](#reproducibility-notes-cnn) for why the sample rate was overridden from the config file's default), `window_ms=200`, `hop_ms=100`, gestures = `rest, fist, open, pinch`.

Environment: Python 3.11.7, `numpy==1.26.4`, `torch==2.9.1+cpu`, `librosa==0.11.0`, `scikit-learn==1.7.2`, `matplotlib==3.10.7`, `scipy==1.16.3`. The exact percentages below should reproduce closely with a fixed seed on these versions; minor floating-point/BLAS differences across platforms or library versions can shift them slightly.

```text
Windows generated:        599   (rest: 299, open: 150, pinch: 150, fist: 0 — see notes)
Spectrogram shape:        (599, 64, 2)   — 64 mel bins × 2 time frames per window
Train / val split:        479 / 120
Model:                     EMGConvNet, 93,412 trainable parameters
Training time:             ~3.0 s for 20 epochs (CPU)
Best validation accuracy:  84.17% (epoch 13 of 20)
Final validation macro-F1: 0.7725
```

Classification report (final epoch, validation set, `zero_division=0`):

```text
              precision    recall  f1-score   support

        rest       0.97      0.97      0.97        64
        fist       0.00      0.00      0.00         0
        open       0.62      0.64      0.63        25
       pinch       0.73      0.71      0.72        31

    accuracy                           0.83       120
   macro avg       0.58      0.58      0.58       120
weighted avg       0.83      0.83      0.83       120
```

The `fist` row has zero support: with `seed=42`, none of the twelve 5-second blocks drawn for this 60-second run happened to be labelled `fist`, so the class is absent from the entire dataset, not just the validation split. This is a property of the random block sampler in `generate_dummy_emg` over a short 60 s draw, not a training failure — macro-F1 is computed over all four configured classes regardless.

<p align="center">
  <img src="assets/cnn/training_curves.png" alt="CNN training and validation loss/accuracy curves" width="800">
</p>
<p align="center"><em>Training/validation loss and accuracy over 20 epochs.</em></p>

<p align="center">
  <img src="assets/cnn/confusion_matrix.png" alt="CNN validation confusion matrix" width="550">
</p>
<p align="center"><em>Validation confusion matrix (rest / fist / open / pinch). Rest is separated cleanly; open and pinch are the main source of confusion, consistent with both being lower-amplitude, adjacent gesture classes in the synthetic generator.</em></p>

---

#### Spectrogram Examples

<p align="center">
  <img src="assets/cnn/spectrogram_gallery_per_class.png" alt="Log-mel spectrograms per gesture class, raw vs preprocessed" width="850">
</p>
<p align="center"><em>Log-mel spectrograms of one representative 5-second gesture block per class, computed with <code>compute_stft_spectrogram</code> (fs=1000 Hz, n_fft=256, hop=128, n_mels=64). Left column: spectrogram of the raw synthetic EMG. Right column: spectrogram of the same block after the pipeline's bandpass → rectify → lowpass(10 Hz) → z-score preprocessing.</em></p>

<p align="center">
  <img src="assets/cnn/model_input_spectrograms.png" alt="Actual (64, 2) CNN input spectrograms per class" width="850">
</p>
<p align="center"><em>The actual tensors the CNN sees: one 200 ms window per class, shape (64 mel bins, 2 time frames) — this is the true input resolution of <code>EMGConvNet</code> under the current windowing configuration.</em></p>

**What the preprocessed spectrograms show.** After `preprocess_raw`'s lowpass at 10 Hz, almost all remaining signal energy sits below the mel filterbank's 20 Hz floor, so the classifier is not looking at raw EMG frequency content (20–450 Hz muscle activation spectrum) in these spectrograms — it is looking at how the low-frequency envelope leaks into the lowest mel bins, and how that leakage is modulated in time as the synthetic gesture bursts turn on and off. This is a legitimate, learnable signal (it is what separates the classes above chance), but it should not be read as "the CNN sees EMG frequency content" — see the note below.

This was checked quantitatively, not just visually: the 4th-order Butterworth lowpass at 10 Hz used in `preprocess_raw` attenuates by **−24 dB at 20 Hz**, **−56 dB at 50 Hz**, and **−184 dB at 450 Hz** (computed via `scipy.signal.freqz`). By the time a signal reaches the mel filterbank's 20 Hz floor, over 99.5% of its power (in linear terms, a gain of ≈0.06) has already been removed, and content near the bandpass's original 450 Hz edge is gone entirely.

---

#### Reproducibility Notes <a name="reproducibility-notes-cnn"></a>

Two implementation details had to be worked around to get an end-to-end run out of the current code, and are recorded here rather than silently patched:

1. **Sample rate / bandpass mismatch.** `configs/preprocessing.yaml` currently sets `sample_rate_hz: 100`, but `train_cnn_dummy.py` preprocesses with a fixed `band=(20, 450)` Hz Butterworth bandpass. A bandpass filter requires `high < Nyquist = fs/2`; at `fs=100 Hz` the Nyquist frequency is 50 Hz, so `high=450` is invalid and `preprocess_raw` raises `ValueError: Invalid bandpass... Ensure 0 < low < high < fs/2`. The spectrogram module's own docstrings/self-tests and `src/cnn/README.md` both assume `fs=1000 Hz` ("200 samples @ 1kHz"), so the results above were produced with `sample_rate_hz` overridden to `1000` on the config object passed into the pipeline. Nothing else was changed. Before running the CNN track from `configs/preprocessing.yaml` as-is, either raise `sample_rate_hz` to something compatible with a 450 Hz bandpass edge (≥ ~1000 Hz), or lower the bandpass/window parameters to match a 100 Hz signal.
2. **`train_val_split` reloads config from a relative path.** `src/cnn/datasets/cnn_dataset.py`'s `train_val_split()` always calls `SpectrogramDataset(..., cfg=None, ...)`, which makes the dataset reload configuration via `load_cfg("configs/preprocessing.yaml")` — a path relative to the current working directory, not the repo root. If the training script is invoked from any directory other than the project root, this silently falls back to the hardcoded defaults in `schemas.py` (which have no `label_map`/`gestures`), and raises `AttributeError: 'SimpleNamespace' object has no attribute 'label_map'`. Run training scripts from the repository root until this is fixed.
3. **`n_fft=256` on a 200-sample window.** `librosa` emits `UserWarning: n_fft=256 is too large for input signal of length=200` for every window, since the FFT size exceeds the window length. `librosa` zero-pads internally so the call still succeeds, but it means part of each spectrogram's frequency resolution is coming from zero-padding rather than real samples — worth reducing `n_fft` (e.g. to 128 or 64) if this is revisited.
4. **Dummy data only.** All figures and metrics above come from `generate_dummy_emg`'s synthetic, band-limited noise bursts, not from recorded EMG. They validate that the spectrogram → dataset → model → training loop runs correctly and can separate synthetic classes above chance; they say nothing about real-world gesture classification accuracy.

---

#### Why This Track Matters

The CNN track is useful because it removes the need to hand-design frequency-domain descriptors: the mel filterbank and convolutional layers learn which time-frequency patterns are discriminative directly from data, rather than relying on the fixed feature set used by the classic ML track. Its main current limitation is upstream of the model — the shared preprocessing pipeline low-pass filters the signal to a 10 Hz envelope before the spectrogram is computed, which limits how much genuine EMG spectral information (as opposed to envelope-modulation timing) the CNN actually has access to. Feeding the CNN track a less aggressively low-passed signal — e.g. the rectified, band-passed signal without the 10 Hz envelope step — is the most direct way to let it exploit real time-frequency muscle-activation structure once real EMG recordings are available.

---

### 3. RNN / BRNN — Temporal Sequence Models

The RNN track models **temporal dependencies across consecutive EMG windows**. Instead of treating each signal window as an isolated sample, this pipeline converts windowed EMG recordings into sequences of feature vectors and trains recurrent neural networks to classify the user's intended movement.

This is especially relevant for prosthetic control because muscle activation patterns evolve over time. A single EMG window may be noisy or ambiguous, while a short sequence of windows can provide a clearer representation of the intended gesture.

```text
Windowed EMG Signal
        ↓
Sequential Feature Extraction
        ↓
Feature Standardization
        ↓
FeatureSequenceDataset
        ↓
GRU / LSTM / Bidirectional Recurrent Model
        ↓
Intent Class Prediction
```

#### Sequence Feature Extraction

The file `src/rnn/features/seq_features.py` contains the feature-extraction logic used before training the recurrent models.

The function `compute_seq_features()` converts each EMG window into a compact numerical feature vector. These feature vectors are then arranged into temporal sequences for RNN-based classification.

The current feature extraction supports multiple groups of descriptors:

| Feature group | Description |
|---|---|
| Statistical features | Mean, standard deviation, minimum, maximum, range, and zero-crossing rate |
| Shape-based features | Signal slope, skewness, and kurtosis-style descriptors |
| Spectral features | Frequency-domain information such as centroid, bandwidth, and spectral power |

In the RNN training pipeline, spectral features are enabled when the feature vectors are generated:

```python
ft_vectors, ft_names = compute_seq_features(
    windows,
    sample_rate,
    spectral_feat=True
)
```

This produces a feature matrix where each row corresponds to one EMG window and each column corresponds to a calculated feature.

#### Sequence Dataset

The file `src/rnn/datasets/sequence_dataset.py` defines the `FeatureSequenceDataset` class.

This dataset groups consecutive EMG feature vectors into fixed-length temporal sequences. Each sequence becomes one training sample for the recurrent model.

```text
Feature vector 1
Feature vector 2
Feature vector 3
...
Feature vector N
        ↓
Sequence of consecutive feature vectors
        ↓
RNN input sample
```

Each dataset sample contains:

| Output | Meaning |
|---|---|
| `x` | Input tensor with shape `[sequence_length, feature_dim]` |
| `y_seq` | Labels for the windows inside the sequence |
| `y` | Majority label for the sequence |
| `length` | Actual sequence length |
| `names` | Feature names used in the input vector |

The current RNN training setup uses overlapping temporal sequences, allowing the model to learn how EMG features change over time.

#### Feature Standardization

Before training, the feature vectors are standardized using a custom `Standardizer`.

Standardization transforms the input features so that each feature has a comparable scale:

```text
standardized_feature = (feature - mean) / standard_deviation
```

This is important because recurrent neural networks are sensitive to feature scale. Without standardization, features with larger numeric values can dominate the learning process.

The trained model artifact stores the standardization parameters together with the model, making it possible to apply the same transformation later during inference.

Stored metadata includes:

```text
standardizer_mean
standardizer_std
feature_names
label_map
```

#### Recurrent Model Architecture

The file `src/rnn/models/model_rnn.py` defines the recurrent model architectures used for temporal EMG classification.

The main recurrent models are:

| Model | Purpose |
|---|---|
| `GRUModel` | Gated Recurrent Unit model for temporal EMG classification |
| `LSTMModel` | Long Short-Term Memory model for temporal EMG classification |

The recurrent architecture follows this structure:

```text
Input sequence: [batch_size, sequence_length, feature_dim]
        ↓
GRU / LSTM recurrent layers
        ↓
Last valid hidden state
        ↓
Fully connected classification layer
        ↓
Class logits
```

The models support:

- Configurable hidden dimension.
- Multiple recurrent layers.
- Dropout.
- Optional bidirectional recurrent processing.
- Variable sequence lengths through the `lengths` argument.

The RNN track is therefore suitable for both simple temporal baselines and more advanced bidirectional sequence models.

#### Training Pipeline

The training script is located at:

```text
src/rnn/models/train_rnn.py
```

The script performs the complete RNN training workflow:

```text
Load preprocessing configuration
        ↓
Generate or load EMG data
        ↓
Apply shared windowing pipeline
        ↓
Extract sequential features
        ↓
Encode gesture labels
        ↓
Standardize feature vectors
        ↓
Build FeatureSequenceDataset
        ↓
Split into training and validation sets
        ↓
Train recurrent model
        ↓
Evaluate validation performance
        ↓
Save model artifact
        ↓
Generate training diagnostics
```

The training loop uses:

- Cross-entropy loss.
- Adam optimization.
- Training and validation split.
- Best-model checkpointing.
- Validation accuracy tracking.
- Confusion-matrix evaluation.

The script also selects the best available compute device, supporting CPU, CUDA, and Apple Silicon acceleration where available.

#### Output Artifact

After training, the RNN pipeline saves a model artifact that contains both the trained weights and the metadata required for reuse.

The saved artifact includes:

```text
state_dict
model_type
input_dim
hidden_dim
num_layers
bidirectional
dropout
label_map
feature_names
standardizer_mean
standardizer_std
```

This makes the trained model easier to reload for later testing, comparison, or future real-time prosthetic-arm inference.

#### Reproduced Results

The numbers below come from `artifacts/best_gru_2026-07-07_19-24-24.pt` (Bi-GRU, `seq_length=32`, `hidden_dim=128`, 48 features — 24 base + delta features, 2 layers, bidirectional), measured on 48,898 validation sequences (Exercise 2, repetitions 9–10, channels 0/1, 18 classes: 17 gestures + rest).

| Metric | Value |
| --- | ---: |
| Overall accuracy | 84.5% |
| Macro F1 | 0.737 |
| Rest share of validation set | 54.4% |
| Rest-only accuracy | 96.8% |
| **Accuracy excluding rest (17 gestures only)** | **70.1%** |
| Per-class F1 range (17 gestures) | 0.679 – 0.794 |

**Read the 84.5% carefully — rest makes up 54.4% of the validation set and is classified at 96.8% accuracy** (resting EMG, near-zero muscle activity, is naturally easy to separate from any gesture). The number that actually reflects gesture-recognition ability is **70.1% excluding rest**. All 17 gesture classes land in a tight 0.68–0.79 F1 band, with no class collapsing toward zero — this is the strongest evidence that the model is genuinely learning per-gesture structure rather than just calling "rest" most of the time.

See [Model Comparison: Classic ML vs RNN](#model-comparison-classic-ml-vs-rnn) below for the full caveats (reduced channel count, rest-class weighting) before treating either figure as final.

<p align="center">
  <img src="assets/rnn/rnn_tsne_embeddings.png" alt="t-SNE of GRU sequence embeddings" width="48%">
  <img src="assets/rnn/rnn_umap_embeddings.png" alt="UMAP (3D) of GRU sequence embeddings" width="48%">
</p>
<p align="center">
  <em>t-SNE of the GRU's learned sequence embeddings (validation set)</em> &nbsp;&nbsp;&nbsp;&nbsp; <em>UMAP (3D) of the same embeddings</em>
</p>

<p align="center">
  <img src="assets/rnn/rnn_confusion_matrix.png" alt="GRU validation confusion matrix" width="48%">
  <img src="assets/rnn/rnn_per_class_f1.png" alt="GRU per-class F1 scores" width="48%">
</p>
<p align="center">
  <em>Validation confusion matrix (row-normalized)</em> &nbsp;&nbsp;&nbsp;&nbsp; <em>Per-class F1, sorted, with macro F1 reference line</em>
</p>

#### Why This Track Matters

The RNN/BRNN track is important because prosthetic control is inherently temporal. Muscle activation is not just defined by one instant of signal activity, but by the pattern of activation across time.

Compared with classic machine learning, the recurrent approach can capture temporal movement dynamics. Compared with CNN-based spectrogram classification, it focuses directly on the evolution of extracted EMG features across consecutive windows.

This makes the RNN/BRNN track a strong candidate for future real-time EEG/EMG prosthetic control, where stable and responsive intent prediction is essential.

---

## Model Comparison: Classic ML vs RNN

This section compares the **Classic ML track** against the **RNN/BRNN track** on the same evaluation split — Exercise 2, validation repetitions 9–10, 2 EMG channels (ch0, ch1), 18 gesture classes (17 gestures + rest).

Only the current best RNN checkpoint is shown below (`artifacts/best_gru_2026-07-07_19-24-24.pt`). Earlier GRU runs (a 62.6%-accuracy baseline and a 73.5%-accuracy "improved" run) are superseded and omitted — the 73.5% checkpoint is also no longer usable (corrupted on disk after training, unrelated to modeling).

### Headline Results

| Model | Val Accuracy | Macro F1 | Notes |
| --- | ---: | ---: | --- |
| Logistic Regression | 43.3% | 0.104 | Linear baseline |
| XGBoost | 44.9% | 0.142 | Gradient-boosted trees |
| Random Forest | 45.0% | 0.145 | Strongest classic ML baseline |
| **Bi-GRU** (`best_gru_2026-07-07_19-24-24.pt`) | **84.5%** | **0.737** | `seq_length=32`, `hidden_dim=128`, 48 features (24 base + delta features), bidirectional, 2 layers |

Measured directly via `rnn_metrics_check.py` on 48,898 validation sequences (not just the checkpoint's self-reported `best_val_acc`):

| Metric | Value |
| --- | ---: |
| Overall accuracy | 84.5% |
| Macro F1 | 0.737 |
| Rest share of validation set | 54.4% |
| Rest-only accuracy | 96.8% |
| **Accuracy excluding rest (17 gestures only)** | **70.1%** |
| Per-class F1 range (17 gestures) | 0.679 – 0.794 |

### Why the RNN Wins

- **Temporal context.** The RNN consumes a *sequence* of 32 consecutive feature windows, letting it learn how muscle activation evolves over time. Classic ML classifies a single window in isolation, with no memory of what came before.
- **Delta features.** 24 of the 48 input features are frame-to-frame deltas of the base statistics, giving the model explicit short-term dynamics instead of forcing it to infer them purely from the raw sequence.
- **Macro F1 confirms this isn't just a rest-class effect.** All 17 gesture classes land in a tight 0.68–0.79 F1 band — no class collapses to near-zero the way several classic ML classes effectively do (see below). The RNN's advantage holds up per-class, not just in aggregate accuracy.

### Where Classic ML Still Has an Edge

Classic ML is not without merit — with a confidence-threshold rejection rule (`tau=0.6`), it reaches **76.8% accuracy on accepted predictions**, higher than the GRU's rest-excluded 70.1%. The tradeoff is that it abstains (rejects) **56% of windows** to get there, i.e. it only commits to a prediction when it's confident, at the cost of coverage. For a real-time prosthetic control signal, this reject-and-wait behavior may or may not be acceptable depending on the target application — it trades responsiveness for reliability.

Classic ML's low macro-F1 (~0.10–0.15) across the board also indicates it performs close to random on many of the 18 gesture classes, even where its raw accuracy looks reasonable — a sign of class imbalance combined with the lack of temporal context making rare/similar gestures hard to separate from single windows. This is the direct contrast to the RNN's 0.737 macro F1.

### Known Caveats — Read Before Trusting the 84.5% Number

These were raised and checked directly against this checkpoint; recorded here rather than silently smoothed over.

1. **84.5% overall accuracy is inflated by the rest class.** Rest makes up 54.4% of the validation set and is classified at 96.8% accuracy (it's an easy class — near-zero muscle activity is naturally distinct from any gesture). The number that actually reflects gesture-recognition ability is **70.1% excluding rest**. Report that number, not 84.5%, when comparing against other gesture-recognition work.
2. **Only 2 of the NinaPro channels are used** (channels 0 and 1), not the full electrode array. The 70.1% rest-excluded, 17-class accuracy should be read in that context — it's a reduced-channel result, and would need to be re-benchmarked against full-channel classic ML/RNN runs (not done here) to know how much accuracy is being left on the table by dropping channels.

### Summary

| Property | Classic ML | RNN (Bi-GRU) |
| --- | --- | --- |
| Best accuracy | 45.0% raw (RF), 76.8% with `tau=0.6` reject option | 84.5% raw, 70.1% excluding rest, no rejection |
| Macro F1 | ~0.10–0.15 (weak — several classes near-random) | 0.737 (all 17 gesture classes in 0.68–0.79 band) |
| Temporal context | None (single window) | Yes (32-window sequence) |
| Coverage | Full, or ~44% with `tau=0.6` rejection | Full (always predicts) |
| Reproducibility | Not re-measured across runs | ~11pp run-to-run spread observed (73.5–84.5%) — not yet stabilized |
| Interpretability | High | Low |
| Best use case | Fast baseline, debugging feature quality | Primary model for continuous real-time control |

**Recommendation:** the RNN track (48-feature delta configuration, `seq_length=32`, `hidden_dim=128`) is the stronger model for prosthetic control — its macro F1 and rest-excluded accuracy both clearly beat classic ML's equivalents, and it doesn't need to reject ambiguous windows to do it. That said, treat 84.5% as an optimistic single-run number: the honest headline figure is **~70% on the 17 real gesture classes**. Classic ML remains useful as a fast, interpretable sanity check on feature separability and data quality before committing to RNN training runs.

---

## Evaluation Protocol

The project can be evaluated using two main modes.

### Within-subject evaluation

The model is trained and tested on data from the same subject, with trials split into training, validation, and test sets.

This measures how well the system can adapt to a specific user.

### Cross-subject evaluation

The model is trained on some subjects and tested on unseen subjects.

This measures how well the system generalizes across users.

---

## Metrics

Recommended metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Macro-F1 | Balanced performance across classes |
| Per-class F1 | Gesture-specific reliability |
| Confusion matrix | Error analysis between classes |
| Latency | Real-time control feasibility |
| Throughput | Inference speed |

The most important metric is **Macro-F1**, because prosthetic control requires reliable performance across all movement classes, not only the most frequent class.

---

## Project Demo

The project includes a physical prosthetic robotic arm prototype used for demonstrations and experimentation.

Demo media can be stored in:

```text
assets/demo/
```

Example:

```md
<p align="center">
  <img src="assets/demo/prosthetic_arm_demo.gif" alt="Prosthetic arm demo" width="700">
</p>
```

Suggested demo caption:

```md
<p align="center">
  <em>Prototype demonstration of biosignal-based prosthetic arm control.</em>
</p>
```

---

## Reproducibility

For reproducible experiments, each run should save:

```text
run.json
config.yaml
metrics.json
confusion_matrix.png
model_weights.pt
```

Recommended information to store:

- Git commit hash.
- Random seed.
- Dataset version.
- Subject split.
- Preprocessing configuration.
- Model hyperparameters.
- Training duration.
- Evaluation metrics.

This makes experiments easier to compare across model tracks.

---

## Roadmap

Planned and potential future improvements:

- [ ] Add complete real-time serial data logger.
- [ ] Add EEG channel support in the common schema.
- [ ] Add multi-channel EMG support.
- [ ] Add synchronized EEG/EMG acquisition.
- [ ] Add live inference with rolling windows.
- [ ] Export trained models to ONNX.
- [ ] Benchmark inference latency on CPU and embedded hardware.
- [ ] Integrate predictions with prosthetic arm control.
- [ ] Add grip-force feedback.
- [ ] Add more presentation photos and demo videos.
- [ ] Improve cross-subject evaluation.
- [ ] Add automated tests for preprocessing and windowing.

---

## Project Status

This repository is currently a **research prototype**.

Current focus:

- Biosignal preprocessing.
- EMG windowing.
- Gesture / intent classification.
- Comparison of classic ML, CNN, and RNN-based approaches.
- Prosthetic arm control integration.

Not yet intended for:

- Clinical use.
- Medical diagnosis.
- Commercial prosthetic deployment.
- Safety-critical autonomous control.

---

## Team

Project developed by students and researchers interested in:

- Biosignal processing
- Machine learning
- EEG/EMG interfaces
- Assistive robotics
- Prosthetic arm control
- Human-machine interaction

Main repository maintainer:

```text
Hugo Arsénio
GitHub: @Hugo132645
```

Team members:

```md
- Tudor-Andrei Dolineaschi — Classic ML
- Maria Daria Dejeu — CNN
- Norbert Cesar — RNN/BRNN
```

---

## License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.

---

## Disclaimer

This project is an educational and research prototype. It is not a certified medical device and should not be used for clinical, diagnostic, or safety-critical applications without proper validation, regulation, and expert supervision.
