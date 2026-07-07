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
  <em>Project presentation and live demonstration booth.</em>
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
- [Evaluation Protocol](#evaluation-protocol)
- [Metrics](#metrics)
- [Data Collection](#data-collection)
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
│   ├── preprocessing.yaml            # Sampling, filtering, windowing and normalization settings
│   └── gestures.yaml                 # Gesture classes and trial protocol
│
├── src/
│   ├── common/
│   │   ├── io/
│   │   │   ├── schemas.py            # Shared constants, schemas and file patterns
│   │   │   └── dummy_data.py         # Synthetic EMG generation for testing
│   │   │
│   │   ├── preprocessing/
│   │   │   ├── pipelines.py          # Raw/envelope preprocessing functions
│   │   │   └── windowing.py          # Signal windowing and majority-label logic
│   │   │
│   │   └── utils/
│   │       └── config.py             # YAML configuration loader
│   │
│   ├── classic_ml/
│   │   ├── features/
│   │   │   ├── time_domain.py        # Time-domain EMG features
│   │   │   └── freq_domain.py        # Frequency-domain EMG features
│   │   │
│   │   ├── datasets/
│   │   │   └── classic_ml_dataset.py # Dataset preparation for classic ML
│   │   │
│   │   └── models/
│   │       └── train_classic.py      # Classic ML training script
│   │
│   ├── cnn/
│   │   ├── transforms/
│   │   │   └── spectrograms.py       # Spectrogram generation
│   │   │
│   │   ├── datasets/
│   │   │   └── cnn_dataset.py        # Dataset preparation for CNN models
│   │   │
│   │   └── models/
│   │       ├── model_cnn.py          # CNN architecture
│   │       └── train_cnn.py          # CNN training script
│   │
│   └── rnn/
│       ├── features/
│       │   └── seq_features.py       # Sequence feature extraction
│       │
│       ├── datasets/
│       │   └── sequence_dataset.py   # Dataset preparation for RNN models
│       │
│       └── models/
│           ├── model_rnn.py          # RNN / BRNN architecture
│           └── train_rnn.py          # RNN training script
│
├── assets/
│   ├── demo/
│   │   ├── prosthetic_arm_demo.gif
│   │   └── prosthetic_arm_demo_frame.jpg
│   │
│   └── presentations/
│       └── 20260213_114047.jpg
│
├── requirements.txt
├── LICENSE
└── README.md
```

The repository currently contains the main project folders `configs`, `src`, `requirements.txt`, `LICENSE`, and `README.md`. The `assets/` folder can be added to include photos, demo GIFs, and presentation material.

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

The classic ML pipeline uses hand-crafted features extracted from each EMG window.

Typical time-domain features include:

- RMS — Root Mean Square
- MAV — Mean Absolute Value
- Standard deviation
- Variance
- Waveform length
- Zero crossings
- Slope sign changes
- Willison amplitude
- Hjorth parameters
- Autoregressive coefficients

Typical frequency-domain features include:

- Welch power spectral density
- Mean frequency
- Median frequency
- Band powers
- Spectral entropy

Possible models:

- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosting / XGBoost

This track is useful because it is interpretable, fast to train, and suitable as a baseline.

---

### 2. CNN — Time-Frequency Classification

The CNN track transforms biosignal windows into time-frequency representations such as spectrograms.

```text
EMG Window
   ↓
Spectrogram / Scalogram
   ↓
CNN
   ↓
Intent Class
```

This approach allows the model to learn spatial patterns in the time-frequency domain.

Potential architectures:

- Lightweight custom ConvNet
- MobileNet-style compact CNN
- CNN with global average pooling

Potential augmentations:

- Additive noise
- Time masking
- Frequency masking
- Small amplitude perturbations

This track is useful for learning signal patterns that may not be captured by hand-crafted features.

---

### 3. RNN / BRNN — Temporal Sequence Models

The RNN track models temporal dependencies across multiple windows.

```text
Sequence of EMG Features
          ↓
RNN / GRU / LSTM / BiLSTM
          ↓
Intent Class
```

Possible architectures:

- RNN
- GRU
- LSTM
- Bidirectional LSTM
- Optional attention layer

This track is useful for modelling movement dynamics over time rather than classifying each window independently.

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

## Data Collection

The project is designed around biosignal acquisition for prosthetic control.

Example EMG setup:

| Component | Example |
|---|---|
| EMG sensor | MyoWare 2.0 |
| Microcontroller | Arduino / Teensy / ESP32 |
| Sampling rate | 500 Hz to 1 kHz |
| Signal type | Raw EMG or envelope |
| Output format | Parquet / CSV |

Example trial protocol:

```text
5 s rest
↓
Cue appears
↓
3 s gesture hold
↓
2 s relaxation
↓
Repeat for each class
```

Recommended protocol:

- Multiple repetitions per gesture.
- Multiple sessions per subject.
- Consistent electrode placement.
- Clear trial markers.
- Rest periods to reduce fatigue.
- Notes for sensor placement, noise, or failed trials.

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
- Norbert Ceaser — RNN/BRNN
```

---

## License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.

---

## Disclaimer

This project is an educational and research prototype. It is not a certified medical device and should not be used for clinical, diagnostic, or safety-critical applications without proper validation, regulation, and expert supervision.
