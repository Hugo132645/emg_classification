# EMG Prosthetic Arm — Intent Classification

End-to-end pipeline for EMG-based gesture recognition to control a prosthetic robotic arm (MyoWare 2.0).  
Three modeling tracks run in parallel and share a common data layer:

- Classic ML (feature-based)
- CNN (time–frequency images)
- RNN/BRNN (temporal sequences)

This repository is structured so every team uses the same schema, preprocessing, and windowing. You can develop and validate models on dummy data before hardware arrives.

---

## Quick Start

1) Clone the repository and create a Python 3.10 virtual environment.
2) Install dependencies from `requirements.txt`.
3) Implement your components in the folders described below.

```bash
git clone https://github.com/<org>/emg-prosthetic.git
cd emg-prosthetic

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Repository Structure

```
emg-prosthetic/
├─ configs/
│  ├─ preprocessing.yaml            # Tunables: fs, filters, windowing, normalization, gestures
│  └─ gestures.yaml                 # Class list & trial protocol (optional)
├─ data/
│  ├─ raw/                          # {subject}/{date}/session_{id}.parquet (to be collected)
│  └─ processed/                    # Windows/features ready for models
├─ src/
│  ├─ common/                       # Shared by all model tracks
│  │  ├─ __init__.py
│  │  ├─ io/
│  │  │  ├─ schemas.py              # Constants, file patterns, class names
│  │  │  └─ dummy_data.py           # Synthetic EMG generator (raw & envelope modes)
│  │  ├─ preprocessing/
│  │  │  ├─ pipelines.py            # preprocess_raw / preprocess_envelope
│  │  │  └─ windowing.py            # window_signal() + majority label
│  │  └─ utils/
│  │     └─ config.py               # YAML loader/helper
│  ├─ classic_ml/                   # Feature-based models
│  │  ├─ features/
│  │  │  ├─ time_domain.py
│  │  │  └─ freq_domain.py
│  │  ├─ datasets/
│  │  │  └─ classic_ml_dataset.py
│  │  └─ models/
│  │     └─ train_classic.py
│  ├─ cnn/                          # Time–frequency image models
│  │  ├─ transforms/
│  │  │  └─ spectrograms.py
│  │  ├─ datasets/
│  │  │  └─ cnn_dataset.py
│  │  └─ models/
│  │     ├─ model_cnn.py
│  │     └─ train_cnn.py
│  └─ rnn/                          # Temporal sequence models
│     ├─ features/
│     │  └─ seq_features.py
│     ├─ datasets/
│     │  └─ sequence_dataset.py
│     └─ models/
│        ├─ model_rnn.py
│        └─ train_rnn.py
├─ tests/                           # Unit tests for common + per-track modules
├─ requirements.txt
└─ README.md
```

Note: The specific training/evaluation scripts are intentionally omitted here. Teams are expected to implement their own CLIs or notebooks in their respective folders using the shared common layer. The team members are also free to adjust the structure depending on what suits their needs.

---

## Requirements

Use Python 3.10 (macOS/Windows/Linux). Suggested `requirements.txt` below pins versions that are widely compatible on CPU-only environments and work well together for signal processing and training.

```
numpy==1.24.4
pandas==2.2.2
pyarrow==16.1.0
scipy==1.11.4
scikit-learn==1.4.2
matplotlib==3.8.4
librosa==0.10.1
torch==2.2.2
torchvision==0.17.2
onnx==1.16.0
onnxruntime==1.18.0
tqdm==4.66.4
pyyaml==6.0.2
```

If you are on Apple Silicon, these CPU wheels work; GPU acceleration is optional and not required for baseline development.

NOTE: This might be adjusted later in a requirements.txt file.

---

## Data Schema & Conventions

Primary format: Parquet (fast I/O, typed columns) with an optional mirrored CSV.  
Path template: `data/raw/{subject_id}/{session_date}/session_{session_id}.parquet`

Columns:
- `timestamp_ms` (int64)
- `emg_ch1_raw` (int16) — raw if available
- `emg_ch1_env` (float32) — envelope if available
- `label` (string) — one of classes in `configs/gestures.yaml`
- `subject_id` (string), `session_id` (string), `trial_id` (int)
- `marker` (int; optional cue onset flag)
- `notes` (string; optional)

Windowing (from `src/common/preprocessing/windowing.py`):
- Defaults: `WINDOW_MS = 200`, `HOP_MS = 100` (configurable via YAML)
- Returns:
  - `windows`: `(N, W)` float32
  - `labels`: list[str] length `N`
  - `times_ms`: `(N, 2)` start/end per window

---

## Shared Common Layer

### `schemas.py`
Centralizes constants: sampling rates, window/hop, gestures, filter params, and file patterns.

### `pipelines.py`
- `preprocess_raw(x, fs, band=(20,450), notch=50, envelope_lp=10, normalize="zscore_session")`  
  Steps: band-pass -> optional notch -> rectify -> low-pass envelope -> normalize.  
  Returns: processed signal (same length) + metadata dict.
- `preprocess_envelope(x, fs, lp=10, normalize="zscore_session")`  
  Steps: low-pass (light) -> normalize.  
  Returns: processed signal (same length) + metadata dict.

### `windowing.py`
- `window_signal(x, fs, win_ms, hop_ms, labels=None, timestamps_ms=None)`  
  Majority-vote window labels; returns `(windows, labels, times_ms)`.

### `dummy_data.py`
- `generate_dummy_emg(seconds, fs, classes, block_s=5, mode="raw"|"envelope")`  
  EMG-like segments with class blocks and noise (to train before hardware arrives).

### `utils/config.py`
Minimal YAML loader (via `pyyaml`).

---

## Modeling Tracks

### Classic ML (feature-based)
- Feature sets per 200 ms window: RMS, MAV, SD, VAR, WL, ZC, SSC, WAMP, Hjorth parameters, AR(4).
- Frequency features: Welch PSD, mean/median freq, band powers (20–150, 150–350 Hz), spectral entropy.
- Baseline models: Logistic Regression (L2), Linear/RBF-SVM, Random Forest/XGBoost.
- Deliverables: scaler + model artifact, confusion matrix, macro-F1, latency estimate.

### CNN (time–frequency images)
- Inputs: log-mel spectrograms (STFT 256 @ 1 kHz, hop 128, 64 mel bins) or CWT scalograms.
- Architectures: MobileNetV3-Small or a 3–4 block ConvNet with GAP.
- Augmentations: additive noise, light time/freq masking.
- Deliverables: ONNX export, macro-F1, inference throughput on CPU.

### RNN/BRNN (temporal sequences)
- Inputs: sequence of feature vectors at 10 Hz over 1–2 s context or raw/envelope with 1D Conv front-end.
- Architectures: Bi-LSTM/GRU (128×2), optional attention.
- Deliverables: ONNX export, macro-F1, decision latency (rolling window inference).

---

## Evaluation Protocol

Two evaluation modes:
1. Within-subject (session split) — user-specific tuning performance.
2. Cross-subject (LOSO) — generalization across people.

Hold-out policy (typical):
- Train 70%, Val 15%, Test 15% (split by trial, not random rows).

Metrics:
- Primary: Macro-F1.
- Secondary: Accuracy, per-class F1, latency (ms), throughput (Hz).
- Reporting: confusion matrices, per-class F1, latency and throughput tables.

---

## Data Collection (when hardware arrives)

- Sensor: MyoWare 2.0
- ADC/MCU: Arduino/Teensy/ESP32 at 1 kHz (raw) or 500 Hz (envelope)
- Labeling: On-screen cue with digital `marker` pin toggled at onset
- Trial protocol per gesture:
  - 5 s rest → cue → 3 s hold → 2 s relax
  - 10 repetitions per gesture
  - 2 sessions/day for 3 days
- Subjects: At least 5; consistent electrode placement per subject.
- Output: Parquet files following the schema above (`data/raw/...`).

---

## Versioning and Reproducibility

- Pin dependencies in `requirements.txt`.
- Save trained artifacts under each track’s `exports/` with a run manifest:
  - `run.json` containing git commit, config hash, seed, split description.
- Recommended seeds and determinism flags for PyTorch where applicable.

---

## Development

Unit tests (recommend `pytest`) should cover:
- Filtering correctness (pass/stopband checks on synthetic inputs)
- Window shapes and label majority logic
- Dataset shapes for each model track

Style:
- Black and isort for formatting and imports.

Git workflow:
- Branch naming: `feature/<area>-<short-desc>`
- Pull requests: include a brief description, sample run output, and tests where relevant.

---

## Roadmap

- Serial logger and Parquet writer for live collection
- Real-time inference demo (rolling buffer, 100 ms updates)
- ONNXRuntime micro-benchmarks on target hardware
- Force/Grip control integration (downstream controller)
- Multi-channel EMG support

---

## Contributing

1. Open an issue describing the change.
2. Create a feature branch and include tests.
3. Submit a PR and tag reviewers.

---

## License

MIT
