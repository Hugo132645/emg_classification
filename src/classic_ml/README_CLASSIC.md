## Classic ML – EMG Gesture Classification

This file implements the feature-based classical machine-learning pipeline for EMG gesture recognition. It uses time-domain and frequency-domain features together with classical ML models. I have not pushed the exports(trained models) and reports(visual graphs) to not overload the repository. In this case the classical ML models compute pretty fast anyway even for longer time periods of dummy data.
The real EMG dataset function is not yet fully functional as the parameters for the path are brute coded inside. The result are accurate and the models do not overfit/underfit. A weird thing happens though and I do not know its meaning. In the PCA plot sometimes clusters are formed, sometimes they overlap and there is always one point randomly placed in the corner of the graph.

## Folder Structure:

classic_ml/ \
├── features/ \
│ ├── time_domain.py \
│ └── freq_domain.py \
├── datasets/ \
│ └── classic_ml_dataset.py \
└── models/ \
└── train_classic.py \
All shared preprocessing (windowing, filtering, dummy data, config loading) lives in src/common.

## Pipeline Overview:

Load configuration (sample rate, filters, windowing, gestures).
Load data: dummy EMG or real EMG (raw or envelope).
Window the signal using majority-vote labeling.
Extract features: time, frequency, or both (time+freq).
Scale features with StandardScaler.
Train four models: Logistic Regression, Linear SVM, RBF SVM, Random Forest.
Select best model using macro-F1.
Generate visual plots.
Save all artifacts and metadata.

## Features:

Time-domain: \
MAV \
RMS \
VAR \
SD \
Waveform Length \
Zero Crossings \
Slope Sign Changes \
Willison Amplitude \
Hjorth Activity / Mobility / Complexity \
AR(4) coefficients \
Frequency-domain (Welch PSD): \
Bandpower 20–150 Hz \
Bandpower 150–350 Hz \
Mean frequency \
Median frequency \
Spectral entropy 

## Default mode:

time+freq — concatenation of both sets.
Visualizations (Auto-Generated)

##The training script saves the following plots in reports/:

Confusion matrix of the best model \
Confusion matrices of all models \
Model comparison bar chart (macro-F1) \
Random Forest feature-importance plot \
Per-class F1 bar plot \
PCA 2D feature-space visualization 

##Output Artifacts

Saved under:
exports/classic_ml/{subject}/{date}/
Includes: \
Best model (.joblib) \
Scaler \
run.json metadata \
Per-model artifacts (one file per classifier) \
All plots (PNG) \
Results table (.csv) 

## How to Run

From the repository root:
python -m src.classic_ml.models.train_classic

## Switch between dummy and real EMG inside the script:

use_dummy = True # or False
Real EMG must follow the Parquet schema described in the main project README.

## Latency:

The script measures per-window inference latency (ms) for the best model.
This is important for real-time prosthetic control.

## Evaluation Metrics:

Macro-F1 (primary metric) \
Accuracy \
Precision (macro) \
Recall (macro) \
Per-class F1 \
Latency (ms) \
Confusion matrices
