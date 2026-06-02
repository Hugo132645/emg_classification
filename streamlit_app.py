from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from src.common.io.schemas import load_cfg
from src.common.preprocessing.pipelines import preprocess_raw, preprocess_envelope
from src.classic_ml.features import time_domain as td
from src.classic_ml.features import freq_domain as fd
from src.cnn.transforms.spectrograms import batch_compute_spectrograms
from src.cnn.models.model_cnn import EMGConvNet
from src.rnn.features.seq_features import compute_seq_features
from src.rnn.models.model_rnn import GRUModel


RECORDING_ROOTS = [
    Path("data/raw"),
    Path("data/input_data"),
    Path("data/processed"),
]

CLASSIC_ARTIFACT_ROOTS = [
    Path("exports/classic_ml"),
    Path("exports/classic_ml_best"),
    Path("exports/classic_ml_online"),
    Path("exports/classic_ml_online_best"),
]

CNN_ARTIFACT_ROOTS = [
    Path("exports/cnn"),
]

RNN_ARTIFACT_ROOTS = [
    Path("artifacts"),
]

DEFAULT_HISTORY_WINDOWS = 100
CLASSIC_FEATURES_PER_CHANNEL = 20

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


@dataclass
class ModelPrediction:
    display_name: str
    track: str
    label: str
    confidence: float | None
    accepted: bool | None
    latency_ms: float
    probs: dict[str, float] | None


@st.cache_data(show_spinner=False)
def find_recording_files() -> list[str]:
    files: list[str] = []
    for root in RECORDING_ROOTS:
        if not root.exists():
            continue
        for ext in ("*.csv", "*.parquet"):
            files.extend(str(p) for p in root.rglob(ext))
    return sorted(set(files))


@st.cache_data(show_spinner=False)
def load_recording(path: str, fs: int) -> tuple[pd.DataFrame, list[str]]:
    p = Path(path)

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    channel_cols = [c for c in df.columns if c.startswith("emg_ch")]
    if not channel_cols:
        channel_cols = [c for c in df.columns if c.startswith("emg_")]
    if not channel_cols and "signal" in df.columns:
        channel_cols = ["signal"]

    if not channel_cols:
        raise ValueError("No signal/EMG columns found.")

    if "timestamp_ms" not in df.columns:
        if "sample_idx" in df.columns:
            df["timestamp_ms"] = (
                df["sample_idx"].to_numpy(dtype=np.float64) / fs
            ) * 1000.0
        else:
            df["timestamp_ms"] = (
                np.arange(len(df), dtype=np.float64) / fs
            ) * 1000.0

    return df.reset_index(drop=True), channel_cols


@st.cache_data(show_spinner=False)
def find_artifact_files() -> list[str]:
    files: list[str] = []

    for root in CLASSIC_ARTIFACT_ROOTS:
        if root.exists():
            files.extend(str(p) for p in root.rglob("*.joblib"))

    for root in CNN_ARTIFACT_ROOTS:
        if root.exists():
            files.extend(str(p) for p in root.rglob("*.pth"))

    for root in RNN_ARTIFACT_ROOTS:
        if root.exists():
            files.extend(str(p) for p in root.rglob("*.pt"))

    return sorted(set(files))


def identify_artifact_type(path: str) -> str:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".joblib":
        return "classic"
    if suffix == ".pth":
        return "cnn"
    if suffix == ".pt":
        return "rnn"

    raise ValueError(f"Unsupported artifact type: {suffix}")


@st.cache_resource(show_spinner=False)
def load_classic_artifact(path: str) -> dict[str, Any]:
    data = joblib.load(path)
    data["track"] = "classic"
    data["path"] = path
    return data


@st.cache_resource(show_spinner=False)
def load_cnn_artifact(path: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=DEVICE)
    num_classes = int(checkpoint["num_classes"])

    model = EMGConvNet(num_classes=num_classes, dropout=0.3).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cfg = load_cfg()
    class_names = list(cfg.gestures)
    if len(class_names) != num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]

    return {
        "track": "cnn",
        "path": path,
        "checkpoint": checkpoint,
        "model": model,
        "class_names": class_names,
    }


@st.cache_resource(show_spinner=False)
def load_rnn_artifact(path: str) -> dict[str, Any]:
    artifact = torch.load(path, map_location=DEVICE)
    model_type = str(artifact["model_type"]).lower()

    if model_type == "gru":
        model = GRUModel(
            input_dim=int(artifact["input_dim"]),
            hidden_dim=int(artifact["hidden_dim"]),
            num_classes=int(artifact["num_classes"]),
            num_layers=int(artifact["num_layers"]),
            bidirectional=bool(artifact["bidirectional"]),
            dropout=float(artifact["dropout"]),
        ).to(DEVICE)
    elif model_type == "lstm":
        raise RuntimeError(
            "Current LSTMModel in the repo is not usable yet. "
            "Use GRU artifacts or fix src/rnn/models/model_rnn.py first."
        )
    else:
        raise ValueError(f"Unsupported RNN model_type: {model_type}")

    model.load_state_dict(artifact["state_dict"])
    model.eval()

    if "label_names" in artifact:
        inv_label_map = {
            i: str(name)
            for i, name in enumerate(artifact["label_names"])
        }
    else:
        inv_label_map = {
            int(v): f"E{k[0]}_G{k[1]}" if isinstance(k, tuple) else str(k)
            for k, v in artifact["label_map"].items()
        }

    return {
        "track": "rnn",
        "path": path,
        "artifact": artifact,
        "model": model,
        "inv_label_map": inv_label_map,
    }


def load_any_artifact(path: str) -> dict[str, Any]:
    kind = identify_artifact_type(path)

    if kind == "classic":
        return load_classic_artifact(path)
    if kind == "cnn":
        return load_cnn_artifact(path)
    if kind == "rnn":
        return load_rnn_artifact(path)

    raise ValueError(f"Unsupported artifact kind: {kind}")


def artifact_preprocess_mode(
    loaded: dict[str, Any],
    fallback: str,
) -> str:
    track = loaded.get("track")

    if track == "rnn":
        return str(loaded.get("artifact", {}).get("preprocess_mode", fallback))

    if track == "classic":
        return str(loaded.get("preprocess_mode", fallback))

    if track == "cnn":
        return str(loaded.get("checkpoint", {}).get("preprocess_mode", fallback))

    return fallback


def artifact_selected_channel_cols(
    loaded: dict[str, Any],
    all_channel_cols: list[str],
    fallback_channel_cols: list[str],
) -> list[str]:
    track = loaded.get("track")

    if track == "rnn":
        selected = loaded.get("artifact", {}).get("selected_channels", None)
    elif track == "classic":
        selected = loaded.get("selected_channels", None)
    elif track == "cnn":
        selected = loaded.get("checkpoint", {}).get("selected_channels", None)
    else:
        selected = None

    if selected is None:
        return fallback_channel_cols

    try:
        indices = [int(i) for i in list(selected)]
    except Exception:
        return fallback_channel_cols

    if not indices:
        return fallback_channel_cols

    if min(indices) < 0 or max(indices) >= len(all_channel_cols):
        return fallback_channel_cols

    return [all_channel_cols[i] for i in indices]


def preprocess_1d(x: np.ndarray, fs: int, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if mode == "none":
        return x

    if mode == "raw":
        nyquist = fs / 2.0

        if nyquist <= 20.0:
            return x

        high = min(450.0, nyquist * 0.95)

        y, _ = preprocess_raw(
            x,
            fs=fs,
            notch_50hz=False,
            band=(20.0, high),
            lp_cut=10.0,
            norm="zscore",
        )
        return np.asarray(y, dtype=np.float32)

    if mode == "envelope":
        y, _ = preprocess_envelope(
            x,
            fs=fs,
            lp_cut=10.0,
            lp_order=2,
            norm="zscore",
        )
        return np.asarray(y, dtype=np.float32)

    raise ValueError(f"Unknown preprocess mode: {mode}")


def preprocess_multichannel(x_sc: np.ndarray, fs: int, mode: str) -> np.ndarray:
    x_sc = np.asarray(x_sc, dtype=np.float32)

    if x_sc.ndim != 2:
        raise ValueError("Expected shape (samples, channels).")

    cols = []
    for ch in range(x_sc.shape[1]):
        cols.append(preprocess_1d(x_sc[:, ch], fs=fs, mode=mode))

    return np.stack(cols, axis=1)


def classic_feature_row(processed_window_sc: np.ndarray, fs: int) -> np.ndarray:
    win_batch = np.transpose(processed_window_sc, (1, 0))[None, :, :]
    x_time = td.extract_td_features_per_window(win_batch)
    x_freq = fd.extract_fd_features_per_window(win_batch, fs=fs)
    return np.hstack([x_time, x_freq]).astype(np.float32)


def expected_classic_channels(artifact: dict[str, Any]) -> int | None:
    scaler = artifact.get("scaler", None)
    if scaler is None:
        return None

    n_features = getattr(scaler, "n_features_in_", None)
    if n_features is None:
        return None

    if n_features % CLASSIC_FEATURES_PER_CHANNEL != 0:
        return None

    return n_features // CLASSIC_FEATURES_PER_CHANNEL


def cnn_input_tensor(processed_window_1d: np.ndarray, fs: int) -> torch.Tensor:
    specs = batch_compute_spectrograms(
        np.asarray([processed_window_1d], dtype=np.float32),
        fs=fs,
        n_fft=256,
        hop_length=128,
        n_mels=64,
        fmin=20.0,
        fmax=float(fs / 2.0),
    )
    return torch.from_numpy(specs[:, None, :, :]).float().to(DEVICE)


def rnn_sequence_tensor(
    signal_sc: np.ndarray,
    end_idx: int,
    fs: int,
    window_ms: int,
    hop_ms: int,
    seq_len: int,
    preprocess_mode: str,
    loaded_rnn: dict[str, Any],
) -> torch.Tensor | None:
    signal_sc = np.asarray(signal_sc, dtype=np.float32)

    if signal_sc.ndim != 2:
        raise ValueError("Expected signal shape (samples, channels).")

    win_len = int(fs * window_ms / 1000)
    hop_len = int(fs * hop_ms / 1000)

    starts = []
    for i in range(seq_len):
        start = end_idx - win_len + 1 - (seq_len - 1 - i) * hop_len
        starts.append(start)

    if starts[0] < 0:
        return None

    windows = []
    for start in starts:
        end = start + win_len
        w = signal_sc[start:end, :]
        w = preprocess_multichannel(w, fs=fs, mode=preprocess_mode)
        windows.append(w)

    windows = np.stack(windows, axis=0).astype(np.float32)

    feats, _ = compute_seq_features(
        windows,
        fs=fs,
        basic_feat=True,
        shape_feat=True,
        spectral_feat=True,
        deltas=False,
    )

    expected_dim = int(loaded_rnn["artifact"]["input_dim"])
    if feats.shape[1] != expected_dim:
        raise ValueError(
            f"RNN feature dimension mismatch: got {feats.shape[1]}, "
            f"expected {expected_dim}. Selected channels probably do not match training."
        )

    mean_ = np.asarray(loaded_rnn["artifact"]["standardizer_mean"], dtype=np.float32)
    std_ = np.asarray(loaded_rnn["artifact"]["standardizer_std"], dtype=np.float32)
    std_ = np.where(std_ < 1e-7, 1.0, std_)

    feats = (feats - mean_) / std_

    return torch.from_numpy(feats[None, :, :]).float().to(DEVICE)


def is_rest_label(label: Any) -> bool:
    label_str = str(label)

    if label_str == "0":
        return True

    if label_str.endswith("_G0"):
        return True

    if label_str.lower() in {"rest", "no_action", "none"}:
        return True

    return False


def predict_classic(
    loaded: dict[str, Any],
    processed_window_sc: np.ndarray,
    fs: int,
    tau_override: float | None,
) -> ModelPrediction:
    artifact = loaded
    model = artifact["model"]
    gestures = list(artifact["gestures"])
    display_name = Path(artifact["path"]).stem
    tau = artifact.get("no_action_threshold", 0.60) if tau_override is None else tau_override

    expected_channels = expected_classic_channels(artifact)
    if expected_channels is not None:
        if processed_window_sc.shape[1] < expected_channels:
            raise ValueError(
                f"Classic artifact expects {expected_channels} channels, "
                f"but only {processed_window_sc.shape[1]} selected."
            )
        processed_window_sc = processed_window_sc[:, :expected_channels]

    x_row = classic_feature_row(processed_window_sc, fs=fs)

    scaler = artifact.get("scaler", None)
    if scaler is not None:
        x_row = scaler.transform(x_row)

    t0 = time.perf_counter()

    confidence = None
    accepted = None
    probs_out = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_row)[0]
        pred_id = int(np.argmax(probs))
        confidence = float(np.max(probs))

        raw_label = gestures[pred_id]
        accepted = confidence >= tau

        if is_rest_label(raw_label):
            label = "no_action"
            accepted = False
        else:
            label = raw_label if accepted else "no_action"

        probs_out = {
            str(gestures[i]): float(probs[i])
            for i in range(len(gestures))
        }
    else:
        pred = model.predict(x_row)[0]
        try:
            pred_id = int(pred)
            label = gestures[pred_id]
        except Exception:
            label = str(pred)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return ModelPrediction(
        display_name=display_name,
        track="classic",
        label=label,
        confidence=confidence,
        accepted=accepted,
        latency_ms=latency_ms,
        probs=probs_out,
    )


def predict_cnn(
    loaded: dict[str, Any],
    raw_window_sc: np.ndarray,
    fs: int,
    preprocess_mode: str,
    tau_override: float | None,
) -> ModelPrediction:
    model = loaded["model"]
    class_names = list(loaded["class_names"])
    display_name = Path(loaded["path"]).stem
    tau = 0.60 if tau_override is None else tau_override

    prob_list = []
    t0 = time.perf_counter()

    for ch in range(raw_window_sc.shape[1]):
        x_1d = preprocess_1d(raw_window_sc[:, ch], fs=fs, mode=preprocess_mode)
        x = cnn_input_tensor(x_1d, fs=fs)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

        prob_list.append(probs)

    probs_mean = np.mean(np.stack(prob_list, axis=0), axis=0)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_id = int(np.argmax(probs_mean))
    confidence = float(np.max(probs_mean))
    accepted = confidence >= tau
    label = class_names[pred_id] if accepted else "no_action"
    probs_out = {
        class_names[i]: float(probs_mean[i])
        for i in range(len(class_names))
    }

    return ModelPrediction(
        display_name=display_name,
        track="cnn",
        label=label,
        confidence=confidence,
        accepted=accepted,
        latency_ms=latency_ms,
        probs=probs_out,
    )


def predict_rnn(
    loaded: dict[str, Any],
    full_signal_sc: np.ndarray,
    end_idx: int,
    fs: int,
    window_ms: int,
    hop_ms: int,
    preprocess_mode: str,
    tau_override: float | None,
) -> ModelPrediction:
    model = loaded["model"]
    inv_label_map = loaded["inv_label_map"]
    display_name = Path(loaded["path"]).stem
    tau = 0.60 if tau_override is None else tau_override

    seq_len = int(loaded["artifact"].get("seq_length", 16))
    t0 = time.perf_counter()

    x = rnn_sequence_tensor(
        signal_sc=full_signal_sc,
        end_idx=end_idx,
        fs=fs,
        window_ms=window_ms,
        hop_ms=hop_ms,
        seq_len=seq_len,
        preprocess_mode=preprocess_mode,
        loaded_rnn=loaded,
    )

    if x is None:
        return ModelPrediction(
            display_name=display_name,
            track="rnn",
            label="not_enough_history",
            confidence=None,
            accepted=None,
            latency_ms=0.0,
            probs=None,
        )

    lengths = torch.tensor([seq_len], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits = model(x, lengths)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_id = int(np.argmax(probs))
    confidence = float(np.max(probs))

    raw_label = inv_label_map[pred_id]
    accepted = confidence >= tau

    if is_rest_label(raw_label):
        label = "no_action"
        accepted = False
    else:
        label = raw_label if accepted else "no_action"

    probs_out = {
        str(inv_label_map[i]): float(probs[i])
        for i in range(len(probs))
    }

    return ModelPrediction(
        display_name=display_name,
        track="rnn",
        label=label,
        confidence=confidence,
        accepted=accepted,
        latency_ms=latency_ms,
        probs=probs_out,
    )


def label_to_gesture_id(label: str) -> int | None:
    if label == "no_action":
        return 0

    if label.startswith("E") and "_G" in label:
        try:
            return int(label.split("_G")[1])
        except Exception:
            return None

    try:
        return int(label)
    except Exception:
        return None


def predict_any(
    loaded: dict[str, Any],
    raw_window_sc: np.ndarray,
    full_signal_sc: np.ndarray,
    end_idx: int,
    fs: int,
    window_ms: int,
    hop_ms: int,
    preprocess_mode: str,
    tau_override: float | None,
) -> ModelPrediction:
    track = loaded["track"]
    mode = artifact_preprocess_mode(loaded, preprocess_mode)

    if track == "classic":
        proc_sc = preprocess_multichannel(raw_window_sc, fs=fs, mode=mode)
        return predict_classic(
            loaded=loaded,
            processed_window_sc=proc_sc,
            fs=fs,
            tau_override=tau_override,
        )

    if track == "cnn":
        return predict_cnn(
            loaded=loaded,
            raw_window_sc=raw_window_sc,
            fs=fs,
            preprocess_mode=mode,
            tau_override=tau_override,
        )

    if track == "rnn":
        return predict_rnn(
            loaded=loaded,
            full_signal_sc=full_signal_sc,
            end_idx=end_idx,
            fs=fs,
            window_ms=window_ms,
            hop_ms=hop_ms,
            preprocess_mode=mode,
            tau_override=tau_override,
        )

    raise ValueError(f"Unsupported track: {track}")


def build_history_df(
    loaded_artifacts: dict[str, dict[str, Any]],
    df: pd.DataFrame,
    selected_channel_cols: list[str],
    end_idx: int,
    fs: int,
    window_ms: int,
    hop_ms: int,
    preprocess_mode: str,
    tau_override: float | None,
    n_windows: int,
) -> pd.DataFrame:
    win_len = int(fs * window_ms / 1000)
    hop_len = int(fs * hop_ms / 1000)

    all_channel_cols = [c for c in df.columns if c.startswith("emg_ch")]
    if not all_channel_cols:
        all_channel_cols = [c for c in df.columns if c.startswith("emg_")]
    if not all_channel_cols and "signal" in df.columns:
        all_channel_cols = ["signal"]

    rnn_seq_lens = [
        int(loaded["artifact"].get("seq_length", 16))
        for loaded in loaded_artifacts.values()
        if loaded.get("track") == "rnn"
    ]

    required_seq_len = max(rnn_seq_lens) if rnn_seq_lens else 1
    first_valid_end = win_len - 1 + (required_seq_len - 1) * hop_len

    rows = []
    first_end = max(first_valid_end, end_idx - (n_windows - 1) * hop_len)

    for idx in range(first_end, end_idx + 1, hop_len):
        start = idx - win_len + 1
        if start < 0:
            continue

        row = {"timestamp_ms": float(df.iloc[idx]["timestamp_ms"])}

        for display_name, loaded in loaded_artifacts.items():
            model_cols = artifact_selected_channel_cols(
                loaded=loaded,
                all_channel_cols=all_channel_cols,
                fallback_channel_cols=selected_channel_cols,
            )

            raw_window_sc = df.iloc[start : idx + 1][model_cols].to_numpy(
                dtype=np.float32
            )
            full_signal_sc = df[model_cols].to_numpy(dtype=np.float32)

            pred = predict_any(
                loaded=loaded,
                raw_window_sc=raw_window_sc,
                full_signal_sc=full_signal_sc,
                end_idx=idx,
                fs=fs,
                window_ms=window_ms,
                hop_ms=hop_ms,
                preprocess_mode=preprocess_mode,
                tau_override=tau_override,
            )

            row[f"{display_name}_label"] = pred.label
            row[f"{display_name}_confidence"] = pred.confidence
            row[f"{display_name}_gesture_id"] = label_to_gesture_id(pred.label)

        rows.append(row)

    return pd.DataFrame(rows)


def init_state(default_idx: int) -> None:
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "cursor_idx" not in st.session_state:
        st.session_state.cursor_idx = default_idx


def ground_truth_at_cursor(df: pd.DataFrame, end_idx: int) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if "restimulus" in df.columns:
        stimulus = int(df.iloc[end_idx]["restimulus"])
        out["restimulus"] = stimulus

        if "exercise" in df.columns:
            exercise = int(df.iloc[end_idx]["exercise"])
            out["label"] = f"E{exercise}_G{stimulus}"
        else:
            out["label"] = str(stimulus)

        if stimulus == 0:
            out["control_label"] = "no_action"
        else:
            out["control_label"] = out["label"]

    if "rerepetition" in df.columns:
        out["rerepetition"] = int(df.iloc[end_idx]["rerepetition"])

    if "subject" in df.columns:
        out["subject"] = int(df.iloc[end_idx]["subject"])

    if "exercise" in df.columns:
        out["exercise"] = int(df.iloc[end_idx]["exercise"])

    if "timestamp_ms" in df.columns:
        out["timestamp_ms"] = float(df.iloc[end_idx]["timestamp_ms"])

    return out


def main() -> None:
    st.set_page_config(page_title="EMG Replay Dashboard", layout="wide")
    st.title("EMG replay dashboard")

    cfg = load_cfg()
    fs = int(cfg.sample_rate_hz)
    window_ms = int(cfg.window_ms)
    hop_ms = int(cfg.hop_ms)
    win_len = int(fs * window_ms / 1000)
    hop_len = int(fs * hop_ms / 1000)

    recording_files = find_recording_files()
    artifact_files = find_artifact_files()

    with st.sidebar:
        st.subheader("Config")

        if not recording_files:
            st.error("No recording files found under data/.")
            return

        recording_path = st.selectbox("Recording", recording_files)
        df, all_channel_cols = load_recording(recording_path, fs=fs)

        selected_channel_cols = st.multiselect(
            "Channels",
            all_channel_cols,
            default=all_channel_cols,
        )
        if not selected_channel_cols:
            st.warning("Select at least one channel.")
            return

        if not artifact_files:
            st.error("No artifacts/checkpoints found.")
            return

        selected_artifact_paths = st.multiselect(
            "Models",
            artifact_files,
            default=[artifact_files[-1]],
        )
        if not selected_artifact_paths:
            st.warning("Select at least one model.")
            return

        preprocess_mode = st.selectbox(
            "Preprocess mode",
            ["none", "raw", "envelope"],
            index=0,
        )

        visible_panels = st.multiselect(
            "Show panels",
            [
                "Current predictions",
                "Ground truth",
                "Probabilities",
                "Prediction history",
                "Raw signal",
                "Processed signal",
                "Metadata",
            ],
            default=[
                "Current predictions",
                "Ground truth",
                "Probabilities",
                "Prediction history",
            ],
        )

        tau_override_enabled = st.checkbox("Override NO ACTION threshold", value=False)
        tau_override = None
        if tau_override_enabled:
            tau_override = st.slider("NO ACTION threshold", 0.0, 1.0, 0.60, 0.01)

        speed_hz = st.slider("Replay speed (windows/sec)", 1, 20, 5)
        history_windows = st.slider("History windows", 5, 300, DEFAULT_HISTORY_WINDOWS)

        has_rnn_selected = any(
            identify_artifact_type(path) == "rnn"
            for path in selected_artifact_paths
        )

        rnn_seq_len_default = 16
        rnn_min_idx = win_len - 1 + (rnn_seq_len_default - 1) * hop_len

        min_valid_idx = rnn_min_idx if has_rnn_selected else win_len - 1
        default_start_idx = min_valid_idx + history_windows * hop_len

        c1, c2, c3 = st.columns(3)
        if c1.button("Play"):
            st.session_state.playing = True
        if c2.button("Pause"):
            st.session_state.playing = False
        if c3.button("Reset"):
            st.session_state.playing = False
            st.session_state.cursor_idx = default_start_idx
            st.rerun()

    max_idx = len(df) - 1
    default_start_idx = min(default_start_idx, max_idx)

    init_state(default_start_idx)

    cursor_idx = st.sidebar.slider(
        "Position (sample index)",
        min_value=min(min_valid_idx, max_idx),
        max_value=max_idx,
        value=min(max(st.session_state.cursor_idx, min_valid_idx), max_idx),
        step=hop_len,
    )
    st.session_state.cursor_idx = cursor_idx

    loaded_artifacts: dict[str, dict[str, Any]] = {}
    artifact_errors: dict[str, str] = {}

    for path in selected_artifact_paths:
        display_name = Path(path).stem
        try:
            loaded_artifacts[display_name] = load_any_artifact(path)
        except Exception as e:
            artifact_errors[display_name] = str(e)

    if artifact_errors:
        for name, err in artifact_errors.items():
            st.warning(f"{name}: {err}")

    if not loaded_artifacts:
        st.error("No usable artifacts were loaded.")
        return

    end_idx = st.session_state.cursor_idx
    start_idx = end_idx - win_len + 1

    predictions: dict[str, ModelPrediction] = {}
    model_channel_cols: dict[str, list[str]] = {}

    for display_name, loaded in loaded_artifacts.items():
        try:
            model_cols = artifact_selected_channel_cols(
                loaded=loaded,
                all_channel_cols=all_channel_cols,
                fallback_channel_cols=selected_channel_cols,
            )

            model_channel_cols[display_name] = model_cols

            raw_window_sc_model = df.iloc[start_idx : end_idx + 1][model_cols].to_numpy(
                dtype=np.float32
            )
            full_signal_sc_model = df[model_cols].to_numpy(dtype=np.float32)

            pred = predict_any(
                loaded=loaded,
                raw_window_sc=raw_window_sc_model,
                full_signal_sc=full_signal_sc_model,
                end_idx=end_idx,
                fs=fs,
                window_ms=window_ms,
                hop_ms=hop_ms,
                preprocess_mode=preprocess_mode,
                tau_override=tau_override,
            )

            predictions[display_name] = pred

        except Exception as e:
            predictions[display_name] = ModelPrediction(
                display_name=display_name,
                track=loaded.get("track", "unknown"),
                label=f"error: {e}",
                confidence=None,
                accepted=None,
                latency_ms=0.0,
                probs=None,
            )

    raw_window_sc = df.iloc[start_idx : end_idx + 1][selected_channel_cols].to_numpy(
        dtype=np.float32
    )

    processed_window_sc = preprocess_multichannel(
        raw_window_sc,
        fs=fs,
        mode=preprocess_mode,
    )

    if "Metadata" in visible_panels:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Samples", len(df))
        m2.metric("Selected channels", len(selected_channel_cols))
        m3.metric("Timestamp (ms)", f"{df.iloc[end_idx]['timestamp_ms']:.1f}")
        m4.metric("Device", str(DEVICE))

        st.write("Model channel usage:")
        for name, cols in model_channel_cols.items():
            st.write(f"{name}: {cols}")

    if "Ground truth" in visible_panels:
        st.subheader("Ground truth")

        gt = ground_truth_at_cursor(df, end_idx)

        if not gt:
            st.info("No ground-truth columns found in this recording.")
        else:
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Ground truth", gt.get("control_label", "unknown"))
            c2.metric("Raw label", gt.get("label", "unknown"))
            c3.metric("Repetition", gt.get("rerepetition", "unknown"))
            c4.metric("Subject", gt.get("subject", "unknown"))

            if "exercise" in gt:
                st.caption(f"Exercise: {gt['exercise']}")

    if "Current predictions" in visible_panels:
        cols = st.columns(len(predictions))
        for col, (display_name, pred) in zip(cols, predictions.items()):
            with col:
                st.subheader(display_name)
                st.caption(pred.track.upper())

                gt = ground_truth_at_cursor(df, end_idx)
                gt_control_label = gt.get("control_label", None)

                st.metric("Prediction", pred.label)

                if gt_control_label is not None:
                    is_correct = pred.label == gt_control_label
                    st.metric("Correct", "Yes" if is_correct else "No")

                if pred.confidence is not None:
                    st.metric("Confidence", f"{pred.confidence:.3f}")

                if pred.accepted is not None:
                    st.metric("Accepted", "Yes" if pred.accepted else "No")

                st.metric("Latency (ms)", f"{pred.latency_ms:.3f}")

    if "Probabilities" in visible_panels:
        st.subheader("Probabilities")

        prob_cols = st.columns(len(predictions))
        for col, (display_name, pred) in zip(prob_cols, predictions.items()):
            with col:
                st.caption(display_name)

                if pred.probs is None:
                    st.info("No probability output available.")
                else:
                    prob_df = pd.DataFrame(
                        {
                            "label": list(pred.probs.keys()),
                            "probability": list(pred.probs.values()),
                        }
                    ).sort_values("probability", ascending=False)

                    prob_df["probability"] = prob_df["probability"].round(4)

                    st.dataframe(
                        prob_df,
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.bar_chart(
                        prob_df.head(20).set_index("label"),
                        use_container_width=True,
                    )

    if "Prediction history" in visible_panels:
        st.subheader("Prediction history")

        hist_df = build_history_df(
            loaded_artifacts=loaded_artifacts,
            df=df,
            selected_channel_cols=selected_channel_cols,
            end_idx=end_idx,
            fs=fs,
            window_ms=window_ms,
            hop_ms=hop_ms,
            preprocess_mode=preprocess_mode,
            tau_override=tau_override,
            n_windows=history_windows,
        )

        if hist_df.empty:
            st.info("No prediction history available yet.")
        else:
            label_cols = [c for c in hist_df.columns if c.endswith("_label")]
            conf_cols = [c for c in hist_df.columns if c.endswith("_confidence")]
            gesture_cols = [c for c in hist_df.columns if c.endswith("_gesture_id")]

            st.write("History rows:", len(hist_df))

            if gesture_cols:
                st.write("Predicted class / gesture ID over time")
                st.line_chart(hist_df.set_index("timestamp_ms")[gesture_cols])

            if conf_cols:
                st.write("Confidence over time")
                st.line_chart(hist_df.set_index("timestamp_ms")[conf_cols])

            if label_cols:
                st.write("Recent predictions")
                display_cols = ["timestamp_ms"] + label_cols + conf_cols
                st.dataframe(
                    hist_df[display_cols].tail(30),
                    hide_index=True,
                    use_container_width=True,
                )

    if "Raw signal" in visible_panels:
        st.subheader("Raw window")
        st.line_chart(pd.DataFrame(raw_window_sc, columns=selected_channel_cols))

    if "Processed signal" in visible_panels:
        st.subheader("Processed window")
        st.line_chart(pd.DataFrame(processed_window_sc, columns=selected_channel_cols))

    if st.session_state.playing:
        next_idx = min(end_idx + hop_len, len(df) - 1)
        if next_idx == end_idx:
            st.session_state.playing = False
        else:
            time.sleep(1.0 / speed_hz)
            st.session_state.cursor_idx = next_idx
            st.rerun()


if __name__ == "__main__":
    main()

