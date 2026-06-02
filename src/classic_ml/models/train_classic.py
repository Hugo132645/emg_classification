from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.classic_ml.datasets.classic_ml_dataset import build_classic_ml_dataset
from src.common.io.dummy_data import generate_dummy_emg
from src.common.io.emg_loader import _search_csv_files, process_emg_for_windowing
from src.common.io.schemas import (
    ensure_parent_dir,
    expand_template,
    load_cfg,
    sanity_check_cfg,
)
from src.common.preprocessing.windowing import window_signal, window_signal_np


try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = e


USE_DUMMY = False

DATA_PATH = "data/input_data/ninapro_db1/csv_files"
SELECTED_CHANNELS = [0, 1]
ALLOWED_EXERCISES = (2,)
DROP_REST = False

SKIP_MODELS = {
    "svm_linear",
    "svm_rbf",
}

NO_ACTION_TAU = 0.60
TEST_SIZE = 0.20
RANDOM_STATE = 0


def _encode_dummy_labels(cfg, y):
    y_arr = np.asarray(y, dtype=object)
    label_map = cfg.label_map
    inv_map = {v: k for k, v in label_map.items()}

    y_int = []
    for label in y_arr:
        label_str = str(label)
        if label_str not in label_map:
            raise KeyError(f"Label '{label_str}' not found in cfg.label_map")
        y_int.append(label_map[label_str])

    return np.asarray(y_int, dtype=np.int32), inv_map


def _decode_labels(y_int, inv_map):
    y_int = np.asarray(y_int)
    return np.asarray([inv_map[int(v)] for v in y_int], dtype=object)


def _estimate_latency_ms(model, X: np.ndarray, n_reps: int = 100) -> float:
    if X.shape[0] == 0:
        return float("nan")

    x_single = X[:1]
    start = time()
    for _ in range(n_reps):
        _ = model.predict(x_single)
    end = time()

    return ((end - start) / n_reps) * 1000.0


def _evaluate_with_rejection(model, X_test, y_test, inv_map, tau=0.60):
    if not hasattr(model, "predict_proba"):
        return None

    probs = model.predict_proba(X_test)
    pred_ids = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    y_true_str = _decode_labels(y_test, inv_map)
    accepted_mask = confs >= tau

    y_pred_reject = []
    for pred_id, conf in zip(pred_ids, confs):
        if conf < tau:
            y_pred_reject.append("no_action")
        else:
            y_pred_reject.append(inv_map[int(pred_id)])

    y_pred_reject = np.asarray(y_pred_reject, dtype=object)

    coverage = float(np.mean(accepted_mask))
    reject_rate = 1.0 - coverage

    metrics = {
        "tau": float(tau),
        "coverage": coverage,
        "reject_rate": reject_rate,
    }

    if np.any(accepted_mask):
        accepted_true = y_true_str[accepted_mask]
        accepted_pred = y_pred_reject[accepted_mask]

        metrics["accepted_accuracy"] = float(
            accuracy_score(accepted_true, accepted_pred)
        )
        metrics["accepted_macro_f1"] = float(
            f1_score(accepted_true, accepted_pred, average="macro")
        )
    else:
        metrics["accepted_accuracy"] = float("nan")
        metrics["accepted_macro_f1"] = float("nan")

    return metrics


def predict_window_with_no_action(artifact, x_row):
    model = artifact["model"]
    tau = artifact.get("no_action_threshold", 0.60)
    gestures = artifact["gestures"]
    inv_map = {i: g for i, g in enumerate(gestures)}

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"Model {type(model).__name__} does not support predict_proba."
        )

    probs = model.predict_proba(x_row)[0]
    pred_id = int(np.argmax(probs))
    conf = float(np.max(probs))

    if conf < tau:
        return {
            "label": "no_action",
            "confidence": conf,
            "probs": probs,
            "accepted": False,
        }

    return {
        "label": inv_map[pred_id],
        "confidence": conf,
        "probs": probs,
        "accepted": True,
    }


def _build_dummy_dataset(cfg):
    fs = cfg.sample_rate_hz
    gestures = cfg.gestures
    seconds = 1000

    df, interval_labels, timestamps_ms = generate_dummy_emg(
        seconds=seconds,
        fs=fs,
        classes=gestures,
    )

    signal = df["signal"].to_numpy(dtype=np.float32)

    windows, window_labels, _ = window_signal(
        x=signal,
        fs=fs,
        window_ms=cfg.window_ms,
        hop_ms=cfg.hop_ms,
        labels=interval_labels,
        timestamps_ms=timestamps_ms,
    )

    X, y, scaler = build_classic_ml_dataset(
        windows=windows,
        window_labels=window_labels,
        fs=fs,
        mode="time+freq",
        scale=True,
    )

    return X, y, scaler


def _build_real_dataset(
    cfg,
    data_path: str = DATA_PATH,
    channels: list[int] | None = None,
    drop_rest: bool = False,
    allowed_exercises: tuple[int, ...] | None = None,
):
    fs = cfg.sample_rate_hz

    csv_files = sorted(Path(p) for p in _search_csv_files(data_path))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_path}")

    X_blocks = []
    y_blocks = []

    for csv_path in csv_files:
        x, labels, rerepetition, subject, exercise = process_emg_for_windowing(
            str(csv_path),
            drop_rest=False,
        )

        if allowed_exercises is not None and int(exercise) not in allowed_exercises:
            continue

        if channels is not None:
            x = x[:, list(channels)]

        if len(x) == 0:
            continue

        windows, window_labels, _ = window_signal_np(
            x=x,
            fs=fs,
            window_ms=cfg.window_ms,
            hop_ms=cfg.hop_ms,
            labels=labels,
        )

        window_labels = np.asarray(window_labels)

        if drop_rest:
            keep = window_labels != 0
            windows = windows[keep]
            window_labels = window_labels[keep]

        if len(windows) == 0:
            continue

        if windows.ndim == 3:
            windows = np.transpose(windows, (0, 2, 1))

        X_part, y_part, _ = build_classic_ml_dataset(
            windows=windows,
            window_labels=window_labels,
            fs=fs,
            mode="time+freq",
            scale=False,
        )

        X_blocks.append(X_part)
        y_blocks.append(y_part)

        print(
            f"Loaded {csv_path.name}: "
            f"subject={subject}, exercise={exercise}, "
            f"windows={len(window_labels)}, features={X_part.shape[1]}, "
            f"labels={sorted(np.unique(window_labels).tolist())}"
        )

    if not X_blocks:
        raise ValueError("No usable NinaPro CSV files found after filtering.")

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print("Built NinaPro classic ML dataset")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Labels:", sorted(set(map(str, y))))

    return X, y, scaler


def _encode_real_labels(y, exercise_id: int = 2):
    unique_labels = sorted(np.unique(y).tolist())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    inv_map = {i: f"E{exercise_id}_G{label}" for label, i in label_map.items()}
    y_int = np.asarray([label_map[label] for label in y], dtype=np.int32)
    return y_int, inv_map


def _make_models():
    models = {
        "logreg": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    else:
        print("Skipping xgboost because it could not be imported:")
        print(XGBOOST_IMPORT_ERROR)

    models = {
        name: model
        for name, model in models.items()
        if name not in SKIP_MODELS
    }

    return models


def _make_artifact_path(cfg, track: str, subject_id: str, timestamp: int) -> Path:
    path_str = expand_template(
        cfg.model_file_template,
        track=track,
        subject_id=subject_id,
        session_date=datetime.now().strftime("%Y-%m-%d"),
        session_id=0,
        timestamp=timestamp,
        model_artifact_ext="joblib",
    )

    path = Path(path_str).with_suffix(".joblib")
    ensure_parent_dir(str(path))
    return path


def main():
    cfg = load_cfg()
    sanity_check_cfg(cfg)

    print("Config loaded.")
    print(f"sample_rate_hz: {cfg.sample_rate_hz}")
    print(f"window_ms: {cfg.window_ms}, hop_ms: {cfg.hop_ms}")
    print(f"use_dummy = {USE_DUMMY}")

    session_date = datetime.now().strftime("%Y-%m-%d")
    timestamp = int(time())

    if USE_DUMMY:
        print("\n[1] Building dataset from DUMMY EMG...")
        X, y, scaler = _build_dummy_dataset(cfg)
        y, inv_map = _encode_dummy_labels(cfg, y)
        subject_id = "DUMMY"
    else:
        print("\n[1] Building dataset from REAL NinaPro EMG...")
        X, y, scaler = _build_real_dataset(
            cfg,
            data_path=DATA_PATH,
            channels=SELECTED_CHANNELS,
            drop_rest=DROP_REST,
            allowed_exercises=ALLOWED_EXERCISES,
        )
        y, inv_map = _encode_real_labels(y, exercise_id=ALLOWED_EXERCISES[0])
        subject_id = "REAL"

    print("Final X shape:", X.shape)
    print("Final y shape:", y.shape)
    print("Class map:", inv_map)

    print("\n[2] Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Train size:", X_train.shape[0])
    print("Test size:", X_test.shape[0])

    print("\n[3] Defining models...")
    models = _make_models()

    if not models:
        raise ValueError("No models left to train. Check SKIP_MODELS.")

    print("Models to train:")
    for name, model in models.items():
        print(f"  {name}: {type(model).__name__}")

    results = []
    best_name = None
    best_model = None
    best_macro_f1 = -1.0
    best_cm = None

    class_ids = sorted(np.unique(y_train))
    class_names = [inv_map[i] for i in class_ids]

    print("\n[4] Training and evaluating models...")
    for name, model in models.items():
        print(f"\nModel: {name}")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=class_ids)

        print("Accuracy:", acc)
        print("Macro-F1:", macro_f1)
        print("Precision (macro):", precision)
        print("Recall (macro):", recall)
        print("Confusion matrix:")
        print(cm)
        print("Classification report:")
        print(
            classification_report(
                y_test,
                y_pred,
                labels=class_ids,
                target_names=class_names,
                zero_division=0,
            )
        )

        results.append(
            {
                "name": name,
                "accuracy": float(acc),
                "macro_f1": float(macro_f1),
                "precision": float(precision),
                "recall": float(recall),
            }
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_name = name
            best_model = model
            best_cm = cm

    print("\n[5] Summary of model scores:")
    df_results = pd.DataFrame(results).sort_values(by="macro_f1", ascending=False)
    print(df_results)

    reports_dir = Path(f"reports/{session_date}/best")
    reports_dir.mkdir(parents=True, exist_ok=True)

    results_path = reports_dir / f"results_table_{timestamp}.csv"
    df_results.to_csv(results_path, index=False)
    print("Saved results table to:", results_path)

    print("\n[6] Estimating latency for best model...")
    latency_ms = _estimate_latency_ms(best_model, X_test, n_reps=200)
    print(f"Estimated model.predict latency per window: {latency_ms:.4f} ms")

    rejection_results = _evaluate_with_rejection(
        best_model,
        X_test,
        y_test,
        inv_map,
        tau=NO_ACTION_TAU,
    )

    if rejection_results is not None:
        rejection_results_path = reports_dir / f"no_action_eval_{timestamp}.json"
        with open(rejection_results_path, "w", encoding="utf-8") as f:
            json.dump(rejection_results, f, indent=2)

        print("Saved NO ACTION evaluation to:", rejection_results_path)
    else:
        rejection_results_path = None
        print("Best model has no predict_proba. Skipping NO ACTION evaluation.")

    print("\n[7] Saving BEST model artifact...")
    best_artifact_path = _make_artifact_path(
        cfg=cfg,
        track="classic_ml_best",
        subject_id=subject_id,
        timestamp=timestamp,
    )

    best_artifact = {
        "model_name": best_name,
        "model": best_model,
        "scaler": scaler,
        "gestures": list(class_names),
        "config": vars(cfg),
        "latency_ms": float(latency_ms),
        "results": results,
        "use_dummy": USE_DUMMY,
        "no_action_threshold": float(NO_ACTION_TAU),
        "supports_predict_proba": hasattr(best_model, "predict_proba"),
        "rejection_results": rejection_results,
        "selected_channels": SELECTED_CHANNELS,
        "allowed_exercises": ALLOWED_EXERCISES,
        "drop_rest": DROP_REST,
    }

    joblib.dump(best_artifact, best_artifact_path)
    print("Saved BEST model artifact to:", best_artifact_path)

    print("\n[8] Saving ALL trained model artifacts...")
    all_base_path = _make_artifact_path(
        cfg=cfg,
        track="classic_ml",
        subject_id=subject_id,
        timestamp=timestamp,
    )

    all_model_artifacts = {}

    for name, model in models.items():
        per_model_path = all_base_path.with_name(
            f"{all_base_path.stem}_{name}.joblib"
        )
        ensure_parent_dir(str(per_model_path))

        per_artifact = {
            "model_name": name,
            "model": model,
            "scaler": scaler,
            "gestures": list(class_names),
            "config": vars(cfg),
            "latency_ms": float(latency_ms),
            "results": results,
            "use_dummy": USE_DUMMY,
            "no_action_threshold": float(NO_ACTION_TAU),
            "supports_predict_proba": hasattr(model, "predict_proba"),
            "selected_channels": SELECTED_CHANNELS,
            "allowed_exercises": ALLOWED_EXERCISES,
            "drop_rest": DROP_REST,
        }

        joblib.dump(per_artifact, per_model_path)
        all_model_artifacts[name] = str(per_model_path)
        print(f"Saved {name} to:", per_model_path)

    run_info = {
        "best_model_artifact": str(best_artifact_path),
        "best_model_name": best_name,
        "macro_f1": float(best_macro_f1),
        "latency_ms": float(latency_ms),
        "results": results,
        "class_names": list(class_names),
        "timestamp": timestamp,
        "use_dummy": USE_DUMMY,
        "results_table_csv": str(results_path),
        "all_model_artifacts": all_model_artifacts,
        "no_action_threshold": float(NO_ACTION_TAU),
        "rejection_results": rejection_results,
        "rejection_results_path": (
            str(rejection_results_path) if rejection_results_path else None
        ),
        "selected_channels": SELECTED_CHANNELS,
        "allowed_exercises": ALLOWED_EXERCISES,
        "drop_rest": DROP_REST,
    }

    run_json_path = best_artifact_path.with_suffix(".run.json")
    with open(run_json_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print("Saved run metadata to:", run_json_path)

    print("\nDone.")
    print("Classic ML artifacts should now be visible to Streamlit if they are under exports/ and end in .joblib.")


if __name__ == "__main__":
    main()