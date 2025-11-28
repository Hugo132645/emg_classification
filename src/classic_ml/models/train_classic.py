# Imports

from __future__ import annotations
import json
from pathlib import Path
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA  # >>> ADDED
import joblib
from src.common.io.schemas import (
    load_cfg,
    sanity_check_cfg,
    expand_template,
    ensure_parent_dir,
    is_raw_mode,
)
from src.common.io.dummy_data import generate_dummy_emg
from src.common.preprocessing.windowing import window_signal
from src.classic_ml.datasets.classic_ml_dataset import build_classic_ml_dataset
from src.common.preprocessing.pipelines import (
    preprocess_raw,
    preprocess_envelope,
)
from datetime import datetime

# Dummy data flag

use_dummy = True  # set to False for real EMG

# Confusion matrix ploted


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    ensure_parent_dir(str(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Simple bar plot for model metrics


def _plot_model_scores_bar(
    df_results: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df_results["name"], df_results[metric])
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.set_title(title)
    plt.tight_layout()
    ensure_parent_dir(str(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Random forest feature importance graph


def _plot_rf_feature_importance(rf_model, feature_names, out_path: Path):
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(sorted_names, sorted_vals)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    ensure_parent_dir(str(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# PPer-clas F1 bar plot


def _plot_per_class_f1(y_true, y_pred, class_names, out_path: Path):
    f1_vals = f1_score(y_true, y_pred, average=None, labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(class_names, f1_vals)
    ax.set_title("Per-Class F1")
    ax.set_ylabel("F1 Score")
    plt.tight_layout()
    ensure_parent_dir(str(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# PCA 2D plot


def _plot_pca_2d(X, y, class_names, out_path: Path):
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    for cls in class_names:
        mask = np.array(y) == cls
        ax.scatter(X2[mask, 0], X2[mask, 1], s=12, label=cls)
    ax.set_title("PCA Feature Space (2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    plt.tight_layout()
    ensure_parent_dir(str(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Latency estimation


def _estimate_latency_ms(model, X: np.ndarray, n_reps: int = 100) -> float:
    if X.shape[0] == 0:
        return float("nan")
    x_single = X[:1]  # one sample
    start = time()
    for _ in range(n_reps):
        _ = model.predict(x_single)
    end = time()
    avg_s = (end - start) / n_reps
    return avg_s * 1000.0


# Dummy dataset build


def _build_dummy_dataset(cfg):
    fs = cfg.sample_rate_hz
    gestures = cfg.gestures  # list of gesture names from YAML
    # 1000 seconds of dummy EMG just for a longer test
    seconds = 1000
    df, interval_labels, timestamps_ms = generate_dummy_emg(
        seconds=seconds,
        fs=fs,
        classes=gestures,
    )
    signal = df["signal"].to_numpy(dtype=np.float32)
    windows, window_labels, times_ms = window_signal(
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


# Real dataset build when needed


def _build_real_dataset(cfg):
    fs = cfg.sample_rate_hz
    subject_id = "S01"
    session_date = "2025-11-06"
    session_id = 1
    raw_path_str = expand_template(
        cfg.raw_file_template,
        subject_id=subject_id,
        session_date=session_date,
        session_id=session_id,
    )
    raw_path = Path(raw_path_str)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Real EMG file not found at {raw_path}. "
            "Create a parquet file matching the README schema or change subject/date/session."
        )
    df = pd.read_parquet(raw_path)
    # Decide which column to use based on config (raw vs envelope mode)
    if is_raw_mode(cfg):
        if "emg_ch1_raw" not in df.columns:
            raise KeyError("Column 'emg_ch1_raw' not found in real EMG parquet.")
        x = df["emg_ch1_raw"].to_numpy(dtype=np.float32)
        x_proc, meta = preprocess_raw(
            x,
            fs=fs,
            notch_50hz=(cfg.notch_hz is not None),
            band=cfg.bandpass_hz or (20.0, 450.0),
        )
    else:
        if "emg_ch1_env" not in df.columns:
            raise KeyError("Column 'emg_ch1_env' not found in real EMG parquet.")
        x = df["emg_ch1_env"].to_numpy(dtype=np.float32)
        x_proc, meta = preprocess_envelope(x, fs=fs)
    # Labels and timestamps per sample
    if "label" not in df.columns or "timestamp_ms" not in df.columns:
        raise KeyError("Real EMG parquet must have 'label' and 'timestamp_ms' columns.")
    sample_labels = df["label"].to_numpy()
    timestamps_ms = df["timestamp_ms"].to_numpy()
    # Windowing with majority-vote labels
    windows, window_labels, times_ms = window_signal(
        x=x_proc,
        fs=fs,
        window_ms=cfg.window_ms,
        hop_ms=cfg.hop_ms,
        labels=sample_labels,
        timestamps_ms=timestamps_ms,
    )
    # Build features + scaler
    X, y, scaler = build_classic_ml_dataset(
        windows=windows,
        window_labels=window_labels,
        fs=fs,
        mode="time+freq",
        scale=True,
    )
    return X, y, scaler


# All models training and evaluation, comparison and output best model with data


def main():
    # 1. Load and sanity check configuration
    cfg = load_cfg()
    sanity_check_cfg(cfg)
    print("Config loaded.")
    print(f"sample_rate_hz: {cfg.sample_rate_hz}")
    print(f"window_ms: {cfg.window_ms}, hop_ms: {cfg.hop_ms}")
    print(f"gestures: {cfg.gestures}")
    print(f"use_dummy = {use_dummy}")
    session_date = datetime.now().strftime("%Y-%m-%d")

    # 2. Build dataset (dummy or real) based on use_dummy
    if use_dummy:
        print("\n[1] Building dataset from DUMMY EMG...")
        X, y, scaler = _build_dummy_dataset(cfg)
    else:
        print("\n[1] Building dataset from REAL EMG + preprocessing...")
        X, y, scaler = _build_real_dataset(cfg)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 3. Train-test split
    print("\n[2] Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )
    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

    # 4. Define candidate models
    print("\n[3] Defining models...")
    models = {
        "logreg": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            multi_class="auto",
        ),
        "svm_linear": SVC(
            kernel="linear",
            C=1.0,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=0,
        ),
    }
    results: list[dict] = []
    best_name: str | None = None
    best_model = None
    best_macro_f1 = -1.0
    best_cm: np.ndarray | None = None
    class_names = sorted(np.unique(y_train))

    # 5. Train and evaluate each model
    print("\n[4] Training and evaluating models...")
    for name, model in models.items():
        print(f"\nModel: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred, labels=class_names)

        print("Accuracy:", acc)
        print("Macro-F1:", macro_f1)
        print("Precision (macro):", precision)
        print("Recall (macro):", recall)
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
        print("Classification report:")
        print(classification_report(y_test, y_pred, labels=class_names))
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

    # 6. Print a simple summary table
    print("\n[5] Summary of model scores:")
    print(f"{'Model':<12} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 34)
    for r in results:
        print(f"{r['name']:<12} {r['accuracy']:>10.3f} {r['macro_f1']:>10.3f}")
    print(f"\nBest model: {best_name} (macro-F1 = {best_macro_f1:.3f})")

    # 6b. Build and save a DataFrame of results
    timestamp = int(time())
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="macro_f1", ascending=False)
    print("\n[5b] Results table:")
    print(df_results)
    results_path = Path(f"reports/{session_date}/results_table_{timestamp}.csv")
    ensure_parent_dir(str(results_path))
    df_results.to_csv(results_path, index=False)
    print("Saved results table to:", results_path)

    # 6c. Bar plot of model scores
    scores_plot_path = Path(
        f"reports/{session_date}/model_scores_macro_f1_{timestamp}.png"
    )
    _plot_model_scores_bar(
        df_results,
        metric="macro_f1",
        out_path=scores_plot_path,
        title="Model comparison (Macro-F1)",
    )
    print("Saved model scores plot to:", scores_plot_path)

    # 7. Latency estimate for the best model
    print("\n[6] Estimating latency for best model...")
    latency_ms = _estimate_latency_ms(best_model, X_test, n_reps=200)
    print(f"Estimated model.predict latency per window: {latency_ms:.4f} ms")

    # 8. Plot confusion matrix for best model
    print("\n[7] Plotting confusion matrix for best model...")
    cm_title = f"Confusion matrix - {best_name}"
    cm_path = Path(
        f"reports/{session_date}/best/confusion_matrix_{best_name}_{timestamp}.png"
    )
    _plot_confusion_matrix(best_cm, class_names, cm_title, cm_path)
    print("Saved confusion matrix to:", cm_path)

    print("\n[7b] Plotting confusion matrices for ALL models...")
    for name, model in models.items():
        y_pred_all = model.predict(X_test)
        cm_all = confusion_matrix(y_test, y_pred_all, labels=class_names)
        cm_title_all = f"Confusion matrix - {name}"
        cm_path_all = Path(
            f"reports/{session_date}/confusion_matrix_{name}_{timestamp}.png"
        )
        _plot_confusion_matrix(cm_all, class_names, cm_title_all, cm_path_all)
        print("Saved confusion matrix for", name, "to:", cm_path_all)

    if "rf" in models:
        from src.classic_ml.features.time_domain import td_feature_names
        from src.classic_ml.features.freq_domain import feature_names_freq

        feature_names = td_feature_names() + feature_names_freq()
        fi_path = Path(f"reports/{session_date}/feature_importance_rf_{timestamp}.png")
        print("\n[7c] Plotting Random Forest feature importance...")
        _plot_rf_feature_importance(models["rf"], feature_names, fi_path)
        print("Saved RF feature importance to:", fi_path)

    print("\n[7d] Plotting per-class F1 scores...")
    y_pred_best = best_model.predict(X_test)
    f1_path = Path(f"reports/{session_date}/per_class_f1_{timestamp}.png")
    _plot_per_class_f1(y_test, y_pred_best, class_names, f1_path)
    print("Saved per-class F1 plot to:", f1_path)

    print("\n[7e] Plotting PCA 2D feature space...")
    pca_path = Path(f"reports/{session_date}/pca_2d_{timestamp}.png")
    _plot_pca_2d(X, y, class_names, pca_path)
    print("Saved PCA 2D plot to:", pca_path)

    # 9. Save best model + scaler + run metadata using model_file_template
    print("\n[8] Saving model artifact...")
    model_artifact_path_str = expand_template(
        cfg.model_file_template,
        track="classic_ml_best",
        subject_id="DUMMY" if use_dummy else "REAL",
        session_date=datetime.now().strftime("%Y-%m-%d"),
        session_id=0,
        timestamp=timestamp,
        model_artifact_ext="joblib",
    )
    model_artifact_path = Path(model_artifact_path_str)
    ensure_parent_dir(str(model_artifact_path))

    artifact = {
        "model_name": best_name,
        "model": best_model,
        "scaler": scaler,
        "gestures": list(class_names),
        "config": vars(cfg),
        "latency_ms": float(latency_ms),
        "results": results,
        "use_dummy": use_dummy,
    }
    joblib.dump(artifact, model_artifact_path)
    print("Saved BEST model artifact to:", model_artifact_path)

    # 9b. Save ALL trained models as separate artifacts
    model_artifact_path_str = expand_template(
        cfg.model_file_template,
        track="classic_ml",
        subject_id="DUMMY" if use_dummy else "REAL",
        session_date=datetime.now().strftime("%Y-%m-%d"),
        session_id=0,
        timestamp=timestamp,
        model_artifact_ext="joblib",
    )
    model_artifact_path = Path(model_artifact_path_str)

    all_model_artifacts: dict[str, str] = {}
    for name, model in models.items():
        per_model_path = model_artifact_path.with_name(
            f"{model_artifact_path.stem}_{name}.joblib"
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
            "use_dummy": use_dummy,
        }
        joblib.dump(per_artifact, per_model_path)
        all_model_artifacts[name] = str(per_model_path)
    print("Saved ALL model artifacts:", all_model_artifacts)

    # 10. Save a simple run.json next to the model
    model_artifact_path_str = expand_template(
        cfg.model_file_template,
        track="classic_ml_best",
        subject_id="DUMMY" if use_dummy else "REAL",
        session_date=datetime.now().strftime("%Y-%m-%d"),
        session_id=0,
        timestamp=timestamp,
        model_artifact_ext="joblib",
    )
    model_artifact_path = Path(model_artifact_path_str)
    run_info = {
        "model_artifact": str(model_artifact_path),
        "best_model_name": best_name,
        "macro_f1": float(best_macro_f1),
        "latency_ms": float(latency_ms),
        "results": results,
        "class_names": list(class_names),
        "timestamp": timestamp,
        "use_dummy": use_dummy,
        "results_table_csv": str(results_path),
        "model_scores_plot": str(scores_plot_path),
        "all_model_artifacts": all_model_artifacts,
    }
    run_json_path = model_artifact_path.with_suffix(".run.json")
    with open(run_json_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print("Saved run metadata to:", run_json_path)


if __name__ == "__main__":
    main()
