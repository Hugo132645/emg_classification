# Imports

from __future__ import annotations
import json
from pathlib import Path
from time import time
import numpy as np
from src.classic_ml.utils.plots import (
    plot_confusion_matrix,
    plot_model_scores_bar,
    plot_rf_feature_importance,
    plot_xgb_feature_importance,
    plot_per_class_f1,
    plot_pca_2d,
    plot_tsne_2d,
    plot_umap_2d,
    plot_tsne_3d,
    plot_umap_3d,
)
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
import joblib
from src.common.io.schemas import (
    load_cfg,
    sanity_check_cfg,
    expand_template,
    ensure_parent_dir,
    is_raw_mode,
)
from src.common.io.dummy_data import generate_dummy_emg
from src.common.preprocessing.windowing import (
    window_signal,
    window_segment_multichannel,
)
from src.classic_ml.datasets.classic_ml_dataset import build_classic_ml_dataset
from src.common.preprocessing.pipelines import (
    preprocess_raw,
    preprocess_envelope,
)
from datetime import datetime
from xgboost import XGBClassifier

# Dummy data flag

use_dummy = True  # set to False for real EMG

# Label encoding


def _encode_labels(cfg, y):
    y_arr = np.asarray(y, dtype=object)
    label_map = cfg.label_map
    inv_map = {v: k for k, v in label_map.items()}

    y_int = []
    for label in y_arr:
        label_str = str(label)
        if label_str not in label_map:
            raise KeyError(f"Label '{label_str}' not found in cfg.label_map")
        y_int.append(label_map[label_str])

    return np.array(y_int, dtype=np.int32), inv_map


def _decode_labels(y_int, inv_map):
    y_int = np.asarray(y_int)
    y_str = []
    for v in y_int:
        v = int(v)
        if v not in inv_map:
            raise KeyError(f"Encoded label '{v}' not found in inverse label map")
        y_str.append(inv_map[v])
    return np.array(y_str, dtype=object)


# NO_ACTION functions


def _predict_with_rejection(model, x_row, inv_map, tau=0.60):
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"Model {type(model).__name__} does not support predict_proba in current config."
        )

    probs = model.predict_proba(x_row)[0]
    pred_id = int(np.argmax(probs))
    conf = float(np.max(probs))

    if conf < tau:
        return "no_action", conf, probs, False

    pred_label = inv_map[pred_id]
    return pred_label, conf, probs, True


def _evaluate_with_rejection(model, X_test, y_test, inv_map, tau=0.60):
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"Model {type(model).__name__} does not support predict_proba in current config."
        )

    probs = model.predict_proba(X_test)  # (N, K)
    pred_ids = np.argmax(probs, axis=1)  # (N,)
    confs = np.max(probs, axis=1)  # (N,)

    y_true_str = _decode_labels(y_test, inv_map)
    y_pred_reject = []
    accepted_mask = confs >= tau

    for pred_id, conf in zip(pred_ids, confs):
        if conf < tau:
            y_pred_reject.append("no_action")
        else:
            y_pred_reject.append(inv_map[int(pred_id)])

    y_pred_reject = np.array(y_pred_reject, dtype=object)

    coverage = float(np.mean(accepted_mask))
    reject_rate = 1.0 - coverage

    accepted_true = y_true_str[accepted_mask]
    accepted_pred = y_pred_reject[accepted_mask]

    metrics = {
        "tau": float(tau),
        "coverage": coverage,
        "reject_rate": reject_rate,
    }

    if np.any(accepted_mask):
        metrics["accepted_accuracy"] = float(
            accuracy_score(accepted_true, accepted_pred)
        )
        metrics["accepted_macro_f1"] = float(
            f1_score(accepted_true, accepted_pred, average="macro")
        )
    else:
        metrics["accepted_accuracy"] = float("nan")
        metrics["accepted_macro_f1"] = float("nan")

    return metrics, y_true_str, y_pred_reject, confs


def predict_window_with_no_action(artifact, x_row):
    model = artifact["model"]
    tau = artifact.get("no_action_threshold", 0.60)
    gestures = artifact["gestures"]
    inv_map = {i: g for i, g in enumerate(gestures)}

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"Model {type(model).__name__} does not support predict_proba in current config."
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


# Change windows format


def _sample_labels_to_intervals(sample_labels, timestamps_ms):
    if len(sample_labels) != len(timestamps_ms):
        raise ValueError("sample_labels and timestamps_ms must have same length")

    if len(sample_labels) == 0:
        return [], []

    interval_labels = []
    interval_timestamps_ms = []

    start_idx = 0
    current_label = sample_labels[0]

    for i in range(1, len(sample_labels)):
        if sample_labels[i] != current_label:
            start_ms = float(timestamps_ms[start_idx])
            end_ms = float(timestamps_ms[i - 1])
            interval_labels.append(str(current_label))
            interval_timestamps_ms.append((start_ms, end_ms))
            start_idx = i
            current_label = sample_labels[i]

    start_ms = float(timestamps_ms[start_idx])
    end_ms = float(timestamps_ms[-1])
    interval_labels.append(str(current_label))
    interval_timestamps_ms.append((start_ms, end_ms))

    return interval_labels, interval_timestamps_ms


# Real dataset build when needed


def _build_real_dataset(
    cfg,
    subject_id: str,
    session_date: str,
    session_id: int,
    channels: list[str] | None = None,
):
    fs = cfg.sample_rate_hz

    raw_path_str = expand_template(
        cfg.raw_file_template,
        subject_id=subject_id,
        session_date=session_date,
        session_id=session_id,
    )
    raw_path = Path(raw_path_str)

    if not raw_path.exists():
        raise FileNotFoundError(f"Real EMG file not found at {raw_path}")

    df = pd.read_parquet(raw_path)

    if "label" not in df.columns or "timestamp_ms" not in df.columns:
        raise KeyError("Real EMG parquet must contain 'label' and 'timestamp_ms'")

    sample_labels = df["label"].to_numpy()
    sample_timestamps_ms = df["timestamp_ms"].to_numpy(dtype=np.float64)

    interval_labels, interval_timestamps_ms = _sample_labels_to_intervals(
        sample_labels,
        sample_timestamps_ms,
    )

    if channels is None:
        if is_raw_mode(cfg):
            channels = [
                c for c in df.columns if c.startswith("emg_ch") and c.endswith("_raw")
            ]
        else:
            channels = [
                c for c in df.columns if c.startswith("emg_ch") and c.endswith("_env")
            ]

    if not channels:
        raise ValueError("No EMG channels found")

    processed_channels = []

    for col in channels:
        x = df[col].to_numpy(dtype=np.float32)

        if is_raw_mode(cfg):
            x_proc, _ = preprocess_raw(
                x,
                fs=fs,
                notch_50hz=(cfg.notch_hz is not None),
                band=cfg.bandpass_hz or (20.0, 450.0),
            )
        else:
            x_proc, _ = preprocess_envelope(x, fs=fs)

        processed_channels.append(np.asarray(x_proc, dtype=np.float32))

    signal_mc = np.stack(processed_channels, axis=1)  # (samples, channels)

    windows = window_segment_multichannel(
        signal_mc,
        fs=fs,
        window_ms=cfg.window_ms,
        hop_ms=cfg.hop_ms,
    )
    windows = np.transpose(windows, (0, 2, 1))  # (N, W, C) -> (N, C, W)
    _, window_labels, _ = window_signal(
        x=signal_mc[:, 0],
        fs=fs,
        window_ms=cfg.window_ms,
        hop_ms=cfg.hop_ms,
        labels=interval_labels,
        timestamps_ms=interval_timestamps_ms,
    )

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
    y, inv_map = _encode_labels(cfg, y)
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
            probability=True,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=0,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
        ),
    }
    results: list[dict] = []
    best_name: str | None = None
    best_model = None
    best_macro_f1 = -1.0
    best_cm: np.ndarray | None = None
    class_ids = sorted(np.unique(y_train))
    class_names = [inv_map[i] for i in class_ids]
    no_action_tau = 0.60

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
        cm = confusion_matrix(y_test, y_pred, labels=class_ids)

        print("Accuracy:", acc)
        print("Macro-F1:", macro_f1)
        print("Precision (macro):", precision)
        print("Recall (macro):", recall)
        print("Confusion matrix (rows=true, cols=pred):")
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
    results_path = Path(f"reports/{session_date}/best/results_table_{timestamp}.csv")
    ensure_parent_dir(str(results_path))
    df_results.to_csv(results_path, index=False)
    print("Saved results table to:", results_path)

    # 6c. Bar plot of model scores
    scores_plot_path = Path(
        f"reports/{session_date}/rest/model_scores_macro_f1_{timestamp}.png"
    )
    plot_model_scores_bar(
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
    plot_confusion_matrix(best_cm, class_names, cm_title, cm_path)
    print("Saved confusion matrix to:", cm_path)

    print("\n[7b] Plotting confusion matrices for ALL models...")
    for name, model in models.items():
        y_pred_all = model.predict(X_test)
        cm_all = confusion_matrix(y_test, y_pred_all, labels=class_ids)
        cm_title_all = f"Confusion matrix - {name}"
        cm_path_all = Path(
            f"reports/{session_date}/rest/confusion_matrix_{name}_{timestamp}.png"
        )
        plot_confusion_matrix(cm_all, class_names, cm_title_all, cm_path_all)
        print("Saved confusion matrix for", name, "to:", cm_path_all)

    # RF feature importance
    if "rf" in models:
        from src.classic_ml.features.time_domain import td_feature_names
        from src.classic_ml.features.freq_domain import feature_names_freq

        feature_names = td_feature_names() + feature_names_freq()
        fi_path = Path(
            f"reports/{session_date}/rest/feature_importance_rf_{timestamp}.png"
        )
        print("\n[7c] Plotting Random Forest feature importance...")
        plot_rf_feature_importance(models["rf"], feature_names, fi_path)
        print("Saved RF feature importance to:", fi_path)

    # XGBoost feature importance
    if "xgboost" in models:
        fi_xgb_path = Path(
            f"reports/{session_date}/rest/feature_importance_xgb_gain_{timestamp}.png"
        )
        print("\n[7d] Plotting XGBoost feature importance...")
        plot_xgb_feature_importance(models["xgboost"], fi_xgb_path)
        print("Saved XGBoost feature importance to:", fi_xgb_path)

    # F1 score
    print("\n[7e] Plotting per-class F1 scores...")
    y_pred_best = best_model.predict(X_test)
    f1_path = Path(f"reports/{session_date}/best/per_class_f1_{timestamp}.png")
    plot_per_class_f1(y_test, y_pred_best, class_ids, class_names, f1_path)
    print("Saved per-class F1 plot to:", f1_path)

    # PCA
    print("\n[7f] Plotting PCA 2D feature space...")
    pca_path = Path(f"reports/{session_date}/best/pca_2d_{timestamp}.png")
    plot_pca_2d(X, y, class_ids, class_names, pca_path)
    print("Saved PCA 2D plot to:", pca_path)

    # t-SNE
    print("\n[7g] Plotting t-SNE 2D feature space...")
    tsne_path = Path(f"reports/{session_date}/online/tsne_2d_{timestamp}.png")
    plot_tsne_2d(X, y, class_ids, class_names, tsne_path)
    print("Saved t-SNE 2D plot to:", tsne_path)

    print("\n[7g-3d] Plotting t-SNE 3D feature space...")
    tsne_3d_path = Path(f"reports/{session_date}/best/tsne_3d_{timestamp}.png")
    plot_tsne_3d(X, y, class_ids, class_names, tsne_3d_path)
    print("Saved t-SNE 3D plot to:", tsne_3d_path)

    # UMAP
    print("\n[7h] Plotting UMAP 2D feature space...")
    umap_path = Path(f"reports/{session_date}/online/umap_2d_{timestamp}.png")
    plot_umap_2d(X, y, class_ids, class_names, umap_path)
    print("Saved UMAP 2D plot to:", umap_path)

    print("\n[7h-3d] Plotting UMAP 3D feature space...")
    umap_3d_path = Path(f"reports/{session_date}/best/umap_3d_{timestamp}.png")
    plot_umap_3d(X, y, class_ids, class_names, umap_3d_path)
    print("Saved UMAP 3D plot to:", umap_3d_path)

    rejection_results = None
    rejection_results_path = None

    if hasattr(best_model, "predict_proba"):
        rejection_results, y_true_reject, y_pred_reject, confs = (
            _evaluate_with_rejection(
                best_model,
                X_test,
                y_test,
                inv_map,
                tau=no_action_tau,
            )
        )

        print(f"tau = {rejection_results['tau']:.2f}")
        print(f"coverage = {rejection_results['coverage']:.3f}")
        print(f"reject_rate = {rejection_results['reject_rate']:.3f}")
        print(f"accepted_accuracy = {rejection_results['accepted_accuracy']:.3f}")
        print(f"accepted_macro_f1 = {rejection_results['accepted_macro_f1']:.3f}")

        rejection_results_path = Path(
            f"reports/{session_date}/best/no_action_eval_{timestamp}.json"
        )
        ensure_parent_dir(str(rejection_results_path))
        with open(rejection_results_path, "w", encoding="utf-8") as f:
            json.dump(rejection_results, f, indent=2)

        print("Saved NO ACTION evaluation to:", rejection_results_path)
    else:
        print("Best model has no predict_proba, skipping NO ACTION evaluation.")

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
        "no_action_threshold": float(no_action_tau),
        "supports_predict_proba": hasattr(best_model, "predict_proba"),
        "rejection_results": rejection_results,
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
            "no_action_threshold": float(no_action_tau),
            "supports_predict_proba": hasattr(model, "predict_proba"),
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
        "no_action_threshold": float(no_action_tau),
        "rejection_results": rejection_results,
        "rejection_results_path": (
            str(rejection_results_path) if rejection_results_path else None
        ),
    }
    run_json_path = model_artifact_path.with_suffix(".run.json")
    with open(run_json_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print("Saved run metadata to:", run_json_path)


if __name__ == "__main__":
    main()
