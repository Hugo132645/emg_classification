from __future__ import annotations

import json
from pathlib import Path
from time import time
from datetime import datetime

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
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.common.io.emg_loader import (
    _search_csv_files,
    load_emg_files,
    process_emg_for_windowing,
)
from src.common.io.schemas import (
    ensure_parent_dir,
    expand_template,
    load_cfg,
    sanity_check_cfg,
)
from src.common.preprocessing.pipelines import preprocess_envelope
from src.common.preprocessing.windowing import window_signal_np

from src.classic_ml.datasets.classic_ml_dataset import build_classic_ml_dataset
from src.classic_ml.features.freq_domain import feature_names_freq
from src.classic_ml.features.time_domain import td_feature_names
from src.classic_ml.models.train_classic import (
    _encode_labels,
    _estimate_latency_ms,
    _evaluate_with_rejection,
)
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

# Online validation settings

DATA_ROOT = "data/input_data/ninapro_db1"
CSV_DIR_NAME = "csv_files"
KEEP_EXERCISES = {2}  # only exercise 2
REBUILD_CSV = False  # set True if you want to regenerate csv_files
NO_ACTION_TAU = 0.60
TEST_SIZE = 0.20
RANDOM_STATE = 0


def _configure_ninapro_ex2_cfg(cfg):
    cfg.gestures = ["rest"] + [f"g{i}" for i in range(1, 18)]
    cfg.label_map = {name: i for i, name in enumerate(cfg.gestures)}
    return cfg


def _label_int_to_str_ex2(label_int: int) -> str | None:
    if label_int == 0:
        return "rest"
    if 1 <= label_int <= 17:
        return f"g{label_int}"
    return None


def _map_labels_ex2(labels_int: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mapped = []
    keep_mask = []

    for lab in labels_int:
        name = _label_int_to_str_ex2(int(lab))
        if name is None:
            keep_mask.append(False)
        else:
            keep_mask.append(True)
            mapped.append(name)

    return np.array(mapped, dtype=object), np.array(keep_mask, dtype=bool)


def _feature_names_multichannel(n_channels: int) -> list[str]:
    base = td_feature_names() + feature_names_freq()
    names = []
    for ch in range(n_channels):
        names.extend([f"ch{ch+1}_{name}" for name in base])
    return names


def _preprocess_multichannel_envelope(x: np.ndarray, fs: float) -> np.ndarray:
    processed_channels = []

    for ch in range(x.shape[1]):
        x_ch = x[:, ch].astype(np.float32)
        x_proc, _ = preprocess_envelope(
            x_ch,
            fs=fs,
            lp_cut=10.0,
            lp_order=2,
            norm="zscore",
        )
        processed_channels.append(np.asarray(x_proc, dtype=np.float32))

    signal_mc = np.stack(processed_channels, axis=1)
    return signal_mc


def _build_online_windows_from_csv(
    cfg, csv_path: str
) -> tuple[np.ndarray, np.ndarray, int]:
    x, labels_int, rerepetition, subject, exercise = process_emg_for_windowing(
        csv_path,
        drop_rest=False,
    )

    if exercise not in KEEP_EXERCISES:
        raise ValueError(f"exercise={exercise} not in KEEP_EXERCISES={KEEP_EXERCISES}")

    labels_str, keep_mask = _map_labels_ex2(labels_int)
    x = x[keep_mask]

    if len(x) == 0:
        raise ValueError(f"No samples left after label filtering for file: {csv_path}")

    fs = cfg.sample_rate_hz
    n_channels = x.shape[1]

    # Ninapro DB1 temporary validation path:
    # treat input as envelope-like and keep pipeline minimal
    signal_mc = _preprocess_multichannel_envelope(x, fs=fs)

    # window_signal_np with multichannel input returns (N, W, C)
    windows, window_labels, times_ms = window_signal_np(
        signal_mc,
        fs=fs,
        window_ms=cfg.window_ms,
        hop_ms=cfg.hop_ms,
        labels=labels_str,
    )

    # classic feature extractors expect (N, C, W)
    windows = np.transpose(windows, (0, 2, 1))

    return windows, np.asarray(window_labels, dtype=object), n_channels


def _build_online_dataset(cfg) -> tuple[np.ndarray, np.ndarray, object, int]:
    data_root = Path(DATA_ROOT)
    csv_root = data_root / CSV_DIR_NAME

    if REBUILD_CSV or not csv_root.exists() or len(list(csv_root.rglob("*.csv"))) == 0:
        print("[online] Building CSV files from .mat files...")
        load_emg_files(
            input_path=str(data_root),
            output_dir_name=CSV_DIR_NAME,
        )

    csv_files = _search_csv_files(str(csv_root))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_root}")

    windows_blocks = []
    labels_blocks = []
    n_channels_ref = None

    for csv_path in csv_files:
        try:
            windows_i, labels_i, n_channels = _build_online_windows_from_csv(
                cfg, csv_path
            )
        except ValueError as e:
            print(f"[skip] {Path(csv_path).name}: {e}")
            continue

        if n_channels_ref is None:
            n_channels_ref = n_channels
        elif n_channels != n_channels_ref:
            raise ValueError(
                f"Inconsistent number of channels across files: "
                f"{n_channels_ref} vs {n_channels}"
            )

        windows_blocks.append(windows_i)
        labels_blocks.append(labels_i)
        print(
            f"[ok] {Path(csv_path).name}: windows={windows_i.shape}, labels={labels_i.shape}"
        )

    if not windows_blocks:
        raise ValueError("No usable files remained after filtering.")

    print("[debug] finished csv loop")
    print(f"[debug] number of kept files: {len(windows_blocks)}")
    print("[debug] concatenating windows...")
    windows = np.concatenate(windows_blocks, axis=0)
    print("[debug] concatenating labels...")
    window_labels = np.concatenate(labels_blocks, axis=0)
    print("[debug] windows shape after concat:", windows.shape)
    print("[debug] labels shape after concat:", window_labels.shape)
    print("[debug] building classic ML dataset...")

    windows = np.concatenate(windows_blocks, axis=0)
    window_labels = np.concatenate(labels_blocks, axis=0)

    X, y, scaler = build_classic_ml_dataset(
        windows=windows,
        window_labels=window_labels,
        fs=cfg.sample_rate_hz,
        mode="time+freq",
        scale=True,
    )

    return X, y, scaler, int(n_channels_ref)


def main():
    # 1. Load config and override label space only for this validation script
    cfg = load_cfg()
    sanity_check_cfg(cfg)  # sanity-check original config first
    cfg = _configure_ninapro_ex2_cfg(cfg)

    print("Config loaded.")
    print(f"sample_rate_hz: {cfg.sample_rate_hz}")
    print(f"window_ms: {cfg.window_ms}, hop_ms: {cfg.hop_ms}")
    print(f"online gestures: {cfg.gestures}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"KEEP_EXERCISES: {KEEP_EXERCISES}")

    session_date = datetime.now().strftime("%Y-%m-%d")

    # 2. Build online dataset
    print("\n[1] Building dataset from Ninapro DB1 Exercise 2...")
    X, y, scaler, n_channels = _build_online_dataset(cfg)
    y, inv_map = _encode_labels(cfg, y)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("n_channels:", n_channels)
    print("unique encoded labels:", np.unique(y))

    # 3. Train-test split
    print("\n[2] Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

    # 4. Define models - copied from original pipeline as much as possible
    print("\n[3] Defining models...")
    models = {
        "logreg": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            multi_class="auto",
        ),
        # "svm_linear": LinearSVC(
        #    C=1.0,
        #   max_iter=5000,
        #    dual=False,
        # ),
        # "svm_linear": SVC(
        #    kernel="linear",
        #    C=1.0,
        #    # probability=True,
        # ),
        # "svm_rbf": SVC(
        #   kernel="rbf",
        #   C=1.0,
        #   gamma="scale",
        #   probability=True,
        # ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
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

    # 5. Train and evaluate each model
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

    # 6. Summary table
    print("\n[5] Summary of model scores:")
    print(f"{'Model':<12} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 34)
    for r in results:
        print(f"{r['name']:<12} {r['accuracy']:>10.3f} {r['macro_f1']:>10.3f}")
    print(f"\nBest model: {best_name} (macro-F1 = {best_macro_f1:.3f})")

    # 6b. Results table
    timestamp = int(time())
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="macro_f1", ascending=False)
    print("\n[5b] Results table:")
    print(df_results)

    results_path = Path(f"reports/{session_date}/online/results_table_{timestamp}.csv")
    ensure_parent_dir(str(results_path))
    df_results.to_csv(results_path, index=False)
    print("Saved results table to:", results_path)

    # 6c. Bar plot of model scores
    scores_plot_path = Path(
        f"reports/{session_date}/online/model_scores_macro_f1_{timestamp}.png"
    )
    plot_model_scores_bar(
        df_results,
        metric="macro_f1",
        out_path=scores_plot_path,
        title="Model comparison on online data (Macro-F1)",
    )
    print("Saved model scores plot to:", scores_plot_path)

    # 7. Latency estimate
    print("\n[6] Estimating latency for best model...")
    latency_ms = _estimate_latency_ms(best_model, X_test, n_reps=200)
    print(f"Estimated model.predict latency per window: {latency_ms:.4f} ms")

    # 8. Best confusion matrix
    print("\n[7] Plotting confusion matrix for best model...")
    cm_title = f"Confusion matrix - {best_name}"
    cm_path = Path(
        f"reports/{session_date}/online/confusion_matrix_{best_name}_{timestamp}.png"
    )
    plot_confusion_matrix(best_cm, class_names, cm_title, cm_path)
    print("Saved confusion matrix to:", cm_path)

    # 8b. All confusion matrices
    print("\n[7b] Plotting confusion matrices for ALL models...")
    for name, model in models.items():
        y_pred_all = model.predict(X_test)
        cm_all = confusion_matrix(y_test, y_pred_all, labels=class_ids)
        cm_title_all = f"Confusion matrix - {name}"
        cm_path_all = Path(
            f"reports/{session_date}/online/confusion_matrix_{name}_{timestamp}.png"
        )
        plot_confusion_matrix(cm_all, class_names, cm_title_all, cm_path_all)
        print("Saved confusion matrix for", name, "to:", cm_path_all)

    feature_names = _feature_names_multichannel(n_channels)

    # 8c. RF feature importance
    if "rf" in models:
        fi_path = Path(
            f"reports/{session_date}/online/feature_importance_rf_{timestamp}.png"
        )
        print("\n[7c] Plotting Random Forest feature importance...")
        plot_rf_feature_importance(models["rf"], feature_names, fi_path)
        print("Saved RF feature importance to:", fi_path)

    # 8d. XGBoost feature importance
    if "xgboost" in models:
        fi_xgb_path = Path(
            f"reports/{session_date}/online/feature_importance_xgb_gain_{timestamp}.png"
        )
        print("\n[7d] Plotting XGBoost feature importance...")
        plot_xgb_feature_importance(models["xgboost"], fi_xgb_path)
        print("Saved XGBoost feature importance to:", fi_xgb_path)

    # 8e. Per-class F1
    print("\n[7e] Plotting per-class F1 scores...")
    y_pred_best = best_model.predict(X_test)
    f1_path = Path(f"reports/{session_date}/online/per_class_f1_{timestamp}.png")
    plot_per_class_f1(y_test, y_pred_best, class_ids, class_names, f1_path)
    print("Saved per-class F1 plot to:", f1_path)

    # 8f. PCA
    print("\n[7f] Plotting PCA 2D feature space...")
    pca_path = Path(f"reports/{session_date}/online/pca_2d_{timestamp}.png")
    plot_pca_2d(X, y, class_ids, class_names, pca_path)
    print("Saved PCA 2D plot to:", pca_path)

    # 8g. t-SNE
    print("\n[7g] Plotting t-SNE 2D feature space...")
    tsne_path = Path(f"reports/{session_date}/online/tsne_2d_{timestamp}.png")
    plot_tsne_2d(X, y, class_ids, class_names, tsne_path)
    print("Saved t-SNE 2D plot to:", tsne_path)

    print("\n[7g-3d] Plotting t-SNE 3D feature space...")
    tsne_3d_path = Path(f"reports/{session_date}/best/tsne_3d_{timestamp}.png")
    plot_tsne_3d(X, y, class_ids, class_names, tsne_3d_path)
    print("Saved t-SNE 3D plot to:", tsne_3d_path)

    # 8h. UMAP
    print("\n[7h] Plotting UMAP 2D feature space...")
    umap_path = Path(f"reports/{session_date}/online/umap_2d_{timestamp}.png")
    plot_umap_2d(X, y, class_ids, class_names, umap_path)
    print("Saved UMAP 2D plot to:", umap_path)

    print("\n[7h-3d] Plotting UMAP 3D feature space...")
    umap_3d_path = Path(f"reports/{session_date}/best/umap_3d_{timestamp}.png")
    plot_umap_3d(X, y, class_ids, class_names, umap_3d_path)
    print("Saved UMAP 3D plot to:", umap_3d_path)

    # 8i. NO ACTION rejection layer
    print("\n[7i] Evaluating NO ACTION rejection layer...")
    rejection_results = None
    rejection_results_path = None

    if hasattr(best_model, "predict_proba"):
        rejection_results, y_true_reject, y_pred_reject, confs = (
            _evaluate_with_rejection(
                best_model,
                X_test,
                y_test,
                inv_map,
                tau=NO_ACTION_TAU,
            )
        )

        print(f"tau = {rejection_results['tau']:.2f}")
        print(f"coverage = {rejection_results['coverage']:.3f}")
        print(f"reject_rate = {rejection_results['reject_rate']:.3f}")
        print(f"accepted_accuracy = {rejection_results['accepted_accuracy']:.3f}")
        print(f"accepted_macro_f1 = {rejection_results['accepted_macro_f1']:.3f}")

        rejection_results_path = Path(
            f"reports/{session_date}/online/no_action_eval_{timestamp}.json"
        )
        ensure_parent_dir(str(rejection_results_path))
        with open(rejection_results_path, "w", encoding="utf-8") as f:
            json.dump(rejection_results, f, indent=2)

        print("Saved NO ACTION evaluation to:", rejection_results_path)
    else:
        print("Best model has no predict_proba, skipping NO ACTION evaluation.")

    # 9. Save best model artifact
    print("\n[8] Saving model artifact...")
    model_artifact_path_str = expand_template(
        cfg.model_file_template,
        track="classic_ml_online_best",
        subject_id="NINAPRO_EX2_ALL",
        session_date=session_date,
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
        "use_dummy": False,
        "data_source": "ninapro_db1_ex2_online_validation",
        "keep_exercises": sorted(list(KEEP_EXERCISES)),
        "no_action_threshold": float(NO_ACTION_TAU),
        "supports_predict_proba": hasattr(best_model, "predict_proba"),
        "rejection_results": rejection_results,
    }
    joblib.dump(artifact, model_artifact_path)
    print("Saved BEST model artifact to:", model_artifact_path)

    # 9b. Save all trained models as separate artifacts
    all_model_base_path_str = expand_template(
        cfg.model_file_template,
        track="classic_ml_online",
        subject_id="NINAPRO_EX2_ALL",
        session_date=session_date,
        session_id=0,
        timestamp=timestamp,
        model_artifact_ext="joblib",
    )
    all_model_base_path = Path(all_model_base_path_str)

    all_model_artifacts: dict[str, str] = {}
    for name, model in models.items():
        per_model_path = all_model_base_path.with_name(
            f"{all_model_base_path.stem}_{name}.joblib"
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
            "use_dummy": False,
            "data_source": "ninapro_db1_ex2_online_validation",
            "keep_exercises": sorted(list(KEEP_EXERCISES)),
            "no_action_threshold": float(NO_ACTION_TAU),
            "supports_predict_proba": hasattr(model, "predict_proba"),
        }
        joblib.dump(per_artifact, per_model_path)
        all_model_artifacts[name] = str(per_model_path)

    print("Saved ALL model artifacts:", all_model_artifacts)

    # 10. Save run.json
    run_info = {
        "model_artifact": str(model_artifact_path),
        "best_model_name": best_name,
        "macro_f1": float(best_macro_f1),
        "latency_ms": float(latency_ms),
        "results": results,
        "class_names": list(class_names),
        "timestamp": timestamp,
        "use_dummy": False,
        "data_source": "ninapro_db1_ex2_online_validation",
        "keep_exercises": sorted(list(KEEP_EXERCISES)),
        "results_table_csv": str(results_path),
        "model_scores_plot": str(scores_plot_path),
        "all_model_artifacts": all_model_artifacts,
        "no_action_threshold": float(NO_ACTION_TAU),
        "rejection_results": rejection_results,
        "rejection_results_path": (
            str(rejection_results_path) if rejection_results_path else None
        ),
    }

    run_info_path = model_artifact_path.with_name(f"run_{timestamp}.json")
    ensure_parent_dir(str(run_info_path))
    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print("Saved run metadata to:", run_info_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
