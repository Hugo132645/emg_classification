from __future__ import annotations

from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from src.common.io.schemas import load_cfg, sanity_check_cfg, ensure_parent_dir
from src.classic_ml.models.train_classic import (
    _build_dummy_dataset,
    _build_real_dataset,
)

# Set this to True only if you want to quickly test with dummy EMG
use_dummy = True


def _build_dataset():
    cfg = load_cfg()
    sanity_check_cfg(cfg)
    print("Config loaded.")
    print(f"sample_rate_hz: {cfg.sample_rate_hz}")
    print(f"window_ms: {cfg.window_ms}, hop_ms: {cfg.hop_ms}")
    print(f"gestures: {cfg.gestures}")
    print(f"use_dummy = {use_dummy}")

    if use_dummy:
        print("Building dataset from DUMMY EMG")
        X, y, _ = _build_dummy_dataset(cfg)
    else:
        print("Building dataset from REAL EMG")
        X, y, _ = _build_real_dataset(cfg)

    print("Dataset shapes:", X.shape, y.shape)
    return X, y


def _compute_regularization_curves(X: np.ndarray, y: np.ndarray):
    # C range
    C_values = np.logspace(-3, 3, 13)

    # All C-based models
    models = {
        "logreg": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            multi_class="auto",
        ),
        "svm_linear": SVC(
            kernel="linear",
            C=1.0,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=1.0,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scoring = "f1_macro"

    curves = {}

    print("\nRunning cross-validated regularization curves")
    for name, base_model in models.items():
        mean_scores = []
        std_scores = []
        print(f"{name}")
        for C in C_values:
            model = base_model.__class__(**base_model.get_params())
            model.set_params(C=C)
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )
            mean_scores.append(scores.mean())
            std_scores.append(scores.std())
        curves[name] = {
            "C_values": C_values,
            "mean_scores": np.array(mean_scores),
            "std_scores": np.array(std_scores),
        }

        best_idx = int(np.argmax(mean_scores))
        print(
            f"best C for {name}: {C_values[best_idx]:.4g} "
            f"(mean {scoring} = {mean_scores[best_idx]:.3f})"
        )

    return curves


def _plot_curves(curves: dict):
    timestamp = int(time())
    out_path = Path(f"reports/regularization/regularization_C_{timestamp}.png")
    ensure_parent_dir(str(out_path))

    fig, ax = plt.subplots(figsize=(7, 5))

    for name, data in curves.items():
        C_vals = data["C_values"]
        means = data["mean_scores"]
        stds = data["std_scores"]

        ax.semilogx(C_vals, means, marker="o", label=name)
        ax.fill_between(C_vals, means - stds, means + stds, alpha=0.2)

    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Regularization curves for C")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print("\nSaved regularization plot to:", out_path)


def main():
    X, y = _build_dataset()
    curves = _compute_regularization_curves(X, y)
    _plot_curves(curves)


if __name__ == "__main__":
    main()
