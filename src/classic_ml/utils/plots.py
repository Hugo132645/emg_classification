from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from src.common.io.schemas import ensure_parent_dir

try:
    import umap
except ImportError:
    umap = None


def plot_confusion_matrix(
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


def plot_model_scores_bar(
    df_results,
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


def plot_rf_feature_importance(rf_model, feature_names, out_path: Path) -> None:
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


def plot_xgb_feature_importance(xgb_model, out_path: Path) -> None:
    from xgboost import plot_importance

    plt.figure(figsize=(8, 6))
    plot_importance(
        xgb_model,
        importance_type="gain",
        max_num_features=20,
    )
    plt.title("XGBoost Feature Importance (gain)")
    plt.tight_layout()
    ensure_parent_dir(str(out_path))
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_per_class_f1(
    y_true,
    y_pred,
    class_ids,
    class_names,
    out_path: Path,
) -> None:
    f1_vals = f1_score(y_true, y_pred, average=None, labels=class_ids)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(class_names, f1_vals)
    ax.set_title("Per-Class F1")
    ax.set_ylabel("F1 Score")

    plt.tight_layout()
    ensure_parent_dir(str(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pca_2d(
    X,
    y,
    class_ids,
    class_names,
    out_path: Path,
) -> None:
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    y_arr = np.asarray(y)

    for cls_id, cls_name in zip(class_ids, class_names):
        mask = y_arr == cls_id
        ax.scatter(X2[mask, 0], X2[mask, 1], s=12, label=cls_name)

    ax.set_title("PCA Feature Space (2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    plt.tight_layout()
    ensure_parent_dir(str(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tsne_2d(
    X,
    y,
    class_ids,
    class_names,
    out_path: Path,
    random_state: int = 0,
) -> None:
    X2 = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    ).fit_transform(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    y_arr = np.asarray(y)

    for cls_id, cls_name in zip(class_ids, class_names):
        mask = y_arr == cls_id
        ax.scatter(X2[mask, 0], X2[mask, 1], s=12, label=cls_name)

    ax.set_title("t-SNE Feature Space (2D)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()

    plt.tight_layout()
    ensure_parent_dir(str(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_umap_2d(
    X,
    y,
    class_ids,
    class_names,
    out_path: Path,
    random_state: int = 0,
) -> bool:
    if umap is None:
        print("Skipping UMAP 2D plot because umap-learn is not installed.")
        return False

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    )
    X2 = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    y_arr = np.asarray(y)

    for cls_id, cls_name in zip(class_ids, class_names):
        mask = y_arr == cls_id
        ax.scatter(X2[mask, 0], X2[mask, 1], s=12, label=cls_name)

    ax.set_title("UMAP Feature Space (2D)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()

    plt.tight_layout()
    ensure_parent_dir(str(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_tsne_3d(
    X,
    y,
    class_ids,
    class_names,
    out_path: Path,
    random_state: int = 0,
) -> None:
    X3 = TSNE(
        n_components=3,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    ).fit_transform(X)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    y_arr = np.asarray(y)

    for cls_id, cls_name in zip(class_ids, class_names):
        mask = y_arr == cls_id
        ax.scatter(
            X3[mask, 0],
            X3[mask, 1],
            X3[mask, 2],
            s=12,
            label=cls_name,
        )

    ax.set_title("t-SNE Feature Space (3D)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    ax.legend()

    plt.tight_layout()
    ensure_parent_dir(str(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_umap_3d(
    X,
    y,
    class_ids,
    class_names,
    out_path: Path,
    random_state: int = 0,
) -> bool:
    if umap is None:
        print("Skipping UMAP 3D plot because umap-learn is not installed.")
        return False

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    )
    X3 = reducer.fit_transform(X)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    y_arr = np.asarray(y)

    for cls_id, cls_name in zip(class_ids, class_names):
        mask = y_arr == cls_id
        ax.scatter(
            X3[mask, 0],
            X3[mask, 1],
            X3[mask, 2],
            s=12,
            label=cls_name,
        )

    ax.set_title("UMAP Feature Space (3D)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.legend()

    plt.tight_layout()
    ensure_parent_dir(str(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True
