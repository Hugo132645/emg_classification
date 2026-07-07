# Importing all needed files
from lib2to3.fixes.fix_input import context

import numpy as np
import torch
import torch.nn as nn

import optuna
from onnxruntime.transformers.optimizer import MODEL_TYPES

from src.rnn.models.model_rnn import GRUModel, LSTMModel
from src.common.io.schemas import (
    load_cfg,
    gesture_to_id,
    expand_template,
    ensure_parent_dir,
)
from src.rnn.datasets.sequence_dataset import (
    Standardizer,
    FeatureSequenceDataset,
    encode_labels_from_cfg,
)
from src.rnn.features.seq_features import compute_seq_features
from src.common.io.dummy_data import generate_dummy_emg
from torch.utils.data import DataLoader
from src.common.preprocessing.windowing import window_signal
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset, Subset
from src.rnn.datasets.sequence_dataset import Standardizer, FeatureSequenceDataset
from src.common.preprocessing.windowing import window_signal_np
from src.common.io.emg_loader import _search_csv_files, process_emg_for_windowing
import copy
from dataclasses import dataclass

# Loading config, seed, device
config = load_cfg()
sd = 42
dvc = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def get_targets_preds(loader, model, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lengths = batch["length"].to(device)

            logits = model(x, lengths)
            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.detach().cpu())
            all_targets.append(y.detach().cpu())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        return all_targets, all_preds


def _cont_runs(labels: np.ndarray, rerepetition: np.ndarray):
    if len(labels) == 0:
        return []

    change_idx = np.where((np.diff(labels) != 0) | (np.diff(rerepetition) != 0))[0] + 1

    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx, len(labels)]

    return [
        (int(s), int(e), int(labels[s]), int(rerepetition[s]))
        for s, e in zip(starts, ends)
    ]


def make_dataloaders_dummy(
    cfg,
    seconds: int = 100,
    batch_size: int = 32,
    seq_length: int = 32,
    seq_stride: int = 8,
    val_ratio: float = 0.3,
    seed: int = None,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    sample_rate = cfg.sample_rate_hz

    # Generating dummy_data
    dat, raw_labels, timestamps = generate_dummy_emg(
        seconds, sample_rate, cfg.gestures, block_s=5, seed=seed
    )

    # Windowing dummy_data
    windows, raw_labels, timestamps = window_signal(
        dat["signal"],
        sample_rate,
        labels=raw_labels,
        timestamps_ms=timestamps,
        window_ms=cfg.window_ms,
        hop_ms=cfg.hop_ms,
    )
    ft_vectors, ft_names = compute_seq_features(
        windows, sample_rate, deltas=True, spectral_feat=True
    )
    labels_int = encode_labels_from_cfg(cfg, raw_labels)

    # Creating a Standardizer
    std = Standardizer()
    dataset = FeatureSequenceDataset(
        feature_vectors=ft_vectors,
        labels=labels_int,
        seq_len=seq_length,
        seq_stride=seq_stride,
        standardizer=std,
        device=None,
    )

    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)
    n_val = int(val_ratio * n)
    val_idxs = indices[:n_val]
    train_idxs = indices[n_val:]

    # Dividing into training and validation dataset
    train_dataset = torch.utils.data.Subset(dataset, train_idxs)
    vals_dataset = torch.utils.data.Subset(dataset, val_idxs)

    # Collation into torch.tensors for training and validation
    def collate_fn(batch):
        return {
            "x": torch.stack([b["x"] for b in batch], dim=0),  # [B, L, D]
            "y_seq": torch.stack([b["y_seq"] for b in batch], dim=0),  # [B, L]
            "y": torch.stack([b["y"] for b in batch], dim=0),  # [B]
            "length": torch.tensor(
                [int(b["length"]) for b in batch], dtype=torch.long
            ),  # [B]
            "names": batch[0]["names"],
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )
    val_loader = DataLoader(
        vals_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    label_names = list(cfg.gestures)
    label_map = {name: i for i, name in enumerate(label_names)}

    return (
        train_loader,
        val_loader,
        input_dim,
        num_classes,
        ft_names,
        std,
        label_names,
        label_map,
    )


def make_dataloaders_np(
    cfg,
    data_path: str,
    batch_size: int = 32,
    seq_length: int = 32,
    seq_stride: int = 1,
    seed: int | None = None,
    drop_rest: bool = False,
    train_reps: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
    val_reps: tuple[int, ...] = (9, 10),
    allowed_exercises: tuple[int, ...] | None = None,
    selected_channels: tuple[int, ...] | None = None,
    include_rest: bool = True,
    rest_val_ratio: float = 0.2,
):
    if include_rest and drop_rest:
        raise ValueError("include_rest=True requires drop_rest=False")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    rng = np.random.RandomState(seed if seed is not None else 42)

    sample_rate = cfg.sample_rate_hz
    csv_files = sorted(Path(p) for p in _search_csv_files(data_path))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_path}")

    file_records = []
    label_pairs = set()

    for csv_path in csv_files:
        # For continuous robotic-arm training, do NOT drop rest here.
        x, raw_labels, rerepetition, subject, exercise = process_emg_for_windowing(
            str(csv_path),
            drop_rest=False,
        )

        if allowed_exercises is not None and int(exercise) not in allowed_exercises:
            continue

        if selected_channels is not None:
            x = x[:, list(selected_channels)]

        if not include_rest:
            keep = raw_labels != 0
            x = x[keep]
            raw_labels = raw_labels[keep]
            rerepetition = rerepetition[keep]

        if len(x) == 0:
            continue

        file_records.append((csv_path, x, raw_labels, rerepetition, subject, exercise))

        for lab in np.unique(raw_labels):
            lab = int(lab)
            if lab == 0 and not include_rest:
                continue
            label_pairs.add((int(exercise), lab))

    if not file_records:
        raise ValueError("No usable EMG")

    sorted_pairs = sorted(label_pairs)
    label_map = {pair: i for i, pair in enumerate(sorted_pairs)}
    label_names = [f"E{ex}_G{lab}" for ex, lab in sorted_pairs]

    feature_blocks = []
    label_blocks = []
    rep_blocks = []
    ft_names = None

    min_len = int(sample_rate * cfg.window_ms / 1000)

    for csv_path, x, raw_labels, rerepetition, subject, exercise in file_records:
        if len(x) < min_len:
            continue

        global_labels = np.array(
            [label_map[(int(exercise), int(lbl))] for lbl in raw_labels],
            dtype=np.int64,
        )

        windows, window_labels, _ = window_signal_np(
            x=x,
            fs=sample_rate,
            window_ms=cfg.window_ms,
            hop_ms=cfg.hop_ms,
            labels=global_labels,
        )

        _, window_reps, _ = window_signal_np(
            x=x[:, 0],
            fs=sample_rate,
            window_ms=cfg.window_ms,
            hop_ms=cfg.hop_ms,
            labels=rerepetition,
        )

        if len(windows) == 0:
            continue

        features, names = compute_seq_features(
            windows=windows,
            fs=sample_rate,
            basic_feat=True,
            shape_feat=True,
            spectral_feat=True,
            deltas=False,
        )

        if ft_names is None:
            ft_names = names

        feature_blocks.append(features)
        label_blocks.append(np.asarray(window_labels, dtype=np.int64))
        rep_blocks.append(np.asarray(window_reps, dtype=np.int64))

        print(
            f"Loaded {csv_path.name}: subject={subject}, exercise={exercise}, "
            f"windows={len(window_labels)}, labels={sorted(np.unique(window_labels).tolist())}, "
            f"reps={sorted(np.unique(window_reps).tolist())}"
        )

    if not feature_blocks:
        raise ValueError("No feature blocks were created.")

    # Fit standardizer only on training-target windows.
    train_standardizer_rows = []

    for features, labels, reps in zip(feature_blocks, label_blocks, rep_blocks):
        train_target_mask = np.isin(reps, train_reps)

        # Rest windows often have repetition 0. Keep some rest in training.
        if include_rest:
            rest_mask = labels == label_map.get((2, 0), -999999)
            rest_train_mask = rest_mask & (rng.rand(len(labels)) >= rest_val_ratio)
            train_target_mask = train_target_mask | rest_train_mask

        if np.any(train_target_mask):
            train_standardizer_rows.append(features[train_target_mask])

    if not train_standardizer_rows:
        raise ValueError("No training windows available for standardizer.")

    std = Standardizer().fit(np.vstack(train_standardizer_rows))

    train_datasets = []
    val_datasets = []

    for features, labels, reps in zip(feature_blocks, label_blocks, rep_blocks):
        dataset = FeatureSequenceDataset(
            feature_vectors=features,
            labels=labels,
            seq_len=seq_length,
            seq_stride=seq_stride,
            standardizer=std,
            device=None,
        )

        train_indices = []
        val_indices = []

        # Dataset item i usually corresponds to a sequence starting at i * seq_stride.
        # Target is normally the last label in the sequence.
        for ds_idx in range(len(dataset)):
            start = ds_idx * seq_stride
            target_idx = start + seq_length - 1

            if target_idx >= len(labels):
                continue

            target_label = int(labels[target_idx])
            target_rep = int(reps[target_idx])

            is_rest = label_names[target_label].endswith("_G0")

            if is_rest and include_rest:
                if rng.rand() < rest_val_ratio:
                    val_indices.append(ds_idx)
                else:
                    train_indices.append(ds_idx)
            elif target_rep in train_reps:
                train_indices.append(ds_idx)
            elif target_rep in val_reps:
                val_indices.append(ds_idx)

        if train_indices:
            train_datasets.append(Subset(dataset, train_indices))

        if val_indices:
            val_datasets.append(Subset(dataset, val_indices))

    if not train_datasets:
        raise ValueError("No training sequences were created.")
    if not val_datasets:
        raise ValueError("No validation sequences were created.")

    train_dataset = ConcatDataset(train_datasets)
    vals_dataset = ConcatDataset(val_datasets)

    def collate_fn(batch):
        return {
            "x": torch.stack([b["x"] for b in batch], dim=0),
            "y_seq": torch.stack([b["y_seq"] for b in batch], dim=0),
            "y": torch.stack([b["y"] for b in batch], dim=0),
            "length": torch.tensor(
                [int(b["length"]) for b in batch],
                dtype=torch.long,
            ),
            "names": batch[0]["names"],
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )

    val_loader = DataLoader(
        vals_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    input_dim = feature_blocks[0].shape[1]
    num_classes = len(label_map)

    print("Continuous GRU dataset built")
    print("Train sequences:", len(train_dataset))
    print("Val sequences:", len(vals_dataset))
    print("Input dim:", input_dim)
    print("Num classes:", num_classes)
    print("Label names:", label_names)

    return (
        train_loader,
        val_loader,
        input_dim,
        num_classes,
        ft_names,
        std,
        label_names,
        label_map,
    )


def run_epoch(loader, model, criterion, optimizer=None, device="mps", grad_clip=None):
    # Running 1 epoch -> one-time training dataset
    if optimizer is None:
        model.eval()
    else:
        model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # Differentiating running epoch between training/validation data -> val not calculating gradient
    context = torch.no_grad() if optimizer is None else torch.enable_grad()
    with context:
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lengths = batch["length"].to(device)

            if optimizer is not None:
                optimizer.zero_grad()

            logits = model(x, lengths)
            loss = criterion(logits, y)

            if optimizer is not None:
                loss.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)

                optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += batch_size

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def build_model(
    model_type,
    input_dim,
    hidden_dim,
    num_classes,
    num_layers,
    dropout,
    bidirectional=False,
):
    model_type = model_type.lower()

    if model_type == "gru":
        return GRUModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
    elif model_type == "lstm":
        return LSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def train_save_model(
    model,
    train_loader,
    val_loader,
    device,
    input_dim,
    num_classes,
    ft_names,
    std,
    label_names,
    label_map,
    model_type="gru",
    hidden_dim=32,
    num_layers=2,
    dropout=0.2,
    bidirectional=False,
    lr=1e-3,
    weight_decay=1e-4,
    label_smoothing=0.05,
    grad_clip=1.0,
    epochs=30,
    early_stopping_patience=8,
    save_dir="./artifacts",
    seq_length=16,
    seq_stride=1,
    selected_channels=None,
    allowed_exercises=None,
    drop_rest=False,
    include_rest=True,
    preprocess_mode="none"
):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    train_loss_log, val_loss_log = [], []
    train_acc_log, val_acc_log = [], []

    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    best_model_path = expand_template(
        f"{save_dir}/best_{model_type}_{timestamp}.pt",
        model_type=model_type,
        timestamp=timestamp,
    )
    ensure_parent_dir(best_model_path)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss, train_acc = run_epoch(
            train_loader,
            model,
            criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
        )
        val_loss, val_acc = run_epoch(
            val_loader, model, criterion, optimizer=None, device=device
        )

        scheduler.step(val_acc)

        train_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        train_acc_log.append(train_acc)
        val_acc_log.append(val_acc)

        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            artifact = {
                "state_dict": best_state,
                "model_type": model_type,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "bidirectional": bidirectional,
                "dropout": dropout,
                "num_classes": num_classes,
                "label_names": label_names,
                "label_map": label_map,
                "feature_names": ft_names,
                "standardizer_mean": std.mean_,
                "standardizer_std": std.std_,
                "best_val_acc": best_val_acc,
                "seq_length": seq_length,
                "seq_stride": seq_stride,
                "selected_channels": selected_channels,
                "allowed_exercises": allowed_exercises,
                "drop_rest": drop_rest,
                "include_rest": include_rest,
                "preprocess_mode": preprocess_mode,
            }

            torch.save(artifact, best_model_path)
            print(f"Saved new best model to {best_model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {
        "train_loss": train_loss_log,
        "val_loss": val_loss_log,
        "train_acc": train_acc_log,
        "val_acc": val_acc_log,
        "best_val_acc": best_val_acc,
        "best_model_path": best_model_path,
    }
    return model, history


def objective(trial, cfg, data_path, device, seed=42):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    seq_length = trial.suggest_categorical("seq_length", [16, 32, 48])
    seq_stride = trial.suggest_categorical("seq_stride", [4, 8, 16])

    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15)

    train_loader, val_loader, input_dim, num_classes, ft_names, std, _, _ = (
        make_dataloaders_np(
            cfg=cfg,
            data_path=data_path,
            batch_size=batch_size,
            seq_length=seq_length,
            seq_stride=seq_stride,
            seed=seed,
            drop_rest=False,
            train_reps=(1, 2, 3, 4, 5, 6, 7, 8),
            val_reps=(9, 10),
            allowed_exercises=(2,),
            selected_channels=(0, 1),
        )
    )

    model = build_model(
        model_type="gru",
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    max_epochs = 10

    for epoch in range(max_epochs):
        run_epoch(
            train_loader,
            model,
            criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=1.0,
        )
        _, val_acc = run_epoch(
            val_loader, model, criterion, optimizer=None, device=device
        )

        best_val_acc = max(best_val_acc, val_acc)

        trial.report(best_val_acc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_acc


def run_hyperparameter_opt(cfg, data_path, device, n_trials=25, seed=42):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, cfg, data_path, device, seed), n_trials=n_trials
    )

    print("Best trial:")
    print(study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    return study


def load_best_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(
        model_type=checkpoint["model_type"],
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        bidirectional=checkpoint["bidirectional"],
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    std = Standardizer()
    std.mean_ = checkpoint["standardizer_mean"]
    std.std_ = checkpoint["standardizer_std"]

    return {
        "model": model,
        "standardizer": std,
        "feature_names": checkpoint["feature_names"],
        "label_names": checkpoint["label_names"],
        "label_map": checkpoint["label_map"],
        "checkpoint": checkpoint,
    }


def predict_with_model(model, loader, device, label_names=None):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            lengths = batch["length"].to(device)

            logits = model(x, lengths)
            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu())

            if "y" in batch:
                all_targets.append(batch["y"].cpu())

    y_pred = torch.cat(all_preds).numpy()

    result = {"y_pred": y_pred}
    if label_names is not None:
        result["pred_names"] = [label_names[i] for i in y_pred]

    if len(all_targets) > 0:
        y_true = torch.cat(all_targets).numpy()
        result["y_true"] = y_true
        if label_names is not None:
            result["true_names"] = [label_names[i] for i in y_true]

    return result


if __name__ == "__main__":
    sd = 42
    config = load_cfg()
    seq_length = 16
    seq_stride = 1
    selected_channels = (0, 1)
    allowed_exercises = (2,)
    drop_rest = False
    include_rest = True
    preprocess_mode = "none"

    (
        train_loader,
        val_loader,
        input_dim,
        num_classes,
        ft_names,
        std,
        label_names,
        label_map,
    ) = make_dataloaders_np(
        cfg=config,
        data_path="data/input_data/ninapro_db1/csv_files",
        batch_size=32,
        seq_length=seq_length,
        seq_stride=seq_stride,
        include_rest=include_rest,
        drop_rest=drop_rest,
        train_reps=(1, 2, 3, 4, 5, 6, 7, 8),
        val_reps=(9, 10),
        seed=sd,
        allowed_exercises=allowed_exercises,
        selected_channels=selected_channels,
        rest_val_ratio=0.1,
    )

    hidden_dim = 64
    num_layers = 2
    dropout = 0.2
    lr = 5e-4
    weight_decay = 1e-4
    epochs = 30

    gru_model = build_model(
        model_type="gru",
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=True,
    ).to(dvc)

    gru_model, history = train_save_model(
        model=gru_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=dvc,
        input_dim=input_dim,
        num_classes=num_classes,
        ft_names=ft_names,
        std=std,
        label_names=label_names,
        label_map=label_map,
        model_type="gru",
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=True,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=0.05,
        grad_clip=1.0,
        epochs=epochs,
        early_stopping_patience=4,
        save_dir="./artifacts",
        seq_length=seq_length,
        seq_stride=seq_stride,
        selected_channels=selected_channels,
        allowed_exercises=allowed_exercises,
        drop_rest=drop_rest,
        include_rest=include_rest,
        preprocess_mode=preprocess_mode,
    )

    print("Best model path:", history["best_model_path"])

    loaded = load_best_model(history["best_model_path"], dvc)
    preds = predict_with_model(loaded["model"], val_loader, dvc, loaded["label_names"])

    y_true = preds["y_true"]
    y_pred = preds["y_pred"]

    cm = compute_confusion_matrix(
        torch.from_numpy(y_true), torch.from_numpy(y_pred), num_classes=num_classes
    ).numpy()

    count = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(count, history["train_loss"], label="train loss")
    ax1.plot(count, history["val_loss"], label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()

    im = ax2.imshow(cm, interpolation="nearest")
    ax2.set_title("Confusion Matrix (validation)")
    ax2.set_xlabel("Predicted label")
    ax2.set_ylabel("True label")

    ax2.set_xticks(np.arange(num_classes))
    ax2.set_yticks(np.arange(num_classes))
    ax2.set_xticklabels(label_names, rotation=45, ha="right")
    ax2.set_yticklabels(label_names)

    plt.colorbar(im, ax=ax2)

    for i in range(num_classes):
        for j in range(num_classes):
            value = cm[i, j]
            ax2.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="black" if value > cm.max() / 2.0 else "white",
                fontsize=8,
            )

    plt.tight_layout()
    plt.show()
