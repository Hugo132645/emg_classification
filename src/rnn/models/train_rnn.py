#Importing all needed files
import numpy as np
import torch
import torch.nn as nn
from torch import device
from torchgen.gen_functionalization_type import return_from_mutable_noop_redispatch

from src.rnn.models.model_rnn import GRUModel, LSTMModel
from src.common.io.schemas import load_cfg, gesture_to_id, expand_template, ensure_parent_dir
from src.rnn.datasets.sequence_dataset import Standardizer, FeatureSequenceDataset, encode_labels_from_cfg
from src.rnn.features.seq_features import compute_seq_features
from src.common.io.dummy_data import generate_dummy_emg
from torch.utils.data import DataLoader
from src.common.preprocessing.windowing import window_signal
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from src.rnn.datasets.sequence_dataset import Standardizer, FeatureSequenceDataset
from src.common.preprocessing.windowing import window_signal_np
from src.common.io.emg_loader import _search_csv_files, process_emg_for_windowing

#Loading config, seed, device
config = load_cfg()
sd = 42
dvc = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))

def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t,p] += 1
    return cm

def get_targets_preds(loader, model, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            lengths = batch['length'].to(device)

            logits = model(x, lengths)
            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.detach().cpu())
            all_targets.append(y.detach().cpu())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        return all_targets, all_preds

def _cont_runs(rerepetition: np.ndarray):
    if len(rerepetition) == 0:
        return []

    change_idx = np.where(np.diff(rerepetition) != 0)[0] + 1
    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx, len(rerepetition)]

    return [(int(s), int(e), int(rerepetition[s])) for s, e in zip(starts, ends)]


def make_dataloaders_dummy(
        cfg, seconds: int = 100, batch_size: int = 32, seq_length: int = 32,
        seq_stride: int = 8, val_ratio: float = 0.3, seed: int = None
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    sample_rate = cfg.sample_rate_hz

    #Generating dummy_data
    dat, raw_labels, timestamps = generate_dummy_emg(seconds, sample_rate, cfg.gestures, block_s=5, seed=seed)

    #Windowing dummy_data
    windows, raw_labels, timestamps = window_signal(dat['signal'], sample_rate,
                                                    labels=raw_labels, timestamps_ms=timestamps, window_ms=cfg.window_ms, hop_ms=cfg.hop_ms)
    ft_vectors, ft_names = compute_seq_features(windows, sample_rate, spectral_feat=True)
    labels_int = encode_labels_from_cfg(cfg, raw_labels)

    #Creating a Standardizer
    std = Standardizer()
    dataset = FeatureSequenceDataset(
        feature_vectors=ft_vectors,
        labels=labels_int,
        seq_len=seq_length,
        seq_stride=seq_stride,
        standardizer=std,
        device=None,
    )

    n=len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)
    n_val = int(val_ratio * n)
    val_idxs = indices[:n_val]
    train_idxs = indices[n_val:]

    #Dividing into training and validation dataset
    train_dataset = torch.utils.data.Subset(dataset, train_idxs)
    vals_dataset = torch.utils.data.Subset(dataset, val_idxs)

    #Collation into torch.tensors for training and validation
    def collate_fn(batch):
        return {
            "x": torch.stack([b["x"] for b in batch], dim=0),          # [B, L, D]
            "y_seq": torch.stack([b["y_seq"] for b in batch], dim=0),  # [B, L]
            "y": torch.stack([b["y"] for b in batch], dim=0),          # [B]
            "length": torch.tensor([int(b["length"]) for b in batch], dtype=torch.long),# [B]
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

    input_dim = ft_vectors.shape[1]
    num_classes = len(cfg.gestures)

    return train_loader, val_loader, input_dim, num_classes, ft_names, std

def make_dataloaders_np(
        cfg,
        data_path: str,
        batch_size: int = 32,
        seq_length: int = 32,
        seq_stride: int = 8,
        seed: int | None = None,
        drop_rest: bool = True,
        train_reps: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
        val_reps: tuple[int, ...] = (9, 10),
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    sample_rate = cfg.sample_rate_hz
    csv_files = sorted(Path(p) for p in _search_csv_files(data_path))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_path}")

    file_records = []
    label_pairs = set()

    for csv_path in csv_files:
        x, raw_labels, rerepetition, subject, exercise = process_emg_for_windowing(
            str(csv_path),
            drop_rest=drop_rest,
        )

        if len(x) == 0:
            continue

        file_records.append((csv_path, x, raw_labels, rerepetition, subject, exercise))

        for lab in np.unique(raw_labels):
            label_pairs.add((exercise, int(lab)))

    if not file_records:
        raise ValueError("No usable EMG")

    sorted_pairs = sorted(label_pairs)
    label_map = {pair: i for i, pair in enumerate(sorted_pairs)}
    label_names = [f"E{ex}_G{lab}" for ex, lab in sorted_pairs]

    train_feature_blocks = []
    train_label_blocks = []
    val_feature_blocks = []
    val_label_blocks = []
    ft_names = None

    min_len = int(sample_rate * cfg.window_ms / 1000)

    for csv_path, x, raw_labels, rerepetition, subject, exercise in file_records:
        global_labels = np.array(
            [label_map[(exercise, int(lbl))] for lbl in raw_labels],
            dtype=np.int64,
        )

        for start, end, rep in _cont_runs(rerepetition):
            x_block = x[start:end]
            y_block = global_labels[start:end]

            if len(x_block) < min_len:
                continue

            windows, window_labels, _ = window_signal_np(
                x_block,
                fs=sample_rate,
                window_ms=cfg.window_ms,
                hop_ms=cfg.hop_ms,
                labels=y_block,
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

            window_labels = np.asarray(window_labels, dtype=np.int64)

            if rep in train_reps:
                train_feature_blocks.append(features)
                train_label_blocks.append(window_labels)
            elif rep in val_reps:
                val_feature_blocks.append(features)
                val_label_blocks.append(window_labels)

    if not train_feature_blocks:
        raise ValueError("No training windows were created.")
    if not val_feature_blocks:
        raise ValueError("No validation windows were created.")

    std = Standardizer().fit(np.vstack(train_feature_blocks))

    train_datasets = [
        FeatureSequenceDataset(
            feature_vectors=F,
            labels=y,
            seq_len=seq_length,
            seq_stride=seq_stride,
            standardizer=std,
            device=None,
        )
        for F, y in zip(train_feature_blocks, train_label_blocks)
    ]

    val_datasets = [
        FeatureSequenceDataset(
            feature_vectors=F,
            labels=y,
            seq_len=seq_length,
            seq_stride=seq_stride,
            standardizer=std,
            device=None,
        )
        for F, y in zip(val_feature_blocks, val_label_blocks)
    ]

    train_dataset = ConcatDataset(train_datasets)
    vals_dataset = ConcatDataset(val_datasets)

    #Collation into torch.tensors for training and validation
    def collate_fn(batch):
        return {
            "x": torch.stack([b["x"] for b in batch], dim=0),          # [B, L, D]
            "y_seq": torch.stack([b["y_seq"] for b in batch], dim=0),  # [B, L]
            "y": torch.stack([b["y"] for b in batch], dim=0),          # [B]
            "length": torch.tensor([int(b["length"]) for b in batch], dtype=torch.long),# [B]
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

    input_dim = train_feature_blocks[0].shape[1]
    num_classes = len(cfg.gestures)

    return train_loader, val_loader, input_dim, num_classes, ft_names, std


def run_epoch(loader, model, optimizer=None, device="mps"):
    #Running 1 epoch -> one-time training dataset
    if optimizer is None:
        model.eval()
    else:
        model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    #Differentiating running epoch between training/validation data -> val not calculating gradient
    context = torch.no_grad() if optimizer is None else torch.enable_grad()
    with context:
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            lengths = batch['length'].to(device)

            if optimizer is not None:
                optimizer.zero_grad()

            logits = model(x, lengths)
            loss = criterion(logits, y)

            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += batch_size

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

if __name__ == "__main__":
    #Testing
    sd = 1
    config = load_cfg()
    train_loader, val_loader, input_dim, num_classes, ft_names, std = make_dataloaders_np(
        cfg=config,
        data_path="/Users/norbertcesar/PycharmProjects/emg_classification/data/input_data/csv_files",
        batch_size=32,
        seq_length=32,
        seq_stride=8,
        drop_rest=True,
        train_reps=(1,2,3,4,5,6,7,8),
        val_reps=(9, 10),
        seed=sd,
    )

    hidden_dim = 32
    num_layers = 2
    bidirectional = False
    lr = 1e-3
    epochs = 5

    gru_model = GRUModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=0.2,
    ).to(dvc)

    GRU_train_loss_log, GRU_val_loss_log, GRU_train_acc_log, GRU_val_acc_log = [], [], [], []
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        train_loss, train_accuracy = run_epoch(train_loader, gru_model, optimizer, device=dvc)
        val_loss, val_accuracy = run_epoch(val_loader, gru_model, optimizer=None, device=dvc)
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_state = gru_model.state_dict()
        GRU_train_loss_log.append(train_loss)
        GRU_val_loss_log.append(val_loss)
        GRU_train_acc_log.append(train_accuracy)
        GRU_val_acc_log.append(val_accuracy)
    count = list(range(1, epochs + 1))

    #Loading only best model state
    if best_state is not None:
        gru_model.load_state_dict(best_state)

    y_true, y_pred = get_targets_preds(val_loader, gru_model, dvc)
    num_classes_cm = max(num_classes, int(max(y_true.max(), y_pred.max())) + 1)
    cm = compute_confusion_matrix(
        torch.from_numpy(y_true),
        torch.from_numpy(y_pred),
        num_classes=num_classes_cm
    ).numpy()


    #Saving artifact
    artifact = {
        "state_dict": gru_model.state_dict(),
        "model_type": "gru",
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "bidirectional": False,
        "dropout": 0.2,
        "label_map": config.label_map,
        "feature_names": ft_names,
        "standardizer_mean": std.mean_,
        "standardizer_std": std.std_,
    }

    #Saving model
    timestamp = datetime.now().strftime("%Y-%m-%d")
    model_path = expand_template("./artifacts/{model_type}_{timestamp}.pt",
                                 model_type='gru',
                                 timestamp=timestamp)
    ensure_parent_dir(model_path)

    torch.save(artifact, model_path)
    print("Model saved to {}".format(model_path))

    count = list(range(1, epochs + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(count, GRU_train_loss_log, label="train loss")
    ax1.plot(count, GRU_val_loss_log, label="val loss")
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
    ax2.set_xticklabels(config.gestures, rotation=45, ha="right")
    ax2.set_yticklabels(config.gestures)

    plt.colorbar(im, ax=ax2)

    for i in range(num_classes):
        for j in range(num_classes):
            value = cm[i, j]
            ax2.text(
                j, i,
                str(value),
                ha="center", va="center",
                color="black" if value > cm.max() / 2.0 else "white",
                fontsize=8,
            )

    plt.tight_layout()
    plt.show()


