#Importing all needed files
import numpy as np
import torch
import torch.nn as nn
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

#Loading config, seed, device
config = load_cfg()
sd = 42
dvc = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))

def make_dataloaders(
        cfg, seconds: int = 100, batch_size: int = 32, seq_length: int = 32,
        seq_stride: int = 8, val_ratio: float = 0.4, seed: int = None
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
    train_loader, val_loader, input_dim, num_classes, ft_names, std = make_dataloaders(
        cfg=config,
        seconds=2000,
        batch_size=32,
        seq_length=32,
        seq_stride=8,
        seed=sd,
    )

    hidden_dim = 32
    num_layers = 2
    bidirectional = False
    lr = 1e-3
    epochs = 7

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

    #Simple plot -> training vs validation data accuracy
    plt.plot(count, GRU_val_acc_log, label="value accuracy", color="red")
    plt.plot(count, GRU_train_acc_log, label="train accuracy", color="green")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


