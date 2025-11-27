import numpy as np
from src.common.io.schemas import load_cfg, gesture_to_id
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional


def encode_labels_from_cfg(cfg, labels_str: list[str])-> np.ndarray:
    labels_str = np.asarray(labels_str)
    labels_int = []
    for g in labels_str:
        labels_int.append(gesture_to_id(cfg, str(g)))
    return np.array(labels_int, dtype=np.int32)

class Standardizer:
    mean_: np.ndarray = None
    std_: np.ndarray = None
    eps: float = 1e-7
    def fit(self, x: np.ndarray):
        self.mean_ = np.mean(x, axis=0, keepdims=True)
        self.std_ = np.std(x, axis=0, keepdims=True)
        self.std_ = np.where(self.std_ < self.eps, 1, self.std_)
        return self
    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must fit before calling transform.")
        return (x-self.mean_) / self.std_
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

@dataclass
class FeatureSequenceDataset(Dataset):
    def __init__(
            self, feature_vectors: np.ndarray, labels: np.ndarray,
            seq_len: int = 32,
            seq_stride: int = 1,
            names: Optional[list[str]] = None,
            standardizer: Optional[Standardizer] = None,
            allow_partial: bool = False,
            device: Optional[str] = None,
            pad_value: Optional[float] = None,
                 ):
        self.allow_partial = allow_partial
        self.pad_value = pad_value
        feature_vectors = np.asarray(feature_vectors, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        if feature_vectors.ndim != 2:
            raise ValueError("Feature vector must have 2 dimensions.")
        if labels.ndim != 1:
            raise ValueError("Labels must have 1 dimension.")
        self.N, self.D = feature_vectors.shape
        self.seq_len = int(seq_len)
        self.seq_stride = int(seq_stride)
        self.device = device
        self.feature_names = names if names is not None else [f"f{i}" for i in range(self.D)]

        self.standardizer = standardizer
        if self.standardizer is not None:
            if self.standardizer.mean_ is None or self.standardizer.std_ is None:
                feature_vectors = self.standardizer.fit_transform(feature_vectors)
            else:
                feature_vectors = self.standardizer.transform(feature_vectors)
        self._feature_vectors = feature_vectors.astype(np.float32)
        self._labels = labels.astype(np.int64)

        self._starts: list[int] = []
        last_full = self.N - self.seq_len
        if last_full > 0:
            self._starts.extend(range(0, last_full+1, self.seq_stride))
        #Only needed if more overlapping batch of context allowed -> tail of data
        if self.allow_partial:
            last_any = self.N - 1
            tail_start = max(0, last_any - (self.seq_len - 1))
            for s in range(tail_start, self.N, self.seq_stride):
                if not self._starts or s > self._starts[-1]:
                    self._starts.append(s)

    def __len__(self) -> int:
        return len(self._starts)

    @staticmethod
    def _mode_label(y_seq: np.ndarray) -> int:
        vals, counts = np.unique(y_seq, return_counts=True)
        return int(vals[np.argmax(counts)])

    def __getitem__(self, index: int):
        s = self._starts[index]
        e = min(s + self.seq_len, self.N)
        L = e - s
        X = self._feature_vectors[s:e]  # [L, D]
        y_seq = self._labels[s:e]  # [L]
        #Only when padding allow_partial is true, can pad the last window which is not a full window
        if L < self.seq_len:
            pad_f = np.full((self.seq_len - L, self.D), self.pad_value, dtype=np.float32)
            pad_y = np.full((self.seq_len - L,), -100, dtype=np.int64)
            X = np.vstack([pad_f, X])  # left-pad
            y_seq = np.concatenate([pad_y, y_seq], axis=0)
            L_out = L
        else:
            L_out = self.seq_len

        y = self._mode_label(y_seq[y_seq >= 0]) if np.any(y_seq >= 0) else -100

        item = {
            "x": torch.from_numpy(X).to(self.device) if self.device else torch.from_numpy(X),
            "y_seq": torch.from_numpy(y_seq).to(self.device) if self.device else torch.from_numpy(y_seq),
            "y": torch.tensor(y, dtype=torch.long, device=self.device) if self.device else torch.tensor(y,
                                                                                                        dtype=torch.long),
            "length": torch.tensor(L_out, dtype=torch.long, device=self.device) if self.device else torch.tensor(L_out,
                                                                                                                 dtype=torch.long),
            "names": self.feature_names,
        }
        return item

if __name__ == "__main__":
    from src.common.io.schemas import id_to_gesture
    from src.common.preprocessing.windowing import window_signal
    from src.rnn.features.seq_features import compute_seq_features
    from src.common.io.dummy_data import generate_dummy_emg

    config = load_cfg()
    gestures = config.gestures
    dat, labels_raw, ts = generate_dummy_emg(
        100, 1000, gestures
    )
    signal = np.asarray(dat['signal'], dtype=np.float32)
    windows, gesture_labels, time_stamps = window_signal(
        signal, 1000,
        timestamps_ms=ts,
        labels=labels_raw
    )

    feature_v, feat_names = compute_seq_features(
        windows=windows,
        fs=1000,
        basic_feat=True,
        shape_feat=True,
        spectral_feat=True,
        deltas=False
    )
    print(f"Feature_vector shape: {feature_v.shape}")
    print(f"Feature_names: {feat_names}    Feature length: {len(feat_names)}")

    lbls_int = encode_labels_from_cfg(gesture_labels)
    print(lbls_int)

    print("First 10 labels (str → id):")
    for i in range(min(10, len(gesture_labels))):
        print(f"  {gesture_labels[i]} -> {lbls_int[i]}")

    std = Standardizer()
    dataset = FeatureSequenceDataset(
        feature_vectors=feature_v,
        labels=lbls_int,
        seq_len=32,
        seq_stride=8,
        names=feat_names,
        standardizer=std,
        allow_partial=True,
        device=None,
    )

    print(f"Number of samples: {len(dataset)}")

    sample = dataset[0]

    print("\nSingle dataset item:")
    print("  x shape     :", sample["x"].shape)  # [L, D]
    print("  y_seq shape :", sample["y_seq"].shape)  # [L]
    print("  y (id)      :", sample["y"].item())
    try:
        print("  y (name)    :", id_to_gesture(config, int(sample["y"].item())))
    except KeyError:
        print("  y (name)    : <id not in cfg.label_map>")

    print("  length      :", sample["length"].item())
    print("  first 5 feature names:", sample["names"][:5])

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda btch: {
            "x": torch.stack([b["x"] for b in btch], dim=0),
            "y_seq": torch.stack([b["y_seq"] for b in btch], dim=0),
            "y": torch.stack([b["y"] for b in btch], dim=0),
            "length": torch.stack([b["length"] for b in btch], dim=0),
            "names": btch[0]["names"],
        },
    )

    batch = next(iter(loader))
    print("\nBatch from DataLoader:")
    print("  x batch shape     :", batch["x"].shape)  # [B, L, D]
    print("  y_seq batch shape :", batch["y_seq"].shape)  # [B, L]
    print("  y batch shape     :", batch["y"].shape)  # [B]
    print("  lengths           :", batch["length"])
