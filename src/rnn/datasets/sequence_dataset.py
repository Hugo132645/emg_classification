import numpy as np
from src.rnn.features.seq_features import compute_seq_features
from src.common.io.dummy_data import generate_dummy_emg
from src.common.io.schemas import load_cfg
from src.common.preprocessing.windowing import window_signal
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional

class Standardizer:
    mean_: np.ndarray = None
    std_: np.ndarray = None
    eps: float = 1e-7
    def fit(self, x: np.ndarray) -> np.ndarray:
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
            self, feature_vectors: np.ndarray, labels: list[str],
            seq_len: int = 32,
            seq_stride: int = 1,
            names: Optional[list[str]] = None,
            standardizer: Optional[Standardizer] = None,
            device: Optional[str] = None
                 ):
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

    def __getitem__(self, i: int):
        s = self._starts[i]
        e = min(s + self.seq_len, self.N)
        L = e - s
        X = self._F[s:e]  # [L, D]
        y_seq = self._labels[s:e]  # [L]
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