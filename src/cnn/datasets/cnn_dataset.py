"""
PyTorch Dataset for EMG Spectrograms.

This module wraps spectrogram arrays and labels into a PyTorch Dataset
for training CNN models on EMG gesture classification.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.common.io.schemas import load_cfg


class SpectrogramDataset(Dataset):
    """
    PyTorch Dataset for spectrogram-based EMG classification.
    
    Wraps numpy arrays of spectrograms and corresponding labels into a
    PyTorch Dataset that can be used with DataLoader.
    
    Parameters
    ----------
    spectrograms : np.ndarray
        Array of shape (N, n_mels, time_frames) containing log-mel spectrograms.
    labels : List[str]
        List of length N containing gesture labels (e.g., 'fist', 'open', 'pinch', 'rest').
    cfg : SimpleNamespace, optional
        Configuration object with label_map. If None, loads from default config.
    transform : callable, optional
        Optional transform to apply to spectrograms (e.g., augmentations).
    
    Attributes
    ----------
    spectrograms : np.ndarray
        Stored spectrograms (N, n_mels, time_frames).
    labels : List[str]
        String labels.
    label_to_idx : dict
        Mapping from label string to integer index.
    idx_to_label : dict
        Mapping from integer index to label string.
    num_classes : int
        Number of unique classes.
    
    Examples
    --------
    >>> spectrograms = np.random.randn(100, 64, 2)  # 100 spectrograms
    >>> labels = ['fist'] * 25 + ['open'] * 25 + ['pinch'] * 25 + ['rest'] * 25
    >>> dataset = SpectrogramDataset(spectrograms, labels)
    >>> print(len(dataset))
    100
    >>> spec_tensor, label_idx = dataset[0]
    >>> print(spec_tensor.shape, label_idx)
    torch.Size([1, 64, 2]) tensor(1)
    """
    
    def __init__(
        self,
        spectrograms: np.ndarray,
        labels: List[str],
        cfg=None,
        transform: Optional[callable] = None
    ):
        """Initialize the dataset."""
        assert len(spectrograms) == len(labels), \
            f"Mismatch: {len(spectrograms)} spectrograms but {len(labels)} labels"
        
        self.spectrograms = spectrograms.astype(np.float32)
        self.labels = labels
        self.transform = transform
        
        # Load config if not provided
        if cfg is None:
            cfg = load_cfg("configs/preprocessing.yaml")
        
        # Create label mappings
        self.label_to_idx = cfg.label_map
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
        # Validate all labels are in the mapping
        unique_labels = set(labels)
        for label in unique_labels:
            if label not in self.label_to_idx:
                raise ValueError(
                    f"Label '{label}' not found in config label_map. "
                    f"Available labels: {list(self.label_to_idx.keys())}"
                )
        
        # Convert string labels to indices
        self.label_indices = [self.label_to_idx[label] for label in labels]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.spectrograms)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.
        
        Returns
        -------
        spectrogram : torch.Tensor
            Spectrogram tensor of shape (1, n_mels, time_frames).
            Channel dimension is added for CNN input.
        label : torch.Tensor
            Label index as a tensor (scalar).
        """
        # Get spectrogram and add channel dimension: (n_mels, T) -> (1, n_mels, T)
        spec = self.spectrograms[idx]
        
        # Apply optional transform
        if self.transform is not None:
            spec = self.transform(spec)
        
        # Add channel dimension
        spec = np.expand_dims(spec, axis=0)  # (n_mels, T) -> (1, n_mels, T)
        
        # Convert to torch tensors
        spec_tensor = torch.from_numpy(spec).float()
        label_tensor = torch.tensor(self.label_indices[idx], dtype=torch.long)
        
        return spec_tensor, label_tensor
    
    def get_label_counts(self) -> dict:
        """
        Count occurrences of each label.
        
        Returns
        -------
        counts : dict
            Dictionary mapping label strings to their counts.
        """
        from collections import Counter
        return dict(Counter(self.labels))
    
    def compute_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced datasets.
        
        Uses inverse frequency weighting: weight_i = N / (n_classes * count_i)
        
        Returns
        -------
        weights : torch.Tensor
            Tensor of shape (num_classes,) containing class weights.
        
        Examples
        --------
        >>> weights = dataset.compute_class_weights()
        >>> criterion = nn.CrossEntropyLoss(weight=weights)
        """
        label_counts = self.get_label_counts()
        total_samples = len(self.labels)
        
        # Initialize weights for all classes
        weights = torch.zeros(self.num_classes, dtype=torch.float32)
        
        # Compute inverse frequency weights
        for label, count in label_counts.items():
            idx = self.label_to_idx[label]
            weights[idx] = total_samples / (self.num_classes * count)
        
        return weights
    
    def get_class_distribution(self) -> None:
        """Print the class distribution for debugging."""
        counts = self.get_label_counts()
        print("Class Distribution:")
        print("-" * 30)
        for label in sorted(counts.keys()):
            idx = self.label_to_idx[label]
            count = counts[label]
            percentage = 100 * count / len(self.labels)
            print(f"  {label:>8} (id={idx}): {count:4d} ({percentage:5.1f}%)")
        print("-" * 30)
        print(f"  Total: {len(self.labels)} samples")


def train_val_split(
    dataset: SpectrogramDataset,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Tuple[SpectrogramDataset, SpectrogramDataset]:
    """
    Split a dataset into training and validation sets.
    
    Parameters
    ----------
    dataset : SpectrogramDataset
        The dataset to split.
    train_ratio : float
        Fraction of data to use for training (default 0.8).
    shuffle : bool
        Whether to shuffle before splitting (default True).
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    train_dataset : SpectrogramDataset
        Training subset.
    val_dataset : SpectrogramDataset
        Validation subset.
    
    Examples
    --------
    >>> train_ds, val_ds = train_val_split(dataset, train_ratio=0.8, seed=42)
    >>> print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    split_idx = int(n_samples * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create new datasets with subsets
    train_spectrograms = dataset.spectrograms[train_indices]
    train_labels = [dataset.labels[i] for i in train_indices]
    
    val_spectrograms = dataset.spectrograms[val_indices]
    val_labels = [dataset.labels[i] for i in val_indices]
    
    train_dataset = SpectrogramDataset(
        train_spectrograms,
        train_labels,
        cfg=None,  # Will reload config internally
        transform=dataset.transform
    )
    
    val_dataset = SpectrogramDataset(
        val_spectrograms,
        val_labels,
        cfg=None,
        transform=None  # No augmentation for validation
    )
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Self-test
    print("Testing SpectrogramDataset...")
    print("=" * 60)
    
    # Create dummy data
    n_samples = 100
    n_mels = 64
    time_frames = 2
    
    spectrograms = np.random.randn(n_samples, n_mels, time_frames).astype(np.float32)
    labels = ['fist'] * 25 + ['open'] * 25 + ['pinch'] * 25 + ['rest'] * 25
    
    # Create dataset
    print("\n[1/4] Creating dataset...")
    dataset = SpectrogramDataset(spectrograms, labels)
    print(f"  ✓ Dataset size: {len(dataset)}")
    print(f"  ✓ Number of classes: {dataset.num_classes}")
    print(f"  ✓ Label mapping: {dataset.label_to_idx}")
    
    # Test __getitem__
    print("\n[2/4] Testing __getitem__...")
    spec_tensor, label_tensor = dataset[0]
    print(f"  ✓ Spectrogram shape: {spec_tensor.shape}")
    print(f"  ✓ Expected shape: (1, {n_mels}, {time_frames})")
    print(f"  ✓ Label type: {type(label_tensor)}, value: {label_tensor.item()}")
    assert spec_tensor.shape == (1, n_mels, time_frames), "Shape mismatch!"
    
    # Test class distribution
    print("\n[3/4] Class distribution:")
    dataset.get_class_distribution()
    
    # Test class weights
    print("\n[4/4] Computing class weights...")
    weights = dataset.compute_class_weights()
    print(f"  ✓ Class weights: {weights}")
    print(f"  ✓ All weights equal (balanced data): {torch.allclose(weights, weights[0] * torch.ones_like(weights))}")
    
    # Test train/val split
    print("\n[Bonus] Testing train/val split...")
    train_ds, val_ds = train_val_split(dataset, train_ratio=0.8, seed=42)
    print(f"  ✓ Train size: {len(train_ds)}")
    print(f"  ✓ Val size: {len(val_ds)}")
    print(f"  ✓ Total: {len(train_ds) + len(val_ds)} (original: {len(dataset)})")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
