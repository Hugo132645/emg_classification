"""
CNN Model for EMG Spectrogram Classification.

This module provides a lightweight ConvNet architecture optimized for
CPU training on log-mel spectrograms of EMG signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    """
    Convolutional block: Conv2D → BatchNorm → ReLU → MaxPool.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple
        Convolution kernel size (default 3).
    pool_size : int or tuple
        MaxPool kernel size (default 2).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2
    ):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # 'same' padding
            bias=False  # BatchNorm makes bias redundant
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolutional block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class EMGConvNet(nn.Module):
    """
    Lightweight ConvNet for EMG spectrogram classification.
    
    Architecture:
        Input: (B, 1, n_mels, time_frames)
        ├─ ConvBlock1: 1 → 32 channels, pool 2x2
        ├─ ConvBlock2: 32 → 64 channels, NO pooling
        ├─ ConvBlock3: 64 → 128 channels, NO pooling
        ├─ Global Average Pooling
        ├─ Dropout (0.3)
        └─ Fully Connected: 128 → num_classes
    
    Parameters
    ----------
    num_classes : int
        Number of gesture classes (default 4).
    dropout : float
        Dropout rate before final FC layer (default 0.3).
    
    Attributes
    ----------
    features : nn.Sequential
        Convolutional feature extractor.
    classifier : nn.Sequential
        Classification head (GAP + Dropout + FC).
    
    Examples
    --------
    >>> model = EMGConvNet(num_classes=4)
    >>> x = torch.randn(8, 1, 64, 2)  # Batch of 8 spectrograms
    >>> logits = model(x)
    >>> print(logits.shape)
    torch.Size([8, 4])
    """
    
    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super(EMGConvNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature extraction: 3 convolutional blocks
        # Only first block pools to handle small inputs (e.g., 64x2 spectrograms)
        self.block1 = ConvBlock(in_channels=1, out_channels=32, kernel_size=3, pool_size=2)
        self.block2 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, pool_size=1)  # No pooling
        self.block3 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, pool_size=1)  # No pooling
        
        # Global Average Pooling (adaptive to any spatial size)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, n_mels, time_frames).
        
        Returns
        -------
        logits : torch.Tensor
            Class logits of shape (B, num_classes).
        """
        # Extract features through convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)  # (B, 128, H', W')
        
        # Global average pooling
        x = self.global_pool(x)  # (B, 128, 1, 1)
        
        # Flatten
        x = torch.flatten(x, 1)  # (B, 128)
        
        # Classify
        logits = self.classifier(x)  # (B, num_classes)
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings before classification.
        
        Useful for visualization (t-SNE, UMAP) or transfer learning.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, n_mels, time_frames).
        
        Returns
        -------
        features : torch.Tensor
            Feature embeddings of shape (B, 128).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    
    Returns
    -------
    num_params : int
        Total number of trainable parameters.
    
    Examples
    --------
    >>> model = EMGConvNet(num_classes=4)
    >>> num_params = count_parameters(model)
    >>> print(f"Model has {num_params:,} trainable parameters")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_shape: Tuple[int, int, int] = (1, 64, 2)) -> None:
    """
    Print a summary of the model architecture.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    input_shape : tuple
        Input shape (C, H, W) without batch dimension (default (1, 64, 2)).
    
    Examples
    --------
    >>> model = EMGConvNet(num_classes=4)
    >>> model_summary(model, input_shape=(1, 64, 2))
    """
    print("=" * 80)
    print("Model Architecture Summary")
    print("=" * 80)
    print(model)
    print("=" * 80)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Estimate memory usage (assuming float32)
    param_memory_mb = (total_params * 4) / (1024 ** 2)  # 4 bytes per float32
    print(f"Parameter memory: {param_memory_mb:.2f} MB")
    
    # Test forward pass
    batch_size = 1
    dummy_input = torch.randn(batch_size, *input_shape)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape:  {tuple(dummy_input.shape)}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Output shape: {tuple(output.shape)}")
    print(f"  Output classes: {output.shape[1]}")
    
    # Feature extraction test
    with torch.no_grad():
        features = model.extract_features(dummy_input)
    print(f"  Feature embedding shape: {tuple(features.shape)}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Self-test
    print("Testing EMGConvNet...\n")
    
    # Create model
    num_classes = 4
    model = EMGConvNet(num_classes=num_classes)
    
    # Print summary
    model_summary(model, input_shape=(1, 64, 2))
    
    # Test with a batch
    print("\n" + "=" * 80)
    print("Testing batch inference...")
    print("=" * 80)
    
    batch_size = 8
    n_mels = 64
    time_frames = 2
    
    x = torch.randn(batch_size, 1, n_mels, time_frames)
    print(f"Input batch shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions shape: {preds.shape}")
    print(f"Sample predictions: {preds[:5].tolist()}")
    print(f"Sample probabilities:\n{probs[:3]}")
    
    # Test feature extraction
    with torch.no_grad():
        features = model.extract_features(x)
    print(f"\nExtracted features shape: {features.shape}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
