"""
Training script for CNN model on dummy EMG data.

This script demonstrates the complete training pipeline:
1. Generate dummy EMG data
2. Preprocess and window the signal
3. Compute spectrograms
4. Train CNN model
5. Evaluate and save results

Designed for CPU-only training as a baseline.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import time

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.common.io.dummy_data import generate_dummy_emg
from src.common.io.schemas import load_cfg
from src.common.preprocessing.pipelines import preprocess_raw
from src.common.preprocessing.windowing import window_signal
from src.cnn.transforms.spectrograms import batch_compute_spectrograms
from src.cnn.datasets.cnn_dataset import SpectrogramDataset, train_val_split
from src.cnn.models.model_cnn import EMGConvNet, count_parameters


# Training Configuration
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
SEED = 42
DEVICE = 'cpu'  # CPU-only training

# Data generation parameters
DURATION_S = 60  # 60 seconds of dummy data
BLOCK_S = 5  # 5-second blocks per gesture


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_data(cfg):
    """
    Generate, preprocess, and prepare spectrogram data.
    
    Returns
    -------
    spectrograms : np.ndarray
        Array of shape (N, n_mels, time_frames).
    labels : list
        List of string labels.
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATA GENERATION & PREPROCESSING")
    print("=" * 70)
    
    fs = cfg.sample_rate_hz
    classes = [g for g in cfg.gestures if g != 'rest']  # Exclude 'rest' for cleaner dummy data
    
    # Generate dummy EMG data
    print(f"\n[1/4] Generating {DURATION_S}s of dummy EMG data...")
    df, interval_labels, timestamps_ms = generate_dummy_emg(
        seconds=DURATION_S,
        fs=fs,
        classes=classes,
        block_s=BLOCK_S,
        mode="raw",
        seed=SEED
    )
    
    signal = df['signal'].values
    print(f"  ✓ Generated {len(signal)} samples ({DURATION_S}s @ {fs}Hz)")
    print(f"  ✓ Gestures: {set(interval_labels)}")
    
    # Preprocess signal
    print("\n[2/4] Preprocessing signal...")
    processed_signal, meta = preprocess_raw(
        signal,
        fs=fs,
        band=(20.0, 450.0),
        lp_cut=10.0,
        norm="zscore"
    )
    print(f"  ✓ Pipeline: bandpass → rectify → lowpass → zscore")
    
    # Window signal
    print("\n[3/4] Windowing signal...")
    windows, window_labels, times_ms = window_signal(
        processed_signal,
        fs=fs,
        window_ms=cfg.window_ms,
        hop_ms=cfg.hop_ms,
        labels=interval_labels,
        timestamps_ms=timestamps_ms
    )
    print(f"  ✓ Created {len(windows)} windows")
    print(f"  ✓ Window: {cfg.window_ms}ms, Hop: {cfg.hop_ms}ms")
    
    # Compute spectrograms
    print("\n[4/4] Computing spectrograms...")
    spectrograms = batch_compute_spectrograms(
        windows,
        fs=fs,
        n_fft=256,
        hop_length=128,
        n_mels=64,
        fmin=20.0,
        fmax=500.0
    )
    print(f"  ✓ Generated {len(spectrograms)} spectrograms")
    print(f"  ✓ Shape: {spectrograms.shape} (N, mels, time)")
    
    return spectrograms, window_labels


def create_dataloaders(spectrograms, labels, cfg):
    """
    Create train and validation DataLoaders.
    
    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    dataset : SpectrogramDataset (full dataset for reference)
    """
    print("\n" + "=" * 70)
    print("STEP 2: DATASET CREATION")
    print("=" * 70)
    
    # Create full dataset
    dataset = SpectrogramDataset(spectrograms, labels, cfg)
    print(f"\n  ✓ Total samples: {len(dataset)}")
    print(f"  ✓ Number of classes: {dataset.num_classes}")
    dataset.get_class_distribution()
    
    # Split into train and validation
    train_dataset, val_dataset = train_val_split(
        dataset,
        train_ratio=TRAIN_SPLIT,
        shuffle=True,
        seed=SEED
    )
    
    print(f"\n  ✓ Train samples: {len(train_dataset)}")
    print(f"  ✓ Val samples: {len(val_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # CPU-only, avoid multiprocessing overhead
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"\n  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, dataset


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for spectrograms, labels in pbar:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for spectrograms, labels in pbar:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved training curves: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved confusion matrix: {save_path}")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("CNN TRAINING ON DUMMY EMG DATA")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Train/Val split: {TRAIN_SPLIT:.0%} / {1-TRAIN_SPLIT:.0%}")
    
    # Set random seed
    set_seed(SEED)
    
    # Load configuration
    cfg = load_cfg("configs/preprocessing.yaml")
    
    # Generate data
    spectrograms, labels = generate_data(cfg)
    
    # Create dataloaders
    train_loader, val_loader, dataset = create_dataloaders(spectrograms, labels, cfg)
    
    # Create model
    print("\n" + "=" * 70)
    print("STEP 3: MODEL INITIALIZATION")
    print("=" * 70)
    
    num_classes = dataset.num_classes
    model = EMGConvNet(num_classes=num_classes, dropout=0.3)
    model = model.to(DEVICE)
    
    num_params = count_parameters(model)
    print(f"\n  ✓ Model: EMGConvNet")
    print(f"  ✓ Number of classes: {num_classes}")
    print(f"  ✓ Trainable parameters: {num_params:,}")
    print(f"  ✓ Parameter memory: {(num_params * 4) / (1024**2):.2f} MB")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"  ✓ Loss: CrossEntropyLoss")
    print(f"  ✓ Optimizer: Adam (lr={LEARNING_RATE})")
    
    # Training loop
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING")
    print("=" * 70)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print summary
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            # Create export directory
            export_dir = Path("exports/cnn")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = export_dir / "best_model_dummy.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': num_classes,
                'config': {
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'train_split': TRAIN_SPLIT,
                    'seed': SEED
                }
            }, model_path)
            
            print(f"  → Saved best model (Val Acc: {val_acc:.2f}%)")
    
    elapsed_time = time.time() - start_time
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("STEP 5: FINAL EVALUATION")
    print("=" * 70)
    
    print(f"\n  Training completed in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    print(f"  Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    # Compute metrics on final validation predictions
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    print(f"  Final Val F1 (macro): {val_f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 70)
    class_names = [dataset.idx_to_label[i] for i in range(num_classes)]
    # Specify labels to handle cases where not all classes appear in validation set
    print(classification_report(val_labels, val_preds, target_names=class_names, 
                                labels=list(range(num_classes)), zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds, labels=list(range(num_classes)))
    print("Confusion Matrix:")
    print(cm)
    
    # Save visualizations
    print("\n" + "=" * 70)
    print("STEP 6: SAVING RESULTS")
    print("=" * 70)
    
    output_dir = Path("outputs/cnn_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=output_dir / "training_curves.png"
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm, class_names,
        save_path=output_dir / "confusion_matrix.png"
    )
    
    print(f"\n  ✓ Model saved: exports/cnn/best_model_dummy.pth")
    print(f"  ✓ Plots saved: {output_dir}/")
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
