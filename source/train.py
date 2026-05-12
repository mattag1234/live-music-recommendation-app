"""
train.py — training loop for the music recommender autoencoder.

Loads preprocessed song features, trains the autoencoder to reconstruct them,
and saves the trained model weights for later inference.

Production version: full dataset, validation split, early stopping.
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'autoencoder.pt')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autoencoder import MusicAutoencoder
from preprocessing import preprocess, save_scaler
from data_utils import get_csv_path


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
N_SAMPLES = None         # None = use the full 1.2M dataset
BATCH_SIZE = 1024        # bigger batches work great on GPU
LEARNING_RATE = 1e-3
N_EPOCHS = 50            # upper bound; early stopping usually cuts us off sooner
VAL_FRACTION = 0.1       # 10% held out for validation
PATIENCE = 5             # stop if val loss doesn't improve for this many epochs
RANDOM_SEED = 777


def train():
    print("=" * 60)
    print("Music Recommender — Autoencoder Training")
    print("=" * 60)

    torch.manual_seed(RANDOM_SEED)

    # Device selection — should be 'cuda' on JupyterHub
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Load + preprocess data
    # ------------------------------------------------------------------
    if N_SAMPLES is None:
        print("\nLoading FULL dataset (this may take a minute)...")
        df = pd.read_csv(get_csv_path())
    else:
        print(f"\nLoading {N_SAMPLES:,} songs...")
        df = pd.read_csv(get_csv_path()).sample(n=N_SAMPLES, random_state=RANDOM_SEED)

    print(f"Loaded {len(df):,} rows from CSV")

    X, ids, scaler = preprocess(df)
    print(f"After preprocessing: {X.shape}")
    save_scaler(scaler)

    # ------------------------------------------------------------------
    # Build train/val DataLoaders
    # ------------------------------------------------------------------
    X_tensor = torch.from_numpy(X).float()
    full_dataset = TensorDataset(X_tensor, X_tensor)

    val_size = int(VAL_FRACTION * len(full_dataset))
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"Train set: {train_size:,} | Val set: {val_size:,}")

    # ------------------------------------------------------------------
    # Model, loss, optimizer
    # ------------------------------------------------------------------
    model = MusicAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Training loop with validation + early stopping
    # ------------------------------------------------------------------
    print("\nStarting training...")
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(N_EPOCHS):
        # --- training pass ---
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = train_loss / n_train_batches

        # --- validation pass ---
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / n_val_batches

        # --- bookkeeping + early stopping ---
        improved = avg_val_loss < best_val_loss
        marker = " ✓ best" if improved else ""
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS} | "
              f"train: {avg_train_loss:.6f} | val: {avg_val_loss:.6f}{marker}")

        if improved:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save the best model so far
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"\nEarly stopping: no improvement for {PATIENCE} epochs")
                break

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()