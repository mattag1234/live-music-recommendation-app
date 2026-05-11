"""
train.py — training loop for the music recommender autoencoder.

Loads preprocessed song features, trains the autoencoder to reconstruct them,
and saves the trained model weights for later inference.
"""

import os 
import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'autoencoder.pt')

from autoencoder import MusicAutoencoder 
from preprocessing import preprocess, save_scaler
from data_utils import get_csv_path


# -----------------------------------------------------------------------------
# Hyperparameters — the knobs you can turn.
# They are generalistc values that are commonly used in the ML community,
# but we will experiment with them later to see how they affect training and recommendations.
# -----------------------------------------------------------------------------
N_SAMPLES = 5_000        # tiny sample for local CPU debugging; bump up on GPU
BATCH_SIZE = 256         # Number of songs per training batch. Adjust based on your GPU memory (try 512 or 1024 if you have a powerful GPU).
LEARNING_RATE = 1e-3     # Adam default is 1e-3, but feel free to experiment with higher/lower values.
N_EPOCHS = 10            # For a real training run, you'll likely want to increase this to 50 or 100 epochs or more.
RANDOM_SEED = 777        # Set a random seed for reproducibility. You can use any integer here.







def train(): 
    # Step 1: Load and preprocess the data
    print("Loading and preprocessing data...")
    torch.manual_seed(RANDOM_SEED)

    # Step 2: Determine which device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Step 3: Load the dataset CSV and preprocess data

    print(f"Loading {N_SAMPLES} songs..")
    df = pd.read_csv(get_csv_path()).sample(n=N_SAMPLES, random_state=RANDOM_SEED)
    X, ids, scaler = preprocess(df)
    print(f"Preprocessed data shape: {X.shape}")

    save_scaler(scaler)  # Save the scaler for later use during inference

    # Step 4: Create PyTorch DataLoader for batching
    X_tensor = torch.from_numpy(X).float()

    dataset = TensorDataset(X_tensor, X_tensor)  # Autoencoder targets are the same as inputs

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Step 5: Initialize the model, loss function, and optimizer

    model = MusicAutoencoder().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error loss for reconstruction
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Step 6: Training loop
    print("Starting training Loop...")
    model.train()  # Set model to training mode

    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0

        for batch_inputs, batch_targets in dataloader:
            # Move batch to the correct device (GPU or CPU)
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()  # Clear gradients from previous step
            
            # Forward pass: compute the model's output and the loss
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()  # Backpropagation: compute gradients   
            optimizer.step()  # Update model weights

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch+1}/{N_EPOCHS} - Average Loss: {avg_loss:.6f}")

    # Step 7: Save the trained model weights for later inference
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()



