""" 
generate_embeddings.py - It produces embeddings for every song in the dataset

Loads the trained autoencoder, runs the dataset through the encoder, saves the resulting (N, 16) embeddings (the end of the encoder part)
in addition to the parallel song IDs as an .npy file. For the readers clarity (and mine) a .npy if unfamilar is a binary file format used to store NumPy arrays, this allows for efficient storage and retrieval of large datasets while preserving the arrays shape and data type. This is usefule for DS and ML for fast data loading compared to the initial .csv that kaggle provided (and also allows for us to processes the data like getting the (N , 16) embedding and parallel song ID.
"""

import os 
import sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_PATH = os.path.join(DATA_DIR, 'autoencoder.pt')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'embeddings.npy')
SONG_IDS_PATH = os.path.join(DATA_DIR, 'song_ids.npy')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autoencoder import MusicAutoencoder
from preprocessing import preprocess, load_scaler
from data_utils import get_csv_path

BATCH_SIZE = 8192

def generate_embeddings():
    print("=" * 60)
    print("Generating Embeddings")
    print("=" * 60) 

    # Step 1: Set up device(s) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # Step 2: Load the trained model
    print("\nLoading trained model...")
    
    model = MusicAutoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({n_params:,} parameters)")

    # Step 3: Load and Preprocess data using the saved scalar
    
    print("\nLoading dataset...")
    df = pd.read_csv(get_csv_path())
    print(f"Loaded {len(df):,} rows from CSV")

    scaler = load_scaler()
    X, ids, _ = preprocess(df, scaler=scaler)

    print(f"After preprocessing: {X.shape}")
    print(f"Encoding {X.shape[0]:,} songs in batches of {BATCH_SIZE:,}...")

    # Step 4: Encode all songs in large batches

    X_tensor = torch.from_numpy(X).float()
    embeddings_list = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), BATCH_SIZE):
            batch = X_tensor[i:i + BATCH_SIZE].to(device, non_blocking=True)
            emb = model.encode(batch)
            embeddings_list.append(emb.cpu().numpy())

            if (i // BATCH_SIZE) % 20 == 0:
                print(f"  Processed {i + len(batch):,} / {len(X_tensor):,} songs")

    
    embeddings = np.concatenate(embeddings_list, axis=0)
    print(f"\nFinal embeddings shape: {embeddings.shape}")
    print(f"Embedding dtype:        {embeddings.dtype}")

    # Step 5: Save embeddings and parallel song IDS
    os.makedirs(DATA_DIR, exist_ok = True)
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(SONG_IDS_PATH, np.array(ids))

    embeddings_mb = os.path.getsize(EMBEDDINGS_PATH) / (1024 * 1024)
    ids_mb = os.path.getsize(SONG_IDS_PATH) / (1024 * 1024)

    print(f"\nSaved embeddings to: {EMBEDDINGS_PATH} ({embeddings_mb:.1f} MB)")
    print(f"Saved song IDs to:   {SONG_IDS_PATH} ({ids_mb:.1f} MB)")
    print("\nDone!")


if __name__ == "__main__":
    generate_embeddings()