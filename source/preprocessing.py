"""
preprocessing.py — feature preprocessing for the music recommender autoencoder.

Handles:
  - filtering corrupt data
  - one-hot encoding the `key` feature
  - standardizing continuous features with StandardScaler

The fitted scaler is saved to disk so identical preprocessing
can be applied to new songs at inference time.
"""
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
# List of continuous features to be standardized (after one-hot encoding 'key').
CONTINUOUS_FEATURES = [
    'danceability',
    'energy',
    'loudness',
    'mode',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
    'duration_ms',
    'time_signature'
]

# `key` (0-11) is handled separately via one-hot encoding.
KEY_COLUMNS = [f'key_{i}' for i in range(12)]

def filter_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Rows with NaN or infinite values or corrupt data in the dataset."""
    # Drop rows with any NaN or infinite values.
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Remove rows with non-positive tempo
    df = df[df['tempo'] > 0] 

    # Remove rows with time signature <= 0
    df = df[df['time_signature'] > 0]

    # Remove rows with <= 5000ms
    df = df[df['duration_ms'] > 5000]
    return df


def one_hot_encode_key(df: pd.DataFrame) -> pd.DataFrame:
    key_dummies = pd.get_dummies(df['key'], prefix='key')

    reindexed = key_dummies.reindex(columns=KEY_COLUMNS, fill_value=0)
    return reindexed

def preprocess(df: pd.DataFrame, scaler: StandardScaler = None) -> tuple:
    """
    Full preprocessing pipeline.
    
    Args:
        df: raw DataFrame loaded from tracks_features.csv
        scaler: optional pre-fitted StandardScaler. If None, fit a new one.
    
    Returns:
        X: numpy array, shape (n_songs, 24), dtype float32 — ready for the autoencoder
        ids: list of Spotify track IDs, length n_songs (parallel to X)
        scaler: the fitted StandardScaler (save this for inference!)
    """

    filtered_df = filter_bad_rows(df)
    ids = filtered_df['id'].tolist()

    continuous_df = filtered_df[CONTINUOUS_FEATURES]                                         # Extract continuous features for scaling. 12 cols, DF
    onehot_df = one_hot_encode_key(filtered_df)                                              # One-hot encode 'key' feature. 12 cols, DF 

    # One-hot encode the 'key' feature and concatenate it back to the DataFrame.

    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(continuous_df)                                        # Fit the scaler on the continuous features
    else:                                                                                   # and transform them. 12 cols, np array
        scaled = scaler.transform(continuous_df)                                            # Transform the continuous features using the
                                                                                            # provided scaler (for inference time). 12 cols, np array
    onehot_array = onehot_df.values                                                         # numpy array, 12 cols

    X = np.concatenate([scaled, onehot_array], axis=1).astype(np.float32)                   # numpy array, shape (n_songs, 24), dtype float32 

    return X, ids, scaler

def save_scaler(scaler: StandardScaler, path: str = None) -> None:
    """Persist the fitted scaler so we can reuse it at inference time."""
    if path is None:
        path = os.path.join(DATA_DIR, 'scaler.pkl')
    os.makedirs(os.path.dirname(path), exist_ok=True)  # safety: create dir if missing
    joblib.dump(scaler, path)


def load_scaler(path: str = None) -> StandardScaler:
    """Load a previously-saved scaler from disk."""
    if path is None:
        path = os.path.join(DATA_DIR, 'scaler.pkl')
    return joblib.load(path)


