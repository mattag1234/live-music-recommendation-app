"""
Downloads the Spotify 1.2M+ Songs dataset from Kaggle (or uses cache if already present).
Returns the path so other scripts can locate the CSV.
"""
import kagglehub
import os

def get_dataset_path():
    """Download (or fetch from cache) and return the directory containing tracks_features.csv."""
    return kagglehub.dataset_download("rodolfofigueroa/spotify-12m-songs")

def get_csv_path():
    """Return the full path to the tracks_features.csv file."""
    return os.path.join(get_dataset_path(), "tracks_features.csv")

if __name__ == "__main__":
    path = get_csv_path()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Dataset CSV located at:\n  {path}")
    print(f"Size: {size_mb:.1f} MB")