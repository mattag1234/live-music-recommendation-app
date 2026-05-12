"""
recommender.py — music similarity engine with era filtering.

Uses autoencoder embeddings for audio similarity, then filters candidates
to within ±10 years of the query song. This prevents absurd cross-era
matches (e.g., 2003 pop song → 1939 Judy Garland) and keeps recommendations
in a coherent musical era.

Note: We originally planned Spotify genre filtering, but Spotify deprecated
the artist-genres field in early 2025, making the data unreliable. Era
filtering is a robust alternative that uses only data already in our dataset.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_csv_path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'embeddings.npy')
SONG_IDS_PATH = os.path.join(DATA_DIR, 'song_ids.npy')

ERA_WINDOW = 10  # ± years for the era filter

print("Initializing recommender...")

_embeddings = np.load(EMBEDDINGS_PATH)
_song_ids = np.load(SONG_IDS_PATH)
_id_to_index = {sid: idx for idx, sid in enumerate(_song_ids)}

_norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
_embeddings_normalized = _embeddings / _norms

# Load song_id → year map
print("Loading song years from CSV...")
_df_years = pd.read_csv(get_csv_path(), usecols=['id', 'year'])
_song_to_year = dict(zip(_df_years['id'], _df_years['year']))

print(f"Loaded {len(_song_ids):,} songs with {_embeddings.shape[1]}-dim embeddings")
print("Recommender ready.")


def recommend(track_id: str, k: int = 10, top_n_pool: int = 200) -> list:
    """
    Top-K most similar songs, filtered to within ±ERA_WINDOW years.
    Falls back to pure audio similarity if too few era-matched results.
    """
    idx = _id_to_index.get(track_id)
    if idx is None:
        return []

    query_vec = _embeddings_normalized[idx]
    similarities = _embeddings_normalized @ query_vec
    top_pool = np.argsort(similarities)[::-1][:top_n_pool + 1]

    query_year = _song_to_year.get(track_id)

    era_filtered = []
    fallback = []

    for i in top_pool:
        candidate_id = str(_song_ids[i])
        if candidate_id == track_id:
            continue

        candidate_year = _song_to_year.get(candidate_id)
        rec = {
            "track_id": candidate_id,
            "similarity": float(similarities[i]),
            "year": int(candidate_year) if pd.notna(candidate_year) else None,
        }

        # Filter by era if both years are known
        if (query_year is not None and candidate_year is not None
                and abs(candidate_year - query_year) <= ERA_WINDOW):
            era_filtered.append(rec)
        else:
            fallback.append(rec)

        if len(era_filtered) >= k:
            break

    if len(era_filtered) >= k:
        return era_filtered[:k]

    return era_filtered + fallback[:k - len(era_filtered)]


if __name__ == "__main__":
    df_meta = pd.read_csv(get_csv_path(), usecols=[
        'id', 'name', 'artists', 'year', 'tempo', 'energy', 'danceability', 'valence'
    ]).set_index('id')

    def describe(track_id):
        try:
            row = df_meta.loc[track_id]
            return (f"{row['name']} by {row['artists']}  ({row['year']})\n"
                    f"    tempo={row['tempo']:.0f}  energy={row['energy']:.2f}  "
                    f"dance={row['danceability']:.2f}  valence={row['valence']:.2f}")
        except KeyError:
            return f"<song {track_id} not in metadata>"

    for idx in [42, 50_000, 800_000]:
        test_id = str(_song_ids[idx])
        print(f"\n{'=' * 70}")
        print(f"Query: {describe(test_id)}")
        print(f"{'=' * 70}")
        recs = recommend(test_id, k=5)
        for rank, rec in enumerate(recs, 1):
            print(f"\n  Rank {rank}  (sim: {rec['similarity']:.4f})")
            print(f"  {describe(rec['track_id'])}")