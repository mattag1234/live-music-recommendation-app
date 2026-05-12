"""
recommender.py — music similarity engine with era filtering and duplicate suppression.

Uses autoencoder embeddings for audio similarity, then filters candidates
to within ±10 years of the query song. This prevents absurd cross-era
matches (e.g., 2003 pop song → 1939 Judy Garland) and keeps recommendations
in a coherent musical era.

We also dedupe by (normalized title, first artist), so the recommender does
not return the same song from a different album/EP/remaster as a "similar"
result. ("Bohemian Rhapsody - Remastered 2011" no longer surfaces as a
recommendation for "Bohemian Rhapsody".)

Note: We originally planned Spotify genre filtering, but Spotify deprecated
the artist-genres field in early 2025, making the data unreliable. Era
filtering is a robust alternative that uses only data already in our dataset.
"""

import os
import pickle
import re
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_bootstrap import DATA_DIR, ensure_data_files

EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'embeddings.npy')
SONG_IDS_PATH = os.path.join(DATA_DIR, 'song_ids.npy')
METADATA_PATH = os.path.join(DATA_DIR, 'song_metadata.pkl')

ERA_WINDOW = 10  # ± years for the era filter

print("Initializing recommender...")
ensure_data_files()

_embeddings = np.load(EMBEDDINGS_PATH)
_song_ids = np.load(SONG_IDS_PATH)
_id_to_index = {sid: idx for idx, sid in enumerate(_song_ids)}

_norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
_embeddings_normalized = _embeddings / _norms

print("Loading song metadata from pickle...")
with open(METADATA_PATH, 'rb') as f:
    # {spotify_id: (year_or_None, name, artist_cleaned)}
    _song_meta: dict = pickle.load(f)

print(f"Loaded {len(_song_ids):,} songs with {_embeddings.shape[1]}-dim embeddings")
print("Recommender ready.")


# Strip everything from the first " - " or "(" or "[" onwards. Catches
# "(Remastered 2015)", " - Live at Wembley", "[Deluxe Edition]", etc.
_TITLE_SUFFIX_RE = re.compile(r'\s*[\(\[\-].*$')

_EMPTY_META = (None, '', '')


def _normalize_title(name) -> str:
    if not isinstance(name, str):
        return ''
    return _TITLE_SUFFIX_RE.sub('', name).strip().lower()


def _first_artist_key(cleaned) -> str:
    if not cleaned:
        return ''
    return cleaned.split(',')[0].strip().lower()


def recommend(track_id: str, k: int = 10, top_n_pool: int = 400) -> list:
    """
    Top-K most similar songs.

    Filters:
      - Drops the query itself.
      - Drops near-duplicates by (normalized title, first artist) so EP /
        deluxe / remaster copies of the query song don't surface.
      - Prefers candidates within ±ERA_WINDOW years; falls back to pure
        audio similarity if too few era-matched results.
    """
    idx = _id_to_index.get(track_id)
    if idx is None:
        return []

    query_vec = _embeddings_normalized[idx]
    similarities = _embeddings_normalized @ query_vec
    top_pool = np.argsort(similarities)[::-1][:top_n_pool + 1]

    query_year, query_name, query_artist = _song_meta.get(track_id, _EMPTY_META)
    query_key = (_normalize_title(query_name), _first_artist_key(query_artist))
    seen_keys = {query_key} if query_key[0] else set()

    era_filtered = []
    fallback = []

    for i in top_pool:
        candidate_id = str(_song_ids[i])
        if candidate_id == track_id:
            continue

        cand_year, cand_name, cand_artist = _song_meta.get(candidate_id, _EMPTY_META)
        cand_key = (_normalize_title(cand_name), _first_artist_key(cand_artist))

        if cand_key[0] and cand_key in seen_keys:
            continue
        seen_keys.add(cand_key)

        rec = {
            "track_id": candidate_id,
            "name": cand_name or None,
            "artist": cand_artist,
            "similarity": float(similarities[i]),
            "year": cand_year,
        }

        if (query_year is not None and cand_year is not None
                and abs(cand_year - query_year) <= ERA_WINDOW):
            era_filtered.append(rec)
        else:
            fallback.append(rec)

        if len(era_filtered) >= k:
            break

    if len(era_filtered) >= k:
        return era_filtered[:k]

    return era_filtered + fallback[:k - len(era_filtered)]


if __name__ == "__main__":
    def describe(track_id):
        meta = _song_meta.get(track_id)
        if meta is None:
            return f"<song {track_id} not in metadata>"
        year, name, artist = meta
        return f"{name} by {artist} ({year})"

    for idx in [412, 40_000, 190_483]:
        test_id = str(_song_ids[idx])
        print(f"\n{'=' * 70}")
        print(f"Query: {describe(test_id)}")
        print(f"{'=' * 70}")
        recs = recommend(test_id, k=5)
        for rank, rec in enumerate(recs, 1):
            print(f"\n  Rank {rank}  (sim: {rec['similarity']:.4f})")
            print(f"  {describe(rec['track_id'])}")
