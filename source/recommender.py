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
import re
import sys
import ast
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

print("Loading song metadata from CSV...")
_df_meta = pd.read_csv(get_csv_path(), usecols=['id', 'name', 'artists', 'year'])
_song_to_year = dict(zip(_df_meta['id'], _df_meta['year']))
_song_to_name = dict(zip(_df_meta['id'], _df_meta['name']))
_song_to_artists_raw = dict(zip(_df_meta['id'], _df_meta['artists']))
del _df_meta

print(f"Loaded {len(_song_ids):,} songs with {_embeddings.shape[1]}-dim embeddings")
print("Recommender ready.")


# Strip everything from the first " - " or "(" or "[" onwards. Catches
# "(Remastered 2015)", " - Live at Wembley", "[Deluxe Edition]", etc.
_TITLE_SUFFIX_RE = re.compile(r'\s*[\(\[\-].*$')


def _clean_artists(raw) -> str:
    """The CSV stores artists as a Python-list-literal string like "['A', 'B']".
    Return a human-readable "A, B"."""
    if not isinstance(raw, str):
        return ''
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return ', '.join(str(x) for x in parsed)
    except (ValueError, SyntaxError):
        pass
    return raw


def _normalize_title(name) -> str:
    if not isinstance(name, str):
        return ''
    return _TITLE_SUFFIX_RE.sub('', name).strip().lower()


def _first_artist_key(raw) -> str:
    cleaned = _clean_artists(raw)
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

    query_year = _song_to_year.get(track_id)
    query_key = (
        _normalize_title(_song_to_name.get(track_id, '')),
        _first_artist_key(_song_to_artists_raw.get(track_id, '')),
    )
    seen_keys = {query_key} if query_key[0] else set()

    era_filtered = []
    fallback = []

    for i in top_pool:
        candidate_id = str(_song_ids[i])
        if candidate_id == track_id:
            continue

        cand_name_raw = _song_to_name.get(candidate_id, '')
        cand_artists_raw = _song_to_artists_raw.get(candidate_id, '')
        cand_key = (_normalize_title(cand_name_raw), _first_artist_key(cand_artists_raw))

        if cand_key[0] and cand_key in seen_keys:
            continue
        seen_keys.add(cand_key)

        candidate_year = _song_to_year.get(candidate_id)
        rec = {
            "track_id": candidate_id,
            "name": cand_name_raw if isinstance(cand_name_raw, str) else None,
            "artist": _clean_artists(cand_artists_raw),
            "similarity": float(similarities[i]),
            "year": int(candidate_year) if pd.notna(candidate_year) else None,
        }

        if (query_year is not None and candidate_year is not None
                and pd.notna(query_year) and pd.notna(candidate_year)
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

    for idx in [412, 40_000, 190_483]:
        test_id = str(_song_ids[idx])
        print(f"\n{'=' * 70}")
        print(f"Query: {describe(test_id)}")
        print(f"{'=' * 70}")
        recs = recommend(test_id, k=5)
        for rank, rec in enumerate(recs, 1):
            print(f"\n  Rank {rank}  (sim: {rec['similarity']:.4f})")
            print(f"  {describe(rec['track_id'])}")
