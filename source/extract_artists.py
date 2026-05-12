"""
extract_artists.py — extract every unique artist ID from the Kaggle dataset.

The Kaggle CSV stores artist_ids as Python-list-strings like "['abc', 'def']".
We parse those, flatten them across all 1.2M songs, and produce a unique set.
The output is `data/unique_artists.txt`, one artist ID per line.

This is step 1 of building the genre-hybrid recommender:
  1. extract_artists.py     → list of artist IDs to fetch genres for
  2. fetch_genres.py        → calls Spotify for each, saves artist_genres.json
  3. recommender.py (mod)   → uses genres to filter recommendations
"""

import os
import sys
import ast
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_csv_path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'unique_artists.txt')


def extract_artists():
    print("=" * 60)
    print("Extracting unique artist IDs")
    print("=" * 60)

    # Load only the column we need — much faster than reading all 24 cols
    print("\nLoading dataset (artist_ids column only)...")
    df = pd.read_csv(get_csv_path(), usecols=['artist_ids'])
    print(f"Loaded {len(df):,} rows")

    # ------------------------------------------------------------------
    # Parse the stringified lists.
    #
    # The artist_ids column looks like this in the CSV:
    #     "['2d0hyoQ5ynDBnkvAbJKORj']"
    #     "['abc', 'def']"
    #
    # That's a *string* that LOOKS like a Python list. We need to actually
    # parse it into a real Python list. `ast.literal_eval` is the safe way
    # to do this — it only evaluates literal Python data (lists, dicts,
    # strings, numbers), so it can't execute arbitrary code (unlike eval()).
    # ------------------------------------------------------------------
    print("Parsing artist_ids column...")
    df['artist_ids'] = df['artist_ids'].apply(ast.literal_eval)
    # ------------------------------------------------------------------
    all_artists = set(df['artist_ids'].explode())  

    print(f"\nFound {len(all_artists):,} unique artists across the dataset")

    # ------------------------------------------------------------------
    # Sort the artists alphabetically before saving.
    #
    # Why? Two reasons:
    #   1. Reproducibility — running this script twice gives the same
    #      file in the same order, which is nice for git diffs.
    #   2. Sets in Python have no guaranteed order, so without sorting,
    #      the output file would be in arbitrary order each run.
    # ------------------------------------------------------------------
    sorted_artists = sorted(all_artists)

    # Make sure the data/ folder exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Write one ID per line — simplest possible format
    with open(OUTPUT_PATH, 'w') as f:
        for aid in sorted_artists:
            f.write(aid + '\n')

    file_size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\nSaved to: {OUTPUT_PATH} ({file_size_kb:.1f} KB)")

    # Estimate API call cost so we know what we're getting into
    BATCH_SIZE = 50
    n_batches = (len(all_artists) + BATCH_SIZE - 1) // BATCH_SIZE  # ceiling division
    rate_limit_per_min = 180  # Spotify's documented rate
    est_minutes = n_batches / rate_limit_per_min

    print(f"\n Estimated fetch cost:")
    print(f"  Total batches:  {n_batches:,} (at 50 artists/batch)")
    print(f"  Est. time:      {est_minutes:.1f} minutes ({est_minutes / 60:.1f} hours)")
    print(f"  This is the wall-clock time for fetch_genres.py to complete.")


if __name__ == "__main__":
    extract_artists()