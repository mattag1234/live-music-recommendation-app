"""
precompute_metadata.py — one-off script.

Reads the 600 MB Kaggle CSV once on a dev machine and writes a compact
pickle of `{spotify_id: (year, name, artist)}` to `data/song_metadata.pkl`.
The pickle is what `recommender.py` loads at runtime — that way the
deployed container never needs pandas or kagglehub.

Artist values in the CSV are stored as Python-list-literal strings like
`"['A', 'B']"`. We pre-clean them here so the runtime doesn't have to
`ast.literal_eval` every lookup.

Run once after you've trained the model and regenerated embeddings:

    python3 precompute_metadata.py
"""

import ast
import os
import pickle
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_csv_path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "song_metadata.pkl")


def clean_artists(raw) -> str:
    if not isinstance(raw, str):
        return ""
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return ", ".join(str(x) for x in parsed)
    except (ValueError, SyntaxError):
        pass
    return raw


def main():
    print("Reading CSV...")
    df = pd.read_csv(get_csv_path(), usecols=["id", "name", "artists", "year"])
    print(f"  loaded {len(df):,} rows")

    print("Building compact metadata dict...")
    meta = {}
    for row in df.itertuples(index=False):
        year = int(row.year) if pd.notna(row.year) else None
        name = row.name if isinstance(row.name, str) else ""
        artist = clean_artists(row.artists)
        meta[row.id] = (year, name, artist)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print(f"Writing {OUT_PATH}...")
    with open(OUT_PATH, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Done. {size_mb:.1f} MB, {len(meta):,} entries.")


if __name__ == "__main__":
    main()
