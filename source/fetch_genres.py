"""
fetch_genres.py — fetch Spotify genres for every artist in our dataset.

Reads data/unique_artists.txt, batches IDs in groups of 50, calls Spotify's
artists endpoint, and saves {artist_id: [genres]} as JSON. Checkpoint-resumable.
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
ARTISTS_FILE = os.path.join(DATA_DIR, 'unique_artists.txt')
GENRES_FILE = os.path.join(DATA_DIR, 'artist_genres.json')

BATCH_SIZE = 50
CHECKPOINT_EVERY = 20


def make_client() -> Spotify:
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    if not client_id or not client_secret:
        raise RuntimeError("SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env")
    return Spotify(auth_manager=SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret,
    ))


def fetch_genres():
    print("=" * 60)
    print("Fetching Spotify genres for all artists")
    print("=" * 60)

    sp = make_client()
    print("Spotify client authenticated.")

    if not os.path.exists(ARTISTS_FILE):
        raise FileNotFoundError(f"{ARTISTS_FILE} not found. Run extract_artists.py first.")

    with open(ARTISTS_FILE) as f:
        all_artists = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(all_artists):,} artists to fetch.")

    if os.path.exists(GENRES_FILE):
        with open(GENRES_FILE) as f:
            artist_genres = json.load(f)
        print(f"Resuming: {len(artist_genres):,} artists already fetched.")
    else:
        artist_genres = {}
        print("Starting fresh.")

    to_fetch = [a for a in all_artists if a not in artist_genres]
    print(f"Need to fetch {len(to_fetch):,} more artists.")

    if not to_fetch:
        print("All artists already fetched!")
        return

    print("\nStarting fetch loop...\n")
    start_time = time.time()
    n_batches_total = (len(to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE

    batch_idx = 0
    while batch_idx * BATCH_SIZE < len(to_fetch):
        batch = to_fetch[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]

        try:
            result = sp.artists(batch)
            for artist in result['artists']:
                if artist is None:
                    continue
                artist_genres[artist['id']] = artist.get('genres', [])
        except SpotifyException as e:
            print(f"  Batch {batch_idx} failed (code {e.http_status}): {e.msg}")
            print(f"  Sleeping 30s and retrying...")
            time.sleep(30)
            continue  # retry same batch
        except Exception as e:
            print(f"  Batch {batch_idx} unexpected error: {e}")
            print(f"  Sleeping 30s and retrying...")
            time.sleep(30)
            continue

        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            eta_min = (n_batches_total - batch_idx) / rate / 60 if rate > 0 else 0
            print(f"  Batch {batch_idx + 1:,}/{n_batches_total:,} | "
                  f"{len(artist_genres):,} artists fetched | "
                  f"ETA: {eta_min:.1f} min")

        if batch_idx % CHECKPOINT_EVERY == 0 and batch_idx > 0:
            with open(GENRES_FILE, 'w') as f:
                json.dump(artist_genres, f)

        batch_idx += 1

    with open(GENRES_FILE, 'w') as f:
        json.dump(artist_genres, f)

    total = len(artist_genres)
    with_genres = sum(1 for g in artist_genres.values() if g)
    without_genres = total - with_genres
    pct_with = 100 * with_genres / total if total > 0 else 0

    print(f"\nSaved to: {GENRES_FILE}")
    print(f"\nSummary:")
    print(f"  Total artists fetched:       {total:,}")
    print(f"  With at least one genre tag: {with_genres:,} ({pct_with:.1f}%)")
    print(f"  With empty genres list:      {without_genres:,}")


if __name__ == "__main__":
    fetch_genres()