---
title: Organic Sound
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI music recommendations from your live Spotify playback.
---

# Organic Sound

A web app that watches what you're listening to on Spotify in real time and suggests songs that sound similar, using a neural network trained on 1.2 million tracks. Click the recommendations to queue them on your own Spotify device, like them, or skip to the next track — all without leaving the browser.

Built for **CSE-108 Lab 9** at UC Merced.

---

## What it does

1. You sign up, log in, and link your Spotify account via OAuth.
2. The app polls your "now playing" track every five seconds.
3. When you click **Get recommendations**, the current track's audio features are fed through an autoencoder we trained from scratch. The encoder's 16-dimensional bottleneck vector becomes a fingerprint for that song.
4. The fingerprint is compared (cosine similarity) against the fingerprints of 1.2M other songs we pre-computed and stored in a 73 MB numpy array.
5. The top matches are filtered to remove (a) the song itself, (b) duplicates from EPs / remasters / live versions, and (c) anything outside a ±10 year era window from the query song.
6. The five best results are looked up on Spotify by name + artist so we can show real album art and queue / like / skip them through the live Spotify API.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Frontend (Jinja templates + vanilla JS, Spotify-dark theme)     │
│  - signup.html, login.html, player.html                          │
└────────────────────────┬─────────────────────────────────────────┘
                         │ HTTP / JSON
┌────────────────────────┴─────────────────────────────────────────┐
│  Flask (app.py)                                                   │
│  - Flask-Login auth, pbkdf2 password hashing                      │
│  - Spotify OAuth blueprint (spotify_api.py)                       │
│  - SQLAlchemy ORM over SQLite (local) or Postgres (deployed)      │
└────────────────────────┬─────────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
┌───────┴────────┐               ┌────────┴────────┐
│  recommender   │               │  spotify        │
│  • loads 73 MB │               │  • user OAuth   │
│    embeddings  │               │  • currently    │
│  • cosine sim  │               │    playing      │
│  • era filter  │               │  • queue/skip/  │
│  • dedupe      │               │    save         │
└────────────────┘               │  • client-creds │
                                 │    catalog      │
                                 │    lookups      │
                                 └─────────────────┘
```

## How the model was trained

The model is a vanilla autoencoder in PyTorch with shape:

```
input (24 audio features) → 64 → 32 → 16 (bottleneck) → 32 → 64 → 24 (reconstruction)
```

The 24 inputs are Spotify-style audio features (tempo, energy, danceability, valence, acousticness, loudness, etc.) plus a one-hot encoding of the song's musical key.

We trained on the [Rodolfo Figueroa Spotify 1.2M dataset](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs) with a 90/10 train/validation split, MSE reconstruction loss, and early stopping on validation loss. Final validation loss: **0.006200** (very tight reconstruction, meaning the 16-dim bottleneck captures essentially all the signal in the 24-dim input).

After training we ran every song through the encoder once and saved the resulting bottleneck vectors as `data/embeddings.npy`. At runtime the app loads this array, L2-normalizes it, and computes similarity via a single matrix-vector dot product — top-K retrieval over 1.2M songs in about 1 ms.

## Notable engineering decisions

A handful of choices that aren't obvious from reading the code:

- **Era filtering, not genre filtering.** The original plan was to hybridize audio similarity with Spotify's genre tags, but Spotify deprecated the artist-genres field in early 2025. Era filtering (±10 years) was a robust substitute that uses only data we already have in the local CSV, and turned out to give surprisingly coherent recommendations.
- **Title + first-artist deduplication.** The Kaggle dataset has the same song re-listed for every album it appears on — original, deluxe edition, remaster, "Live at Wembley," etc. Without dedupe the top-K is mostly the *exact same song* you asked about, which is useless. We normalize titles by stripping anything after the first " - " / "(" / "[" and key on `(normalized_title, first_artist)`.
- **`pbkdf2:sha256` for password hashing.** Werkzeug's default switched to `scrypt`, which uses an OpenSSL function unavailable in macOS's LibreSSL build of Python 3.9. We force pbkdf2 to keep dev environments working.
- **Search-based metadata enrichment instead of batch track lookup.** As of November 2024, Spotify's `GET /v1/tracks/?ids=...` endpoint returns 403 for unapproved developer apps. We fall back to `GET /v1/search?q=...` (one call per recommended track) using client-credentials auth. This also fixes a latent issue: many of the Kaggle dataset's track IDs are now stale (Spotify has retired or relinked them over five years), so even if the batch endpoint worked, the IDs wouldn't. Searching by name + artist gives us the *current* Spotify ID and album art for queue / like to actually work.
- **`load_dotenv(override=True)`.** Spotify-related env vars in a developer's shell can shadow values in `.env`, which is confusing during onboarding. We tell python-dotenv that `.env` wins.
- **Pre-computed metadata pickle.** The CSV is 600 MB and pulled via kagglehub. To avoid that dependency at runtime, `precompute_metadata.py` is run once on a dev machine to write a 92 MB pickle containing just `{spotify_id: (year, name, artist)}`. The deployed container ships only the pickle.

## Local development

### Prerequisites
- Python 3.11 (matches the deployed Docker image; 3.9 also works locally)
- A Spotify Developer app with `http://127.0.0.1:5000/api/spotify/callback` added to the redirect URI list
- Roughly 1 GB of disk for the dataset and ~300 MB of free RAM

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the repo root:

```bash
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SPOTIFY_REDIRECT_URI=http://127.0.0.1:5000/api/spotify/callback
SECRET_KEY=any-long-random-string
```

### First-run data generation

These are one-off steps you only do once (or whenever the dataset changes):

```bash
cd source

# 1. Train the autoencoder (~30 min on CPU, faster on GPU)
python3 train.py

# 2. Generate embeddings for all 1.2M songs
python3 generate_embeddings.py

# 3. Build the compact metadata pickle used at runtime
python3 precompute_metadata.py
```

After this `data/` contains: `autoencoder.pt`, `scaler.pkl`, `embeddings.npy`, `song_ids.npy`, `song_metadata.pkl`.

### Run the app

```bash
cd source
python3 app.py
# → http://127.0.0.1:5000
```

> macOS users: if Flask refuses to start because port 5000 is in use, that's AirPlay Receiver. Disable it in **System Settings → General → AirDrop & Handoff**, or run on a different port.

## Deployment (HuggingFace Spaces, Docker SDK)

The app deploys to a HuggingFace Space running a custom Docker image. Free tier gives 16 GB RAM and 2 vCPU — plenty for the 173 MB of in-memory embeddings.

The data files (`embeddings.npy`, `song_ids.npy`, `song_metadata.pkl`, `autoencoder.pt`) live in a private HuggingFace Dataset and are downloaded to the container's filesystem on first start. The Postgres database is hosted on Neon.

**Required Space secrets** (set in the Space's Settings → Variables and secrets):

| Name | Source |
|---|---|
| `SPOTIFY_CLIENT_ID` | Spotify Developer Dashboard |
| `SPOTIFY_CLIENT_SECRET` | Spotify Developer Dashboard |
| `SPOTIFY_REDIRECT_URI` | `https://<user>-<space>.hf.space/api/spotify/callback` (also add to Spotify Dashboard) |
| `SECRET_KEY` | Any long random string |
| `DATABASE_URL` | Neon Postgres connection string |
| `HF_DATASET_REPO` | Path to the private dataset holding the model files |
| `HF_TOKEN` | A read-scope HF token with access to the dataset |

## Project layout

```
.
├── README.md                 ← you are here
├── requirements.txt
├── Dockerfile                ← HF Spaces build recipe
├── data/                     ← gitignored; trained model + embeddings live here
│   ├── autoencoder.pt
│   ├── scaler.pkl
│   ├── embeddings.npy
│   ├── song_ids.npy
│   └── song_metadata.pkl
├── notebooks/
│   └── 01_explore.ipynb      ← data exploration
└── source/
    ├── app.py                ← Flask app, page routes, recommendation API
    ├── models.py             ← User / Song / LikedSong SQLAlchemy models
    ├── spotify.py            ← Spotipy wrappers: OAuth, now-playing, queue, search
    ├── spotify_api.py        ← Flask blueprint for the /api/spotify/* OAuth flow
    ├── recommender.py        ← runtime engine: embeddings, similarity, filters
    ├── autoencoder.py        ← PyTorch model definition
    ├── preprocessing.py      ← StandardScaler + key one-hot encoding
    ├── train.py              ← training loop with early stopping
    ├── generate_embeddings.py← runs the trained encoder over all 1.2M songs
    ├── precompute_metadata.py← builds the runtime metadata pickle
    ├── data_utils.py         ← kagglehub dataset path helper
    ├── extract_artists.py    ← one-off: dumps unique artist names
    ├── fetch_genres.py       ← one-off: legacy genre fetching
    ├── templates/
    │   ├── signup.html
    │   ├── login.html
    │   └── player.html
    └── static/
        ├── css/styles.css
        └── javascript/
```

## Tech stack

- **Backend**: Python 3.11, Flask, Flask-Login, Flask-SQLAlchemy, spotipy
- **ML**: PyTorch (training), numpy (runtime inference), scikit-learn (preprocessing)
- **Database**: SQLite locally, Postgres in production
- **Frontend**: Jinja templates, vanilla JavaScript, hand-rolled CSS (no framework)
- **Deployment**: Docker on HuggingFace Spaces, Neon Postgres, HuggingFace Datasets for model artifacts

## Credits

- Dataset: [Spotify 1.2M Songs](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs) by Rodolfo Figueroa
- Github: [live-music-recommendation-app](https://github.com/mattag1234/live-music-recommendation-app)
- Spotify Web API
- Built by Matthew Aguirre, Eduardo Torres, and collaborators for CSE-108 (UC Merced)

## License

MIT
