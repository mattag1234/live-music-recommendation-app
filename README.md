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

**A live "DJ co-pilot" that recommends songs based on what you're listening to right now, using a neural audio embedding trained on 1.2 million tracks.**

Connect your Spotify account, press play on anything, and the app uses a custom-trained PyTorch autoencoder to find sonically similar songs from a 1.2M-track catalog — then queues them, likes them, or skips the current track for you, all without leaving the browser.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Spaces-2496ED?logo=docker&logoColor=white)](https://huggingface.co/docs/hub/spaces-sdks-docker)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Built for **CSE-108 Lab 9** at UC Merced.

---

## Demo

<!-- Add a screenshot or GIF here -->
> _Screenshot / GIF coming soon._
>
> Live demo: `https://mattag1234-organic-sound.hf.space` _(public-mode, dev-tier Spotify access — see "Limitations" below)_

---

## What it does

1. You sign up, log in, and link your Spotify account via OAuth.
2. The app polls your "now playing" track every five seconds.
3. When you click **Get recommendations**, the current track is looked up in a precomputed 1.2M-vector embedding index. Cosine similarity (a single matrix-vector dot product) returns the top candidates in under 10 ms.
4. Candidates are filtered for (a) the query itself, (b) album/EP/remaster duplicates, and (c) era coherence (±10 years from the query song).
5. The final shortlist is enriched with live Spotify metadata via search, then displayed as cards with **Queue / Like** buttons that drive your real Spotify session.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Frontend  (Jinja templates + vanilla JS, Spotify-dark theme)        │
│  signup.html · login.html · player.html                              │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ HTTP / JSON
┌─────────────────────────────┴────────────────────────────────────────┐
│  Flask backend (app.py)                                              │
│  • Flask-Login auth + pbkdf2:sha256 password hashing                 │
│  • Spotify OAuth blueprint (spotify_api.py)                          │
│  • SQLAlchemy ORM (SQLite local · Postgres deployed)                 │
└──────────────┬──────────────────────────────────────┬────────────────┘
               │                                      │
       ┌───────┴─────────┐                ┌───────────┴────────────┐
       │  recommender    │                │  spotify integration   │
       │  ──────────     │                │  ──────────────────    │
       │  • PyTorch      │                │  • per-user OAuth      │
       │    encoder      │                │  • now-playing read    │
       │  • L2-normed    │                │  • queue / skip / save │
       │    embeddings   │                │  • client-creds search │
       │  • cosine sim   │                │    (fallback metadata) │
       │  • era filter   │                │                        │
       │  • title dedupe │                │                        │
       └─────────────────┘                └────────────────────────┘
```

### Inference path (live recommendation)

```
Spotify API           Local catalog               PyTorch encoder
───────────           ─────────────               ───────────────
"currently      →     look up the      →         encoder-only       →  16-dim vector
 playing"             track's 24-dim              forward pass           (the "fingerprint")
 track ID             feature row                                              │
                                                                               ▼
                                                                cosine similarity
                                                                vs all 1.2M song
                                                                fingerprints
                                                                (single BLAS GEMV)
                                                                               │
                                                                               ▼
                                                            top-K filtered by era,
                                                            deduped by title/artist
                                                                               │
                                                                               ▼
                                                       enriched + pushed to Spotify queue
```

---

## Results

| Metric | Value |
|---|---|
| Songs in training set (after filtering) | **1,201,203** |
| Model parameters | **8,424** |
| Latent embedding dimension | **16** (from 24 input features → 33% compression) |
| Best validation MSE | **0.006200** (epoch 23) |
| Train/val loss gap at best epoch | **< 1.5%** (no overfitting) |
| Inference latency (1.2M-vector top-K search) | **< 10 ms** on CPU |
| Training hardware | NVIDIA RTX 6000 Ada Generation |
| Production hardware | HuggingFace Spaces · CPU Basic · 16 GB RAM |

Three-phase training curve: coarse structure (epochs 1–5) → plateau and fine pattern discovery (6–16) → convergence (17–23) → early stopping (28).

---

## Quick Start

```bash
# Clone and set up the environment
git clone https://github.com/mattag1234/live-music-recommendation-app.git
cd live-music-recommendation-app
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure Spotify credentials
cat > .env <<EOF
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:5000/api/spotify/callback
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
EOF

# (Optional) one-time data preparation — trains the model, generates
# embeddings, and builds the runtime metadata pickle. Skip if you
# already have the data/ artifacts.
cd source
python3 train.py                  # ≈ 25 min on GPU, 4–6 hr on CPU
python3 generate_embeddings.py    # ≈ 2 min
python3 precompute_metadata.py    # ≈ 30 sec

# Run
python3 app.py        # → http://127.0.0.1:5000
```

> **macOS note:** macOS Control Center's AirPlay Receiver claims port 5000. Disable it in **System Settings → General → AirDrop & Handoff**, or run on a different port.

---

## Tech Stack

| Layer | Stack |
|---|---|
| ML | PyTorch 2.9 · NumPy · scikit-learn (StandardScaler) |
| Backend | Python 3.9+ · Flask 3.1 · Flask-Login · Flask-SQLAlchemy · spotipy 2.26 |
| Storage | SQLite (local dev) · Postgres (production, via Neon) |
| Auth | Flask-Login sessions · Werkzeug pbkdf2:sha256 hashing |
| Frontend | Jinja2 · vanilla JavaScript · hand-rolled CSS (no framework) |
| Data | Rodolfo Figueroa's [Spotify 1.2M Songs](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs) via `kagglehub` |
| Deployment | Docker · HuggingFace Spaces · HuggingFace Datasets (model artifacts) · gunicorn 23 |

---

## Model Design

A vanilla feedforward autoencoder, deliberately small:

```
Encoder
  Input(24) ── Dense(64) + ReLU ── Dense(32) + ReLU ── Dense(16)
                                                          │
                                                          ▼
                                                     latent fingerprint
                                                          │
Decoder                                                   ▼
  Output(24) ── Dense(64) + ReLU ── Dense(32) + ReLU ── Dense(16)
```

- **No activation** on the bottleneck or on the reconstruction output — the latent space and reconstructions are unconstrained, which empirically gives a cleaner embedding geometry than ReLU/sigmoid bottlenecks.
- **Two hidden layers of 64 and 32 units** — right-sized for 24-dim input. Deeper would overfit; shallower would underfit nonlinear feature interactions.
- **16-dim bottleneck (67% of input dim)** — small enough to force the network to learn structure, large enough to retain near-perfect reconstruction.
- **8,424 parameters total.** Trains in 23 epochs on a single GPU.

### Input features (24 dimensions)

12 continuous: `danceability`, `energy`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `time_signature`. Z-score standardized.

12 binary: one-hot encoding of `key` (musical key, 0–11). Avoids imposing a false linear ordering on circular pitch classes.

---

## Engineering Decisions

### 1. Train on GPU once. Infer on CPU forever.

Embeddings for all 1.2M songs are computed in batches of 8,192 after training, then frozen as `embeddings.npy`. At inference the autoencoder isn't loaded at all — the server only reads the precomputed vectors and the user's current track ID. **Production ML separation** of expensive training from cheap serving.

### 2. L2-normalize once. Reduce cosine similarity to BLAS.

```
naïve:           top-K cosine over 1.2M songs → N divisions + N norms per query
this codebase:   L2-normalize all embeddings at startup (cost: ~73 MB of math)
                 per query: similarities = E @ q   (one matrix-vector multiply)
```

A single NumPy `@` against a 1.2M × 16 matrix dispatches to BLAS, which exploits CPU vector instructions for ~10–20× speedup over Python loops. Result: sub-10 ms top-K on a single CPU core.

### 3. Persistent scaler. Same preprocessing in training and inference.

The `StandardScaler` is fit *once* during training, then `joblib.dump`-ed to `data/scaler.pkl`. Inference loads the saved scaler and calls `.transform()` only — never `.fit_transform()`. Re-fitting at inference would yield garbage embeddings because the normalization basis would differ from training. The `key` one-hot pipeline uses `reindex(columns=KEY_COLUMNS, fill_value=0)` so any single song produces the identical 24-column structure the model was trained on.

### 4. Era filtering instead of genre filtering (the pivot).

Originally planned a hybrid: audio similarity scored, then re-ranked by Spotify artist-genre overlap. Mid-project I discovered Spotify silently deprecated the `genres` field on artist responses starting in March 2025 — values now empty or stale. A major batch endpoint was removed in February 2026.

Pivoted to **era filtering** using the `year` column already in our dataset. Recommendations are constrained to within ±10 years of the query song. Prevents cross-era weirdness (a 2003 pop song matching a 1939 musical with coincidentally similar audio features) and uses only data we control.

### 5. Title-and-artist deduplication after retrieval.

The raw catalog has dozens of duplicate songs per artist (original, deluxe edition, remaster, live, anniversary edition…). Without dedup, the top-5 for "Bohemian Rhapsody" was *five different recordings of "Bohemian Rhapsody"*. Solution:

- Normalize titles by stripping the first `" - "`, `"("`, or `"["` and everything after.
- Compute a dedup key of `(normalized_title, first_artist_lowercased)`.
- Skip candidates whose key matches the query's or any earlier pick.

Pool size bumped from 200 → 400 to absorb the drops.

### 6. Search-based metadata enrichment (resilience to a second Spotify policy change).

Spotify's `GET /v1/tracks/?ids=...` batch endpoint started 403'ing for Default-mode apps in late 2024. The codebase falls back to `GET /v1/search`, which still works, using **three layers of quoted queries**:

```python
queries = [
    f'track:"{name}" artist:"{first_artist}"',  # strict
    f'"{name}" "{first_artist}"',                # loose
    f'"{name}"',                                 # last resort
]
```

Quoting is essential — unquoted apostrophes/dashes inside `track:` qualifiers silently break the parser. This also incidentally fixes the fact that some Kaggle dataset IDs have drifted (Spotify retires/relinks tracks over time), since we now look up by name+artist instead of by ID.

### 7. Production hygiene: env-driven config, single gunicorn worker, lazy data download.

- `SECRET_KEY`, `DATABASE_URL`, `PORT` all read from environment with safe local-dev fallbacks. Legacy `postgres://` URLs are auto-normalized to `postgresql://` for SQLAlchemy 2.x.
- gunicorn runs with **one worker, four threads** — single process so the 173 MB embedding array isn't duplicated per worker, threads to handle concurrent requests (the workload is I/O-bound on Spotify).
- Model artifacts (`embeddings.npy`, `song_ids.npy`, `song_metadata.pkl`) are downloaded from a private HuggingFace Dataset to `/home/user/app/data` on first container start via `data_bootstrap.py`. Keeps the Docker image small and lets us iterate on the model without rebuilding the image.
- A one-off `precompute_metadata.py` collapses the 600 MB Kaggle CSV into a 92 MB pickle, removing pandas and kagglehub from the runtime dependency graph.

### 8. Python 3.9 compatibility quirks.

`pbkdf2:sha256` is forced for password hashing because macOS's LibreSSL Python build lacks `hashlib.scrypt`, which Werkzeug 3 uses by default. `from __future__ import annotations` is imported at the top of `spotify.py` so PEP 604 union syntax (`X | None`) parses on 3.9.

---

## Project Structure

```
.
├── README.md                          ← you are here
├── requirements.txt
├── Dockerfile                         ← HF Spaces build recipe
├── .dockerignore
├── docs/
│   └── PROJECT_REPORT.pdf             ← detailed technical writeup
├── data/                              ← gitignored; model artifacts
│   ├── autoencoder.pt                   trained PyTorch weights
│   ├── scaler.pkl                       fitted StandardScaler
│   ├── embeddings.npy                   1.2M × 16 latent vectors (73 MB)
│   ├── song_ids.npy                     parallel array of Spotify IDs (100 MB)
│   └── song_metadata.pkl                compact id → (year, name, artist) (92 MB)
├── notebooks/
│   └── 01_explore.ipynb               exploratory data analysis
└── source/
    ├── app.py                         Flask app, page routes, recommendation API
    ├── models.py                      SQLAlchemy User / Song / LikedSong
    ├── spotify.py                     spotipy wrappers (OAuth, playback, search)
    ├── spotify_api.py                 Flask blueprint for /api/spotify/* OAuth flow
    ├── recommender.py                 inference: embeddings, similarity, filters
    ├── data_bootstrap.py              first-run download from HF Dataset
    ├── autoencoder.py                 PyTorch model definition
    ├── preprocessing.py               StandardScaler + key one-hot encoder
    ├── train.py                       training loop with early stopping
    ├── generate_embeddings.py         runs trained encoder over the full catalog
    ├── precompute_metadata.py         builds the compact runtime metadata pickle
    ├── data_utils.py                  kagglehub dataset helper
    ├── templates/                     Jinja templates: signup, login, player
    └── static/                        css + js
```

---

## Limitations & Future Work

Honest scope of what this system *cannot* do:

- **Audio features ≠ genre.** Two songs with identical numerical profiles can feel completely different musically. The model has no concept of timbre, language, lyrical content, or cultural context. It will happily recommend a metal song that has the same tempo/energy/key as a Latin pop song.
- **Similarity scores bunch near 1.0.** A 16-dim latent space tends to produce cosine similarities of 0.99+ across most top results. Absolute scores aren't very interpretable; only *relative* ranking matters.
- **Content-only has a ceiling.** Spotify's own recommender uses audio features as just one signal among many (collaborative filtering, listening history, lyrical embeddings, editorial curation). A purely content-based system can't match that.
- **Catalog drift.** The training data is from 2020. Some Spotify track IDs in our dataset have since been retired or relinked. The search-by-name enrichment compensates by looking up current IDs at query time.
- **Spotify Developer mode caps.** While in Default Mode, the app is capped at 25 listed test users. Extended Quota Mode is required for general public use and is gated on Spotify approval.

### Future work

- **Collaborative-filtering signals.** Augment the audio embedding with implicit user-listening data (skip rates, repeat plays). Hybrid systems consistently outperform pure-content systems.
- **CLAP embeddings.** Replace hand-crafted audio features with [CLAP](https://github.com/LAION-AI/CLAP)-style audio-language contrastive embeddings learned directly from waveforms.
- **Track-genre tags.** Now that Spotify's `genres` field is unreliable, scrape Last.fm / MusicBrainz / Discogs tags as a re-ranking signal.
- **Online learning from queue/skip behavior.** Treat user clicks on rec cards as implicit feedback and adapt rankings per user.

---

## Deployment (HuggingFace Spaces · Docker)

The app runs as a Docker Space on HuggingFace's free CPU tier. Free tier is sufficient: 16 GB RAM accommodates the in-memory embedding catalog comfortably, and gunicorn is configured to stay in a single worker so the 173 MB array isn't duplicated.

Model artifacts (`embeddings.npy`, `song_ids.npy`, `song_metadata.pkl`) live in a private HuggingFace Dataset and are downloaded to the container on first request. Production database is hosted on Neon (managed Postgres, free tier).

**Required Space secrets** (set in *Settings → Variables and secrets*):

| Name | Source |
|---|---|
| `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` | Spotify Developer Dashboard |
| `SPOTIFY_REDIRECT_URI` | `https://<user>-<space>.hf.space/api/spotify/callback` (add to Spotify Dashboard as well) |
| `SECRET_KEY` | `python3 -c "import secrets; print(secrets.token_hex(32))"` |
| `DATABASE_URL` | Neon connection string |
| `HF_DATASET_REPO` / `HF_TOKEN` | Private dataset path + read-scope token |

---

## For My Resume

Drop-in bullets, quantified and concrete:

- **Designed and trained a PyTorch autoencoder on 1.2M Spotify songs** to learn 16-dimensional audio embeddings, achieving a validation MSE of 0.0062 with no overfitting (train/val loss gap < 1.5%) at 8,424 parameters.
- **Engineered an end-to-end ML pipeline** spanning data filtering, preprocessing, GPU training (NVIDIA RTX 6000 Ada), embedding generation over the full 1.2M-song catalog, and CPU-based real-time inference at < 10 ms per top-K query via L2-normalized BLAS GEMV.
- **Pivoted recommender architecture from genre-hybrid to era-based filtering** after discovering Spotify's mid-project deprecation of the artist-genres API field — demonstrating resilience to upstream dependency changes in production ML systems.
- **Built a full-stack Flask web application** with per-user Spotify OAuth, session-based authentication with pbkdf2-SHA256 password hashing, SQLAlchemy ORM persistence, and live Spotify queue / skip / save integration.
- **Deployed to production on HuggingFace Spaces via Docker** with model artifacts (173 MB of embeddings) externalized to a private HuggingFace Dataset, managed Postgres on Neon, and a memory-aware gunicorn configuration (single worker + threaded) to handle concurrent requests without duplicating in-memory state.
- **Hardened the Spotify integration against three breaking API changes** (deprecated genres field, restricted batch tracks endpoint, dev-mode quota enforcement) by introducing a three-tier quoted-search fallback and client-credentials authorization for catalog metadata.

---

## Credits

- **Dataset**: [Spotify 1.2M+ Songs](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs) by Rodolfo Figueroa.
- **APIs**: [Spotify Web API](https://developer.spotify.com/documentation/web-api).
- **GitHub**: [live-music-recommendation-app](https://github.com/mattag1234/live-music-recommendation-app)
- **Course**: CSE-108 (UC Merced), Spring 2026.

Built by [Matthew Aguirre](https://github.com/mattag1234), Eduardo Torres, and collaborators for CSE-108 Lab 9.

## License

[MIT](LICENSE)
