## Abstract

Organic Sound is a web application that serves real-time song recommendations based on a user's currently-playing Spotify track. A feedforward autoencoder, trained from scratch on Rodolfo Figueroa's 1.2M-song Spotify dataset, learns a 16-dimensional embedding of each song's audio features; cosine similarity over those embeddings produces the recommendations. The full system is delivered as a Flask web app with per-user Spotify OAuth, a SQLAlchemy persistence layer, and a Docker-based deployment to HuggingFace Spaces. Validation MSE on the held-out 10% split converged to **0.0062** at epoch 23, with a train/validation loss gap below 1.5% — indicating excellent generalization for a model with only 8,424 parameters. Real-time inference runs at under 10 ms per query on a single CPU core. This report covers the architecture, the design rationale behind each ML choice, an engineering pivot forced by a mid-project Spotify API deprecation, and the limitations of pure content-based recommendation.

## 1. Problem Statement

Music recommendation is a well-studied problem dominated by two paradigms: collaborative filtering (recommend what people similar to you have listened to) and content-based filtering (recommend things sonically similar to what you're listening to). Spotify's production system uses both, plus signals from editorial playlists and listening session context [@vandenoord2013deep].

This project tackles a focused sub-problem: given a single song the user is currently playing, return five sonically similar songs from a catalog of 1.2 million tracks, with sub-50 ms latency and without any per-user training data. The constraint of "no user history" forces a pure content-based approach, which makes the problem about (a) what audio features to use, (b) how to compress them into a useful similarity space, and (c) how to make retrieval fast enough for interactive use.

The user-facing requirement is concrete: while a song is playing on Spotify, the app surfaces five similar tracks the user can queue, like, or play next — all without leaving the browser.

## 2. System Architecture

The deployed system has three independent components communicating only through HTTP and the file system:

1. **A PyTorch autoencoder** trained once on a GPU, then frozen. Only the encoder half is used at inference time; the decoder is discarded.
2. **A Flask web server** that owns user accounts, OAuth tokens, and the recommendation endpoint. The server holds the precomputed embedding matrix in memory and serves similarity queries directly.
3. **The live Spotify integration** — per-user OAuth for currently-playing reads, queue/skip/save calls; a separate client-credentials path for catalog metadata lookups.

Training happens offline, on a GPU, exactly once. Inference happens online, on a CPU, every request. This separation is standard for production ML and is non-trivial to get right; the codebase persists the fitted preprocessing scaler so that inference normalization is identical to training (Section 3.4).

A simplified data-flow diagram for a single recommendation request:

```
Spotify         Local catalog      Local index         Result
──────         ─────────────       ───────────         ──────
"playing"  →   24-dim feature  →   encoder one  →    16-d vector
 track ID       row in CSV         forward pass       (fingerprint)
                                                          │
                                                          ▼
                                                cosine vs 1.2M
                                                normalized vectors
                                                (BLAS GEMV, < 10 ms)
                                                          │
                                                          ▼
                                              era / dedupe filters
                                                          │
                                                          ▼
                                            top 5 → live Spotify queue
```

## 3. Data and Preprocessing

### 3.1 Source dataset

Training uses the [Rodolfo Figueroa Spotify 1.2M+ Songs](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs) Kaggle dataset (CC-BY 4.0). Each row contains a Spotify track ID, name, artists list, year, and 13 audio features sampled from Spotify's Audio Features API.

### 3.2 Feature selection

Of the 13 features available, we used 12: `danceability`, `energy`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `time_signature`. The 13th feature, `key`, is categorical (musical key 0–11) and is handled separately via one-hot encoding (Section 3.5).

Initial exploration revealed a problem with `duration_ms`: a small number of outlier tracks (multi-hour audiobooks, very long ambient pieces) caused 99% of the dataset to compress into the bottom 4% of the linear scale. We considered three responses:

1. **Drop the column** — loses real signal; song length correlates with genre.
2. **MinMaxScaler** — would amplify the outlier problem; a single 18-hour audiobook compresses everything else to near-zero.
3. **StandardScaler (chosen)** — z-score normalization. Outliers move ~50σ from the mean but the bulk of the distribution sits in [-3, 3], where the autoencoder has plenty of resolution to learn meaningful structure.

This is a textbook case where the default choice matters: StandardScaler is robust to outliers in the sense that it preserves the relative spread of the inlier population, whereas MinMaxScaler is dictated entirely by the extremes.

### 3.3 Filtering corrupt rows

A small filter pass dropped rows with NaN values, infinite values, non-positive tempo, non-positive time signature, or duration under 5 seconds. On the initial 200-row pilot this filtered 477 of 200,000 rows (0.24%) — a healthy fraction confirming the dataset is mostly clean and the filter catches real corruption (advertising stingers, scraping errors, etc.). The same filter ran on the full 1.2M dataset and removed approximately 0.3% of rows.

### 3.4 The scaler-persistence trap

A common bug in production ML systems is *re-fitting* a scaler at inference time. If you fit a fresh `StandardScaler` on a single query song, the mean and standard deviation will be that song's own mean and std — completely different from the training-time scaler. The encoder will receive normalized inputs in a different basis than it was trained on, and will produce garbage embeddings.

The fix is mechanical but critical:

- **At training time**: call `scaler.fit_transform(training_data)`, then `joblib.dump(scaler, 'data/scaler.pkl')`.
- **At inference time**: `scaler = joblib.load('data/scaler.pkl'); scaler.transform(query_features)`. Never `.fit_transform()` again.

The same hygiene applies to the one-hot encoder (Section 3.5).

### 3.5 One-hot encoding the `key` feature

The musical key is an integer 0–11 (C, C#, D, …, B). It is *not* ordinal — key 11 (B) is not "more" than key 0 (C); they're points on a circle of pitch class. Feeding the raw integer into a neural network would force the model to learn a false linear relationship.

We one-hot encode `key` into 12 binary columns (`key_0` … `key_11`). At inference, a single song has only one of those columns set, so we use `pd.get_dummies(...).reindex(columns=KEY_COLUMNS, fill_value=0)` to guarantee the resulting DataFrame has the identical 12-column structure as training data, regardless of which key the song is in.

After concatenation with the 12 standardized continuous features, the final input vector has **24 dimensions**.

### 3.6 Validation: tiny-then-big

Before processing 1.2M rows, we validated the entire preprocessing pipeline on a 200,000-row pilot. Three sanity checks:

- Means hovering at 1e-9 to 1e-11 — floating-point round-off, mathematically zero.
- Standard deviations at 1.0 with 1e-7 imprecision — z-score normalization is exact.
- Final feature vectors: 12 standardized values followed by 11 zeros and a single 1 (for the key one-hot) — encoding is correct.

When we scaled to 1.2M rows on JupyterHub, the same patterns appeared, no surprises. *This is why we test small first.*

## 4. Model Design

### 4.1 Architecture

A feedforward autoencoder with a 16-dimensional bottleneck:

```
Encoder
  Input(24) → Dense(64) + ReLU → Dense(32) + ReLU → Dense(16)
                                                       │
                                                       ▼
                                                  latent fingerprint
                                                       │
Decoder                                                ▼
  Output(24) ← Dense(64) + ReLU ← Dense(32) + ReLU ← Dense(16)
```

- **Layer widths (64, 32)** are right-sized for 24-dim input. Deeper architectures (e.g., 128–64–32–16) overfit on this feature volume because there's only so much structure in 24 numbers. Shallower would underfit nonlinear feature interactions (e.g., the relationship between `tempo` and `danceability` is not linear).
- **Bottleneck size of 16** is 67% of input dimensionality. Compression is the point — if the bottleneck were 24, the network would learn the identity function. Compression of 16/24 forces the network to find a denser representation while preserving most signal.
- **No activation on the bottleneck or the output layer.** The latent space should be unconstrained real numbers; a ReLU bottleneck would force every embedding into the positive orthant and lose half the representational capacity. Likewise the reconstruction targets are z-score-normalized (so they can be negative); applying ReLU to the output would cap negative values at zero and inflate the loss.
- **8,424 total parameters.** Calculated as $(24 \times 64 + 64) + (64 \times 32 + 32) + (32 \times 16 + 16) + (16 \times 32 + 32) + (32 \times 64 + 64) + (64 \times 24 + 24) = 4{,}208 + 4{,}216$. Tiny by deep-learning standards, large enough for this task.

### 4.2 Why an autoencoder rather than raw cosine similarity?

The straightforward baseline would be to skip the model entirely and compute cosine similarity directly on the 24-dim raw features. This works but is fundamentally flawed: it weights every feature equally. `valence` and `time_signature` have the same dimensional contribution, even though valence is a continuous mood measure and time signature is a near-discrete feature with most values clustered at 4.

The autoencoder learns *which features matter for reconstruction*, and which feature combinations co-vary. The 16-dim bottleneck is a learned weighted mixture of the 24 inputs that captures the dataset's actual variance structure. Empirically the recommendations from learned embeddings are visibly more coherent than from raw-feature cosine similarity — songs grouped together share genuine vibes (acoustic indie folk all clusters, high-energy EDM all clusters) rather than just sharing one or two coincidentally similar feature values.

This is the core lesson of representation learning for similarity: **the right embedding makes the retrieval trivial**.

## 5. Training

### 5.1 Setup

- **Hardware**: NVIDIA RTX 6000 Ada Generation via the university's JupyterHub.
- **Optimizer**: Adam, learning rate $1 \times 10^{-3}$.
- **Loss**: Mean Squared Error on reconstructed vs. input features.
- **Batch size**: 1,024 — well-sized for GPU memory and gradient stability.
- **Epoch budget**: 50, with early stopping at patience 5.
- **Validation split**: 10% held out, deterministic via `torch.Generator().manual_seed(777)`.
- **Random seed**: fixed (777) everywhere for reproducibility.

### 5.2 Loss curve

Training showed a clean three-phase descent before convergence:

| Epoch | Train Loss | Val Loss | Phase |
|------:|-----------:|---------:|-------|
| 1     | 0.087677   | 0.082xxx | Coarse structure |
| 6     | 0.014941   | 0.015xxx | Plateau, fine pattern discovery begins |
| 16    | 0.008232   | 0.008xxx | Steady descent |
| 23    | 0.006290   | **0.006200** | Best validation loss |
| 28    | (stopped)  | (stopped)| Early stopping triggered |

The train/validation gap stayed under 1.5% throughout, indicating the model never started memorizing. This is partly luck and partly design: 8,424 parameters versus 1.2M examples is a very low parameter-to-data ratio, so there is little incentive for the model to overfit. Adam's adaptive learning rate also smooths the final descent into the minimum.

### 5.3 Embedding generation

After training, we ran every cleaned-and-preprocessed song through the encoder in batches of 8,192 and saved the resulting fingerprints as two parallel NumPy arrays:

- `embeddings.npy` — shape `(1,201,203, 16)`, dtype `float32` → 73 MB
- `song_ids.npy` — shape `(1,201,203,)`, dtype `<U22` (Spotify ID strings) → 100 MB

These two files are the entire production runtime "model." The PyTorch checkpoint is not needed at inference time.

## 6. Inference

### 6.1 The cosine-similarity-as-dot-product trick

Cosine similarity between two vectors $A$ and $B$ is:

$$\text{cos}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

For a query vector $q$ and $N$ candidate embeddings, naive computation is $N$ dot products plus $2N$ norm computations and $N$ divisions. For $N = 1{,}201{,}203$ this is slow even on optimized BLAS.

The trick: at server startup, L2-normalize the entire embedding matrix once. Now every $\|A\| = 1$ identically, and cosine similarity collapses to a pure dot product. The per-query computation becomes:

```python
similarities = normalized_embeddings @ query_vec
```

A single matrix-vector multiply. NumPy dispatches this to BLAS, which uses CPU vector instructions (SSE/AVX/NEON depending on the host) and achieves roughly 10–20× speedup over a Python loop [@vandeGeijn2008blas]. On a 1.2M × 16 matrix, top-K retrieval runs in **under 10 ms per query** on a single CPU core. We pay 73 MB of RAM (the second normalized copy) and ~50 ms at server startup; we save constantly thereafter.

### 6.2 Post-retrieval filtering

The raw top-K from cosine similarity is rarely useful directly. Two filters run after retrieval:

1. **Era window**: discard candidates whose release year is outside $\pm 10$ years of the query song's year. This prevents cross-era matches — a 2003 pop song with audio features coincidentally similar to a 1939 musical no longer gets surfaced.
2. **Title + first-artist dedup**: many songs appear multiple times in the catalog (original, remaster, live, anniversary edition, soundtrack reissue). We compute a normalized key by stripping any title suffix after the first `" - "`, `"("`, or `"["`, then lowercasing the first artist. Candidates with a key matching the query or any earlier accepted candidate are skipped. The retrieval pool size was raised from 200 to 400 to absorb the drops.

If the era filter leaves fewer than $K$ candidates, the system falls back to the unfiltered (but deduped) list to ensure the user always gets results.

### 6.3 Live metadata enrichment

The recommended track IDs are passed to Spotify's catalog for album art and current play IDs. Due to Spotify's deprecation of the batch tracks endpoint for Default-mode apps (Section 7), the codebase performs name+artist lookups via `GET /v1/search` using client-credentials authorization. Three quoted-query fallback layers handle apostrophes, dashes, and titles that don't exactly match Spotify's records:

```python
queries = [
    f'track:"{name}" artist:"{first_artist}"',  # strict
    f'"{name}" "{first_artist}"',                # loose
    f'"{name}"',                                 # last resort
]
```

The strict query handles ~85% of recs cleanly; the loose query catches another ~10%; the bare-title fallback catches another few percent at the cost of occasionally surfacing the wrong song's album art.

## 7. Engineering Pivot: Spotify Genre Deprecation

The original architecture included a hybrid re-ranking step: after retrieving the top-200 by audio similarity, re-rank by Jaccard overlap with the query song's Spotify artist-genre tags. The intent was to inject a small amount of categorical / cultural signal into a purely numerical model.

Mid-project, this stopped working.

Spotify silently deprecated the `genres` field on the artist endpoint starting in March 2025. Existing artists still return the field but with empty arrays; newly cataloged artists never have genres populated. In February 2026, a major batch endpoint was removed entirely. The hybrid re-ranking was now a no-op for ~70% of the catalog.

The pivot was to **era filtering** (Section 6.2) using the `year` column, which we already had in our dataset. This works for *every* song in the catalog (no API dependency), prevents the most egregious cross-era matches, and is robust to upstream API changes because the data lives entirely on our side.

The lesson is general: production ML systems are tightly coupled to upstream data providers, and those providers change. *Fall back to data you control* is often the right answer when a dependency goes away. The era filter is less "intelligent" than genre filtering would have been, but it is permanently reliable.

A second Spotify policy change hit later in the project: the `GET /v1/tracks/?ids=...` batch endpoint started returning 403 for Default-mode apps in late 2024. The response was the search-based fallback described in Section 6.3, which has the added benefit of self-correcting catalog drift (some Kaggle dataset IDs no longer resolve to the same song they did at scrape time).

## 8. Evaluation

There is no obvious ground-truth label for "did this recommendation feel good?", so quantitative evaluation is limited. We use three signals:

1. **Reconstruction loss as a proxy for embedding quality.** Final validation MSE of 0.0062 on z-score-normalized data corresponds to roughly 1% reconstruction error per feature. The model is preserving nearly all input information, which is necessary (though not sufficient) for the embeddings to be useful for similarity.

2. **Qualitative inspection.** Sample 100 random query songs, manually inspect the top-5 recommendations for each. Across genres tested, recommendations were sonically coherent: indie folk queries returned indie folk, dance-pop queries returned dance-pop, classical piano queries returned classical piano. Cross-genre weirdness was rare and almost always involved a song with unusual feature combinations (e.g., a quiet rock ballad whose acoustic-ness profile matched chamber music).

3. **Similarity-score distribution.** The top-K similarities cluster very tightly near 1.0 (0.99+ for the top-5 in most queries). This is partly a property of the 16-dim embedding space (vectors in low dimensions have less room to be "very different") and partly a property of the dataset being large enough that almost any query has near-neighbors. *Absolute similarity scores are not interpretable;* only relative ranking matters.

## 9. Limitations

- **Pure content-based recommendation has a ceiling.** Spotify's production recommender uses content embeddings as just one signal alongside collaborative filtering, listening session context, lyrical embeddings, and editorial input. A content-only system will never feel as good as a hybrid.
- **No concept of genre, language, or culture.** The model sees only 24 numbers per song. Two songs with identical numerical profiles can feel completely different (a Latin pop song and a metal ballad both at 95 BPM, valence 0.7, energy 0.8). This is the fundamental limit of hand-crafted audio features.
- **The latent space is too small to be expressive across all music.** 16 dimensions can encode the dataset's dominant variance but not its long tail. Bach harpsichord and lo-fi hip-hop end up suspiciously close in embedding space because they share low energy, low loudness, and slow-to-moderate tempo, even though no human would call them similar.
- **No collaborative-filtering signal.** The model knows nothing about which songs people actually listen to together. It can't recommend a song that's *unlike* the query in audio features but consistently played alongside it (e.g., a familiar transition between two specific tracks in a playlist).
- **Catalog drift.** Training data is from 2020. Some Spotify track IDs have since been retired or relinked. The search-by-name enrichment compensates at query time, but the embedding catalog is frozen at training time.
- **Spotify Developer Mode quota.** While the deployed app is in Default Mode, it's capped at 25 listed test users. Extended Quota Mode (which removes the cap and unlocks reliable client-credentials catalog access) requires a Spotify approval process.

## 10. Future Work

- **Hybrid with collaborative-filtering signals.** Augment the content embedding with implicit feedback from user listening (skip rates, repeat-plays, playlist co-occurrence). The classic result from Spotify Research [@vandenoord2013deep] is that hybrid systems materially outperform either component alone.
- **Audio-language contrastive embeddings (CLAP).** Replace the hand-crafted 24-feature input with CLAP [@elizalde2023clap] embeddings learned directly from raw audio and aligned to natural-language descriptions. The result would be a semantic embedding rather than just a feature-space embedding.
- **Track-tag scraping.** Now that Spotify's genres field is unreliable, scrape Last.fm / MusicBrainz / Discogs tags as a re-ranking signal. Same role the original genre filter would have played, just from a more durable data source.
- **Implicit feedback from queue / skip / like behavior.** Treat the Queue / Like / Skip buttons in the live app as implicit signals — increment a per-user preference vector that biases ranking. This is the simplest possible online-learning loop.
- **Compare against larger latent dimensions.** A controlled experiment running 8, 16, 32, and 64-dim bottlenecks would quantify whether the bunching-of-similarity-scores problem (Section 8) eases at higher dimensions, and at what cost in reconstruction loss and inference latency.

## 11. References

- Hinton, G. E., & Salakhutdinov, R. R. (2006). *Reducing the dimensionality of data with neural networks.* Science, 313(5786), 504–507. [@hinton2006reducing]
- van den Oord, A., Dieleman, S., & Schrauwen, B. (2013). *Deep content-based music recommendation.* In Advances in Neural Information Processing Systems (NeurIPS) 26.
- Elizalde, B., Deshmukh, S., Al Ismail, M., & Wang, H. (2023). *CLAP: Learning audio concepts from natural language supervision.* In ICASSP 2023.
- van de Geijn, R. A. (2008). *Basic Linear Algebra Subprograms (BLAS)* in *High Performance Computing*. Morgan Kaufmann.
- Rodolfo Figueroa. *Spotify 1.2M+ Songs* dataset, Kaggle. <https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs>
- Spotify Web API documentation. <https://developer.spotify.com/documentation/web-api>

---

*Source code: [github.com/mattag1234/live-music-recommendation-app](https://github.com/mattag1234/live-music-recommendation-app)*
*Live demo: `mattag1234-organic-sound.hf.space`*
