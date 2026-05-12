FROM python:3.11-slim

# HF Spaces convention: run as a non-root user with UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# Install dependencies first so this layer caches across code changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application source. Data files are downloaded at runtime via data_bootstrap.py
# from a private HF Dataset (configured by HF_DATASET_REPO + HF_TOKEN env vars).
COPY --chown=user source/ ./source/

EXPOSE 7860

# Single worker so the 173 MB embedding array isn't loaded multiple times.
# Threads handle concurrent requests (the workload is I/O-bound on Spotify).
# 120s timeout covers the cold-start path: HF download + numpy load + first request.
CMD gunicorn --chdir source --bind 0.0.0.0:7860 \
    --workers 1 --threads 4 --timeout 120 \
    app:app
