FROM python:3.12-slim

WORKDIR /app

# System deps for scientific Python wheels (scipy/scikit-learn) to build
# cleanly on slim images that lack prebuilt wheels for some platforms.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ and models_artifacts/ hold the SQLite lab database and trained
# model files. Mount a volume here in production so they survive
# container restarts/redeploys (see hosting notes) -- without a volume,
# this directory is just the container's own ephemeral filesystem.
RUN mkdir -p data models_artifacts
VOLUME ["/app/data", "/app/models_artifacts"]

# Default for local `docker run`/docker-compose, where nothing sets $PORT.
ENV PORT=8501
EXPOSE 8501

HEALTHCHECK CMD curl --fail "http://localhost:${PORT}/_stcore/health" || exit 1

# Shell form (not JSON-array) so $PORT is actually substituted -- Render,
# Railway and most PaaS hosts inject their own PORT env var and expect
# the container to bind to it rather than a hardcoded port.
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0
