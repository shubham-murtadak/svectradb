# Multi-stage optimized production Dockerfile for SvectraDB
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py README.md ./
COPY svectra ./svectra

# Build wheel package
RUN pip install --no-cache-dir build && python -m build --wheel

# Runtime Stage
FROM python:3.11-slim

LABEL maintainer="Shubham Murtadak <shubham@svectradb.io>"
LABEL description="SvectraDB - High-Performance AI Vector Database"

# Create unprivileged user for security
RUN groupadd -r svectra && useradd -r -g svectra -d /data svectra

WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /build/dist/*.whl .
RUN pip install --no-cache-dir *.whl && rm -f *.whl

# Create data directory and assign ownership
RUN mkdir -p /data && chown -R svectra:svectra /data

USER svectra

# Persistent storage mount
VOLUME ["/data"]

# Expose HTTP API & Dashboard
EXPOSE 8080

# Environment configurations
ENV SVECTRA_DATA_DIR=/data
ENV PYTHONUNBUFFERED=1

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/v1/health')" || exit 1

ENTRYPOINT ["svectra-server", "--host", "0.0.0.0", "--port", "8080", "--data-dir", "/data"]
