# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

WORKDIR /app

# Install mindforge-kb with optional embeddings. Build with
# --build-arg INSTALL_EXTRAS= to skip embeddings and ship a slimmer image.
ARG INSTALL_EXTRAS=[embeddings]
RUN pip install --no-cache-dir "mindforge-kb${INSTALL_EXTRAS}"

# Default storage lives outside the image so it persists via volume mount.
ENV MINDFORGE_ROOT=/data/mindforge
VOLUME ["/data/mindforge"]

ENTRYPOINT ["mindforge"]
CMD ["--help"]
