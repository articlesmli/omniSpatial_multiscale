# Stage 1: Build and environment preparation
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system build dependencies
FROM mambaorg/micromamba:1.5.8

# Copy and create the named Conda environment from environment.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml /tmp/environment.yaml
RUN micromamba env create -n omnispatial_env -f /tmp/environment.yaml && \
    micromamba clean --all --yes

# Set working directory
WORKDIR /app

# Copy project files
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
ENV PYTHONPATH=/app

# Use micromamba run so commands execute inside the conda environment
CMD ["micromamba", "run", "-n", "omnispatial_env", "pytest", "tests/"]
