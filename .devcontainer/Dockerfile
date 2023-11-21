# syntax=docker/dockerfile:1.5

ARG RAPIDS_VERSION=24.02
ARG LLVM_VERSION=16
ARG CUDA_VERSION=12.0
ARG UBUNTU_VERSION=22.04

FROM rapidsai/devcontainers:${RAPIDS_VERSION}-cpp-llvm${LLVM_VERSION}-cuda${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}

ENV CUDAARCHS="RAPIDS"
ENV CUDA_VERSION="${CUDA_VERSION}"

ENV PYTHONSAFEPATH="1"
ENV PYTHONUNBUFFERED="1"
ENV PYTHONDONTWRITEBYTECODE="1"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libaio-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir 'conan<2'
