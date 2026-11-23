#!/usr/bin/env bash
# Builds the training image and runs the whole pipeline on an NVIDIA GPU:
# synthetic samples (generate.sh), augmented features (prep.py), then the
# model (train.py). Run training/download.sh first. Writes into ./data;
# the model lands in data/trained_models/computah.
set -euo pipefail
cd "$(dirname "$0")/.."
podman build -q -t computah-train training
exec podman run --rm --device nvidia.com/gpu=all --memory=10g --shm-size=2g \
  -v "$PWD/data:/work/data" -v "$PWD/training:/work/scripts:ro" \
  -e PYTHONPATH=/work/data/microWakeWord:/work/data/piper-sample-generator \
  computah-train bash -c 'set -e; cd /work; bash scripts/generate.sh; python3 scripts/prep.py; python3 scripts/train.py'
