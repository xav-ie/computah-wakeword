#!/usr/bin/env bash
# Builds the training image and runs the whole pipeline on an NVIDIA GPU:
# synthetic samples (generate.sh), augmented features (prep.py), then the
# model (train.py), then the held-out clips benchmark.py scores (heldout.sh).
# Run training/download.sh first. Writes into ./data;
# the model lands in data/trained_models/computah-v2.
set -euo pipefail
cd "$(dirname "$0")/.."
podman build -q -t computah-train training
exec podman run --rm --device nvidia.com/gpu=all --memory=12g --shm-size=2g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -v "$PWD/data:/work/data" -v "$PWD/training:/work/scripts:ro" \
  -e PYTHONPATH=/work/data/microWakeWord:/work/data/piper-sample-generator \
  computah-train bash -c 'set -e; cd /work; bash scripts/generate.sh; python3 scripts/prep.py; python3 scripts/train.py; bash scripts/heldout.sh'
