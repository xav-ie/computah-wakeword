#!/usr/bin/env bash
# The held-out positives benchmark.py counts catches on: "computah" from five
# single-speaker Piper voices (American and British) that training never
# uses, at three speeds and two noise levels, into data/heldout_pos.
# Run inside the training image (it has piper-tts).
set -euo pipefail
cd /work/data
mkdir -p heldout_pos
for voice in heldout_voices/*.onnx; do
  name=$(basename "$voice" .onnx)
  for phrase in "computah" "hey computah"; do
    for speed in 0.85 1.0 1.2; do
      for noise in 0.5 0.8; do
        out="heldout_pos/${name}_${phrase// /_}_${speed}_${noise}.wav"
        [ -s "$out" ] || echo "$phrase" | piper -m "$voice" --length-scale "$speed" --noise-scale "$noise" -f "$out" 2>/dev/null
      done
    done
  done
done
ls heldout_pos | wc -l
