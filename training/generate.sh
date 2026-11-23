#!/usr/bin/env bash
# Synthetic clips for training, under data/v2: "computah" (pos) and the
# sound-alikes it must not wake on (neg), each from both Piper LibriTTS
# generators in the same mix, so the model learns the word and not the
# sound of one generator. Speakers are blended (slerp), and speed and voice
# noise vary per clip.
set -euo pipefail
cd /work
export PYTHONPATH=/work/data/piper-sample-generator
GENERATORS=(
  /work/data/piper-sample-generator/models/en_US-libritts_r-medium.pt
  /work/data/piper-sample-generator/models/en-us-libritts-high.pt
)

gen() { # text count outdir
  local i=0 model
  for model in "${GENERATORS[@]}"; do
    local out="$3/$i"
    i=$((i + 1))
    [ -d "$out" ] && [ "$(ls "$out" | wc -l)" -ge "$2" ] && continue
    python3 -m piper_sample_generator "$1" --model "$model" --max-samples "$2" --batch-size 10 \
      --slerp-weights 0.2 0.5 0.8 \
      --length-scales 0.75 0.9 1.0 1.15 1.3 1.4 \
      --noise-scales 0.5 0.667 0.8 0.9 \
      --noise-scale-ws 0.6 0.8 1.0 \
      --output-dir "$out"
  done
}

# About 25,000 positives: 1,800 per phrase per generator.
i=0
for p in "computah" "computa" "kum pew tah" "com pyoo tah" "khum pyoota" "hey computah" "okay computah"; do
  gen "$p" 1800 "data/v2/pos/$i"
  i=$((i + 1))
done

# About 25,000 sound-alikes: 310 per phrase per generator. "computer" with
# the r is one: only the r-dropped "computah" should wake it.
i=0
for p in "commuter" "compute" "computing" "computers" "come to" "pewter" "Jupiter" "amputate" \
         "compadre" "cucumber" "complete" "compost" "competitor" "computed" "come here" "potato" \
         "pasta" "commuting" "can you" "come on" \
         "computer" "the computer" "my computer" "computer!" "compu" "puter" "come pewter" \
         "compute a" "commuters" "computational" "computation" "competition" "complicated" \
         "copper" "come put her" "come pick her" "cappuccino" "kombucha" "compound" "a pewter"; do
  gen "$p" 310 "data/v2/neg/$i"
  i=$((i + 1))
done
