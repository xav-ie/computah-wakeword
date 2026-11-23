#!/usr/bin/env bash
set -euo pipefail
cd /work
GEN=/work/data/piper-sample-generator/models/en_US-libritts_r-medium.pt
export PYTHONPATH=/work/data/piper-sample-generator
gen() { # text count outdir
  [ -d "$3" ] && [ "$(ls "$3" | wc -l)" -ge "$2" ] && return 0
  python3 -m piper_sample_generator "$1" --model "$GEN" --max-samples "$2" --batch-size 25 --output-dir "$3"
}
i=0
for p in "computah" "computa" "kum pew tah" "com pyoo tah" "khum pyoota" "computah!"; do
  gen "$p" 1700 "data/pos/$i"; i=$((i+1))
done
i=0
for p in "commuter" "compute" "computing" "computers" "come to" "pewter" "Jupiter" "amputate" \
         "compadre" "cucumber" "complete" "compost" "competitor" "computed" "come here" "potato" \
         "pasta" "commuting" "can you" "come on"; do
  gen "$p" 300 "data/neg/$i"; i=$((i+1))
done
