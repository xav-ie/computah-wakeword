#!/usr/bin/env bash
# Builds the harder catch test in data/hard: the 106 held-out Piper clips
# (data/heldout_phonemes, from pipeline.sh) and up to 12 bare and 12 "hey"
# clips from each of four held-out Kokoro voices that pass the same phoneme
# screen, each through a real room and into AMI or People's Speech audio at
# 10, 5 and 0 dB. Needs pipeline.sh, ./download.sh and ../farfield.sh first.
# Score it as the clean clips are: ../benchmark.py --positives data/hard ...
set -euo pipefail
cd "$(dirname "$0")/../.."
podman build -q -t computah-kokoro training/openwakeword/kokoro
RUN="podman run --rm -v $PWD/data:/work/data -v $PWD/training:/work/scripts:ro"
K=/work/data/kokoro_cand
[ -f data/kokoro_cand.done ] || {
  $RUN computah-kokoro python3 /work/scripts/openwakeword/kokoro/heldout.py $K
  touch data/kokoro_cand.done
}
$RUN --device nvidia.com/gpu=all computah-oww bash -c "
set -euo pipefail
S=/work/scripts/openwakeword
python3 \$S/phonescore.py $K > /work/data/kokoro_cand.tsv
cd \$S && python3 - <<'EOF'
import shutil
from collections import defaultdict
from pathlib import Path
from phonejudge import judge
out = Path('/work/data/heldout_kokoro'); shutil.rmtree(out, ignore_errors=True); out.mkdir()
per = defaultdict(int)
for line in open('/work/data/kokoro_cand.tsv'):
    p, ph = line.rstrip('\n').split('\t')
    p = Path(p); voice = '_'.join(p.name.split('_')[:2]); kind = '_hey_' in p.name
    if judge(ph)[0] == 'good' and per[voice, kind] < 12:
        shutil.copy(p, out / f'kokoro_{p.name}'); per[voice, kind] += 1
print('kokoro held-out clips:', sum(per.values()))
EOF
R=/work/data/oww/src/rirs/RIRS_NOISES/real_rirs_isotropic_noises
M='/work/data/media/ami_sdm /work/data/media/peoples_dirty'
rm -rf /work/data/hard
python3 /work/scripts/hardtest.py /work/data/hard/piper \$R \$M -- /work/data/heldout_phonemes
python3 /work/scripts/hardtest.py /work/data/hard/kokoro \$R \$M -- /work/data/heldout_kokoro
"
