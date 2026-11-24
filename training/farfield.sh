#!/usr/bin/env bash
# The two far-field sets benchmark.py measures false wakes on, beside
# LibriSpeech test-clean: AMI meeting room recordings (single distant
# microphone, test split, 9 h) and People's Speech "dirty" (noisy found
# audio, 2 of 11 test shards, 9.8 h). Rebuilt into continuous 16 kHz wavs
# in data/media/ami_sdm and data/media/peoples_dirty. Run in the training
# image (it has pyarrow, soundfile and scipy).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
mkdir -p data/media
cd data/media
get() { # url file sha256
  if [ ! -s "$2" ] || ! echo "$3  $2" | sha256sum -c --quiet - 2>/dev/null; then
    curl -fL --retry 5 -o "$2.part" "$1"
    echo "$3  $2.part" | sha256sum -c --quiet -
    mv -f "$2.part" "$2"
  fi
}
AMI=https://huggingface.co/datasets/edinburghcstr/ami/resolve/46f28f2503e2ec48f8867a84eef356c70476beab/sdm
get $AMI/test-00000-of-00004.parquet ami_sdm_test_0.parquet 90eaf6b410dff4ab03a90af0ed6377c7e0947df55d08645382d4f5e36fd8a864
get $AMI/test-00001-of-00004.parquet ami_sdm_test_1.parquet e8c9b3539a3c074a5ac99c8c9a00ba672da5ea95501fc3ef515c23db2ba8771f
get $AMI/test-00002-of-00004.parquet ami_sdm_test_2.parquet 40b5ecf718d528f5695288a6fe67076492020c1e761aabb6cdc778cdd5fb6129
get $AMI/test-00003-of-00004.parquet ami_sdm_test_3.parquet f0bceadc1368be231fd6c4f6e688750aa39742f6c364fba7ef3790b74d709372
PS=https://huggingface.co/datasets/MLCommons/peoples_speech/resolve/f10597c5d3d3a63f8b6827701297c3afdf178272/dirty
get $PS/test-00000-of-00011.parquet ps_dirty_test_0.parquet 27661190976870a27c63c048afc8ee65ded8efb1255c17de9a4f84dc38d3d51f
get $PS/test-00001-of-00011.parquet ps_dirty_test_1.parquet 00a1ed5395f82c54ce08e0d99ab56172599dfca2875cdd90e5265b9db96ffae3
[ -d ami_sdm ] && [ -d peoples_dirty ] || python3 "$ROOT/training/farfield.py" "$PWD"
