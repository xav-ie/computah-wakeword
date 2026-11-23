#!/usr/bin/env bash
# Fetches everything training reads into ./data, each file at a fixed
# revision and checked against the hash it was trained with.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/dl
cd data

get() { # url file sha256
  if [ ! -s "$2" ] || ! echo "$3  $2" | sha256sum -c --quiet - 2>/dev/null; then
    curl -fL --retry 5 -o "$2.part" "$1"
    echo "$3  $2.part" | sha256sum -c --quiet -
    mv -f "$2.part" "$2"
  fi
}
clone() { # url dir commit
  [ -d "$2/.git" ] || git clone -q "$1" "$2"
  git -C "$2" checkout -q "$3"
}

clone https://github.com/kahrendt/microWakeWord microWakeWord 4665173cd35f1cff9a61e06fc427f124766c488e
clone https://github.com/rhasspy/piper-sample-generator piper-sample-generator 2971426a55072f7d22fec416ca7800df8bd23207
mkdir -p piper-sample-generator/models
get https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt \
  piper-sample-generator/models/en_US-libritts_r-medium.pt e95ee53770bf598c354a6e6dbfc95ccb259aeeb501d35a86be8a767429ab0ff6

MWW=https://huggingface.co/datasets/kahrendt/microwakeword/resolve/0da95f94302ca2f4aae3b18fc6560fa6d2bba3d1
get $MWW/dinner_party.zip dl/dinner_party.zip 18a0885d595ced7faa73736a8680206f5ba6e80113ca3e6ce43130e510aac18f
get $MWW/dinner_party_eval.zip dl/dinner_party_eval.zip e08e23a5e654dd4415a1c41a4b6cd02d1f983c821512367ad937bac1aeed8299
get $MWW/no_speech.zip dl/no_speech.zip 722dbcc967275c64c9581ecb85729549830a9094ce605ee22b2ddab9dbabf3c8
get $MWW/speech.zip dl/speech.zip 68b6b811b347b8321169f9e75ea2b9be73245943b4470af9532ca288d9f1b3cc
get https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/bfef24fd1e1334e4631b917c98e0dbc3dd653ab2/fma_xs.zip \
  dl/fma_xs.zip e5876c13cfb0f7ef668327c75d1c40bc4a2ed9d5b8e62ce383d093319c9ff663
# The second generator: positives and sound-alikes come from both in the
# same mix, so the model cannot learn one generator's sound.
get https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt \
  piper-sample-generator/models/en-us-libritts-high.pt 5258f21adf77c29248d9a89d51ca9492f6e75a498140584dcbed0edcab81960a

# LibriSpeech: dev-clean as plain-speech negatives (prep.py), test-clean
# (other speakers) for benchmark.py's false wakes. 16 kHz mono wav.
SLR=https://openslr.elda.org/resources/12
get $SLR/dev-clean.tar.gz dl/dev-clean.tar.gz 76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3
get $SLR/test-clean.tar.gz dl/test-clean.tar.gz 39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23
for set in dev test; do
  [ -d "librispeech_${set}_16k" ] && continue
  mkdir -p "librispeech_${set}.tmp" "librispeech_${set}_16k.tmp"
  tar -xzf "dl/${set}-clean.tar.gz" -C "librispeech_${set}.tmp"
  find "librispeech_${set}.tmp" -name "*.flac" | while read -r f; do
    ffmpeg -loglevel error -y -i "$f" -ac 1 -ar 16000 "librispeech_${set}_16k.tmp/$(basename "${f%.flac}").wav"
  done
  mv "librispeech_${set}_16k.tmp" "librispeech_${set}_16k"
  rm -rf "librispeech_${set}.tmp"
done

# Held-out voices for benchmark.py's catches (heldout.sh): single-speaker
# Piper voices training never uses.
VOICES=https://huggingface.co/rhasspy/piper-voices/resolve/e5349433ed883a32f893a434dc43a6c66b95ef7c/en
mkdir -p heldout_voices
while read -r path sha sha_json; do
  name=$(basename "$path")
  get "$VOICES/$path.onnx" "heldout_voices/$name.onnx" "$sha"
  get "$VOICES/$path.onnx.json" "heldout_voices/$name.onnx.json" "$sha_json"
done <<'VOICES'
en_GB/alan/medium/en_GB-alan-medium 0a309668932205e762801f1efc2736cd4b0120329622adf62be09e56339d3330 c0f0d124e5895c00e7c03b35dcc8287f319a6998a365b182deb5c8e752ee8c1e
en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium 62ba183a504cfcd8816ec1b50ee856ed318ccc9dd22fd0bedd7d839c95dfa20e 2d28804145e37cf6b277d17417769a2c84f6e4157273678dfe495240cc6e00e9
en_US/amy/medium/en_US-amy-medium cf9ccdec59f05adee47173f6fb4d89e8b0c5f30c178748d2572fe4efd1d2724c 27490c2197e3e7b566b051fc0b1dae21200e72b44c209731419effdbbe0789e5
en_US/lessac/medium/en_US-lessac-medium 5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0
en_US/ryan/high/en_US-ryan-high b3990d7606e183ec8dbfba70a4607074f162de1a0c412e0180d1ff60bb154eca c6d3b98f08315cb4bebf0d49d50fc4ff491b503c64b940cd3d5ca28543b48011
VOICES
