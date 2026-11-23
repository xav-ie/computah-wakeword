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
