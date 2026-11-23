---
license: cc-by-4.0
language: en
tags:
  - microwakeword
  - wake-word
  - esphome
  - home-assistant
  - tflite
---

# Computah wake word

A [microWakeWord](https://github.com/kahrendt/microWakeWord) model that wakes
on "Computah" (computer, said with the r dropped). It runs anywhere
microWakeWord models do: ESPHome voice devices, Home Assistant's
linux-voice-assistant and other Wyoming satellites.

| File | What |
|---|---|
| `model/computah.tflite` | The streaming, quantized model (61 KB) |
| `model/computah.json` | Its manifest: cutoff 0.6, 10 ms feature step, window of 5 |

This is version 2. Version 1 (cutoff 0.94) is in the history; v2 halves its
false wakes on speech and catches as much.

## Using it

Put both files in one folder and point the satellite at it, for example
linux-voice-assistant:

```sh
linux-voice-assistant --wake-word-dir ./model --wake-model computah
```

Keep the manifest and the model named alike (`computah.json` and
`computah.tflite`). The satellite names the wake word after the manifest and
the loaded model after the file the manifest points to; when the two differ,
it drops the model without a word and never wakes.

## How well it works

`training/benchmark.py` runs a model the way a satellite does over 5.4 hours
of read speech (LibriSpeech test-clean, speakers training never hears) for
false wakes, and over 60 clips of "computah" from five Piper voices training
never uses (American and British) for catches. `results/benchmark.txt`:

| Model | Cutoff | False wakes per hour | Caught |
|---|---|---|---|
| **Computah v2** | **0.6** (shipped) | **4.07** | **57/60** |
| Computah v2 | 0.8 | 2.96 | 52/60 |
| Computah v2 | 0.97 | 0.93 | 42/60 |
| Computah v1 | 0.94 | 8.33 | 56/60 |
| openWakeWord "computer" (community) | 0.7 | 0.93 | 57/60 |

Raise `probability_cutoff` in the manifest for fewer false wakes, lower it for
fewer misses. Read speech close to the microphone is the hard case; on the
far-field `dinner_party_eval` microWakeWord tests with
(`results/tflite_streaming_roc.txt`) v2 reaches 0.56 false wakes an hour with
13% misses at cutoff 0.91.

**Limits:** every positive example is a synthetic voice (two Piper LibriTTS
generators), English only. Accents beyond American and British, real voices
and far-field rooms are not benchmarked.

## How it was trained

- **Positives:** 25,200 Piper clips of "computah", four respellings and
  "hey computah" / "okay computah", half from each of two generators
  (`en_US-libritts_r-medium`, `en-us-libritts-high`), speakers blended and
  speed and voice noise varied, `training/generate.sh`.
- **Sound-alikes:** 24,800 clips of 40 near-misses from the same two
  generators in the same mix ("computer", "commuter", "compu", "puter",
  "cappuccino", …). With every clip from one generator, v1 learned the
  generator's sound as much as the word.
- **Augmentation:** room impulse responses (MIT, probability 0.7),
  background music (FMA xsmall), EQ, distortion, pitch, noise and gain,
  `training/prep.py`.
- **Negative sets:** microWakeWord's `speech`, `dinner_party` and
  `no_speech`, and LibriSpeech dev-clean as plain speech. Speech belongs among
  the negatives: a trial run that instead mixed it under the positives as
  background learned that speech means "computah".
- **Model:** microWakeWord's MixedNet, 20,000 steps then 10,000 at a fifth of
  the learning rate, with time and frequency masking (5/2/5/2),
  `training/train.py` (parameters in `results/training_parameters.yaml`).

## Training it again

Needs an NVIDIA GPU, podman with CDI and about 100 GB of disk.

```sh
training/download.sh   # pinned sources, each checked by SHA-256
training/pipeline.sh   # samples, features, model, held-out clips
training/benchmark.py --negatives data/librispeech_test_16k --positives data/heldout_pos \
  --micro model/computah.json
```

Everything is pinned (base image digest, every Python package, the
microWakeWord and piper-sample-generator commits, each dataset's revision),
but a retrain gives an equivalent model, not the same bytes: sample
generation, augmentation and GPU training are random.

## License

The model and its manifest are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):
use them anywhere, crediting "Computah wake word by xav-ie". Its training voices
come from LibriTTS and LibriTTS-R through Piper (CC BY 4.0), its negatives
from microWakeWord's datasets, LibriSpeech (CC BY 4.0) and FMA, each under its
own license.
