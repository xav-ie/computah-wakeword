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
| `model/computah.json` | Its manifest: cutoff 0.94, 10 ms feature step, window of 5 |

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

Measured by microWakeWord's streaming test on held-out synthetic clips (misses)
and the `dinner_party_eval` set (false wakes), `results/tflite_streaming_roc.txt`:

| Cutoff | Misses | False wakes per hour |
|---|---|---|
| 0.99 | 18.4% | 0.00 |
| 0.97 | 13.8% | 0.38 |
| **0.94** (shipped) | **10.4%** | **0.56** |
| 0.90 | 8.3% | 1.31 |

Raise `probability_cutoff` in the manifest for fewer false wakes, lower it for
fewer misses.

**Limits:** every positive example is a synthetic voice (Piper, LibriTTS-R
speakers), English only. On one real speaker it caught each of four tries at
a desk microphone; accents and far-field rooms are not measured.

## How it was trained

- **Positives:** 10,200 Piper clips of "computah" and five spellings of it
  ("computa", "kum pew tah", …), `training/generate.sh`.
- **Hard negatives:** 6,000 Piper clips of 20 sound-alikes ("commuter",
  "compute", "Jupiter", "potato", …).
- **Augmentation:** room impulse responses (MIT), background music (FMA
  xsmall), EQ, distortion, pitch, noise and gain, `training/prep.py`.
- **Negative sets:** microWakeWord's `speech`, `dinner_party` and `no_speech`.
- **Model:** microWakeWord's MixedNet, 10,000 steps,
  `training/train.py` (parameters in `results/training_parameters.yaml`).

## Training it again

Needs an NVIDIA GPU, podman with CDI and about 45 GB of disk.

```sh
training/download.sh   # pinned sources, each checked by SHA-256
training/pipeline.sh   # samples, features, model
```

Everything is pinned (base image digest, every Python package, the
microWakeWord and piper-sample-generator commits, each dataset's revision),
but a retrain gives an equivalent model, not the same bytes: sample
generation, augmentation and GPU training are random.

## License

The model and its manifest are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):
use them anywhere, crediting "Computah wake word by xav-ie". Its training voices
come from LibriTTS-R through Piper (CC BY 4.0), its negatives from
microWakeWord's datasets and FMA, each under its own license.
