---
license: cc-by-nc-sa-4.0
language: en
tags:
  - openwakeword
  - wake-word
  - home-assistant
  - tflite
---

# Computah wake word

An [openWakeWord](https://github.com/dscripka/openWakeWord) model that wakes
on "Computah": computer said the American way with the r dropped,
com-pew-TAH. It runs anywhere openWakeWord models do: Home Assistant's
linux-voice-assistant, Wyoming satellites and openWakeWord itself.

| File | What |
|---|---|
| `model/computah.tflite` | The model (208 KB); openWakeWord's built-in melspectrogram and embedding models feed it |
| `model/computah.json` | Its manifest: openWakeWord, cutoff 0.9 |

This is version 3. Versions 1 and 2, microWakeWord models, are in the
history. Over 18.7 hours of far-field audio v3 woke falsely once, where v1
woke 83 times and v2 118, and it does not wake on "computer".

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

`training/benchmark.py` runs a model the way a satellite does. False wakes
come from three sets, each played as one continuous stream: read speech
close to the microphone (LibriSpeech test-clean, 5.4 h), meetings from one
distant microphone (AMI SDM test, 8.9 h) and noisy found audio (People's
Speech dirty, 9.8 h). Catches come from 106 clips of "computah" and
"hey computah" by five Piper voices training never uses. `results/benchmark.txt`:

| Model | Cutoff | LibriSpeech | AMI meetings | People's Speech | Caught |
|---|---|---|---|---|---|
| **Computah v3** | **0.9** (shipped) | **0.00/h** | **0.11/h** | **0.00/h** | **105/106** |
| Computah v3 | 0.5 | 0.19/h | 0.45/h | 0.41/h | 106/106 |
| Computah v2 (microWakeWord) | 0.6 | 4.07/h | 10.12/h | 2.86/h | 106/106 |
| Computah v1 (microWakeWord) | 0.94 | 8.33/h | 4.72/h | 4.19/h | 106/106 |
| openWakeWord "computer" (community) | 0.7 | 0.93/h | 1.91/h | 1.43/h | 100/106 |

"computer" said by the three American held-out voices wakes v3 on 0 of 12
clips and the "computer" model on 11.

Lower `probability_cutoff` in the manifest if it misses you; down to 0.5 it
still wakes falsely less than once an hour on every set.

**Limits:** every positive example is a synthetic voice, English only, and
the held-out clips pass the same pronunciation screen as the training clips.
Real voices and accents beyond American and British are not benchmarked.

## How it was trained

The fix that mattered was in the data. Spelled "computah" ("computah",
"computa", "kum pew tah"), as v1 and v2 were trained, came out of most
synthetic voices as "computer"; a listening check found that a fifth of v2's
held-out clips said "computer" too. v3's positives come from phonemes
instead, and every clip is screened before training sees it.

- **Positives:** 105,000 candidates of "computah" and "hey computah" from
  exact phonemes (stress on TAH, or on PEW with an open "ah"), from both
  Piper LibriTTS generators (904 speakers, blended) and 24 Piper voice models
  (186 speakers), each faded and padded so no voice ends in a click
  (`training/openwakeword/positives.py`).
- **Screening:** a wav2vec2 phoneme recognizer transcribes every candidate
  (`phonescore.py`), and only clips heard as com-pew-tah with no r at the
  end are kept (`phonejudge.py`, `select.py`): 21,335 of them, at most 600
  per voice model. On clips a person had labelled, it rejected every one
  they called bad.
- **Sound-alikes:** 24,800 clips of 40 near-misses from the LibriTTS
  generators (`training/generate.sh`) and 14,400 of 29 from the 24 voice
  models (`negatives_voices.py`). The voice models leave out "computer" and
  "compute a": through them they sounded like "computah" and taught the model
  to reject the wake word.
- **General negatives:** openWakeWord's 2,000 hours of ACAV100M features.
- **Augmentation and model:** openWakeWord's own (room impulse responses,
  background music, noise, EQ; a 32-unit DNN over its speech embeddings),
  50,000 steps (`training/openwakeword/computah.yml`).

## Training it again

Needs an NVIDIA GPU with 8 GB, podman with CDI and about 150 GB of disk.

```sh
training/download.sh                # pinned sources, each checked by SHA-256
training/openwakeword/download.sh   # openWakeWord's features and 24 voice models
training/openwakeword/pipeline.sh   # sound-alikes, candidates, screening, model
training/farfield.sh                # the AMI and People's Speech benchmark sets
training/benchmark.py --negatives data/media/ami_sdm --positives data/heldout_phonemes \
  --openwakeword data/oww/computah.tflite 0.9
```

Everything is pinned (base image, Python packages, repository commits, each
dataset's and model's revision), but a retrain gives an equivalent model,
not the same bytes: generation, augmentation and training are random.
`training/pipeline.sh` and the files beside it train v2, the microWakeWord
model.

## License

The model and its manifest are [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/),
credited "Computah wake word by xav-ie": its general negatives
(openWakeWord's ACAV100M features) and 4 of its 24 voice models (semaine,
hfc_female, hfc_male, l2arctic) are non-commercial, so the model is too, as
openWakeWord's own models are. Versions 1 and 2 in the history remain
CC BY 4.0. The other sources (LibriTTS and LibriTTS-R through Piper,
LibriSpeech, AMI, People's Speech, FMA, the other voice models) are each
under their own license.
