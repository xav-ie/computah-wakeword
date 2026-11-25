---
license: cc-by-4.0
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
| `model/computah.json` | Its manifest: openWakeWord, cutoff 0.95 |

This is version 4, trained only on openly licensed data, so unlike v3 it
is CC BY 4.0 and free for commercial use. It is as accurate as v3: over
18.7 hours of far-field audio it woke falsely 3 times (v3 once, v1 83
times, v2 118), it catches a little more in noise, and it does not wake on
"computer". Earlier versions are in the history.

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
"hey computah" by five Piper voices training never uses. The harder test
puts those clips and 82 from four held-out Kokoro voices through real rooms
and into meeting or street audio at 10, 5 and 0 dB (564 clips).
`results/benchmark.txt`:

| Model | Cutoff | LibriSpeech | AMI meetings | People's Speech | Caught | Harder test |
|---|---|---|---|---|---|---|
| **Computah v4** | **0.95** (shipped) | **0.00/h** | **0.22/h** | **0.10/h** | **105/106** | **178/564** |
| Computah v4 | 0.9 | 0.00/h | 0.34/h | 0.10/h | 106/106 | 201/564 |
| Computah v4 | 0.5 | 0.19/h | 1.12/h | 0.31/h | 106/106 | 252/564 |
| Computah v3 | 0.9 | 0.00/h | 0.11/h | 0.00/h | 105/106 | 173/564 |
| Computah v2 (microWakeWord) | 0.6 | 4.07/h | 10.12/h | 2.86/h | 106/106 | |
| Computah v1 (microWakeWord) | 0.94 | 8.33/h | 4.72/h | 4.19/h | 106/106 | |
| openWakeWord "computer" (community) | 0.7 | 0.93/h | 1.91/h | 1.43/h | 100/106 | 111/564 |

"computer" said by the three American held-out voices wakes v4 on 0 of 12
clips and the "computer" model on 11.

Lower `probability_cutoff` in the manifest if it misses you, at the cost of
more false wakes in meetings (about one an hour at 0.5); raise it to 0.97
for fewer. Training's negatives include other AMI meetings and other
LibriSpeech speakers, so People's Speech is the fairest of the three sets.

**Limits:** every positive example is a synthetic voice, English only, and
the held-out clips pass the same pronunciation screen as the training clips.
Real voices and accents beyond American and British are not benchmarked.

## How it was trained

The fix that mattered was in the data. Spelled "computah" ("computah",
"computa", "kum pew tah"), as v1 and v2 were trained, came out of most
synthetic voices as "computer"; a listening check found that a fifth of v2's
held-out clips said "computer" too. v3's positives come from phonemes
instead, and every clip is screened before training sees it.

- **Positives:** about 100,000 candidates of "computah" and "hey computah" from
  exact phonemes (stress on TAH, or on PEW with an open "ah"), from both
  Piper LibriTTS generators (904 speakers, blended) and 20 Piper voice models,
  each faded and padded so no voice ends in a click
  (`training/openwakeword/positives.py`).
- **Screening:** a wav2vec2 phoneme recognizer transcribes every candidate
  (`phonescore.py`), and only clips heard as com-pew-tah with no r at the
  end are kept (`phonejudge.py`, `select.py`): 20,001 of them, at most 600
  per voice model. On clips a person had labelled, it rejected every one
  they called bad.
- **Sound-alikes:** 24,800 clips of 40 near-misses from the LibriTTS
  generators (`training/generate.sh`) and 12,000 of 29 from the 20 voice
  models (`negatives_voices.py`). The voice models leave out "computer" and
  "compute a": through them they sounded like "computah" and taught the model
  to reject the wake word.
- **General negatives:** 546 hours of openWakeWord features from openly
  licensed audio (`neg_features.py`): LibriSpeech train-clean-100 and 360
  (414 h), MUSAN's CC BY and public domain recordings (86 h; music under
  share-alike or no-derivatives terms is left out by `musan_cc_by.py`) and
  AMI's distant microphone train split (46 h). The validation set openWakeWord
  tunes against is LibriSpeech dev-clean and dev-other (9.3 h).
- **Augmentation and model:** openWakeWord's own (room impulse responses,
  background clips, noise, EQ; a 32-unit DNN over its speech embeddings),
  50,000 steps (`training/openwakeword/computah.yml`). The background clips
  are MUSAN music, speech and noise in about equal parts: with music and
  noise alone the model caught a third less in noisy rooms.

What did not help, on the same benchmarks: Kokoro voices among the positives,
a model four times as wide, and 100,000 steps, which made the model sure of
its false wakes.

## Training it again

Needs an NVIDIA GPU with 8 GB, podman with CDI and about 250 GB of disk.

```sh
training/download.sh                # pinned sources, each checked by SHA-256
training/openwakeword/download.sh   # negatives' audio, room responses, 20 voice models
training/openwakeword/pipeline.sh   # sound-alikes, candidates, screening, negatives, model
training/farfield.sh                # the AMI and People's Speech benchmark sets
training/openwakeword/hardtest.sh   # the harder test
training/benchmark.py --negatives data/media/ami_sdm --positives data/heldout_phonemes \
  --openwakeword data/oww/computah.tflite 0.95
```

Everything training uses is pinned (base image, Python packages, repository
commits, each dataset's and model's revision; the harder test's Kokoro
weights are not), but a retrain gives an equivalent model,
not the same bytes: generation, augmentation and training are random.
`training/pipeline.sh` and the files beside it train v2, the microWakeWord
model.

## License

The model and its manifest are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
credited "Computah wake word by xav-ie". Everything it was trained on is
openly licensed: LibriTTS and LibriTTS-R through Piper, LibriSpeech, AMI and
MUSAN's CC BY part (CC BY 4.0), MUSAN's public domain recordings, the MIT
room impulse responses, and 20 Piper voice models, each under its own
license; the four non-commercial voice models (semaine, hfc_female,
hfc_male, l2arctic) are left out. Version 3 in the history is CC BY-NC-SA 4.0,
versions 1 and 2 CC BY 4.0.
