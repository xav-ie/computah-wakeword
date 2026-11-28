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
on "Computah": computer said with the r dropped, com-pew-tah, the final
"tah" as in "lobstah". It runs anywhere openWakeWord models do: Home Assistant's
linux-voice-assistant, Wyoming satellites and openWakeWord itself.

| File | What |
|---|---|
| `model/computah.tflite` | The model (208 KB); openWakeWord's built-in melspectrogram and embedding models feed it |
| `model/computah.json` | Its manifest: openWakeWord, cutoff 0.95 |

This is version 5, the first tested on a real voice. v3 and v4 learned a
long "ah" at the end that a speaker of the target accent hears as "ta-da",
and v4 caught 1 of 16 of that speaker's real takes. v5 learns the endings
the speaker approved and 20 of their takes, and catches 14 of the 16. It
woke falsely 0 times in 24 hours of speech, where v4 woke 3 times, and it
does not wake on "computer". It is CC BY 4.0, free for commercial use. Earlier
versions are in the history.

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
Speech dirty, 9.8 h). Catches come from:

- **Real voice:** 16 takes of "computah" and "hey computah" by one speaker
  of the target accent on a desk microphone, never trained on
  (`takes/heldout`), and 23 takes of "computer", "compute", other words,
  speech and silence that must not wake it (`takes/not`).
- **Approved endings:** 240 clips by five Piper voices training never uses,
  in the four endings that speaker approved ("tah", "dah", "duh", "taw").
- **In noise:** those 240 through real rooms and into meeting or street
  audio at 10, 5 and 0 dB (720 clips).
- **Old ending:** v3 and v4's 106 clips in the long "ah" ending, and their
  564 in noise, for comparison with earlier versions.

`results/benchmark.txt`:

| Model | Cutoff | LibriSpeech | AMI meetings | People's Speech | Real voice | Wakes on others | Approved endings | In noise |
|---|---|---|---|---|---|---|---|---|
| **Computah v5** | **0.95** (shipped) | **0.00/h** | **0.00/h** | **0.00/h** | **14/16** | **0/23** | **171/240** | **46/720** |
| Computah v5 | 0.9 | 0.00/h | 0.22/h | 0.00/h | 15/16 | 1/23 | 193/240 | 74/720 |
| Computah v5 | 0.5 | 0.00/h | 0.90/h | 0.00/h | 15/16 | 6/23 | 212/240 | 120/720 |
| Computah v4 | 0.95 | 0.00/h | 0.22/h | 0.10/h | 1/16 | 0/23 | 114/240 | 29/720 |
| Computah v4 | 0.5 | 0.19/h | 1.12/h | 0.31/h | 4/16 | 0/23 | 141/240 | 59/720 |

Lowering the cutoff below 0.95 catches little more of the real voice and
starts waking on that speaker's "computer" and "compute": the four
"computer" takes score 0.50 to 0.90. Keep `probability_cutoff` at 0.95
unless it misses you; raise it to 0.97 for fewer wakes, at a cost in noise.
"computer" said by the three American held-out voices wakes v5 on 0 of 12
clips.

On v3 and v4's old-ending sets v5 catches less (95/106, and 55/564 in noise
where v4 caught 178): it learned not to treat the long "ah" as the wake
word. In noise every version misses most of the approved endings; that is
the next thing to improve.

**Limits:** the real-voice test is one speaker, and 20 of their takes are in
training, so v5 may fit that voice better than others'. The other positives
are synthetic voices, English only. Accents beyond American and British are
not benchmarked.

## How it was trained

The fix that mattered was in the data. Spelled "computah" ("computah",
"computa", "kum pew tah"), as v1 and v2 were trained, came out of most
synthetic voices as "computer"; a listening check found that a fifth of v2's
held-out clips said "computer" too. v3's positives come from phonemes
instead, and every clip is screened before training sees it.

- **Positives:** 21,951 clips of "computah" and "hey computah" from exact
  phonemes in the four endings a speaker of the target accent approved
  (kəmpjˈuːtɐ, kəmpjˈuːɾɐ, kəmpjˈuːɾʌ, kəmpjˈuːtɔ), from both Piper LibriTTS
  generators (904 speakers, blended), each faded and padded so no voice
  ends in a click (`training/openwakeword/pipeline.sh`, `positives.py`).
  Not the long "ah" of v3 and v4 (heard as "ta-da") and not a schwa (heard
  as "computer"). They are not screened: the phoneme recognizer v3 and v4
  screened with did not agree with the listener on these endings.
- **Real takes:** 20 takes by that speaker (`takes/train`), trimmed to the
  word and linked 550 times each, 11,000 clips, so augmentation puts every
  copy in a different room and background (`takes.py`). A model trained on
  the takes alone caught that voice but woke on its "computer"; the synthetic
  clips teach the difference.
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

What did not help, on v4's benchmarks: Kokoro voices among the positives,
a model four times as wide, and 100,000 steps, which made the model sure of
its false wakes.

## Training it again

Needs an NVIDIA GPU with 8 GB, podman with CDI and about 250 GB of disk.

```sh
training/download.sh                # pinned sources, each checked by SHA-256
training/openwakeword/download.sh   # negatives' audio, room responses, 20 voice models
training/openwakeword/pipeline.sh   # sound-alikes, positives, held-out clips, negatives, model
training/farfield.sh                # the AMI and People's Speech benchmark sets
training/openwakeword/hardtest.sh   # the tests in noise
training/benchmark.py --negatives data/media/ami_sdm --positives data/heldout_end \
  --openwakeword data/oww/computah.tflite 0.95
training/benchmark.py --negatives takes/not --positives takes/heldout \
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
openly licensed: the takes in `takes/` (CC BY 4.0, given by their speaker),
LibriTTS and LibriTTS-R through Piper, LibriSpeech, AMI and
MUSAN's CC BY part (CC BY 4.0), MUSAN's public domain recordings, the MIT
room impulse responses, and 20 Piper voice models, each under its own
license; the four non-commercial voice models (semaine, hfc_female,
hfc_male, l2arctic) are left out. Version 3 in the history is CC BY-NC-SA 4.0,
versions 1 and 2 CC BY 4.0.
