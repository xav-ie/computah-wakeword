#!/usr/bin/env python3
import json
import os
import subprocess
import urllib.request
from pathlib import Path

from mmap_ninja.ragged import RaggedMmap
from tqdm import tqdm

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

os.chdir("/work/data")

if not Path("mit_rirs").exists():
    Path("mit_rirs.tmp").mkdir(exist_ok=True)
    # The revision trained with: 270 room impulse responses at 16 kHz.
    repo = "https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses"
    rev = "b824a1ef2821f112fda0b9cb26e4278c62b425bb"
    listing = repo.replace("/datasets/", "/api/datasets/") + f"/revision/{rev}"
    files = [s["rfilename"] for s in json.load(urllib.request.urlopen(listing))["siblings"]]
    for f in tqdm([f for f in files if f.startswith("16khz/") and f.endswith(".wav")]):
        urllib.request.urlretrieve(f"{repo}/resolve/{rev}/{f}", f"mit_rirs.tmp/{f.split('/')[-1]}")
    os.rename("mit_rirs.tmp", "mit_rirs")

if not Path("fma_16k").exists():
    if not Path("fma").exists():
        subprocess.run(["unzip", "-q", "dl/fma_xs.zip", "-d", "fma"], check=True)
    Path("fma_16k.tmp").mkdir(exist_ok=True)
    for mp3 in tqdm(list(Path("fma").glob("**/*.mp3"))):
        out = f"fma_16k.tmp/{mp3.stem}.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), "-ac", "1", "-ar", "16000", out], check=False)
    os.rename("fma_16k.tmp", "fma_16k")

Path("negative_datasets").mkdir(exist_ok=True)
for name in ["dinner_party", "dinner_party_eval", "no_speech", "speech"]:
    if not Path(f"negative_datasets/{name}").exists():
        subprocess.run(["unzip", "-q", f"dl/{name}.zip", "-d", "negative_datasets"], check=True)

# The ambient set train.py validates and tests false wakes on: the first of
# dinner_party_eval's four validation recordings (about 2 hours, to keep
# validation quick) and all of its testing recordings.
if not Path("eval_small/validation_ambient").exists():
    Path("eval_small").mkdir(exist_ok=True)
    if not Path("eval_small/testing_ambient").exists():
        os.symlink("../negative_datasets/dinner_party_eval/testing_ambient", "eval_small/testing_ambient")
    full = RaggedMmap("negative_datasets/dinner_party_eval/validation_ambient/chime6_dev_eval_mmap")
    Path("eval_small/validation_ambient.tmp").mkdir(parents=True, exist_ok=True)
    RaggedMmap.from_lists("eval_small/validation_ambient.tmp/x_mmap", [full[0]])
    os.rename("eval_small/validation_ambient.tmp", "eval_small/validation_ambient")

augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.1,
        "TanhDistortion": 0.1,
        "PitchShift": 0.1,
        "BandStopFilter": 0.1,
        "AddColorNoise": 0.1,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "RIR": 0.7,
    },
    impulse_paths=["mit_rirs"],
    # Music only. A trial run that also mixed speech under "computah" at -5
    # to 10 dB learned that speech itself means "computah" (19 false wakes an
    # hour on LibriSpeech); speech goes in as its own negative set below.
    background_paths=["fma_16k"],
    background_min_snr_db=-5,
    background_max_snr_db=10,
    min_jitter_s=0.195,
    max_jitter_s=0.205,
)


def features(src, out_root):
    clips = Clips(
        input_directory=src,
        file_pattern="**/*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )
    # slide is how many shifted copies of each clip are kept. Validation
    # keeps 4, not 10: the trainer holds the whole validation set in memory
    # (and on the GPU) at once, and 25k clips at 10 copies did not fit.
    for split, split_name, repetition, slide in [
        ("training", "train", 2, 10),
        ("validation", "validation", 1, 4),
        ("testing", "test", 1, 1),
    ]:
        if Path(out_root, f"{split}.done").exists():
            continue
        out_dir = Path(out_root, split)
        out_dir.mkdir(parents=True, exist_ok=True)
        spectrograms = SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=slide, step_ms=10)
        RaggedMmap.from_generator(
            out_dir=str(out_dir / "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(split=split_name, repeat=repetition),
            batch_size=100,
            verbose=True,
        )
        Path(out_root, f"{split}.done").touch()


features("v2/pos", "v2/features_pos")
features("v2/neg", "v2/features_neg")

# LibriSpeech dev-clean as a negative set of plain speech, in 3 s pieces and
# unaugmented like microWakeWord's own speech sets. Its speakers are apart
# from test-clean, which the benchmark measures false wakes on.
if not Path("v2/features_librispeech/training.done").exists():
    out_dir = Path("v2/features_librispeech/training")
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = Clips(
        input_directory="librispeech_dev_16k",
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )
    spectrograms = SpectrogramGeneration(clips=clips, step_ms=10, split_spectrogram_duration_s=3.0)
    RaggedMmap.from_generator(
        out_dir=str(out_dir / "librispeech_dev_mmap"),
        sample_generator=spectrograms.spectrogram_generator(split="train"),
        batch_size=100,
        verbose=True,
    )
    Path("v2/features_librispeech/training.done").touch()
print("prep done")
