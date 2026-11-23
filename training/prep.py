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
        "RIR": 0.5,
    },
    impulse_paths=["mit_rirs"],
    background_paths=["fma_16k"],
    background_min_snr_db=-5,
    background_max_snr_db=10,
    min_jitter_s=0.195,
    max_jitter_s=0.205,
)


def features(src, out_root):
    if Path(out_root, "done").exists():
        return
    clips = Clips(
        input_directory=src,
        file_pattern="**/*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )
    for split, split_name, repetition, slide in [
        ("training", "train", 2, 10),
        ("validation", "validation", 1, 10),
        ("testing", "test", 1, 1),
    ]:
        out_dir = Path(out_root, split)
        out_dir.mkdir(parents=True, exist_ok=True)
        spectrograms = SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=slide, step_ms=10)
        RaggedMmap.from_generator(
            out_dir=str(out_dir / "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(split=split_name, repeat=repetition),
            batch_size=100,
            verbose=True,
        )
    Path(out_root, "done").touch()


features("pos", "features_pos")
features("neg", "features_neg")
print("prep done")
