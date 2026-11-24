#!/usr/bin/env python3
# The room impulse responses and background music openWakeWord augments
# with, the same as ../prep.py builds for microWakeWord.
import json, os, subprocess, urllib.request
from pathlib import Path
os.chdir("/work/data")
if not Path("mit_rirs").exists():
    Path("mit_rirs.tmp").mkdir(exist_ok=True)
    repo = "https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses"
    rev = "b824a1ef2821f112fda0b9cb26e4278c62b425bb"
    listing = repo.replace("/datasets/", "/api/datasets/") + f"/revision/{rev}"
    files = [s["rfilename"] for s in json.load(urllib.request.urlopen(listing))["siblings"]]
    for f in [f for f in files if f.startswith("16khz/") and f.endswith(".wav")]:
        urllib.request.urlretrieve(f"{repo}/resolve/{rev}/{f}", f"mit_rirs.tmp/{f.split('/')[-1]}")
    os.rename("mit_rirs.tmp", "mit_rirs")
if not Path("fma_16k").exists():
    if not Path("fma").exists():
        subprocess.run(["unzip", "-q", "dl/fma_xs.zip", "-d", "fma"], check=True)
    Path("fma_16k.tmp").mkdir(exist_ok=True)
    for mp3 in Path("fma").glob("**/*.mp3"):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), "-ac", "1", "-ar", "16000",
                        f"fma_16k.tmp/{mp3.stem}.wav"], check=False)
    os.rename("fma_16k.tmp", "fma_16k")
