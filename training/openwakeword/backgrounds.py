#!/usr/bin/env python3
# The room impulse responses openWakeWord augments with, the same as
# ../prep.py fetches for microWakeWord.
import json, os, urllib.request
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
