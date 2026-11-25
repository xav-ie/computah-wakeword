#!/usr/bin/env python3
# The harder catch test: each held-out "computah" clip through a real room
# (a recorded impulse response training never uses) and laid into far-field
# audio (AMI meetings or People's Speech) at 10, 5 and 0 dB, word at 1.5 s of
# a 4 s clip. Deterministic (seeded).
#   hardtest.py OUT RIR_DIR BACKGROUND_DIR... -- POSITIVE_DIR...
# openwakeword/hardtest.sh builds it from the benchmark sets.
import random, sys, wave
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

args = sys.argv[1:]; sep = args.index("--")
out, rir_dir, bg_dirs, pos_dirs = Path(args[0]), Path(args[1]), args[2:sep], args[sep + 1:]
rng = random.Random(11)

def mono16k(path):
    a, r = sf.read(str(path), dtype="float32", always_2d=True)
    a = a[:, 0]
    return a if r == 16000 else resample_poly(a, 16000, r).astype("float32")

rirs = [mono16k(p) for p in sorted(rir_dir.rglob("*.wav")) if "noise" not in p.name]  # responses only
bgs = [(p, sf.info(str(p)).frames) for d in bg_dirs for p in sorted(Path(d).glob("*.wav"))]
out.mkdir(parents=True, exist_ok=True)
n = 0
for d in pos_dirs:
    for p in sorted(Path(d).glob("*.wav")):
        word = mono16k(p)
        for snr in (10, 5, 0):
            rir = rirs[rng.randrange(len(rirs))]
            rir = rir[np.argmax(np.abs(rir)):]  # direct path first
            wet = fftconvolve(word, rir)[: len(word) + 8000]
            wet /= np.abs(wet).max() + 1e-9
            for _ in range(20):  # a background stretch that is not silence
                bp, frames = bgs[rng.randrange(len(bgs))]
                start = rng.randrange(0, max(1, frames - 64000))
                bg, _ = sf.read(str(bp), start=start, frames=64000, dtype="float32")
                if len(bg) == 64000 and np.sqrt((bg ** 2).mean()) > 1e-3:
                    break
            active = wet[np.abs(wet) > 0.05 * np.abs(wet).max()]
            gain = np.sqrt((bg ** 2).mean()) * 10 ** (snr / 20) / (np.sqrt((active ** 2).mean()) + 1e-9)
            mix = bg.copy()
            seg = wet[: 64000 - 24000] * gain
            mix[24000:24000 + len(seg)] += seg
            mix /= max(1.0, np.abs(mix).max() / 0.95)
            sf.write(out / f"{snr:02d}dB_{p.stem}.wav", mix, 16000, subtype="PCM_16")
            n += 1
print("hard test clips:", n)
