#!/usr/bin/env python3
# "computah" and "hey computah" candidates, 40 of each, from four Kokoro
# voices training never hears (two American, two British), from exact
# phonemes as the Piper clips are, faded and padded the same way. Screened by
# ../hardtest.sh before use.
#   heldout.py OUT
import random, sys
from pathlib import Path
import numpy as np, soundfile as sf
from kokoro import KPipeline
from scipy.signal import resample_poly

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = random.Random(4)
pipes = {"a": KPipeline(lang_code="a"), "b": KPipeline(lang_code="b")}
VOICES = ["af_nova", "am_onyx", "bf_lily", "bm_daniel"]
# Stress on TAH, an h-ending variant and stress on PEW, as for Piper.
PH = {"a": [("kəmpjˌutˈɑ", 5), ("kəmpjˌutˈɑh", 3), ("kəmpjˈutɑh", 2)],
      "b": [("kəmpjˌuːtˈɑː", 5), ("kəmpjˌuːtˈɑːh", 3), ("kəmpjˈuːtɑːh", 2)]}

def finish(a, rate):
    # Fade the last 30 ms, add 200 ms of silence, peak at 0.9, 16 kHz.
    a = np.asarray(a, dtype=np.float32).copy()
    f = int(0.03 * rate); a[-f:] *= np.linspace(1, 0, f)
    a = np.concatenate([a / max(np.abs(a).max(), 1e-6) * 0.9, np.zeros(int(0.2 * rate), np.float32)])
    return resample_poly(a, 16000, rate)

for name in VOICES:
    lang = name[0]
    for i in range(80):
        ph = rng.choices([p for p, _ in PH[lang]], [w for _, w in PH[lang]])[0]
        word = f"[computah](/{ph}/)"
        text = f"Hey {word}." if i % 2 else f"{word}."
        a = np.concatenate([r.audio.numpy() for r in pipes[lang](text, voice=name, speed=rng.uniform(0.85, 1.2))])
        sf.write(out / f"{name}_{'hey_' if i % 2 else ''}computah_{i}.wav", finish(a, 24000), 16000, subtype="PCM_16")
print("kokoro held-out candidates:", len(list(out.glob("*.wav"))))
