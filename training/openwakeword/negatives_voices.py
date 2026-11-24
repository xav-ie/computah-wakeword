#!/usr/bin/env python3
# Sound-alikes ("commuter", "Jupiter", "cappuccino", ...) from 24 Piper voice
# models (186 speakers), 600 per model, into openWakeWord's clip folders,
# every 10th to test. Nothing that sounds like "computah": no "computer"
# family and no "compute a"; with them, the model learned to reject
# "computah" itself.
import random, wave
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from piper import PiperVoice, SynthesisConfig

VOICES = Path("/work/data/train_voices")
OUT = Path("/work/data/oww/computah")
NEG = ["commuter", "compute", "computing", "come to", "Jupiter", "amputate", "compadre", "cucumber",
       "complete", "compost", "competitor", "computed", "come here", "potato", "pasta", "commuting",
       "can you", "come on", "compu", "puter", "computational", "computation", "competition",
       "complicated", "copper", "come pick her", "cappuccino", "kombucha", "compound"]

def run(model_path):
    voice = PiperVoice.load(str(model_path))
    speakers = voice.config.num_speakers or 1
    rng = random.Random(model_path.stem)
    for i in range(600):
        out = OUT / f"negative_{'test' if i % 10 == 0 else 'train'}" / f"pv_{model_path.stem}_{i}.wav"
        if out.exists():
            continue
        cfg = SynthesisConfig(speaker_id=i % speakers if speakers > 1 else None, length_scale=rng.uniform(0.75, 1.4),
                              noise_scale=rng.uniform(0.5, 0.9), noise_w_scale=rng.uniform(0.6, 1.0))
        chunks = list(voice.synthesize(rng.choice(NEG), syn_config=cfg))
        audio = np.concatenate([c.audio_float_array for c in chunks])
        rate = chunks[0].sample_rate
        audio = resample_poly(audio, 16000, rate) if rate != 16000 else audio
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return model_path.stem

if __name__ == "__main__":
    for d in ("negative_train", "negative_test"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    with Pool(12) as pool:
        for line in pool.imap_unordered(run, sorted(VOICES.glob("*.onnx"))):
            print(line, flush=True)
