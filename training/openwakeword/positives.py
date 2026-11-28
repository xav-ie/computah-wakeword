#!/usr/bin/env python3
# "computah" clips from exact phonemes, not spellings: spelled "computah"
# came out of most voices as "computer".
#   positives.py heldout_end OUT_DIR  the 5 held-out voices in the approved endings, 240 clips
#   positives.py heldout OUT_DIR      300 per held-out voice in v3 and v4's "ah" ending,
#                                     screened afterwards by phonescore.py and phonejudge.py
#   positives.py voices OUT_ROOT N    N per training Piper model, into OUT_ROOT/<model>/ (v4)
#   positives.py fade DIR...          fade and pad clips another generator made
import random, sys, wave
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly

# The endings a listener with the target accent approved: "tah", "dah"
# (soft t), "duh" and "taw". Not the long "ah" below (heard as "ta-da") nor
# a schwa (heard as "computer").
ENDINGS = {"tah": "kəmpjˈuːtɐ", "dah": "kəmpjˈuːɾɐ", "duh": "kəmpjˈuːɾʌ", "taw": "kəmpjˈuːtɔ"}

# v3 and v4's positives, now only the old held-out set. Weighted toward the stress patterns whose open "ah" survives most voices
# (phoneme scorer pass rates: com-pew-TAH 26%, com-PEW-tahh 14%, -tˌɑː 8%).
BARE = ["kəmpjˌuːtˈɑː"] * 5 + ["kəmpjˈuːtɑːh"] * 3 + ["kəmpjˈuːtˌɑː"] * 2
HEY = ["hˈeɪ " + b for b in BARE]

def finish(a, rate):
    # Fade the last 30 ms (some voices stop mid-sound and click), add 200 ms
    # of silence, peak at 0.9, 16 kHz int16.
    a = np.asarray(a, dtype=np.float32).copy()
    f = int(0.03 * rate); a[-f:] *= np.linspace(1, 0, f)
    a = np.concatenate([a / max(np.abs(a).max(), 1e-6) * 0.9, np.zeros(int(0.2 * rate), np.float32)])
    a = resample_poly(a, 16000, rate) if rate != 16000 else a
    return (np.clip(a, -1, 1) * 32767).astype(np.int16)

def write(path, pcm):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())

def say(voice, phonemes, cfg):
    # Not voice.synthesize("[[ ]]"): Piper 1.3 adds an empty sentence after a
    # phoneme block, which comes out as a burst of noise.
    return finish(voice.phoneme_ids_to_audio(voice.phonemes_to_ids(list(phonemes)), cfg), voice.config.sample_rate)

def train_voice(args):
    from piper import PiperVoice, SynthesisConfig
    model, root, count = args
    voice = PiperVoice.load(str(model)); speakers = voice.config.num_speakers or 1
    rng = random.Random("v5" + model.stem)
    (root / model.stem).mkdir(parents=True, exist_ok=True)
    for i in range(count):
        out = root / model.stem / f"{i}.wav"
        cfg = SynthesisConfig(speaker_id=i % speakers if speakers > 1 else None, length_scale=rng.uniform(0.75, 1.4),
                              noise_scale=rng.uniform(0.5, 0.9), noise_w_scale=rng.uniform(0.6, 1.0))
        write(out, say(voice, rng.choice(HEY) if rng.random() < 0.3 else rng.choice(BARE), cfg))
    return model.stem

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "voices":
        root, count = Path(sys.argv[2]), int(sys.argv[3])
        with Pool(12) as pool:
            for s in pool.imap_unordered(train_voice, [(m, root, count) for m in sorted(Path("/work/data/train_voices").glob("*.onnx"))]):
                print(s, flush=True)
    elif mode == "heldout_end":
        from piper import PiperVoice, SynthesisConfig
        out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
        for model in sorted(Path("/work/data/heldout_voices").glob("*.onnx")):
            voice = PiperVoice.load(str(model))
            for end, ph in ENDINGS.items():
                for tag, text in (("computah", ph), ("hey_computah", "hˈeɪ " + ph)):
                    for speed in (0.85, 1.0, 1.2):
                        for noise in (0.5, 0.8):
                            write(out / f"{model.stem}_{tag}_{end}_{speed}_{noise}.wav",
                                  say(voice, text, SynthesisConfig(length_scale=speed, noise_scale=noise)))
        print("heldout_end", len(list(out.glob("*.wav"))))
    elif mode == "heldout":
        from piper import PiperVoice, SynthesisConfig
        out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
        for model in sorted(Path("/work/data/heldout_voices").glob("*.onnx")):
            voice = PiperVoice.load(str(model))
            rng = random.Random("heldout" + model.stem)
            for tag, phrases in (("computah", BARE), ("hey_computah", HEY)):
                for i in range(150):
                    speed, noise = rng.choice((0.85, 1.0, 1.2)), rng.choice((0.5, 0.667, 0.8))
                    write(out / f"{model.stem}_{tag}_{i}_{speed}_{noise}.wav",
                          say(voice, rng.choice(phrases), SynthesisConfig(length_scale=speed, noise_scale=noise)))
        print("heldout", len(list(out.glob("*.wav"))))
    elif mode == "fade":
        for d in sys.argv[2:]:
            for p in Path(d).glob("*.wav"):
                with wave.open(str(p)) as w:
                    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768; r = w.getframerate()
                write(p, finish(a, r))
        print("faded")
