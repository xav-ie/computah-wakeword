#!/usr/bin/env python3
# Transcribes clips into phonemes with a wav2vec2 phoneme recognizer
# (facebook/wav2vec2-lv-60-espeak-cv-ft) and prints path<TAB>phonemes per
# clip. Screens synthetic "computah" clips (phonejudge.py judges them).
import sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
NAME, REV = "facebook/wav2vec2-lv-60-espeak-cv-ft", "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = Wav2Vec2Processor.from_pretrained(NAME, revision=REV)
model = Wav2Vec2ForCTC.from_pretrained(NAME, revision=REV).eval().to(dev)
if dev == "cuda":
    model = model.half()
files = [str(p) for a in sys.argv[1:] for p in (sorted(Path(a).rglob("*.wav")) if Path(a).is_dir() else [Path(a)])]
files.sort(key=lambda f: Path(f).stat().st_size)  # similar lengths per batch, little padding

def load(f):
    a, r = sf.read(f, dtype="float32")
    assert r == 16000, f
    return f, (a - a.mean()) / (a.std() + 1e-7)

def collate(items):
    n = max(len(a) for _, a in items)
    mask = torch.tensor([[1] * len(a) + [0] * (n - len(a)) for _, a in items])
    return [f for f, _ in items], torch.from_numpy(np.stack([np.pad(a, (0, n - len(a))) for _, a in items])), mask

for names, x, mask in DataLoader(list(range(len(files))), batch_size=int(__import__("os").environ.get("BATCH", "32")), num_workers=8,
                           collate_fn=lambda idx: collate([load(files[i]) for i in idx])):
    with torch.no_grad():
        ids = model(x.to(dev).to(next(model.parameters()).dtype), attention_mask=mask.to(dev)).logits.argmax(-1).cpu()
    for f, t in zip(names, proc.batch_decode(ids)):
        print(f"{f}\t{t}", flush=True)
