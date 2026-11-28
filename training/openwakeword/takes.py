#!/usr/bin/env python3
# Xavier's own "computah" takes (takes/train) as training positives: each
# trimmed to the word (50 ms before the first loud 10 ms frame to 100 ms after
# the last; loud is above max(400, 8% of the loudest frame)), then linked
# COPIES times into the positive clips, so openWakeWord's augmentation puts
# each copy in a different room and background. One copy each is too few:
# 20 takes against 33,000 negatives trained a model that never wakes.
#   takes.py TAKES_DIR TRIM_DIR POSITIVE_DIR COPIES
import math, os, struct, sys, wave
from pathlib import Path

takes, trim, pos, copies = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
trim.mkdir(parents=True, exist_ok=True)
for src in sorted(takes.glob("*.wav")):
    with wave.open(str(src)) as w:
        d = w.readframes(w.getnframes())
    s = struct.unpack(f"<{len(d) // 2}h", d); fr = 160
    e = [math.sqrt(sum(x * x for x in s[i:i + fr]) / fr) for i in range(0, len(s) - fr, fr)]
    thr = max(400, 0.08 * max(e)); loud = [i for i, v in enumerate(e) if v > thr]
    a, b = max(0, (loud[0] - 5) * fr), min(len(s), (loud[-1] + 10) * fr)
    out = trim / src.name
    with wave.open(str(out), "wb") as o:
        o.setnchannels(1); o.setsampwidth(2); o.setframerate(16000); o.writeframes(struct.pack(f"<{b - a}h", *s[a:b]))
    for k in range(copies):
        link = pos / f"real_{src.stem}_{k:04d}.wav"
        if not link.exists():
            os.link(out, link)
print("takes:", len(list(trim.glob("*.wav"))), "x", copies)
