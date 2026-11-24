#!/usr/bin/env python3
# Rebuilds continuous 16 kHz recordings from the parquet shards: AMI meetings
# with each utterance at its begin_time (gaps silent), People's Speech
# sources with their numbered pieces joined in order.
import io, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, soundfile as sf
from scipy.signal import resample_poly

M = Path(sys.argv[1])

def load(b):
    a, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    return a if sr == 16000 else resample_poly(a, 16000, sr).astype("float32")

meetings = defaultdict(list)
for f in sorted(M.glob("ami_sdm_test_*.parquet")):
    for row in pq.read_table(f, columns=["meeting_id", "audio", "begin_time"]).to_pylist():
        meetings[row["meeting_id"]].append((row["begin_time"], load(row["audio"]["bytes"])))
out = M / "ami_sdm"; out.mkdir(exist_ok=True)
for mid, segs in meetings.items():
    end = max(int(b * 16000) + len(a) for b, a in segs)
    buf = np.zeros(end, dtype="float32")
    for b, a in segs:
        s = int(b * 16000); buf[s:s + len(a)] = a
    sf.write(out / f"{mid}.wav", buf, 16000, subtype="PCM_16")
    print(mid, round(end / 16000 / 60, 1), "min", flush=True)

sources = defaultdict(list)
for f in sorted(M.glob("ps_dirty_test_*.parquet")):
    for row in pq.read_table(f, columns=["id", "audio"]).to_pylist():
        m = re.match(r"(.*)_(\d+)$", row["audio"]["path"].rsplit(".", 1)[0])
        key, idx = (m.group(1), int(m.group(2))) if m else (row["id"], 0)
        sources[key].append((idx, load(row["audio"]["bytes"])))
out = M / "peoples_dirty"; out.mkdir(exist_ok=True)
total = 0
for i, (key, parts) in enumerate(sources.items()):
    a = np.concatenate([p for _, p in sorted(parts, key=lambda x: x[0])])
    total += len(a)
    sf.write(out / f"{i:04d}.wav", a, 16000, subtype="PCM_16")
print("peoples_dirty", len(sources), "sources", round(total / 16000 / 3600, 2), "h")
