#!/usr/bin/env python3
# openWakeWord features from openly licensed audio. "windows" writes general
# negatives in the shape of openWakeWord's own ACAV100M features: 16 embedding
# frames (2 s of audio) every 1.28 s, float16, digital silence skipped.
# "stream" writes one continuous (frames, 96) stream, a validation set.
#   neg_features.py windows|stream OUT.npy SOURCE...
# SOURCE is a folder of audio or AMI parquet shards.
import io, subprocess, sys
from pathlib import Path
import numpy as np
from openwakeword.utils import AudioFeatures

def decode(path):
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                          capture_output=True, check=True).stdout

def audio(sources):
    """Yields int16 arrays, one per recording."""
    for s in sources:
        s = Path(s)
        if s.suffix == ".parquet":
            import pyarrow.parquet as pq, soundfile as sf
            from scipy.signal import resample_poly
            for row in pq.read_table(s, columns=["audio"]).to_pylist():
                a, r = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
                a = a.mean(axis=1)
                a = a if r == 16000 else resample_poly(a, 16000, r)
                yield (np.clip(a, -1, 1) * 32767).astype(np.int16)
        else:
            for f in sorted(p for p in s.rglob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3")):
                try:
                    yield np.frombuffer(decode(f), np.int16)
                except subprocess.CalledProcessError:
                    pass

def main():
    import openwakeword.utils as u; u.download_models()
    mode, out, sources = sys.argv[1], sys.argv[2], sys.argv[3:]
    F = AudioFeatures(ncpu=14)
    if mode == "stream":
        frames = [F.embed_clips(a[None, :len(a) // 1280 * 1280], batch_size=1)[0] for a in audio(sources) if len(a) > 32000]
        np.save(out, np.concatenate(frames).astype(np.float32))
        print(out, sum(len(f) for f in frames), "frames"); return
    WIN, HOP, BATCH = 32000, 20480, 512
    buf, chunks, n = [], [], 0
    def flush():
        nonlocal buf
        x = np.stack(buf); buf = []
        chunks.append(F.embed_clips(x, batch_size=BATCH, ncpu=14).astype(np.float16))
    for a in audio(sources):
        for i in range(0, len(a) - WIN + 1, HOP):
            w = a[i:i + WIN]
            if np.abs(w).max() < 100:  # digital silence
                continue
            buf.append(w); n += 1
            if len(buf) == BATCH * 8:
                flush(); print(out, n, "windows", f"{n * 1.28 / 3600:.1f} h", flush=True)
    if buf:
        flush()
    x = np.concatenate(chunks)
    np.save(out, x)
    print(out, x.shape, f"{len(x) * 1.28 / 3600:.1f} h")

if __name__ == "__main__":
    main()
