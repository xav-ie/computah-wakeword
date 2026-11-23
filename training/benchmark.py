#!/usr/bin/env python3
"""Scores wake word models the way a satellite runs them.

False wakes: every negative file played as one continuous stream, counting
wakes per hour with a 2 s refractory period. Catches: each positive clip on
its own, padded with a second of silence either side, counting the clips it
wakes on.

    pip install pymicro-wakeword pyopen-wakeword   # and ffmpeg on PATH
    training/benchmark.py --negatives data/librispeech_test_16k \\
        --positives data/heldout_pos \\
        --micro model/computah.json --cutoffs 0.5 0.6 0.7 \\
        --openwakeword computer_v2.tflite 0.7
"""
import argparse
import subprocess
from pathlib import Path

from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures

CHUNK = 640  # 20 ms of 16 kHz 16 bit mono
SILENCE = b"\0" * 32000


def pcm(path: Path) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True,
        check=True,
    ).stdout


class Micro:
    def __init__(self, config: str, cutoff: float):
        self.model = MicroWakeWord.from_config(config)
        self.model.probability_cutoff = cutoff
        self.features = MicroWakeWordFeatures()

    def feed(self, chunk: bytes) -> bool:
        return any(self.model.process_streaming(f) for f in self.features.process_streaming(chunk))


class OpenWW:
    def __init__(self, tflite: str, cutoff: float):
        from pyopen_wakeword import OpenWakeWord, OpenWakeWordFeatures

        self.model = OpenWakeWord.from_model(model_path=tflite)
        self.features = OpenWakeWordFeatures.from_builtin()
        self.cutoff = cutoff

    def feed(self, chunk: bytes) -> bool:
        return any(
            p > self.cutoff for e in self.features.process_streaming(chunk) for p in self.model.process_streaming(e)
        )


def wakes(model, audio: bytes) -> int:
    count, last, t = 0, -9.0, 0.0
    for i in range(0, len(audio) - CHUNK + 1, CHUNK):
        t = i / 32000
        if model.feed(audio[i : i + CHUNK]) and t - last > 2:
            count, last = count + 1, t
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negatives", required=True, help="folder of speech or noise without the wake word")
    parser.add_argument("--positives", required=True, help="folder of wake word clips")
    parser.add_argument("--micro", action="append", default=[], help="microWakeWord manifest (.json)")
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.94, 0.97])
    parser.add_argument("--openwakeword", nargs=2, action="append", default=[], metavar=("TFLITE", "CUTOFF"))
    args = parser.parse_args()

    negatives = [pcm(f) for f in sorted(Path(args.negatives).rglob("*")) if f.suffix in (".wav", ".flac")]
    positives = [SILENCE + pcm(f) + SILENCE for f in sorted(Path(args.positives).rglob("*.wav"))]
    hours = sum(len(a) for a in negatives) / 32000 / 3600
    print(f"negatives: {hours:.2f} h; positives: {len(positives)} clips")

    runs = [(f"{config} @{c}", lambda config=config, c=c: Micro(config, c)) for config in args.micro for c in args.cutoffs]
    runs += [(f"{tflite} @{c}", lambda tflite=tflite, c=c: OpenWW(tflite, float(c))) for tflite, c in args.openwakeword]
    for name, make in runs:
        model = make()
        false = sum(wakes(model, a) for a in negatives)
        caught = sum(wakes(make(), a) > 0 for a in positives)
        print(f"{name}: {false / hours:.2f} false wakes/h ({false}), caught {caught}/{len(positives)}", flush=True)


if __name__ == "__main__":
    main()
