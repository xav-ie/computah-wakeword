#!/usr/bin/env python3
# Keeps the candidates phonejudge.py passes (heard as com-pew-tah, no r):
# training positives into openWakeWord's clip folders, every 10th to test, at
# most 600 per Piper voice model and 2000 per LibriTTS folder; and up to 12
# bare and 12 "hey" clips per held-out voice into heldout_phonemes.
#   select.py SCORES.tsv   (phonescore.py's output over data/cand)
import shutil, sys
from collections import defaultdict
from pathlib import Path
from phonejudge import judge

D = Path("/work/data")
good = defaultdict(list)
for line in open(sys.argv[1]):
    path, phonemes = line.rstrip("\n").split("\t")
    if judge(phonemes)[0] == "good":
        path = Path(path); good[path.parent].append(path)

oww = D / "oww/computah"
n = 0
for folder, files in sorted(good.items()):
    if folder.name == "heldout":
        continue
    piper = folder.parent.name == "pv"
    for f in sorted(files, key=lambda f: int(f.stem))[: 600 if piper else 2000]:
        tag = ("p5v_" if piper else "p5l_") + folder.name
        shutil.copy(f, oww / f"positive_{'test' if n % 10 == 0 else 'train'}" / f"{tag}_{f.stem}.wav")
        n += 1
print("training positives kept:", n)

out = D / "heldout_phonemes"
shutil.rmtree(out, ignore_errors=True); out.mkdir()
per = defaultdict(int)
for f in sorted(good.get(D / "cand/heldout", [])):
    voice = f.name.split("_computah")[0].removesuffix("_hey")
    kind = "hey" if "_hey_computah" in f.name else "bare"
    if per[voice, kind] < 12:
        shutil.copy(f, out / f.name); per[voice, kind] += 1
print("held-out kept:", sum(per.values()), dict(per))
