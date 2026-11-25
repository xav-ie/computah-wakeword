#!/usr/bin/env python3
# Links the MUSAN recordings that are CC BY or public domain into
# data/oww/src/musan_by. Speech and noise are all public domain; music is
# kept only when its LICENSE entry names an attribution or public domain
# license with no share-alike, no-derivatives or non-commercial term, and
# dropped when it has no entry at all. Royalty Free Music (rfm) is CC BY 3.0
# for the whole folder.
import collections, os, re
from pathlib import Path

root = Path("/work/data/oww/src/musan/musan")
out = Path("/work/data/oww/src/musan_by")
ok, bad = set(), set()
for lic in sorted(root.glob("music/*/LICENSE")):
    for block in re.split(r"\n=+\s*\n", lic.read_text(errors="replace")):
        ids = set(re.findall(r"\b(music-[\w-]+-\d+)\b", block))
        if not re.search(r"CC BY|Attribution|Public Domain|CC0", block, re.I):
            continue
        restricted = re.search(r"BY-SA|BY-ND|BY-NC|Share.?Alike|NonCommercial|NoDeriv|\bNC\b", block, re.I)
        (bad if restricted else ok).update(ids)
kept, dropped = collections.Counter(), collections.Counter()
for w in sorted(root.glob("*/*/*.wav")):
    rel = w.relative_to(root)
    if rel.parts[0] == "music" and rel.parts[1] != "rfm" and (w.stem not in ok or w.stem in bad):
        dropped[rel.parent.as_posix()] += 1
        continue
    (out / rel.parent).mkdir(parents=True, exist_ok=True)
    if not (out / rel).exists():
        os.link(w, out / rel)
    kept[rel.parent.as_posix()] += 1
for k in sorted(set(kept) | set(dropped)):
    print(f"{k}: kept {kept[k]}, dropped {dropped[k]}")
