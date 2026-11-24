#!/usr/bin/env python3
# Judges phonemizer output (path<TAB>phonemes lines on stdin or a file) as
# good "computah": PEW (p j uː) then t/ɾ/d and an open, r-less final vowel.
import re, sys
VOWEL_END = {"ʌ", "ə", "ɑ", "ɑː", "a", "aː", "ɐ"}
def norm(ph):
    t = [re.sub(r"\d", "", x) for x in ph.split()]
    return [{"th": "t", "ph": "p", "kh": "k", "y": "uː", "u": "uː"}.get(x, x) for x in t if x]
def judge(ph):
    t = norm(ph)
    s = " ".join(t)
    if any("ɚ" in x or "ɹ" in x or "ɝ" in x for x in t[-3:]): return "bad", "r at end"
    if not re.search(r"p (j )?uː", s): return "bad", "no PEW"
    m = re.search(r"p (j )?uː (t|ɾ|d) (\S+)$", s)
    if not m or m.group(3) not in VOWEL_END: return "bad", "ending"
    return "good", ""
if __name__ == "__main__":
    for line in (open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin):
        p, ph = line.rstrip("\n").split("\t") if "\t" in line else (line.strip(), "")
        v, why = judge(ph); print(f"{v}\t{why}\t{p}\t{ph}")
