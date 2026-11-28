#!/usr/bin/env bash
# Trains Computah v5, the openWakeWord model, on an NVIDIA GPU. Run
# ../download.sh and ./download.sh first. Writes into ./data; the model lands
# in data/oww/computah.tflite and the held-out clips in data/heldout_end and
# data/heldout_phonemes.
#
#   1. Sound-alikes: ../generate.sh (its LibriTTS half; the spelled
#      positives it also makes are not used) and negatives_voices.py.
#   2. "computah" from exact phonemes in the four approved endings, from
#      both LibriTTS generators, and the 20 takes in takes/train, 550
#      copies each. The held-out voices in the approved endings make
#      data/heldout_end.
#   3. v3 and v4's held-out set (data/heldout_phonemes, the long "ah"
#      ending): candidates from the held-out voices, kept when phonescore.py
#      and phonejudge.py hear com-pew-tah with no r.
#   4. General negatives and the validation set as openWakeWord features,
#      from MUSAN (its CC BY and public domain part), LibriSpeech and AMI.
#   5. openWakeWord augments the clips, computes features and trains;
#      to_tflite.py converts the result.
set -euo pipefail
cd "$(dirname "$0")/../.."
podman build -q -t computah-train training
podman build -q -t computah-oww training/openwakeword
exec podman run --rm --device nvidia.com/gpu=all --memory=14g --shm-size=2g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -e PYTHONPATH=/work/data/piper-sample-generator \
  -v "$PWD/data:/work/data" -v "$PWD/training:/work/scripts:ro" -v "$PWD/takes:/work/takes:ro" computah-oww bash -c '
set -euo pipefail
cd /work
S=/work/scripts/openwakeword
O=/work/data/oww/computah
python3 $S/backgrounds.py

# 1. Sound-alikes.
bash scripts/generate.sh
mkdir -p $O/negative_train $O/negative_test $O/positive_train $O/positive_test
find data/v2/neg -name "*.wav" | sort | awk "{n=split(\$0,p,\"/\"); print \$0, (NR%10==0 ? \"test\" : \"train\"), p[n-2]\"_\"p[n-1]\"_\"p[n]}" |
  while read -r src split name; do
    out=$O/negative_$split/$name
    [ -s "$out" ] || ffmpeg -loglevel error -y -i "$src" -ac 1 -ar 16000 "$out"
  done
python3 $S/negatives_voices.py

# 2. Positives.
gen() { # model tag phonemes count
  local out=/work/data/pos/lt/${1}_$2
  [ -f $out.done ] && return
  rm -rf $out
  python3 -m piper_sample_generator "$3" --phoneme-input --model /work/data/piper-sample-generator/models/$1.pt \
    --max-samples $4 --batch-size 10 --slerp-weights 0.2 0.5 0.8 --length-scales 0.75 0.9 1.0 1.15 1.3 \
    --noise-scales 0.5 0.667 0.8 0.9 --noise-scale-ws 0.6 0.8 1.0 --output-dir $out
  python3 $S/positives.py fade $out
  touch $out.done
}
for model in en_US-libritts_r-medium en-us-libritts-high; do
  gen $model TAHc "kəmpjˈuːtɐ" 3000
  gen $model TAHs "kəmpjˈuːɾɐ" 3000
  gen $model TUHs "kəmpjˈuːɾʌ" 2500
  gen $model TAWc "kəmpjˈuːtɔ" 2500
  gen $model heyTAHc "hˈeɪ kəmpjˈuːtɐ" 1500
  gen $model heyTUHs "hˈeɪ kəmpjˈuːɾʌ" 1500
done
i=0
for f in $(find /work/data/pos/lt -name "*.wav" | sort); do
  i=$((i + 1)); tag=$(basename $(dirname $f)); split=train; [ $((i % 10)) -eq 0 ] && split=test
  [ -e $O/positive_$split/p6_${tag}_$(basename $f) ] || ln $f $O/positive_$split/p6_${tag}_$(basename $f)
done
python3 $S/takes.py /work/takes/train /work/data/takes_trimmed $O/positive_train 550
[ -f /work/data/heldout_end.done ] || { python3 $S/positives.py heldout_end /work/data/heldout_end; touch /work/data/heldout_end.done; }

# 3. The held-out set of v3 and v4.
[ -f /work/data/cand/heldout.done ] || { python3 $S/positives.py heldout /work/data/cand/heldout; touch /work/data/cand/heldout.done; }
python3 $S/phonescore.py /work/data/cand > /work/data/cand/scores.tsv
cd $S && python3 select.py /work/data/cand/scores.tsv && cd /work

# 4. General negatives and validation set.
python3 $S/musan_cc_by.py
N=/work/data/oww/neg; R=/work/data/oww/src
mkdir -p $N
[ -s $N/musan.npy ] || python3 $S/neg_features.py windows $N/musan.npy $R/musan_by
[ -s $N/librispeech.npy ] || python3 $S/neg_features.py windows $N/librispeech.npy $R/train-clean-100 $R/train-clean-360
[ -s $N/ami.npy ] || python3 $S/neg_features.py windows $N/ami.npy $R/ami_train/*.parquet
[ -s $N/validation.npy ] || python3 $S/neg_features.py stream $N/validation.npy $R/dev-clean $R/dev-other

# 5. Features, training, conversion.
cd /work/data/oww
python3 -c "import openwakeword.utils as u; u.download_models()"
python3 -m openwakeword.train --training_config $S/computah.yml --augment_clips --overwrite
python3 -m openwakeword.train --training_config $S/computah.yml --train_model || true  # exits 1 without onnx_tf, after writing the ONNX model
python3 $S/to_tflite.py computah.onnx computah.tflite
'
