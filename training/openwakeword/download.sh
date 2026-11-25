#!/usr/bin/env bash
# What the openWakeWord model needs on top of ../download.sh, each at a fixed
# revision and checked by hash: the openly licensed audio its general
# negatives and validation set are made from, real room impulse responses for
# the harder test, and the 20 Piper voice models positives and sound-alikes
# come from.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p data/oww/src/ami_train data/train_voices
cd data
get() { # url file sha256
  if [ ! -s "$2" ] || ! echo "$3  $2" | sha256sum -c --quiet - 2>/dev/null; then
    curl -fL --retry 5 -o "$2.part" "$1"
    echo "$3  $2.part" | sha256sum -c --quiet -
    mv -f "$2.part" "$2"
  fi
}
# General negatives: MUSAN (OpenSLR 17, CC BY 4.0; music, speech and noise),
# LibriSpeech train-clean-100 and train-clean-360 (OpenSLR 12, CC BY 4.0)
# and AMI's single distant microphone train split (CC BY 4.0). LibriSpeech
# dev-clean and dev-other are the validation set; OpenSLR 28's real room
# impulse responses (Apache 2.0) feed the harder test. openslr.org serves
# near 1 MB/s per connection, so the large archives take hours.
SLR=https://openslr.elda.org/resources
get $SLR/17/musan.tar.gz oww/src/musan.tar.gz 25533b73eca0b8e73463d4e0dca7109ef56e8dd81775ec2eb3ebd32356391f31
get $SLR/12/train-clean-100.tar.gz oww/src/train-clean-100.tar.gz d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2
get $SLR/12/train-clean-360.tar.gz oww/src/train-clean-360.tar.gz 705fe3e5db46fac9f062625834d3c6cbef61beeed4a6d88378a5cd386a2ff915
get $SLR/12/dev-clean.tar.gz oww/src/dev-clean.tar.gz 76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3
get $SLR/12/dev-other.tar.gz oww/src/dev-other.tar.gz 12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365
get $SLR/28/rirs_noises.zip oww/src/rirs_noises.zip 3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7
AMI=https://huggingface.co/datasets/edinburghcstr/ami/resolve/46f28f2503e2ec48f8867a84eef356c70476beab/sdm
while read -r i sha; do
  get "$AMI/train-000$i-of-00027.parquet" "oww/src/ami_train/train_$i.parquet" "$sha"
done <<'LIST'
00 049d32bfb1b948485eb5f522b8ab7a9dd8709bbc5b9e1740a23738b6d84539a5
01 8f31337d48e8b114d15b1f4fcbb90344181092ab640e564fa74d52e1d34d858f
02 955445b94afeaa8813c38bcd1ad3342d64db82aea0a972b17f83628e78f61e0f
03 f2e1546d6b02f7b4478ea7def00e83651a80e0a2d332c116b202e255015faa12
04 400d129d834bf797975211253a02b3eb6d727d71399563b5e48aa8bcf961b99a
05 29265c0cc123e106ae8c0237eb47608244988922b7bb0f9a277e070fd775ee34
06 908ffc70e60bd30a283141df950f5ce80f4a11931f3b9e3f4b75ca0a7658b831
07 06fed36421d027fb6aa2fa45a6e59d5c147a4c2e369be752da8300010bae562e
08 cb231b55c3ab09c6512d53c2fddf19268f700c12f9b4cd470895aede8c59b63a
09 3eb8a5b267e62060ba710c821a4cb1c84e8454cb54ba199153637f4c85fe1701
10 d621b54553580bfca51a65621c84813109d9c74663f65982a775282f889dd87b
11 b1b47ce200de72f5b213186fe8ca6bc46ee35d39ffd1fa950256526a5af87fd9
12 58aef986f83dd5e6239042cb2d1c22e2bfcf17235d0287fbb28f96c1ac045bd4
13 4a298b2c7851d6e06dcf402eeb11f2dade3f26950dff2f99675aab2345e53f71
14 493de68cb70026fef7a97b2782a5136ee243c60b42c8eec2dfde7ca0c838ca3e
15 a4c459ea964b9e19cb5146cf21756b942241eb06bc189e67c2feaa706de774f1
16 9472e56503a30569e3f1d2f85b11036c4de85363582817822e13b2912733ca06
17 8c2403839f56eae6444d757d2d5f908a6d5a6e4518d5086ea354e37fcb68b0fa
18 2d7be46e2ae9da34eb2ceefae7a2693ff3bb422446c9e3c74aae4175a7cff306
19 d9b05211e963df3c14718b4952d7421debc53ff6afe64c79491bc92b61dac5af
20 db458fca8a7ce8d6c984b4beefd32a75fa8fbfba0248ba1098c1e1a2b23f44ee
21 524981139dff54b62efeea87bb9a0645b1d6a7e75d94c4a73157a7170fbfc104
22 0a2ac150c8f7b982c44dc08af1e43c0c5fe50dacdff7719bd476452493a2da92
23 3e3152da85f9ee7eac3316c49971d03a8707619c412ea1c16ce8cfbb7e7b0576
24 256ab2d13f99c84ea2dd11d3b5bb586b0f7020abaa1ffb769d3f960176c3cf68
25 51cfbe9a9c016ec8c2cb7873971426d8e8603c4246dcbe20c857f61e519920b9
26 c2892fe94063b9fd76750a892d37556f3cdf10b88f7dfbc6074587634e0bbacf
LIST
for f in musan train-clean-100 train-clean-360 dev-clean dev-other; do
  [ -d oww/src/$f ] || { mkdir -p oww/src/$f.tmp; tar xzf oww/src/$f.tar.gz -C oww/src/$f.tmp; mv oww/src/$f.tmp oww/src/$f; }
done
[ -d oww/src/rirs ] || { unzip -q oww/src/rirs_noises.zip -d oww/src/rirs.tmp; mv oww/src/rirs.tmp oww/src/rirs; }

# Piper voice models. semaine, hfc_female, hfc_male and l2arctic are left out:
# their data is non-commercial.
VOICES=https://huggingface.co/rhasspy/piper-voices/resolve/e5349433ed883a32f893a434dc43a6c66b95ef7c/en
while read -r path sha sha_json; do
  name=$(basename "$path")
  get "$VOICES/$path.onnx" "train_voices/$name.onnx" "$sha"
  get "$VOICES/$path.onnx.json" "train_voices/$name.onnx.json" "$sha_json"
done <<'LIST'
en_GB/alba/medium/en_GB-alba-medium 1f9da809b6da1ef4190bb5dd6cce4f4af803274539e9cf9e0693085150bac3d7 ed4b1f015a8aae5eb26d4bdeed7b712c660f5698a0dd954aa6df9c8be60b900b
en_GB/aru/medium/en_GB-aru-medium cbdeeec713222dc17fb8f52212b62b30b8ddbaa521a6781022ed86a4a136e5bc 3575c270d9bc0f8b3fe4e9a0190eef03492a4f35d4dffe5f1cbd0de258b6f3c8
en_GB/cori/medium/en_GB-cori-medium 2e0a55e140de672aa19b182602b407607bd2e6c35fc6020c0b1516053c02e995 7204f5ead17759278419129290e51ebb414f09f7ac6cd0e567532592956bbbac
en_GB/northern_english_male/medium/en_GB-northern_english_male-medium 41199d55776c142c151e018cf77bea80ca5f01b7b834f81cd781df5b751bf2fb 0159d481609238e47e31114496d004b86d7b49d1f95a81adfb6e5ff758d99013
en_GB/southern_english_female/low/en_GB-southern_english_female-low f2f37aed1b3a093476f719d1379ba0c0b1b1cf6f1ef99288e2ebf502971a07c3 a1af43310f8756161506e849a2863d342a98b0853af9e2437581d39e9922d0e5
en_GB/vctk/medium/en_GB-vctk-medium fc3d92d266b1a6c03af2a38215e0eee430ca7fb360f641d7c3dcb17c7670c0f3 8cc5a7678bf893c35cb6fde5d0812fb7e2b63e1eef54db250afd525277471f2d
en_US/arctic/medium/en_US-arctic-medium 483303e294947a3ec2f910ea96093d876e1640f5772e9d89e511d6c82c667286 db2ca1a55db01cdd3ce28ae63037ac525133e9e00ca557430dec572643235efe
en_US/bryce/medium/en_US-bryce-medium dc9caa6c313199ffb5ac698b6e542fa6cba388aeaf2731e25262e33b9810aef1 7ceb1bc4af6d4e41b6d1edbb86c67e91e01eaa71f66db4cd0ae92ac704d415be
en_US/danny/low/en_US-danny-low 56a9ae9499e961514f060aac2866cb323a21e5989592fc4f208e56fdc323ab64 191dea1ba9863199d8ca2e9048f83b219219fb450f62767ee02f1bc4568ce4f4
en_US/hifi_tts_92/medium/en_US-hifi_tts_92-medium 14d19c09e5b08c9416ef856fd669118f91813b0772714e3d989adbd70f05ff91 e593b75cb2795f02e3094831754746d1a9a5abfeb1f05f948c8d1753874e288b
en_US/joe/medium/en_US-joe-medium abf4b998e34045cbd00eaf050de2de620f81ded6eb8b85a24b022a1c2b3f08d2 9b91a652090ffa8a49e36569afa9acc8ead2b3a22a950e8e6ca11914f3e85d56
en_US/john/medium/en_US-john-medium 789c6c875726e627ddee93d51d8727859abe9c091c3d141591f4b83c2072e988 af60f177b6b550f3d7a302720c0fb89e7f94a82b5dca464775ef63b1c69ba09a
en_US/kathleen/medium/en_US-kathleen-medium 173115e17ff8efb7aa895a4957fb0bccf7c6d9048d82f8c5821902a68f959cc3 9cb87b28715abfe6d73a70df52283fe29849950ecce74d2c36556db91d93c124
en_US/kristin/medium/en_US-kristin-medium 5849957f929cbf720c258f8458692d6103fff2f0e3d3b19c8259474bb06a18d4 5681426d4aead22195de70531eeeeddb46493cfaffc5764b2ea3db73428b651c
en_US/kusal/medium/en_US-kusal-medium 438ae25bb305b2a7f6d632327d6102df25011f793e8222fa9db876e7321df8f3 ddd3c4dfd8b4f568150c934fb94912dd788d44db87f4f0a328c469d7a6761f41
en_US/ljspeech/medium/en_US-ljspeech-medium 6f52a751e2349abe7a76735eb09dc1875298c77ea2342ffd2fef79ff81b87f22 141d612cc0a95ed7efc1ca936b845c2364967f2e9217c5dbfcf69fc4d6c65860
en_US/mike/medium/en_US-mike-medium 6876866e3dfadc7f044bc5e4fbac16c47b5a1e79eaec04e3bb0c2c6e9da8a857 8c6e040667a91624501e4b30c460ef389ad3c995164788c89dd9522151ecbc4c
en_US/norman/medium/en_US-norman-medium b9739443232a80a59c7d18810dd856899bf16a7964725f5ab81ea49b1351cb71 6c2db7f558a4a8deb9fe822583c1c5105f6c4e834dd0f9de8ad17a888ee9fe1d
en_US/reza_ibrahim/medium/en_US-reza_ibrahim-medium 99f0c31464a2120831ca87d079e10a9a2b3e426cc1ee662d80ff9042df15cd3c 465ddf1702917fe617b7d69ed81301d6a2f39f083a754bd1cf6db8955d09a381
en_US/sam/medium/en_US-sam-medium c89f55564302e0d0e7e5baa0f45ab21d932018a334cfe6622abe46cc5713ac00 d46b4a474fbb49e74818f90c1439d453e5203e0306ef6ed6d17319bc03b89cac
LIST
