#!/usr/bin/env bash
# Downloads the pinned Kitten TTS 2 files (tools/speech/kitten/pins.json) into ROOT and checks every sha256.
#   tools/speech/kitten/fetch.sh ROOT [--lm packed|full|emb4] [--decoder default|student_w4] [--cpp]
# ROOT/kitten-tts-2/<revision>/ and ROOT/chatterbox-turbo/<revision>/ then hold what the worker loads (its
# `weights`). Files already present with the pinned hash are kept. --cpp adds the GGUF and TorchScript decoder of
# the kitten-tts-2-cpp runtime. Nothing is executed from the downloads.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=${1:?usage: fetch.sh ROOT [--lm packed|full|emb4] [--decoder default|student_w4] [--cpp]}
shift
lm=packed decoder=default cpp=0
while [ $# -gt 0 ]; do
  case $1 in
    --lm) lm=$2; shift 2 ;;
    --decoder) decoder=$2; shift 2 ;;
    --cpp) cpp=1; shift ;;
    *) echo "fetch.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

# One line a file: repo revision dir path size sha256.
plan=$(python3 -I - "$here/pins.json" "$lm" "$decoder" "$cpp" <<'EOF'
import json, sys
pins, lm, decoder, cpp = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4] == "1"
k, s = pins["kitten"], pins["s3gen"]
paths = list(k["common"]) + k["lm"][lm] + k["decoders"][decoder]
if cpp:
    paths += ["cpp/model-tq2_1.gguf", "cpp/default/decoder.pt", "cpp/default/voices.json"]
for p in paths:
    size, digest = pins["files"]["kitten"][p]
    print(k["repo"], k["revision"], k["dir"], p, size, digest)
for p in [s["file"]] + s.get("extra", []):
    size, digest = pins["files"]["s3gen"][p]
    print(s["repo"], s["revision"], s["dir"], p, size, digest)
EOF
)

sha() { sha256sum "$1" | cut -d' ' -f1; }
n=0
while read -r repo rev dir path size digest; do
  dest="$root/$dir/$rev/$path"
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ] && [ "$(sha "$dest")" = "$digest" ]; then
    n=$((n + 1)); continue
  fi
  mkdir -p "$(dirname "$dest")"
  curl -fsSL --retry 3 -o "$dest.part" "https://huggingface.co/$repo/resolve/$rev/$path"
  got=$(sha "$dest.part")
  if [ "$got" != "$digest" ]; then
    rm -f "$dest.part"
    echo "fetch.sh: $repo/$path sha256 $got, pinned $digest" >&2
    exit 1
  fi
  mv "$dest.part" "$dest"
  n=$((n + 1))
done <<< "$plan"
echo "fetch.sh: $n files verified under $root (lm $lm, decoder $decoder)"
