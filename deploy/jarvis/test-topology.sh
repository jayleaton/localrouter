#!/usr/bin/env bash
# CPU-only checks of the Jarvis deploy's topologies and the configs install.sh renders (bash and jq; no GPU, no build,
# nothing installed): deploy/jarvis/test-topology.sh
# - every shipped topology (topologies/*.json) passes topo_check for each of its nodes, and kits.sh makes its kits;
# - topo_check refuses broken topologies with the reason;
# - one box (no TOPOLOGY): the config is today's, without the tools that need a Spark of their own (`own_spark`);
# - flashnext-2spark: each node's config has every node's tools, the topology's defaults, the block without
#   `defaults`, `self`, `machine` and `host` for that node, and the stand-in not kept loaded.
# Prints one line a check and exits 1 when any fails.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/topology.sh"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/lr-topo-test.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
fails=0
pass() { echo "ok    $1"; }
fail() { echo "FAIL  $1${2:+: $2}"; fails=$((fails + 1)); }
# expect NAME JQ-FILTER JSON: the filter is true of the JSON
expect() { if jq -e "$2" <<< "$3" > /dev/null 2>&1; then pass "$1"; else fail "$1" "$(jq -c "$2" <<< "$3" 2>&1 | head -c 300)"; fi; }
# render [VAR=VALUE...]: the config install.sh would write, in a scratch prefix
render() { env -u TOPOLOGY -u NODE -u TOOLS -u PORT -u HOST JARVIS="$tmp/jarvis" DEPLOY="$tmp/jarvis/deploy" RENDER_ONLY=1 "$@" bash "$here/install.sh"; }

# ---- shipped topologies
for f in "$here"/topologies/*.json; do
  name=$(basename "$f")
  for n in $(topo_nodes "$f"); do
    why=$(topo_check "$f" "$n")
    if [ -z "$why" ]; then pass "$name: node $n passes topo_check"; else fail "$name: node $n" "$why"; fi
  done
  if bash "$here/kits.sh" "$f" "$tmp/kits-$name" > /dev/null; then pass "$name: kits.sh"; else fail "$name: kits.sh"; fi
done

# ---- broken topologies, each refused with its reason
base=$here/topologies/flashnext-2spark.json
refused() { # refused NAME JQ-EDIT REASON
  jq "$2" "$base" > "$tmp/bad.json"
  local why
  why=$(topo_check "$tmp/bad.json" spark-2)
  if [[ "$why" == *"$3"* ]]; then pass "refused: $1"; else fail "refused: $1" "got: ${why:-nothing}"; fi
}
refused "stand-in on a node" '.nodes[0].stand_in = {"qwen3.8-27b": "qwen3.8-flash-next"}' "lists stand_in models, which only the front runs"
refused "stand-in for an unknown model" '.nodes[1].stand_in = {"nope": "qwen3.8-27b"}' "names a model that is not a tool"
refused "stand-in for its own model" '.nodes[1].stand_in = {"knowline-4b": "qwen3.8-27b"}' "one of the models of the front itself"
refused "stand-in not on the front" '.nodes[1].models -= ["qwen3.8-27b"] | .nodes[0].models += ["qwen3.8-27b"]' "is not one of the models of the front"
refused "stand-in of another kind" '.nodes[1].stand_in = {"qwen3.8-flash-next": "whisper-large-v3-turbo"}' "a transcription model"
refused "stand-in and fallback" '.nodes[1].fallback = ["qwen3.8-flash-next"]' "under both fallback and stand_in"
refused "default no node runs" '.defaults.chat_completion = "kitten"' "which no node runs"
refused "default of no capability" '.defaults.chat = "qwen3.8-27b"' "is not a capability"

# ---- one box: unchanged, no own_spark tool
one=$(render) || fail "one box: install.sh RENDER_ONLY=1"
expect "one box: no topology, chat default qwen3.8-27b" '(has("topology") | not) and .defaults.chat_completion == "qwen3.8-27b"' "$one"
expect "one box: every tool but qwen3.8-flash-next" '[.tools[].id] == ["qwen3.8-27b", "fishaudio/drama-3", "whisper-large-v3-turbo", "qwen-image-2.1-turbo", "minimax-h3", "knowline-4b", "idol"]' "$one"
expect "one box: no own_spark key, chat kept loaded" 'all(.tools[]; has("own_spark") | not) and (.tools[0].keep_loaded == true)' "$one"

# ---- two Sparks: qwen3.8-flash-next on spark-1, the rest on spark-2 with qwen3.8-27b standing in
for n in spark-1 spark-2; do
  c=$(render TOPOLOGY="$base" NODE="$n") || { fail "$n: install.sh RENDER_ONLY=1"; continue; }
  expect "$n: every node's tools" '[.tools[].id] | sort == (["qwen3.8-flash-next", "qwen3.8-27b", "fishaudio/drama-3", "whisper-large-v3-turbo", "qwen-image-2.1-turbo", "minimax-h3", "knowline-4b", "idol"] | sort)' "$c"
  expect "$n: chat default qwen3.8-flash-next, the others as one box" '.defaults.chat_completion == "qwen3.8-flash-next" and .defaults.text_to_speech == "fishaudio/drama-3"' "$c"
  expect "$n: topology block with self, no defaults" ".topology.self == \"$n\" and (.topology | has(\"defaults\") | not) and .topology.nodes[1].stand_in[\"qwen3.8-flash-next\"] == \"qwen3.8-27b\"" "$c"
  expect "$n: the stand-in is not kept loaded, qwen3.8-flash-next is" '(.tools[] | select(.id == "qwen3.8-27b") | .keep_loaded) == false and (.tools[] | select(.id == "qwen3.8-flash-next") | .keep_loaded) == true' "$c"
  expect "$n: qwen3.8-flash-next booked 111.5 GiB, served by serve.sh, no own_spark key" '.tools[] | select(.id == "qwen3.8-flash-next") | .resident_bytes == 119722213376 and (.options.server[0] | endswith("/tools/chat/qwen38-flash-next/serve.sh")) and (has("own_spark") | not)' "$c"
  expect "$n: nothing left unrendered" 'tostring | test("@[A-Z_]+@") | not' "$c"
done
c1=$(render TOPOLOGY="$base" NODE=spark-1 2> /dev/null || echo '{}')
expect "spark-1: machine spark-1, listens on its first ConnectX-7 address" '.machine == "spark-1" and .host == "192.168.100.11"' "$c1"
c2=$(render TOPOLOGY="$base" NODE=spark-2 2> /dev/null || echo '{}')
expect "spark-2: machine spark-2, the front on loopback" '.machine == "spark-2" and .host == "127.0.0.1"' "$c2"
k1=$(cat "$tmp/kits-flashnext-2spark.json/spark-1/node.env" 2> /dev/null)
k2=$(cat "$tmp/kits-flashnext-2spark.json/spark-2/node.env" 2> /dev/null)
if [[ "$k1" == *"TOOLS='qwen3.8-flash-next'"* ]]; then pass "kit spark-1: TOOLS is qwen3.8-flash-next"; else fail "kit spark-1" "$k1"; fi
if [[ "$k2" == *"TOOLS='qwen3.8-27b fishaudio/drama-3"* ]]; then pass "kit spark-2: TOOLS starts with the stand-in and voice"; else fail "kit spark-2" "$k2"; fi

if [ "$fails" = 0 ]; then echo "all passed"; else echo "$fails failed"; exit 1; fi
