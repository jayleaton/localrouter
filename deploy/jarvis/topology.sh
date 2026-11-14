# Several Sparks (sourced; jq): a topology file is the daemon's `topology` block (docs/dev/MULTI-SPARK.md,
# Configuration), `self` left out, plus optionally `defaults` (capability -> tool id, over config.json.in's: install.sh
# writes them into the config, not into the block); a node is one machine in it. Used by install.sh, kits.sh, and
# tools/install's bootstrap.sh and doctor.sh.
topo_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# the tool ids of config.json.in
topo_tool_ids() { sed -e 's/@DRAMA_PORT@/0/' -e 's/@DRAMA_RESIDENT@/0/' "$topo_here/config.json.in" | jq -r '[.tools[].id] | join(" ")'; }
# the tools of config.json.in that can be split over nodes (tensor parallel: options.rank_env says how)
topo_rank_ids() { sed -e 's/@DRAMA_PORT@/0/' -e 's/@DRAMA_RESIDENT@/0/' "$topo_here/config.json.in" | jq -r '[.tools[] | select(.options.rank_env | type == "object") | .id] | join(" ")'; }
# each tool of config.json.in and its kind, as a JSON object
topo_tool_kinds() { sed -e 's/@DRAMA_PORT@/0/' -e 's/@DRAMA_RESIDENT@/0/' "$topo_here/config.json.in" | jq -c '[.tools[] | {key: .id, value: .kind}] | from_entries'; }
# topo_check FILE NODE: why FILE with NODE is not a topology this checkout deploys, one reason a line (none: it is)
topo_check() {
  [ -f "$1" ] || { echo "no topology file $1"; return; }
  jq -r --arg n "$2" --arg ids " $(topo_tool_ids) " --arg ranked " $(topo_rank_ids) " --argjson kinds "$(topo_tool_kinds)" '
    def known: . as $m | $ids | contains(" " + $m + " ");
    def caps: ["text_to_image", "image_edit", "text_to_video", "image_to_video", "text_to_speech", "speech_to_text",
               "chat_completion", "typed_decision", "voice_clone", "image_to_3d"];
    if type != "object" then "not a JSON object"
    elif (.nodes | type) != "array" or (.nodes | length) == 0 then "no nodes"
    else .nodes as $ns
      | (if .mode != "multi" then "mode is \(.mode | tojson), not \"multi\"" else empty end),
        (if any($ns[]; .id == $n) then empty else "no node \($n | tojson) (have: \([$ns[].id] | join(" ")))" end),
        ([$ns[] | select(.role == "front")] | length | if . != 1 then "\(.) nodes with role front, needs exactly one" else empty end),
        ($ns | group_by(.id)[] | select(length > 1) | "node \(.[0].id | tojson) is listed twice"),
        ($ns[] | select((.addresses | type) != "array" or (.addresses | length) == 0) | "node \(.id | tojson) has no addresses"),
        ($ns[] | .id as $i | (.models // [])[], (.fallback // [])[] | select(known | not)
          | "node \($i | tojson) names \(tojson), which is not a tool in deploy/jarvis/config.json.in"),
        ([$ns[] | (.models // [])[]] | group_by(.)[] | select(length > 1) | .[0] as $m | select($ranked | contains(" " + $m + " ") | not)
          | "\($m | tojson) is on more than one node, which is tensor parallel: its tool in deploy/jarvis/config.json.in needs options.rank_env"),
        # stand-ins (the front only): a model of another node -> one of the models of the front, of the same kind
        ($ns[] | select(.role != "front" and (.stand_in // {} | length) > 0) | "node \(.id | tojson) lists stand_in models, which only the front runs"),
        ($ns[] | select(.role == "front") | . as $f | (.stand_in // {}) | to_entries[] | .key as $k | .value as $v
          | if ($k | known | not) or ($v | known | not) then "front stand_in \({($k): $v} | tojson) names a model that is not a tool in deploy/jarvis/config.json.in"
            elif ($f.models // [] | index($k)) then "front stand_in names \($k | tojson), one of the models of the front itself"
            elif ([$ns[] | (.models // [])[] | select(. == $k)] | length) > 1 then "\($k | tojson) is tensor parallel and has no stand-in"
            elif ($f.fallback // [] | index($k)) then "\($k | tojson) is under both fallback and stand_in of the front: one or the other"
            elif ($f.models // [] | index($v)) | not then "the stand-in for \($k | tojson), \($v | tojson), is not one of the models of the front"
            elif $kinds[$k] != $kinds[$v] then "the stand-in for \($k | tojson) (\($kinds[$k])) is \($v | tojson), a \($kinds[$v]) model"
            else empty end),
        # defaults: capabilities of the daemon, each naming a model some node runs
        (.defaults // {} | to_entries[] | . as $d
          | if (caps | index($d.key)) == null then "defaults has \($d.key | tojson), which is not a capability (\(caps | join(" ")))"
            elif ([$ns[] | (.models // [])[]] | index($d.value)) == null then "defaults.\($d.key) is \($d.value | tojson), which no node runs"
            else empty end)
    end' "$1" 2> /dev/null || echo "$1 is not valid JSON, or not shaped like the daemon's topology block"
}
# topo_local FILE NODE: the tools NODE runs, space separated: its own models, and on the front its fallback ones
topo_local() { jq -r --arg n "$2" '.nodes[] | select(.id == $n) | (.models // []) + (if .role == "front" then .fallback // [] else [] end) | join(" ")' "$1"; }
# topo_all FILE: every node's models (the tools each node's config lists)
topo_all() { jq -r '[.nodes[] | (.models // [])[]] | unique | join(" ")' "$1"; }
topo_nodes() { jq -r '.nodes[].id' "$1"; }
topo_addrs() { jq -r --arg n "$2" '.nodes[] | select(.id == $n) | .addresses[]' "$1"; }
topo_addr() { jq -r --arg n "$2" '.nodes[] | select(.id == $n) | .addresses[0]' "$1"; }   # its first: a ConnectX-7 link
topo_port() { jq -r --arg n "$2" '.nodes[] | select(.id == $n) | .port // 8190' "$1"; }
topo_role() { jq -r --arg n "$2" '.nodes[] | select(.id == $n) | .role // "node"' "$1"; }
# topo_block FILE NODE: the config's topology block for NODE
topo_block() { jq -c --arg n "$2" 'del(.defaults) + {self: $n}' "$1"; }
