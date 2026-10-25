#!/usr/bin/env bash
# One deploy kit per node of a topology (README.md, Several Sparks):
#   deploy/jarvis/kits.sh TOPOLOGY OUTDIR
# OUTDIR/NODE/ gets topology.json (the topology file, checked, `self` left out) and node.env (NODE, TOPOLOGY: the
# topology.json beside it, TOOLS: the tools the node runs, its own and on the front its fallback ones, and PORT).
# Copy OUTDIR/NODE to that node; there, from a checkout of the same commit:
#   set -a; . KIT/node.env; set +a; deploy/jarvis/install.sh && deploy/jarvis/start.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
die() { echo "$(basename "$0"): $*" >&2; exit 1; }
source "$here/topology.sh"
[ $# = 2 ] || die "usage: deploy/jarvis/kits.sh TOPOLOGY OUTDIR"
topology=$1 out=$2
command -v jq > /dev/null || die "needs jq"
[ ! -e "$out" ] || [ -z "$(ls -A "$out" 2> /dev/null)" ] || die "$out exists and is not empty"
nodes=$(topo_nodes "$topology" 2> /dev/null) || die "$topology is not a topology file (topology.sh)"
for n in $nodes; do
  why=$(topo_check "$topology" "$n")
  [ -z "$why" ] || die "$topology: ${why//$'\n'/; }"
done
for n in $nodes; do
  mkdir -p "$out/$n"
  jq 'del(.self)' "$topology" > "$out/$n/topology.json"
  {
    echo "# node $n of a LocalRouter topology (deploy/jarvis/kits.sh, $(date -u +%F)); source it with set -a, then install.sh"
    echo "NODE=$n"
    echo 'TOPOLOGY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/topology.json'
    echo "TOOLS='$(topo_local "$topology" "$n")'"
    echo "PORT=$(topo_port "$topology" "$n")"
  } > "$out/$n/node.env"
  echo "kit: $out/$n ($(topo_role "$topology" "$n"), $(topo_addr "$topology" "$n"):$(topo_port "$topology" "$n"): $(topo_local "$topology" "$n"))"
done
echo "copy each kit to its node; install the nodes that are not the front first, then the front (README.md, Several Sparks)"
