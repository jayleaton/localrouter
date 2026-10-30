#!/usr/bin/env bash
# What a box lacks for a profile (INSTALL.md), each with the command that fixes it: the hardware, the prerequisites,
# every install step of the manifest (deploy/profiles/PROFILE.json, by its own `done` test), then the service: unit,
# health, the models' states and memory. Changes nothing.
#   tools/install/doctor.sh --profile jarvis-gb10 [--models chat,tts,stt,image,video] [--prefix DIR] [--port N] [--unit NAME]
#     [--topology FILE --node ID [--ssh-user NODE=USER]...]
# With --topology (this box is node ID; deploy/jarvis/topology.sh) it checks the node's models only, and the link: the
# RoCE devices and MTU 9000 on the interfaces that carry this node's addresses, and for every other node a ping of each
# of its addresses, ssh (as USER, default the current user) and its daemon's /health.
# Exit 0 when nothing is missing (warnings do not count), 1 otherwise.
set -uo pipefail
source "$(dirname "$0")/lib.sh"

profile="" list=all PREFIX=$HOME/jarvis PORT=8190 HTTPS_PORT=10443 UNIT=localrouter-jarvis
topology="" node="" port_set=0 sshu=() listen=127.0.0.1
while [ $# -gt 0 ]; do
  case $1 in
    --profile) profile=${2:?}; shift 2 ;;
    --models) list=${2:?}; shift 2 ;;
    --prefix) PREFIX=$(realpath -m "${2:?}"); shift 2 ;;
    --port) PORT=${2:?}; port_set=1; shift 2 ;;
    --https-port) HTTPS_PORT=${2:?}; shift 2 ;;
    --unit) UNIT=${2:?}; shift 2 ;;
    --topology) topology=${2:?}; shift 2 ;;
    --node) node=${2:?}; shift 2 ;;
    --ssh-user) sshu+=("${2:?}"); shift 2 ;;
    -h | --help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument $1 (see --help)" ;;
  esac
done
[ -n "$profile" ] || die "--profile is required (have: $(profiles))"
command -v jq > /dev/null || { echo "MISSING jq"; echo "  fix: sudo apt-get install -y jq"; exit 1; }
p=$(profile_file "$profile") || exit 1
[ -z "$topology$node" ] || use_topology
mapfile -t keys < <(selected "$p" "$list")
step_env "$p"
boot="tools/install/bootstrap.sh --profile $profile --models $(IFS=,; echo "${keys[*]}") --prefix $PREFIX --port $PORT --https-port $HTTPS_PORT --unit $UNIT${topology:+ --topology $topology --node $node}"
bad=0
ok() { echo "ok      $*"; }
miss() { echo "MISSING $1"; echo "        fix: $2"; bad=1; }
warn() { echo "warning $1"; [ -n "${2:-}" ] && echo "        fix: $2"; }

echo "== box: $(hw_summary)"
while read -r line; do
  [ -n "$line" ] || continue
  case $line in warning:*) warn "${line#warning: }" ;; *) miss "$line" "this box does not fit $profile (INSTALL.md, Profiles)" ;; esac
done < <(hw_match "$p")
avail=$(hw_disk_free "$PREFIX")
echo "        disk free under $PREFIX: $(gib "$avail") GiB"
[ $(($(hw_avail) / 1073741824)) -ge 88 ] || systemctl --user is-active -q "$UNIT" || warn "only $(gib "$(hw_avail)") GiB of memory is available: the GPU steps need up to 88 GiB free and the warm models about 107 GiB" "see what holds it: nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv (INSTALL.md step 6)"

echo "== prerequisites"
while read -r q; do
  if ok_test "$q"; then ok "$(jq -r .title <<< "$q")"; else miss "$(jq -r .title <<< "$q")" "$(jq -r .fix <<< "$q")"; fi
done < <(prereqs "$p" "${keys[@]}")

echo "== install steps (under $PREFIX)"
while read -r s; do
  if is_done "$s"; then ok "$(step_id "$s"): $(jq -r .title <<< "$s")"
  else miss "$(step_id "$s"): $(jq -r .title <<< "$s")" "$boot   (it resumes at the first step not done)"; fi
done < <(steps "$p" "${keys[@]}")

echo "== service $UNIT on $listen:$PORT"
url=http://$listen:$PORT
if [ ! -f "$HOME/.config/systemd/user/$UNIT.service" ]; then
  miss "the unit $UNIT is not installed" "$boot"
elif ! systemctl --user is-active -q "$UNIT"; then
  miss "the unit $UNIT is installed but not running ($(systemctl --user is-active "$UNIT"))" "bash $repo/deploy/jarvis/start.sh   (why it stopped: journalctl --user -u $UNIT -n 50)"
else
  ok "unit $UNIT active since $(systemctl --user show "$UNIT" -p ActiveEnterTimestamp --value)"
  [ "$(systemctl --user is-enabled "$UNIT" 2> /dev/null)" = enabled ] || warn "the unit does not start at boot" "systemctl --user enable $UNIT"
  if ! curl -fs -m 5 "$url/health" > /dev/null; then
    miss "the daemon does not answer on $url/health" "journalctl --user -u $UNIT -n 50; then systemctl --user restart $UNIT"
  else
    ok "health on $url"
    t=$(curl -fs -m 10 "$url/v1/tools")
    for k in "${keys[@]}"; do
      m=$(model "$p" "$k")
      id=$(jq -r .id <<< "$m")
      st=$(jq -r --arg id "$id" '.data[] | select(.id == $id) | .state' <<< "$t")
      if [ -z "$st" ]; then miss "$k ($id) is not in the service's config" "$boot"
      elif [ "$(jq -r .warm <<< "$m")" = true ] && [ "$st" != ready ]; then
        err=$(jq -r --arg id "$id" '.data[] | select(.id == $id) | .last_error' <<< "$t")
        miss "$k ($id) should be warm, is $st${err:+ (last error: $err)}" "wait for it (start.sh waits until warm), else journalctl --user -u $UNIT and $PREFIX/deploy/data/logs/"
      else ok "$k ($id): $st"; fi
    done
    q=$(curl -fs -m 10 "$url/v1/models" | jq -r '[.data[] | select(.qualified == false) | .id] | join(" ")')
    [ -z "$q" ] || miss "not GPU-qualified on this box: $q" "$boot   (it runs each model's qualify step)"
    echo "        MemAvailable $(jq -r '.mem_available_bytes / 1073741824 * 10 | round / 10' <<< "$t") GiB, reserve $(jq -r '.reserve_bytes / 1073741824' <<< "$t") GiB, voice sessions $(jq -r '.voice_sessions // "?"' <<< "$t")"
  fi
fi
if [ "$HTTPS_PORT" != 0 ] && [ "$listen" = 127.0.0.1 ]; then   # (a node that is not the front serves the front only)
  if ! command -v tailscale > /dev/null; then warn "no tailscale: reach the service with ssh -L $PORT:127.0.0.1:$PORT (deploy/jarvis/README.md)"
  elif tailscale serve status 2> /dev/null | grep -q "127.0.0.1:$PORT"; then ok "tailscale serve: HTTPS $HTTPS_PORT to 127.0.0.1:$PORT"
  else warn "tailscale serve has no HTTPS entry for 127.0.0.1:$PORT" "sudo tailscale serve --bg --https=$HTTPS_PORT http://127.0.0.1:$PORT"; fi
fi

if [ -n "$topology" ]; then
  echo "== nodes ($topology; this is $node, $(topo_role "$topology" "$node"))"
  # the RoCE devices: "NETDEV STATE" a line (ACTIVE: up)
  if command -v ibdev2netdev > /dev/null; then   # mlx5_0 port 1 ==> enp1s0f0np0 (Up)
    roce=$(ibdev2netdev 2> /dev/null | awk '{print $5, ($6 == "(Up)" ? "ACTIVE" : $6)}')
  elif command -v rdma > /dev/null; then         # link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev enp1s0f0np0
    roce=$(rdma link show 2> /dev/null | awk '{n = ""; s = ""; for (i = 1; i < NF; i++) { if ($i == "netdev") n = $(i + 1); if ($i == "state") s = $(i + 1) } if (n != "") print n, s}')
  else
    roce=""
    warn "neither ibdev2netdev nor rdma here: the RoCE devices are not checked" "sudo apt-get install -y rdma-core (rdma), or DOCA/MLNX OFED (ibdev2netdev)"
  fi
  ifaddrs=$(ip -o -4 addr show 2> /dev/null)
  for a in $(topo_addrs "$topology" "$node"); do
    ifc=$(awk -v a="$a" '{split($4, x, "/"); if (x[1] == a) print $2}' <<< "$ifaddrs" | first)
    if [ -z "$ifc" ]; then miss "this node's address $a is on no interface here" "set it on its ConnectX-7 port (netplan), or correct the topology file"; continue; fi
    mtu=$(ip -o link show dev "$ifc" 2> /dev/null | sed -n 's/.* mtu \([0-9]*\) .*/\1/p')
    st=$(awk -v n="$ifc" '$1 == n {print $2}' <<< "$roce" | first)
    if [ -n "$roce" ] && [ -z "$st" ]; then ok "$a on $ifc (no RoCE device: not a ConnectX-7 port; MTU $mtu not checked)"; continue; fi
    [ -z "$st" ] || [ "$st" = ACTIVE ] || miss "the RoCE device of $ifc ($a) is $st, not up" "check the cable and the port (ibdev2netdev, rdma link); sudo ip link set dev $ifc up"
    if [ "$mtu" = 9000 ]; then ok "$a on $ifc: MTU 9000$([ "$st" = ACTIVE ] && echo ", RoCE up")"
    else miss "$ifc ($a) has MTU ${mtu:-?}, the link needs 9000" "sudo ip link set dev $ifc mtu 9000, and mtu: 9000 for it in netplan so it stays"; fi
  done
  tf=$(jq -r '.token_file // ""' "$topology")
  [ "$(topo_role "$topology" "$node")" != front ] || [ -z "$tf" ] || [ -r "$tf" ] ||
    warn "the front's node token $tf is missing (nodes that require pairing refuse the front without it)" "on each other node: deploy/jarvis/pair.sh token front --scopes manage; put that token in $tf"
  for n in $(topo_nodes "$topology"); do
    [ "$n" != "$node" ] || continue
    u=$(id -un) port=$(topo_port "$topology" "$n") login="" health=""
    for x in "${sshu[@]}"; do [ "${x%%=*}" = "$n" ] && u=${x#*=}; done
    for a in $(topo_addrs "$topology" "$n"); do
      if ping -c 1 -W 1 "$a" > /dev/null 2>&1; then ok "node $n: $a answers ping"
      else miss "node $n: $a does not answer ping" "check the cable and the addresses on both nodes (ip -4 addr), then: ping -c 3 $a"; fi
      [ -n "$login" ] || ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$u@$a" true < /dev/null > /dev/null 2>&1 || login=$u@$a
      [ -n "$health" ] || [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://$a:$port/health" 2> /dev/null)" != 200 ] || health=http://$a:$port
    done
    if [ -n "$login" ]; then ok "node $n: ssh $login"
    else miss "node $n: no ssh login as $u on $(topo_addrs "$topology" "$n" | paste -sd ' ')" "ssh-copy-id $u@$(topo_addr "$topology" "$n") (another user: --ssh-user $n=USER)"; fi
    if [ -n "$health" ]; then ok "node $n: its daemon answers on $health/health"
    else warn "node $n: no daemon answers /health on port $port (not installed or not started yet?)" "on $n: tools/install/bootstrap.sh --profile $profile --topology FILE --node $n, then deploy/jarvis/start.sh"; fi
  done
fi

[ $bad = 0 ] && { echo "== nothing missing"; exit 0; }
echo "== something is missing: do each fix above (bootstrap.sh does the install steps and resumes where it stopped)"
exit 1
