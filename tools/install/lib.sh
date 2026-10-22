# Shared by bootstrap.sh and doctor.sh (sourced): the box's hardware, the profile's manifest (deploy/profiles/*.json)
# and its steps. A step is {id, title, run, done, fix} (+ gpu, min_free_gib, needs_accept): `done` is a shell test that
# is true once the step's result is in place, so a finished step is skipped and a re-run resumes at the first unfinished
# one; `run` makes it so; `fix` says what to do when `run` fails. Every command runs under bash with step_env's variables.

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
die() { echo "$(basename "$0"): $*" >&2; exit 1; }
gib() { awk -v b="${1:-0}" 'BEGIN{printf "%.1f", b / 1073741824}'; }
first() { awk 'NR == 1'; }   # the first line, reading the rest (no SIGPIPE for the producer under pipefail)

profiles() { (cd "$repo/deploy/profiles" && ls *.json | sed 's/\.json$//' | tr '\n' ' '); }
# profile_file NAME: the manifest of a profile
profile_file() {
  local f="$repo/deploy/profiles/$1.json"
  [ -f "$f" ] || die "no profile '$1' (have: $(profiles))"
  jq -e '.models | length > 0' "$f" > /dev/null ||
    die "the profile manifest $f is not valid (jq says why above): a bug in this checkout, not on this box; report it"
  echo "$f"
}

# ---- the box
hw_arch() { uname -m; }
hw_os() { (. /etc/os-release 2> /dev/null && echo "${ID:-unknown} ${VERSION_ID:-}"); }
hw_gpu() { nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | first; }
hw_driver() { nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null | first; }
# the GPU's own memory in bytes; 0 on a unified-memory GPU (GB10 reports none: its memory is the system's)
hw_vram() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2> /dev/null | awk 'NR == 1 {v = $1 + 0} END {printf "%d", v * 1048576}'; }
hw_mem() { awk '/^MemTotal:/{printf "%d", $2 * 1024}' /proc/meminfo; }
hw_avail() { awk '/^MemAvailable:/{printf "%d", $2 * 1024}' /proc/meminfo; }
hw_cuda() { "${NVCC:-/usr/local/cuda/bin/nvcc}" --version 2> /dev/null | sed -n 's/.*release \([0-9.]*\),.*/\1/p'; }
# free bytes on the filesystem that holds DIR (or its nearest existing parent)
hw_disk_free() {
  local d=$1
  while [ ! -e "$d" ]; do d=$(dirname "$d"); done
  df -B1 --output=avail "$d" | awk 'END {print $1}'
}
version_ge() { [ -n "$1" ] && [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | first)" = "$2" ]; }
hw_summary() {
  echo "$(hw_arch), $(hw_os), GPU '$(hw_gpu)' (driver $(hw_driver)), CUDA $(hw_cuda), memory $(gib "$(hw_mem)") GiB$([ "$(hw_vram)" = 0 ] && echo " unified"), $(gib "$(hw_avail)") GiB available"
}

# hw_match PROFILE_FILE: one line a requirement the box misses (none: it matches); "warning:" lines do not block
hw_match() {
  local p=$1 want
  want=$(jq -r .hardware.arch "$p"); [ "$(hw_arch)" = "$want" ] || echo "arch $(hw_arch), the profile needs $want"
  want=$(jq -r .hardware.gpu "$p"); hw_gpu | grep -q "$want" || echo "GPU '$(hw_gpu)', the profile needs '$want'"
  want=$(jq -r .hardware.min_memory_gib "$p")
  [ "$(hw_mem)" -ge $((want * 1073741824)) ] || echo "memory $(gib "$(hw_mem)") GiB, the profile needs $want GiB"
  want=$(jq -r .hardware.min_driver "$p"); version_ge "$(hw_driver)" "$want" || echo "NVIDIA driver '$(hw_driver)', the profile needs $want or newer"
  want=$(jq -r .hardware.min_cuda "$p"); version_ge "$(hw_cuda)" "$want" || echo "CUDA '$(hw_cuda)' (nvcc at ${NVCC:-/usr/local/cuda/bin/nvcc}), the profile needs $want or newer"
  want=$(jq -r .hardware.os "$p"); [ "$(hw_os)" = "$want" ] || echo "warning: OS '$(hw_os)', the profile was tested on '$want'"
}

# ---- the manifest
models() { jq -r '.models[].key' "$1"; }
# selected PROFILE_FILE LIST: the keys in LIST (comma separated; "all": every model), in the manifest's order
selected() {
  local p=$1 list=${2:-all} k
  [ "$list" = all ] && { models "$p"; return; }
  for k in ${list//,/ }; do
    jq -e --arg k "$k" '.models[] | select(.key == $k)' "$p" > /dev/null || die "no model '$k' in $(basename "$p") (have: $(models "$p" | tr '\n' ' '))"
  done
  models "$p" | while read -r k; do [[ ",$list," == *",$k,"* ]] && echo "$k"; done
}
model() { jq -c --arg k "$2" '.models[] | select(.key == $k)' "$1"; }
# steps PROFILE_FILE KEYS...: the steps in order, as compact JSON lines (each with its model's key, "host" for the host's)
steps() {
  local p=$1; shift
  jq -c '.host.steps[] | . + {model: "host"}' "$p"
  for k in "$@"; do jq -c --arg k "$k" '.models[] | select(.key == $k) | .steps[] | . + {model: $k}' "$p"; done
}
# prereqs PROFILE_FILE KEYS...: the prerequisites that apply to these models
prereqs() {
  local p=$1; shift
  jq -c --arg keys " $* " '.prereqs[] | select((.for // null) == null or any(.for[]; . as $f | $keys | contains(" " + $f + " ")))' "$p"
}

# step_env PROFILE_FILE: exports what step commands use (PREFIX, PORT, HTTPS_PORT and UNIT must be set)
step_env() {
  export REPO=$repo PROFILE_FILE=$1 PREFIX J=$PREFIX JARVIS=$PREFIX PORT HTTPS_PORT UNIT
  export PATH=$PREFIX/bin:$PATH NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
  export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
  export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}
  local k v
  while IFS=$'\t' read -r k v; do export "$k=$(eval "echo \"$v\"")"; done < <(jq -r '.env | to_entries[] | [.key, .value] | @tsv' "$1")
}

is_done() { bash -c "$(jq -r .done <<< "$1")" > /dev/null 2>&1; }
ok_test() { bash -c "$(jq -r .test <<< "$1")" > /dev/null 2>&1; }
step_id() { jq -r '.model + "/" + .id' <<< "$1"; }
