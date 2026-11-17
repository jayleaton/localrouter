#!/usr/bin/env bash
# Regression tests of lib.sh's AMD detection against recorded hosts (tests/fixtures/amd/CASE: a sysfs tree, rocminfo
# output, ROCm's version file, nvidia-smi answers), with no GPU: bash tools/install/test_amd.sh
#   strix-halo  Ryzen AI Max+ 395 / Radeon 8060S (gfx1151): an APU whose GTT pool is its GPU memory; connector entries
#               (card1-DP-1) beside the card
#   discrete    an Intel CPU (rocminfo's first agent) and iGPU (card0) beside a Radeon RX 7900 XTX whose VRAM is real
#   nvidia      an NVIDIA GPU and an AMD card on one box: the NVIDIA path, unchanged
#   none        no GPU
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); fx=$(cd "$here/../../tests/fixtures/amd" && pwd)
fails=0 n=0
scratch=$(mktemp -d); trap 'rm -rf "$scratch"' EXIT
check() {   # NAME GOT WANT
  n=$((n + 1))
  if [ "$2" = "$3" ]; then echo "PASS $1"; else echo "FAIL $1: got '$2', want '$3'"; fails=$((fails + 1)); fi
}
# run CASE CMD...: CMD under lib.sh with the case's sysfs, ROCm and stub tools (each absent tool fails, as on a host
# without it); host binaries other than coreutils, awk, jq and sort are not reachable
run() {
  local c=$fx/$1; shift
  local bin; bin=$(mktemp -d "$scratch/bin.XXXXXX")
  for t in awk basename cat cut df dirname grep head id jq realpath sed sort tr uname; do ln -s "$(command -v $t)" "$bin/$t"; done
  printf '#!/bin/sh\n[ -f %q ] && exec cat %q\nexit 1\n' "$c/rocminfo.txt" "$c/rocminfo.txt" > "$bin/rocminfo"
  cat > "$bin/nvidia-smi" <<SH
#!/bin/sh
f=$c/nvidia-smi.txt; [ -f "\$f" ] || exit 9
case "\$*" in -L) k=-L ;; *--query-gpu=*) k=\$(echo "\$*" | sed 's/.*--query-gpu=\([a-z_.]*\).*/\1/') ;; *) exit 2 ;; esac
sed -n "s/^\$k=//p" "\$f"
SH
  printf '#!/bin/sh\n[ -f %q ] && exec cat %q\nexit 1\n' "$c/nvcc.txt" "$c/nvcc.txt" > "$bin/nvcc"
  printf '#!/bin/sh\nexit 1\n' > "$bin/hipconfig"
  chmod +x "$bin/rocminfo" "$bin/nvidia-smi" "$bin/nvcc" "$bin/hipconfig"
  ( PATH=$bin LR_SYSFS=$c/sys ROCM_PATH=$c/rocm NVCC=$bin/nvcc
    source "$here/lib.sh"; "$@" )
  rm -rf "$bin"
}
yes_no() { "$@" && echo yes || echo no; }
profile() { local f; f=$(mktemp "$scratch/profile.XXXXXX"); echo "$1" > "$f"; echo "$f"; }
match() { run "$1" hw_match "$2" | grep -E '^(ROCm|GPU target|GTT)' | cut -d' ' -f1-2 | tr '\n' ';'; }

# strix-halo
check "strix-halo: AMD path" "$(run strix-halo yes_no hw_amd)" yes
check "strix-halo: card (connectors skipped)" "$(run strix-halo amd_card)" "$fx/strix-halo/sys/class/drm/card1/device"
check "strix-halo: gfx" "$(run strix-halo amd_gfx)" gfx1151
check "strix-halo: GPU name" "$(run strix-halo hw_gpu)" "AMD Radeon 8060S Graphics (gfx1151)"
check "strix-halo: ROCm" "$(run strix-halo hw_rocm)" 7.2.4
check "strix-halo: GTT" "$(run strix-halo hw_gtt)/$(run strix-halo hw_gtt_used)" 103079215104/57671680
check "strix-halo: the 4 GiB carve-out is not VRAM (unified)" "$(run strix-halo hw_vram)" 0
check "strix-halo: driver (in-tree amdgpu: the kernel)" "$(run strix-halo hw_driver)" "amdgpu $(uname -r)"
s=$(run strix-halo hw_summary)
check "strix-halo: summary" "$(echo "$s" | sed 's/memory [0-9.]* GiB/memory M GiB/; s/, [0-9.]* GiB available/, A GiB available/; s/^[^,]*, [^,]*, //; s/driver amdgpu [^)]*/driver amdgpu K/')" \
  "GPU 'AMD Radeon 8060S Graphics (gfx1151)' (driver amdgpu K), ROCm 7.2.4, memory M GiB unified, GTT 96.0 GiB (0.1 GiB in use), A GiB available"
p=$(profile '{"hardware": {"min_memory_gib": 0, "min_rocm": "7.2", "gfx": "gfx1151", "min_gtt_gib": 90, "os": "x"}}')
check "strix-halo: fits min_rocm 7.2, gfx1151, min_gtt_gib 90" "$(match strix-halo "$p")" ""
p=$(profile '{"hardware": {"min_memory_gib": 0, "min_rocm": "7.10", "gfx": "gfx1100", "min_gtt_gib": 100, "os": "x"}}')
check "strix-halo: refused by min_rocm 7.10, gfx1100, min_gtt_gib 100" "$(match strix-halo "$p")" "ROCm '7.2.4';GPU target;GTT 96.0;"

# discrete
check "discrete: AMD path" "$(run discrete yes_no hw_amd)" yes
check "discrete: the AMD card, not the Intel one" "$(run discrete amd_card)" "$fx/discrete/sys/class/drm/card1/device"
check "discrete: GPU name (the GPU agent, not the Intel CPU)" "$(run discrete hw_gpu)" "Radeon RX 7900 XTX (gfx1100)"
check "discrete: VRAM above GTT is the GPU's own" "$(run discrete hw_vram)" 25753026560
check "discrete: driver (amdgpu-dkms version)" "$(run discrete hw_driver)" "amdgpu 6.12.12"
check "discrete: ROCm 6.4.1 fails min_rocm 7.2" "$(match discrete "$(profile '{"hardware": {"min_memory_gib": 0, "min_rocm": "7.2", "os": "x"}}')")" "ROCm '6.4.1';"

# nvidia: what the NVIDIA path said before AMD detection existed
check "nvidia: not the AMD path" "$(run nvidia yes_no hw_amd)" no
check "nvidia: GPU" "$(run nvidia hw_gpu)" "NVIDIA GB10"
check "nvidia: driver" "$(run nvidia hw_driver)" 580.95.05
check "nvidia: unified (no VRAM figure)" "$(run nvidia hw_vram)" 0
s=$(run nvidia hw_summary)
check "nvidia: summary unchanged" "$(echo "$s" | sed 's/memory [0-9.]* GiB/memory M GiB/; s/, [0-9.]* GiB available/, A GiB available/; s/^[^,]*, [^,]*, //')" \
  "GPU 'NVIDIA GB10' (driver 580.95.05), CUDA 13.0, memory M GiB unified, A GiB available"
check "nvidia: the AMD keys are ignored when unset" "$(match nvidia "$(profile '{"hardware": {"min_memory_gib": 0, "os": "x"}}')")" ""

# none
check "none: not AMD" "$(run none yes_no hw_amd)" no
check "none: no GPU" "$(run none hw_gpu)" ""
check "none: GTT 0" "$(run none hw_gtt)" 0
check "none: no ROCm" "$(run none hw_rocm)" ""

echo "$((n - fails))/$n passed"
[ $fails = 0 ]
