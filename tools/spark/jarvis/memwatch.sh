#!/usr/bin/env bash
# Samples whole-machine memory while a measured step runs; GB10 device memory is system memory, so a model's real
# footprint is the drop in MemAvailable (CUDA allocations do not show in RSS).
#   memwatch.sh OUT.csv [INTERVAL_S]      runs until killed; columns: ms, MemAvailable KiB, GPU MiB (nvidia-smi, all procs)
#   memwatch.sh --summary OUT.csv [BASELINE_KIB]   peak drop from BASELINE (default: the first sample), in GiB
set -euo pipefail
if [[ ${1:-} == --summary ]]; then
    awk -F, -v base="${3:-}" 'NR==1{next} NR==2&&base==""{base=$2} {if(min==""||$2<min){min=$2;at=$1}; if($3>g)g=$3}
        END{printf "baseline_kib=%d min_avail_kib=%d peak_drop_gib=%.2f at_ms=%d peak_gpu_mib=%d samples=%d\n",
            base,min,(base-min)/1048576,at,g,NR-1}' "$2"
    exit
fi
out=${1:?usage: memwatch.sh OUT.csv [INTERVAL_S]}
dt=${2:-0.25}
echo "ms,mem_available_kib,gpu_used_mib" > "$out"
while :; do
    avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    gpu=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2> /dev/null | awk '{s+=$1} END{print s+0}')
    echo "$(date +%s%3N),$avail,$gpu" >> "$out"
    sleep "$dt"
done
