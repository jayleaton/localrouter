#!/usr/bin/env bash
# The D2 attribution probe job on a DGX Spark (GB10), under the shared staging runner's D2 memory policy
# (gb10-unified-d2-v2; contract and PASS / FAIL rules in D2-EVIDENCE.md): the gate a node's D2 payload needs.
#   run-probe.sh torch    dgx01, d2-probe-torch: the original D2 bundle's interpreter (/opt/k6-oracle-d2) runs
#                         d2-allocator-guard.py --probe (torch.empty(X, uint8).fill_(1), synchronize)
#   run-probe.sh native   dgx02, d2-probe-native: k6v-qual --probe (cuMemAlloc through the 24 GiB ledger, cuMemsetD8,
#                         synchronize)
# Six sealed phases, each held 10 s: P0 pre-context (no CUDA library loaded), P1 context, P2 alloc 4 GiB, P3 free,
# P4 alloc 16 GiB, P5 free. The probe writes each phase record (probe-phases.jsonl) as the phase ends, and the allocator
# evidence from P1 on (allocator-{ready.json, metrics.jsonl, terminal.json}, a sample every 0.5 s); the runner
# correlates its own samples with each phase window. After the probe is reaped, the reaped record
# (allocator-producer-exit.json). Nothing else runs; the process sees only the `env -i` whitelist below.
# Env (the runner's): K6_OUTPUT, K6_MEMORY_POLICY, K6_MEMORY_NONCE, K6_JOB_ID, K6_ALLOCATOR_RECEIPT, K6_ALLOCATOR_LOG;
# torch also ORACLE_ROOT (/opt/k6-oracle-d2) and ORACLE_MANIFEST_SHA256 (the original bundle's, checked by the payload
# job's env-probe segment, not here: the probe imports torch from the read-only mount only).
set -uo pipefail
KIT=${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${K6_OUTPUT:?K6_OUTPUT must name the output directory}
KIND=${1:?usage: run-probe.sh torch|native}
[[ ${K6_MEMORY_POLICY:-} == gb10-unified-d2-v2 ]] || { echo "[d2-probe] needs the runner's D2 memory policy" >&2; exit 2; }
mkdir -p $OUT/logs
VARS=(K6_MEMORY_POLICY=$K6_MEMORY_POLICY K6_MEMORY_NONCE=${K6_MEMORY_NONCE:?} K6_JOB_ID=${K6_JOB_ID:?} K6_OUTPUT=$OUT
      K6_ALLOCATOR_RECEIPT=${K6_ALLOCATOR_RECEIPT:?} K6_ALLOCATOR_LOG=${K6_ALLOCATOR_LOG:?}
      ${CUDA_VISIBLE_DEVICES:+CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES} ${NVIDIA_VISIBLE_DEVICES:+NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES})
case $KIND in
torch)
    OR=/opt/k6-oracle-d2
    [[ ${ORACLE_ROOT:-} == "$OR" ]] || { echo "[d2-probe] ORACLE_ROOT must be $OR" >&2; exit 2; }
    PY=$OR/python/bin/python3.13
    ENV=(env -i PATH=/usr/bin:/bin HOME=/tmp LANG=C.UTF-8 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
         PYTHONPATH=$OR/site-packages PYTORCH_ALLOC_CONF=backend:native,throw_on_cudamalloc_oom:True
         PYTORCH_CUDA_ALLOC_CONF=backend:native,throw_on_cudamalloc_oom:True "${VARS[@]}")
    PROBE=($PY -s $TOOLS/d2-allocator-guard.py --probe)
    REAP=($PY -I $TOOLS/d2-allocator-guard.py --record-exit -) ;;
native)
    ENV=(env -i PATH=/usr/bin:/bin HOME=/tmp LANG=C.UTF-8 "${VARS[@]}")
    PROBE=($KIT/bin/k6v-qual --probe)
    REAP=($KIT/bin/k6v-qual --record-exit) ;;
*) echo "usage: run-probe.sh torch|native" >&2; exit 2 ;;
esac
echo "[d2-probe $(date +%T)] $KIND"
t0=$(date +%s)
timeout --kill-after=10 300 "${ENV[@]}" "${PROBE[@]}" > $OUT/logs/probe.log 2>&1
rc=$?
"${ENV[@]}" "${REAP[@]}" $rc || { echo "[d2-probe] reaped record failed"; exit 67; }
echo "{\"step\": \"probe-$KIND\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" | tee -a $OUT/steps.jsonl
chmod -R a+rX $OUT   # the container is root; the results are the user's to read
exit $rc
