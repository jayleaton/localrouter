#!/usr/bin/env bash
# The D2 oracle job on a DGX Spark (GB10, aarch64, sm_121), inside the arm64 NGC PyTorch image, offline, under the shared
# staging runner's D2 memory policy (gb10-unified-d2-v2; contract in D2-EVIDENCE.md). It runs the pinned native stack
# from the original D2 bundle (pybundle.sh: CPython 3.13, torch 2.14.1+cu130, diffusers 0.40.0, NumPy 2.5.3), mounted
# read-only at /opt/k6-oracle-d2, never the image's Python or torch.
# The job is a sealed sequence of segments, each a fresh process under d2-allocator-guard.py (its entry point: the
# 38 GiB native Torch cap, ready receipt, a sample each second, terminal receipt, each bound to the segment's own nonce):
#   env-probe   the bundle (every file against its manifest, the manifest against ORACLE_MANIFEST_SHA256), the
#               interpreter and the pinned versions, with a CUDA context
#   s_dec f_dec s_enc n_enc n_dec   oracle.py --cases <case>: a fresh process per case, so every case sees the native
#               generator, its own CUDA context and caching allocator and mem_get_info as a standalone run
# After each segment the launcher reaps the process and writes its reaped record (--record-exit) before the next one
# starts; a segment that fails, times out or would not fit JOB_BUDGET_S ends the sequence (the runner fails the job).
# Each process runs under `env -i` with only the whitelist below, so the image's PYTHONPATH, LD_LIBRARY_PATH (its CUDA /
# cuDNN libraries) and torch overrides (e.g. TORCH_ALLOW_TF32_CUBLAS_OVERRIDE) do not reach it; env.json lists the CUDA
# libraries the process actually mapped.
# Outputs under $OUT: allocator-<segment>-{ready.json, metrics.jsonl, terminal.json, producer-exit.json}, oracle/ (the
# record set: ops.jsonl, blobs/, env.json, summary.json), logs/<step>.log, steps.jsonl ({"step","rc","s"}),
# memavail.log (MemAvailable each second: GB10's memory is the system's) and summary.jsonl (each step's last JSON line,
# then the MemAvailable start / minimum / peak drop).
# Env (the runner's): K6_OUTPUT, K6_MEMORY_POLICY, K6_MEMORY_NONCE, K6_JOB_ID, K6_SEGMENT_NONCES; (the spec's) VAE (default
# /models/vae), ORACLE_ROOT (/opt/k6-oracle-d2), ORACLE_MANIFEST_SHA256 (the original bundle's), CASES (default
# "s_dec f_dec s_enc n_enc n_dec"), TIMEOUT_<SEGMENT> (default 300 s, n_dec 1200 s; ENV_PROBE for env-probe),
# JOB_BUDGET_S (default 2580 s, from the script's start).
set -uo pipefail
T_START=$(date +%s)
KIT=${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT=${K6_OUTPUT:?K6_OUTPUT must name the output directory}
VAE=${VAE:-/models/vae}
TOOLS=${TOOLS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
CASES=${CASES:-s_dec f_dec s_enc n_enc n_dec}
BUDGET=${JOB_BUDGET_S:-2580}
OR=/opt/k6-oracle-d2
[[ ${ORACLE_ROOT:-} == "$OR" ]] || { echo "[d2-oracle] ORACLE_ROOT must be $OR" >&2; exit 2; }
[[ ${ORACLE_MANIFEST_SHA256:-} == ead020cbc102a5cb10fdbef43f85dfa63aa14585d1835ce0b7f3bc686eac6d1d ]] \
    || { echo "[d2-oracle] ORACLE_MANIFEST_SHA256 is not the original D2 bundle's" >&2; exit 2; }
[[ ${K6_MEMORY_POLICY:-} == gb10-unified-d2-v2 ]] || { echo "[d2-oracle] needs the runner's D2 memory policy" >&2; exit 2; }
PY=$OR/python/bin/python3.13
GUARD=$TOOLS/d2-allocator-guard.py
mkdir -p $OUT/logs
( while :; do echo "$(date +%s) $(awk '/MemAvailable/ {print $2}' /proc/meminfo)"; sleep 1; done ) > $OUT/memavail.log &
MEM=$!
finish() { kill $MEM 2>/dev/null
    awk 'NR == 1 {a = $2} m == "" || $2 < m {m = $2} END {printf "{\"memavail_start_gib\": %.1f, \"memavail_min_gib\": %.1f, \"peak_used_gib\": %.1f}\n", a / 1048576, m / 1048576, (a - m) / 1048576}' \
        $OUT/memavail.log | tee -a $OUT/summary.jsonl
    chmod -R a+rX $OUT; }   # the container is root; the results are the user's to read
trap finish EXIT
run() { local name=$1; shift; echo "[d2-oracle $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $OUT/logs/$name.log 2>&1; local rc=$?
    tail -n 1 $OUT/logs/$name.log | grep '^{' >> $OUT/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" | tee -a $OUT/steps.jsonl; return $rc; }

run preflight bash -c 'echo "image python: $(python -V 2>&1), arch $(uname -m)"; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.used --format=csv,noheader;
    nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; awk "/MemAvailable/ {print \$2 / 1048576 \" GiB available\"}" /proc/meminfo;
    echo "image env dropped for the oracle: $(env | cut -d= -f1 | grep -E "^(PYTHON|LD_|TORCH|PYTORCH|CUDA|CUBLAS|CUDNN|NVIDIA|NCCL|OMP)" | tr "\n" " ")"'

# each segment's environment: nothing from the image but these
ENV=(env -i PATH=/usr/bin:/bin HOME=/tmp LANG=C.UTF-8 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
     PYTHONPATH=$OR/site-packages PYTORCH_ALLOC_CONF=backend:native,throw_on_cudamalloc_oom:True
     PYTORCH_CUDA_ALLOC_CONF=backend:native,throw_on_cudamalloc_oom:True
     K6_MEMORY_POLICY=$K6_MEMORY_POLICY K6_MEMORY_NONCE=${K6_MEMORY_NONCE:?} K6_JOB_ID=${K6_JOB_ID:?} K6_OUTPUT=$OUT
     K6_SEGMENT_NONCES=${K6_SEGMENT_NONCES:?}
     ${CUDA_VISIBLE_DEVICES:+CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES} ${NVIDIA_VISIBLE_DEVICES:+NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES})

R=$OUT/oracle; mkdir -p $R
for seg in env-probe $CASES; do
    v=TIMEOUT_${seg^^}; v=${v//-/_}; t=${!v:-$([[ $seg == n_dec ]] && echo 1200 || echo 300)}
    if (( $(date +%s) - T_START + t + 30 > BUDGET )); then
        echo "{\"step\": \"segment-$seg\", \"rc\": \"not started\", \"reason\": \"timeout $t s no longer fits JOB_BUDGET_S $BUDGET\"}" | tee -a $OUT/steps.jsonl
        break
    fi
    if [[ $seg == env-probe ]]; then payload=(); else
        payload=(-- $TOOLS/oracle.py --upstream $KIT/upstream --vae $VAE --inputs $KIT/inputs --out $R --cases $seg --device cuda); fi
    run segment-$seg timeout --kill-after=10 $t "${ENV[@]}" $PY -s $GUARD --segment $seg "${payload[@]}"
    rc=$?
    # the reaped record: written only now that the producer is gone, before anything else starts
    "${ENV[@]}" $PY -I $GUARD --record-exit $seg $rc || { echo "[d2-oracle] reaped record for $seg failed"; exit 67; }
    if [[ $seg != env-probe ]]; then
        # fold the process's exit into summary.json (a killed case leaves "started")
        "${ENV[@]}" $PY -I -c 'import json, sys
p, c, rc = sys.argv[1], sys.argv[2], int(sys.argv[3]); d = json.load(open(p)) if __import__("os").path.exists(p) else {"cases": {}}
e = d["cases"].setdefault(c, {"status": "not started"}); e["exit"] = rc
if rc in (124, 137) and e["status"] in ("started", "not started"): e["status"] = "timeout"
json.dump(d, open(p, "w"), indent=1)' $R/summary.json $seg $rc
    fi
    (( rc == 0 )) || { echo "[d2-oracle] segment $seg exited $rc: the sequence ends here"; break; }
done
echo "[d2-oracle $(date +%T)] done: $OUT"
