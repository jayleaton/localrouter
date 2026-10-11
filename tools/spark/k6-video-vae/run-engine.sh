#!/usr/bin/env bash
# The D2 engine job on a DGX Spark (GB10, aarch64, sm_121), inside the arm64 NGC PyTorch image (for its driver mounts
# only: nothing of the image's CUDA toolkit or Python is used). It runs k6v-qual (src/engines/kandinsky6/video_vae/
# qual.zig, built for aarch64-linux-gnu) on this component's sm_121a/sm_120a fatbins, the real checkpoint and the kit's
# inputs, all cases in one process (the engine probes no memory and draws no noise, so cases share nothing), under the
# runner's D2 memory policy (gb10-unified-d2-v1; contract in D2-EVIDENCE.md): every device buffer goes through k6v-qual's
# 24 GiB ledger, and the process writes the allocator evidence (ready receipt, a sample each second, terminal receipt);
# after it is reaped, `k6v-qual --record-exit` writes the reaped record.
# Runs under the shared staging runner (tools/kandinsky6/staging/run-node.py: immutable image ID, --memory/--memory-swap,
# read-only root and kit, models at /models, K6_OUTPUT, deadlines, memory and disk watch), or by hand with the same env.
# Steps: verify the kit's SHA256SUMS (the runner's sealed kit) or MANIFEST.sha256 (build-kit.sh's), then k6v-qual with
# one timeout, under `env -i` with only the runner's variables below (nothing of the image reaches it), then the reaped
# record.
# Outputs under $OUT: allocator-{ready.json, metrics.jsonl, terminal.json, producer-exit.json}, engine/ (the record set: ops.jsonl, blobs/, env.json, summary.json), logs/<step>.log,
# steps.jsonl ({"step","rc","s"}), memavail.log (MemAvailable each second: GB10's memory is the system's) and
# summary.jsonl (each step's last JSON line, then the MemAvailable start / minimum / peak drop).
# Env: the runner's K6_OUTPUT, K6_MEMORY_POLICY, K6_MEMORY_NONCE, K6_JOB_ID, K6_ALLOCATOR_RECEIPT, K6_ALLOCATOR_LOG;
# KIT (default: the directory above this script: bin/k6v-qual, fatbin/{conv,ops}.fatbin, inputs/), VAE (default /models/vae), CASES (default "s_dec,f_dec,s_enc,n_enc,n_dec"), TIMEOUT (default
# 1200 s), FREE_BYTES (default unset: decodes run the unbounded plan; set it to the oracle's recorded mem_get_info to
# match a memory-tiled oracle plan).
set -uo pipefail
KIT=${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VAE=${VAE:-/models/vae}
OUT=${K6_OUTPUT:?K6_OUTPUT must name the output directory}
[[ ${K6_MEMORY_POLICY:-} == gb10-unified-d2-v1 ]] || { echo "[d2-engine] needs the runner's D2 memory policy" >&2; exit 2; }
CASES=${CASES:-s_dec,f_dec,s_enc,n_enc,n_dec}
mkdir -p $OUT/logs
( while :; do echo "$(date +%s) $(awk '/MemAvailable/ {print $2}' /proc/meminfo)"; sleep 1; done ) > $OUT/memavail.log &
MEM=$!
finish() { kill $MEM 2>/dev/null
    awk 'NR == 1 {a = $2} m == "" || $2 < m {m = $2} END {printf "{\"memavail_start_gib\": %.1f, \"memavail_min_gib\": %.1f, \"peak_used_gib\": %.1f}\n", a / 1048576, m / 1048576, (a - m) / 1048576}' \
        $OUT/memavail.log | tee -a $OUT/summary.jsonl
    chmod -R a+rX $OUT; }   # the container is root; the results are the user's to read
trap finish EXIT
run() { local name=$1; shift; echo "[d2-engine $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $OUT/logs/$name.log 2>&1; local rc=$?
    tail -n 1 $OUT/logs/$name.log | grep '^{' >> $OUT/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" | tee -a $OUT/steps.jsonl; return $rc; }

run preflight bash -c 'echo "arch $(uname -m)"; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.used --format=csv,noheader;
    nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; awk "/MemAvailable/ {print \$2 / 1048576 \" GiB available\"}" /proc/meminfo'

M=$([[ -f $KIT/SHA256SUMS ]] && echo SHA256SUMS || echo MANIFEST.sha256)
run manifest bash -c "cd $KIT && sha256sum -c --quiet --strict $M && echo '{\"manifest\": \"$M\", \"files\": '\$(wc -l < $M)'}'" \
    || { echo "[d2-engine] kit manifest check failed: not running the engine"; exit 1; }

# the engine's environment: nothing from the image but these
ENV=(env -i PATH=/usr/bin:/bin HOME=/tmp LANG=C.UTF-8 K6_MEMORY_POLICY=$K6_MEMORY_POLICY K6_MEMORY_NONCE=${K6_MEMORY_NONCE:?}
     K6_JOB_ID=${K6_JOB_ID:?} K6_OUTPUT=$OUT K6_ALLOCATOR_RECEIPT=${K6_ALLOCATOR_RECEIPT:?} K6_ALLOCATOR_LOG=${K6_ALLOCATOR_LOG:?}
     ${CUDA_VISIBLE_DEVICES:+CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES} ${NVIDIA_VISIBLE_DEVICES:+NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES})
run engine timeout --kill-after=10 ${TIMEOUT:-1200} "${ENV[@]}" $KIT/bin/k6v-qual --vae $VAE/diffusion_pytorch_model.safetensors \
    --inputs $KIT/inputs --conv $KIT/fatbin/conv.fatbin --ops $KIT/fatbin/ops.fatbin --out $OUT/engine --cases $CASES \
    ${FREE_BYTES:+--free-bytes $FREE_BYTES}
rc=$?
# the reaped record: written only now that the producer is gone
"${ENV[@]}" $KIT/bin/k6v-qual --record-exit $rc || { echo "[d2-engine] reaped record failed"; exit 67; }
echo "[d2-engine $(date +%T)] done: $OUT"
