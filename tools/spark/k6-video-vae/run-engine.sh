#!/usr/bin/env bash
# The D2 engine job on a DGX Spark (GB10, aarch64, sm_121), inside the arm64 NGC PyTorch image (for its driver mounts
# only: nothing of the image's CUDA toolkit or Python is used). It runs k6v-qual (src/engines/kandinsky6/video_vae/
# qual.zig, built for aarch64-linux-gnu) on this component's sm_121a/sm_120a fatbins, the real checkpoint and the kit's
# inputs, all cases in one process (the engine probes no memory and draws no noise, so cases share nothing).
# Steps: verify the kit's MANIFEST.sha256 (binary, fatbins, inputs), then k6v-qual with one timeout.
# Outputs under $OUT: engine/ (the record set: ops.jsonl, blobs/, env.json, summary.json), logs/<step>.log,
# steps.jsonl ({"step","rc","s"}), memavail.log (MemAvailable each second: GB10's memory is the system's) and
# summary.jsonl (each step's last JSON line, then the MemAvailable start / minimum / peak drop).
# Env: KIT (default /workspace/d2-kit: bin/k6v-qual, fatbin/{conv,ops}.fatbin, inputs/, vae/), VAE (default $KIT/vae),
# OUT (default $KIT/results/engine-<utc>), CASES (default "s_dec,f_dec,s_enc,n_enc,n_dec"), TIMEOUT (default 1200 s),
# FREE_BYTES (default unset: decodes run the unbounded plan; set it to the oracle's recorded mem_get_info to match a
# memory-tiled oracle plan).
set -uo pipefail
KIT=${KIT:-/workspace/d2-kit}
VAE=${VAE:-$KIT/vae}
OUT=${OUT:-$KIT/results/engine-$(date -u +%Y%m%d-%H%M%S)}
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

run manifest bash -c "cd $KIT && sha256sum -c --quiet --strict MANIFEST.sha256 && echo '{\"manifest\": \"ok\", \"files\": '\$(wc -l < MANIFEST.sha256)'}'" \
    || { echo "[d2-engine] kit manifest check failed: not running the engine"; exit 1; }

run engine timeout --kill-after=30 ${TIMEOUT:-1200} $KIT/bin/k6v-qual --vae $VAE/diffusion_pytorch_model.safetensors \
    --inputs $KIT/inputs --conv $KIT/fatbin/conv.fatbin --ops $KIT/fatbin/ops.fatbin --out $OUT/engine --cases $CASES \
    ${FREE_BYTES:+--free-bytes $FREE_BYTES}
echo "[d2-engine $(date +%T)] done: $OUT"
