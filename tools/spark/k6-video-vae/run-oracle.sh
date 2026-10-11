#!/usr/bin/env bash
# The D2 oracle job on a DGX Spark (GB10, aarch64, sm_121), inside the arm64 NGC PyTorch image, offline. It runs the
# pinned native stack from a sealed oracle root (the shared staging oracle prefix, or pybundle.sh's bundle: CPython 3.13,
# torch 2.14.1+cu130, diffusers 0.40.0), never the image's Python or torch: that interpreter runs under `env -i` with only the variables below, so the image's
# PYTHONPATH, LD_LIBRARY_PATH (its CUDA / cuDNN libraries) and torch overrides (e.g. TORCH_ALLOW_TF32_CUBLAS_OVERRIDE)
# do not reach it; env.json lists the CUDA libraries the process actually mapped.
# Runs under the shared staging runner (tools/kandinsky6/staging/run-node.py: immutable image ID, --memory/--memory-swap,
# read-only root and kit, models at /models, K6_OUTPUT, deadlines, memory and disk watch), or by hand with the same env.
# Steps: verify the oracle stack (every file against its MANIFEST.sha256, that manifest against ORACLE_MANIFEST_SHA256),
# assert the pinned versions, then oracle.py once per case (a fresh process each, so every case sees the native generator
# and mem_get_info state) with a per-case timeout; a case whose timeout no longer fits JOB_BUDGET_S is not started.
# Outputs under $OUT: oracle/ (the record set: ops.jsonl, blobs/, env.json, summary.json), logs/<step>.log,
# steps.jsonl ({"step","rc","s"}), memavail.log (MemAvailable each second: GB10's memory is the system's) and
# summary.jsonl (each step's last JSON line, then the MemAvailable start / minimum / peak drop).
# Env: KIT (default: the directory above this script: inputs/, upstream/, tools/), OUT (default $K6_OUTPUT; required),
# VAE (default /models/vae), ORACLE_ROOT (default /opt/k6-oracle: bin/python3.13 with its own site-packages, or the kit
# builder's layout python/bin/python3.13 + site-packages/), ORACLE_MANIFEST_SHA256 (required), CASES (default
# "s_dec f_dec s_enc n_enc n_dec"), TIMEOUT_<CASE> (default 300 s, n_dec 1200 s), JOB_BUDGET_S (default 2580 s, from
# the script's start), ORACLE_LD_LIBRARY_PATH (default empty: the driver's libcuda comes from the loader cache the
# container toolkit sets).
set -uo pipefail
T_START=$(date +%s)
KIT=${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT=${OUT:-${K6_OUTPUT:?K6_OUTPUT (or OUT) must name the output directory}}
VAE=${VAE:-/models/vae}
TOOLS=${TOOLS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
CASES=${CASES:-s_dec f_dec s_enc n_enc n_dec}
BUDGET=${JOB_BUDGET_S:-2580}
OR=${ORACLE_ROOT:-/opt/k6-oracle}
WANT=${ORACLE_MANIFEST_SHA256:?ORACLE_MANIFEST_SHA256 must pin the oracle stack}
if [[ -x $OR/bin/python3.13 ]]; then PY=$OR/bin/python3.13; PYPATH=; else PY=$OR/python/bin/python3.13; PYPATH=$OR/site-packages; fi
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

# the oracle stack: every file against its manifest, the manifest against the sealed hash
run manifest bash -c "cd $OR && [[ \$(sha256sum MANIFEST.sha256 | cut -d' ' -f1) == $WANT ]] && sha256sum -c --quiet --strict MANIFEST.sha256 \
    && echo '{\"manifest\": \"ok\", \"root\": \"$OR\", \"sha256\": \"$WANT\", \"files\": '\$(wc -l < MANIFEST.sha256)'}'" \
    || { echo "[d2-oracle] oracle stack manifest check failed: not running the oracle"; exit 1; }

# the oracle's environment: nothing from the image but these
ENV=(env -i PATH=/usr/bin:/bin HOME=/tmp LANG=C.UTF-8 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
     PYTHONPATH=$PYPATH LD_LIBRARY_PATH=${ORACLE_LD_LIBRARY_PATH:-}
     ${CUDA_VISIBLE_DEVICES:+CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES} ${NVIDIA_VISIBLE_DEVICES:+NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES})
run bundle-probe "${ENV[@]}" $PY -s -c 'import json, sys, torch, diffusers, torchvision
print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__, "torch_file": torch.__file__, "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(), "diffusers": diffusers.__version__, "torchvision": torchvision.__version__,
    "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)}))
import accelerate, numpy, safetensors   # the pinned native stack (upstream uv.lock, extra consumer), exactly
got = {"torch": torch.__version__, "torchvision": torchvision.__version__, "diffusers": diffusers.__version__,
       "accelerate": accelerate.__version__, "numpy": numpy.__version__, "safetensors": safetensors.__version__,
       "python": sys.version.split()[0].rsplit(".", 1)[0]}
want = {"torch": "2.14.1+cu130", "torchvision": "0.29.1+cu130", "diffusers": "0.40.0", "accelerate": "1.15.0",
        "numpy": "2.5.3", "safetensors": "0.8.0", "python": "3.13"}
bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
sys.exit(f"oracle stack differs from the pins: {bad}" if bad else 0)' \
    || { echo "[d2-oracle] the oracle stack does not start or differs from the pins: not running the oracle"; exit 1; }

R=$OUT/oracle; mkdir -p $R
for c in $CASES; do
    v=TIMEOUT_${c^^}; t=${!v:-$([[ $c == n_dec ]] && echo 1200 || echo 300)}
    if (( $(date +%s) - T_START + t + 30 > BUDGET )); then
        echo "{\"step\": \"oracle-$c\", \"rc\": \"not started\", \"reason\": \"timeout $t s no longer fits JOB_BUDGET_S $BUDGET\"}" | tee -a $OUT/steps.jsonl
        continue
    fi
    run oracle-$c timeout --kill-after=30 $t "${ENV[@]}" $PY -s $TOOLS/oracle.py --upstream $KIT/upstream --vae $VAE \
        --inputs $KIT/inputs --out $R --cases $c --device cuda
    rc=$?
    # fold the process's exit into summary.json (a killed case leaves "started")
    "${ENV[@]}" $PY -I -c 'import json, sys
p, c, rc = sys.argv[1], sys.argv[2], int(sys.argv[3]); d = json.load(open(p)) if __import__("os").path.exists(p) else {"cases": {}}
e = d["cases"].setdefault(c, {"status": "not started"}); e["exit"] = rc
if rc in (124, 137) and e["status"] in ("started", "not started"): e["status"] = "timeout"
json.dump(d, open(p, "w"), indent=1)' $R/summary.json $c $rc
done
echo "[d2-oracle $(date +%T)] done: $OUT"
