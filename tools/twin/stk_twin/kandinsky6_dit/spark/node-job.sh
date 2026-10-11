#!/usr/bin/env bash
# One Kandinsky-6 DiT GPU job on one DGX Spark node, from an immutable kit (kit.sh), in a window the owner granted:
#   DEADLINE=<unix> WORK=<dir> KIT=<tarball> KIT_SHA=<sha256> VENV=<venv dir> bash node-job.sh <job> [--preflight]
# Jobs (each step is one python process; a step's max minutes is its timeout, its output budget is declared up front):
#   smoke   import (twin + pinned upstream, every module from the kit), kernels (load the three fatbins, digests ==
#           MANIFEST), test_ops_small (`test_ops --small --fp32-linear tf32_rna`)
#   twin    test_ops_<mode> at the checkpoint's widths, for mode in tf32_rna tf32_truncate fp32, one after another
#   ops     gate_ops_tf32_rna: every `gate --phase ops` site; gate_fp32_<mode>: its fp32/* sites for the other modes
#   blocks  gate_blocks_<mode>: `gate --phase blocks --fp32-linear <mode>` for each mode
# --preflight checks the kit sha, the venv (python >= 3.13, torch 2.14.1 built for CUDA 13.0, CUDA available, device
# capability 12.1), free disk, MemAvailable, the GPU lock and other GPU compute apps; prints ok / FAILED and exits.
#
# Hard contract (each a check; the job refuses to start, or a step refuses to start, otherwise):
#   kit     sha256(KIT) == KIT_SHA before anything is extracted; extracted once into $WORK/kit-<sha12>, then chmod -R a-w
#           and never modified; every run re-checks every file against the kit's MANIFEST.txt (and that MANIFEST
#           against the tarball's) before each step. Python runs with the kit as its read-only cwd, no bytecode written.
#   time    STOP_NEW = DEADLINE - 20 min: no step starts at or after it, and none starts unless its max minutes end by
#           CLEANUP = DEADLINE - 10 min (a step's timeout is also capped at CLEANUP). At CLEANUP the guard terminates our
#           step's process group (KILL 10 s later); at DEADLINE it KILLs the group and stops this script.
#   memory  bounded and measured for the whole job, <= 100 GiB total. The job must run in its own cgroup v2 with
#           memory.max <= 96 GiB (`docker run --memory 96g`, or `systemd-run --scope -p MemoryMax=96G` on the host);
#           preflight refuses a shared or unlimited cgroup. Every process this job starts (guard, tar, python, page
#           cache) is charged there, so memory.current is the job's host-side total and memory.peak its high-water
#           mark. Device allocations made through the driver are not cgroup-charged, so the guard adds GPU memory
#           as nvidia-smi's device-wide memory.used (conservative: every app on the device counts against us),
#           sampled every second. FAIL CLOSED: the query must exit 0 within 10 s with every row numeric; [N/A],
#           "Not Supported", an empty or partly numeric answer, a timeout or an error fails preflight, refuses a step
#           and kills a running step. No estimate stands in for it. Outside the cgroup, HELPER_RESERVE_GIB (2: the
#           `docker run` / `systemd-run` client and the login shell) is counted as a fixed reserve. The total is
#           cgroup + GPU + reserve <= TOTAL_GIB (100): the guard kills the step above it, and a step starts only if
#           the total + its declared peak fits. MemAvailable >= MIN_GIB (8) stays as a supplemental headroom guard.
#           Logged: memavail.log (every sample), guard.log (actions).
#   disk    a step does not start if du($WORK) + its output budget exceeds 200 GB, or the filesystem lacks the budget
#           + 5 GB; du before and after every step is recorded.
#   GPU     one heavy job per node: an exclusive flock on $WORK/.gpu.lock, held for the whole run (refused if another
#           k6 job holds it). Other GPU compute apps are reported (preflight, env.json, cleanup), never touched.
#   scope   no docker inside the job, no image builds, no network: proxies unset, NCCL/UCX/MPI pinned to loopback with IB off, HF/pip
#           offline. Only $WORK is written (TMPDIR and every cache are under it); only our own process groups are killed.
#   oracle  NVIDIA_TF32_OVERRIDE, TORCH_ALLOW_TF32* and CUBLASLT_LOG_* must not be set (refused); this script never
#           sets them (gate.py sets CUBLASLT_LOG_* itself where it wants them).
# Optional env: MIN_GIB (8), SEED (0, passed to test_ops / gate), GATE_MAX_GIB (90, gate --max-gib), STEP_MAX_MIN (the
# max minutes of every twin / ops / blocks step; defaults 20 / 40 (+10 per further fp32/* form) / 20), MODES (a subset
# of "tf32_rna tf32_truncate fp32", in order: splits a job into discrete runs; ops runs every site under the first).
# Output: $WORK/results/<job>-<UTC>/: env.json, nvidia-smi-q.txt, preflight.log, job.log, steps.tsv, memavail.log,
# guard.log, <step>/{log,exit_code,<receipts>}, SUMMARY.txt; then $WORK/results/<job>-<UTC>.tar.gz and its sha256.
set -uo pipefail
umask 022
export LC_ALL=C

usage() { echo "usage: DEADLINE=<unix> WORK=<dir> KIT=<tarball> KIT_SHA=<sha256> VENV=<venv> bash node-job.sh {smoke|twin|ops|blocks} [--preflight]" >&2; exit 2; }
JOB=${1:-}
OPT=${2:-}
case $JOB in smoke | twin | ops | blocks) ;; *) usage ;; esac
[[ -z $OPT || $OPT == --preflight ]] || usage
(( $# <= 2 )) || usage
PREFLIGHT_ONLY=0
[[ $OPT == --preflight ]] && PREFLIGHT_ONLY=1

die() { echo "node-job: $*" >&2; exit 1; }
: "${WORK:?WORK (the only directory this job writes) is required}"
: "${KIT:?KIT (the kit tarball) is required}"
: "${KIT_SHA:?KIT_SHA (its sha256) is required}"
: "${VENV:?VENV (the python 3.13 + torch 2.14.1 cu130 venv) is required}"
[[ $WORK == /* && $WORK != / && $WORK != *..* ]] || die "WORK must be an absolute path other than / (got $WORK)"
[[ $KIT_SHA =~ ^[0-9a-f]{64}$ ]] || die "KIT_SHA must be 64 lowercase hex digits"
if (( PREFLIGHT_ONLY == 0 )); then
    : "${DEADLINE:?DEADLINE (unix seconds the window ends) is required}"
fi
abspath() { if [[ $1 == /* ]]; then echo "$1"; else echo "$PWD/$1"; fi; }
KIT=$(abspath "$KIT")
VENV=$(abspath "$VENV")
[[ -z ${DEADLINE:-} || ${DEADLINE:-} =~ ^[0-9]+$ ]] || die "DEADLINE must be unix seconds"
MIN_GIB=${MIN_GIB:-8}
SEED=${SEED:-0}
GATE_MAX_GIB=${GATE_MAX_GIB:-90}
for v in MIN_GIB SEED GATE_MAX_GIB; do [[ ${!v} =~ ^[0-9]+$ ]] || die "$v must be a non-negative integer"; done
[[ -z ${STEP_MAX_MIN:-} || ${STEP_MAX_MIN:-} =~ ^[1-9][0-9]*$ ]] || die "STEP_MAX_MIN must be a positive integer"
MODES=${MODES:-tf32_rna tf32_truncate fp32}   # a subset splits twin / ops / blocks into discrete, resumable jobs
for m in $MODES; do [[ " tf32_rna tf32_truncate fp32 " == *" $m "* ]] || die "MODES: unknown Fp32Linear form $m"; done
[[ -n ${MODES// /} ]] || die "MODES is empty"

# contract constants (not overridable)
TOTAL_GIB=100   # cgroup (memory.current) + GPU memory + helper reserve: the contract's node total for this job
HELPER_RESERVE_GIB=2   # processes outside the job's cgroup: the docker / systemd-run client and the login shell
CG_MAX_GIB=96   # the job's cgroup memory.max must be at most this (leaves room for the GPU term's own accounting)
DISK_GB=200
STOP_NEW_S=1200
CLEANUP_S=600
GIB_KIB=1048576
MIN_KIB=$(( MIN_GIB * GIB_KIB ))
MAX_KIB=$(( TOTAL_GIB * GIB_KIB ))
SHA12=${KIT_SHA:0:12}
KD=$WORK/kit-$SHA12
LOCK=$WORK/.gpu.lock
# the venv's python, the image's /usr/local/bin/python, or the shared prefix's bin/python3.13
PY=$VENV/bin/python
[[ -x $PY ]] || PY=$VENV/bin/python3.13
TWIN=$KD/repo/tools/twin
if [[ -n ${DEADLINE:-} ]]; then
    STOP_NEW=$(( DEADLINE - STOP_NEW_S ))
    CLEANUP_AT=$(( DEADLINE - CLEANUP_S ))
fi

# ---- environment: offline, loopback-only, caches under $WORK, oracle settings refused if preset
ENV_BAD=""
for v in $(compgen -e); do
    case $v in NVIDIA_TF32_OVERRIDE | TORCH_ALLOW_TF32* | CUBLASLT_LOG_*) ENV_BAD+="$v=${!v} " ;; esac
done
unset http_proxy https_proxy ftp_proxy all_proxy no_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY NO_PROXY
export NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo NCCL_P2P_DISABLE=1
export UCX_TLS=tcp,self UCX_NET_DEVICES=lo OMPI_MCA_btl=self,tcp OMPI_MCA_btl_tcp_if_include=lo OMPI_MCA_pml=ob1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1 PIP_NO_INPUT=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 PYTHONPATH=$TWIN
export TMPDIR=$WORK/tmp XDG_CACHE_HOME=$WORK/cache TORCH_HOME=$WORK/cache/torch TRITON_CACHE_DIR=$WORK/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$WORK/cache/inductor CUDA_CACHE_PATH=$WORK/cache/nv
export STK_K6_SRC=$KD/kandinsky6-upstream STK_K6_DIT_FATBINS=$KD/fatbins KIT_SHA
unset STK_K6_CONFIG   # the twin's built-in Pro-distill config, never a file from elsewhere

avail_kib() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
JOB_BASE=$(avail_kib)   # MemAvailable before this job does anything (supplemental and fallback accounting)
# this job's cgroup v2 directory (the container's own, or the systemd scope it was started in)
CG=/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup 2> /dev/null)
cg_kib() { local v; v=$(cat "$CG/$1" 2> /dev/null) || { echo -1; return; }; [[ $v =~ ^[0-9]+$ ]] && echo $(( v / 1024 )) || echo -1; }
cg_max_kib() { local v; v=$(cat "$CG/memory.max" 2> /dev/null) || { echo -1; return; }; [[ $v == max ]] && echo -2 || echo $(( v / 1024 )); }
# device-wide GPU memory.used in KiB, or -1 unless nvidia-smi exits 0 within 10 s and every row (blank rows included) is
# numeric (fail closed). The trailing rc line keeps $(...) from dropping trailing blank rows.
gpu_kib() {
    local out rc
    out=$(timeout 10 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null; echo "rc=$?")
    rc=${out##*rc=}; out=${out%rc=*}
    [[ $rc == 0 ]] || { echo -1; return; }
    awk 'BEGIN {n = 0; bad = 0} {gsub(/^[ \t]+|[ \t]+$/, ""); if ($0 ~ /^[0-9]+$/) {s += $0; n++} else bad = 1}
         END {if (n > 0 && !bad) print s * 1024; else print -1}' <<< "${out%$'\n'}"
}
gib() { awk -v k="$1" 'BEGIN {printf "%.1f", k / 1048576}'; }
du_bytes() { du -sxb "$WORK" 2> /dev/null | cut -f1; }
df_bytes() { df -B1 --output=avail "$WORK" | tail -1 | tr -dc 0-9; }
gpu_apps() { timeout 30 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2> /dev/null | tr '\n' ';'; }

# ---- the steps: "name max_min out_gb peak_gib fatal" (estimates in README.md; max_min is the hard timeout)
steps() {
    local m
    case $JOB in
    smoke)
        echo "import 3 1 4 1"
        echo "kernels 3 1 4 1"
        echo "test_ops_small 15 1 8 1" ;;
    twin) for m in $MODES; do echo "test_ops_$m ${STEP_MAX_MIN:-20} 1 24 0"; done ;;
    # only the fp32/* sites depend on the Fp32Linear form: every site once (tf32_rna), then fp32/* for the others
    ops) local first=1
        for m in $MODES; do
            if (( first )); then echo "gate_ops_$m ${STEP_MAX_MIN:-40} 5 24 0"; first=0
            else echo "gate_fp32_$m ${STEP_MAX_MIN:-10} 1 16 0"; fi
        done ;;
    blocks) for m in $MODES; do echo "gate_blocks_$m ${STEP_MAX_MIN:-20} 10 32 0"; done ;;
    esac
}
total_out_gb() { steps | awk '{s += $3} END {print s + 1}'; }   # + 1 GB: the extracted kit and the results tarball
max_peak_gib() { steps | awk '$4 > p {p = $4} END {print p + 0}'; }
first_max_min() { steps | awk 'NR == 1 {print $2}'; }

IMPORT_PY='
import importlib, importlib.util, json, sys
from pathlib import Path
kd = Path(sys.argv[1]).resolve()
import torch
from stk_twin.kandinsky6_dit import block, kernels, native, test_ops
mods = ["block", "kernels", "native", "test_ops"]
if importlib.util.find_spec("stk_twin.kandinsky6_dit.gate") is not None:
    importlib.import_module("stk_twin.kandinsky6_dit.gate")
    mods.append("gate")
u = native.upstream()
assert u.src == (kd / "kandinsky6-upstream").resolve(), f"upstream from {u.src}, not the kit"
stray = [f"{n}: {m.__file__}" for n, m in sorted(sys.modules.items())
         if n.split(".")[0] in ("stk_twin", "kandinsky") and getattr(m, "__file__", None)
         and not Path(m.__file__).resolve().is_relative_to(kd)]
assert not stray, f"modules from outside the kit: {stray}"
optional = {n: importlib.util.find_spec(n) is not None for n in ("flash_attn", "flash_attn_interface", "sageattention")}
print(json.dumps({"ok": True, "twin": mods, "upstream": str(u.src), "torch": torch.__version__,
                  "optional_attention_installed": optional,
                  "loaded": sorted(n for n in sys.modules if n.split(".")[0] in ("stk_twin", "kandinsky"))}, indent=1))
'
KERNELS_PY='
import json, sys
from pathlib import Path
kd, out = Path(sys.argv[1]), sys.argv[2]
import torch
from stk_twin.kandinsky6_dit import kernels as K
torch.cuda.init()
k = K.Kernels(fatbins=str(kd / "fatbins"))
man = {}
for line in (kd / "MANIFEST.txt").read_text().splitlines():
    if line.startswith("# fatbin: "):
        name, sha = line.split()[2:4]
        man[name] = sha
want = {n: man.get(f"k6_dit_{n}.fatbin") for n in K.IMAGES}
ok = k.digests == want
r = {"ok": ok, "loaded": k.digests, "manifest": want, "device": torch.cuda.get_device_name(),
     "capability": list(torch.cuda.get_device_capability())}
Path(out).write_text(json.dumps(r, indent=1))
print(json.dumps(r, indent=1))
sys.exit(0 if ok else 1)
'
VENV_PY='
import importlib.util, sys
ok = True
def say(good, msg):
    global ok
    ok = ok and good
    print(("  ok  " if good else "  BAD ") + msg)
v = sys.version_info
say(v >= (3, 13), f"python {sys.version.split()[0]} (>= 3.13) at {sys.executable}")
try:
    import torch
except Exception as e:
    say(False, f"import torch: {e!r}")
    sys.exit(1)
say(torch.__version__.split("+")[0] == "2.14.1", f"torch {torch.__version__} (2.14.1)")
say((torch.version.cuda or "").startswith("13.0"), f"torch built for CUDA {torch.version.cuda} (13.0)")
avail = torch.cuda.is_available()
say(avail, f"torch.cuda.is_available() {avail}")
if avail:
    cap = torch.cuda.get_device_capability()
    say(tuple(cap) == (12, 1), f"device {torch.cuda.get_device_name()} capability {cap[0]}.{cap[1]} (12.1)")
for n in ("flash_attn", "flash_attn_interface", "sageattention"):
    if importlib.util.find_spec(n) is not None:
        print(f"  note optional {n} is installed (the native reference uses sdpa; not required)")
sys.exit(0 if ok else 1)
'
ENV_PY='
import json, os, platform, socket, subprocess, sys
out, kd, job, smi = sys.argv[1:5]
def sh(*c):
    try:
        return subprocess.run(c, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"error: {e!r}"
summary = {}
keys = ("Driver Version", "CUDA Version", "Product Name", "Product Architecture", "Persistence Mode",
        "Addressing Mode", "Compute Mode", "Bus Id", "GPU Current Temp", "Power Draw")
for line in open(smi, errors="replace"):
    k, _, val = line.partition(":")
    k = k.strip()
    if k in keys and k not in summary:
        summary[k] = val.strip()
import torch
dev = {}
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    dev = {"name": p.name, "capability": [p.major, p.minor], "total_memory": p.total_memory,
           "multi_processor_count": p.multi_processor_count}
manifest = [l.strip() for l in open(os.path.join(kd, "MANIFEST.txt")) if l.startswith("#")]
names = [k for k in os.environ if k.startswith(("NCCL_", "UCX_", "OMPI_", "CUDA_", "CUBLAS", "TORCH", "NVIDIA_",
                                                "PYTHON", "STK_", "HF_", "TRANSFORMERS_", "PIP_"))]
json.dump({
    "job": job, "hostname": socket.gethostname(), "kernel": " ".join(platform.uname()),
    "kit_sha256": os.environ["KIT_SHA"], "kit_dir": kd, "kit_manifest_header": manifest,
    "driver": sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
    "gpu": sh("nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,power.limit",
              "--format=csv,noheader"),
    "nvidia_smi_q_summary": summary,
    "gpu_compute_apps": sh("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"),
    "python": sys.version, "python_executable": sys.executable, "torch": torch.__version__,
    "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "device": dev,
    "env": {k: os.environ[k] for k in sorted(names)},
}, open(out, "w"), indent=1)
'

# the argv of a step, in CMD
step_cmd() {
    local s=$1 out=$2 m
    case $s in
    import) CMD=("$PY" -s -c "$IMPORT_PY" "$KD") ;;
    kernels) CMD=("$PY" -s -c "$KERNELS_PY" "$KD" "$out/kernels.json") ;;
    test_ops_small)
        CMD=("$PY" -s -m stk_twin.kandinsky6_dit.test_ops --small --fp32-linear tf32_rna --seed "$SEED"
             --fatbins "$KD/fatbins" --out "$out/receipt.json") ;;
    test_ops_*)
        m=${s#test_ops_}
        CMD=("$PY" -s -m stk_twin.kandinsky6_dit.test_ops --fp32-linear "$m" --seed "$SEED"
             --fatbins "$KD/fatbins" --out "$out/receipt.json") ;;
    gate_ops_* | gate_blocks_* | gate_fp32_*)
        m=${s#gate_*_}
        local phase=${s#gate_} sites=()
        phase=${phase%%_*}
        [[ $phase == fp32 ]] && phase=ops sites=(--sites 'fp32/*')
        CMD=("$PY" -s -m stk_twin.kandinsky6_dit.gate --out "$out" --fatbins "$KD/fatbins" --phase "$phase"
             --fp32-linear "$m" --seed "$SEED" --max-gib "$GATE_MAX_GIB" "${sites[@]}") ;;
    *) return 1 ;;
    esac
}

kit_has_gate() {
    if [[ -d $KD ]]; then
        [[ -f $KD/repo/tools/twin/stk_twin/kandinsky6_dit/gate.py ]]
    else
        tar -tzf "$KIT" ./repo/tools/twin/stk_twin/kandinsky6_dit/gate.py > /dev/null 2>&1
    fi
}

# every file of the extracted kit against its MANIFEST, that MANIFEST against the tarball's, and nothing extra
verify_kit() {
    local d=$1 files listed
    [[ -f $d/MANIFEST.txt ]] || { echo "kit: $d has no MANIFEST.txt"; return 1; }
    tar -xzOf "$KIT" ./MANIFEST.txt | cmp -s - "$d/MANIFEST.txt" || { echo "kit: $d/MANIFEST.txt differs from the tarball's"; return 1; }
    (cd "$d" && grep -v '^#' MANIFEST.txt | sha256sum -c --quiet --strict -) || { echo "kit: $d differs from its MANIFEST"; return 1; }
    files=$(cd "$d" && find . ! -type d ! -path ./MANIFEST.txt | wc -l)
    listed=$(grep -vc '^#' "$d/MANIFEST.txt")
    (( files == listed )) || { echo "kit: $d holds $files files, MANIFEST lists $listed"; return 1; }
}

kit_sha_ok() {
    local got
    [[ -f $KIT && -r $KIT ]] || { echo "kit: $KIT is not a readable file"; return 1; }
    got=$(sha256sum "$KIT" | cut -d' ' -f1)
    [[ $got == "$KIT_SHA" ]] || { echo "kit: sha256 $got != KIT_SHA $KIT_SHA"; return 1; }
}

# ---- preflight: report every check, then ok / FAILED
preflight() {
    local bad=0 t used free need avail apps left first
    echo "preflight $JOB on $(hostname) at $(date -u +%FT%TZ)"
    for t in flock setsid timeout sha256sum tar gzip du df awk cmp nvidia-smi; do
        command -v "$t" > /dev/null || { echo "FAIL missing tool: $t"; bad=1; }
    done
    mkdir -p "$WORK" "$WORK/tmp" "$WORK/cache" "$WORK/results" 2> /dev/null && [[ -w $WORK ]] ||
        { echo "FAIL WORK $WORK is not writable"; bad=1; }
    if kit_sha_ok; then echo "ok   kit sha256 $KIT_SHA"; else echo "FAIL kit sha256"; bad=1; fi
    if [[ -d $KD ]]; then
        if verify_kit "$KD"; then echo "ok   extracted kit $KD matches its MANIFEST"; else echo "FAIL extracted kit $KD"; bad=1; fi
    fi
    if [[ $JOB == ops || $JOB == blocks ]]; then
        if kit_has_gate; then echo "ok   kit has kandinsky6_dit/gate.py"; else echo "FAIL the kit has no kandinsky6_dit/gate.py ($JOB needs it)"; bad=1; fi
    fi
    if [[ -z $ENV_BAD ]]; then
        echo "ok   no TF32 / cuBLASLt log overrides in the environment"
    else
        echo "FAIL oracle settings preset in the environment: $ENV_BAD"; bad=1
    fi
    if [[ -x $PY ]]; then
        echo "venv $VENV:"
        (cd / && timeout 300 "$PY" -I -c "$VENV_PY") || { echo "FAIL venv"; bad=1; }
    else
        echo "FAIL no python at $PY"; bad=1
    fi
    used=$(du_bytes); free=$(df_bytes); need=$(( $(total_out_gb) * 1000000000 ))
    echo "disk: WORK uses $(( used / 1000000000 )) GB, $(( free / 1000000000 )) GB free; $JOB declares $(total_out_gb) GB"
    (( used + need <= DISK_GB * 1000000000 )) || { echo "FAIL WORK usage + budget over $DISK_GB GB"; bad=1; }
    (( free >= need + 5000000000 )) || { echo "FAIL filesystem has under budget + 5 GB free"; bad=1; }
    local cgmax cgcur gk
    cgmax=$(cg_max_kib); cgcur=$(cg_kib memory.current)
    if (( cgmax == -1 || cgcur < 0 )); then
        echo "FAIL no cgroup v2 memory accounting at $CG (memory.max / memory.current unreadable)"; bad=1
    elif (( cgmax == -2 || cgmax > CG_MAX_GIB * GIB_KIB )); then
        echo "FAIL cgroup $CG memory.max is $( (( cgmax == -2 )) && echo unlimited || echo "$(gib "$cgmax") GiB"): run in its own cgroup with memory.max <= $CG_MAX_GIB GiB"; bad=1
    else
        echo "ok   cgroup $CG: memory.max $(gib "$cgmax") GiB, memory.current $(gib "$cgcur") GiB, memory.peak $( p=$(cg_kib memory.peak); (( p < 0 )) && echo n/a || echo "$(gib "$p") GiB")"
    fi
    gk=$(gpu_kib)
    if (( gk >= 0 )); then
        echo "ok   GPU memory accounting: nvidia-smi memory.used, $(gib "$gk") GiB used now (+ helper reserve $HELPER_RESERVE_GIB GiB)"
    else
        echo "FAIL GPU memory: nvidia-smi gave no complete numeric memory.used ([N/A], unsupported, timeout or error): the job total cannot be bounded"; bad=1
    fi
    if [[ -n ${ORACLE_MANIFEST:-} ]]; then
        if [[ $(sha256sum < "$ORACLE_MANIFEST" | cut -d' ' -f1) == "${ORACLE_MANIFEST_SHA256:-}" ]]; then
            echo "ok   oracle prefix MANIFEST $ORACLE_MANIFEST sha256 $ORACLE_MANIFEST_SHA256"
            if [[ ${ORACLE_VERIFY:-1} == 1 ]]; then
                if (cd "$VENV" && sha256sum -c --quiet --strict "$ORACLE_MANIFEST") > "$WORK/tmp/oracle-verify.log" 2>&1; then
                    echo "ok   every oracle prefix file matches its MANIFEST"
                else
                    echo "FAIL oracle prefix files differ from the MANIFEST ($WORK/tmp/oracle-verify.log)"; bad=1
                fi
            fi
        else
            echo "FAIL oracle prefix MANIFEST sha256 is not ORACLE_MANIFEST_SHA256"; bad=1
        fi
    fi
    avail=$(avail_kib)
    echo "memory: MemAvailable $(gib "$avail") GiB; $JOB's largest declared peak $(max_peak_gib) GiB + MIN_GIB $MIN_GIB"
    (( avail >= ($(max_peak_gib) + MIN_GIB) * GIB_KIB )) || { echo "FAIL MemAvailable too low"; bad=1; }
    if [[ ${HOLD_LOCK:-0} == 1 ]]; then
        echo "ok   GPU lock $LOCK held by this job"
    elif flock -n "$LOCK" true 2> /dev/null; then
        echo "ok   GPU lock $LOCK free"
    else
        echo "FAIL GPU lock $LOCK is held by another k6 job"; bad=1
    fi
    apps=$(gpu_apps)
    echo "GPU compute apps (report only, never touched): ${apps:-none}"
    if [[ -n ${DEADLINE:-} ]]; then
        left=$(( STOP_NEW - $(date +%s) )); first=$(first_max_min)
        echo "time: $(( left / 60 )) min until no new work (DEADLINE - 20 min); first step needs up to $first min"
        (( $(date +%s) + first * 60 <= CLEANUP_AT && left > 0 )) || { echo "FAIL not enough time before the deadline"; bad=1; }
    fi
    echo "preflight $JOB: $([[ $bad == 0 ]] && echo ok || echo FAILED)"
    return $bad
}

if (( PREFLIGHT_ONLY )); then
    preflight
    exit
fi

# ================================================================================================ the run
mkdir -p "$WORK/results" "$WORK/tmp" "$WORK/cache" || die "cannot create directories under $WORK"
# the lock, held until this script exits (children inherit it: a surviving child keeps the GPU marked busy)
exec 9>> "$LOCK" || die "cannot open $LOCK"
flock -n 9 || die "$LOCK is held by another k6 job: not starting"
START=$(date +%s)
RUN=$JOB-$(date -u +%Y%m%d-%H%M%S)
R=$WORK/results/$RUN
mkdir "$R" || die "cannot create $R"
log() { echo "[k6 $JOB $(date -u +%T)] $*" | tee -a "$R/job.log"; }
printf 'step\tstatus\trc\tminutes\tpeak_gib\tmax_min\tdu_before_gb\tdu_after_gb\n' > "$R/steps.tsv"
GUARD=""
CUR_PID=""
CUR_STEP=""
RESULT="incomplete"

record() {   # name status rc minutes peak_gib max_min du_before du_after
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$R/steps.tsv"
}

finalize() {
    local tarball sha
    trap - EXIT
    trap '' INT TERM   # finish packing whatever arrives
    if [[ -n $CUR_PID ]]; then
        kill -TERM -- "-$CUR_PID" 2> /dev/null && sleep 5
        kill -KILL -- "-$CUR_PID" 2> /dev/null
        record "$CUR_STEP" "killed: job stopped" "-" "-" "-" "-" "-" "-"
        log "$CUR_STEP: process group $CUR_PID killed (job stopped)"
    fi
    if [[ -n $GUARD ]]; then kill "$GUARD" 2> /dev/null; wait "$GUARD" 2> /dev/null; fi
    log "cleanup: du WORK $(( $(du_bytes) / 1000000000 )) GB; GPU compute apps: $(gpu_apps)"
    {
        echo "k6 DiT node job: $JOB"
        echo "host: $(hostname)"
        echo "kit: $KIT_SHA"
        echo "started: $(date -u -d "@$START" +%FT%TZ)  ended: $(date -u +%FT%TZ)  deadline: $(date -u -d "@$DEADLINE" +%FT%TZ)"
        echo "result: $RESULT"
        echo "memory: cgroup $CG memory.peak $( p=$(cg_kib memory.peak); (( p < 0 )) && echo n/a || echo "$(gib "$p") GiB"), memory.max $(gib "$(cg_max_kib)") GiB; GPU nvidia-smi memory.used; helper reserve $HELPER_RESERVE_GIB GiB"
        echo
        column -t -s $'\t' "$R/steps.tsv" 2> /dev/null || cat "$R/steps.tsv"
        echo
        echo "guard actions:"
        sed 's/^/  /' "$R/guard.log" 2> /dev/null
    } > "$R/SUMMARY.txt"
    rm -f "$R"/.step "$R"/.peak.* "$R"/.killed.*
    tarball=$WORK/results/$RUN.tar.gz
    if tar -C "$WORK/results" --sort=name -czf "$tarball" "$RUN"; then
        sha=$(sha256sum "$tarball" | cut -d' ' -f1)
        echo "$sha  $RUN.tar.gz" > "$tarball.sha256"
        cat "$R/SUMMARY.txt"
        echo "results: $tarball"
        echo "RESULTS_SHA=$sha"
    else
        echo "node-job: packing $R failed; the directory is intact" >&2
    fi
}

# the guard: one sample a second; kills only our current step's process group

guard() {
    local now avail pid name base cur="" peak=0 cgk gpu=0 drop use reason tick=0 term_at=0
    while :; do
        now=$(date +%s); avail=$(avail_kib); pid=""; name=""; base=0
        [[ -f $R/.step ]] && read -r pid name base < "$R/.step"
        drop=0; use=0
        cgk=$(cg_kib memory.current)
        gpu=$(gpu_kib)
        drop=$(( JOB_BASE - avail )); (( drop < 0 )) && drop=0
        use=$(( (cgk < 0 ? 0 : cgk) + (gpu < 0 ? 0 : gpu) + HELPER_RESERVE_GIB * GIB_KIB ))
        if [[ -n $pid ]]; then
            [[ $name == "$cur" ]] || { cur=$name; peak=0; term_at=0; }
            if (( use > peak )); then peak=$use; echo "$peak" > "$R/.peak.$name"; fi
            reason=""
            (( use > MAX_KIB )) && reason="job total $(gib "$use") GiB (cgroup $(gib "$cgk") + GPU $(gib "$gpu") + reserve $HELPER_RESERVE_GIB) over TOTAL_GIB $TOTAL_GIB"
            (( gpu < 0 )) && reason="GPU memory query failed (no complete numeric memory.used): job total unbounded"
            (( cgk < 0 )) && reason="cgroup memory.current unreadable: job total unbounded"
            (( avail < MIN_KIB )) && reason="MemAvailable $(gib "$avail") GiB under MIN_GIB $MIN_GIB"
            (( now >= CLEANUP_AT )) && reason="cleanup point (DEADLINE - 10 min)"
            if [[ -n $reason ]]; then
                if (( term_at == 0 )); then
                    [[ -f $R/.killed.$name ]] || echo "$reason" > "$R/.killed.$name"
                    echo "$(date -u +%T) guard: $name: $reason: TERM process group $pid" >&2
                    kill -TERM -- "-$pid" 2> /dev/null; term_at=$now
                elif (( now - term_at >= 10 )); then
                    echo "$(date -u +%T) guard: $name: still alive 10 s after TERM: KILL process group $pid" >&2
                    kill -KILL -- "-$pid" 2> /dev/null; term_at=$now
                fi
            fi
        fi
        if (( now >= DEADLINE )); then
            echo "$(date -u +%T) guard: DEADLINE reached: KILL ${pid:+process group $pid and }stopping the job" >&2
            [[ -n $pid ]] && kill -KILL -- "-$pid" 2> /dev/null
            kill -TERM "$MAIN" 2> /dev/null
        fi
        echo "$now $avail ${name:--} $cgk $gpu $drop $use"
        tick=$(( tick + 1 ))
        sleep 1
    done
}

# one step: refuse unless it fits in time, disk and memory; run it in its own session under a timeout
run_step() {
    local name=$1 max=$2 outgb=$3 peak=$4 fatal=$5 S=$R/$1 now used free avail tos rc t0 status pk dua
    now=$(date +%s)
    used=$(du_bytes); free=$(df_bytes); avail=$(avail_kib)
    local dub=$(( used / 1000000000 ))
    if (( now >= STOP_NEW )); then
        log "$name: not started: past DEADLINE - 20 min"; record "$name" "not-run: time" - - - "$max" "$dub" -; return 1
    fi
    if (( now + max * 60 > CLEANUP_AT )); then
        log "$name: not started: needs up to $max min, $(( (CLEANUP_AT - now) / 60 )) min left before cleanup"
        record "$name" "not-run: time" - - - "$max" "$dub" -; return 1
    fi
    if (( used + outgb * 1000000000 > DISK_GB * 1000000000 || free < (outgb + 5) * 1000000000 )); then
        log "$name: not started: WORK $dub GB + budget $outgb GB over $DISK_GB GB, or $(( free / 1000000000 )) GB free"
        record "$name" "not-run: disk" - - - "$max" "$dub" -; return 1
    fi
    local cgk g
    cgk=$(cg_kib memory.current); g=$(gpu_kib)
    if (( cgk < 0 || g < 0 )); then
        log "$name: not started: cgroup memory.current or GPU memory.used unreadable (fail closed)"
        record "$name" "not-run: memory unaccounted" - - - "$max" "$dub" -; return 1
    fi
    if (( avail < (peak + MIN_GIB) * GIB_KIB || cgk + g + (HELPER_RESERVE_GIB + peak) * GIB_KIB > MAX_KIB )); then
        log "$name: not started: MemAvailable $(gib "$avail") GiB under its declared peak $peak + MIN_GIB $MIN_GIB, or cgroup $(gib "$cgk") + GPU $(gib "$g") + reserve $HELPER_RESERVE_GIB + peak $peak GiB over TOTAL_GIB $TOTAL_GIB"
        record "$name" "not-run: memory" - - - "$max" "$dub" -; return 1
    fi
    if ! verify_kit "$KD" >> "$R/job.log" 2>&1; then
        log "$name: not started: the extracted kit no longer matches its MANIFEST"
        record "$name" "not-run: kit modified" - - - "$max" "$dub" -; return 1
    fi
    step_cmd "$name" "$S" || { log "$name: unknown step"; return 1; }
    mkdir -p "$S"
    tos=$(( max * 60 )); (( CLEANUP_AT - now < tos )) && tos=$(( CLEANUP_AT - now ))
    log "$name: start (max $max min, timeout ${tos} s, budget $outgb GB, declared peak $peak GiB, MemAvailable $(gib "$avail") GiB, WORK $dub GB)"
    printf '%q ' "${CMD[@]}" > "$S/command.txt"
    t0=$(date +%s)
    (cd "$TWIN" && exec setsid timeout --signal=TERM --kill-after=30 "$tos" "${CMD[@]}") < /dev/null > "$S/log" 2>&1 &
    CUR_PID=$!; CUR_STEP=$name
    echo "$CUR_PID $name $avail" > "$R/.step.tmp" && mv -f "$R/.step.tmp" "$R/.step"
    wait "$CUR_PID"; rc=$?
    rm -f "$R/.step"
    kill -KILL -- "-$CUR_PID" 2> /dev/null   # anything the step left behind in its session
    CUR_PID=""
    echo "$rc" > "$S/exit_code"
    pk=$(gib "$(cat "$R/.peak.$name" 2> /dev/null || echo 0)")
    dua=$(( $(du_bytes) / 1000000000 ))
    if [[ -f $R/.killed.$name ]]; then status="killed by guard: $(cat "$R/.killed.$name")"
    elif (( rc == 0 )); then status=ok
    elif (( rc == 124 || rc == 137 )); then status="timeout after $tos s"
    elif (( rc == 1 )); then status="failed (receipt written if the tool got that far)"
    else status="error rc $rc"
    fi
    local mins
    mins=$(awk -v s="$(( $(date +%s) - t0 ))" 'BEGIN {printf "%.1f", s / 60}')
    record "$name" "$status" "$rc" "$mins" "$pk" "$max" "$dub" "$dua"
    log "$name: $status (rc $rc, $mins min, peak $pk GiB, WORK $dub -> $dua GB)"
    (( rc == 0 )) && return 0
    (( fatal == 0 && rc == 1 )) && return 0   # a test that ran and failed: its receipt stands, the next mode still runs
    return 1
}

MAIN=$$
trap finalize EXIT
trap 'RESULT="stopped by signal"; exit 143' INT TERM
log "job $JOB, kit $KIT_SHA, deadline $(date -u -d "@$DEADLINE" +%FT%TZ), results $R"

HOLD_LOCK=1 preflight 2>&1 | tee "$R/preflight.log"
(( PIPESTATUS[0] == 0 )) || { RESULT="preflight failed: not started"; log "$RESULT"; exit 1; }

# ---- the kit: verified, extracted once, read-only
if [[ -d $KD ]]; then
    verify_kit "$KD" >> "$R/job.log" 2>&1 || { RESULT="extracted kit $KD does not match: not started (left as is)"; log "$RESULT"; exit 1; }
    log "kit: $KD present and verified"
else
    kit_sha_ok >> "$R/job.log" 2>&1 || { RESULT="kit sha256 mismatch: not started"; log "$RESULT"; exit 1; }
    if tar -tzf "$KIT" | grep -Eq '^/|(^|/)\.\.(/|$)'; then RESULT="kit has unsafe paths: not started"; log "$RESULT"; exit 1; fi
    PART=$WORK/.kit-$SHA12.partial.$$
    mkdir "$PART" || { RESULT="cannot create $PART"; log "$RESULT"; exit 1; }
    if ! timeout 600 tar -C "$PART" --no-same-owner --no-same-permissions -xzf "$KIT" || ! verify_kit "$PART" >> "$R/job.log" 2>&1; then
        rm -rf "$PART"; RESULT="kit extraction failed: not started"; log "$RESULT"; exit 1
    fi
    mv -T "$PART" "$KD" || { rm -rf "$PART"; RESULT="cannot place $KD"; log "$RESULT"; exit 1; }
    chmod -R a-w "$KD" || { RESULT="chmod -R a-w $KD failed: not started"; log "$RESULT"; exit 1; }
    log "kit: extracted to $KD (read-only)"
fi
[[ ! -w $KD/MANIFEST.txt ]] || { RESULT="$KD is writable: not started"; log "$RESULT"; exit 1; }

timeout 30 nvidia-smi -q > "$R/nvidia-smi-q.txt" 2>&1
(cd / && timeout 300 "$PY" -I -c "$ENV_PY" "$R/env.json" "$KD" "$JOB" "$R/nvidia-smi-q.txt") >> "$R/job.log" 2>&1 ||
    log "env.json: collection failed (see job.log)"

guard > "$R/memavail.log" 2> "$R/guard.log" &
GUARD=$!
log "guard: pid $GUARD; memavail.log columns: epoch MemAvailable_kib step cgroup_current_kib gpu_used_kib(-1 = query failed) memavail_drop_kib total_kib"

RESULT=ok
mapfile -t STEPS < <(steps)
for line in "${STEPS[@]}"; do
    read -r name max outgb peak fatal <<< "$line"
    if ! run_step "$name" "$max" "$outgb" "$peak" "$fatal"; then
        RESULT="stopped at $name"
        break
    fi
    [[ $(tail -1 "$R/steps.tsv" | cut -f2) == ok ]] || RESULT="completed with failures"
done
log "job $JOB: $RESULT"
[[ $RESULT == ok ]]
