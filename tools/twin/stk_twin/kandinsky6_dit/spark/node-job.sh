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
#   memory  the owner's GB10 uma-node policy (2026-10-12), all of (a)-(d); cgroups and nvidia-smi bound nothing on GB10:
#           (a) Torch: every CUDA process of a step caps torch's caching allocator at the step's declared Torch budget
#               with set_per_process_memory_fraction before its first allocation, reads it back and refuses unless
#               readback == set and cap <= budget (budget.py; STK_K6_TORCH_BUDGET_BYTES, records in <step>/caps).
#               After the step, every record is checked, and there must be at least one per CUDA process.
#           (b) non-allocator overhead (CUDA context, module images, cuBLAS / cuDNN state, host python): not yet
#               measured for this exact job, so every process counted gets the owner's unmeasured minimum, 4 GiB with
#               2x margin = 8 GiB: CUDA_OVERHEAD_GIB 8 per CUDA process, HOST_PROC_GIB 8 per host-only process and
#               JOB_SHELL_GIB 8 for this script, its guard, tar and the supervisor. The job's processes make no native
#               allocation outside torch (fatbins are loaded into torch's context and launched on torch tensors), so
#               the raw allocation ledger is empty (test_policy.py lists every native allocation call site).
#               step budget = cuda_procs x (torch + CUDA_OVERHEAD_GIB) + host_procs x HOST_PROC_GIB + JOB_SHELL_GIB.
#           (c) host guard: a step starts only if MemAvailable >= its budget + FLOOR_GIB (12, never lower). Steps
#               run one at a time, so that is the sum of the budgets running at once. supervise.py runs each step in
#               its own session, as its child subreaper; it samples MemAvailable every 0.1 s and SIGKILLs the whole
#               process tree at once below the floor, on a sampling gap > 0.5 s, an unreadable /proc, a kernel
#               oom_kill, its timeout or a signal.
#           (d) sticky STOP_ALL: after.sh / classify.py (the D1 classifier's rules), run by the operator after the job's
#               scope ends; this script writes job-result.json for them.
#           The job must run in its own cgroup v2 scope with memory.max <= 96 GiB (`systemd-run --scope -p
#           MemoryMax=96G`): a host accounting and refusal check, never a GPU bound. memory.current, and the VmRSS of
#           launcher / login ancestors outside the scope, are measured every second and logged. Discovery fails
#           closed: an unreadable or malformed stat or cgroup of a live ancestor (or of this job) refuses every step,
#           never a partial list; an ancestor that exits mid-scan restarts it, and a live helper that becomes
#           unreadable kills the running step. A job inside a container is refused: its launcher is outside the pid
#           namespace and cannot be measured from here. Logged: memavail.log (every second), guard.log (actions),
#           <step>/supervision.json.
#   disk    a step does not start if du($WORK) + its output budget exceeds 200 GB, or the filesystem lacks the budget
#           + 5 GB; du before and after every step is recorded.
#   GPU     one heavy job per node: an exclusive flock on $WORK/.gpu.lock, held for the whole run (refused if another
#           k6 job holds it). Other GPU compute apps are reported (preflight, env.json, cleanup), never touched.
#   scope   no docker inside the job, no image builds, no network: proxies unset, NCCL/UCX/MPI pinned to loopback with IB off, HF/pip
#           offline. Only $WORK is written (TMPDIR and every cache are under it); only our own process groups are killed.
#   oracle  NVIDIA_TF32_OVERRIDE, TORCH_ALLOW_TF32* and CUBLASLT_LOG_* must not be set (refused); this script never
#           sets them (gate.py sets CUBLASLT_LOG_* itself where it wants them).
# Optional env: FLOOR_GIB (12; lower is refused), SEED (0, passed to test_ops / gate), STEP_MAX_MIN (the max minutes
# of every twin / ops / blocks step; defaults 20 / 40 (+10 per further fp32/* form) / 20), MODES (a subset of
# "tf32_rna tf32_truncate fp32", in order: splits a job into discrete runs; ops runs every site under the first),
# RUN_ID (the results directory name; default <job>-<UTC>; an existing one is never reused).
# Output: $WORK/results/<RUN_ID>/: env/ (env.json), nvidia-smi-q.txt, preflight.log, job.log, steps.tsv, memavail.log,
# guard.log, <step>/{log,exit_code,supervision.json,caps/,<receipts>}, SUMMARY.txt, job-result.json; then
# $WORK/results/<RUN_ID>.tar.gz and its sha256.
# --budgets prints the step table with each step's Torch and total budget, then exits (no other check runs).
set -uo pipefail
umask 022
export LC_ALL=C

usage() { echo "usage: DEADLINE=<unix> WORK=<dir> KIT=<tarball> KIT_SHA=<sha256> VENV=<venv> bash node-job.sh {smoke|twin|ops|blocks} [--preflight|--budgets]" >&2; exit 2; }
JOB=${1:-}
OPT=${2:-}
case $JOB in smoke | twin | ops | blocks) ;; *) usage ;; esac
[[ -z $OPT || $OPT == --preflight || $OPT == --budgets ]] || usage
(( $# <= 2 )) || usage
PREFLIGHT_ONLY=0
[[ $OPT == --preflight ]] && PREFLIGHT_ONLY=1

die() { echo "node-job: $*" >&2; exit 1; }
if [[ $OPT == --budgets ]]; then   # the step table only: nothing below reads or writes these
    WORK=${WORK:-/budgets-only} KIT=${KIT:-/budgets-only} VENV=${VENV:-/budgets-only}
    KIT_SHA=${KIT_SHA:-0000000000000000000000000000000000000000000000000000000000000000}
fi
: "${WORK:?WORK (the only directory this job writes) is required}"
: "${KIT:?KIT (the kit tarball) is required}"
: "${KIT_SHA:?KIT_SHA (its sha256) is required}"
: "${VENV:?VENV (the python 3.13 + torch 2.14.1 cu130 venv) is required}"
[[ $WORK == /* && $WORK != / && $WORK != *..* ]] || die "WORK must be an absolute path other than / (got $WORK)"
[[ $KIT_SHA =~ ^[0-9a-f]{64}$ ]] || die "KIT_SHA must be 64 lowercase hex digits"
if (( PREFLIGHT_ONLY == 0 )) && [[ $OPT != --budgets ]]; then
    : "${DEADLINE:?DEADLINE (unix seconds the window ends) is required}"
fi
abspath() { if [[ $1 == /* ]]; then echo "$1"; else echo "$PWD/$1"; fi; }
KIT=$(abspath "$KIT")
VENV=$(abspath "$VENV")
[[ -z ${DEADLINE:-} || ${DEADLINE:-} =~ ^[0-9]+$ ]] || die "DEADLINE must be unix seconds"
FLOOR_GIB=${FLOOR_GIB:-12}
SEED=${SEED:-0}
for v in FLOOR_GIB SEED; do [[ ${!v} =~ ^[0-9]+$ ]] || die "$v must be a non-negative integer"; done
(( FLOOR_GIB >= 12 )) || die "FLOOR_GIB $FLOOR_GIB is under the owner's 12 GiB MemAvailable floor"
[[ -z ${RUN_ID:-} || ${RUN_ID:-} =~ ^[a-z0-9][a-z0-9-]*$ ]] || die "RUN_ID must match ^[a-z0-9][a-z0-9-]*$"
[[ -z ${STEP_MAX_MIN:-} || ${STEP_MAX_MIN:-} =~ ^[1-9][0-9]*$ ]] || die "STEP_MAX_MIN must be a positive integer"
MODES=${MODES:-tf32_rna tf32_truncate fp32}   # a subset splits twin / ops / blocks into discrete, resumable jobs
for m in $MODES; do [[ " tf32_rna tf32_truncate fp32 " == *" $m "* ]] || die "MODES: unknown Fp32Linear form $m"; done
[[ -n ${MODES// /} ]] || die "MODES is empty"

# contract constants (not overridable)
CG_MAX_GIB=96          # the job's cgroup memory.max must be at most this: host accounting and refusal, not a GPU bound
CUDA_OVERHEAD_GIB=8    # policy (b), unmeasured for this job: 2 x the 4 GiB minimum, per CUDA process
HOST_PROC_GIB=8        # the same, per host-only process (gate's parent)
JOB_SHELL_GIB=8        # the same, for this script, its guard, tar and the supervisor
POLL_S=0.1             # policy (c): supervise.py samples MemAvailable this often ...
MAX_GAP_S=0.5          # ... and kills the step if two samples are further apart than this
DISK_GB=200
STOP_NEW_S=1200
CLEANUP_S=600
GIB_KIB=1048576
FLOOR_KIB=$(( FLOOR_GIB * GIB_KIB ))
SHA12=${KIT_SHA:0:12}
KD=$WORK/kit-$SHA12
LOCK=$WORK/.gpu.lock
# the venv's python, the image's /usr/local/bin/python, or the shared prefix's bin/python3.13
PY=$VENV/bin/python
[[ -x $PY ]] || PY=$VENV/bin/python3.13
TWIN=$KD/repo/tools/twin
SUP=$TWIN/stk_twin/kandinsky6_dit/spark/supervise.py
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
JOB_BASE=$(avail_kib)   # MemAvailable before this job does anything (supplemental headroom only)
# this job's cgroup v2 directory (the systemd scope it was started in)
CG=/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup 2> /dev/null)
cg_kib() { local v; v=$(cat "$CG/$1" 2> /dev/null) || { echo -1; return; }; [[ $v =~ ^[0-9]+$ ]] && echo $(( v / 1024 )) || echo -1; }
cg_max_kib() { local v; v=$(cat "$CG/memory.max" 2> /dev/null) || { echo -1; return; }; [[ $v == max ]] && echo -2 || echo $(( v / 1024 )); }
# ---- launcher / login helper discovery (begin)
# This job's ancestors outside its cgroup, up to (excluding) pid 1 and the systemd manager, recorded once as
# "pid:starttime". Fail closed: unless every live ancestor's /proc stat and cgroup read and parse, HELPER_ERR is set
# and helper_kib reports -1, so a partial chain is never accepted. An ancestor that exits mid-scan (its child is
# re-parented) restarts the scan; a chain still changing after 5 scans fails too.
proc_read() { cat "/proc/$1/$2" 2> /dev/null; }
proc_alive() { [[ -d /proc/$1 ]]; }
# the unified (v2) cgroup path of pid $1: exactly one "0::/..." line
cg_of() {
    local t
    t=$(proc_read "$1" cgroup) || return 1
    t=$(awk -F: '$1 == "0" && $2 == "" {print $3; n++} END {exit n != 1}' <<< "$t") || return 1
    [[ $t == /* ]] || return 1
    echo "$t"
}
# "ppid starttime comm" of pid $1, from a /proc/$1/stat whose pid field and shape check out
stat_of() {
    local st pid comm f
    st=$(proc_read "$1" stat) || return 1
    [[ $st == *" ("*") "* ]] || return 1
    pid=${st%% (*}; [[ $pid == "$1" ]] || return 1
    comm=${st#* (}; comm=${comm%) *}
    read -r -a f <<< "${st##*) }"
    (( ${#f[@]} >= 20 )) && [[ ${f[1]} =~ ^[0-9]+$ && ${f[19]} =~ ^[0-9]+$ ]] || return 1
    echo "${f[1]} ${f[19]} $comm"
}
HELPERS=""; HELPER_ERR=""
helper_scan_once() {   # 0 complete, 1 unreadable or malformed (HELPER_ERR set), 2 the chain changed: scan again
    local c=$$ p pp s s2 cg start comm mine depth=0
    mine=$(cg_of $$) || { HELPER_ERR="this job's own /proc cgroup is unreadable or malformed"; return 1; }
    s=$(stat_of $$) || { HELPER_ERR="this job's own /proc stat is unreadable or malformed"; return 1; }
    read -r p _ <<< "$s"
    while (( p != 1 )); do
        (( p > 0 )) || { HELPER_ERR="the parent of $c is outside this pid namespace"; return 1; }
        (( ++depth <= 256 )) || { HELPER_ERR="ancestor chain deeper than 256"; return 1; }
        if s=$(stat_of "$p") && cg=$(cg_of "$p") && s2=$(stat_of "$p"); then
            [[ $s == "$s2" ]] || return 2
        else
            # p exited if its child c has been re-parented; if c still names p as parent, a live ancestor is unreadable
            s2=$(stat_of "$c") || return 2
            read -r pp _ <<< "$s2"
            (( pp == p )) && { HELPER_ERR="live ancestor $p: /proc stat or cgroup unreadable or malformed"; return 1; }
            return 2
        fi
        read -r pp start comm <<< "$s"
        [[ $comm == systemd ]] && return 0
        [[ $cg == "$mine" ]] || HELPERS+="$p:$start "
        c=$p; p=$pp
    done
}
helper_scan() {
    local i r
    for i in 1 2 3 4 5; do
        HELPERS=""; HELPER_ERR=""
        helper_scan_once; r=$?
        (( r == 0 )) && return 0
        (( r == 1 )) && { HELPERS=""; return 0; }
    done
    HELPERS=""; HELPER_ERR="the ancestor chain kept changing during 5 scans"
}
# summed VmRSS (KiB) of the recorded helpers still alive, or -1 if discovery failed or a live one is unreadable
helper_kib() {
    [[ -z $HELPER_ERR ]] || { echo -1; return; }
    local h p s st v k=0
    for h in $HELPERS; do
        p=${h%%:*}
        s=$(stat_of "$p") || { proc_alive "$p" && { echo -1; return; }; continue; }
        read -r _ st _ <<< "$s"; [[ $st == "${h#*:}" ]] || continue   # exited; the pid now names another process
        s=$(proc_read "$p" status) || { proc_alive "$p" && { echo -1; return; }; continue; }
        v=$(awk '/^VmRSS:/ {r = $2; n++} /^State:[ \t]+[ZX]/ {z = 1} END {print n == 1 ? r : (n == 0 && z ? 0 : "x")}' <<< "$s")
        [[ $v =~ ^[0-9]+$ ]] || { echo -1; return; }
        k=$(( k + v ))
    done
    echo "$k"
}
# ---- launcher / login helper discovery (end)
in_container() { [[ -f /.dockerenv || -f /run/.containerenv ]] || grep -qE '(docker|containerd|kubepods|libpod)' /proc/1/cgroup 2> /dev/null; }
helper_scan   # once, at start: the ancestors that launched this job
gib() { awk -v k="$1" 'BEGIN {printf "%.1f", k / 1048576}'; }
du_bytes() { du -sxb "$WORK" 2> /dev/null | cut -f1; }
df_bytes() { df -B1 --output=avail "$WORK" | tail -1 | tr -dc 0-9; }
gpu_apps() { timeout 30 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2> /dev/null | tr '\n' ';'; }

# ---- the steps: "name max_min out_gb torch_gib cuda_procs host_procs fatal" (estimates in README.md; max_min is the
# hard timeout; torch_gib is each CUDA process's declared Torch budget, its caching-allocator cap; cuda_procs and
# host_procs count the processes alive at once: gate runs a host-only parent and one CUDA child at a time)
steps() {
    local m
    echo "env 3 1 1 1 0 1"
    case $JOB in
    smoke)
        echo "import 3 1 1 1 0 1"
        echo "kernels 3 1 1 1 0 1"
        echo "test_ops_small 15 1 4 1 0 1" ;;
    twin) for m in $MODES; do echo "test_ops_$m ${STEP_MAX_MIN:-20} 1 16 1 0 0"; done ;;
    # only the fp32/* sites depend on the Fp32Linear form: every site once (tf32_rna), then fp32/* for the others
    ops) local first=1
        for m in $MODES; do
            if (( first )); then echo "gate_ops_$m ${STEP_MAX_MIN:-40} 5 16 1 1 0"; first=0
            else echo "gate_fp32_$m ${STEP_MAX_MIN:-10} 1 16 1 1 0"; fi
        done ;;
    blocks) for m in $MODES; do echo "gate_blocks_$m ${STEP_MAX_MIN:-20} 10 28 1 1 0"; done ;;
    esac
}
# a step's total budget in GiB: policy (a) + (b) for every process alive at once
budget_gib() { echo $(( $2 * ($1 + CUDA_OVERHEAD_GIB) + $3 * HOST_PROC_GIB + JOB_SHELL_GIB )); }
total_out_gb() { steps | awk '{s += $3} END {print s + 1}'; }   # + 1 GB: the extracted kit and the results tarball
max_budget_gib() { local n x o t c h f b=0 m=0; while read -r n x o t c h f; do b=$(budget_gib "$t" "$c" "$h"); (( b > m )) && m=$b; done < <(steps); echo "$m"; }
first_max_min() { steps | awk 'NR == 1 {print $2}'; }
if [[ $OPT == --budgets ]]; then
    printf 'step\tmax_min\ttorch_gib\tcuda_procs\thost_procs\tbudget_gib\tadmission_gib\n'
    while read -r n x o t c h f; do
        b=$(budget_gib "$t" "$c" "$h"); printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$x" "$t" "$c" "$h" "$b" $(( b + FLOOR_GIB ))
    done < <(steps)
    echo "floor_gib $FLOOR_GIB cuda_overhead_gib $CUDA_OVERHEAD_GIB host_proc_gib $HOST_PROC_GIB job_shell_gib $JOB_SHELL_GIB poll_s $POLL_S max_gap_s $MAX_GAP_S"
    exit 0
fi

IMPORT_PY='
import importlib, importlib.util, json, sys
from pathlib import Path
kd = Path(sys.argv[1]).resolve()
import torch
from stk_twin.kandinsky6_dit import budget
cap = budget.apply("import")
from stk_twin.kandinsky6_dit import block, kernels, native, test_ops
mods = ["block", "budget", "kernels", "native", "test_ops"]
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
print(json.dumps({"ok": True, "twin": mods, "upstream": str(u.src), "torch": torch.__version__, "torch_cap": cap,
                  "optional_attention_installed": optional,
                  "loaded": sorted(n for n in sys.modules if n.split(".")[0] in ("stk_twin", "kandinsky"))}, indent=1))
'
KERNELS_PY='
import json, sys
from pathlib import Path
kd, out = Path(sys.argv[1]), sys.argv[2]
import torch
from stk_twin.kandinsky6_dit import budget, kernels as K
torch.cuda.init()
cap = budget.apply("kernels")
k = K.Kernels(fatbins=str(kd / "fatbins"))
man = {}
for line in (kd / "MANIFEST.txt").read_text().splitlines():
    if line.startswith("# fatbin: "):
        name, sha = line.split()[2:4]
        man[name] = sha
want = {n: man.get(f"k6_dit_{n}.fatbin") for n in K.IMAGES}
ok = k.digests == want
r = {"ok": ok, "loaded": k.digests, "manifest": want, "torch_cap": cap, "device": torch.cuda.get_device_name(),
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
avail = torch.cuda.is_available()   # the device count only: preflight runs unsupervised, so it opens no CUDA context
say(avail, f"torch.cuda.is_available() {avail}")
import subprocess
q = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], capture_output=True,
                   text=True, timeout=30)
rows = [r.split(", ") for r in q.stdout.strip().splitlines()]
say(q.returncode == 0 and len(rows) == 1 and rows[0][-1] == "12.1", f"device {q.stdout.strip()!r} capability 12.1")
say(not torch.cuda.is_initialized(), "no CUDA context opened by preflight")
for n in ("flash_attn", "flash_attn_interface", "sageattention"):
    if importlib.util.find_spec(n) is not None:
        print(f"  note optional {n} is installed (the native reference uses sdpa; not required)")
sys.exit(0 if ok else 1)
'
ENV_PY='
import json, os, platform, socket, subprocess, sys
out, kd, job, smi = sys.argv[1:5]
import torch
from stk_twin.kandinsky6_dit import budget
cap = budget.apply("env")
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
    "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "device": dev, "torch_cap": cap,
    "env": {k: os.environ[k] for k in sorted(names)},
}, open(out, "w"), indent=1)
'

# the argv of a step, in CMD
step_cmd() {
    local s=$1 out=$2 torch=$3 m
    case $s in
    env) CMD=("$PY" -s -c "$ENV_PY" "$out/env.json" "$KD" "$JOB" "$R/nvidia-smi-q.txt") ;;
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
             --fp32-linear "$m" --seed "$SEED" --max-gib "$torch" "${sites[@]}") ;;
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
    for t in flock timeout sha256sum tar gzip du df awk cmp nvidia-smi; do
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
    local cgmax cgcur
    cgmax=$(cg_max_kib); cgcur=$(cg_kib memory.current)
    if (( cgmax == -1 || cgcur < 0 )); then
        echo "FAIL no cgroup v2 memory accounting at $CG (memory.max / memory.current unreadable)"; bad=1
    elif (( cgmax == -2 || cgmax > CG_MAX_GIB * GIB_KIB )); then
        echo "FAIL cgroup $CG memory.max is $( (( cgmax == -2 )) && echo unlimited || echo "$(gib "$cgmax") GiB"): run in its own cgroup with memory.max <= $CG_MAX_GIB GiB"; bad=1
    else
        echo "ok   cgroup $CG: memory.max $(gib "$cgmax") GiB, memory.current $(gib "$cgcur") GiB, memory.peak $( p=$(cg_kib memory.peak); (( p < 0 )) && echo n/a || echo "$(gib "$p") GiB")"
    fi
    if in_container; then
        echo "FAIL running inside a container: its launcher is outside this pid namespace and cannot be measured; run on the host under systemd-run --scope"; bad=1
    fi
    local hk; hk=$(helper_kib)
    if [[ -n $HELPER_ERR ]]; then
        echo "FAIL launcher / login helper discovery: $HELPER_ERR"; bad=1
    elif (( hk >= 0 )); then
        echo "ok   launcher / login helpers outside the cgroup: pid:starttime ${HELPERS:-none}, $(gib "$hk") GiB RSS (logged every second)"
    else
        echo "FAIL a launcher / login helper is alive but its memory is unreadable"; bad=1
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
    echo "memory: MemAvailable $(gib "$avail") GiB; $JOB's largest step budget $(max_budget_gib) GiB + floor $FLOOR_GIB GiB (owner policy (c))"
    (( avail >= ($(max_budget_gib) + FLOOR_GIB) * GIB_KIB )) || { echo "FAIL MemAvailable under the largest step budget + floor"; bad=1; }
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
RUN=${RUN_ID:-$JOB-$(date -u +%Y%m%d-%H%M%S)}
R=$WORK/results/$RUN
[[ ! -e $R && ! -e $R.tar.gz ]] || die "results $R already exist: a run id is never reused"
mkdir "$R" || die "cannot create $R"
log() { echo "[k6 $JOB $(date -u +%T)] $*" | tee -a "$R/job.log"; }
printf 'step\tstatus\trc\tminutes\tmax_min\ttorch_gib\tbudget_gib\tmin_avail_gib\tmax_gap_s\tcaps\tdu_before_gb\tdu_after_gb\n' > "$R/steps.tsv"
GUARD=""
CUR_PID=""
CUR_STEP=""
RESULT="incomplete"

record() {   # name status rc minutes max_min torch_gib budget_gib min_avail_gib max_gap_s caps du_before du_after
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$R/steps.tsv"
}

# stop a running step: TERM its supervisor, which SIGKILLs the step's whole process tree at once; if the supervisor
# is gone or still alive 2 s later, SIGKILL the step's session (from its session file) and the supervisor
stop_step() {
    local pid=$1 S=$2 i sid
    kill -TERM "$pid" 2> /dev/null
    for (( i = 0; i < 20; i++ )); do kill -0 "$pid" 2> /dev/null || return 0; sleep 0.1; done
    read -r _ sid < "$S/.session" 2> /dev/null && [[ $sid =~ ^[0-9]+$ ]] && kill -KILL -- "-$sid" 2> /dev/null
    kill -KILL "$pid" 2> /dev/null
}

JOBRES_PY='
import json, os, sys
r, job, kit, run, result, started, ended, floor = sys.argv[1:9]
rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(r, "steps.tsv"))]
out = {"schema": "k6-dit-job-result-v1", "job": job, "kit_sha256": kit, "run": run, "result": result,
       "started": int(started), "ended": int(ended), "floor_gib": int(floor),
       "steps": [dict(zip(rows[0], x)) for x in rows[1:]]}
tmp = os.path.join(r, "job-result.json.tmp")
with open(tmp, "w") as f:
    json.dump(out, f, indent=1)
os.replace(tmp, os.path.join(r, "job-result.json"))
'

finalize() {
    local tarball sha ended py
    trap - EXIT
    trap '' INT TERM   # finish packing whatever arrives
    if [[ -n $CUR_PID ]]; then
        stop_step "$CUR_PID" "$R/$CUR_STEP"
        wait "$CUR_PID" 2> /dev/null
        record "$CUR_STEP" "killed: job stopped" - - - - - - - - - -
        log "$CUR_STEP: stopped (job stopped; its supervisor kills the whole tree)"
    fi
    if [[ -n $GUARD ]]; then kill "$GUARD" 2> /dev/null; wait "$GUARD" 2> /dev/null; fi
    log "cleanup: du WORK $(( $(du_bytes) / 1000000000 )) GB; GPU compute apps: $(gpu_apps)"
    ended=$(date +%s)
    {
        echo "k6 DiT node job: $JOB"
        echo "host: $(hostname)"
        echo "kit: $KIT_SHA"
        echo "started: $(date -u -d "@$START" +%FT%TZ)  ended: $(date -u -d "@$ended" +%FT%TZ)  deadline: $(date -u -d "@$DEADLINE" +%FT%TZ)"
        echo "result: $RESULT"
        echo "memory (owner GB10 policy): per-step Torch cap + CUDA_OVERHEAD_GIB $CUDA_OVERHEAD_GIB per CUDA process + HOST_PROC_GIB $HOST_PROC_GIB per host process + $JOB_SHELL_GIB; MemAvailable floor $FLOOR_GIB GiB sampled every $POLL_S s (gap <= $MAX_GAP_S s), whole-tree SIGKILL"
        echo "host accounting: cgroup $CG memory.peak $( p=$(cg_kib memory.peak); (( p < 0 )) && echo n/a || echo "$(gib "$p") GiB"), memory.max $(gib "$(cg_max_kib)") GiB; helpers ${HELPERS:-none} logged"
        echo
        column -t -s $'\t' "$R/steps.tsv" 2> /dev/null || cat "$R/steps.tsv"
        echo
        echo "guard actions:"
        sed 's/^/  /' "$R/guard.log" 2> /dev/null
    } > "$R/SUMMARY.txt"
    for py in "$PY" /usr/bin/python3; do
        (cd / && "$py" -I -c "$JOBRES_PY" "$R" "$JOB" "$KIT_SHA" "$RUN" "$RESULT" "$START" "$ended" "$FLOOR_GIB") 2>> "$R/job.log" && break
    done
    rm -f "$R"/.step "$R"/.killed.* "$R"/*/.session
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

# the guard: one sample a second for host accounting and time; the MemAvailable floor is supervise.py's (0.1 s)
guard() {
    local now avail pid name cur="" cgk hk reason stopped=0
    while :; do
        now=$(date +%s); avail=$(avail_kib); pid=""; name=""
        [[ -f $R/.step ]] && read -r pid name < "$R/.step"
        cgk=$(cg_kib memory.current); hk=$(helper_kib)
        if [[ -n $pid ]]; then
            [[ $name == "$cur" ]] || { cur=$name; stopped=0; }
            reason=""
            (( cgk < 0 )) && reason="cgroup memory.current unreadable: host accounting lost"
            (( hk < 0 )) && reason="launcher / login helper memory unreadable: host accounting lost"
            (( now >= CLEANUP_AT )) && reason="cleanup point (DEADLINE - 10 min)"
            if [[ -n $reason ]] && (( stopped == 0 )); then
                [[ -f $R/.killed.$name ]] || echo "$reason" > "$R/.killed.$name"
                echo "$(date -u +%T) guard: $name: $reason: stopping the step (supervisor $pid kills its whole tree)" >&2
                stop_step "$pid" "$R/$name"; stopped=1
            fi
        fi
        if (( now >= DEADLINE )); then
            echo "$(date -u +%T) guard: DEADLINE reached: stopping ${name:+step $name and }the job" >&2
            [[ -n $pid ]] && stop_step "$pid" "$R/$name"
            kill -TERM "$MAIN" 2> /dev/null
        fi
        echo "$now $avail ${name:--} $cgk $hk"
        sleep 1
    done
}

# the step's policy evidence: its supervision record and every Torch cap record (owner policy (a) and (c));
# prints "verdict<TAB>min_avail_gib<TAB>max_gap_s<TAB>caps<TAB>kill reason or -"
POLICY_PY='
import glob, json, os, sys
S, torch_gib, cuda, rc, max_gap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
bad = []
try:
    sv = json.load(open(os.path.join(S, "supervision.json")))
except (OSError, ValueError) as e:
    print(f"BAD\t-\t-\t-\tno supervision record: {e!r}")
    sys.exit()
kill = (sv.get("kill") or {}).get("reason")
if sv.get("schema") != "k6-dit-step-supervision-v1" or sv.get("tree_verified") is not True:
    bad.append("process tree not shown gone")
if kill is None and not (isinstance(sv.get("max_gap_s"), (int, float)) and sv["max_gap_s"] <= max_gap):
    bad.append("supervision gap %s s" % sv.get("max_gap_s"))
budget = torch_gib * 1073741824
caps = sorted(glob.glob(os.path.join(S, "caps", "cap-*.json")))
for c in caps:
    try:
        r = json.load(open(c))
        ok = (r["schema"] == "k6-dit-torch-cap-v1" and r["budget_bytes"] == budget
              and r["fraction_readback"] == r["fraction_set"] and r["cap_bytes"] <= budget)
    except (OSError, ValueError, KeyError, TypeError):
        ok = False
    if not ok:
        bad.append(f"Torch cap record {os.path.basename(c)} invalid")
if kill is None and rc in (0, 1) and len(caps) < cuda:
    bad.append(f"{len(caps)} Torch cap record(s) for {cuda} CUDA process(es)")
ma = sv.get("min_avail")
print("\t".join(["BAD" if bad else "ok", f"{ma / 1073741824:.1f}" if isinstance(ma, int) else "-",
                 str(sv.get("max_gap_s", "-")), str(len(caps)), "; ".join(bad) or kill or "-"]))
'

# one step: refuse unless it fits in time, disk and memory (budget + floor); run it under supervise.py
run_step() {
    local name=$1 max=$2 outgb=$3 torch=$4 cuda=$5 host=$6 fatal=$7 S=$R/$1 now used free avail tos rc t0 status dua
    local budget dub cgk h verdict mavail gap ncaps why mins
    budget=$(budget_gib "$torch" "$cuda" "$host")
    now=$(date +%s)
    used=$(du_bytes); free=$(df_bytes); avail=$(avail_kib)
    dub=$(( used / 1000000000 ))
    if (( now >= STOP_NEW )); then
        log "$name: not started: past DEADLINE - 20 min"; record "$name" "not-run: time" - - "$max" "$torch" "$budget" - - - "$dub" -; return 1
    fi
    if (( now + max * 60 > CLEANUP_AT )); then
        log "$name: not started: needs up to $max min, $(( (CLEANUP_AT - now) / 60 )) min left before cleanup"
        record "$name" "not-run: time" - - "$max" "$torch" "$budget" - - - "$dub" -; return 1
    fi
    if (( used + outgb * 1000000000 > DISK_GB * 1000000000 || free < (outgb + 5) * 1000000000 )); then
        log "$name: not started: WORK $dub GB + budget $outgb GB over $DISK_GB GB, or $(( free / 1000000000 )) GB free"
        record "$name" "not-run: disk" - - "$max" "$torch" "$budget" - - - "$dub" -; return 1
    fi
    cgk=$(cg_kib memory.current); h=$(helper_kib)
    if (( cgk < 0 || h < 0 )); then
        log "$name: not started: cgroup memory.current or launcher / login helper memory unreadable (fail closed)"
        record "$name" "not-run: host accounting" - - "$max" "$torch" "$budget" - - - "$dub" -; return 1
    fi
    if (( avail < (budget + FLOOR_GIB) * GIB_KIB )); then
        log "$name: not started: MemAvailable $(gib "$avail") GiB under its budget $budget GiB + floor $FLOOR_GIB GiB"
        record "$name" "not-run: memory" - - "$max" "$torch" "$budget" - - - "$dub" -; return 1
    fi
    if ! verify_kit "$KD" >> "$R/job.log" 2>&1; then
        log "$name: not started: the extracted kit no longer matches its MANIFEST"
        record "$name" "not-run: kit modified" - - "$max" "$torch" "$budget" - - - "$dub" -; return 1
    fi
    step_cmd "$name" "$S" "$torch" || { log "$name: unknown step"; return 1; }
    mkdir -p "$S/caps"
    tos=$(( max * 60 )); (( CLEANUP_AT - now < tos )) && tos=$(( CLEANUP_AT - now ))
    log "$name: start (max $max min, timeout ${tos} s, Torch $torch GiB x $cuda, budget $budget GiB + floor $FLOOR_GIB, MemAvailable $(gib "$avail") GiB, out $outgb GB, WORK $dub GB)"
    printf '%q ' "${CMD[@]}" > "$S/command.txt"
    t0=$(date +%s)
    (cd "$TWIN" && exec env STK_K6_TORCH_BUDGET_BYTES=$(( torch * 1073741824 )) STK_K6_CAP_DIR="$S/caps" \
        "$PY" -I -s "$SUP" --floor-bytes $(( FLOOR_GIB * 1073741824 )) --poll "$POLL_S" --max-gap "$MAX_GAP_S" \
        --timeout "$tos" --record "$S/supervision.json" --session-file "$S/.session" -- "${CMD[@]}") < /dev/null > "$S/log" 2>&1 &
    CUR_PID=$!; CUR_STEP=$name
    echo "$CUR_PID $name" > "$R/.step.tmp" && mv -f "$R/.step.tmp" "$R/.step"
    wait "$CUR_PID"; rc=$?
    rm -f "$R/.step"
    CUR_PID=""
    echo "$rc" > "$S/exit_code"
    IFS=$'\t' read -r verdict mavail gap ncaps why < <(cd / && "$PY" -I -c "$POLICY_PY" "$S" "$torch" "$cuda" "$rc" "$MAX_GAP_S" 2>> "$R/job.log")
    dua=$(( $(du_bytes) / 1000000000 ))
    if [[ $verdict != ok ]]; then status="policy unverified: ${why:-no policy record}"
    elif [[ -f $R/.killed.$name ]]; then status="killed by guard: $(cat "$R/.killed.$name")"
    elif [[ $why != - ]]; then status="killed by supervisor: $why"
    elif (( rc == 0 )); then status=ok
    elif (( rc == 1 )); then status="failed (receipt written if the tool got that far)"
    else status="error rc $rc"
    fi
    mins=$(awk -v s="$(( $(date +%s) - t0 ))" 'BEGIN {printf "%.1f", s / 60}')
    record "$name" "$status" "$rc" "$mins" "$max" "$torch" "$budget" "${mavail:--}" "${gap:--}" "${ncaps:--}" "$dub" "$dua"
    log "$name: $status (rc $rc, $mins min, min MemAvailable ${mavail:--} GiB, max sample gap ${gap:--} s, $ncaps Torch cap record(s), WORK $dub -> $dua GB)"
    [[ $verdict == ok ]] || return 1
    [[ $status == ok ]] && return 0
    (( fatal == 0 && rc == 1 )) && [[ $why == - ]] && return 0   # a test that ran and failed: its receipt stands
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
[[ -f $SUP ]] || { RESULT="the kit has no spark/supervise.py: not started"; log "$RESULT"; exit 1; }

timeout 30 nvidia-smi -q > "$R/nvidia-smi-q.txt" 2>&1   # NVML only: no CUDA context outside a supervised step

guard > "$R/memavail.log" 2> "$R/guard.log" &
GUARD=$!
log "guard: pid $GUARD; memavail.log columns: epoch MemAvailable_kib step cgroup_current_kib helpers_rss_kib(-1 = unreadable)"

RESULT=ok
mapfile -t STEPS < <(steps)
for line in "${STEPS[@]}"; do
    read -r name max outgb torch cuda host fatal <<< "$line"
    if ! run_step "$name" "$max" "$outgb" "$torch" "$cuda" "$host" "$fatal"; then
        RESULT="stopped at $name"
        break
    fi
    [[ $(tail -1 "$R/steps.tsv" | cut -f2) == ok ]] || RESULT="completed with failures"
done
log "job $JOB: $RESULT"
[[ $RESULT == ok ]]
