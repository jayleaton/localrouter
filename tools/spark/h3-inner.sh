#!/usr/bin/env bash
# The H3 window's inside (h3-window.sh starts it): tools/twin/pod/v3.sh on a DGX Spark, in the NGC PyTorch container
# (aarch64, sm_121) with $LOCALROUTER_HOME at /workspace/localrouter. Steps, in order (names are the log and ONLY names):
#   preflight, setup (video venv, ComfyUI 0.37.0, comfy-kitchen, tfvideo, the H3 weights), kitchen-probe, weights,
#   test-h3-ops, test-te32, test-vae-audio, test-vae-video, test-kitchen, build (the Turbo-LoRA pack),
#   capture, capture-avae, capture-vvae, zig-install, zig-build, h3-te, h3-replay, h3-avae, h3-vvae, generate, h3-e2e.
# A failing step does not stop the others. Each step: $R/NAME.log, its last JSON line to summary.jsonl, and
# {"step","rc","s"} to steps.jsonl; MemAvailable is sampled every second (memavail.log; GB10's memory is the system's)
# and its peak drop goes to summary.jsonl at the end.
# Env: SKIP_SETUP=1 (also skips kitchen-probe, weights), SKIP_TWIN_TESTS=1, SKIP_PACK=1 (reuse $P), SKIP_CAPTURE=1
# (no capture or generate steps; the existing captures are reused and kept), ONLY="h3-te h3-replay" (only these steps;
# captures kept), KEEP_CAPTURES=1. Captures are otherwise deleted after the check that uses them.
# WITH_COMFY (default 1 here, since the twin tests below compare against ComfyUI 0.37.0 and comfy-kitchen): 0 sets up only
# the twin's own stack and skips kitchen-probe and test-h3-ops, test-vae-audio, test-vae-video, test-kitchen (test-te32 then runs its
# ComfyUI-free checks only). The
# build / capture / generate steps never need ComfyUI.
# KITCHEN_SRC_BUILD=1 reinstalls comfy-kitchen from the source tree setup clones ($STK/src/comfy-kitchen, v0.2.35) if
# the wheel pip chose has no CUDA backend or no sm_120/121 code (see kitchen-probe.log). UNTESTED: its build recipe is
# the repo's, it needs nvcc and cmake in the image.
# What pod/video-setup.sh does that may not hold on aarch64 / GB10 (the logs say; nothing here papers over it):
#   * comfy-kitchen: ComfyUI's requirements.txt pulls the PyPI wheel; 0.2.35 has no sdist, an aarch64 manylinux wheel
#     (cp310/cp311/cp312-abi3) and a pure-Python py3-none-any wheel. The README lists CUDA builds only for x86_64 and
#     Windows. If the aarch64 wheel has no CUDA extension (or none for sm_12x) the twin runs kitchen's eager/Triton
#     paths and the bit comparisons against "the wheel" (test_te32, test_kitchen, the INT8 attention) change meaning.
#   * the other ComfyUI requirements (av, comfy-aimdo, ...) need aarch64 wheels; pip fails the setup step if not.
#   * torch/torchvision/torchaudio/triton/numpy are the container's, pinned; torchaudio is stubbed if absent.
set -uo pipefail
export STK=/workspace/localrouter
export WITH_COMFY=${WITH_COMFY:-1}
ONLY=${ONLY:-}
R=$STK/results/${H3_RUN:-h3-$(date -u +%Y%m%d-%H%M%S)}; mkdir -p $R
sel() { [[ -z $ONLY || " $ONLY " == *" $1 "* ]]; }
run() { local name=$1; shift; sel $name || return 0; echo "[h3 $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    tail -n 1 $R/$name.log | grep '^{' >> $R/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; return $rc; }
( while :; do echo "$(date +%s) $(awk '/MemAvailable/ {print $2}' /proc/meminfo)"; sleep 1; done ) > $R/memavail.log &
MEM=$!
finish() { kill $MEM 2>/dev/null
    awk 'NR == 1 {a = $2} m == "" || $2 < m {m = $2} END {printf "{\"memavail_start_gib\": %.1f, \"memavail_min_gib\": %.1f, \"peak_used_gib\": %.1f}\n", a / 1048576, m / 1048576, (a - m) / 1048576}' \
        $R/memavail.log >> $R/summary.jsonl
    chmod -R a+rX $R; }   # the container is root; the results are the user's to read
trap finish EXIT

# keep captures when they are reused, or when only some steps run
KEEP=0; [[ ${SKIP_CAPTURE:-0} == 1 || -n $ONLY || ${KEEP_CAPTURES:-0} == 1 ]] && KEEP=1
drop() { [[ $KEEP == 1 ]] || rm -rf "$@"; }
cap() { local name=$1 dir=$2; shift 2; [[ ${SKIP_CAPTURE:-0} == 1 ]] && return 0; sel $name && rm -rf $dir; run $name "$@"; }

# pod/video-env.sh puts the weights on the pod's /root; here they are on $STK (one copy, the Spark's disk is local).
# The patched copy of pod/ is the one run; video-setup.sh sources the env file beside itself.
POD=/tmp/h3-pod
rm -rf $POD; cp -r $STK/twin/pod $POD
sed -i 's#^export VMODELS=/tmp/localrouter/models/minimax-h3 .*#export VMODELS=$STK/models/minimax-h3#' $POD/video-env.sh
grep -q '^export VMODELS=\$STK/' $POD/video-env.sh || { echo "h3-inner: video-env.sh VMODELS patch failed" >&2; exit 1; }
source $POD/video-env.sh
P=$STK/packs/h3-turbo-nvfp4; CAP=$STK/captures
C=$CAP/h3-step1; CA=$CAP/h3-avae; CV=$CAP/h3-vvae; CG=$CAP/h3-gen
mkdir -p $STK/packs $CAP $STK/results

run preflight bash -c 'echo "{\"python\": \"$(python -V 2>&1)\", \"arch\": \"$(uname -m)\"}"; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader;
    nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; awk "/MemAvailable/ {print \$2 / 1048576 \" GiB available\"}" /proc/meminfo;
    df -h /workspace/localrouter | tail -1; python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"'

if [[ ${SKIP_SETUP:-0} != 1 ]]; then
    run setup bash $POD/video-setup.sh
    source $POD/video-env.sh   # again: on a fresh $STK the venv exists only now
    [[ ${KITCHEN_SRC_BUILD:-0} != 1 || $WITH_COMFY != 1 ]] || run kitchen-build env CUDA_HOME=/usr/local/cuda \
        pip install --no-deps --no-build-isolation --force-reinstall $STK/src/comfy-kitchen
    [[ $WITH_COMFY != 1 ]] || run kitchen-probe python - <<'PY'
import glob, importlib.metadata as m, json, os, re, subprocess
import comfy_kitchen as ck
d = os.path.dirname(ck.__file__)
sos = sorted(glob.glob(d + "/**/*.so", recursive=True))
out = {"version": m.version("comfy-kitchen"), "wheel_tags": [l.split(": ")[1] for l in (m.distribution("comfy-kitchen").read_text("WHEEL") or "").splitlines() if l.startswith("Tag:")],
       "so": [os.path.relpath(s, d) for s in sos], "sm": {}}
for s in sos:
    sms = set()
    for flag in ("--list-elf", "--list-ptx"):
        r = subprocess.run(["/usr/local/cuda/bin/cuobjdump", flag, s], capture_output=True, text=True)
        sms |= set(re.findall(r"sm_\d+[a-z]?", r.stdout))
    out["sm"][os.path.basename(s)] = sorted(sms)
out["cuda_extension"] = bool(sos)
out["has_sm120_or_121"] = any({"sm_120", "sm_121", "sm_120a", "sm_121a"} & set(v) for v in out["sm"].values())
print(json.dumps(out))
PY
    run weights bash -c 'for f in diffusion_models/minimax_h3_fl2va_pruned_bf16 text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq vae/minimax_h3_video_vae_fp16 vae/minimax_h3_audio_vae_fp32 loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16; do
        [[ -s '$VMODELS'/$f.safetensors ]] || { echo "missing $f" >&2; m=1; }; done; du -sh '$VMODELS'/*; exit ${m:-0}'
fi

VA=$VMODELS/vae/minimax_h3_audio_vae_fp32.safetensors; VV=$VMODELS/vae/minimax_h3_video_vae_fp16.safetensors
TE=$VMODELS/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
if [[ ${SKIP_TWIN_TESTS:-0} != 1 ]]; then
    [[ $WITH_COMFY != 1 ]] || run test-h3-ops python -m stk_twin.h3.test_ops
    run test-te32 python -m stk_twin.h3.test_te32 --ckpt $TE
    [[ $WITH_COMFY != 1 ]] || run test-vae-audio python -m stk_twin.h3.test_vae_audio
    [[ $WITH_COMFY != 1 ]] || run test-vae-video python -m stk_twin.h3.test_vae_video
    [[ $WITH_COMFY != 1 ]] || run test-kitchen python -m stk_twin.h3.test_kitchen   # our kernel copies against the installed comfy-kitchen: the aarch64 wheel's proof
fi
if [[ ${SKIP_PACK:-0} != 1 ]]; then
    sel build && rm -rf $P
    run build python -m stk_twin.h3.build --models $VMODELS --lora --out $P
fi
cap capture $C python -m stk_twin.h3.capture --models $VMODELS --pack $P --out $C
cap capture-avae $CA python -m stk_twin.h3.capture_avae --models $VMODELS --out $CA
cap capture-vvae $CV python -m stk_twin.h3.capture_vvae --models $VMODELS --out $CV

# zig 0.17.0 (aarch64) and the localrouter binary, built as m5.sh does; the binary stays in $STK/h3-out for re-runs
ZIG=$STK/zig/zig
[[ -x $ZIG ]] || run zig-install bash -c "mkdir -p $STK/zig && curl -fsSL https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz | tar -xJ -C $STK/zig --strip-components=1"
export ZIG_GLOBAL_CACHE_DIR=$STK/zig-cache
if sel zig-build; then
    SRC=/tmp/localrouter/stk-src; rm -rf $SRC; mkdir -p $SRC; cp -r $STK/repo/. $SRC/; cd $SRC
    run zig-build $ZIG build -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121,120 --prefix $STK/h3-out fatbins install
fi
S=$STK/h3-out/bin/localrouter
run h3-te $S check h3-te $TE $C
run h3-replay $S check h3-replay $P $C
drop $C
run h3-avae $S check h3-avae $VA $CA
drop $CA
run h3-vvae $S check h3-vvae $VV $CV
drop $CV
cap generate $CG python -m stk_twin.h3.generate --models $VMODELS --pack $P --out $CG
run h3-e2e $S check h3-e2e $P $TE $VA $VV $CG
drop $CG
echo "[h3 $(date +%T)] done: $R"
