#!/usr/bin/env bash
# M3c on one small sm_120 pod (RTX PRO 4500 Blackwell): Zig 0.17 on the volume, the stk binary with every fatbin for
# sm_120 and sm_121 built by the image's nvcc, the unit tests, the sm_120 SASS check (sm_121 passed offline), then the
# replay bit gate of both captures (NVFP4, FP8). Output: $STK/results/m3c-build-*/ and
# $STK/bin/stk (+ fatbins) for the replay pod.
set -uo pipefail
source "$(dirname "$0")/env.sh"
R=$STK/results/m3c-build-$(date -u +%Y%m%d-%H%M%S); mkdir -p $R; ln -sfn $R $STK/results/m3c-build-latest
run() { local name=$1; shift; echo "[m3c $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; return $rc; }
ZIG=$STK/zig/zig
if [[ ! -x $ZIG ]]; then
    run zig-install bash -c "mkdir -p $STK/zig && curl -fsSL https://ziglang.org/download/0.17.0/zig-x86_64-linux-0.17.0.tar.xz | tar -xJ -C $STK/zig --strip-components=1"
fi
SRC=/tmp/localrouter/stk-src; rm -rf $SRC; mkdir -p $SRC; cp -r $STK/repo/. $SRC/   # build on local disk, not the volume
cd $SRC
export ZIG_GLOBAL_CACHE_DIR=$STK/zig-cache
NVCC=/usr/local/cuda/bin/nvcc
run fetch $ZIG build --fetch
run build $ZIG build -Doptimize=ReleaseSafe -Dnvcc=$NVCC -Dsm=121,120 --prefix $R/out fatbins install
run test-unit $ZIG build test-unit
run sass-nvfp4 env NATIVE=1 SMS=120 TF_PY=$STK/src/TensorFold TF_REF=$TENSORFOLD_TAG OUT=$R/sass FATBINS=$R/out/fatbin bash kernels/cuda/sass_check.sh
mkdir -p $STK/bin && cp -r $R/out/. $STK/bin/
pcp() { mkdir -p "$2"; (cd "$1" && find . -type f -print0 | xargs -0 -P 16 -I{} sh -c 'mkdir -p "$(dirname "$0/{}")" && cp -u "{}" "$0/{}"' "$2"); }
for p in nvfp4 fp8s; do
    run copy-$p bash -c "$(declare -f pcp); pcp $STK/packs/qwen-image-2.1-$p /tmp/localrouter/packs/$p && pcp $STK/captures/m3-$p /tmp/localrouter/captures/$p"
    run replay-$p $R/out/bin/localrouter check qwen-replay /tmp/localrouter/packs/$p /tmp/localrouter/captures/$p
    tail -n 1 $R/replay-$p.log | grep '^{' >> $R/summary.jsonl
done
ls -la $R/out/bin $R/out/fatbin > $R/artifacts.txt 2>&1
echo "[m3c $(date +%T)] done: $R"
