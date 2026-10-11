#!/usr/bin/env bash
# Builds the D2 (Kandinsky-6 video VAE) qualification kit on the dev host (x86-64, no GPU), for the GB10 jobs
# run-oracle.sh and run-engine.sh. Heavy: run it under the host's heavy-job gate.
#   KIT=DIR tools/spark/k6-video-vae/build-kit.sh
# KIT must already hold pybundle/, pybundle.json, upstream/ and inputs/ (pybundle.sh, the pinned upstream checkout,
# inputs.py). This adds:
#   bin/k6v-qual          zig build -Dtarget=aarch64-linux-gnu in src/engines/kandinsky6/video_vae (ReleaseSafe)
#   fatbin/{conv,ops}.fatbin  kernels/cuda/kandinsky6/video_vae/*.cu through kernels/cuda/nvcc-docker.sh, sm_121a + sm_120a
#   tools/                this directory's scripts, as committed
#   build.json            the repo head, tree state, zig and nvcc versions
#   MANIFEST.sha256       every kit file except pybundle/ (pinned by pybundle/MANIFEST.sha256, listed here) and vae/;
#                         its own sha256 is the kit's identity
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
KIT=${KIT:?KIT=DIR}
VV=$REPO/src/engines/kandinsky6/video_vae
KS=$REPO/kernels/cuda/kandinsky6/video_vae
for d in pybundle upstream inputs; do [[ -e $KIT/$d ]] || { echo "KIT lacks $d/" >&2; exit 1; }; done
mkdir -p $KIT/bin $KIT/fatbin $KIT/tools
OBJ=$REPO/zig-out/k6v-kit   # under the repo: nvcc-docker.sh mounts only the repo
mkdir -p $OBJ

(cd $VV && zig build -Dtarget=aarch64-linux-gnu --prefix $OBJ/zig)
install -m 0755 $OBJ/zig/bin/k6v-qual $KIT/bin/k6v-qual

FLAGS=(-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__
       -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 -fatbin
       -gencode=arch=compute_121a,code=sm_121a -gencode=arch=compute_120a,code=sm_120a -Xptxas -v)
for f in conv ops; do
    (cd $REPO && kernels/cuda/nvcc-docker.sh "${FLAGS[@]}" -o $OBJ/$f.fatbin $KS/$f.cu) 2>&1 | tee $OBJ/$f.nvcc.log
    install -m 0644 $OBJ/$f.fatbin $KIT/fatbin/$f.fatbin
done
NVCC=$(cd $REPO && kernels/cuda/nvcc-docker.sh --version | tail -2 | tr '\n' ' ')

for f in inputs.py oracle.py compare.py pybundle.sh run-oracle.sh run-engine.sh build-kit.sh; do
    install -m 0644 $REPO/tools/spark/k6-video-vae/$f $KIT/tools/$f
done
chmod 0755 $KIT/tools/*.sh

cd $REPO
cat > $KIT/build.json <<EOF
{"head": "$(git rev-parse HEAD)", "dirty": $([[ -z $(git status --porcelain -- src/engines/kandinsky6 kernels/cuda/kandinsky6 tools/spark/k6-video-vae) ]] && echo false || echo true),
 "zig": "$(zig version)", "nvcc": "$NVCC", "nvcc_image": "${STK_NVCC_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}",
 "target": "aarch64-linux-gnu", "gencode": ["sm_121a", "sm_120a"]}
EOF
cd $KIT
find bin fatbin tools inputs upstream build.json pybundle.json pybundle/MANIFEST.sha256 -type f -not -path '*/.git/*' \
    -not -name '*.pyc' | LC_ALL=C sort | xargs -d '\n' sha256sum > MANIFEST.sha256
echo "{\"kit\": \"$KIT\", \"manifest_sha256\": \"$(sha256sum MANIFEST.sha256 | cut -d' ' -f1)\", \"files\": $(wc -l < MANIFEST.sha256)}"
