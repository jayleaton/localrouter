#!/usr/bin/env bash
# Builds the D2 (Kandinsky-6 video VAE) qualification kit on the dev host (x86-64, no GPU), for the GB10 jobs
# run-probe.sh, run-oracle.sh and run-engine.sh. Heavy: run it under the host's heavy-job gate.
#   KIT=DIR tools/spark/k6-video-vae/build-kit.sh
# KIT must already hold pybundle/, pybundle.json, upstream/ and inputs/ (pybundle.sh, the pinned upstream checkout,
# inputs.py). This adds:
#   bin/k6v-qual          zig build -Dtarget=aarch64-linux-gnu in src/engines/kandinsky6/video_vae (ReleaseSafe)
#   fatbin/{conv,ops}.fatbin  kernels/cuda/kandinsky6/video_vae/*.cu through kernels/cuda/nvcc-docker.sh, sm_121a + sm_120a
#   tools/                this directory's scripts, as committed
#   build.json            the repo head, tree state, zig and nvcc versions
#   MANIFEST.sha256       every kit file except pybundle/ (pinned by pybundle/MANIFEST.sha256, listed here) and vae/;
#                         its own sha256 is the kit's identity
#   $KIT.operator-spec.json (beside the kit) the D2 jobs for the shared staging sealer (tools/kandinsky6/staging/seal.py): per-node
#                         immutable image IDs (NODE_IMAGES, the nodes' `docker image inspect` receipt), artifacts with
#                         sha256, the vae assets, budgets; the oracle jobs name the original pybundle/ (pinned there,
#                         checked against this kit's) as the runner's read-only D2 oracle root.
# TOOLS_ONLY=1 restages tools/ and the spec over an existing kit without rebuilding bin/ or fatbin/, which must still
# match MANIFEST.sha256 (the checked binary and kernels stay byte-identical).
# FATBIN_FROM=DIR (a new KIT, bin/ rebuilt) takes fatbin/ from an earlier kit instead of nvcc, when the kernel sources are
# unchanged since that kit's head: its fatbins are checked against its MANIFEST.sha256 and its nvcc version is kept.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
KIT=${KIT:?KIT=DIR}
VV=$REPO/src/engines/kandinsky6/video_vae
KS=$REPO/kernels/cuda/kandinsky6/video_vae
for d in pybundle upstream inputs; do [[ -e $KIT/$d ]] || { echo "KIT lacks $d/" >&2; exit 1; }; done
NODE_IMAGES=${NODE_IMAGES:-/home/sireggy/Documents/.t3/k6-staging/node-image-inspect.json}
mkdir -p $KIT/bin $KIT/fatbin $KIT/tools
if [[ ${TOOLS_ONLY:-0} == 1 ]]; then
    (cd $KIT && grep -E '  (bin|fatbin)/' MANIFEST.sha256 | sha256sum -c --strict --quiet) \
        || { echo "TOOLS_ONLY: bin/ or fatbin/ differ from MANIFEST.sha256" >&2; exit 1; }
    NVCC=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["nvcc"])' $KIT/build.json)
else
OBJ=$REPO/zig-out/k6v-kit   # under the repo: nvcc-docker.sh mounts only the repo
mkdir -p $OBJ

(cd $VV && zig build -Dtarget=aarch64-linux-gnu --prefix $OBJ/zig)
install -m 0755 $OBJ/zig/bin/k6v-qual $KIT/bin/k6v-qual

if [[ -n ${FATBIN_FROM:-} ]]; then
    OLD=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["head"])' $FATBIN_FROM/build.json)
    [[ -z $(git -C $REPO diff --name-only $OLD HEAD -- kernels/cuda/kandinsky6/video_vae) && -z $(git -C $REPO status --porcelain -- kernels/cuda/kandinsky6/video_vae) ]] \
        || { echo "FATBIN_FROM: kernel sources changed since $OLD" >&2; exit 1; }
    (cd $FATBIN_FROM && grep -E '  fatbin/' MANIFEST.sha256 | sha256sum -c --strict --quiet) \
        || { echo "FATBIN_FROM: fatbin/ differs from its MANIFEST.sha256" >&2; exit 1; }
    install -m 0644 $FATBIN_FROM/fatbin/conv.fatbin $FATBIN_FROM/fatbin/ops.fatbin $KIT/fatbin/
    NVCC=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["nvcc"])' $FATBIN_FROM/build.json)
else

FLAGS=(-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__
       -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 -fatbin
       -gencode=arch=compute_121a,code=sm_121a -gencode=arch=compute_120a,code=sm_120a -Xptxas -v)
for f in conv ops; do
    (cd $REPO && kernels/cuda/nvcc-docker.sh "${FLAGS[@]}" -o $OBJ/$f.fatbin $KS/$f.cu) 2>&1 | tee $OBJ/$f.nvcc.log
    install -m 0644 $OBJ/$f.fatbin $KIT/fatbin/$f.fatbin
done
NVCC=$(cd $REPO && kernels/cuda/nvcc-docker.sh --version | tail -2 | tr '\n' ' ')
fi
fi

chmod -R u+w $KIT/tools
for f in inputs.py oracle.py compare.py operator_spec.py pybundle.sh run-oracle.sh run-engine.sh run-probe.sh \
         d2-allocator-guard.py d2_evidence.py D2-EVIDENCE.md build-kit.sh; do
    install -m 0644 $REPO/tools/spark/k6-video-vae/$f $KIT/tools/$f
done
chmod 0755 $KIT/tools/*.sh

chmod u+w $KIT $KIT/MANIFEST.sha256 $KIT/build.json 2>/dev/null || true
rm -f $KIT/SEAL.json   # a seal names one manifest; a restaged kit is sealed again after its commit
cd $REPO
cat > $KIT/build.json <<EOF
{"head": "$(git rev-parse HEAD)", "dirty": $([[ -z $(git status --porcelain -- src/engines/kandinsky6 kernels/cuda/kandinsky6 tools/spark/k6-video-vae) ]] && echo false || echo true),
 "zig": "$(zig version)", "nvcc": "$NVCC", "nvcc_image": "${STK_NVCC_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}",
 "target": "aarch64-linux-gnu", "gencode": ["sm_121a", "sm_120a"]}
EOF
cd $KIT
find bin fatbin tools inputs upstream build.json pybundle.json pybundle/MANIFEST.sha256 -type f -not -path '*/.git/*' \
    -not -name '*.pyc' | LC_ALL=C sort | xargs -d '\n' sha256sum > MANIFEST.sha256
python3 -I $REPO/tools/spark/k6-video-vae/operator_spec.py --kit $KIT --node-images $NODE_IMAGES \
    --assets /home/sireggy/Documents/.t3/k6-staging/assets-1018d582/progress.json > $KIT.operator-spec.json
echo "{\"kit\": \"$KIT\", \"manifest_sha256\": \"$(sha256sum MANIFEST.sha256 | cut -d' ' -f1)\", \"files\": $(wc -l < MANIFEST.sha256), \"operator_spec\": \"$KIT.operator-spec.json\", \"operator_spec_sha256\": \"$(sha256sum $KIT.operator-spec.json | cut -d' ' -f1)\"}"
