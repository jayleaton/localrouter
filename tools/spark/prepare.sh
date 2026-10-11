#!/usr/bin/env bash
# The weights half of `install.sh`, inside the NGC PyTorch container (the twin's PyTorch) with $LOCALROUTER_HOME at
# /workspace/localrouter. MODELS: "image", "video" or both (default). Idempotent: a finished part (its manifest present) is skipped.
#   image: the official Qwen-Image 2.1 checkpoint downloaded once, then the packs the image tool serves (checked byte for
#          byte against the committed digests) and each precision's Triton cubins for this GPU (PRECISIONS: "fp8s", the
#          default; "fp8s nvfp4" adds the faster NVFP4). Result: weights/qwen-image-2.1/ (fp8s/, nvfp4/, te/, vae/).
#          IMAGE_MODELS ("qwen-image-2.1 qwen-image-2.1-turbo", the default; either alone) picks the checkpoints. Turbo
#          downloads only its own files (the DiT, the scheduler, model_index.json with its 8 sample_sigmas): its text
#          encoder, processor and VAE are Qwen-Image 2.1's tensors, so they are linked from that snapshot and its te/ and
#          vae/ packs serve both. Its DiT packs are turbo-<precision>/ beside the others (for each precision with
#          committed scales: acts-<p>-turbo.json and digests-turbo-<p>.json).
#   video: the MiniMax H3 checkpoints (Comfy-Org/MiniMax-H3, about 61 GB) into models/minimax-h3/ in the layout of
#          tools/twin/pod/video-setup.sh (diffusion_models, text_encoders, vae, loras; resumable, existing files kept), the
#          video venv (WITH_COMFY=0: no ComfyUI is needed to build), and the Turbo-LoRA NVFP4 pack (python -m
#          stk_twin.h3.build, about 15 minutes). Result: weights/minimax-h3/, which the minimax_h3 engine reads as its
#          `weights` directory: h3-turbo-nvfp4/ (the pack) beside text_encoders/ and vae/. The encoder and the VAEs are
#          hard links to the downloaded files (symlinks would point out of weights/, which is all the service container
#          mounts, at /models); the DiT and LoRA checkpoints stay in models/ and are only needed to build the pack.
set -euo pipefail
export STK=/workspace/localrouter
MODELS=${MODELS:-image video}
want() { [[ " $MODELS " == *" $1 "* ]]; }
source $STK/twin/pod/env.sh
step() { echo "[prepare $(date +%T)] $*"; }
for m in $MODELS; do [[ $m == image || $m == video ]] || { echo "prepare: MODELS has '$m'; use image, video or both" >&2; exit 2; }; done

# ---- image
if want image; then
bash $TWIN/pod/setup.sh > /dev/null
source $STK/twin/pod/env.sh   # the venv exists only now on a fresh $LOCALROUTER_HOME
export MODEL=$MODEL_VOL STK_ATTN=stk STK_OPS=stk STK_TE=stk STK_VAE=stk
W=$STK/weights/qwen-image-2.1; PK=$STK/repo/tools/twin/packs; C=$STK/captures
T="python -m stk_twin"
mkdir -p $W

IMAGE_MODELS=${IMAGE_MODELS:-qwen-image-2.1 qwen-image-2.1-turbo}
wantimg() { [[ " $IMAGE_MODELS " == *" $1 "* ]]; }
for p in ${PRECISIONS:-fp8s}; do
    wantimg qwen-image-2.1 || continue
    pp=${p%s}   # the pack command's name: nvfp4 or fp8
    if [[ ! -f $W/$p/manifest.json ]]; then
        step "pack $p"
        $T pack --model $MODEL --precision $pp --acts $PK/acts-$pp.json --out $W/$p
    fi
    python $PK/check.py $W/$p $PK/digests-$p.json > /dev/null || { echo "pack $p differs from the reference digests" >&2; exit 1; }
    if [[ ! -f $W/$p/triton/triton.json ]]; then   # tfimage's Triton kernels, compiled for this GPU by one short capture
        step "triton $p"
        rm -rf $C/prepare-$p
        $T capture --model $MODEL --pack $W/$p --size 512x512 --steps 2 --out $C/prepare-$p > /dev/null
        $T triton --model $MODEL --pack $C/prepare-$p --out $W/$p
        rm -rf $C/prepare-$p
    fi
done
if wantimg qwen-image-2.1-turbo; then
    TW=$STK/models/Qwen-Image-2.1-Turbo; TURBO_REV=d65dbc9a7e8f6b5479e33dee6030eaab2a906509
    if [[ ! -f $TW/.complete ]]; then
        step "download Qwen-Image-2.1-Turbo (its own files, about 14 GB)"
        hf download Qwen/Qwen-Image-2.1-Turbo --revision $TURBO_REV --include "transformer/*" "scheduler/*" model_index.json LICENSE README.md \
            --local-dir $TW --max-workers 16 > /dev/null && touch $TW/.complete
    fi
    for d in text_encoder processor vae; do [[ -e $TW/$d ]] || ln -s ../Qwen-Image-2.1/$d $TW/$d; done
    for p in ${PRECISIONS:-fp8s}; do
        pp=${p%s}
        [[ -f $PK/acts-$pp-turbo.json ]] || { echo "prepare: no committed Turbo scales for $p (acts-$pp-turbo.json); skipping turbo-$p" >&2; continue; }
        if [[ ! -f $W/turbo-$p/manifest.json ]]; then
            step "pack turbo-$p"
            $T pack --model $TW --repo Qwen/Qwen-Image-2.1-Turbo --precision $pp --acts $PK/acts-$pp-turbo.json --out $W/turbo-$p
        fi
        python $PK/check.py $W/turbo-$p $PK/digests-turbo-$p.json > /dev/null || { echo "pack turbo-$p differs from the reference digests" >&2; exit 1; }
        if [[ ! -f $W/turbo-$p/triton/triton.json ]]; then
            step "triton turbo-$p"
            rm -rf $C/prepare-turbo-$p
            $T capture --model $TW --pack $W/turbo-$p --size 512x512 --out $C/prepare-turbo-$p > /dev/null
            $T triton --model $TW --pack $C/prepare-turbo-$p --out $W/turbo-$p
            rm -rf $C/prepare-turbo-$p
        fi
    done
fi
[[ -f $W/te/manifest.json ]] || { step "pack te"; $T pack --model $MODEL --precision te --out $W/te; }
[[ -f $W/vae/manifest.json ]] || { step "pack vae"; $T pack --model $MODEL --precision vae --out $W/vae; }
python -c "import sys; sys.path.insert(0, '$TWIN'); from stk_twin.files import readable; readable('$W')"
step "image done: $(du -sh $W | cut -f1) in $W"
fi

# ---- video
if want video; then
V=$STK/models/minimax-h3            # the downloads (video-setup.sh's volume copy)
H=$STK/weights/minimax-h3           # what the engine reads
PACK=$H/h3-turbo-nvfp4
CKPT=(diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
      text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
      vae/minimax_h3_video_vae_fp16.safetensors
      vae/minimax_h3_audio_vae_fp32.safetensors
      loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors)
SHIPPED=("${CKPT[@]:1:3}")          # the files the engine reads beside the pack
pack_done() { [[ -f $PACK/manifest.json && -f $PACK/tokenizer.json ]]; }
shipped_done() { local f; for f in "${SHIPPED[@]}"; do [[ -s $H/$f ]] || return 1; done; }
have_ckpt() { local f; for f in "${CKPT[@]}"; do [[ -s $V/$f ]] || return 1; done; }

if ! { pack_done && shipped_done; }; then
    # video-setup.sh with its weights on the volume ($STK/models/minimax-h3) rather than the pod's /root, as h3-inner.sh
    # patches it: the patched copy of pod/ is the one run (video-setup.sh sources the env file beside itself)
    POD=/tmp/h3-pod
    rm -rf $POD; cp -r $TWIN/pod $POD
    sed -i 's#^export VMODELS=/tmp/localrouter/models/minimax-h3 .*#export VMODELS=$STK/models/minimax-h3#' $POD/video-env.sh
    grep -q '^export VMODELS=\$STK/' $POD/video-env.sh || { echo "prepare: video-env.sh VMODELS patch failed" >&2; exit 1; }
    export WITH_COMFY=0
    # the venv, TensorFold and tfvideo, and the five checkpoints (hf download into a scratch directory, then a rename:
    # a half-downloaded file never has its final name, a re-run resumes it and skips the finished ones). It does not
    # fail on a dead download, so look at the files and go again (3 tries) before giving up.
    for try in 1 2 3; do
        step "video: venv and checkpoints (try $try)"
        bash $POD/video-setup.sh || true
        have_ckpt && break
    done
    have_ckpt || { echo "prepare: the MiniMax-H3 checkpoints are not all in $V (network? disk?):" >&2
                   for f in "${CKPT[@]}"; do [[ -s $V/$f ]] || echo "  missing $f" >&2; done; exit 1; }
    source $POD/video-env.sh   # again: on a fresh $LOCALROUTER_HOME the venv exists only now
    python -c "import torch, tensorfold.cuda.nvfp4.checkpoint" || { echo "prepare: the video venv is not usable" >&2; exit 1; }

    if ! pack_done; then
        step "pack h3-turbo-nvfp4 (the Turbo LoRA merged, NVFP4 scales calibrated; about 15 minutes)"
        rm -rf $PACK   # a half-written pack from an earlier try
        mkdir -p $H
        python -m stk_twin.h3.build --models $VMODELS --lora --out $PACK
        pack_done || { echo "prepare: the H3 build left no manifest and tokenizer in $PACK" >&2; exit 1; }
    fi

    # the engine's layout: text_encoders/ and vae/ beside the pack, as real files inside weights/ (hard links, so no
    # second copy; a copy where the two directories are on different filesystems)
    for f in "${SHIPPED[@]}"; do
        mkdir -p $H/$(dirname $f)
        [[ -s $H/$f ]] && continue
        ln -f $V/$f $H/$f.part 2> /dev/null || cp $V/$f $H/$f.part
        mv $H/$f.part $H/$f
    done
    shipped_done || { echo "prepare: $H lacks the encoder or VAE files" >&2; exit 1; }
fi
python -c "import sys; sys.path.insert(0, '$TWIN'); from stk_twin.files import readable; readable('$H')"
step "video done: $(du -shL $H | cut -f1) in $H (the checkpoints in $V are only needed to rebuild the pack)"
fi
