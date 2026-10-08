#!/usr/bin/env bash
# V3 (one PRO 6000): the twin's GPU tests (elementwise ops, 32B text encoder, audio and video VAEs against ComfyUI),
# the H3 shipping pack (Turbo 8-step LoRA merged in fp32, static NVFP4 scales calibrated on bf16 trajectories) and one
# recorded shipping-form step (text encoder included) for the Zig replays; pack and capture kept on the volume. Then
# the Zig build, `localrouter check h3-te`, `h3-replay`, `h3-avae` and `h3-vvae` on them (VAE captures: one decode each). Last the
# whole pipeline: the twin's `h3.generate` (text to video + audio, 768x448, 56 frames, 8 steps, light recording) and
# `localrouter check h3-e2e` replaying it in Zig (ids, light ops chained, frames' sha256, waveform).
set -uo pipefail
source "$(dirname "$0")/video-env.sh"
R=$STK/results/v3-$(date -u +%Y%m%d-%H%M%S); mkdir -p $R; ln -sfn $R $STK/results/v3-latest
run() { local name=$1; shift; echo "[v3 $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    tail -n 1 $R/$name.log | grep '^{' >> $R/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; return $rc; }
run setup bash $TWIN/pod/video-setup.sh
source $TWIN/pod/video-env.sh
P=/tmp/localrouter/packs/h3-turbo-nvfp4; C=/tmp/localrouter/captures/h3-step1; CA=/tmp/localrouter/captures/h3-avae; CV=/tmp/localrouter/captures/h3-vvae; CG=/tmp/localrouter/captures/h3-gen
VA=$VMODELS/vae/minimax_h3_audio_vae_fp32.safetensors; VV=$VMODELS/vae/minimax_h3_video_vae_fp16.safetensors
TE=$VMODELS/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
run test-h3-ops python -m stk_twin.h3.test_ops
run test-te32 python -m stk_twin.h3.test_te32 --ckpt $TE
run test-vae-audio python -m stk_twin.h3.test_vae_audio
run test-vae-video python -m stk_twin.h3.test_vae_video
run build python -m stk_twin.h3.build --models $VMODELS --lora --out $P
run capture python -m stk_twin.h3.capture --models $VMODELS --pack $P --out $C
run capture-avae python -m stk_twin.h3.capture_avae --models $VMODELS --out $CA
run capture-vvae python -m stk_twin.h3.capture_vvae --models $VMODELS --out $CV
( mkdir -p $STK/packs && cp -r $P $STK/packs/ ) &
SRC=/tmp/localrouter/stk-src; rm -rf $SRC; mkdir -p $SRC; cp -r $STK/repo/. $SRC/; cd $SRC
export ZIG_GLOBAL_CACHE_DIR=$STK/zig-cache
run zig-build $STK/zig/zig build -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121,120 --prefix $R/out fatbins install
run h3-te $R/out/bin/localrouter check h3-te $TE $C
run h3-replay $R/out/bin/localrouter check h3-replay $P $C
run h3-avae $R/out/bin/localrouter check h3-avae $VA $CA
run h3-vvae $R/out/bin/localrouter check h3-vvae $VV $CV
rm -rf $CG
run generate python -m stk_twin.h3.generate --models $VMODELS --pack $P --out $CG
run h3-e2e $R/out/bin/localrouter check h3-e2e $P $TE $VA $VV $CG
wait
echo "[v3 $(date +%T)] done: $R"
