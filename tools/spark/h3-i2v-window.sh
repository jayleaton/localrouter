#!/usr/bin/env bash
# Sourced by h3-inner.sh for H3_I2V=1; reuses run/cap/drop, model paths, pack and the built binary.
# Stage a real PNG at $LOCALROUTER_HOME/input/h3-i2v.png before the window. The container path defaults to
# /workspace/stk/input/h3-i2v.png (H3_I2V_IMAGE overrides). H3_I2V_SIZE=768x448, H3_I2V_FRAMES=56,
# H3_I2V_STEPS=8 and H3_I2V_SEED=7 (H3_I2V_PROMPT overrides the capture/generation prompt); ONLY selects the step names below. JPEG/WebP are unsupported.
# The ComfyUI gates measure numerical agreement, not bit equality; engine replay must match the twin's bits.
# Full capture includes every encoder/vision/text/refiner/DiT op and can be large. KEEP_CAPTURES=1 preserves it.
# render-h3-i2v keeps an MP4, interleaved stereo waveform, source PNG and result.json under the results directory;
# ffmpeg with libx264/AAC is required. A mux failure preserves the raw generation capture for investigation.
# No window is opened here: h3-window.sh owns the exclusive GPU window and restores it on exit.

h3_i2v_render() {
    local capture=$1 results=$2 size=$3
    [[ -s $capture/frames.u8 && -s $capture/audio.f32 ]] || return 1
    # The NGC image may lack ffmpeg; apt installs it (only the MP4 needs it).
    command -v ffmpeg > /dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null; } || return 1
    # Twin waveform bytes are [1,2,N] planar; raw f32le input to ffmpeg needs interleaved stereo.
    python - "$capture/audio.f32" "$results/h3-i2v-audio.f32" <<'PYMUX'
import sys
import numpy as np
samples=np.fromfile(sys.argv[1],dtype=np.float32)
if not len(samples) or len(samples)%2: raise SystemExit("invalid planar stereo waveform")
samples.reshape(2,-1).T.copy().tofile(sys.argv[2])
PYMUX
    local rc=$?
    [[ $rc == 0 ]] || return "$rc"
    ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size "$size" -framerate 24 -i "$capture/frames.u8" \
        -f f32le -ar 32000 -ac 2 -i "$results/h3-i2v-audio.f32" -c:v libx264 -pix_fmt yuv420p \
        -c:a aac -shortest -movflags +faststart "$results/h3-i2v.mp4" || return $?
    cp "$capture/result.json" "$results/h3-i2v-result.json" || return $?
    cp "$capture/input.png" "$results/h3-i2v-input.png" || return $?
    echo '{"pass":true,"artifact":"h3-i2v.mp4","fps":24,"sample_rate":32000,"channels":2}'
}

h3_i2v_window() {
    local image=${H3_I2V_IMAGE:-$STK/input/h3-i2v.png}
    local size=${H3_I2V_SIZE:-768x448} frames=${H3_I2V_FRAMES:-56}
    local steps=${H3_I2V_STEPS:-8} seed=${H3_I2V_SEED:-7}
    local ci=$CAP/h3-i2v cg=$CAP/h3-i2v-gen
    local width=${size%x*} height=${size#*x}
    if [[ ! -s $image && ${SKIP_CAPTURE:-0} == 1 ]]; then
        [[ ! -s $ci/input.png ]] || image=$ci/input.png
        [[ -s $image || ! -s $cg/input.png ]] || image=$cg/input.png
    fi
    if [[ ! -s $image ]]; then
        ONLY= run h3-i2v-input bash -c 'echo "{\"pass\":false,\"error\":\"stage a real input PNG under the mounted input directory\"}"; exit 1'
        return 1
    fi
    if [[ ! $width =~ ^[0-9]+$ || ! $height =~ ^[0-9]+$ ]]; then
        ONLY= run h3-i2v-input bash -c 'echo "{\"pass\":false,\"error\":\"H3_I2V_SIZE must be WIDTHxHEIGHT\"}"; exit 1'
        return 1
    fi
    if [[ ! $steps =~ ^[0-9]+$ ]] || (( 10#$steps < 2 )); then
        ONLY= run h3-i2v-input bash -c 'echo "{\"pass\":false,\"error\":\"H3_I2V_STEPS must be at least 2 to gate the distinct conditioning timestep\"}"; exit 1'
        return 1
    fi
    # Preserve the actual input outside capture directories before cap clears selected fresh runs.
    [[ $(realpath "$image") == $(realpath -m "$R/h3-i2v-input.png") ]] || cp "$image" "$R/h3-i2v-input.png" || return $?
    image=$R/h3-i2v-input.png
    # One-tile encoder gate and a tiled canvas with nonuniform trailing overlaps. The test also exercises
    # structural cases at 32x48, 256x256, 448x768 and 768x1344, independent of the real-weight gate size.
    if [[ ${SKIP_TWIN_TESTS:-0} != 1 ]]; then
        if [[ $WITH_COMFY == 1 ]]; then
            run test-h3-i2v-vae-single env VVAE="$VV" VENC_H=96 VENC_W=128 python -m stk_twin.h3.test_vae_video_enc
            run test-h3-i2v-vae-tiled env VVAE="$VV" VENC_H=448 VENC_W=768 python -m stk_twin.h3.test_vae_video_enc
            run test-h3-i2v-position python -m stk_twin.h3.test_vision_position --comfy "$COMFY"
            run test-h3-i2v-vision python -m stk_twin.h3.test_te_vision --ckpt "$TE" --image "$image" --size "$size"
            run test-h3-i2v-dit python -m stk_twin.h3.gate --models "$VMODELS" --image "$image" --size "$size" \
                --frames "$frames" --steps "$steps" --seed "$seed" --attn bf16
        else
            run h3-i2v-references bash -c 'echo "{\"pass\":false,\"error\":\"H3 image conditioning requires WITH_COMFY=1 for the reference gates\"}"; exit 1'
        fi
    fi
    cap capture-i2v "$ci" python -m stk_twin.h3.capture_i2v --models "$VMODELS" --pack "$P" --out "$ci" \
        --image "$image" --size "$size" --frames "$frames" --steps "$steps" --step 1 --seed "$seed" ${H3_I2V_PROMPT:+--prompt "$H3_I2V_PROMPT"}
    run h3-i2v "$S" check h3-i2v "$P" "$TE" "$VV" "$ci" && drop "$ci"
    cap generate-i2v "$cg" python -m stk_twin.h3.generate --models "$VMODELS" --pack "$P" --out "$cg" \
        --image "$image" --size "$size" --frames "$frames" --steps "$steps" --seed "$seed" ${H3_I2V_PROMPT:+--prompt "$H3_I2V_PROMPT"}
    run h3-e2e-i2v "$S" check h3-e2e "$P" "$TE" "$VA" "$VV" "$cg"
    local replay_rc=$?
    run render-h3-i2v h3_i2v_render "$cg" "$R" "$size"
    local render_rc=$?
    [[ $replay_rc != 0 || $render_rc != 0 ]] || drop "$cg"
}
h3_i2v_window
