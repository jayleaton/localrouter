#!/usr/bin/env bash
# The video twin's environment (idempotent), beside the image twin's: a venv over the image's torch with the twin's own
# requirements (safetensors, transformers, tokenizers, numpy) and tfvideo (jayleaton/minimax-h3-tensorfold-rtx 711d942);
# the MiniMax H3 weights on the pod's local disk (and once on the volume). Sourced env: video-env.sh.
# WITH_COMFY=1 adds the reference stack: ComfyUI 0.37.0's requirements and checkout, comfy-kitchen 0.2.35. Only the
# comparison tests need it (gate.py, test_ops, test_vae_audio, test_vae_video, test_kitchen, comfy_run.py); the build /
# capture / generate path does not.
set -uo pipefail
source "$(dirname "$0")/video-env.sh"
mkdir -p $STK/src $VMODELS_VOL
if [[ ! -f $VVENV/bin/activate ]]; then
    python -m venv --system-site-packages $VVENV
    source $VVENV/bin/activate
    pip freeze --all | grep -iE '^(torch|torchvision|torchaudio|triton|numpy)==' > $VVENV/constraints.txt
fi
source $VVENV/bin/activate
if [[ $WITH_COMFY == 1 ]]; then
    [[ -d $STK/src/ComfyUI ]] || git clone -q --depth 1 --branch v0.37.0 https://github.com/comfyanonymous/ComfyUI $STK/src/ComfyUI
    # ComfyUI's requirements without torch itself (the image's) and the GUI-only ones. Its web frontend, workflow templates
    # and docs (hundreds of MB of UI assets) only matter to run the ComfyUI server (stk_twin.comfy_run): COMFY_SERVER=1.
    skip='torch|torchvision|torchaudio|comfy-angle|PyOpenGL'
    [[ ${COMFY_SERVER:-0} == 1 ]] || skip="$skip|comfyui-frontend-package|comfyui-workflow-templates|comfyui-embedded-docs"
    grep -vE "^($skip)([<>=~ ]|\$)" $STK/src/ComfyUI/requirements.txt > $VVENV/requirements.txt
    pip install -q -c $VVENV/constraints.txt -r $VVENV/requirements.txt
else
    # the twin's own requirements (the lower bounds ComfyUI 0.37.0's requirements.txt gives them; TODO: pin exact versions)
    printf '%s\n' 'numpy>=1.25.0' 'safetensors>=0.4.2' 'transformers>=4.50.3' 'tokenizers>=0.13.3' huggingface_hub > $VVENV/requirements.txt
    pip install -q -c $VVENV/constraints.txt -r $VVENV/requirements.txt
fi
python -c "import torchaudio" 2>/dev/null || cp -r $TWIN/pod/torchaudio_stub/torchaudio "$(python -c 'import site; print(site.getsitepackages()[0])')/"
# TensorFold (the NVFP4 checkpoint format the pack writer and tfvideo use), as the image twin's setup installs it
[[ -d $STK/src/TensorFold ]] || git clone -q --depth 1 --branch $TENSORFOLD_TAG https://github.com/ashhart/TensorFold $STK/src/TensorFold
python -c "import tensorfold" 2>/dev/null || pip install -q --no-deps -e $STK/src/TensorFold
[[ -d $STK/src/minimax-h3-tensorfold-rtx ]] || { git clone -q https://github.com/jayleaton/minimax-h3-tensorfold-rtx $STK/src/minimax-h3-tensorfold-rtx \
    && git -C $STK/src/minimax-h3-tensorfold-rtx checkout -q $TFVIDEO_REV; }
if [[ $WITH_COMFY == 1 ]]; then
    [[ -d $STK/src/comfy-kitchen ]] || { git clone -q https://github.com/Comfy-Org/comfy-kitchen $STK/src/comfy-kitchen \
        && git -C $STK/src/comfy-kitchen checkout -q v$KITCHEN_VERSION; }
    python - <<'PY'
import torch, comfy_kitchen, importlib.metadata as m
print({"torch": torch.__version__, "comfy_kitchen": m.version("comfy-kitchen"), "gpu": torch.cuda.get_device_name()})
from comfy_kitchen.sage_attention import is_available
print({"int8_attention": is_available(torch.device("cuda"))})
PY
else
    python -c "import torch; print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()})"
fi
# the weights: local disk first (fast), the volume keeps a copy for the next pod
fetch() { # REPO FILE SUBDIR
    local dst=$VMODELS/$3/$(basename $2)
    [[ -f $dst ]] && return 0
    if [[ -f $VMODELS_VOL/$3/$(basename $2) ]]; then mkdir -p $VMODELS/$3; cp $VMODELS_VOL/$3/$(basename $2) $dst; return; fi
    hf download $1 $2 --local-dir $VMODELS/.dl > /dev/null && mkdir -p $VMODELS/$3 && mv $VMODELS/.dl/$2 $dst
}
fetch Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors diffusion_models &
fetch Comfy-Org/MiniMax-H3 text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors text_encoders &
fetch Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_fp16.safetensors vae &
fetch Comfy-Org/MiniMax-H3 vae/minimax_h3_audio_vae_fp32.safetensors vae &
fetch Comfy-Org/MiniMax-H3 loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors loras &
wait
du -sh $VMODELS/*
# keep a copy on the volume for later pods (in the background; the job goes on)
( for d in diffusion_models text_encoders vae loras; do mkdir -p $VMODELS_VOL/$d; cp -n $VMODELS/$d/* $VMODELS_VOL/$d/; done
  touch $VMODELS_VOL/.complete ) > /dev/null 2>&1 &
