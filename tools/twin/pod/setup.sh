#!/usr/bin/env bash
# One-time (idempotent) setup on the volume: a venv over the image's torch, TensorFold, tfimage, the weights.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p $STK/src $STK/models
if [[ ! -f $STK/venv/bin/activate ]]; then
    python -m venv --system-site-packages $STK/venv
    source $STK/venv/bin/activate
    # the image's torch is a pre-release pip would replace: pin it, triton and numpy to what is installed
    pip freeze --all | grep -iE '^(torch|triton|numpy)==' > $STK/venv/constraints.txt
    pip install -q -c $STK/venv/constraints.txt "diffusers==0.41.0" "transformers>=4.57" accelerate
fi
[[ -d $STK/src/TensorFold ]] || git clone -q --depth 1 --branch $TENSORFOLD_TAG https://github.com/ashhart/TensorFold $STK/src/TensorFold
pip install -q --no-deps -e $STK/src/TensorFold
if [[ ! -d $STK/src/qwen-image21-tensorfold-rtx ]]; then
    git clone -q https://github.com/jayleaton/qwen-image21-tensorfold-rtx $STK/src/qwen-image21-tensorfold-rtx
    git -C $STK/src/qwen-image21-tensorfold-rtx checkout -q $TFIMAGE_REV
fi
[[ -f $MODEL_VOL/.complete ]] || { hf download Qwen/Qwen-Image-2.1 --local-dir $MODEL_VOL --max-workers 16 > /dev/null && touch $MODEL_VOL/.complete; }
python - <<'PY'
import torch, triton, diffusers, transformers, tensorfold
print({"torch": torch.__version__, "triton": triton.__version__, "diffusers": diffusers.__version__,
       "transformers": transformers.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()})
PY
du -sh $MODEL_VOL
