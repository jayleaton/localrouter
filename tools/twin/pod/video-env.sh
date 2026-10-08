# Sourced by every video twin job on a pod: the image twin's env plus the video stack's paths and pins.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export VVENV=$STK/venv-video
export VMODELS=/tmp/localrouter/models/minimax-h3              # local disk
export VMODELS_VOL=$STK/models/minimax-h3           # the volume's copy
# WITH_COMFY=1: also ComfyUI 0.37.0 + comfy-kitchen, the REFERENCE for gate.py, test_ops / test_vae_* / test_kitchen and
# comfy_run.py. The twin's build / capture / generate path (stk_twin.h3.build, .capture, .generate, the captures) does not
# need it: no $COMFY, no comfy on PYTHONPATH. Default 0 here; tools/spark/h3-inner.sh defaults it to 1 for now.
export WITH_COMFY=${WITH_COMFY:-0}
if [[ $WITH_COMFY == 1 ]]; then export COMFY=$STK/src/ComfyUI; else unset COMFY; fi   # v0.37.0
export TFVIDEO_REV=711d942                          # jayleaton/minimax-h3-tensorfold-rtx
export KITCHEN_VERSION=0.2.35                       # comfy-kitchen, ComfyUI 0.37.0's pin
export PYTHONPATH=$TWIN${COMFY:+:$COMFY}:$STK/src/minimax-h3-tensorfold-rtx${PYTHONPATH:+:$PYTHONPATH}
if [[ -f $VVENV/bin/activate ]]; then source $VVENV/bin/activate; fi
