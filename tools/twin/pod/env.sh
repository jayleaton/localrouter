# Sourced by every twin job on a pod: paths on the network volume, pinned sources, the venv.
export STK=/workspace/localrouter
export TWIN=$STK/twin
export STK_KROOT=$STK/kernels                      # LocalRouter's kernels/: .cu shared with Zig, frozen tables
export MODEL_VOL=$STK/models/Qwen-Image-2.1          # the volume copy (reads at ~20 MB/s by mmap)
export MODEL=/tmp/localrouter/models/Qwen-Image-2.1          # local NVMe copy, made by pod/local.sh
export HF_HOME=$STK/hf-cache
export TORCH_EXTENSIONS_DIR=$STK/torch-ext       # TensorFold's JIT kernels, built once per GPU architecture
export TFIMAGE_REV=f91845527f095bbc1b87bd9b3960f893e7c9efc1   # jayleaton/qwen-image21-tensorfold-rtx
export TENSORFOLD_TAG=v0.6.1                      # ashhart/TensorFold, 17c73e1
export PYTHONPATH=$TWIN:$STK/src/qwen-image21-tensorfold-rtx${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export PATH=/usr/local/cuda/bin:$PATH                # ssh sessions miss it; Triton needs ptxas
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export CPATH=/usr/local/cuda/include${CPATH:+:$CPATH}       # Triton builds its cuda_utils with gcc and needs cuda.h
if [[ -f $STK/venv/bin/activate ]]; then source $STK/venv/bin/activate; fi
