# Kandinsky-6 audio kernels: provenance and qualification state

`audio.cu`, `src/engines/kandinsky6/audio/{vae,vocoder,launch}.zig` and `src/engines/kandinsky6/audio/fixtures/` are
LocalRouter code (Apache-2.0). No third-party source text is copied or translated into them. They implement
published mathematical definitions and graph facts, recorded per operator in `vae.zig` (`contract`).

## Design sources

| Source | License | What was taken |
| --- | --- | --- |
| kandinskylab/kandinsky-6 `eac06e39`, `mmaudio/ext/autoencoder/vae.py`, `vae_modules.py`, `features_utils.py`, `algo/postprocess_audio.py` | MIT, KandinskyLab 2026 | Decoder graph: channels, block order, clamp placement, upsample, QKV reshape and axes, latent unscale, mel affine |
| HF `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers` `1018d582`, `audio_vae/config.json`, `vocoder/config.json` | model card MIT | Sizes: scaling 0.417, 44.1 kHz mono, mel 128, upsample rates 8/4/2/2/2/2, kernels 16/8/4/4/4/4, AMP kernels 3/7/11, dilations 1/3/5, SnakeBeta log scale, no tanh, no final bias |
| kandinsky-6 `mmaudio/ext/bigvgan_v2/bigvgan.py`, `activations.py` (NVIDIA BigVGAN v2) | MIT, NVIDIA 2024 (adapted from HiFi-GAN, MIT) | AMPBlock1 order, stage averaging, SnakeBeta expression, clamp at the end |
| kandinsky-6 `bigvgan_v2/alias_free_activation/torch/{resample,filter}.py` (from junjun3518/alias-free-torch) | Apache-2.0 | Activation1d padding/stride/crop arithmetic. The 12-tap filters are loaded from the checkpoint |
| lucidrains/denoising-diffusion-pytorch `c8155b1d` `karras_unet_1d.py` | MIT, Phil Wang 2020 | Scaled SiLU expression `silu(x) / 0.596`, residual expression `(a(1-t) + bt) / sqrt((1-t)^2 + t^2)` |
| PyTorch 2.14 conv1d / conv_transpose1d / clamp / interpolate docs | BSD-3-Clause | Primitive semantics: padding, output lengths, nearest-exact, NaN-propagating clamp |
| PyTorch 2.14.1 `aten/src/ATen/native/cuda/PersistentSoftmax.cuh` (`softmax_warp_forward`) | BSD-3-Clause | The fp32 softmax's order for rows of <= 1024: lane layout, max and sum butterflies, comparators, NaN for a zero sum (`k6a_sdpa_math`) |
| NVIDIA CUTLASS split-K GEMM partition | BSD-3-Clause | The K slice length `round_up(ceil(K / splits), 4)` for fp32 (`k6a_sdpa_math` `kslice`); the split count is the one cuBLASLt logged |

Restricted and not used: NVIDIA EDM2 `edm2_utils.py` (CC BY-NC-SA 4.0), imported by the native `MPConv1D`, `normalize`,
`mp_silu` and `mp_sum`. Its text was not read, copied, translated or called by this code. The CPU oracle fixtures in
`fixtures/` came from a black-box invocation in Captain's audio-closure research (synthetic inputs, no model weights).

The activation normalization and the stored -> prepared weight transform were narrowed only by black-box comparison:
compositions of public PyTorch primitives (`torch.linalg.vector_norm`, `torch.add(..., alpha=)`, true division) were
scored for exact bit equality against those fixtures, and the winning composition was then written independently in Zig
and CUDA. The restricted implementation was not read or imported for this. The CPU reduction orders (`vae.zig`
`NormReduce`) were found the same way, against `torch.linalg.vector_norm` on random rows. The GPU harness may import the
native module (and through it `edm2_utils.py`) only as the black-box reference it is, after checking its sha256; nothing
from it is copied.

Native load-path facts (MIT `autoencoder.py`, `vae_audio.py`, `pipeline/factory.py` at `eac06e39`): the decoder weights
are prepared once on the host CPU (`load_state_dict`, `remove_weight_norm`, then `.to(device)`). The vocoder loads the
HF `1018d582` checkpoint's plain `weight` tensors (667 fp32, no `weight_g` / `weight_v` / parametrizations) after removing
weight norm, so no fold runs.

If BigVGAN or alias-free-torch source is ever copied rather than reimplemented, the copy must carry the BigVGAN
`LICENSE` and `incl_licenses/LICENSE_1..8` closure. The release inventory decides which notices ship.

## Qualification state

The audio chain is qualified on SM121 (GB10). With the policies listed in `QUALIFICATION.md`, the full untraced run is
bit-equal to the pinned reference for latent, mel, wave and raw PCM, and so is every decoder and vocoder hook.
`QUALIFICATION.md` records the evidence, hashes, hardware, precision, seed and commands. Admission (`admission.zig`)
still lists the audio contracts as pending; that is a separate decision.

The notes below are how individual operators were narrowed before that gate. Where they disagree with
`QUALIFICATION.md`, the record is current. The FFT-engine and SDPA-GEMM records came from an earlier probe environment;
under the reference's pinned precision settings the originals ran the CUTLASS / xmma / legacy kernels that the record
lists, and the native paths reproduce those.

Native bits reproduced offline (diagnostic only; the kernels below are not yet run on SM121):

- activation normalization `x / (eps + ||x|| * sqrt(1/n))`, denominator in one rounding, true division. CPU oracle:
  serial reduction, 15/15 cases, 0 of 40462 values unequal. SM121 native: PyTorch's own CUDA reduction order
  (`k6a_l2norm_torch`, split per shape and device by vae.zig `torchSplit`) reproduces the D3 window's native output
  sha on all seven real shapes and the zero / tiny / impulse inputs; none of the `k6a_l2norm` orders does.
- decoder weight preparation `w / (eps + ||w_row|| * sqrt(1/fan)) / sqrt(fan)`, eps 1e-4 explicit, on the host at
  load. Native Grace host (aarch64): 4 fused lane sums added in order (`lanes4_fused`), fused denominator: 29/29
  decoder convs of the selected checkpoint sha-equal to the D3 window's native dumps. x86 AVX2 (8 lane sums):
  5/5 synthetic fixtures, diagnostic.
- magnitude-preserving residual: torch.lerp's small-weight form `fma(t, b - a, a)` times the fp32 reciprocal of
  sqrt((1-t)^2 + t^2) reproduces the native sha of the D3 window's [1, 2048, 436] case with specials
  (`k6a_mp_residual`); the earlier weighted-sum form did not. The second D3 job A window (`--probe`) ran the fixed
  kernels on SM121: the PyTorch-order normalization is zero unequal on all 11 cases and the lerp residual on its case.
- scaled SiLU: the probe dumps show torch's silu equals `x / (1 + expf(-x))` bit for bit and native mp_silu equals that
  times `f32(1 / 0.596)`, the reciprocal of the Python double rounded once (one ulp above `1.0f / f32(0.596)`): 0 of
  892928 values unequal offline. The scalar reciprocal is now built that way for every divisor (0.417, sqrt(0.58) and 3
  give the same bits either way). The fixed constant has not run on SM121.

Recorded, not reproduced (gates stay open):
- the decoder's K 3 convs (Cin 512 / 1024 / 2048) and the vocoder's dilation-1 convs with K >= 3 run cuDNN 9.24's FFT
  engine (engine id 4: `fft1d_r2c` / complex `gemvx` / `fft1d_c2r`), which is not reproduced by any order tried offline;
  conv_in, the 1x1 convs, qkv and the dilated convs run implicit-GEMM engines, which `.fma` reproduces.
- the middle attention's SDPA (fp32, head dimension 2048) dispatches MATH. Its steps composed from public ops (q and
  k^T scaled by sqrt(1 / sqrt(D)), matmul, softmax, matmul) reproduce the native output; `probs @ v` (cuBLASLt algo 16)
  is one serial fma chain, reproduced offline. The scores GEMM at T 218 (cuBLASLt algo 20, SIMT 64x64 tile, 8x5
  stages, 12 K splits, compute-type reduction) is reproduced offline (0 of 47524 unequal) by 12 K slices of 172, each
  an fma chain from 0, summed in slice order; at T 436 cuBLASLt logged 6 splits with an in-place reduction, not
  verified. The softmax follows torch's warp softmax order; with a correctly rounded exp offline it is not exact, and
  CUDA's `expf` is not correctly rounded, so only SM121 can settle it. `k6a_sdpa_math` (split-K scores, warp softmax)
  stays a comparison candidate until it runs there. Strict FLASH is not accepted.

Not on the native path, refused: the vocoder weight-norm fold.

## Route A: withdrawn

The scoped vendor exception (cuDNN for the recorded FFT convolutions, cuBLASLt for the SDPA's GEMMs) is withdrawn: the
native kernels reproduce every audio operator (`QUALIFICATION.md`). Its operators (`conv1d_vendor_fft`,
`sdpa_gemm_vendor`), plans (`native_plans.zig`), admission (`route_a.zig`), cuDNN graph (`cudnn.zig`) and the central
lock's candidate rows were removed; no audio code opens a vendor library.

Open choices that the native fixtures must settle: the per-term rounding of each reduction (`Acc`) and how each scalar
division executes (`Div`). Neither has a default.

Build: `nvcc -fatbin -O3 --fmad=false` plus LocalRouter's torch flags, no `--use_fast_math`, `-gencode` for sm_121a /
sm_120a. These are the same flags as `kernels/cuda/minimax/vae_audio.cu`. The build wiring belongs to the integration
owner.
