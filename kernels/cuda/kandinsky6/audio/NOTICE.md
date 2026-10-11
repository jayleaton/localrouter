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

Restricted and not used: NVIDIA EDM2 `edm2_utils.py` (CC BY-NC-SA 4.0), imported by the native `MPConv1D`, `normalize`,
`mp_silu` and `mp_sum`. Its text was not read, copied, translated or called by this code. The CPU oracle fixtures in
`fixtures/` came from a black-box invocation in Captain's audio-closure research (synthetic inputs, no model weights).
They exercise ordinary convolution over oracle-prepared weights only.

If BigVGAN or alias-free-torch source is ever copied rather than reimplemented, the copy must carry the BigVGAN
`LICENSE` and `incl_licenses/LICENSE_1..8` closure. The release inventory decides which notices ship.

## Qualification state (gpu-qualification-pending)

No kernel here is qualified. The required gate is zero unequal values against the native pinned Python twin on SM121,
checked per operator, per chain, on the full mel and on raw PCM. CPU diagnostics do not count toward it.

Unresolved, rejected before allocation, no substitute:

- activation normalization (residual axis 1, packed QKV axis 2): every independent family mismatches the oracle
- decoder stored -> prepared weight transform: every family mismatches, and the oracle transform is not idempotent
- vocoder weight-norm fold: the PyTorch formula is known, but the checkpoint state and target bits are unverified
- middle attention SDPA (fp32, head dimension 2048): the native backend is not fixtured; strict FLASH is not accepted

Open choices that the native fixtures must settle: the per-term rounding of each reduction (`Acc`) and how each scalar
division executes (`Div`). Neither has a default.

Build: `nvcc -fatbin -O3 --fmad=false` plus LocalRouter's torch flags, no `--use_fast_math`, `-gencode` for sm_121a /
sm_120a. These are the same flags as `kernels/cuda/minimax/vae_audio.cu`. The build wiring belongs to the integration
owner.
