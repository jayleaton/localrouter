# MiniMax H3 audio VAE decoder: the twin's port

Source: ComfyUI 0.37.0 `comfy/ldm/minimax/audio_vae.py` (`MiniMaxH3AudioVAE.decode`, `BigVGAN`, `AMPBlock1`,
`Activation1d`, `UpSample1d`, `DownSample1d`, `SnakeBeta`), `comfy/sd.py` `VAE.decode`, `comfy_extras/nodes_audio.py`
`vae_decode_audio`. Ours: `kernels/cuda/minimax/vae_audio.cu` (+ `h3_gemm_f32` of `ops.cu`),
`tools/twin/stk_twin/h3/vae_audio.py`, test `stk_twin/h3/test_vae_audio.py`. Context: `VIDEO-PORT.md` section 7.

Input: the sampler's final audio latent, fp32 `[1, 32, 2, A]` AFTER `process_latent_out` (the `* 0.25` is done by the
caller: the sampler's `h3_scale32`). Output: fp32 `[1, 2, A * 800]` at 32 kHz. The encoder, `pre_block`, `mean_proj`,
`logs_proj` are not used and not loaded.

## 1. Checkpoint facts (`minimax_h3_audio_vae_fp32.safetensors`, header only)

* All 918 tensors F32. Weight norm is ALREADY folded: plain `*.weight` (no `weight_g` / `weight_v`).
* Decoder tensors used (the rest are `encoder.*`, `pre_block.*`, `mean_proj.*`, `logs_proj.*`):
  `latents_mean [32]`, `latents_std [32]`; `dec_in_proj.{weight [2048, 32, 1], bias [2048]}`;
  `decoder.conv_pre.{weight [1024, 2048, 7], bias [1024]}`;
  `decoder.ups.{i}.0.{weight [Cin, Cout, k], bias [Cout]}` (ConvTranspose layout `[Cin, Cout, K]`):
  i = 0: [1024, 512, 9], 1: [512, 256, 9], 2: [256, 128, 4], 3: [128, 64, 4], 4: [64, 32, 4], 5: [32, 16, 4], 6: [16, 8, 4];
  `decoder.resblocks.{3 i + j}.convs1.{n}.{weight [ch, ch, K_j], bias [ch]}`, `.convs2.{n}.*` (same shapes), K_j = 3, 7, 11
  for j = 0, 1, 2, ch = 512 >> i; `decoder.resblocks.{b}.activations.{0..5}.{act.alpha [ch], act.beta [ch],
  upsample.filter [1, 1, 12], downsample.lowpass.filter [1, 1, 12]}`;
  `decoder.activation_post.{act.alpha [8], act.beta [8], upsample.filter, downsample.lowpass.filter}`;
  `decoder.conv_post.weight [1, 8, 7]` (no bias).
* The Kaiser-sinc filters ARE stored (persistent buffers, loaded with `strict=True`), per activation, one
  `upsample.filter` and one `downsample.lowpass.filter`; the same 12 values for every channel of the layer. We load
  the stored ones (never recompute); the test prints the distance to `kaiser_sinc_filter1d(0.25, 0.3, 12)`.
* `alpha` / `beta` are stored as LOGS; the kernel applies `exp`.
* Metadata `minimax_h3_audio_vae`: `latents_mean` / `latents_std` (same values as the tensors), sample_rate 32000,
  `decoder_rates [5, 5, 2, 2, 2, 2, 2]`, `decoder_dim 1024`, `latent_dim 2048`.

## 2. Layout and the op list

Activations are fp32, channel-last `[B, T, C]` with `B = Bb * S = 2` (the stereo channels are independent batch items,
channel 0 first). Every convolution is `im2col + h3_gemm_f32` (`out = fmaf chain over k ascending from 0, then + bias`,
tiles of 64 x 64; an output's bits depend only on its own row of `col` and its own row of the weight). Weights are
repacked on the host by pure permutations (no arithmetic).

| # | name (REC) | kind | arithmetic |
|---|---|---|---|
| 1 | `avae.latent_in` | latent_in | `rows[b, t, c] = fadd(fmul(z[0, c, b, t], std[c]), mean[c])` (two roundings, no fma); the `permute(0, 2, 1, 3).reshape(b * s, c, t)` is the index map |
| 2 | `avae.dec_in_proj` | conv1d k 1 | `Conv1d(32 -> 2048)`: GEMM `[B A, 32] x [2048, 32]^T + bias`; order: ci ascending |
| 3 | `avae.conv_pre` | conv1d k 7 | `Conv1d(2048 -> 1024, pad 3)`; order: tap k ascending (outer), ci ascending (inner), then + bias |
| 4 | `avae.ups.{i}`, i = 0..6 | conv_transpose1d | `ConvTranspose1d(k, stride u, padding (k - u) / 2)`, u = 5, 5, 2, 2, 2, 2, 2; section 3 |
| 5 | `avae.res.{i}.{j}.{n}.a1` | activation1d | `activations[2 n]` of block `resblocks[3 i + j]`, j = 0..2, n = 0..2: up x2, SnakeBeta, down x2 (section 4) |
| 6 | `.c1` | conv1d | `convs1[n]`, K_j in {3, 7, 11}, dilation {1, 3, 5}[n], pad `(K d - d) / 2`, zero padding, same length |
| 7 | `.a2` | activation1d | `activations[2 n + 1]` |
| 8 | `.c2` | conv1d | `convs2[n]`, dilation 1, pad `(K - 1) / 2` |
| 9 | `.add` | add | `x = fadd(c2_out, x)` (`xt.add_(x)`; the block's input x is never modified in place) |
| 10 | `avae.avg.{i}` | avg3 | `s = fadd(fadd(r0, r1), r2)` (`xs = rb0(x); xs += rb1(x); xs += rb2(x)`), then `x = fmul(s, 1.0f / 3.0f)` (torch `xs.div_(3)` on CUDA multiplies by the fp32 reciprocal; flag `recip`, `AVG3_RECIP = True`; `recip = 0` is `fdiv(s, 3.0f)`; the test pins which one the pod's torch does) |
| 11 | `avae.post` | activation1d | `activation_post` (SnakeBeta(8)) |
| 12 | `avae.conv_post` | conv1d k 7 | `Conv1d(8 -> 1, pad 3, no bias)`, output `[2, L, 1]` = waveform `[2, L]` |
| 13 | `avae.clamp` | clamp | `v < -1 ? -1 : (v > 1 ? 1 : v)` (NaN passes, as `clamp_`) |
| 14 | `avae.std_scale` | std_scale | section 5 |
| 15 | `avae.div_scale` | div_scale | `y = fdiv(audio, sc)` |

Op count per decode: 3 + 7 x (1 ups + 3 blocks x 3 layers x 5 + 1 avg) + 3 + 2 = 337 recorded ops (the Activation1d and
convolution ops are composites of 1 to 3 kernel launches plus `h3_gemm_f32`, the intermediates unrecorded). `ins` of a
conv op: `x`; weights are named by `attrs.weight` (checkpoint key); activation attrs name the layer prefix.

Naming scheme: `avae.<stage>.<...>`; `avae.res.{i}.{j}.{n}.<a1|c1|a2|c2|add>` with i = upsample stage (0..6), j = AMP
block within the stage (0..2, kernel 3 / 7 / 11), n = layer within the block (0..2, dilation 1 / 3 / 5). The
checkpoint's `resblocks.{3 i + j}` and `activations.{2 n}` (a1) / `{2 n + 1}` (a2).

## 3. ConvTranspose1d without a scatter

`out[t_o] = bias + sum x[t_i, ci] w[ci, co, k]` over `t_o + pad = t_i u + k`. For output phase `r = (t_o + pad) mod u`
and `q = (t_o + pad) div u`: taps `k = r + j u` (`j < J = ceil((K - r) / u)`) read `t_i = q - j`. The phase's outputs are
`m = 0..L-1`, `q = m + qoff`, `qoff = 1 if r < pad else 0`, `t_o = u (m + qoff) + r - pad` (every `t_o` in `[0, u L)` is
produced by exactly one phase; requires `pad < u`, true for (9, 5) and (4, 2)). Each phase is one `im2col` (`col[b L + m,
j C + ci] = x[b, m + qoff - j, ci]`, zero outside `[0, L)`) + `h3_gemm_f32` with the bias + a store into the interleaved
output. Summation order of an output: `j` ascending (= `k` ascending), `ci` ascending, one fmaf chain from 0, then +
bias. The output length is `u L` for these paddings (checked against torch's `(L - 1) u - 2 pad + K`).

## 4. Activation1d

* `UpSample1d` (ratio 2, 12 taps, filter `f`): replicate pad 5 / 5, `conv_transpose1d(stride 2, groups C)`, `* 2`, crop
  `[15 : -15]`. Output `o` (padded position `p = o + 15`): `acc = fma-chain over j = 0..5 of f[(p & 1) + 2 j] *
  x[clamp((p >> 1) - j - 5, 0, T - 1)]` (k ascending, from 0), `y = fmul(acc, 2.0f)` (exact). `[B, T, C] -> [B, 2T, C]`.
* `SnakeBeta` per element, per channel `c`: `a = expf(alpha[c])`, `ib = fdiv(1, fadd(expf(beta[c]), (float)1e-9))`,
  `t = sinf(fmul(a, x)); t = fmul(t, t); t = fmul(t, ib); y = fadd(t, x)` (torch: `sin(alpha*x)`, `t.mul_(t)`,
  `.mul_((beta + 1e-9).reciprocal())`, `.add_(x)`). libm used: `expf`, `sinf` (CUDA device libm, no fast-math).
* `DownSample1d` (stride 2, 12 taps): replicate pad 5 / 6, `conv1d(stride 2, groups C)`: `y[o] = fma-chain over k =
  0..11 of f[k] * x[clamp(2 o + k - 5, 0, T2 - 1)]`, k ascending from 0. `[B, 2T, C] -> [B, T, C]`.
* Filters: the stored `upsample.filter` and `downsample.lowpass.filter` of the layer (all 12 taps, any asymmetry kept).

## 5. Post-decode normalisation (`vae_decode_audio`)

`std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0; std[std < 1] = 1; audio /= std`, on `[1, 2, L]`: one std over
both stereo channels (`n = 2 L`), unbiased. Ours (`avae_std_scale`, one block of 256 threads per batch row): fp64,
thread `t` sums elements `t, t + 256, ...` ascending, the 256 partials are tree-combined (`s[t] += s[t + st]`, st = 128
down to 1), `mean = sum / n`, second pass the same order over `(x - mean)^2`, `var = sum2 / (n - 1)`,
`std = float(sqrt(var))`, `s = fmul(std, 5.0f)`, `sc = s < 1 ? 1 : s`. torch's CPU path (ComfyUI's intermediate
device) is an fp64 Welford; the results agree to fp64 noise before the one rounding to fp32, so `sc` is normally
bit-equal; torch's CUDA path (if the pod keeps the output on the GPU) is an fp32 Welford and can differ by a few
ulps: the test prints which of the two ours equals. The division is a true IEEE fp32 division (CPU path; also a CUDA
tensor over a one-element CUDA tensor). torch on CUDA divides by a CPU scalar tensor by multiplying with the
reciprocal: not the path ComfyUI takes here (the `std` is a `[1, 1, 1]` tensor on the same device as `audio`).

## 6. Shapes (per stereo item, channel-last `[T, C]`; batch 2)

| stage | A = 93 (L = 74 400) | A = 207 (L = 165 600) |
|---|---|---|
| latent / dec_in_proj / conv_pre | `[93, 32]` / `[93, 2048]` / `[93, 1024]` | `[207, 32]` / `[207, 2048]` / `[207, 1024]` |
| ups.0 + blocks 0 (k 3/7/11), C 512 | `[465, 512]` | `[1035, 512]` |
| ups.1, C 256 | `[2325, 256]` | `[5175, 256]` |
| ups.2, C 128 | `[4650, 128]` | `[10350, 128]` |
| ups.3, C 64 | `[9300, 64]` | `[20700, 64]` |
| ups.4, C 32 | `[18600, 32]` | `[41400, 32]` |
| ups.5, C 16 | `[37200, 16]` | `[82800, 16]` |
| ups.6, C 8 | `[74400, 8]` | `[165600, 8]` |
| activation1d inside a stage | `[2 T, C]` between up and down | same |
| conv_post / clamp / out | `[74400, 1]` -> `[1, 2, 74400]` | `[165600, 1]` -> `[1, 2, 165600]` |

Largest im2col: `[2 T, K C]` = 2 x 14.6 M floats (58 MB x 2) at K = 11 in every stage from 1 on; conv_pre `[2 A, 14336]`.
No banding is needed; bits do not depend on any chunking (row independence of the GEMM).

## 7. Differences from ComfyUI and why

* **No TF32**: ComfyUI's cuDNN convolutions run with `torch.backends.cudnn.allow_tf32` at its default (True), so on
  Ampere+ its fp32 convs may be TF32 (10-bit mantissa products). Ours are strict fp32 fma chains. The test compares
  against ComfyUI with TF32 disabled (like-for-like) and reports how far TF32 moves ComfyUI's own output.
* **Summation orders**: cuDNN's are unspecified and algorithm-dependent (and `cudnn.benchmark` can change them); ours are
  fixed (sections 2 to 5) so the Zig engine is bit-exact with the twin. The expected disagreement with ComfyUI is
  fp32 rounding noise (no TF32), not a model difference; the test measures SNR / max |diff| and the error of both against
  an fp64 run of ComfyUI's module.
* **Bias add after the sum** (one `fadd`), cuDNN may add it in its epilogue or per tile.
* **Depthwise filters as direct 6- and 12-tap fma chains** (cuDNN's grouped conv / conv_transpose use their own order).
* **std in fp64 with a fixed order**, true division (section 5).
* **Avg of the three blocks** fused into one kernel with torch's rounding sequence (two fp32 adds, then the reciprocal
  multiply).
* `torch.exp` / `torch.sin` are CUDA `expf` / `sinf`: same functions as ours IF the CUDA toolkit's libdevice matches
  (bits of `expf`, `sinf` are toolkit-version-dependent in the last ulp rarely); the extension is built with
  `--fmad=false`, no fast-math, no ftz.

## 8. Risks for bit-exactness (twin vs Zig engine)

1. The Zig build must use the same nvcc / libdevice and flags (`--fmad=false`, IEEE div/sqrt, no ftz, no fast-math) for
   `expf` / `sinf` in `avae_snake`; everything else in the file is `__f*_rn` / `__fmaf_rn` and exact whatever the flags.
   Safest: run the kernels from the same fatbin in both.
2. `h3_gemm_f32` (fixed tile / k order) is shared with the DiT's fp32 GEMMs; any change to it changes these bits.
3. `AVG3_RECIP`: the choice is validated by the test on the pod's torch; it only decides ComfyUI-likeness, not twin/engine
   equality (both use the same flag).
4. A large `|alpha x|` in SnakeBeta amplifies a last-ulp difference of `sinf` or of the preceding conv into larger
   output differences: end-to-end SNR against ComfyUI is therefore not a tight bound; the fp64 comparison is the
   meaningful accuracy statement.
5. Untested here (no GPU): the CUDA file and the extension have not been compiled; index maps were checked against
   definitional references on the CPU (conv, conv_transpose phases, up / down filters).
