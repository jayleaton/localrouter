# VAE decoder port (Qwen-Image 2.1, one frame)

Source of truth: diffusers 0.41 `AutoencoderKLQwenImage21` (`autoencoder_kl_qwenimage21.py`), config
`vae/config.json`: base_dim 96, decoder_base_dim 144, z_dim 64, dim_mult [1,2,4,8,8], num_res_blocks 2,
is_residual true, patch_size null, in/out_channels 4, temperal_downsample [F,T,T,T] (so temperal_upsample = [T,T,T,F]),
attn_scales []. The twin is `tools/twin/stk_twin/vae_ops.py` (`patch(vae)`); the kernels are
`kernels/cuda/qwen_image/vae.cu` (this file's `stk_*` names) and `gemm.cu` (`stk_gemm_bf16`). Everything is bf16 in
global memory, fp32 inside a kernel, batch 1, layout CHW (`x[c][y][x]`, row-major).

## What the image path does differently from the video decoder

`vae.decode(z[1,64,1,h,w])` runs `_decode` with one frame (`first_chunk=True`) and a fresh `feat_cache` (all `None`):

* `QwenImage21CausalConv3d` is a `Conv2d` on the squeezed frame: `F.pad(x, (l, r, t, b))` zeros, then the conv. Its
  `cache_x` is always `None`.
* The "upsample3d" mode of `QwenImage21Resample` degrades to "upsample2d": with an empty cache it only stores the string
  "Rep"; there is no frame doubling and `time_conv` is never used. (Blocks 0..2 are "upsample3d", block 3 "upsample2d";
  all four are identical for one frame.)
* `QwenImage21DupUp3D(first_chunk=True)` drops the first `factor_t - 1` output frames: for factor_t = 2 only the second of
  the two time slots survives (index `fti = 1` below), for factor_t = 1 nothing is dropped (`fti = 0`).
* Tiling is off (`use_tiling = False`); no patchify (`patch_size` null); the encoder, `quant_conv` and all
  `time_conv` weights are unused. The final `torch.clamp(out, -1, 1)` stays a plain min/max.
* The attention block's docstring says causal; it is not: `F.scaled_dot_product_attention(q, k, v)` with no mask, one
  head, `E = C`, default scale `1/sqrt(C)`.
* `out_channels` is 4 in the config: the decoded image has 4 channels and `postprocess` makes an RGBA PIL image (the twin
  reads the channel count from the weights; check `conv_out.weight.shape[0]` on the real checkpoint).

## Primitives

Convolution `conv(x[C,H,W], w[Cout,C,kh,kw], b[Cout], stride, pads(t,b,l,r))` -> `y[Cout,Ho,Wo]`,
`Ho = (H + t + b - kh)/stride + 1` (the decoder only uses stride 1: 3x3 pad 1 -> same size, 1x1 pad 0).

1. `stk_im2col(x, cols, C,H,W, kh,kw, stride, pad_t,pad_b,pad_l,pad_r, up=1, Kpad, p0, np)`: for output pixel
   `p = oy*Wo + ox` in `[p0, p0+np)` and `k = c*kh*kw + ky*kw + kx`:
   `cols[p-p0][k] = x[c][oy*stride - pad_t + ky][ox*stride - pad_l + kx]` (0 outside the image; 0 for `k >= C*kh*kw`).
   `Kpad = ceil8(C*kh*kw)`. In this decoder `C*kh*kw` is already a multiple of 8 everywhere (64*9, 1152*9, 288, ...,
   144*9), so there is no padding in practice; the twin pads the weight with zero columns if it ever happens. `up = 2`
   reads `x[y/2][x/2]` (virtual nearest upsample; the twin does not use it, it runs `stk_upsample_nearest2x`).
   launch: block 256, grid `ceil(np*Kpad/256)`.
2. `stk_gemm_bf16(A=cols[np,Kpad], B=Wm[Coutp,Kpad], bias[Coutp], C=pm[np,Coutp], M=np, N=Coutp, K=Kpad, lda=Kpad,
   ldb=Kpad, ldc=Coutp)`. The pixels are on M and the output channels on N because the GEMM's bias is indexed by N
   (the output channel). `Wm` is the PyTorch weight `[Cout,Cin,kh,kw]` viewed as `[Cout, Cin*kh*kw]` (no copy). `Coutp = Cout`
   rounded up to even (ldc must be even; zero rows and zero bias are appended): only `conv_out` (Cout = 4) is that
   small and 4 is even, so no padding happens in practice.
3. `stk_transpose_bf16(pm, y, R=np, Cc=Cout, ldx=Coutp, ldy=Ho*Wo)` at `y + p0`: `y[co][p] = pm[p-p0][co]`.
   launch: block (32,8), grid `(ceil(R/32), ceil(Cc/32))`.

The twin runs steps 1-3 in bands of whole output rows so the columns stay under 512 MiB (`STK_VAE_BAND_BYTES`; the largest
conv, 288->144 at 1024x1024, has K = 2592 and would need 5.4 GB at once). A GEMM row's bits do not depend on M, N or the
grid, so any banding (or none, or an implicit-GEMM that gathers the taps on the fly) gives the same bits; the Zig engine is
free to pick its own. In the twin the whole conv is ONE recorded op (kind `conv2d`, name = module path) whose input is
`x` and whose output is `y` (CHW), weights named in `attrs.weight` / `attrs.bias`.

Other kernels (all `extern "C"`, launch shapes in `vae.cu`):

* `stk_channel_rms_norm(x[C,HW], gamma[C], y, C, HW, scale)`: per pixel `ss = fmaf chain of x_c^2, c ascending`;
  `den = max(sqrtf(ss), 1e-12f)`; `n = bf16(x_c / den)` (IEEE div); `t1 = bf16(n * scale)`; `t2 = bf16(t1 * gamma_c)`;
  `y = bf16(t2 + 0.0f)`. `scale = float(sqrt(C))` (python `dim**0.5`, cast to fp32). `gamma` is the parameter
  `*.gamma` `[C,1,1,1]` (or `[C,1,1]` in the attention block) flattened. Reproduces
  `F.normalize(x.float(), dim=1).to(bf16) * scale * gamma + 0.0` (three bf16 roundings; torch's own channel sum order
  differs from ours, so the twin is not bit-identical to the original there, it is identical to Zig).
* `stk_silu(x, y, n)` from `ops.cu`: `y = bf16(x / (1 + expf(-x)))`.
* `stk_add_bf16(x, z, y, n)`: `y = bf16(x + z)`.
* `stk_upsample_nearest2x(x[C,H,W], y[C,2H,2W])`: `y[c][Y][X] = x[c][Y>>1][X>>1]`. (torch nearest-exact with scale 2.0:
  source = floor((dst + 0.5) * 0.5) = dst >> 1; the module's `.float()` ... `.type_as(x)` round trip of a bf16 is exact.)
* `stk_dup_up(x[Cin,H,W], y[Cout,2H,2W], Cout, H, W, fs=2, factor, repeats, fti)`: with `factor = factor_t*fs*fs` and
  `repeats = Cout*factor/Cin`: `y[o][Y][X] = x[(o*factor + s)/repeats][Y/2][X/2]`,
  `s = fti*fs*fs + (Y%fs)*fs + (X%fs)` (integer division). Derivation: `repeat_interleave(repeats, dim=1)` makes channel
  `j` a copy of `j/repeats`; `view(B, Cout, ft, fs, fs, T, H, W)` splits `j = o*factor + s`, `s = (fti*fs + fh)*fs + fw`;
  `permute(0,1,5,2,6,3,7,4)` + view interleaves them as `(t,ft)`, `(H,fh)`, `(W,fw)`; the first_chunk slice keeps `ft = factor_t-1`.
* `stk_transpose_bf16` (above), also used by the attention block.
* `stk_softmax_rows(S[rows,cols], P[rows,ldp], cols, ldp, scale)`: `m = max_j S_j`; `e_j = expf((S_j - m) * scale)`;
  `sum` = per-thread strided partials (thread t: j = t, t+256, ... ascending, `__fadd_rn`), xor-shuffle tree over each
  warp (offsets 16,8,4,2,1), then the 8 warp sums added in warp order; `P_j = bf16(e_j / sum)` (IEEE div); columns
  `[cols, ldp)` are zero. launch: block 256, grid `rows`.
* `stk_chan_affine(x[C,HW], mean[C], std[C], y)`: `y = bf16(bf16(x*std_c) + mean_c)` (the pipeline's
  `latents * std + mean`; mean/std are `torch.tensor(config.latents_*).to(bf16)`).
* `stk_to_u8_hwc(x[C,H,W], u[H,W,C])`: `d = clamp(bf16(bf16(x*0.5) + 0.5), 0, 1)`; `u = uint8(rintf(d * 255.0f))`
  (diffusers `postprocess`: `(x*0.5+0.5).clamp(0,1)` in bf16, then numpy float32 `(x*255).round().astype(uint8)`).

## Blocks

`ResBlock(name, cin, cout)` (QwenImage21ResidualBlock), input `x`:

```
h  = conv_shortcut(x)                 1x1 conv cin->cout, only if cin != cout (else h = x)       name.conv_shortcut
t  = rms_norm(x, norm1.gamma)                                                                    name.norm1
t  = silu(t)                                                                                     name.act1
t  = conv3x3(t, conv1)                cin -> cout, pad 1                                          name.conv1
t  = rms_norm(t, norm2.gamma)                                                                    name.norm2
t  = silu(t)                                                                                     name.act2
t  = conv3x3(t, conv2)                cout -> cout, pad 1 (dropout p = 0: identity)               name.conv2
y  = t + h                                                                                       name.add
```

`Attn(name, C)` (mid block, one head over L = H*W tokens), input `x [C,H,W]`:

```
xn  = rms_norm(x, attn.norm.gamma)          [C, L]                                               name.norm
xt  = transpose(xn)                         [L, C]                                               name.xt
q   = gemm(xt, Wq, bq)                      [L, C]   Wq = to_qkv.weight.view(3C, C)[0:C],   bq = to_qkv.bias[0:C]       name.q
k   = gemm(xt, Wk, bk)                      [L, C]   rows C:2C                                    name.k
v   = gemm(xt, Wv, bv)                      [L, C]   rows 2C:3C                                   name.v
vt  = transpose(v) into [C, Lpad]           Lpad = ceil8(L); columns L..Lpad zero                 name.vt
S   = gemm(q, k)                            [L, L]   S = q k^T, bf16 (no scale yet), no bias      name.scores
P   = softmax_rows(S, scale = 1/sqrt(C))    [L, Lpad], zero columns past L                        name.softmax
O   = gemm(P, vt)                           [L, C]   O = P v (A = P with K = Lpad, B = vt)        name.pv
po  = gemm(O, proj.weight.view(C, C), proj.bias)   [L, C]                                         name.proj
pc  = transpose(po)                         [C, L]                                               name.proj_t
y   = pc + x                                                                                     name.add
```

(q/k/v are three GEMMs on slices of one weight, contiguous row ranges; the Zig side may use `lda = 3C` on a single fused
qkv GEMM instead only if it keeps the same per-element K order, which it does since a GEMM element depends only on its
two rows. Torch's chunk order is q, k, v by channel blocks of C. SDPA itself is not bit-reproducible by us: flash
attention keeps P unnormalised in bf16; ours rounds S to bf16, then P to bf16. The PSNR test measures the effect.)

`Dup(name, cin, cout, factor_t)`: `stk_dup_up` as above with `fs = 2`.

`UpBlock(name, cin, cout, up)` (QwenImage21ResidualUpBlock), input `x_in`:

```
x = x_in
x = ResBlock(cin, cout)(x); x = ResBlock(cout, cout)(x); x = ResBlock(cout, cout)(x)      name.resnets.{0,1,2}
if up:
  x = upsample_nearest2x(x)                                                              name.upsampler.resample.0
  x = conv3x3(x, upsampler.resample.1)   cout -> cout, pad 1                              name.upsampler.resample.1
  x = x + Dup(x_in, cin, cout, factor_t = 2 for blocks 0..2, 1 for block 3)               name.avg_shortcut, name.shortcut_add
```

## Decoder, `latents[1,64,h,w]` (normalised) to image

h = w = 64 for 1024x1024 (L = 4096 tokens in the attention). Channel widths: dims = 144 * [8, 8, 8, 4, 2, 1] =
[1152, 1152, 1152, 576, 288, 144].

```
z      = chan_affine(latents, mean, std)                  [64, h, w]                      vae.denorm
z      = conv1x1(z, post_quant_conv)  64 -> 64                                            vae.post_quant_conv
x      = conv3x3(z, decoder.conv_in)  64 -> 1152, pad 1                                   vae.decoder.conv_in
x      = ResBlock(1152, 1152)         vae.decoder.mid_block.resnets.0
x      = Attn(1152)                   vae.decoder.mid_block.attentions.0
x      = ResBlock(1152, 1152)         vae.decoder.mid_block.resnets.1
x      = UpBlock 0: 1152 -> 1152, at  h x w   -> upsample to 2h  (Dup factor_t 2)         vae.decoder.up_blocks.0
x      = UpBlock 1: 1152 -> 1152, at 2h x 2w  -> upsample to 4h  (Dup factor_t 2)
x      = UpBlock 2: 1152 ->  576, at 4h x 4w  -> upsample to 8h  (Dup factor_t 2; Dup repeats 4)
x      = UpBlock 3:  576 ->  288, at 8h x 8w  -> upsample to 16h (Dup factor_t 1; Dup repeats 2; fti 0)
x      = UpBlock 4:  288 ->  144, at 16h x 16w, no upsampler, no Dup
x      = rms_norm(x, decoder.norm_out.gamma)                                              vae.decoder.norm_out
x      = silu(x)                                                                          vae.decoder.act_out
img    = conv3x3(x, decoder.conv_out) 144 -> Cout (4), pad 1                              vae.decoder.conv_out
img    = clamp(img, -1, 1)            (torch, exact)
u8     = to_u8_hwc(img)               [16h, 16w, Cout]                                    vae.to_u8
```

Dup repeats per block: block 0: Cin = Cout = 1152, factor 8, repeats 8; block 1: same; block 2: 1152 -> 576, repeats 4;
block 3: 576 -> 288, factor 4, repeats 2.

Op counts (1024x1024, h = w = 64): 17 ResBlocks (2 mid + 15 in up blocks), 3 of them with a 1x1 `conv_shortcut`
(first resnet of blocks 2, 3, 4).

| kind | count | notes |
|---|---|---|
| conv2d (im2col + gemm + transpose) | 44 | 34 resblock 3x3, 3 shortcut 1x1, 4 upsampler 3x3, conv_in, conv_out, post_quant_conv |
| channel_rms_norm | 36 | 34 resblock, 1 attention, norm_out |
| silu | 35 | 34 resblock, act_out |
| add | 22 | 17 resblock, 4 shortcut_add, 1 attention |
| upsample_nearest2x | 4 | |
| dup_up | 4 | |
| attention (mid) | 1 | 6 GEMMs (q, k, v, scores, pv, proj), 3 transposes, 1 softmax |
| chan_affine, to_u8 | 1 + 1 | ends of the pipeline |

Shapes at 1024x1024 (CHW): conv_in out [1152,64,64]; attention S [4096,4096] bf16 (33.5 MB); block 0 out [1152,128,128];
block 1 out [1152,256,256]; block 2 resnets at [576,256,256] then upsampled to [576,512,512]; block 3 resnets [288,512,512]
then [288,1024,1024]; block 4 [144,1024,1024]; image [Cout,1024,1024]. The widest im2col is block 4's `resnets.0.conv1`
(288 -> 144 at 1024^2: K = 2592) and the upsampler of block 3 (288 -> 288 at 1024^2: K = 2592).

## Weight names (safetensors, `vae/diffusion_pytorch_model.safetensors`, bf16)

```
post_quant_conv.{weight [64,64,1,1], bias}
decoder.conv_in.{weight [1152,64,3,3], bias}
decoder.mid_block.resnets.{0,1}.{norm1.gamma [C,1,1,1], conv1.{weight,bias}, norm2.gamma, conv2.{weight,bias}}
decoder.mid_block.attentions.0.{norm.gamma [C,1,1], to_qkv.{weight [3C,C,1,1], bias [3C]}, proj.{weight [C,C,1,1], bias}}
decoder.up_blocks.{i}.resnets.{j}.{norm1.gamma, conv1.*, norm2.gamma, conv2.*, conv_shortcut.* (if cin != cout)}
decoder.up_blocks.{0..3}.upsampler.resample.1.{weight [C,C,3,3], bias}          (resample.0 is the parameter-free upsample)
decoder.up_blocks.{0..2}.upsampler.time_conv.*                                    (unused for one frame)
decoder.norm_out.gamma [144,1,1,1]
decoder.conv_out.{weight [Cout,144,3,3], bias}
```

The RMS norm has no bias parameter (`bias = 0.0`). `config.latents_mean` / `latents_std` (64 each) feed `chan_affine`.

## Recorder names

Every op above is a `REC.op` named `vae.<module path>` (the module names in the tables) with these kinds: `conv2d`,
`channel_rms_norm`, `silu`, `add`, `upsample_nearest2x`, `dup_up`, `gemm`, `transpose`, `softmax_rows`, `chan_affine`,
`to_u8_hwc`. Inputs are the activations (`x`, or `a`/`b` for the two activation-by-activation GEMMs), output `y`;
weights are referenced by name in `attrs`. Every `attrs` carries `impl`.
