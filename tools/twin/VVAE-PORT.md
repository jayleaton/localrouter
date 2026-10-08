# Video VAE decoder port (MiniMax H3, ComfyUI 0.37.0 `comfy/ldm/minimax/vae.py`, fp16 checkpoint)

The decode path of `MiniMaxH3VideoVAE` on kernels LocalRouter owns: `kernels/cuda/minimax/gemm_f16.cu` (the GEMM) and `kernels/cuda/minimax/vae_video.cu` (everything else),
bound for the twin in `tools/twin/stk_twin/h3/vae_video.py`, checked on a GPU by `stk_twin/h3/test_vae_video.py`. Background and file:line references for what ComfyUI does:
`VIDEO-PORT.md` section 6. Only the decoder is ported (the encoder, `quant_conv`, `mask_token` are never run). Batch 1.

Nothing here has been compiled or run (no GPU / nvcc on the authoring machine): every claim below about bits is by construction and by reading the sources, and the checks in
section 8 are what turns it into fact on a pod.

## 1. The checkpoint (`minimax_h3_video_vae_fp16.safetensors`)

562 tensors, all F16, no `comfy_quant` keys (the int8 ConvRot file is a different path and is refused by `from_checkpoint`). 444 are used: 2.42 B parameters, 4.85 GB.

| tensor | shape |
|---|---|
| `latents_mean`, `latents_std` | `[24]` (persistent buffers: the file's fp16 values are the ones used) |
| `post_quant_conv.weight` / `.bias` | `[24, 24, 1, 1, 1]` / `[24]` |
| `decoder.x_embedder.weight` / `.bias` | `[2048, 24]` / `[2048]` |
| `decoder.register_tokens` | `[1, 4, 2048]` |
| `decoder.transformer_blocks.{0..35}.norm1.weight`, `.norm2.weight`, `.scale1`, `.scale2` | `[2048]` |
| `...attn.to_qkv.weight` / `.bias` | `[6144, 2048]` / `[6144]` |
| `...attn.to_out.weight` / `.bias` | `[2048, 2048]` / `[2048]` |
| `...ff.w1.weight` / `.bias` | `[16384, 2048]` / `[16384]` (gate = first 8192 outputs, up = last 8192) |
| `...ff.w2.weight` / `.bias` | `[2048, 8192]` / `[2048]` |
| `decoder.norm_out.weight` / `.bias` | `[2048]` (LayerNorm) |
| `decoder.proj_out.weight` / `.bias` | `[3072, 2048]` / `[3072]` (3072 = 3 * 4 * 16 * 16) |
| `decoder.mask_token` | `[1, 1, 2048]`, unused |
| `encoder.*`, `quant_conv.*` | 118 tensors, 0.18 B parameters, unused |

36 layers, 32 heads x 64, dim 2048, FFN 8192 (gated), RoPE theta 100 on 48 of 64 dims, eps 1e-5, 4 register tokens + 1 zero suffix token. Constants not in the file: the ImageNet
mean / std (`vae.py:17-18`, **rounded to fp16 by the module's `.to(fp16)` and used as fp32**), `rope inv_freq` (8 values, fp16-rounded), `qk_norm_scale` (ones).

## 2. What `decode` does (the plan)

Input: latents `[1, 24, T, H, W]` (fp32 from `VAEDecode`; ComfyUI does `.to(device, dtype=fp16)`; so do we). Output: uint8 `[F, 16 H, 16 W, 3]`, `F = 17 (T-2)/5 + 5` (T = 5k + 2).

1. `z = z * std + mean`, fp16, two torch ops (two roundings), per channel (`vv_denorm`).
2. `decode_temporal`: `pad_tokens = (-(T + 3)) % 5` latent frames repeat the last frame (we clamp the frame index in the gather instead of building the padded tensor);
   `num_chunks = (T + 3 + pad) / 5 - 1` (at least 1; one more chunk of padding if 0). Clip `i` is latent frames `[5i, 5i + 7)` (7 frames, so 28 pixel frames).
3. Each clip is tiled spatially (`tiled_decode`, section 3), the tiles decoded one by one (section 4), blended into a fp16 clip canvas `[3, 28, 16 H, 16 W]`.
4. A clip's canvas is cut into two halves: `j = 0`: canvas frames `[3, 20)` (17 frames, the first `frame_pre_padding = 3` dropped), `j = 1`: `[23, 28)` (5 frames). The previous clip's carried 5 frames are
   blended into the first 5 frames of the `j = 0` half (`blend(dec_overlap, chunk, 5, dim = frames)`), the part is written; the `j = 1` half is carried (the last clip writes it as is).
   `write_part` = finalise + convert (section 6), at `write_pos`, truncated to the frame count `decode_output_shape` allots (the padding at the end is cut).
5. `T = 1`: one clip of one latent frame, decoded to 4 frames, only the last is finalised.

Plan for T = 37 (the production video, 124 frames): 7 clips `[0,7) [5,12) [10,17) [15,22) [20,27) [25,32) [30,37)`, no padding; 17 frames written per clip, +5 at the end = 124.
T = 7 (the check size, 22 frames): 1 clip, 17 + 5.

## 3. Spatial tiling (`split_tiles`, tile 256 px, overlap >= 64 px, grid 16 px; pixels)

`N = ceil(len / 256)` grown until `256 N - 64 (N - 1) - len >= 0`; `(256 N - 64 (N-1) - len) // 16` units of 16 px are added round-robin to the `N - 1` overlaps; starts are cumulative `256 - overlap`.
All tiles are 256 px in a dimension when there are several; a dimension of <= 256 px is one tile of its own length.

| length (px) | tiles | starts | overlaps | cropped lengths placed in the canvas |
|---|---|---|---|---|
| 448 | 2 | 0, 192 | 64 | 192, 256 |
| 768 | 4 | 0, 160, 336, 512 | 96, 80, 80 | 160, 176, 176, 256 |
| 1344 | 7 | 0, 176, 352, 528, 704, 896, 1088 | 80, 80, 80, 80, 64, 64 | 176, 176, 176, 176, 192, 192, 256 |

Production sizes (width x height): **768 x 448** -> columns `768` (4 tiles), rows `448` (2 tiles): 8 tiles a clip. **1344 x 768** -> columns `1344` (7), rows `768` (4): 28 tiles a clip.
Each tile is `[24, 7, 16, 16]` latents (clip of 7 frames x 256 px / 16). The 768 x 448 video at T = 37 is 8 x 7 = 56 tile decodes, 1344 x 768 is 28 x 7 = 196.

Order and arithmetic per tile `(i, j)` (row `i`, column `j`), exactly `tiled_decode`, all in fp16 (`vv_place_tile`):

* the raw decoded tile `b`; the **y blend** (`i > 0`): rows `r < ey = y_overlap[i-1]` of `b` are blended with the last `ey` rows of the **raw** tile above (`a`), weight position `r`;
* then the **x blend** (`j > 0`): columns `c < ex = x_overlap[j-1]` of that result are blended with the last `ex` columns of the **raw** (un-blended in y and x) tile to the left, all `256` rows, position `c`
  (this is what ComfyUI does: the left tail is cloned before any blend of the left tile, the tile is y-blended before the x blend; the seam is not "correct", it is replicated);
* crop the trailing `y_overlap[i]` rows (if not the last row) and `x_overlap[j]` columns (if not the last column); place at `(out_y, out_x)`; `out_x` advances by the cropped width, `out_y` by the row's cropped height.

`blend(a, b, ext, position p)` (fp16): `w_b = half(float(p) * (1 / ext))` (`positions / blend_extent`: torch divides a tensor by a Python scalar as a multiplication by the fp32 reciprocal `1.0f / ext`),
`w_a = half(1 - w_b)`, `out = half(half(a * w_a) + half(b * w_b))` (three torch ops, three roundings; each product of two halves is exact in fp32). `ext = min(a.len, b.len, extent)`.
Temporal blend: `ext = min(carried frames, part frames, 5)`, position = frame index in the part.

## 4. One tile: the ViT3D decoder, op by op (fp16 in memory; fp32 inside a kernel; one rounding to half per torch op)

`S = Tn * h * w + 5` tokens (1797 for the 7 x 16 x 16 tile). `rows` are `[tokens, C]` row-major, token = `(t * h + y) * w + x`.

| # | op (kernel) | arithmetic |
|---|---|---|
| 1 | `gather` (`vv_gather_rows`) | tile crop of the denormalised latents to rows `[nP, 24]`, frame index clamped to the last latent frame (data move) |
| 2 | `pqc` (`stk_gemm_f16`) | `post_quant_conv` (1x1x1 conv) as GEMM: `half(sum_k x_k w_k + b)`; `[nP, 24] x [24, 24]^T` |
| 3 | `embed` (GEMM) | `x_embedder`: `[nP, 24] x [2048, 24]^T + b` into rows `0 .. nP-1` of `h [S, 2048]` (ComfyUI's `flatten(2).transpose(1, 2)` is this layout) |
| 4 | `suffix` (`vv_suffix`) | rows `nP .. nP+3` = the 4 register tokens, row `nP+4` = 0 |
| 5 | `rope` (`vv_rope_table`) | the rotation table `[S, 24, 4]` fp16, section 4.1 |
| per layer `L0 .. L35` | | |
| a | `L{i}.norm1` (`vv_rms_norm`) | `n = half(x * rsqrt(sum(x^2) / 2048 + 1e-5) * w)`: fp32, sum = thread-strided partials (256 threads, ascending index) + shared-memory tree, `rsqrtf`, `(x * r) * w`, ONE rounding (torch's fused `F.rms_norm`) |
| b | `L{i}.qkv` (GEMM) | `n . Wqkv^T + b` -> `[S, 6144]` = `[S, 32 heads, 192]`: head h at `192 h`: q `+0..63`, k `+64..127`, v `+128..191` |
| c | `L{i}.rr` (`vv_rms_rope`) | the comfy-kitchen kernel for fp16, section 4.2, in place on q and k |
| d | `L{i}.vt` (`vv_vt`) | `V` transposed per head to `[32, 64, SP]`, `SP = ceil(S / 8) * 8` (1800), zero pad |
| e | `L{i}.sc` (GEMM, batch 32) | scores `[32, S, SP]` = `half(q . k^T)`, K = 64, no bias, no scale (pad columns untouched: stay 0) |
| f | `L{i}.sm` (`vv_softmax`) | in place, per row: `x = float(s) * 0.125` (exact), `m = max`, `e = expf(x - m)`, `sum` (per-thread ascending partials, xor-shuffle tree 16..1, 8 warp sums in order), `P = half(e / sum)` (IEEE divide), columns `S .. SP-1` written 0 |
| g | `L{i}.pv` (GEMM, batch 32) | `O_h = half(P_h . v_h)`, K = SP (zero P pad x zero V pad), written to columns `64 h ..` of the attention rows `[S, 2048]` |
| h | `L{i}.nan` (`vv_nan_to_num`) | `torch.nan_to_num`: NaN -> 0, +-inf -> +-65504 |
| i | `L{i}.out` (GEMM + epilogue) | `y = half(O . Wo^T + b)`; `x = half(float(x) + float(y) * float(scale1))` (`torch.addcmul(x, y, scale1)`, in place) |
| j | `L{i}.norm2` (`vv_rms_norm`) | as (a), with `norm2.weight` |
| k | `L{i}.w1` (GEMM) | `half(n . W1^T + b)` -> `[S, 16384]` |
| l | `L{i}.swi` (`vv_swiglu`) | `half(half(g / (1 + expf(-g))) * u)`, `g` = columns `0 .. 8191`, `u` = `8192 ..` (`F.silu(gate).mul_(up)`: the SiLU is rounded first) |
| m | `L{i}.w2` (GEMM + epilogue) | `y = half(a . W2^T + b)`; `x = half(float(x) + float(y) * float(scale2))`, K = 8192 |
| 6 | `norm_out` (`vv_layer_norm`) | rows `0 .. nP-1` only: `mean = sum / 2048`, `var = sum((x - mean)^2) / 2048` (two passes, each ascending partials + tree), `half(fma((x - mean) * rstd, w, b))`, `rstd = rsqrtf(var + 1e-5)` |
| 7 | `proj` (GEMM) | `[nP, 2048] x [3072, 2048]^T + b` |
| 8 | `unshuf` (`vv_unshuffle`) | `view(1, T, H, W, 3, 4, 16, 16).permute(0, 4, 1, 5, 2, 6, 3, 7)` -> `[3, 4 Tn, 16 h, 16 w]` (data move) |

### 4.1 The GEMM (`stk_gemm_f16`, `gemm_f16.cu`)

`C[M, N] = A[M, K] . B[N, K]^T`, block tile 128 x 128 x 32, 256 threads, two-stage `cp.async`, `ldmatrix`, `mma.sync.m16n8k16.f16.f16.f32` with an fp32 accumulator over K tiles ascending,
no split-K. Epilogue per element: `v = acc; if bias: v = v + float(bias)` (fp32 add); `o = half(v)`; with `res`: `o = half(float(res) + float(o) * float(rscale))`. An output's bits depend only on its two input rows.
Batched by `blockIdx.z` (`sA, sB, sC` element strides) for the 32 heads. Requirements: `K % 8 == 0`, `lda, ldb % 8 == 0`, 16-byte aligned A and B, `ldc` even.

Four kernels, one arithmetic (`gemm_f16.cu` header has the argument): `stk_gemm_f16_ref` is the version above; `stk_gemm_f16` (256 x 128 x 32 tile, three stages, 8 warps of 64 x 64, 92,160 bytes of dynamic shared memory, one block a
multiprocessor), `_s` (128 x 128, two stages) and `_n` (64 x 64, three stages, 4 warps) are one template that changes only the tile, the stages, the grid order (the m tile fastest) and the C store (a half2 where aligned); every output element
still sees `acc = 0`, then the same k16 chunks in ascending order into the same `mma.sync m16n8k16`, then the same epilogue. The launch picks by shape (`Ops.pick` / `gemm_pick`): K <= 64 `_s`, N <= 64 `_n`, else the wide one;
`STK_VVAE_REF=1` runs the reference everywhere. `test_vae_video.py` ("gemm_variants_*") proves every kernel equal to the reference with `torch.equal` on int16 views over every shape the decoder launches.
The attention of a tile that is not recorded runs `STK_VVAE_HEADS` (default 2) heads at a time (qk^T, softmax, P.V on the first heads' worth of the score buffer, window offsets `h0 * 192`, `h0 * 64 * SP`, `h0 * 64`): the same
per-head launches, so the same bits ("attention_groups_*"), with the scores (6.5 MB a head) in the L2 instead of 207 MB through memory four times. `STK_VVAE_PROF=1` prints the GPU milliseconds of a decode by class.
This replaces cuBLAS (ComfyUI's `F.linear`); the order of the fp32 accumulation inside a K tile is the tensor core's, so cuBLAS and we agree to fp32 rounding only (not bitwise).

### 4.2 The rotation table (`vv_rope_table`)

`create_token_ids` in fp16: per axis `i` of `n`: `q = half((i + 0.5) * (1.0f / n))`, `id = half(half(2 q) - 1)`; suffix tokens have id 0.
`RotaryEmbeddingND`: `angle = ((float)(2 pi) * id) * float(inv_freq[k])` (two fp32 roundings; `inv_freq = half(1 / 100 ** arange(0, 1, 0.125))`, computed on the host exactly as ComfyUI constructs it),
pair index `axis * 8 + k` (axis order t, y, x), table entry `(c, -s, s, c) = half(cosf), half(-sinf), half(sinf), half(cosf)`. No fast-math: `cosf` / `sinf` are the CUDA library's, as torch's `cos` / `sin` on CUDA.

### 4.3 RMSNorm + RoPE (`vv_rms_rope`), the kitchen kernel for fp16

comfy-kitchen v0.2.35 `rope_kernel<half, half, half, HasRms, SplitHalf, HasK, InPlace>` with unit scale, head dim 64, rot_dim 48 (one warp a (token, head), four warps a block):

* `sum = fmaf(v, v, sum)` over the lane's elements (lane, lane + 32), then `sum += shfl_down(sum, 16, 8, 4, 2, 1)`; lane 0's value is broadcast;
* `rrms = rsqrtf(sum / 64 + eps)`; `x' = half(float(x) * rrms)` (rounded to half BEFORE the rotation); pairs `(i, i + 24)`, `i < 24`;
* `y0 = f00 * x0' + f01 * x1'`, `y1 = f10 * x0' + f11 * x1'` in fp32, each rounded to half; dims 48 .. 63: `x'` only; v untouched.

**Differences from the wheel (all intended to be bit-neutral; check them with `test_vae_video` "rope_vs_kitchen_wheel"):**
1. `/ 64` is written as `* 0.015625f`: exact either way (the wheel is built with `--use_fast_math`, so its divide is `div.approx`, exact for a power of two); the following add is a separate `__fadd_rn` (the wheel may contract it to an fma, whose product is exact, same result).
2. `rsqrtf` is the same hardware `rsqrt.approx` (the wheel's `.ftz` form differs only for denormal inputs, impossible here since `eps = 1e-5`).
3. The unit-scale multiply `x * rrms * 1.0` is omitted (exact).
4. **The FMA contraction of `f00 * x0 + f01 * x1`**: nvcc (the wheel is built with `-fmad=true`) fuses one of the two products. We write `fma(f00, x0, f01 * x1)` (variant 0, nvcc's usual form for `a*b + c*d`). The kernel file also has `vv_rms_rope_v1` (the other product fused) and `_v2` (unfused) so the test can report which one the installed wheel equals; **if it is not variant 0, change `vv_rms_rope` (and nothing else)**.
5. One pair a lane (the wheel handles two adjacent pairs a lane in its vectorised variant); the arithmetic per element is the same.

### 4.4 Attention (decided, not ComfyUI's flash)

ComfyUI calls `optimized_attention` (SDPA: flash / cuDNN / efficient, fp16, softmax state in fp32, P rounded to fp16 inside the kernel). We compute `half(q k^T)` (the scores are rounded to fp16: a logit of 32..64 has an absolute
error up to 0.03, 0.004 after the 1/8 scale: the largest single deviation from ComfyUI's numerics), a fp32 softmax, `half(P v)`. The attention is therefore NOT bitwise ComfyUI's and cannot be made so without reimplementing the flash kernel.

## 5. Recorder names (`REC.op(name, kind, attrs, **ins)`)

`vvae.denorm` (kind `denorm`), then per clip `c` (0 .. num_chunks-1), per tile row `r`, column `k`, with `T = vvae.c{c}.t{r}_{k}`:

```
T.gather  T.pqc  T.embed  T.suffix  T.rope
T.L{0..35}.norm1  .qkv  .rr  .vt  .sc  .sm  .pv  .nan  .out  .norm2  .w1  .swi  .w2
T.norm_out  T.proj  T.unshuf  T.place
```
and the clip-level `vvae.c{c}.part0` / `vvae.c{c}.part1` (kind `finalize`, the temporal blend + finalise + uint8; `part1` only for the last clip's carried frames; `T = 1`: `vvae.c0.part0`).
Kinds: `gemm_f16` (attrs M, N, K, batch, epilogue `none | bias | bias+addcmul`), `rms_norm_f16`, `layer_norm_f16`, `rms_rope_f16`, `vt`, `softmax_rows`, `nan_to_num`, `swiglu_f16`, `gather_rows`, `suffix`, `rope_table`, `unshuffle`, `place_tile`, `finalize`, `denorm`.
Names are unique per decode. A full capture of a 1344 x 768 video is about 196 tiles x 36 layers x 14 ops (scores alone are 207 MB a blob), so record selectively: `decode(latents, rec=lambda chunk, row, col: ...)` toggles the recorder per tile
(`row = col = -1` for the clip-level ops and the denorm). The scratch score buffer is zero-initialised once so recorded outputs are deterministic.

Kernel launches (grids are in the `launch:` comment of each kernel): GEMM grid `(ceil(N/128), ceil(M/128), batch)` x 256 for the reference, `(ceil(M/BM), ceil(N/BN), batch)` for the others; the `.sc` call uses `lda = ldb = 6144`, `sA = sB = 192`, `b_off = 64`, `ldc = SP`, `sC = S * SP`;
the `.pv` call `A = P (lda = SP, sA = S * SP)`, `B = V^T (ldb = SP, sB = 64 * SP)`, `ldc = 2048`, `sC = 64`, `K = SP`.

## 6. Finalise and uint8 (`vv_finalize`)

`_finalize_pixels`: `p = float(half) * std_c` (fp32 multiply; `part * pixel_std.to(float32)` promotes), `p = p + mean_c` (fp32 add, separate torch op), `clamp(0, 1)` (NaN-propagating); std / mean are `half(0.229, 0.224, 0.225)` /
`half(0.485, 0.456, 0.406)` converted back to fp32. Then `save_to`: `u = p * 255` (fp32), `clamp(0, 255)`, `.byte()` = truncation toward zero. A NaN pixel (impossible after the nan_to_num unless the weights are bad) becomes 0 (undefined in torch).
The temporal blend and finalise are one kernel (the same per-element arithmetic, no intermediate store).

## 7. Differences from ComfyUI, and why

| | ComfyUI | here | why |
|---|---|---|---|
| linears | cuBLAS fp16 (`F.linear`), algorithm by shape | `stk_gemm_f16`, fp32 accumulation in a fixed order | determinism across M; the engine launches the same kernel |
| bias | cuBLAS epilogue | fp32 add before the fp16 rounding | decided |
| `post_quant_conv` | cuDNN conv3d 1x1x1 (+ the 2.9 / 2.10 workaround path) | GEMM `[nP, 24] x [24, 24]^T` | a 1x1x1 conv is a GEMM; decided |
| attention | SDPA (flash / cuDNN) | `q k^T` GEMM -> row softmax -> `P v` GEMM, fp16 scores | decided (head dim 64, one 1797-token tile) |
| RMSNorm | `F.rms_norm` (torch's kernel; its reduction order is torch's) | tree reduction in a fixed order, one rounding | fixed order; torch's exact order is not pinned |
| LayerNorm | `F.layer_norm` (Welford) | two-pass mean / variance, fixed order | same |
| rms+rope | comfy-kitchen `rms_rope_split_half_` | our kernel, same arithmetic (4.3) | file budget; bit-equal if the contraction guess is right |
| tiles per decoder call | `min(4, free_vram // 128 MiB)` | 1 | per-tile math is identical; only cuBLAS's choice changes |
| head | all `S` rows, 5 suffix rows dropped | the `nP` patch rows only | a row's bits depend on that row alone |
| padded last latent frames | `torch.cat` of repeated frames | clamped index in the gather | same values |
| dtype handling | `.to(fp16)` of the module | fp16 checkpoint tensors as they are | the file is fp16 |
| int8 ConvRot VAE | `comfy_quant` path (int8 linears, int8 attention) | not supported | out of scope; refused at load |
| output | fp32 `[1, F, H, W, 3]`, then `save_to` | uint8 `[F, H, W, 3]` (and fp32 on request) | the engine needs the frames |

## 8. Undetermined or risky for bit-exactness (what the pod run decides)

1. **Our bits are the engine's, not ComfyUI's**: GEMM order, attention decomposition and the norm reductions differ from ComfyUI's, so the end-to-end check is a PSNR / max-abs comparison (`VVAE_MIN_PSNR` 40, `VVAE_MAX_ABS` 32 are guesses to be replaced by measured numbers).
2. **RoPE contraction** (4.3, item 4): a guess about nvcc; `rope_vs_kitchen_wheel` settles it (the wheel at the version ComfyUI 0.37.0 pins, built `--use_fast_math -fmad=true`).
3. **`F.rms_norm` on fp16**: torch's fused kernel (one rounding of `(x * rstd) * w`) is assumed, as for the DiT's bf16 `h3_rms_norm`; a composite path (round `x * rstd`, then multiply by `w` as a second fp16 op) differs by up to 1 ulp. Which one the pod's torch takes is not determined from the sources here.
4. **torch scalar division by reciprocal**: the blend weights and `create_token_ids` assume `tensor / python_scalar` is `x * (1.0f / s)` on CUDA (`div_true_kernel_cuda`'s CPU-scalar branch). The structure check (blend arithmetic over several overlaps) and the rope-table check test this.
5. **`addcmul` contraction** is argued bit-neutral (the product of two fp16 values is exact in fp32); `gemm_addcmul_epilogue` compares with `torch.addcmul`.
6. **`silu` / `exp` / `cos` / `sin` / `rsqrt`**: the same CUDA math functions as torch's kernels only if LocalRouter's versions agree and torch's silu uses the accurate `expf` (not `__expf`); checked by `swiglu` and `rope_table_vs_comfy`.
7. **`post_quant_conv` accumulation**: if cuDNN picks an fp16-accumulating algorithm for the 1x1x1 conv the GEMM (fp32 accumulation) differs at the 1e-3 relative level on a 24-term dot product; affects the first op only.
8. **Attention score precision** (4.4) is the largest numerical deviation; if the PSNR check is short of the mark, the first remedy is fp32 scores in a larger buffer (a GEMM variant with fp32 output), which would change `vv_softmax`'s input type.
9. **Memory**: the weights are 4.85 GB, the score scratch 207 MB, a 1344 x 768 clip canvas 173 MB plus the previous row's raw tiles (about 11 MB each); the twin holds the whole uint8 video (124 x 768 x 1344 x 3 = 384 MB) on the GPU.
10. **SDPA batch / head ordering and `nan_to_num`**: ComfyUI applies `nan_to_num` after the head merge; we apply it in place on the `[S, 2048]` rows (same elements). `torch.nan_to_num` leaves finite values untouched, so it is exact.
11. **Throughput** was not measured: about 9 TFLOP a tile, 196 tiles for the production video.
