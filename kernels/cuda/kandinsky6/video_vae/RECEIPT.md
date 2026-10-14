# Kandinsky-6 video VAE kernels: ABI, source and precision receipt

Component D2 of the Kandinsky-6 29B Pro-distill port: the video VAE encoder (I2V first frame) and decoder (latent to RGB
frames). It is the same VAE for every Kandinsky-6 checkpoint of this family and does not depend on the DiT variant.

| Path | Role |
|---|---|
| `kernels/cuda/kandinsky6/video_vae/conv.cu` | the fp16 implicit-GEMM convolution / linear / attention GEMM |
| `kernels/cuda/kandinsky6/video_vae/ops.cu` | GroupNorm, softmax, transpose, boundary conversions, posterior, blends, copies |
| `src/engines/kandinsky6/video_vae/launch.zig` | parameter blocks, `Launcher`, weight layout, GroupNorm / resnet / attention sequences, host simulator |
| `src/engines/kandinsky6/video_vae/tiles.zig` | tile plans, frame contract, memory policy |
| `src/engines/kandinsky6/video_vae/decode.zig` | postprocess_video: bf16 latent to uint8 RGB |
| `src/engines/kandinsky6/video_vae/encode.zig` | encode_i2va_first_frame's VAE part: uint8 RGB to latent and distribution |
| `src/engines/kandinsky6/video_vae/testdata/vae_tensors.txt` | the pinned checkpoint header (names, dtypes, shapes) |

## Source pins

| | |
|---|---|
| Upstream | `kandinskylab/kandinsky-6` `eac06e39b904940baff2c408e925cdb0c4529269` |
| `kandinsky/core/components/hunyuan_vae.py` | sha256 `f5e8dd402aba625f23aef86ab7d5324dc6a4e8585b72917a3bf27bd46745bbaf` |
| `kandinsky/core/components/vae_video.py` | sha256 `7ad34af7bbf99319e223ebaf5a2048e15d324692740d0fbcbc0718030df1ec28` |
| `kandinsky/core/algo/prepare_latents.py` | sha256 `edd9cdef58ce31a20442eea568621578fd129722325992850b31cdaa95751ee2` |
| `kandinsky/core/algo/postprocess_video.py` | sha256 `476474b6827476d06c62b0e70957bd2f807e04261676292c3b116b9087c92847` |
| Checkpoint | `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers` `1018d5828716e8bea3de99609f45c924de9c5564` |
| `vae/config.json` | sha256 `8dd1ef5ce49d2a56cdb5ea4c6553244320cd7185918463578c4c08a31f55e50a` |
| `vae/diffusion_pytorch_model.safetensors` header (28,648 bytes) | sha256 `ec504903cc22243034f7fec06892e5d82970b0c527bdc22e5ce85a2a924471f0` |
| Reference runtime (upstream `uv.lock`) | torch 2.14.1 (cu130), diffusers 0.40.0 |

Only the config and the safetensors header (an HTTP range read) were fetched; no weights. The header has 248 fp32
tensors, 246,478,803 parameters; `launch.zig` enumerates exactly these names and shapes (tested against the fixture).

## Graph

```
encode (I2V, opt_tiling=False: always one direct pass)
  uint8 RGB -> rgb8_in -> conv_in 3->128 -> down0 [2 resnets 128, conv s(1,2,2)] -> down1 [128->256, conv s(2,2,2)]
  -> down2 [256->512, conv s(2,2,2)] -> down3 [2 resnets 512] -> mid [resnet, attention, resnet]
  -> GroupNorm+SiLU -> conv_out 512->32 -> quant_conv 1x1x1 -> posterior (mean, std, latent * 0.476986)

decode (per temporal tile of 5 latent frames at the native size; 14 tiles for 31 latent frames)
  fp32 latent -> latent_in (half(z * float(1 / 0.476986))) -> post_quant_conv 1x1x1 -> conv_in 16->512
  -> mid [resnet, attention, resnet] -> up0 [3 resnets 512, nearest (1,2,2) + conv] -> up1 [3 x 512, (2,2,2) + conv]
  -> up2 [512->256, (2,2,2) + conv] -> up3 [256->128] -> GroupNorm+SiLU -> conv_out 128->3
  -> blend_t with the previous tile (8 frames) -> rgb_out of the kept frames into the caller's uint8 output
```

Every causal convolution is `F.pad(replicate: time k-1 front, space k/2 both sides)` then a stride-s convolution; the
nearest upsample (first frame spatial only, the rest also x2 in time) is folded into the convolution's gather, which
reads the same source element the materialised upsample would hold. Residual adds and the conv biases are in the GEMM
epilogue with the reference's roundings.

## Native call contract

Pinned to `eac06e39` and checkpoint `1018d58`, following the Captain's native contract correction of 2026-10-11 (which
cites `pro-distill.yaml:12-20`, `pipeline/config.py:134`, `pipeline.py:345,437-443,495-511,616-627`,
`denoise_loop.py:30-64`, `factory.py:159-162`). The line numbers below are in the pinned files hashed under Source pins.

- Default image conditioning is `pretrain`: the encoded first frame goes into separate `cond[:,0]` channels plus a
  mask; the noisy video latent keeps 31 frames and frame 0 is not overwritten. T2V and default I2V therefore both
  decode 31 latent frames to 121 frames at 24 fps (`decode.zig` takes the frame count as given). Explicit `i2v` also
  keeps 31 frames. The optional `tail_cond_first_frame` scheme (32 latent frames, reference removed before decode at
  `pipeline.py:616-627`) is a different branch, not released here; this component has no tail handling.
- Encode input (`prepare_latents.py:214-237`): `TF.pil_to_tensor` (uint8), the cover resize `TF.resize` and centre
  `TF.crop` on that uint8 tensor (or the `max_area` aspect path), then `tensor / 127.5 - 1.0` and `.to(float16)`.
  Decoding, resizing and cropping the image belong to the caller; `encode.zig` starts at the uint8 canvas.
- `latent_dist.sample()` (`prepare_latents.py:236`) passes no generator: `eps` comes from torch's implicit default CUDA
  generator, drawn before the seeded video noise (all 31 frames, video generator) and the audio noise (seed + 1).
  Fixtures must capture that `eps` (or the generator state) separately; the video seed does not determine it.
- Decode (`postprocess_video.py:32-37`): `frames / scaling_factor`, `.to(float16)`, `vae.decode`,
  `((x.clamp(-1, 1) + 1) * 127.5).to(uint8)`. The VAE loads as fp16 (`vae_video.py:11`). In the Pro-distill sampler
  the latent reaching it is fp32, not the DiT's bf16: the sigmas are fp32 tensors of shape `(batch, 1, ...)`
  (`piflow_sampler.py:88,123-129`), so `policy_rollout_fm`'s `x_t + velocity * (sigma_t_minus - sigma_t)` and
  `torch.where` (`piflow_math.py:182-185`) promote the bf16 state to fp32 from the first outer step on;
  `sample_video` keeps that rollout (`piflow_sampler.py:156-164`) and `pipeline.py:616-631` passes it to
  `postprocess_video` without a cast. (Reported by the central runtime and checked against the pinned source; the
  oracle's f_dec and n_dec cases decode fp32 latents.)
- Backend settings: the native pipeline sets `torch.set_float32_matmul_precision("high")` before model construction
  (`factory.py:162`). That governs fp32 matmuls and does not set convolution precision; the VAE's convolutions,
  linears and attention run on fp16 operands. Neither that flag nor `allow_tf32` identifies the SM121 cuDNN / cuBLASLt
  / SDPA algorithm, its accumulation or its reduction order. This receipt infers no native arithmetic from flags. The
  oracle is the pinned native pipeline with its own settings, unchanged: do not set `highest`, disable TF32 or force
  cuDNN determinism or algorithms to make these kernels match.

## ABI

- The caller owns the CUDA driver, context, stream, modules and every buffer (docs/dev/TENSORFOLD-RUNTIME.md contract):
  components take device addresses (`u64`) and issue `Launch`es through `launch.Launcher`; they import no CUDA runtime
  and never allocate, load, free or synchronise. `launch.zig` documents the adapter to `cuda.launch` (one by-value
  struct parameter per kernel, all shared memory static, at most 40,960 bytes).
- Kernel symbols are `k6v_<name>` (`launch.symbol`), in two images: `conv` (conv.cu) and `ops` (ops.cu) (`launch.image`).
  Parameter blocks are asserted equal in size and offsets on both sides (Zig `comptime`, C++ `static_assert`).
- Weights: the loader converts each checkpoint fp32 tensor to fp16 with round-to-nearest-even (torch `.to(float16)`);
  convolution and linear weights are repacked to `[co][ldw]` rows, `k = ((dt*kh + dy)*kw + dx)*ci + c`, zero padded to
  `ldw = round_up(k, 8)` (`launch.repackConv`). Biases, GroupNorm weights fp16 `[c]`.
- Activations are channels last (`[T][H][W][C]`). Decode input: the packed THWC latent (`[frames][lh][lw][16]`), all
  31 frames for T2V and for default I2V (see Native call contract), fp32 (the Pro-distill sampler's output,
  `launch.latent_in_dtype = .f32`) or bf16, chosen by `decode.Io.latent_dtype`. The tile plan is the caller's:
  `tiles.decodePlan(frames, lh, lw, policy)` with the caller's free-memory figure; the component probes no memory.
  Decode output: uint8 `[(frames-1)*4+1][8 lh][8 lw][3]`. Encode output: fp16 `[lt][lh][lw][16]` latent (and optionally
  mean / std); the caller supplies `eps` (fp16 standard normal noise) for `.sample()`, or none for `.mode()`. Native
  draws `eps` from torch's implicit default generator (see Native call contract); this component owns no RNG.
- Scratch is one caller buffer sized by `decode.scratchBytes(plan)` / `encode.scratchBytes(frames, h, w)` and carved by
  a bump arena in a fixed order; the graphs refuse a smaller buffer. Native sizes: decode 31 x 60 x 108 =
  14,895,317,248 bytes (four 3.61 GB activations of 17 x 480 x 864 x 256 fp16, attention over 32,400 tokens, two
  17-frame fp16 RGB tiles); the same decode under the 8 GB-free policy (2 x 2 spatial tiles) 4,231,848,704 bytes;
  I2V encode at 480 x 864 500,593,408 bytes. Weights: 492,957,606 bytes as fp16 (plus the k padding of conv_in).

## Precision contract

Native reference: AutoencoderKLHunyuanVideo loaded with `torch_dtype=float16`: fp16 parameters and activations. Each
row below is the arithmetic of this implementation. The "Native" column records what the pinned source fixes and what
it leaves to the backend; it is not evidence of native bits.

Qualification status: the decoder is qualified on SM121 for the qualified plan (QUALIFICATION.md, "SM121 qualification
record": the whole decode bit-equal to the reference, every policy exact on every fixture of its op; the production
path's own gate, with no VAE options, passed: `K6_VAE_PROD_EXACT`, and the full pipeline `K6_FULL_E2E_EXACT`). The decode runs
`launch.zig` `qualified` unconditionally; the "component's own" orders below remain for the encoder (unqualified) and as
the fixtures' baselines. Any other plan, the encoder and I2V stay unqualified until fixtures of the native pipeline,
captured with its own settings, match these kernels byte for byte. Record with each fixture:
operand dtypes and shapes, the effective precision and cuDNN / SDPA settings, environment overrides, the torch / CUDA /
cuDNN / cuBLAS versions, and the reference outputs. A mismatch is fixed in these kernels, never by changing the oracle,
a threshold or the reduction order to suit them.

| Operation | Implementation | Native |
|---|---|---|
| Convolution | fp32 accumulation over k16 chunks ascending (`mma.sync` m16n8k16), then `half(half(acc) + bias)`; the K sequence is tap-major, or under a conv rule (the decode's `qualified.conv`, per exact shape) channel tiles of T outer and taps inner (`ctile`, CUTLASS's optimized fprop iterator order), optionally alternating channel tiles (`split` 2, cudnn's xmma fprop with grid z = 2) or k16 chunks (`split_by` chunk: two interleaved mma chains) between two fp32 accumulators joined before the epilogue (the `_k2` kernels), or with `kslot` quad32 the two chunks of each 32-channel block taking alternate 4-channel quads (the decoder conv_out's cudnn `indexed_wo_smem` kernel, measured by `vae_conv_oracle`), the order the convolution fixture selects among candidates (tile sizes, k8/k16 MMAs, split-K) against torch's cudnn bits | cuDNN fp16 conv, then `output.add_(bias)`; the two roundings here are an implementation choice, and cuDNN's accumulation and order on SM121 are undeclared until fixtures |
| Linear (attention q, k, v, out) | `half(acc + bias)`; in the decode (`qualified.attn_qkv_bias`, separate) q, k and v take `half(half(acc) + bias)`; k16 chunks ascending | to_out: cuBLASLt with the bias epilogue (one rounding). q, k, v: the input is non-contiguous (`group_norm(h.transpose(1, 2)).transpose(1, 2)`), so `at::linear` runs `matmul` then `add_(bias)` (torch 2.14.1 `Linear.cpp`; the reference cuBLASLt log shows `cublasLtHSHMatmul` without an epilogue for them). The attention fixture (`vae_attn_isolate`, conv-fixture format) checks the order and both bias modes against torch |
| Residual add | `half(h + res)` in the epilogue | separate fp16 add: same |
| GroupNorm(32, eps 1e-6) | two-pass fp32 (sum, then centred squares with fmaf) in fixed blocks of 2,048 positions and a fixed tree; mean and variance divided in fp64; `rstd = rsqrtf(var + eps)`; `y = half(fmaf(x, rstd*gamma, fmaf(-rstd*gamma, mean, beta)))` | torch 2.14.1 (`group_norm_kernel.cu` RowwiseMomentsCUDAKernel<c10::Half, float>, source-read): one 512-thread block a group, Welford over the group's NCTHW elements at stride 512, warp-shuffle then block combine, `var = m2 / n`, `rstd = rsqrtf(var + float(half(eps)))` (eps cast to the input dtype first); apply as here; SiLU a separate fp16 op. `k6v_gn_welford` transcribes it and is the decode's (`qualified.gn`, V 7); V pins nvcc's fmaf contractions, V1 / V3 / V5 / V7 all exact on every GroupNorm fixture. First divergence of the 832-wide replay (trace v1): `decoder.mid_block.resnets.0.nonlinearity.0`, 879 of 2,375,680 values, inputs bit-equal |
| SiLU | `half(s / (1 + expf(-s)))` on the fp16 norm output | torch silu: same expression |
| Attention (1 head, dim 512, frame-causal mask) | `S = fp32 q.k`, `x = S * 512^-0.5`, fp32 max / `expf` / fixed-tree sum, `P = half(e / l)`, `O = half(fp32 P.V)` (row blocks of 1,024); in the decode (`qualified.attn_sdpa`, variant 524) the reference's mem-efficient kernel replayed (`ops.cu` k6v_sdpa_memeff: 32-query x 128-key blocks, exp2-domain online softmax, unnormalised `half(p)`, an fp32 output accumulator rescaled per key block; the variant selects the contractions, exp2 / reciprocal forms and the GEMMs' instruction shape, m16n8k8 as in the source or m16n8k16) | diffusers `Attention` (`hunyuan_vae.py:238`, `upcast_softmax=True`) through SDPA; on SM121 the call runs torch's mem-efficient kernel (fmha_cutlassF_f16_aligned_32x128_gmem_sm80; flash and cuDNN have no kernel for it), captured by the attention isolate and forced-backend checks |
| Nearest upsample, replicate pad | index arithmetic only | exact |
| Decode input | fp32 `z`: `half(z * inv)`; bf16 `z`: `half(float(bf16(float(z) * inv)))`; `inv = float(1.0 / 0.476986)` = `0x40062d04` | torch 2.14.1 `div_true_kernel_cuda` (`BinaryDivTrueKernel.cu:35-52`, file sha256 `db073437…`): a CPU scalar divisor becomes `a * inv_b` in opmath float with `inv_b = float(1.0 / double(b))`, the quotient in the input dtype. The double and the float reciprocal are the same bits for this factor; over every finite fp32 input, `half(x * inv)` differs from `half(x / float(0.476986))` (true fp32 division) for 35,718 inputs, so the reciprocal form is the one to match. Source-derived; bits still to be confirmed by the SM121 fixture (f_dec, n_dec) |
| Decode output | `uint8(trunc(half(half(clamp(x,-1,1) + 1) * 127.5)))` | same ops |
| I2V input | `half(float(u) / 127.5 - 1)` (true division) | CPU float division then `.to(float16)` |
| Posterior | `std = half(expf(half(0.5 * clamp(logvar, -30, 20))))`, `x = half(mean + half(std * eps))`, `latent = half(x * float(0.476986))` | same ops |
| Tile blends | `b = half(half(a * float(1 - y/e)) + half(b * float(y/e)))`, weights from double | same ops |

No numerical tolerance is used or proposed: every gate is bit equality. Bits do not depend on the
convolution variant chosen (wide / narrow, vector / scalar gather), the grid, M or N: each output element sees the
same k16 chunks in the same order. The attention row-block size (1,024) sets the P.V k-extent of a block and is fixed.

## Tiling and temporal behaviour

`tiles.zig` transcribes get_enc_optimal_tiling / get_dec_optimal_tiling / _temporal_tiled_decode / tiled_decode. The
reference chooses tiles from `torch.cuda.mem_get_info()`; this runtime takes an explicit policy (`.unbounded` = ample
free memory, the native 864 x 480 decode needs about 14.4 GB free to stay untiled spatially; `.free_bytes` replays a
recorded run). Native fixtures must be captured under the same policy. At 864 x 480, 31 latent frames: 14 temporal tiles of 5 latent
frames (17 decoded frames), stride 2, first tile keeps 9 frames, middle tiles 8, last 16, 8-frame linear blends;
121 frames emitted once each. Frames above 2^31 tile-elements (e.g. 1280 x 720) tile spatially regardless of memory.
There is no temporal feature cache in this VAE: tiles are independent decoder passes (their GroupNorm statistics are
per tile) meeting through in-place linear blends, which `decode.zig` replays in the reference's order (later blends read
earlier, already-blended tiles). Where the reference's full-tile ranges stop before the latent's edge, its output is
the kept boxes concatenated (cropped to 8 lh x 8 lw), narrower or shorter than nominal: the plan carries that emitted
extent apart from the nominal one and the decode writes it, with no edge tile, shift or padding (the reference run with
2,301,403,136 free bytes: a 2 x 4 grid, 112 passes, 832 x 480 of the nominal 864). Plans the reference would fail
(zero stride, empty range) are refused. I2V encode (`opt_tiling=False`) is always a direct pass; tiled encodes are outside this contract.

## H3 / Qwen primitive audit

No existing primitive is reused. LocalRouter's H3 video VAE is a ViT decoder (no 3D convolution, GroupNorm, causal
padding or temporal upsampling); Qwen-Image's VAE is single-frame bf16 with zero-padded im2col in channel-major k order
and no GroupNorm. The H3 fp16 GEMM's epilogue (`half(acc + bias)`) is the linear rounding, not the convolution's two
roundings, and its k order is torch's `[ci][kt][kh][kw]`, not the tap-major order the channels-last gather needs. These
kernels are therefore independent files with their own symbols; no DiT or H3 file is edited.

## Checks

Host (Zig 0.17.0): `zig test src/engines/kandinsky6/video_vae/decode.zig` and `.../encode.zig` (tiles and launch tests
run through both), `zig fmt --check src/engines/kandinsky6/video_vae`. The decode tests run the whole launch stream
in `launch.Sim`: every launch's bytes in a declared region, no write overlapping the call's reads, grids consistent,
and the blend / copy / emission kernels executed on the host against a transcription of the reference's loops (31,
30 and 3 latent frames; a 2 x 3 spatial grid whose 32-row blends exceed its 24-row stride). Mutations of the blend order
or of the temporal blend fail these tests.

Device: compile-only `nvcc -cubin` for `sm_121a` with torch's extension flags (`-D__CUDA_NO_HALF_OPERATORS__`
`-D__CUDA_NO_HALF_CONVERSIONS__` `-D__CUDA_NO_BFLOAT16_CONVERSIONS__` `-D__CUDA_NO_HALF2_OPERATORS__`
`--expt-relaxed-constexpr -std=c++20 -O3`), CUDA 13.3. Result (2026-10-11, base `5d5cc87`): exit 0, no warnings; 15
entry points, all with 0-byte stack frames and no spills; the most is 118 registers with 40,960 bytes of shared memory
(the wide convolution).

Repository (same run, Zig 0.17.0): `zig build` exit 0; `zig build test` exit 0; the component tests ran 17 and 14 tests
and all passed; `zig fmt --check build.zig build src tests` reports the same 21 findings as base, none new. These files
are not yet reachable from `build.zig`, so `zig build` and `zig build test` cover the unchanged base graph. The
`zig build test` log contains a test runner `failed command:` diagnostic despite exit 0. This line is inherited: the
D0 baseline `zig build test` log for the same base graph (`~/.cache/k6-d0/base-test.log`, sha256
`4c582e561fee753a1e51ed8a61a13b60db457b9813455b721c7ecc552cd776b9`) has the same diagnostic on line 8, followed by
`Build Summary: 22/22 steps succeeded; 77/77 tests passed`.

Pending approved SM121 hardware (not claimed): every operator alone, chained blocks, encoder and decoder end to end and
raw RGB / latent / distribution bytes equal to SM121 fixtures of the pinned native pipeline (captured as described under
Precision contract, with the separately captured `eps`); measured peak memory and per-phase GPU time. Until then every
path here is unqualified. Integration (build wiring of the two fatbins into a
module, `zig build test` reachability of these files, the loader using `forEachTensor` / `repackConv`) belongs to the
pipeline / integration owner.
