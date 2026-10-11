# Kandinsky-6 29B Pro-distill DiT kernels

The transformer of Kandinsky-6.0 Pro-distill 5 s (Hugging Face `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`,
revision `1018d5828716e8bea3de99609f45c924de9c5564`), ported from upstream kandinsky-6 at
`eac06e39b904940baff2c408e925cdb0c4529269`: `kandinsky/core/components/{dit,piflow_dit,rope}.py`,
`kandinsky/core/tensors.py`, `kandinsky/runtime/kernels/sdpa.py`. Standard Pro is not part of this release.

| File | Contents |
| --- | --- |
| `gemm.cu` | `k6_gemm_bf16` (bf16 tensor-core GEMM, bias or GELU epilogue), `k6_gemv_tf32` (the time / modulation linears on TF32 tensor cores, unqualified vs native), `k6_gemv_f32` (strict-FP32 diagnostic) |
| `attention.cu` | `k6_attention`: streaming bidirectional attention, head dim 128, optional key mask |
| `ops.cu` | norms (LayerNorm, AdaLN `norm_mod`, per-head RMSNorm + RoPE), gated residual, casts, timestep sinusoid, patchify / unpatchify / grid split |
| `frag.cuh` | mma / ldmatrix / cp.async helpers (copy of the image engine's, own namespace) |
| `fatbins.zig` | the embedding module for the build (see Integration) |
| `src/engines/kandinsky6/dit/layout.zig` | config, typed views, shapes, modulation offsets, the 4471-tensor manifest |
| `src/engines/kandinsky6/dit/attention.zig` | attention sites, mask contract, streaming plan |
| `src/engines/kandinsky6/dit/launch.zig` | `Kernels`: module load and one checked launcher a kernel |
| `src/engines/kandinsky6/dit/block.zig` | weights binding, scratch `Plan`, `Dit.textBlock` / `forwardBlock` / `forward`, capture hook |
| `tools/twin/stk_twin/kandinsky6_dit/` | the Python twin (same fatbins through the driver) and native-upstream comparisons |

## ABI

The pipeline owns everything with a lifetime, as `docs/dev/TENSORFOLD-RUNTIME.md` (TensorFold v1.0.7) sets out: one
`cuda.Driver`, one `cuda.Context`, one blocking `cuda.Stream`, the weight buffers, inputs, outputs and the scratch.
The DiT borrows them and never allocates, synchronizes, loads files or creates CUDA objects other than its three
modules.

```zig
const dit = @import("kandinsky6_dit"); // root: src/engines/kandinsky6/dit/block.zig
var kernels = try dit.launch.Kernels.load(&driver, .{ .gemm = k.gemm, .attention = k.attention, .ops = k.ops });
defer kernels.unload();
var text_blocks: [8]dit.TextBlock = undefined;
var blocks: [60]dit.FusedBlock = undefined;
const weights = try dit.bindWeights(dit.layout.Config.distill, &lookup, &text_blocks, &blocks); // lookup.get(name) ?u64
const plan = try dit.Plan.init(dit.layout.Config.distill, shape, dit.Plan.default_ffn_rows);
// scratch: plan.bytes() bytes, 256-byte aligned, owned by the caller
const d: dit.Dit = .{ .k = &kernels, .s = stream, .w = &weights, .p = plan, .base = scratch.ptr, .fp32_linear = math, .hook = null };
try d.forward(inputs, outputs);
```

- Tensors are `layout.View`: device address, dtype (`bf16`, `f32`, `u8`, `i32`), rows, cols and row stride `ld` in
  elements. Every activation is [tokens, features] row-major with a token's heads contiguous (head h at h * 128).
- Weights: immutable BF16 views named as the checkpoint names them. The loader casts the file's 767 F32 tensors to
  BF16 round-to-nearest-even before binding, as upstream `runtime/weights.py:205-207` does (there are no FP32 weight
  islands at run time); `layout.forEachTensor` lists names, file dtypes and shapes, and its test pins the archived
  header (4471 tensors, 30,139,423,760 elements, listing sha256 `a5339cbe...c3bc9aa`).
- `Inputs` (B = 1, the native PiFlow call, `piflow_sampler.py:103-114`): `x_video` [T*H*W, 33] channels last fp32 or
  bf16 (cat of latent 16, cond 16, mask 1); `x_audio` [218, 40]; `text` [L, 3584] bf16 (L = 1024 padded); `pooled`
  [1, 768] bf16; `time_v`, `time_a` [1, 1] fp32 (sigma * 1000); `freqs_v` [1, 2048], `freqs_a` [1, 1024] fp32 (the
  TimeEmbeddings buffers as the module holds them); `visual_rope` [N, 256], `audio_rope` [218, 256],
  `text_rope_v` / `text_rope_a` [L, 256] fp32 (the RoPE modules' own [S, 1, 64, 2, 2] tables, head axis dropped);
  `mask` the text key mask (u8 [L], nonzero = keep, with the host-known valid count). T2V and I2V share the native
  default `visual_cond_scheme = "pretrain"` (`pipeline/config.py:134`, no override in `pro-distill.yaml:12-20`): I2V
  puts the reference latent in the separate `cond[:, 0]` channels with the frame-0 mask, keeps the noisy video state,
  31 latent frames and 50,220 tokens at 864x480 (`core/algo/denoise_loop.py:30-31,49-50,60,64`). The explicit `i2v`
  overwrite and the optional `tail_cond_first_frame` branch (32 frames, 51,840 tokens, token-type ids) are not
  supported: neither has a native-byte gate, and the DiT takes no type ids.
- `Outputs`: the PiFlow grids, grid-major and contiguous: video bf16 [10 * T * H * W, 16] = [10, T, H, W, 16], audio
  bf16 [10 * 218, 40] = [10, 218, 40], i.e. `view(.., 10, dx).movedim(-2, 1)` made contiguous.
- Scratch: `Plan.init(config, shape, ffn_rows).bytes()`. Linear in the token count, no score matrix; 4.0 GB at the
  864x480 T2V preset (50,220 tokens) with `ffn_rows = 4096`. Rows are independent in every kernel, so `ffn_rows` only
  trades scratch for launches, never bits.
- `Hook`: called after each named step with its output view, named by the upstream module path
  (`video_text_transformer_blocks.2.attn`, `visual_transformer_blocks.17.video_dec_block.pre_ca`,
  `visual_transformer_blocks.17.av_cross_attention`, `out_layer`, `video_velocity_grid`, ...); for captures and per-op
  events. The hook runs before anything reads the view, so a parity harness may also overwrite it in stream order
  (inject the native capture of that op) to qualify everything downstream independently of the op itself.
- `fp32_linear` (`launch.Fp32Linear`, no default): the math of the TimeEmbeddings and Modulation linears. Native's
  `high` matmul policy permits TF32 (or an FP32 fallback); the executed algorithm is uncaptured and every form is
  unqualified (see the precision receipt). `tf32_rna` / `tf32_truncate` pick the TF32 operand rounding, `fp32` is a
  diagnostic.
- `Dit.forwardBlock(block, i, BlockIo)` and `Dit.textBlock(...)` run one block for block-level capture and timing.

## Kernels

| Kernel | Grid / block / shared | Requirements |
| --- | --- | --- |
| `k6_gemm_bf16` | (ceil(N/128), ceil(M/128)) / 256 / 40,960 static | any K and strides (`vec` 0 stages element-wise); `vec` 1: K, lda, ldb % 8 == 0, A, B 16-byte aligned |
| `k6_gemv_tf32` | (ceil(N/64), ceil(M/16)) / 256 / 0 | W contiguous [N, K]; sm_80+ (`cvt.rna.tf32`, `mma` m16n8k8 tf32) |
| `k6_gemv_f32` | ceil(N/8) / 256 / 0 | W contiguous [N, K] |
| `k6_attention` | (ceil(W/64), heads, 1) / 128 / 32,768 dynamic | ldq, ldo even; ldk, ldv % 8, k, v 16-byte aligned; >= 1 visible key |
| `k6_layer_norm`, `k6_norm_mod` | rows / 256 | |
| `k6_qk_norm_rope` | ceil(rows * heads / 8) / 256 | head dim 128 |
| elementwise / layout | ceil(n/256) or (ceil(dim/256), rows) / 256 | contiguous where the launcher says so |

Build flags: torch's nvcc defaults plus `-O3` (no fast math, default `--fmad`), `-gencode
arch=compute_{121a,120a},code=sm_{121a,120a}`, as `build/qwen.zig` and `build/minimax.zig` build their kernels. The
kernels never rely on contraction: every fp32 step whose rounding matters is an explicit `__fmul_rn` / `__fadd_rn` /
`__fmaf_rn`.

## Precision receipt

Two layers, as CONTRIBUTING and the discovery handoff require, and both are bit-equality gates; there is no
threshold and no waiver.

1. Production vs the twin: the twin launches the same fatbins with the same arguments, so every op, every chain and
   the whole forward must be bit-equal on the target GPU (zero unequal values, zero unmatched ops).
2. Native upstream (torch 2.14.1+cu130, attention engine forced to `sdpa`, beautifier off, cache / compile / offload
   off, `set_float32_matmul_precision("high")` as `factory.py:162` sets it, never forced to `highest` or TF32 off) vs this
   engine: the acceptance is
   bit-equality. Native's reductions run in library kernels whose algorithm is chosen at run time (cuBLASLt
   heuristics, the SDPA backend and its split), so for every reduction op below the native kernel and its
   parameters must first be captured on SM121 (`CUBLASLT_LOG_LEVEL=5` for algorithm, tile and split-K; the torch
   profiler for the SDPA and norm kernels), then this kernel matched to it or the op declared unsupported. Until
   that capture exists every row marked UNQUALIFIED is exactly that: not equal, not waived, not claimed.

| Op | Upstream (torch 2.14.1 CUDA) | Here | Native status |
| --- | --- | --- | --- |
| bf16 Linear (+bias) | cuBLASLt bf16, fp32 compute, bias epilogue, heuristic algorithm (may split-K) | mma.sync fp32 accumulation, K ascending in tiles of 32, no split-K; bias added to the fp32 sum, one rounding | UNQUALIFIED: algorithm uncaptured |
| FFN proj + GELU | bf16 GEMM (cublasGemmEx), round, `F.gelu` exact in fp32, round | same two roundings in one epilogue | GEMM as above; epilogue elementwise-equal |
| fp32 Linear (TimeEmbeddings linear_1 / linear_2 `dit.py:54-64`, Modulation `dit.py:124-131`) | FP32 operands and output at the linear boundary (weights BF16 at load, `runtime/weights.py:205-207`, widened by `.float()`), under `set_float32_matmul_precision("high")` (`pipeline/factory.py:159-162`), which permits a TF32 algorithm (torch 2.14 maps `high` to the cuBLASLt `CUBLAS_COMPUTE_32F_FAST_TF32` compute type) without fixing the SM121 algorithm or instruction stream | `Dit.fp32_linear`, no default: `k6_gemv_tf32` (TF32 operands, `tf32_rna` or `tf32_truncate`, mma m16n8k8 fp32 accumulation, K ascending, no split-K, bias once) or `k6_gemv_f32` (strict FP32, diagnostic) | UNQUALIFIED: the executed algorithm (TF32 or FP32 fallback, operand rounding, tile, split-K) is uncaptured. Neither form is claimed native; the oracle keeps its `high` policy (never `highest` to suit a kernel) |
| LayerNorm (affine, bf16) | `vectorized_layer_norm_kernel` T = bf16: block (32, 4), Welford with `1.0f / count` multiplies, shuffle-down then shared tree, `gamma * (rstd * (x - mean)) + beta` | torch's statistics and output expressions ported in plain operators at torch's block shape (`ops.cu`, BSD-3 provenance) | UNQUALIFIED until GPU: expected equal; the twin asserts it against torch, and nvcc's contraction must match torch's build (SASS of both is comparable on the host) |
| AdaLN `norm_mod` | `vectorized_layer_norm_kernel` T = fp32 on `x.float()`, then `* (scale + 1)`, `+ shift` as separate fp32 ops, `.to(bf16)` | the same port with fp32 loads of the widened bf16, then the separate steps with explicit `_rn` | UNQUALIFIED until GPU: as LayerNorm |
| RMSNorm (q / k) | `_fused_rms_norm` = the same kernel with `rms_norm`: one (32, 4) block a 128-wide row, warp 0 sums `sigma2 + val * val`, shuffle-down to lane 0, eps = FLT_EPSILON, `w * (rstd * x)` | one warp a row with torch's expressions and lane 0's sum (the other warps add exact zeros in torch) | UNQUALIFIED until GPU: expected equal, asserted by the twin |
| RoPE | `(rope * x.float()).sum(-1)` with the module's fp32 table, then bf16 | reads the same table, same products and one add | elementwise: expected equal |
| Gated residual | `(x.float() + gate.float() * out.float()).to(bf16)` | same | elementwise: expected equal |
| Attention | SDPA: flash (FA2) unmasked, mem-efficient with the additive mask; scale 1/sqrt(128) | streaming online softmax over 64-key tiles, ex2.approx, p rounded to bf16 for P.V, div.full; scale 0x3db504f3 | UNQUALIFIED: tiling, exp and split differ; the native kernel must be captured and matched |
| bf16 adds (time + pooled) | fp32 add, one rounding | same | elementwise: expected equal |
| Time sinusoid | `outer(t, freqs)` fp32 (elementwise, not a GEMM), `cos` / `sin` (libdevice) | same | expected equal given the same `freqs` |
| Layout (patchify, unpatchify, grid split, casts) | views / permutes / `.to(bf16)` | same | exact |

Inputs whose bits the pipeline must produce as upstream does: the RoPE tables (built on the CPU at load, cos / sin on
the GPU), `TimeEmbeddings.freqs` (CPU `torch.exp` at load) and the latents. A native receipt records, per captured op, the loaded and cast operand dtypes and
shapes, the effective precision settings and environment overrides, the Torch / CUDA / cuBLAS versions, the selected
library algorithm and the reference output bytes. The reference latent of I2V comes from `vae.encode(...).latent_dist.
sample()` without an explicit generator (`prepare_latents.py:236`), drawn before the seeded 31-frame video noise
(`prepare_latents.py:26-29`; audio seed + 1, `:58-60`), so DiT input fixtures are captured tensors, never re-derived
from the seed alone.

## Coverage

Every operator of the fused forward (`dit.py:855-935`, `piflow_dit.py:34-45`) at the checkpoint's flags
(`ca_rope`, `cross_gates`, `fix_modulation`, `text_token_padding`, `visual_token_type_num_embeddings = 2`):

- text, pooled, audio embeddings (bf16 Linear + affine LayerNorm); visual patch embedding (132 features); the token-type
  embedding is bound but unused (only the unsupported tail branch adds it);
- TimeEmbeddings (sinusoid, fp32 linear, SiLU, fp32 linear, bf16) + pooled projection;
- 2 x 4 TransformerEncoderBlocks (6-way modulation, masked self-attention with text RoPE, FFN);
- 60 FusedTransformerDecoderBlocks: video / audio self-attention (RoPE3D / RoPE1D x 0.144), text cross-attention
  (masked keys, no RoPE) with the video residual deferred, va / av modulation with cross gates, V<-A and A<-V
  attention from the pre-CA tensors with both ropes, FFN 4096->16384 and 2048->7168;
- OutLayer (2-way modulation, norm_mod, 4096->640, unpatchify) and OutLayerAudio (norm_mod, second LayerNorm,
  2048->400), with the 10-grid split.

Attention sites (`attention.Site`): 50,220 x 50,220 x 32 heads, 218 x 218 x 16, 1024 x 1024 masked (both text
stacks), 50,220 x 1024 and 218 x 1024 masked, 50,220 x 218 x 32 and 218 x 50,220 x 16. Ragged tails: 50,220 =
784 x 64 + 44 queries, 218 = 3 x 64 + 26 keys; every kernel bounds-checks rows and keys.

## Integration (for the build owner)

These paths are new and touch no central file. To wire them (serialized with other central edits):

1. `build/kandinsky6.zig` (or the engine's build file): for each of `gemm`, `attention`, `ops`, an nvcc
   `-fatbin` run of `kernels/cuda/kandinsky6/dit/<name>.cu` with the flags above, added to a module rooted at
   `kernels/cuda/kandinsky6/dit/fatbins.zig` as anonymous imports `k6_dit_fatbin_<name>`, with an options module
   `k6_dit_options` holding `with_kernels: bool = (nvcc != null)`; install as `fatbin/k6_dit_<name>.fatbin`.
2. A module rooted at `src/engines/kandinsky6/dit/block.zig` importing `cuda`; its `addTest` runs the layout,
   attention, launch and block tests (host only, no GPU).
3. `zig fmt --check` already covers `src/`; add nothing else. NOTICE: `kernels/cuda/kandinsky6/dit/attention.cu` and
   `frag.cuh` derive from TensorFold's prefill attention and fragment helpers (Apache-2.0), like the image engine's;
   the norm statistics in `ops.cu` port PyTorch v2.14.1 `aten/src/ATen/native/cuda/layer_norm_kernel.cu` (BSD-3-Clause).

## Checks

- Host: `zig fmt --check`; the module's tests under Zig 0.17.0 against TensorFold v1.0.7's host runtime; nvcc
  compile of the three files for sm_121a and sm_120a (resource usage recorded in the PR).
- GPU (pending, approved SM121 window only): `STK_K6_DIT_FATBINS=DIR python -m stk_twin.kandinsky6_dit.test_ops
  --fp32-linear {tf32_rna,tf32_truncate,fp32} --out receipt.json` from `tools/twin` (twin self-consistency and the
  per-op / per-capture native comparison at the oracle's own settings), then Zig vs twin per op / per block / full
  forward bit-equality, native SM121 byte fixtures for every UNQUALIFIED row (algorithm evidence included), per-op
  GPU events and measured scratch.
