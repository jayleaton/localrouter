# Kandinsky-6 video VAE: SM121 qualification contract

How the native oracle (pinned upstream torch) and this component's kernels are run on the target GPU and compared
byte for byte. Both sides emit the same named checkpoints in the same record format; a host tool compares them after
the window. No threshold is applied: a checkpoint is equal or it is not.

## Jobs

| Job | Runs | Needs |
|---|---|---|
| `oracle` | the pinned native code (`kandinsky-6@eac06e39`, torch 2.14.1+cu130, diffusers 0.40.0, Python 3.13) on the real checkpoint, with forward hooks | the offline aarch64 Python bundle, the upstream source, `vae/`, the inputs |
| `engine` | `k6v-qual` (`src/engines/kandinsky6/video_vae/qual.zig`, loading the sm_121a/sm_120a fatbins) on the same checkpoint and inputs | the binary, the fatbins, `vae/`, the inputs |
| `compare` | host only, after both jobs: records by name, equality of sha256, first differing element of raw blobs | both record sets |

The two GPU jobs share nothing at run time and run on separate nodes in parallel, under the shared staging runner
(`tools/kandinsky6/staging/run-node.py`): each node's own immutable image ID, enforced container memory, read-only root
and kit, the checkpoint at `/models`, outputs in `K6_OUTPUT`, window deadlines and a memory / GPU / disk watch.
`tools/spark/k6-video-vae/operator_spec.py` writes the jobs for its sealer; the oracle takes the pinned native stack from
this component's original pybundle (copied unchanged to each node, mounted read-only) and checks every file of it, and
the pinned versions, before it runs. The engine never consumes oracle output: the posterior is checked with the kit's
own `eps`, which the oracle also applies through the native ops.

GB10 reports no GPU memory counter, so the jobs run under the D2 memory policy `gb10-unified-d2-v2`
(`tools/spark/k6-video-vae/D2-EVIDENCE.md`). On each node an attribution probe (`run-probe.sh`) runs first, and its
PASS gates that node's payload. The oracle is a sealed sequence of guarded processes, each with its own evidence:
`env-probe`, then one per case. Each runs under a 38 GiB Torch allocator cap. In the engine, every device buffer goes
through a 24 GiB ledger. These caps are allocation evidence: the context, modules and driver allocations outside them are a
declared overhead in each job's process budget, which the runner admits against host MemAvailable.

## Cases

Inputs are generated once by the kit builder (`tools/spark/k6-video-vae/inputs.py`, NumPy PCG64 seeds) and shipped as raw
little-endian files, so both sides read identical bytes.

| Case | Input | Path | Records |
|---|---|---|---|
| `s_dec` | latent bf16 `[3][8][12][16]` | `postprocess_video` (direct decode, 9 frames 64 x 96) | every checkpoint, raw |
| `f_dec` | latent fp32 `[3][8][12][16]` | the same with the Pro-distill sampler's fp32 latent | every checkpoint, raw |
| `s_enc` | image uint8 `[64][96][3]`, eps fp16 `[1][8][12][16]` | `encode_i2va_first_frame` (height 64, width 96) | every checkpoint, raw |
| `n_enc` | image uint8 `[480][864][3]`, eps fp16 `[1][60][108][16]` | the same at the native canvas | every checkpoint hashed; inputs, moments, mean, std, latents raw |
| `n_dec` | latent fp32 `[31][60][108][16]` | the native 121-frame decode (14 temporal tiles) | first-tile checkpoints hashed; `decode.input` and `decode.rgb` raw |

Native tiling is chosen by `torch.cuda.mem_get_info()`; the oracle records the free bytes and the tiling the model
chose. The engine probes no memory: it runs `tiles.decodePlan(.., .unbounded)`, or `.free_bytes` with the oracle's
recorded figure (`k6v-qual --free-bytes`), and records its plan. A different plan fails the case.

## Checkpoints

Names are the native module names relative to the VAE; a module checkpoint is that module's forward output, in its
first call only (later temporal tiles are not recorded). Tensors are recorded channels-last: 5-D `(B, C, T, H, W)`
outputs as `[T][H][W][C]` (batch 1), 3-D `(B, N, C)` as `[N][C]`.

- `post_quant_conv`, `quant_conv`
- `{encoder,decoder}.conv_in`, `{encoder,decoder}.conv_act`, `{encoder,decoder}.conv_out`
- `encoder.down_blocks.{i}.resnets.{j}`, `encoder.down_blocks.{i}.downsamplers.0`
- `decoder.up_blocks.{i}.resnets.{j}`, `decoder.up_blocks.{i}.upsamplers.0`
- `{encoder,decoder}.mid_block.resnets.{0,1}`, `{encoder,decoder}.mid_block.attentions.0`
- within a resnet `R`: `R.nonlinearity.0` (SiLU of norm1), `R.conv1`, `R.nonlinearity.1` (SiLU of norm2),
  `R.conv_shortcut` when present; the resnet output is `R` (conv2 + residual)
- within the attention `A`: `A.group_norm` (`[N][C]`), `A.to_q`, `A.to_k`, `A.to_v`; its output `A` includes the
  residual

- tile assembly (decode), each method's return value at its first call, `[T][H][W][3]`: `blend_v` and `blend_h` (the
  spatial tile after that blend), `tiled_decode` (a temporal tile's assembled frames, when spatially tiled), `blend_t`
  (a temporal tile after its dropped frame and blend)

Not recorded, because the kernels fuse them: GroupNorm outputs before SiLU (`norm1`, `norm2`, `conv_norm_out`), conv2
before the residual add, `to_out.0` before the residual add, and the decoder's fp16 output before the uint8 conversion.

Boundary checkpoints (not modules):

| Name | Value |
|---|---|
| `decode.input` | the fp16 tensor passed to `vae.decode` (`[T][H][W][16]`) |
| `decode.rgb` | the uint8 frames `[T][H][W][3]` |
| `encode.canvas` | the uint8 image after the native resize and crop (`[H][W][3]`); the engine's input must equal it |
| `encode.input` | the fp16 tensor passed to `vae.encode` (`[1][H][W][3]`) |
| `encode.mean`, `encode.std` | `latent_dist.mean`, `latent_dist.std` (`[1][h][w][16]`) |
| `encode.latent_kit` | `(mean + std * eps_kit) * scaling_factor` with the native ops |
| `encode.eps_native`, `encode.latent_native` | oracle only: the implicit-generator draw of `latent_dist.sample()` and the native latent, kept as pipeline fixtures |

## Record format

A record set is a directory:

- `ops.jsonl`: one line per checkpoint in emission order, `{"i", "case", "name", "dtype", "shape", "sha", "raw"}`;
  `dtype` is `f16`, `bf16`, `u8` or `f32`; `sha` is the sha256 of the contiguous little-endian bytes; `raw` says
  whether `blobs/<sha>` holds them.
- `blobs/<sha>`: raw bytes.
- `env.json`: GPU name and compute capability, driver, CUDA, and for the oracle torch, cuDNN, diffusers versions, the
  precision settings in force (`float32_matmul_precision`, `allow_tf32` flags, cuDNN `benchmark`/`deterministic`, SDPA
  backends enabled), `mem_get_info`, the chosen tilings, and a profiler list of the kernels each case launched; for the
  engine the binary's sha256 and the fatbins' sha256.
- `summary.json`: per case wall time, peak device memory, exit status.

The oracle keeps the pinned pipeline's own settings (`torch.set_float32_matmul_precision("high")` as
`pipeline/factory.py` does before construction, and nothing else changed). It never sets `highest`, disables TF32 or
forces cuDNN algorithms.

## SM121 qualification record

The native video VAE decode (latent → fp16 decoder input → 112 decoder passes → blends → uint8 frames) is bit-equal to
the pinned native Python reference on the target GPU, under the qualified plan. This is the evidence that qualifies it
(CONTRIBUTING, "Engine correctness"). No threshold is applied: every compared tensor is byte-identical. It records
arithmetic, not admission: `admission.zig`'s contracts are a separate decision.

### Result (trace v7)

| Item | Value |
|---|---|
| `k6.vae.decode` frames | uint8 [1, 3, 121, 480, 832], sha256 `65e5693adef7c61864613c3f28b408f5406eecfc391172c2497fca4a1340df26` (reference 2), native bit-equal |
| `k6.aux.vae.decoder_in` | fp16 [1, 16, 31, 60, 108], `d228763c5c0936b51470cce68131e9bbe0788a69af7d69a3f75cb0dd510d0c3d`, bit-equal |
| Trace | 75 / 75 checkpoints equal (decoder pass 1 and the first blends / tile assembly), first divergent: none |
| Compare | `stk_twin.kandinsky6.compare` over `k6.vae.,k6.aux.vae.`: 77 pairs, 0 failed |
| Blend fixture | 10 / 10 cases bit-equal (upstream `blend_t` / `blend_v` / `blend_h` vs `k6v_blend`) |

Earlier traces (v3 to v6) proved decoder pass 0 checkpoint by checkpoint with the same policies as they became exact.
Trace v7 traced pass 1 and compared the whole decode's frames, all 112 passes.

### Reference

| Item | Value |
|---|---|
| Reference | reference 2, `gpu-e2e-reference-v2/reference` (immutable; `SHA256SUMS` checked by every job); its recorded `note_free_bytes` 2301403136 gives the plan: 2 x 4 spatial tiles x 14 temporal = 112 passes, 832x480 |
| Replay | VAE-only replay of reference 2's own latent with the free memory forced to that figure; it reproduces reference 2's frames and decoder input bit for bit (`replay-self`, every job) |
| Upstream | kandinskylab/kandinsky-6 `eac06e39b904940baff2c408e925cdb0c4529269`, `pro-distill`; checkpoint HF `Kandinsky-6.0-Pro-distill-5s-Diffusers` `1018d582` |
| Stack | Python 3.13.16, torch 2.14.1+cu130, numpy 2.5.4: the `k6-oracle-full-v2` prefix (archive `106e608b…bc85`, freeze `18f5e02f…4451`) |

### Native run

| Item | Value |
|---|---|
| Hardware | NVIDIA GB10 (DGX Spark), compute capability 12.1 |
| Binary | `localrouter` sha256 `3014cb24f9252b33650cfb498b25294d4a8a22e25825b37487fc210fa1183eb0` (check v82 and hardware v78 passed on that tree) |
| Build | `video_vae/conv.cu` and `ops.cu`: `-O3`, LocalRouter's torch flags, sm_121a / sm_120a |
| Command | `localrouter check k6-components --stages vae --oracle gpu-e2e-reference-v2/reference --vae-trace pass-1` with the policies below as options (the trial form); the production decode now runs them unconditionally |
| Production path | `k6-vae-prod-v1`: the default decode with no VAE options, frames [1, 3, 121, 480, 832] bit-equal (`K6_VAE_PROD_EXACT`); `k6-full-e2e-v1`: text, DiT steps 0-9, VAE and audio, 17 pairs, 0 failed, 0 unequal values, audio chain qualified (`K6_FULL_E2E_EXACT`). Binary `6f0812066f4f976179659a4cc97c0243d98b5efbb6d53898e89421b6abf55d4b` (check v84, hardware v80). Evidence: checks/vae-prod-v1 |

### Qualified policies

`video_vae/launch.zig` `qualified` holds them; the decode applies them on every run. Each is a candidate exact on every
fixture of its op (a tie on one input proves nothing), then the whole decode bit-equal.

| Op | Policy | Evidence (checks/…) |
|---|---|---|
| Plan | `vae_free_bytes` 2301403136 (the default) | the reference's recorded free memory; trace v1's replay |
| GroupNorm statistics | `torch_welford` V7: torch's RowwiseMomentsCUDAKernel Welford, nvcc's contractions | `vae-gn-v1` (V1 / V3 / V5 / V7 exact on the first-tile fixture), `vae-blend-v1` gn probe: 8 shapes / distributions, V1 / V3 / V5 / V7 exact on every case the decoder reaches |
| Convolutions | 13 shape rules (`qualified.conv`): channel tile 16 / 64 / 128 outer, taps inner; split-K 2 by tile for the 512-channel mid / up0 / up1 convs; conv_out tap-major with `kslot` quad32 | `vae-conv-v1`..`v2` (fixture, every candidate vs cuDNN), `vae-attn-v1` (conv_out's order measured on the device: 216 k16 steps, one chain), `vae-sdpa-v1` (all 13 production-exact over whole outputs, `K6_VAE_CONV_ALL_EXACT`) |
| Linears (attention q, k, v, out) | k16 ascending; q / k / v bias separate (`half(half(acc) + bias)`), to_out fused | `vae-attn-v1` (fixture: 4 / 4 production-exact on torch's y) |
| Attention core | `k6v_sdpa_memeff` variant 524: torch's mem-efficient kernel, m16n8k8 GEMMs, fused middle combine, last block `fma(a, O, b·A)` | `vae-sdpa-v1` (backend: EFFICIENT_ATTENTION, flash / cuDNN no kernel), `vae-sdpa-v2` (P·V steps measured: k8 32,512 / 32,512; variants exact on the call and every probe: 524 525 556 557 588 589, equivalent for a 0 / −inf mask) |
| Blends, residuals, SiLU, upsample, pad, decode input / output | as RECEIPT.md (torch's ops, no choice) | `vae-blend-v1` blend fixture; the traces |

Not qualified: any other plan (its tile shapes have no conv rule and fall back to tap-major), the encoder and I2V.

### The one supported resolution: 832x480 landscape

Kandinsky-6 serves 832x480 landscape only, because that is the frame the decode is proven bit-exact on. The DiT still
generates its native 864x480 (60 x 108 latent). The qualified plan's tiles stop 32 columns before that edge, as
upstream's do for that plan, so the decoded and written frame is 832x480. Other sizes would need other tile plans, so
other tile shapes, other cuDNN choices and their own qualification.
- **Requests:** the tool's `check` refuses any other width, height or orientation before the request is queued. The
  refusal names 832x480. A request that names no size gets 832x480 (`engine.Entry.default_size`, used by HTTP, MCP and
  `gen`).
- **Config:** production options refuse any VAE plan but the qualified one (`vae_free_bytes` / `vae_free_mb`), so the
  written size cannot drift. The diagnostic gates may still replay other plans.
