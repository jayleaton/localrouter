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
