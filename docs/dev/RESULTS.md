# Results

Measured numbers per milestone. Every row names the GPU, the size, warm or cold, and the commit. Raw logs are in
`results/` (not committed); hardware runs are recorded in local development logs.

## M2: the Qwen-Image 2.1 twin (2026-10-07, a rented pod, 1x RTX PRO 6000, 42 min, $1.46)

Twin at 12fd79c + fixes (`tools/twin`): diffusers 0.41.0 (text encoder, prompt template, sigmas, VAE),
transformers 5.19.0, torch 2.13.0a0 (NGC 26.07, CUDA 13.3), Triton 3.7.1, TensorFold v0.6.1, tfimage f918455.
Weights: `Qwen/Qwen-Image-2.1` (31 GB) on the volume at `models/`, copied to local NVMe per pod in 14 s (the volume
reads at about 20 MB/s through mmap page faults, so models are never loaded from it directly).

### bf16 gate: the twin against diffusers

Same prompt, the same noise, 1024x1024. Relative L2 of the velocity:

| sigma | twin vs diffusers | diffusers vs itself, cuDNN vs memory-efficient attention (floor) | ratio |
| ---: | ---: | ---: | ---: |
| 1.0 | 0.0094 (cos 0.99996) | 0.0075 | 1.25 |
| 0.6 | 0.0072 (cos 0.99997) | 0.0056 | 1.27 |
| 0.2 | 0.0143 (cos 0.99990) | 0.0112 | 1.28 |

- **Gate:** twin within 1.5x of the attention-backend floor at every step. **Pass.**
- **Whole image (25 steps):** 27.9 dB PSNR against the diffusers pipeline: the same image, differing in fine detail
  (`docs/assets/m2-gate-pair.jpg`).
- The twin differs from diffusers only where it should:
  - Triton fused norms and RoPE;
  - cuDNN attention;
  - the prefix kept per prompt.

### Precision and speed (warm, same prompt, 25 steps, one PRO 6000)

| Precision | 1024x1024 sampling | per step | 608x480 sampling | peak memory |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 (TensorFold prompt GEMMs, static scales) | **2.37 s** | 94 ms | 0.63 s | 27.5 GiB |
| FP8 (cuBLASLt `_scaled_mm`, per-call scales) | 4.24 s | 169 ms | 1.02 s | 30.4 GiB |
| bf16 | 4.72 s | | | |

- **Other stages:** text encode 0.02 s and VAE decode 0.19 s at 1024x1024. A 1 MP image end to end is about 2.6 s
  with NVFP4.
- **Peak memory:** it includes the bf16 text encoder (16 GB) and the VAE, all resident.
- **FP8 is barely faster than bf16 here.** The quantization around `_scaled_mm` is unfused and runs per call. M3 measures TensorFold's own FP8 prompt GEMM
  (`prompt(A8)`, static scales like NVFP4) before the FP8 path is fixed.
- **Renders:** `docs/assets/m2-renders.jpg` shows the six gate prompts, rows bf16 / FP8 / NVFP4.
  - FP8 tracks bf16.
  - NVFP4 keeps the subject and changes layout on some prompts (sign framing, cup position), as the original engine
    documented.

### Packs and captures (on the volume)

- **Packs:**
  - `packs/qwen-image-2.1-nvfp4`: 585 tensors. Each linear has codes, scales, a global scale and a calibrated
    input scale (128 linears, 8 calibration prompts).
  - `packs/qwen-image-2.1-fp8`: 329 tensors.
  - Manifests (sha256 per tensor) are in the local image-conversion results directory.
- **Captures:** step 0 of a 512x512, 4-step run, plus the prefix build. Each holds every GPU op with its inputs and
  outputs:
  - `captures/m2-nvfp4`: 727 ops, 4.1 GB;
  - `captures/m2-fp8`: 918 ops, 6.5 GB.
  - Programs and cubins (no blobs) are in `results/captures/`.

| Op kind (one step + prefix) | count | how the Zig engine gets it |
| --- | ---: | --- |
| `adaln`, `rms_rope`, `swiglu` | 129, 128, 64 (FP8 only) | Triton cubins from the capture (3 kernels) |
| `linear` NVFP4, `nvfp4_mlp` | 128, 64 | TensorFold v0.6.1 `quant4`, `gemm_ws`, `gemm_gu_ck` (`.cu` copied) |
| `linear` FP8 | 256 | cuBLASLt today; TensorFold `prompt(A8)` if M3's measurement favours it |
| `attention` | 64 | LocalRouter's kernel (from TensorFold's `prefill_attention.cu`), which the twin then adopts |
| `dense_bf16` (time, modulation, text, in / out) | 8 | small GEMMs: decided in M3 (cuBLASLt with the twin's algorithm, or the twin moves to ours) |
| `gated_residual`, `copy`, `silu`, `tanh`, `gelu_tanh`, `rms_norm_f32`, `time_sinusoid`, `euler` | 128, 64, 3, 2, 1, 1, 1, 1 | small CUDA kernels written to torch's exact arithmetic, bit-gated against the capture |

## M3a/M3b: LocalRouter's own kernels in the twin (2026-10-07, a rented pod, 1x PRO 6000, 30 min, $1.05)

The twin now runs every small op on LocalRouter's kernels (`kernels/cuda/qwen_image/ops.cu`) and attention on the
toolkit's kernel (`attention.cu`, TensorFold's prefill attention with a key count and a causal flag). The Zig engine
launches these same sources.

- **Ops against torch** (`stk_twin.test_ops`, 146 checks, all pass, all deterministic):
  - Bit-identical to torch: SiLU, tanh, GELU-tanh, the gated residual, the time sinusoid, the Euler step, the row
    copies and the FP8 quantization.
  - Ours by design, within 1 bf16 ulp of torch:
    - the dense bf16 GEMM: fixed blocked fp32 order, row-invariant;
    - the text RMSNorm: fixed tree order.
- **Attention against cuDNN** (`attncheck`): relative L2 1.2e-4 at 1 MP, the same distance to an fp32 reference as
  cuDNN's own (0.00227), deterministic.
  - Speed: 1.36 ms a call against cuDNN's 0.87 ms (32 calls a step: +16 ms at 1 MP). **Open: the attention kernel must
    reach cuDNN's speed** (the M3 speed gate).
- **bf16 gate with every toolkit kernel adopted:** relative L2 to diffusers 0.0081 / 0.0071 / 0.0133 at sigma
  1.0 / 0.6 / 0.2, that is 1.08x / 1.27x / 1.19x the attention-backend floor. Pass.
  - A first run with a single sequential fp32 sum in the dense kernel reached 1.56x at sigma 1.0.
  - Summing each 32-wide K tile and then the tile sums (still deterministic) brought it to 1.08x.
- **Step GPU time at 1 MP, NVFP4** (torch profiler, this pod):
  - torch/cuDNN path 98 ms;
  - toolkit kernels 117 ms: attention 43.9 ms, NVFP4 GEMMs 58 ms, RoPE 4.8 ms, gated residual 4.1 ms, the rest
    under 5 ms.
- **Wall-clock benches on this pod are CPU-bound** (about 240 ms a step for both paths, the GPU unthrottled): the twin's
  Python launch loop outran this host's CPU. Twin timings compare only within one pod, and kernel time is measured on
  the GPU.
- **FP8 on TensorFold's prompt GEMM (A8, static scales from calibration, margin 2x)** replaces cuBLASLt `_scaled_mm`.
  - One kernel family for both precisions; no cuBLASLt in the runtime image.
  - Its GPU-time comparison with the cuBLASLt path is pending (the wall-clock numbers above are CPU-bound).
- **Captures for the Zig replay** (on the volume; programs and cubins in `results/captures/`), each with Triton's
  signature and full metadata a launch:
  - `captures/m3-nvfp4`: 727 ops;
  - `captures/m3-fp8s`: 855 ops.

## M3: the Qwen-Image DiT in Zig (2026-10-07, a rented pod, 1x RTX PRO 4500 Blackwell, sm_120, 18 min, $0.22)

`localrouter check qwen-replay PACK CAPTURE` runs step 0 (the text prefix through 32 blocks, then the 32-block step and the
Euler update) through the Zig forward. It matches every op to the twin's capture by name and compares every output
byte for byte:
- **alone:** each op from the captured inputs;
- **chained:** from Zig's own outputs, starting from Zig's own noise.

| Pack | Alone | Chained | Result |
| --- | --- | --- | --- |
| NVFP4 (`m3-nvfp4`, 512x512) | 723 / 723 outputs identical | 723 / 723 identical | **pass** |
| FP8 on TensorFold A8 (`m3-fp8s`) | 851 / 851 | 851 / 851 | **pass** |

- **Hardware:** the captures were made on an RTX PRO 6000 and replayed on an RTX PRO 4500. Same architecture, other
  part; the bits hold.
- **Negative control:** the same pack against the M2 capture (cuDNN attention, torch ops, libm RoPE) differs from the
  first text op on, as it must.
- **Kernel sources:**
  - `python3 tools/kernels/sync.py`: the 8 TensorFold v0.6.1 copies equal their sources at the tag.
  - SASS: all 29 NVFP4 kernels of ours equal to TensorFold's Python build on sm_120 (on the pod) and sm_121 (offline),
    fresh compiles and the shipped fatbins alike.
- **Host-side tables and repacks**, bit-equal to the twin and checked offline:
  - RoPE tables (93,312 angles);
  - noise (300,000 normals);
  - `log` (400,000 values);
  - the FP8 fragment order and NVFP4 scale layout (against TensorFold's own Python).

| Step time, same GPU (RTX PRO 4500) | Zig (GPU events, host gaps included) | twin (kernel time only) |
| --- | ---: | ---: |
| NVFP4, 1024x1024 | 233.9 ms | 227.4 ms |
| NVFP4, 512x512 | 45.6 ms | 44.0 ms |
| FP8, 1024x1024 | 317.8 ms | 310.0 ms |
| FP8, 512x512 | 65.9 ms | 62.3 ms |

- **Kernels:** the same in both, by construction. Zig's few percent over the twin's pure kernel time is host time
  inside the step: per-step RoPE upload with a sync, and per-block prefix copies. Both go when the tables are cached
  per size and the step is captured as a CUDA graph.
- **Wall clock:** the twin is CPU-bound on pods (M3b), so the Zig engine is faster end to end.
- **Load times:** weights 1.0 s (NVFP4, 4.2 GB) and 5.5 s (FP8, 7.3 GB) from local NVMe, including the GPU repack.
- **Open item: attention speed.** LocalRouter's attention is 1.36 ms a call against cuDNN's 0.87 ms at 1 MP on a
  PRO 6000. It is about 38 % of an NVFP4 step, so it is the first speed task: a faster kernel lands behind the same
  launch and becomes the twin's reference after its numeric gate against cuDNN.

## M4: Qwen-Image end to end in Zig (2026-10-07, a rented pod, 1x RTX PRO 4500 Blackwell, sm_120, 46 min, $0.55)

Prompt to pixels in one process: TensorFold's tokenizer on the pipeline's template, the Qwen3-VL 8B text path on the
toolkit's kernels (new: `te.cu`, a deterministic tensor-core bf16 GEMM `gemm.cu`, the attention with 8 KV heads and a
causal mask), the M3 DiT, diffusers' sigma schedule in numpy's float32 order (`sampler.zig`), and the VAE decoder as
im2col + GEMM plus `vae.cu`. Three packs (`stk_twin pack --precision nvfp4|fp8s|te|vae`); tfimage's Triton cubins per SM
inside the DiT pack (`stk_twin triton`). Results are retained locally in the image-port results directory.

| Gate | Result |
| --- | --- |
| New kernels vs transformers / torch formulas (`test_ops`, 171 checks) | text encoder ops 0 ulp; GEMM within one bf16 ulp + the fp32 accumulation bound of fp64, deterministic, row and column blocking invariant |
| Twin text encoder on our kernels vs transformers (10 prompts) | cosine >= 0.99934, deterministic |
| Twin VAE on our kernels vs diffusers (1024x1024) | 38.4 dB PSNR; denorm and uint8 bit-exact with the pipeline; deterministic |
| Zig text encoder replay (`localrouter check qwen-te`) | 613/613 alone and chained; token ids and context equal |
| Zig VAE replay (`localrouter check qwen-vae`) | 157/157 alone and chained at 512x512 and 1024x1024 |
| DiT replay after the GQA attention change | 723/723 (unchanged) |
| End to end (`localrouter check qwen-e2e`): sigmas, context, final latents, pixels | byte-equal for all 6 gate prompts x {512, 1024} (NVFP4, 4 steps), NVFP4 and FP8 at 512 and 1024 from full captures, and NVFP4 and FP8 at 1024x1024 with 25 steps |
| The tool through the API (`localrouter serve`, `/v1/images/generations`, worker loads, generates, unloads) | NVFP4: one 1024x1024 25-step PNG in 17 s wall; FP8: 22 s |

Warm times, 1024x1024, 25 steps (sm_120, PRO 4500):

| | encode | prefix + sample | decode | total | load (cold, local NVMe) |
| --- | --- | --- | --- | --- | --- |
| Zig NVFP4 | 39 ms | 6.05 s | 578 ms | 6.67 s | 8.0 s |
| twin NVFP4 | 47 ms | 11.07 s | 575 ms | 11.69 s | |
| Zig FP8 | 39 ms | 8.16 s | 579 ms | 8.78 s | 12.5 s |
| twin FP8 | 52 ms | 13.13 s | 576 ms | 13.76 s | |

The VAE decode on our kernels is 568 ms against diffusers' cuDNN convolutions at 353 ms: the next speed item after
attention (implicit GEMM convolutions, no im2col round trip). Full captures at 1024x1024 with the VAE recorded are
34 GB; the 6-prompt gate uses light captures (request, sigmas, context, latents, pixels).

![M4: Zig renders, NVFP4 and FP8 (25 steps, prompt 0), and the tool's API output for prompt 1, NVFP4 and FP8](../assets/m4-zig-renders.jpg)

### Attention speed, first measurement (same pod)

`pattn2_kernel` keeps the shipping kernel's arithmetic and makes the staging and masking parameters; `attnbench`
requires bitwise equality with the shipping kernel at every shape. All 10 variants are bit-equal. Median CUDA-event
times, ms:

| shape | shipping | best bit-equal (4 warps, two 64-key slots, 3 blocks/SM, unmasked full tiles) | cuDNN |
| --- | --- | --- | --- |
| DiT step 1024x1024 (4096 x 4176 keys, 32 heads) | 2.635 | 2.229 (1.18x) | 1.705 |
| DiT step 512x512 | 0.235 | 0.157 (1.50x) | 0.149 |
| prefix (80, causal) | 0.016 | 0.014 | 0.021 |
| text encoder (80, causal, 8 KV heads) | 0.017 | 0.014 | |

Bit-preserving tuning closes about half the gap at 1 MP; the rest needs the arithmetic itself to change (a new
declared reference, re-captured).

## Speed: attention and step graphs (2026-10-07, a rented pod, 1x RTX PRO 4500 Blackwell, 20 min, $0.24)

Attention variants at the DiT's 1 MP step (4096 queries x 4176 keys, 32 heads), median CUDA-event ms; error is the
relative L2 against an fp32 reference (`stk_twin.attnbench`, local attention-benchmark results):

| kernel | arithmetic | ms | vs pattn_kernel's bits | rel. error |
| --- | --- | --- | --- | --- |
| pattn_kernel<128, 8, 1, 8> (was shipping) | TensorFold's | 2.634 | | 2.27e-3 |
| **pattn2_kernel<128, 4, 1, 2, 64, true, 3> (shipping now)** | TensorFold's | **2.221** | equal | 2.27e-3 |
| pattn3_kernel, best (FlashAttention-2 exponent and row sums) | new | 2.116 | differ | 2.27e-3 |
| pattn4_kernel, 32 rows a warp | TensorFold's | 3.5-4.9 | equal | 2.27e-3 |
| cuDNN (SDPA) | | 1.703 | | 2.27e-3 |

What decided it: the kernel is latency bound, not arithmetic or shared-memory bound. FlashAttention-2's arithmetic buys
5 % more and would need a new reference; two 16-row tiles a warp (each K/V fragment feeding two MMAs) keeps the bits but
needs 254 registers and 64-96 KB of shared memory, leaving 4-8 warps a multiprocessor, and is 1.3-2x slower. The
shipping variant (4 warps, 32 KB, three blocks a multiprocessor, no per-score masking on full tiles) is bit-equal, so
every M3/M4 capture still replays: verified on the new engine against captures made with the old kernel (DiT 723/723,
text encoder 613/613, pixels equal) and the twin's new captures against Zig (pixels equal). The remaining 1.3x to cuDNN
is about 7 % of an image; GB10's own ranking comes from `attnbench` in the Spark window.

Step graphs (each DiT step's ~700 launches captured once per size and replayed; RoPE tables kept per size): pixels
equal with and without; at 1 MP on this GPU the host was already hidden behind the GPU (5.99 s vs 6.00 s for 25
steps). They stay for smaller images and the Spark's host.

| 1024x1024, 25 steps, NVFP4, warm | encode | prefix + sample | decode | total |
| --- | --- | --- | --- | --- |
| Zig, M4 | 39 ms | 6.05 s | 578 ms | 6.67 s |
| Zig, now | 39 ms | 5.51 s | 577 ms | 6.13 s |
| twin | 47 ms | 10.70 s | 573 ms | 11.32 s |

## M5: GB10 (2026-10-07, a DGX Spark sm_121, 07:39-09:17 UTC, `docs/dev/RUNBOOK-M5.md` run by the owner's operator)

**Bits: 51 of 51 pass on sm_121.** Packs made on the Spark from the official checkpoint are byte-equal to the pods'
(NVFP4 585 tensors, FP8 457); replays alone and chained: DiT 723/723, text encoder 613/613, VAE 157/157; the 29 NVFP4
symbols match the fatbin's SASS; prompt to pixels (6 gate prompts at 512 and 1024 with 4 steps, 1024 and 576 with 25
steps, both precisions): pixels byte-equal to the twin in all 16 runs, warm and cold, with and without step graphs.

| GB10, warm, 25 steps | encode | prefix + sample | decode | total | twin's sample |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVFP4 1024x1024 | 102 ms | 11.35 s | 1.26 s | **12.7 s** | 14.70 s |
| FP8 1024x1024 | 104 ms | 15.40 s | 1.23 s | 16.7 s | 18.87 s |
| NVFP4 576x576 | 99 ms | 2.98 s | 388 ms | **3.47 s** | 4.24 s |
| FP8 576x576 | 102 ms | 4.55 s | 390 ms | 5.04 s | 5.71 s |

4 steps: 1024x1024 3.13 s, 512x512 0.80 s (NVFP4). Engine load 15-17 s. Step graphs change nothing at 1 MP (11.29 s
without) and nothing measurable at 0.3 MP: GB10's host keeps up too. The VAE decode is 1.23 s at 1 MP (0.58 s on the
PRO 4500), about 10 % of an image: the next speed target on this machine.

**Through the API** (Docker image built natively on DGX OS, arm64; page cache dropped before the cold request):
cold 1024x1024 NVFP4 33.4 s (load included), warm 12.6 s, warm 576x576 3.5 s, FP8 1024x1024 with its tool load 39.6 s.
Self-test in the image: image 16 ms, video 306 ms.

**Memory** (MemAvailable each second; docker stats misses GB10's GPU allocations): 38.1 GiB at the window's peak, in
the text encoder's packing (Python); the engine itself 25-30 GiB NVFP4 (512 to 1024) and 28-32 GiB FP8. The runbook had
guessed about 70 GB. The image tool's resident estimates are now 32,000 MiB NVFP4 and 36,000 MiB FP8 (were 48,000).

**Attention on GB10** (`attnbench`, the DiT's 1 MP shape): the shipping `pattn2_kernel<128, 4, 1, 2, 64, true, 3>`
(w4s64n2b3) is still the fastest bit-equal variant, 3.72 ms against 5.31 ms for the reference kernel (1.43x; 1.51x at
0.25 MP) and 3.38 ms for cuDNN (0.91x cuDNN's speed). FlashAttention-2 arithmetic would reach 3.44 ms but changes the
bits. Nothing to adopt: GB10's ranking matches the PRO 4500's.

**Fixed after the window** (each worked around by hand on the day): the worker exited at once when `localrouter serve` was a
container's PID 1 (its orphan check compared the parent with 1; it now compares with the daemon's pid, verified with
the daemon as PID 1 of a pid namespace); packs were written 0600 by root and unreadable to the image's uid 10001 (pack
writers now set 0755/0644); `m5.sh` sourced its venv before `setup.sh` made it; `m5-host.sh` now samples MemAvailable.

## M8: the one-command install on a Spark (2026-10-07, a Spark, a 39 min window)

`bash tools/spark/install.sh` on a fresh install directory: 25 min end to end, rc 0. The checkpoint download took 4 min. The
packs (NVFP4, FP8, text encoder, VAE: 26 GB) took 17 min, every one byte-equal to the reference digests. The native
image build and service start took 3 min, then the MCP check (tools/list, one 512x512 image). A re-run reuses
everything: 30 s, rc 0.

From the dev box over the network, through MCP (`http://<spark>:8190/mcp`, the official SDK
client and plain JSON-RPC):

| request (1024x1024 unless noted, 25 steps) | wall | engine | MemAvailable after |
| --- | ---: | ---: | ---: |
| NVFP4, cold (after `release_models`) | 31.1 s | 12.4 s | 81.3 GiB |
| NVFP4, warm | 12.4 s | 12.4 s | 81.4 GiB |
| NVFP4 576x576, warm | 3.5 s | 3.5 s | 81.4 GiB |
| FP8, cold (NVFP4 stays loaded) | 34.1 s | 16.7 s | 46.2 GiB |

Unloading: `release_models` gives all of it back (117.1 GiB available); left idle, the model unloads at its 300 s
TTL by itself (83.1 GiB at +274 s, 117.1 GiB at +304 s). The returned PNG matched its prompt. After the run, the
stack was taken down and the co-tenant service restored through its own window script (verified: it answered a test
prompt, watchdogs active). Fixed from the run: `list_models` named the container instead of the host; a re-run now always recreates
the container so an edited `tools.json` is read.
## Load speed: weights at the drive's speed (2026-10-07, a Spark, three short windows)

A cold image (the model released first, 1024x1024, 25 steps, through the API; the PNGs byte-equal throughout):

| loader | NVFP4 cold | load | FP8 cold | warm |
| --- | ---: | ---: | ---: | ---: |
| per tensor: pageable read, cuMemAlloc, synchronous copy | 30.8-34.3 s | ~18.5 s | 39.3-39.9 s | 12.4 s |
| pinned ring, async copies, slabs, no per-linear sync | 29.8-30.8 s | 17.3 s | 40.5 s | 12.5 s |
| the same with direct I/O (O_DIRECT) | **16.1-16.2 s** | **3.6 s** | **21.7 s** | 12.4 s |

The load report (now in every worker log, and in `localrouter check qwen-e2e`) is what found it: after the first change the
copies waited 2 ms in all and `pread` took 15.3 of 17.3 s. On GB10 reading through the page cache runs at 0.95 GB/s;
the drive gives 12 GB/s to direct reads (one stream, 64 MB). With direct I/O the 18.5 GB read in 1.6 s; the rest of
the 3.6 s is the context (0.25 s), NVFP4 repacking, and the VAE's scratch (0.6 s). A cold image is now the load plus
the generation (12.4 s).

## M7 speed round: MiniMax H3 on GB10 (2026-10-08, a Spark, one window)

Everything below is bit-identical: the old and new paths agree byte for byte, and Zig agrees with the twin.

**Video VAE decode.** GEMM tiles of 256 x 128 with a 3-stage pipeline (the m tile fastest), attention two heads at a
time. Every GEMM shape, and the frames old against new and Zig against twin, are equal. The GEMM sum is 1.94x faster.

| | 56 frames | 124 frames |
| --- | ---: | ---: |
| decode before | 13.85 s | 32.2 s |
| decode now | **7.14 s** | **16.7 s** |

**DiT step** (`localrouter check h3-step-bench`, 768x448, GPU events, median). The replay of a fresh capture is 639/639
equal, alone and chained, and every fast configuration's velocities equal the reference's.

| | 56 frames (5,967 tokens) | 124 frames (12,915 tokens) |
| --- | ---: | ---: |
| reference schedule | 1,739 ms | 4,621 ms |
| fused gate_add + norm_mod, attention as rows | **1,674 ms** (1.04x) | **4,472 ms** (1.03x) |
| fused + CUDA graph | 1,680 ms | 4,476 ms |

The step is GPU-bound: under 1 ms between ops, so the graph saves nothing. Of the fused step at 56 frames, the GEMMs
are 776 ms, the attention kernel 349 ms, and the memory-bound elementwise ops about 550 ms (attention prep 132, gate +
norm 125, quantize 105, SwiGLU 104, RMSNorm + RoPE 83). At 124 frames attention is 1,580 of 4,472 ms. Next: fold
quantize into its producers, and rope into attention prep; then the attention kernel for long clips.

**Full clips** (`localrouter check h3-bench`, engine only, text encoder resident, two runs equal):

| | 56 frames (2.3 s) | 124 frames (5.2 s) |
| --- | ---: | ---: |
| encode / sample / audio / video | 0.91 / 13.4 / 0.10 / 7.2 s | 0.92 / 35.8 / 0.20 / 17.0 s |
| warm total | **21.7 s** (was 29 s) | **53.9 s** |
| load (cold) | 10.5 s | 20.2 s |
| peak memory (MemAvailable drop) | 36.7 GiB | 38.8 GiB |

Unloading the text encoder between requests lowers the peak only to 36.0 / 37.9 GiB (the peak is the decode) and costs
13-14 s a request, so it stays resident; the tool's estimate (40,000 MiB plus the frames) covers the measured peak.
Over MCP from another machine, a 5 s clip took 71 s from cold.

## Qwen-Image 2.1 Turbo on GB10 (2026-10-10, a Spark, one window, `tools/spark/turbo-window.sh`)

Checkpoints `Qwen/Qwen-Image-2.1` d26bb61 and `Qwen/Qwen-Image-2.1-Turbo` d65dbc9, NGC PyTorch 26.07, driver
580.178.04. Turbo's transformer has base's architecture with its own weights; its text encoder and VAE are base's. Its
FP8 pack uses its own calibration (`tools/twin/packs/acts-fp8-turbo.json`, digests in `digests-turbo-fp8s.json`) and
carries the checkpoint's 8 `sample_sigmas` (shift 1, no dynamic shifting).

**Zig against the twin** (`localrouter check qwen-e2e`, FP8): sigmas, text context, final latents and decoded pixels
equal, warm and without step graphs, in all 5 cases.

| | steps | size | prompts | warm sampling (twin) |
| --- | ---: | --- | --- | ---: |
| Qwen-Image 2.1 (regression) | 25 | 1024x1024 | 0 | 15.34 s (19.05 s) |
| Turbo | 8 | 512x512 | 0, 1 | 1.07 / 1.06 s (1.42 s) |
| Turbo | 8 | 1024x1024 | 0, 1 | 4.91 / 4.91 s (6.18 s) |

**bf16 twin against diffusers** (1024x1024, same inputs; floor: diffusers cuDNN against memory-efficient attention, on
this GB10). `stk_twin gate` requires cos >= 0.9999 at every step: it **fails** at sigma 0.2 for Turbo, and for
Qwen-Image 2.1 on this GB10 too (the M2 pass above was on a PRO 6000). Against the floor (the M2 criterion, 1.5x):

| sigma | Turbo: cos, rel L2 | Turbo floor | ratio | 2.1: cos, rel L2 | 2.1 floor | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.99997, 0.0074 | 0.0062 | 1.18 | 0.99997, 0.0077 | 0.0074 | 1.04 |
| 0.6 | 0.99998, 0.0056 | 0.0053 | 1.07 | 0.99998, 0.0071 | 0.0058 | 1.23 |
| 0.2 | 0.99980, 0.0201 | 0.0149 (cos 0.99989) | 1.36 | 0.99988, 0.0155 | 0.0115 | 1.34 |

Whole image against the diffusers pipeline: Turbo 24.9 dB (8 steps), 2.1 26.8 dB (25 steps).

**Through the daemon** (`tools/spark/compare_models.py`, `tools/twin/prompts/compare.json`, 1024x1024, FP8, each model
with its own defaults, 3 warm repeats, every repeat byte-identical):

| | first request (cold, prompt 0) | warm median, prompts 0 / 1 / 2 | load |
| --- | ---: | ---: | ---: |
| Turbo, 8 steps | 11.3 s | 6.33 / 6.35 / 6.37 s | 4.9 s, 21.3 GiB |
| Qwen-Image 2.1, 25 steps | 21.7 s | 16.84 / 16.87 / 16.88 s | 4.9 s, 21.3 GiB |

The same seed gives both the same starting noise, not the same trajectory. Turbo set the poster's text exactly
("MIDNIGHT BLUE JAZZ", "LISBON 2027") where 2.1 broke "JAZZ"; both placed the still life's objects as asked (apple left,
mug with spoon centre, three lemons stacked right, plant behind), with no visible artifacts. In the portrait, 2.1 has
the sea spray and a deeper scene; Turbo's keeper looks younger than seventy, the spray is faint and the light flatter.
