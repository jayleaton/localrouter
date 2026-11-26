# ROCm: MiniMax H3-Image, then H3, native on gfx1151 (the plan)

LocalRouter does not depend on ComfyUI. On Strix Halo (Radeon 8060S, gfx1151, RDNA3.5, unified memory over GTT)
the MiniMax models ran through ComfyUI (#157, closed; the generic worker is `tools/comfyui` from #156). This plan
replaces that route with LocalRouter's own engines on a HIP backend, in three stacked series:

1. **H3-Image**: text to image, and edits (masked and whole scene).
2. **H3**: text or image to video with audio.
3. **Remove `tools/comfyui`**, its CI step (`.github/workflows/ci.yml:35-36`) and its mention in
   `docs/ARCHITECTURE.md:137`.

Nothing here is measured on gfx1151 yet. Every number marked *estimate* is arithmetic, not a run.

## 1. What we start from

**H3-Image is a second checkpoint of the H3 DiT, not the video model at one frame.** It also has no engine of its
own on CUDA today: `minimax_h3` declares only `text_to_video` and `image_to_video`. Its weights are published as
`Comfy-Org/MiniMax-H3-Image`. The safetensors headers were read for this plan, with no weights downloaded:

| Piece | H3-Image | H3 video (`src/engines/minimax_h3`) |
| --- | --- | --- |
| DiT | 50 blocks, dim 5376, 56 heads x 128, SwiGLU 14336, `condition_proj` 5120 -> 5376, token refiner, `adaln_t_table` [1025, 8], `adaln_proj` 3 x 6 x 5376 | the same shapes (`dit.zig`: `layers`, `dim`, `heads`, `ffn`, `text_dim`, `grid`, `ada_n`) |
| Output head | `final_layer.video_out` [576, 5376] fp32: 6 PDD heads x 96, one per step (`config.transformer.pdd_steps` 6 and `pdd_sigmas`) | [96, 5376], plus `audio_out` |
| Checkpoints | `minimax_h3_image_pruned_bf16` 40.2 GB, `_int8_convrot` 21.0 GB, `_w4a8` 12.5 GB | NVFP4 pack built from bf16 (`stk_twin.h3.build`) |
| VAE | `minimax_h3_image_vae_fp16` 1.17 GB. The encoder has the video VAE encoder's keys (`conv_in` 3 -> 128, `conv_out` -> 48, `quant_conv` 48). The decoder is a ViT with `x_embedder` 24 -> 1024, 4 register tokens, a mask token, 24 blocks of width 1024, gated FFN 8192 -> 4096 and `proj_out` 1024 -> 768 (3 x 16 x 16) | the video decoder ViT (`vae_video_tile.zig`): 36 blocks, width 2048 |
| Text encoder | Qwen3-VL 32B, the H3 encoder (ComfyUI uses its `int8_convrot` file) | `te32.zig` on `qwen3vl_32b_minimax_h3_nvfp4_awq` |
| Licence | MiniMax H3 Image Community License. Authorization is needed above USD 1 million revenue, **or for any hosted service to third parties** | MiniMax H3 Community License |

So series 1 reuses the DiT, the token refiner, the 32B text encoder, the vision tower (`te_vision.zig`, from I2V)
and the VAE encoder (`vae_video_enc.zig`). It adds a PDD sampler, an image VAE decoder (the tile ViT, parameterised
by depth and width), the edit conditioning and the HIP backend. The first task of series 1 is reading ComfyUI's
`MiniMaxH3ImageConditioning` and H3-Image model code. Those findings go into `docs/dev/H3-IMAGE.md`, written the way
`H3-I2V.md` was, before any engine code.

**What #157 measured (ComfyUI on gfx1151, ROCm 7.2.4, torch 2.14.1+rocm7.2; a reference only):**

| Job | Settings | Time | GTT peak |
| --- | --- | --- | --- |
| Text to image | Q4_K_M GGUF, 1024², euler/simple, 6 steps | 47.4 s | 29.6 GiB |
| Masked edit | Q4_K_M, about 1 MP, 6 steps; ComfyUI edits the whole frame, then `ImageCompositeMasked` pastes it back through the mask | 104-111 s | 30.7 GiB |
| Whole-scene edit | Q8_0, about 2 MP, 30 steps | 384 s | 37.2 GiB |

A masked edit is therefore a whole-frame edit plus a pixel composite on the host, not latent inpainting. The native
engine does the same, so the composite runs on the CPU and can be checked bit for bit there.

**What #155 (smfworks, in flight, not touched here) adds, and what this plan builds on:**
`deploy/profiles/strix-halo-gfx1151.json` (preflight `rocminfo | grep gfx1151`, `min_gtt_gib` 90, `min_rocm` 7.2),
`deploy/strix-halo/` (env, config render, systemd-run start/stop/status), the ROCm Whisper lock
(`tools/speech/whisper/rocm/`, torch 2.13.0 rocm7.2), the Gufo container pin and the hash tools. Its memory budget is
computed in shell (`render.sh`: `BUDGET_BYTES = hw_gtt - GTT_MARGIN_BYTES`). GPU, ROCm and GTT detection and the
doctor's AMD hints are already on main from #153 (`tools/install/lib.sh` `hw_gtt`, `hw_gtt_used`, with fixtures in
`tests/fixtures/amd`). This plan adds no second detector or profile. The engine's tool entry joins #155's config
template once #155 lands (PR 1.8), and the daemon learns the GTT ceiling itself (PR 1.2), so the shell's budget
arithmetic becomes an override, not the only guard.

## 2. The HIP backend in the build and the engine interface

### Build

| Option | Meaning |
| --- | --- |
| `-Dgpu=cuda` (default on Linux) / `-Dgpu=hip` | Which runtime and kernel images the GPU engines get. One backend per binary, so no runtime dispatch between vendors. |
| `-Dhip=<clang++ or hipcc>` | Builds the HIP code objects, like `-Dnvcc`. Without it the images are empty slices: host build and `zig build test` stay GPU-free. |
| `-Dgfx=gfx1151` | Offload targets, comma separated (as `-Dsm`). |

`build/hip.zig` mirrors `build/cuda.zig` and `build/minimax.zig`. Each kernel source is compiled as
`clang++ -x hip --offload-arch=gfx1151 --offload-device-only -O3 <flags> -o <name>.hsaco` and embedded with
`addAnonymousImport`. The flags are a fixed list per source, written next to the source like `kitchen_flags`, and
they are exactly the ones the twin's torch-ROCm extension passes:

- `-ffp-contract=on`, matching nvcc's default `-fmad=true`, or `-ffp-contract=off` where the CUDA source says
  `--fmad=false`.
- No `-ffast-math`.
- `-D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1`, torch's half and bf16 guards.

`kernels/hip/isa_check.sh` is the analogue of `sass_check.sh`. It runs offline with no GPU. It builds each kernel the
twin's way and the Zig way, then compares the `llvm-objdump -d` text of every kernel. Equal ISA is the evidence that
the twin and the engine run the same code.

### Runtime

`src/gpu/hip/` dlopens `libamdhip64.so` (ROCm 7.2 is #155's floor). It exposes the subset of TensorFold's `cuda`
module that the H3 engines use: `Driver.open`, `Context`, `Module.load`/`function`, `Function.allowDynamicShared`
(`hipFuncSetAttribute`), `Stream`, `Event`, `DeviceBuffer`/`HostBuffer` (`hipMalloc`, `hipHostMalloc`), `launch`
(`hipModuleLaunchKernel` takes the same `kernelParams` array, so `launch.Args` is unchanged) and `graph`
(`hipStreamBeginCapture`, `EndCapture`, `GraphInstantiate`, `GraphLaunch`).

The engines import `gpu`, not `cuda`. `build.zig` binds `gpu` to TensorFold's runtime or to `src/gpu/hip`. This is a
rename in `minimax_h3` and in the `qwen_image` files it shares (`pack`, `upload`, `weights`, `ops_launch`, `dit`
Io/Probe, `noise`, `smath`, `memcap`). It is a refactor, not a compatibility layer: one stack, no shim. A host test
checks that both modules declare every name the engines use, so the HIP side cannot drift.

### Dispatch in `launch.zig`

`launch.zig` already takes a module image and names its kernels. On HIP:

- `extern "C"` kernels (`h3_*`, `te32_*`, the VAE ops) keep their names, so `Ops.load` is unchanged and only receives
  the HIP image.
- Mangled template names (comfy-kitchen's) differ by the bf16 type's name. Each backend gets its own symbol table, as
  `nvfp4.zig` lists instantiations.
- Where the algorithm differs (attention and the block linears, section 3), `launch.zig` gets one function per role,
  `attention(...)` and `linear(...)`, chosen by `gpu.backend` at comptime. `dit.zig` calls the role and keeps its op
  names and probes, so `localrouter check` replays the same capture format on both vendors.

## 3. Kernels H3-Image needs, and their HIP form on RDNA3.5

**gfx1151 facts that decide the port:**

- HIP on gfx11 compiles wave32 by default, so `warpSize` is 32 as on CUDA. Warp-shuffle reductions keep their shape;
  HIP's `__shfl_*_sync` take a 64-bit mask.
- WMMA exists, with `v_wmma_f32_16x16x16_{f16,bf16}` and `v_wmma_i32_16x16x16_{iu8,iu4}`. On RDNA3 each half-wave
  must hold a duplicate of the A and B fragments.
- RDNA3.5 has **no FP8 and no FP4 matrix instructions** (FP8 WMMA arrived with gfx12 / RDNA4).
- It has no `cp.async` and no `ldmatrix`. Tiles are staged through registers into LDS with explicit double buffering.

| Kernel (CUDA source) | H3-Image use | HIP form |
| --- | --- | --- |
| `minimax/ops.cu` (norm_mod, gate_add(+norm_mod), swiglu, rms, final_mod, patchify, euler, `h3_gemm_f32`, ...) | every DiT step, PDD heads, patch projection | One source for both vendors through `kernels/common/gpu.cuh` (bf16 type, intrinsics, shuffles). No tensor cores, written-out roundings. CUDA SASS must not move (`sass_check.sh`). |
| `minimax/kitchen/rms_rope.cu` | per-head RMSNorm + RoPE | Ported copy under `kernels/hip/minimax/kitchen/`. It is built with `--use_fast_math` on CUDA. The HIP copy writes the same math out explicitly, so its bits are its own. UPSTREAM.md entry: comfy-kitchen has no ROCm backend; offer the port upstream as a general HIP backend, or record why not. |
| `minimax/kitchen` INT8 attention (`qk_int_sv_i8`, quant_qk, quant_v) | DiT attention | **Not ported for H3-Image.** At 1 MP the sequence is about 1.3 k tokens (1,024 image tokens: VAE /16, patch 2, plus text). Attention is about 2 TFLOP a forward against about 53 TFLOP of linears (*estimate*), so INT8 attention buys nothing here. H3-Image on HIP uses the bf16 attention below. Series 2 revisits it at about 37 k tokens. |
| `qwen_image/attention.cu` (TensorFold's bf16 prefill attention, mma.sync m16n8k16, D 128) | token refiner; on HIP, also the DiT | New HIP kernel `kernels/hip/attention.hip`: bf16 WMMA for QKᵀ and PV with fp32 accumulation, fp32 online softmax, a fixed key order and no split-K. Ours, not CK or AOTriton, so its order is written down and the twin runs the same code. |
| `qwen_image/gemm.cu` (deterministic bf16 GEMM, mma.sync, cp.async, ldmatrix) | `condition_proj`, refiner, and on HIP **every block linear** of a bf16 pack | New HIP kernel `kernels/hip/gemm_bf16.hip` with the same contract: bf16 in, fp32 accumulation over K tiles in ascending order, no split-K, so a row's bits depend on neither M, N nor the grid. It uses `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` directly. rocWMMA is used only as a cross-check in tests: a pinned header dependency for one fragment wrapper is not worth it, and the builtin keeps the order visible. hipBLASLt is ruled out: it is not ours, and its algorithm choice is not bit-stable. |
| `minimax/gemm_f16.cu` (fp16 mma.sync) | image VAE decoder linears and attention | `kernels/hip/gemm_f16.hip`, the f16 twin of the bf16 GEMM above |
| `minimax/vae_video.cu`, `vae_enc.cu` | image VAE decoder ops; encoder (edits) | single source via `gpu.cuh` |
| `minimax/te32.cu` (fp32 GEMM with NVFP4 AWQ weights dequantized on the fly) | 32B text encoder | single source. It is weight-only dequantization on CUDA cores, so FP4's missing hardware does not matter. Keep the same checkpoint (15.7 GB, smaller than `int8_convrot`'s 27 GB). |
| `minimax/te_vision.cu`, `qwen_image/vision.cu` | edits: the reference image through the vision tower | single source; its GEMMs go through the bf16 GEMM above |
| TensorFold NVFP4 `gemm_ck`/`gemm_ws`/`act`/`lane4` | H3 video's DiT linears on CUDA | **Not on gfx1151** (Blackwell block-scaled MMA). See precision below. |
| Triton AOT (`qwen_image/triton_k.zig`) | not used by H3 | none |

### Precision, and what it does to bit-exactness

The rule (CONTRIBUTING, "Engine correctness") is bit-exact against the twin **on the target GPU**. Bits compare only
within one architecture (INVENTORY.md section 5). For ROCm that means:

- **The twin runs on gfx1151.** `stk_twin` loads the same `.hip` sources as torch-ROCm extensions. The ISA check
  proves the twin and the engine run the same code. The engine is then bit-exact with that twin op by op, chained,
  and prompt to pixels.
- **Cross-vendor bits are not expected.** WMMA and mma.sync accumulate inside the instruction in their own fixed
  orders, and HIP's `expf`/`rsqrtf` are not CUDA's. Where a kernel is pure fp32 with written-out roundings (ops.cu,
  te32.cu), equal bits across vendors are possible. They are reported when observed, but they are not a gate.
- **H3-Image's first pack is bf16** (40.2 GB). It is the published precision with no quantisation question, and the
  bf16 GEMM above runs it on both vendors, so the same pack also gives a CUDA H3-Image (bf16 `Linear` kind in
  `qwen_image/weights.zig`). The reference baseline is written once: the twin's bf16 forward against ComfyUI's bf16
  forward on gfx1151 (cosine per step, as M7's structure gate), then LPIPS and DINO against #157's outputs at the
  same seeds, as INVENTORY section 1 did for Qwen-Image.
- **NVFP4 and FP8 packs on gfx1151** could only be weight-only (W4A16, W8A16), dequantised to bf16 in the GEMM's
  load stage. That is a different precision from CUDA's W4A4/W8A8. It saves memory, not time, and needs its own
  documented baseline. It is not planned for H3-Image.
- **`int8_convrot` (W8A8, 21.0 GB)** is the memory option: `iu8` WMMA accumulates exactly in int32, so the GEMM
  core is order-independent and could even match CUDA's bits given the same quantiser. It is a later PR in series 1
  with its own baseline, if memory on a shared box needs it (section 4).

## 4. Memory: GTT unified memory, resident and working bytes

On Strix Halo the GPU's allocations come out of system RAM through the GTT pool. `amdgpu.gttsize`, 96 GiB on #155's
box, caps it. The BIOS carve-out reported as "VRAM" is small and is not the budget. #155's README notes that
allocations past GTT fail even when MemAvailable has room. So:

- **Budget.** `src/sched/memory.zig` gets a Linux `gpu_ceiling` the way macOS has one. On an AMD GPU whose
  `mem_info_vram_total` is small next to `mem_info_gtt_total` (an APU), the budget is
  `min(MemAvailable, gtt_total - gtt_used)`. A discrete card keeps its VRAM figure. The sysfs readings are tested
  against #153's fixtures (`tests/fixtures/amd/strix-halo`, `discrete`). This is generic to AMD APUs, not to one
  box. `budget_bytes` and `reserve_bytes` still override it, so #155's rendered values keep working.
- **Page cache.** As on GB10, the page cache is the GPU's memory. Pack reads already use `POSIX_FADV_DONTNEED`
  (`qwen_image/pack.zig`). The same path serves HIP.
- **Resident bytes (`needs`, pure, *estimate*).** DiT bf16 40.2 GB + text encoder 15.7 GB + VAE 1.2 GB + buffers
  for the configured `max_pixels`, about 58 GB. With `unload_te` (an option that exists) it is about 42 GB, but each
  new prompt reloads 15.7 GB.
- **Working bytes.** Activations and attention workspace for the request's size. Edits add the reference latents and
  vision tokens per reference image, the same split as `reference_working_bytes`.
- Every figure is replaced by a measured peak from the gfx1151 run, taken from `gtt_used`, as `default_resident_mb`
  was measured on GB10.
- **The trade-off for JJ.** #155's voice stack books Q3 chat (62 GiB) + TTS + ASR inside 96 GiB GTT. A 58 GB bf16
  H3-Image therefore evicts the chat model, while int8 (about 39 GB) still evicts it but leaves more room afterwards.
  The scheduler handles either correctly. The choice affects only how often the chat model reloads.

## 5. Gates

ComfyUI is a development-time reference for the baseline gate only, never a runtime dependency of any engine.

**Here, with no GPU** (this dev box has none; heavy steps run in a Davit box):

- `zig build` and `zig build test` with `-Dgpu=cuda` and `-Dgpu=hip`, both with empty images.
- Host tests:
  - the HIP runtime's symbol table against the CUDA module's API;
  - the PDD schedule, bit-equal with the twin's plan;
  - the bf16 H3-Image pack reader on a synthetic pack;
  - `needs`/`check` limits and refusals;
  - the GTT budget from fixtures;
  - the masked composite and Lanczos resize (`image.zig`), bit-exact against the twin's CPU path (PIL / numpy);
  - the edit layout and prompt tokens (`H3-IMAGE.md` findings).
- `kernels/hip/isa_check.sh` once ROCm's clang and device libraries are installed: it compiles for gfx1151 with no
  GPU and checks the twin's ISA against the engine's.
- `sass_check.sh` on the single-source kernels, proving CUDA's SASS did not move.
- The twin's structure on CPU at a small size (256², 2 blocks): the H3-Image module against ComfyUI's code, which
  catches layout mistakes before a GPU window.

**On gfx1151**, in order, each recorded with hardware, ROCm version, precision, seeds and commands:

| Gate | What must hold |
| --- | --- |
| G0 `stk_twin` op tests (`test_ops`, `test_kitchen`-style, new `test_wmma`) | each HIP kernel against its torch reference within the twin's stated tolerance, and bit-equal across runs and grid shapes (the row-independence claim) |
| G1 `localrouter check h3i-replay`, `h3-te`, `h3i-vae` | every op alone, then chained, equal to the gfx1151 twin capture |
| G2 one DiT step chained, fast schedule vs reference schedule | velocities equal; the fused and graph schedules equal to the eager one byte for byte (as `h3-step-bench`) |
| G3 `localrouter check h3i-e2e` | prompt to decoded uint8 pixels equal to the twin; a warm rerun identical |
| G4 edits `h3i-edit-e2e`, masked and whole scene | reference encode, every step and the final pixels (after the composite) equal |
| Baseline | twin bf16 vs ComfyUI bf16 on gfx1151 (cosine a step); LPIPS/DINO vs #157 at the same seeds; time and GTT peak vs #157's table |

## 6. The PR stack (stacked drafts on `main`)

Series 1, H3-Image. Each PR runs `zig build` and `zig build test` (Zig 0.17.0), CI and `zig fmt` on touched files:

| PR | Content | Gate it carries |
| --- | --- | --- |
| 1.0 | this plan | — |
| 1.1 | `gpu` seam: `-Dgpu`, `src/gpu/hip` runtime, `build/hip.zig` (`-Dhip`, `-Dgfx`), engines import `gpu` | host tests; CUDA SASS unchanged |
| 1.2 | GTT budget in `sched/memory.zig` | fixture tests |
| 1.3 | `kernels/common/gpu.cuh` and the single-source kernels building for HIP; the twin's ROCm extension loader; `isa_check.sh` | ISA check; G0 for these kernels |
| 1.4 | HIP bf16 and f16 WMMA GEMMs and bf16 attention; bf16 `Linear` kind (also on CUDA); kitchen rms_rope HIP copy plus its UPSTREAM.md entry | G0 |
| 1.5 | `docs/dev/H3-IMAGE.md` (ComfyUI findings); H3-Image twin (`stk_twin/h3i`), bf16 pack build, captures, structure gate | CPU structure; baseline vs ComfyUI |
| 1.6 | engine: DiT image variant (no audio, PDD heads), image VAE decoder, PDD sampler, `minimax_h3_image` entry (`text_to_image`, its own licence) | G1-G3 |
| 1.7 | edits: references through `te_vision` and the VAE encoder, the masked composite, `image_edit` | G4 |
| 1.8 | Strix Halo tool entry on #155's template (after #155 merges), measured resident/working bytes, a qualification file | measured times and GTT peaks |
| 1.9 (if wanted) | `int8_convrot` W8A8 pack on `iu8` WMMA | its own baseline |

Series 2, H3 video, stacked on series 1:

1. Audio VAE and the video VAE tiles on HIP (f16 GEMM from 1.4).
2. The video DiT precision on gfx1151. The NVFP4 pack cannot run there, so the choice is bf16 or `int8_convrot`,
   with a documented baseline.
3. Attention at about 37 k tokens: an `iu8` WMMA port of comfy-kitchen's INT8 attention, measured against the bf16
   kernel.
4. I2V and the gates `h3-replay`, `h3-avae`, `h3-vvae` and `h3-e2e` on gfx1151.

Series 3, one PR: remove `tools/comfyui`, its CI step and the `ARCHITECTURE.md` mention. It lands only after the
native engines serve every job the worker served on Strix Halo.

## 7. Open points

- Which gfx1151 machine runs the gates (JJ is confirming). Nothing touches any GPU until then.
- Shipping precision of H3-Image on a shared 96 GiB box: bf16 first, then `int8_convrot` (this plan), or
  `int8_convrot` first.
- The H3-Image licence forbids hosted services to third parties without authorization. The engine's `license` says so
  (as `minimax_h3`'s does), and deployments must respect it.
