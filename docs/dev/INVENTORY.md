# Inventory: what exists before LocalRouter (2026-10-07)

What we start from: paths, models, recorded numbers, and what each piece gives LocalRouter. Numbers are copied from
the source docs named in each row; none were re-measured here.

## 1. Image: Qwen-Image 2.1 on TensorFold (`tfimage`)

| | |
| --- | --- |
| Repo | `github.com/jayleaton/qwen-image21-tensorfold-rtx` (public, `main` f918455). The author's working copy has an extra `docs/OPENAI-SERVER.md` the public repo lacks. Read-only clone on the dev box: `ref/qwen-image21-tensorfold-rtx` |
| Language | Python + Triton (about 1,000 lines of engine in `tfimage/`), TensorFold v0.6.1 (17c73e1) as an unmodified submodule for the NVFP4 / FP8 GEMMs |
| Host | ComfyUI 0.37 (custom loader node). ComfyUI still runs the text encoder, sampler and VAE. The standalone OpenAI server drives a private headless ComfyUI; the ComfyUI-free port is listed as pending |
| Model | DiT: 32 single-stream blocks, dim 4096, 32 heads x 128, SwiGLU 12288, one shared modulation, about 7.1B parameters. Latent: VAE /16, 64 channels, no patchify (1 MP = 4,096 tokens). Text encoder: Qwen3-VL 8B (ComfyUI's `int8_convrot`). VAE: `qwen_image_2.1_vae_bf16` (676 MB). Sampler: euler, simple schedule, 25 steps, CFG 1 |
| Engine tricks | NVFP4 W4A4 linears on TensorFold's prompt GEMMs (`gemm_ws` bulk-copy tiles), `mlp_prompt` (gate/up with a SwiGLU epilogue writing FP4 rows), fused Triton adaLN / RMSNorm+RoPE / SwiGLU, static activation scales from a calibration run, prefix K/V (text + reference tokens) kept per prompt, cuDNN attention forced |
| Kernels it uses | TensorFold `src/tensorfold/cuda/nvfp4/`: `prompt.cu`, `gemm_ws.cu`, `gemm_ck.cu`, `act.cu`, `nvfp4q.cuh`, `mma4.cuh`, `swiglu4.cuh` (quant4, gemm, gemm_ws, mlp_prompt) |
| Perf (RTX 5070 Ti, sm_120, 16 GB) | 1024x1024, 25 steps, warm, end to end: **8.0-8.4 s** (ComfyUI GGUF Q6_K: 33.0 s). DiT step 0.29-0.30 s: attention 101 ms, NVFP4 GEMMs 124 ms, other 25 ms. 480x608: 1.9 s. FP8: 16.7-17.8 s at 1 MP. Text encoder 4.2 s first prompt, 0.26 s warm. VAE decode about 1 s. Engine load from its converted cache about 2 s. Source: `docs/dev/RESULTS.md` sections 2 and 4 |
| Memory | NVFP4 DiT 3.66 GiB, engine peak 4.3 GiB at 1 MP (about 2.4 GB working set at 2048x2048). Converted cache 4.2 GB |
| Exactness | bf16 path vs ComfyUI's forward: relative L2 0.005, cosine 0.99996. NVFP4 single step vs bf16: 7.3 % relative L2 (FP8 2.0 %). Quality gate (6 prompts): NVFP4 mean LPIPS 0.193, DINO 0.908 (DINO bar 0.95 failed, two layout-sensitive prompts); FP8 LPIPS 0.126, DINO 0.920 |
| Weights | Public on Hugging Face: `Qwen/Qwen-Image-2.1` (diffusers layout: `transformer/` 2 shards, `text_encoder/` 4 shards, `vae/`) and `Comfy-Org/Qwen-Image-2.1` (`diffusion_models/qwen_image_2.1_bf16` and `int8_convrot`, `text_encoders/qwen3vl_8b_bf16` / `int8_convrot` / `w4a8`, `vae/qwen_image_2.1_vae_bf16`). None on the dev box. The engine itself reads the Q6_K GGUF (city96) and caches its NVFP4 conversion |

## 2. Video: MiniMax H3 on TensorFold (`tfvideo`)

| | |
| --- | --- |
| Repo | `github.com/jayleaton/minimax-h3-tensorfold-rtx` (public, `main` 711d942). Read-only clone on the dev box: `ref/minimax-h3-tensorfold-rtx` |
| Language | Python + Triton (about 1,700 lines in `tfvideo/`), TensorFold v0.6.1 submodule, comfy-kitchen's INT8 attention |
| Host | ComfyUI: the engine replaces only the 50 transformer blocks. Packing [text, keyframes, audio, video], per-token timesteps, RoPE, PDD output heads, text encoder, sampler, both VAEs and the mp4 stay in ComfyUI |
| Model | Audio + video. DiT: 50 single-stream blocks, hidden 5376, 56 heads x 128, SwiGLU 14336, per-token modulation, about 19.3B parameters (checkpoint `minimax_h3_fl2va_pruned_int8_convrot`, 21 GB int8 with a 256-wide Hadamard rotation). Text encoder: Qwen3-VL 32B (`qwen3vl_32b_minimax_h3_nvfp4_awq`). Video VAE: a 36-layer ViT over 256 px tiles (fp16, or Comfy-Org's `int8_convrot`). Audio VAE fp32. Sampler: res_multistep, simple, no CFG; Turbo LoRA (lightx2v) for 8 steps. 1344x768, 124 frames (5 s, 24 fps) = 37 latent frames x 1,008 patches, about 37,800 tokens |
| Perf (RTX 5070 Ti) | 1344x768 5 s video + audio, warm, new seed: stock ComfyUI 623 s; engine NVFP4 20 steps 273 s; + Turbo 8 steps 133 s; + sparse attention 96 s; + int8 video VAE **81 s** (sampling 67 s, VAE 10 s, audio + mp4 4 s). A new prompt adds the 32B text encoder (about 50 s cold, seconds warm). Source: `docs/dev/RESULTS.md` sections 5 and 7 |
| Where a step goes | one block at S = 37,790: attention 148 ms (66 %, INT8), NVFP4 GEMMs 64 ms (28 %), elementwise 14 ms. Attention, not the weights, is most of the work (about 2.2 PFLOP a forward vs 1.5 of linears) |
| Memory | NVFP4 DiT 10.1 GiB. On 16 GB it streams blocks from pinned RAM; a Spark holds everything resident |
| Weights | Public: `Comfy-Org/MiniMax-H3` (`diffusion_models/minimax_h3_fl2va_*` in bf16, int8_convrot, fp8_scaled, w6a8; `ref2va` variants; Turbo LoRAs 4-step 768p and 8-step) and `lightx2v/Minimax-h3-Turbo`. None on the dev box |
| Exactness | bf16 engine vs ComfyUI on captured steps: video cos 0.9983-0.9998, audio 0.9997-0.9999. NVFP4: video cos 0.956-0.992. Whole videos are bitwise reproducible across runs (dense and LoRA workflows) |

## 3. A model gateway (prior art for the API)

| | |
| --- | --- |
| Repo | the author's gateway project, branch `feat/local-image-gen` (unmerged, running). Rust tray (Tauri, hyper) + TypeScript hub |
| Gateway | `apps/tray/src-tauri/src/models/gateway.rs`: OpenAI-compatible proxy on the dev box, port 8890 (`/v1`). `GET /v1/models` entries carry `kind` (`chat` / `image` / `embedding`), `machine`, `running`. Routes any `/v1/*` by the `model` field (JSON or multipart). Error envelope `{"error":{"message","type","param","code"}}` with 404 unknown model, 409 switched while waiting, 503 failed to start, 504 not ready in time, 502 upstream unreachable |
| Lifecycle | `models/manager.rs`: one model per machine, swap = stop then start, readiness by polling `GET {localUrl}/models` each second, start timeout 900 s default, process group killed with the tray, last 8 log lines in errors. No idle shutdown, no async jobs (sync with a 45-minute turn limit), no video kind |
| Image contract in use | `POST /v1/images/generations` `{model, prompt, n, size "WxH", response_format "b64_json", seed, steps}`; `POST /v1/images/edits` multipart with `image[]` (up to 5). `seed` and `steps` are the agreed extensions. Model id `qwen-image-2.1` |
| Reuse | the `/v1/models` extras, the error envelope and status mapping, readiness via `GET /v1/models`, `seed`/`steps`, "a started process is stopped only through its own handle". LocalRouter adds what it lacks: memory-budgeted residency, idle unload, async video jobs |

## 4. TensorFold's Zig engine and a large-model port (conventions)

| | |
| --- | --- |
| Upstream | the TensorFold Zig checkout, branch `zig-flashnext` (88c424e). Zig 0.17.0. CUDA runtime `zig/src/cuda/` (driver via dlopen, context, memory, streams, modules, launch, graphs, cuBLASLt, NCCL, Triton AOT cubins), about 2,000 lines, no torch. Core: `safetensors.zig`, `checkpoint.zig`, HF tokenizer (parity with `tokenizers`), Jinja templates. Kernels embedded as fatbins built by `-Dnvcc=` or supplied by `-Dfatbins=`. Its `build.zig` exposes no public modules, so a dependent builds modules from its paths |
| Existing kernels worth reusing | `zig/kernels/cuda/prefill_attention.cu`: TensorFold's bf16 flash attention (mma.sync, D 128); `qmm_frag.cuh` fragments; `torch_ops/` (bit-exact replacements of torch ops) |
| Large-model port | a TensorFold worktree on its own branch (read only). Its docs cover the port, kernels, performance and tensor parallelism |
| Conventions taken from it | interfaces first (vtables), tests run the real implementation; `.cu` kernels copied, not rewritten (`sync.py` writes copies from git refs, `--check` fails on drift; SASS equality checked against the Python build); Triton kernels launched from captured cubins (`aot.zig`); bit-exact gates per op (alone and chained replay of a captured Python run) then per forward; GPU tests on blocking streams; launches take device pointers only; on GB10 read weights with `POSIX_FADV_DONTNEED` (the page cache is GPU memory); files of 600 lines or fewer with one-line comments; pod jobs stream results to the volume every 30 s |
| Recorded speed context | the large-model port's Zig M2b: load 99 s a rank for 57.7 GB from local NVMe on PRO 6000 pods; GB10 is emulated on PRO 6000 at 48 SMs (`proto/gb10emu.cuh`) |

## 5. Infrastructure

| | |
| --- | --- |
| Rented-GPU tooling | scripts for unattended job pods (session, hold, idle guard, staging); every pod writes a row to a ledger. PRO 6000 (sm_120) is the Blackwell stand-in; bits compare only within one architecture |
| Docker on the dev box | `nvcr.io/nvidia/pytorch:26.07-py3` is present (nvcc for fatbins, as the large-model port's `nvcc-docker.sh` uses) |
| Sparks | two Sparks run another service in production. Off limits; windows on them are coordinated with that service's owner |
| Dev box | x86_64, 16 cores, 27 GiB RAM (at most 4 parallel CPU workers), no GPU |
