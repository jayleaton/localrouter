# LocalRouter: plan

One DGX Spark (GB10, aarch64, sm_121, 128 GB unified memory) runs several generative tools on demand: images first,
then video with audio, later more. A tool loads, generates, and unloads. Agents call it over HTTP, OpenAI-compatible
where an OpenAI shape exists. The engines are ported to TensorFold's Zig architecture, the way an earlier large-model port was
(its docs live in the TensorFold worktree). What exists today is in `docs/dev/INVENTORY.md`.

## 1. Shape of the system

```
 agent / curl / localrouter CLI
        |  HTTP :8190  (/v1/models, /v1/images/*, /v1/videos/*, /v1/tools/*)
 +------v-------------------------------------------------------------+
 | localrouter serve  (one process, always on, tiny: no CUDA, no weights)     |
 |   api ──> jobs ──> scheduler (owns the GPU: memory budget, queue)  |
 |                      |  Tool vtable                                |
 |                      v                                             |
 |               ProcessTool  ──stdin/stdout JSON lines──┐            |
 +-------------------------------------------------------│------------+
                                                         v
                         localrouter worker <tool>   (one process per loaded tool)
                           Engine vtable: qwen_image | minimax_h3 | testpattern | ...
                           CUDA context, weights, kernels; writes outputs to the job dir
```

- **One daemon, one worker process per loaded tool.**
  - **Unload is process exit.** The OS takes back every byte, so the memory returns to the co-tenant service or the next tool
    with no allocator leak or fragmentation to reason about.
  - **A CUDA fault stays in its worker.** A sticky CUDA error poisons a context, so it costs one job and the daemon
    stays up.
  - **Cancel is kill.** The scheduler kills the worker's process group and reloads on the next job.
  - **Any language can serve a tool.** The worker protocol is a few JSON lines, so a tool can live behind it before
    it is ported (a Python reference, say) and swap to its Zig engine without the daemon changing.
  - **Cost:** a process spawn and a CUDA context (well under a second) on each load, small next to the weight load.
- **The daemon never touches CUDA.** It reads free memory from `/proc/meminfo` (on GB10, device memory is system
  memory) and the workers report their real resident bytes.
- **Everything is one binary**, `localrouter`, built by Zig: `localrouter serve`, `localrouter worker <tool>`, `localrouter gen ...` (the CLI client)
  and `localrouter check ...` (gates). Engines are compiled in; an external tool is a `cmd` in the tool config.

## 2. Interfaces

All interfaces are vtables (`ptr` + `*const VTable`), as in TensorFold. A tool is described by the config, and the
code behind it is an `Engine`.

### 2.1 `Tool` (daemon side, `src/tool/tool.zig`)

| Call | Contract |
| --- | --- |
| `info() Info` | id, kind (`image` / `video` / later `audio`, ...), the request kinds it accepts, display name |
| `needs(req) Needs` | `resident` bytes while loaded + `working` bytes this request adds at its peak (from size, frames, n). Pure, cheap, no I/O. Admission uses it before `load` |
| `load(deadline) !Loaded` | make the tool ready. Returns the measured resident bytes (they replace the estimate) |
| `generate(job, sink) !Output` | run one request. `sink` takes progress (step, of, phase) and is polled for cancel. Output files go in the job's directory |
| `unload()` | release everything. Never fails; for a process tool it is exit, then SIGKILL after a grace period |
| `health() Health` | `unloaded` / `loading` / `ready` / `busy` / `failed{reason, log tail}` |

`ProcessTool` is the only daemon-side implementation. It spawns `localrouter worker <id>` (or the config's `cmd`) in its
own process group, so the process dies with the daemon, and proxies the calls above as JSON lines.

### 2.2 Worker protocol (`src/tool/wire.zig`)

One JSON object a line, each direction. Inputs and outputs are files in the job directory, never inline bytes.

```
daemon -> worker   {"op":"load"}
worker -> daemon   {"ok":true,"resident":4294967296}            | {"ok":false,"error":"...","type":"..."}
daemon -> worker   {"op":"generate","id":"img_..","dir":"/data/jobs/img_..","request":{...}}
worker -> daemon   {"progress":{"phase":"denoise","step":7,"of":25}}   (any number)
worker -> daemon   {"ok":true,"files":["0.png"],"meta":{"seed":42,"ms":8123}}
daemon -> worker   {"op":"exit"}   (unload)
```

### 2.3 `Engine` (worker side, `src/engine/engine.zig`)

Same verbs as `Tool` minus process control: `load`, `generate`, `unload`, `needs`. `localrouter worker` wraps any `Engine` in
the protocol loop. Tests drive an `Engine` directly or through a real worker process; the daemon cannot tell them
apart.

### 2.4 Requests (`src/tool/request.zig`)

A tagged union per kind, validated by the API before it reaches the scheduler:
- `image`: prompt, negative prompt, size, n, seed, steps, guidance, references (edits), precision.
- `video`: prompt, size, seconds, fps, seed, steps, audio, first frame.

A new kind adds a union arm, so the compiler finds every switch that must handle it.

### 2.5 Scheduler (`src/sched/scheduler.zig`)

- **One executor thread owns the GPU.** One job runs at a time, since a DiT step already fills the GPU.
- **Queue:** FIFO. Two loaded tools that fit together both stay loaded, so alternating requests do not reload.
- **Budget:**
  - Admission needs `sum(resident of loaded) + needs(req).resident + needs(req).working <= budget`.
  - `budget = min(configured cap, MemAvailable - reserve)`, read when the job is admitted.
  - When it does not fit, idle tools are unloaded least recently used first.
  - When it still does not fit, the job fails with 503 `insufficient_memory`, never an OOM. That protects a
    co-resident service.
- **Residency:** `idle_ttl` per tool (default 120 s; 0 means unload right after each job), plus
  `POST /v1/tools/release`, which unloads everything so the co-tenant's start script can claim the Spark.
- **Failure:**
  - A worker that dies or misses its deadline fails the job (503 with the log tail).
  - The tool goes back to `unloaded`; the next job retries the load once.
  - `load` and `generate` have deadlines from the config.
- **Cost:** queue and accounting are O(1) per job (tools are a short array; no scans per step).

### 2.6 API (`src/api/`)

| Route | Shape |
| --- | --- |
| `GET /health` | `{"ok":true}` |
| `GET /v1/models` | OpenAI list. Each entry adds the gateway's `kind`, `running` (loaded), `machine`, plus `resident_bytes`. The gateway's readiness probe (`GET {url}/models`) works unchanged |
| `POST /v1/images/generations` | OpenAI. Extras: `seed`, `steps`, `guidance`, `negative_prompt`. `response_format` `b64_json` (default) or `url` (served from `/v1/files/...`). Synchronous: holds the request through the queue and render |
| `POST /v1/images/edits` | OpenAI multipart, `image[]` up to 5 (milestone 6) |
| `POST /v1/videos` | OpenAI Videos API: returns the video job `{id, object:"video", status, progress, model, size, seconds, created_at}` at once |
| `GET /v1/videos/{id}`, `GET /v1/videos/{id}/content`, `DELETE /v1/videos/{id}` | poll, download (`video/mp4`), cancel or delete |
| `GET /v1/tools`, `POST /v1/tools/{id}/load`, `POST /v1/tools/{id}/unload`, `POST /v1/tools/release` | residency and memory, for operators and for the co-tenant's start script |

- Errors use the gateway's envelope and statuses: 400 invalid request, 404 unknown model, 409 cancelled,
  503 load failed or insufficient memory, 504 deadline.
- Every job is an async job internally. The synchronous image route just waits on it.
- Outputs live under `data_dir/jobs/<id>/` and are deleted after `keep_outputs` (default 24 h).

### 2.7 Agent access

- **HTTP:** the routes above. `localrouter gen image "prompt" -o x.png` and `localrouter gen video ...` are thin clients over them.
- **Agent skill:** `docs/AGENTS.md`, as a plain agent skill.
- **A model gateway:** one catalog entry points a gateway at LocalRouter's `/v1`. The gateway then routes every
  LocalRouter model, and LocalRouter does its own swapping.

## 3. Engines: what is ported and what is reused

The reference for every port is a **Linux Python twin** of the existing engines.
- **What it is:** `tfimage` / `tfvideo` + TensorFold v0.6.1, run headless on a pod. The parts the engines do not own
  yet (text encoder, sampler, VAE, packing) come from the official diffusers-format weights and code where diffusers
  supports the model (`Qwen/Qwen-Image-2.1` is published that way), else from ComfyUI's code used as a library. Either
  way it is pinned, and no ComfyUI server runs.
- **Shared kernels:** where the twin calls a CUDA kernel, the Zig engine launches the same `.cu` (copied by
  `sync.py`, SASS-checked).
- **Where Zig cannot reproduce a torch or cuDNN op bit for bit** (cuDNN attention and convolutions):
  - we write the kernel once;
  - the twin calls it as a torch extension;
  - the kernel gets its own numeric gate against the op it replaces.
- **Result:** the end-to-end gate stays bit-exact, and fidelity to ComfyUI is tracked separately.

| Piece | Today | Port / reuse |
| --- | --- | --- |
| CUDA runtime (driver, memory, streams, modules, launch, graphs, cuBLASLt) | TensorFold `zig/src/cuda` | **Reuse as is**: a pinned `build.zig.zon` dependency on `zig-flashnext`; modules built from its paths, with no fatbins of its own (`with_kernels = false`) |
| safetensors, mmap, HF tokenizer, Jinja template | TensorFold `zig/src/core` | **Reuse** (Qwen3-VL's tokenizer and chat template) |
| NVFP4 quantize, prompt GEMMs, `gemm_ws`, `mlp_prompt` | TensorFold v0.6.1 `cuda/nvfp4/*.cu` | **Copy** with `sync.py` (same lines and namespaces, so the same SASS); launches ported with host-tested tile policies |
| Fused adaLN, RMSNorm+RoPE, SwiGLU, gated residual | Triton (`tfimage/kernels.py`, `tfvideo/kernels.py`) | First the captured Triton cubins (`aot.zig`, bit-exact by construction). Then CUDA rewrites where the profile says they pay, each bit-gated against the cubin |
| Image attention (bf16, D 128, prefix + target) | cuDNN through torch SDPA | **Write**: extend TensorFold's `prefill_attention.cu` to non-causal with a kept prefix. The twin switches to it. Gate: cos >= 0.99999 vs cuDNN, and speed >= cuDNN on PRO 6000 |
| Video attention (S about 38k) | comfy-kitchen INT8 (SageAttention-style) | **Copy** the `.cu` (Apache-2.0). An FP8 / FP4 QK rewrite is a later perf milestone |
| Qwen-Image 2.1 DiT | `tfimage/qwen_image21.py` (300 lines) | **Port** to `src/engines/qwen_image/` (config, pack, block, forward, prefix K/V) |
| MiniMax H3 DiT, packing, PDD heads | `tfvideo/minimax_h3.py` + ComfyUI's `MiniMaxH3Model` | **Port**. No block streaming on a Spark: everything stays resident |
| Qwen3-VL text encoder (8B image, 32B video) | ComfyUI | **Port** once as a dense prefill on the same linears (NVFP4 or bf16 by config). The vision tower comes with image edits |
| Image VAE decoder (and encoder for edits) | ComfyUI, cuDNN convolutions | **Port**: implicit-GEMM conv kernels shared with the twin |
| Video VAE (36-layer ViT over tiles), audio VAE | ComfyUI | **Port**: the ViT reuses the DiT's linears and attention; audio VAE in fp32 |
| Samplers (euler / simple, res_multistep), Turbo LoRA merge | ComfyUI | **Port** (small, host side, golden-tested against ComfyUI's sigmas) |
| PNG | ComfyUI / PIL | **Write**: `std.compress.flate` + CRC, about 100 lines |
| mp4 with audio | ComfyUI (PyAV) | **Reuse ffmpeg** in the image: raw frames and PCM piped in. NVENC is optional later |
| Weight conversion (GGUF / int8 ConvRot to NVFP4 + activation scales) | `tfimage/store.py`, `calibrate.py` | First the twin's converter writes a **pack** (safetensors + a manifest with digests). Later `stk convert` in Zig, once the forward exists for calibration |

Engine code is laid out like a TensorFold family (`config`, `pack`, `load`, `kernels*`, `block`, `forward`,
`engine`), so it can move upstream into `zig/src/families/` later without a rewrite.

## 4. Packaging: Docker

**Docker, one image, multi-stage.** Why:
- DGX OS ships Docker with NVIDIA's container toolkit, and the Spark's production service already runs that way.
  Operators use one mechanism.
- The runtime needs pinned user-space libraries: cuBLASLt (the bit gates depend on its version), ffmpeg, and later
  cuDNN-free kernels only. An image pins them; a bare binary would depend on whatever CUDA the host has.
- A public recipe becomes `docker run --gpus all -p 8190:8190 -v <models>:/models -v <data>:/data ...`.
  The driver (`libcuda`) comes from the host through the container toolkit, as it must.
- Cost: none on hot paths. The binary is native; the container adds no runtime overhead on the GPU.

The build stage is the NGC CUDA devel image: nvcc for `sm_121` (Spark) and `sm_120` (pods), plus Zig 0.17. The
runtime stage is CUDA base + cuBLASLt + ffmpeg + `localrouter`, built for `linux/arm64` (Spark) and `linux/amd64` (RTX and PRO
6000 hosts). Weights are a volume, never baked in. The container runs as a non-root user; GPU memory is guarded by
the scheduler's budget, since cgroups do not bound GB10 device allocations reliably.

A bare binary (`zig build` + systemd) stays possible for development; it is not the published path.

## 5. Gates

- **Bits:**
  - Same GPU architecture, same inputs: byte-equal outputs, per op (alone and chained replay of a captured twin
    run), per forward, then per image or video.
  - sm_120 bits are proven on rented PRO 6000 pods; sm_121 bits on a Spark in a window.
  - A new kernel that replaces a torch or cuDNN op also gets a numeric gate (cosine and max relative error, fixed
    before the run) against that op.
- **Speed:**
  - The Zig engine is no slower than the twin on the same GPU, per step and end to end.
  - GB10 numbers are measured only on a Spark.
  - Every speed claim names the GPU, size, steps, warm or cold, and the commit.
- **Quality:** the existing LPIPS / DINOv2 gate vs stock ComfyUI is kept as a report, not a merge gate (it measures
  the precision choice, not the port).

## 6. Milestones

| # | Milestone | Done when (gates) | GPU |
| --- | --- | --- | --- |
| M0 | Repo, build, pinned TensorFold runtime dependency | `zig build test` green on x86_64; `zig build -Dtarget=aarch64-linux-gnu` cross-builds `localrouter` | none |
| M1 | LocalRouter core: `Tool` / `Engine` vtables, worker protocol, `ProcessTool`, scheduler (budget, LRU, idle TTL, deadlines), HTTP API (models, images, videos jobs, tools, errors), `localrouter gen` client, PNG encoder, `testpattern` engine (deterministic seeded image or video; also the selftest), Dockerfile | Integration tests start the real server and real worker processes: concurrent requests, budget eviction, a worker crash fails one job and the server survives, cancel, idle unload, video job lifecycle. Idle daemon RSS under 30 MB; API overhead under 5 ms a request excluding generation; image builds for amd64 and arm64 and `localrouter check selftest` passes in it | none |
| M2 | Qwen-Image Linux twin and capture (**done**, `docs/dev/RESULTS.md`) | bf16 twin within 1.5x of diffusers' own attention-backend floor at every step (measured 1.25-1.28x); 6 gate prompts rendered in bf16, FP8 and NVFP4; packs with digests; one step plus the prefix captured for each precision | pod: 42 min, $1.46 |
| M3 | Qwen DiT in Zig (**done**, `docs/dev/RESULTS.md`; attention now 1.19x faster with the same bits, 1.3x from cuDNN) | `sync.py` clean, SASS equal on sm_120 and sm_121 (29/29); step 0 replay bit-exact alone and chained for NVFP4 (723/723) and FP8 (851/851); toolkit attention within its numeric gate vs cuDNN (1.2e-4) but 1.56x slower per call, the first speed task; Zig step = twin kernel time + 3 % host time | pods: $1.27 |
| M4 | Image tool end to end in Zig (**done**, `docs/dev/RESULTS.md`): Qwen3-VL 8B text path, sampler, VAE decoder, PNG, in LocalRouter | Same seed and prompt give the same pixels as the twin on sm_120 (6 prompts, 2 sizes, both precisions; met); end to end no slower than the twin (6.7 s vs 11.7 s NVFP4 at 1 MP, 25 steps; met); cold load measured (8.0 s NVFP4, 12.5 s FP8) | pod: 46 min, $0.55 |
| M5 | Spark window for images (**done** 2026-10-07, `docs/dev/RESULTS.md`: 51/51 bit gates on sm_121, 12.7 s NVFP4 at 1 MP, 38 GiB peak; three container bugs fixed after) | Runbook run by the user: sm_121 bits vs the twin on GB10, then GB10 warm and cold times at 1 MP and 0.3 MP, peak memory, Docker on DGX OS | Spark window |
| M6 | Image edits (reference images: Qwen3-VL vision tower, VAE encoder, `/v1/images/edits`) | Bit-equal to the twin on the edit workflow | pod |
| M7 | MiniMax H3 video + audio: twin, DiT, INT8 attention, Qwen3-VL 32B, video and audio VAEs, res_multistep, Turbo merge, mp4 with audio | Per-step bits as M3; whole video bit-equal to the twin on sm_120; end to end no slower than the twin | pods, then a Spark window |
| M8 | Public recipe (one-command `tools/spark/install.sh`, MCP on `/mcp`, the agent skill `skill/localrouter/` done; image on GHCR waits for the go-ahead) | README, `docs/AGENTS.md`, compose file, image on GHCR; needs the user's go-ahead to push | none |

Speed work after parity is measured, not guessed: CUDA graphs per step, FP8 or FP4 attention for video, sparse
attention, and GB10-specific tiles. Each lands behind the same launch interface with its own bit gate (or a declared
new reference).

## 7. Risks and how the plan meets them

| Risk | Mitigation |
| --- | --- |
| ComfyUI owns the text encoders, VAEs and samplers today, so the twin needs ComfyUI's code | The twin imports ComfyUI's modules as a library on the pod (pinned commit), with no server. Each piece is replaced by the twin's own code only when the Zig port of it lands |
| Bits across cuBLASLt versions | The Docker image pins cuBLASLt. The twin runs in the same CUDA version on the pod |
| GB10 has 48 SMs against the 5070 Ti's 70 and the PRO 6000's 188 | Speed gates on pods are relative (Zig vs twin). Absolute GB10 numbers come only from the Spark window. The 48-SM emulation (`gb10emu.cuh`) guides tile choices offline |
| Page cache is GPU memory on GB10 | Weight reads use `POSIX_FADV_DONTNEED`, as the earlier port's `Pack.read` does |
| A co-resident service | Live `MemAvailable` in every admission, a reserve, and `/v1/tools/release` |
| Weight licences and distribution of converted packs | Packs are built locally from the official weights by default. Publishing packs is the user's decision (TODO.md) |

## 8. Working rules

- Commit per milestone step on `main` here; no remotes, no pushes.
- Files of 600 lines or fewer, one-line comments, `zig build test` green before each commit.
- Subagents get disjoint paths. Rented pods follow the pod tooling's rules: each pod under $3,
  under $15 in total without asking, no launch under a $35 balance, delete when done.
- The Sparks are only touched in a window the user grants, through a runbook.
