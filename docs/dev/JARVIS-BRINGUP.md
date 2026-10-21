# Jarvis box bring-up on DGX-01

These are hardware measurements for the single-GB10 "Jarvis box": chat (Qwen3.8-27B), voice in (Whisper) and out
(Kitten TTS 2, Fish Drama 3), and image and video generation, all on one DGX Spark. Every figure below was measured
on DGX-01 on 2026-10-21 by the scripts in `tools/spark/jarvis/`. dgx-02 was not used.

| | |
| --- | --- |
| Machine | DGX-01: GB10 (sm_121), 128 GB unified memory (121.7 GiB `MemTotal`), driver 580.178.04, CUDA 13.0, Ubuntu 24.04.5, aarch64 |
| Idle | 113.8–115.2 GiB `MemAvailable` (about 120 GB) before every run; no jobs, no containers, no GPU processes (checked before each heavy step) |
| LocalRouter | origin/main `eefae95`, plus this branch's `tools/spark/prepare.sh` fix (see [Step 4](#4-localrouter)) |
| TensorFold | release v1.1.1 `tensorfold-1.1.1-linux-aarch64.tar.gz`, sha256 `2e0103a544e64354bfec65bd8a052ede17aa9be1f6a56d8e37c868af9139cb39` (matches the release's `.sha256`, and every file passed the archive's own `SHA256SUMS`) |
| Memory metric | GB10 device memory is system memory, and CUDA allocations do not show in RSS. "Resident" is therefore the drop in `MemAvailable` from the idle baseline, sampled every 0.5 s (`memwatch.sh`). Per-process `nvidia-smi` figures and the servers' own reports are given alongside |

## Memory budget

| Model | Served by | Resident GiB (loaded, idle) | Peak GiB in use | Cold load s | Warm speed |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen3.8-27B MLX 4-bit + DFlash2, `--parallel 1 --prompt-cache-gib 2` (**Jarvis config**) | TensorFold 1.1.1 | **19.2** | **21.1** at a 30,056-token prompt | 29.9 (page cache dropped); 12.2 warm cache | 38.6–41.2 tok/s chat, 74.7–77.8 tok/s code; TTFT 1.51–1.53 s at 1.06k tokens |
| Same, TensorFold defaults (`--parallel auto`: 8 KV streams, 8 GiB prompt cache) | TensorFold 1.1.1 | 38.3–38.4 | 41.6 at 30,056 tokens | 31.5; 13.8 warm cache | 38.1–41.2 chat, 72.0–77.8 code; TTFT 1.51–1.55 s |
| Same, defaults, `--no-drafts` | TensorFold 1.1.1 | 35.6 | 37.9 | 9.2 warm cache | 13.3–13.4 tok/s; TTFT 1.54–1.55 s |
| Whisper large-v3-turbo (fp16, sdpa, Silero VAD) | LocalRouter `cmd` worker | 3.6 (worker reports 2.47; `nvidia-smi` 2.0) | 3.6 | 9.6 load; 17.6 first request | 1.39–1.50 s for 7.1 s of audio (RTF 0.20) |
| Kitten TTS 2 (packed LM, CUDA) | LocalRouter `cmd` worker | 10.6 (worker reports 8.8; `nvidia-smi` 4.9) | 10.6 | about 21 load; 32.1 first request | **RTF 1.50–1.52** (6.3–7.1 s of audio in 9.5–10.8 s): slower than real time; whole replies only |
| Fish Drama 3 (`fishaudio/drama-3@e0a9218a`), native | TensorFold 1.1.1 `serve` | 39.7–40.0 (38.58 GB of weights) | 40.0 (41.8 with default flags) | 29.7 (page cache dropped); 23.1–23.7 warm cache | 8 s sentence: first audio **1.40–1.41 s** streamed, **RTF 0.94** |
| Qwen-Image 2.1 Turbo FP8, 1024x1024, 8 steps | LocalRouter `qwen_image_turbo` | 35.3–39.3 while loaded | 39.3 | 9.3 (21.3 GiB read) | 1024x1024 in 11.6–16.6 s including the load |
| MiniMax H3 NVFP4 Turbo, 768x448, 8 steps | LocalRouter `minimax_h3` | 38.8–39.4 while loaded | 39.4 | 7.5 (30.0 GiB read) | 1 s clip in 25.3 s, 5 s clip in 65.6 s, each including the load |
| LocalRouter daemon (idle) | `localrouter serve` | 0.01 (14 MiB RSS) | | under 0.1 | |

Notes on reading the table:
- "Cold" for TensorFold means `/proc/sys/vm/drop_caches` was written first, so weights come from NVMe. The very first
  27B start on a machine adds 7.4 s of CUDA window-cost calibration, which is then cached in
  `~/.cache/tensorfold/window-costs`.
- The LocalRouter cold times are the first request after the daemon starts: worker spawn, load and one generation.
- The image and video rows come from the co-residency runs. The image figure varies with what else holds memory
  (35.3 GiB with Drama 3 resident, 39.3 GiB without).

## Co-residency verdict

**The whole voice loop stays warm together, with room for one generation job at a time.**

| State (measured together) | In use, GiB | `MemAvailable`, GiB |
| --- | ---: | ---: |
| Idle machine | 0 | 113.8–115.0 |
| 27B chat (Jarvis config) | 19.3–19.4 | 94.5–95.5 |
| + Whisper + Kitten (`keep_loaded`) | 30.8–31.8 | 82.9–83.1 |
| + Drama 3 (TensorFold native) | **69.9–71.0** | **43.9** |
| + one 1024x1024 image job (Turbo) | 106.4–106.7 at the peak | 7.1–8.5 |

What evicted what, through LocalRouter's scheduler (runs `co-full`, `co-full-5s`, `co-nodrama-5s`):

| Job | With chat + Whisper + Kitten + Drama 3 warm | With chat + Whisper + Kitten warm (no Drama 3) |
| --- | --- | --- |
| Image, 1024x1024 | Nothing unloaded. Admitted with 43.9 GiB free; done in 16.3–16.6 s. The idle image model then holds about 36 GiB for its idle TTL, leaving 7.1–8.5 GiB free, which is **below the daemon's own 8 GiB `reserve_bytes`** | Nothing unloaded; done in 11.6 s |
| Video, after that image | The idle image model is unloaded. **Kitten was also unloaded in one of two runs** (the 1 s run; the 5 s run kept it): the margin is a few GiB and depends on `MemAvailable` at the moment of admission. Whisper stayed. Video done (1 s: 25.3 s; 5 s: 65.6 s) | The idle image model is unloaded (43 GiB free was short of 39 GiB + 8 GiB reserve). Kitten and Whisper stay. 5 s clip done in 65.6 s |
| Chat and Drama 3 | Never touched: they run outside LocalRouter. The scheduler sees them only as less `MemAvailable` and cannot unload them | |

Consequences for the design:
1. **Image and video are never resident together** beside the voice loop: the second job always evicts the first.
   Each takes about 39 GiB.
2. **After a video, the evicted TTS stays down.** A `keep_loaded` tool comes back only when it fits without
   evicting anything. The idle video model held 39 GiB and left 15.8 GiB free, so Kitten could not come back. It
   stays unloaded until the video model's idle TTL expires (600 s in the test config). On the Jarvis box, generation
   models want a short `idle_ttl_s` (or 0) so that voice comes back right after the job.
   - **Defaults, this branch:** `install.sh` now writes `idle_ttl_s` 60 for image models and 30 for video
     (`IMAGE_IDLE_TTL_S`, `VIDEO_IDLE_TTL_S`; both were 300). `docker/tools.json` and the README example match.
     A reload costs 9.3 s (image) or 7.5 s (video), so a burst of jobs still reuses the loaded model.
     `IMAGE_KEEP_WARM` now defaults to 0: the default image model loads on demand. Voice and chat come first on
     this box, and chat (up to 24 GiB), Whisper (3.6), Drama 3 (up to 54) and Kitten (10.6) leave no room for a
     resident 36 GiB image model, while an image reload costs only about 9 s. `IMAGE_KEEP_WARM=1` restores the old
     behavior for an image-only box.
   - **PR #14 (chat as a tool) does not close this gap.** Its `loadKept` is main's: a kept-warm tool reloads only
     when `fits(resident)` without evicting anything (`src/sched/scheduler.zig:178` on main; the same check in #14).
     That holds for every kind, Whisper and Kitten included. So after a video, TTS comes back only once the video's
     TTL ends: with the new 30 s default, about 30 s + 2 s of grace + Kitten's 21 s load. #14's chat tool (priority
     100) waits the same way. #14's `tests/jarvis.zig` gives its video tool `idle_ttl_s: 1`, which hides this.
   - **The fix belongs in the scheduler.** A kept-warm tool that does not fit should be allowed to evict idle tools
     of lower priority, the way a request's admission does. #14 also checks resident bytes only when a kept-warm
     tool returns, not working bytes, so that tool's first job may have to evict again.
3. **Admission does not hold the reserve after a load.** It checks the request's estimate against
   `MemAvailable - reserve_bytes`, but the idle model it leaves behind can push the machine below the reserve (7.1
   GiB measured). Nothing was killed. DGX-01's earlyoom (`-m 2,1`) acts only below 2 % available, about 2.4 GiB.
   Its `--avoid` list spares `tensorfold-` processes but not LocalRouter's Python workers, so those would be its
   first victims.
4. **LocalRouter has no `chat` kind**, so the 27B and Drama 3 (TensorFold servers) sit outside its scheduler. To
   rank the voice loop above generation jobs, LocalRouter has two options. It can run them as tools (a `cmd`
   adapter around `tensorfold serve`, or a new kind). Or it can let the config declare them as external residents
   with their measured size, so the scheduler accounts for them instead of only noticing smaller `MemAvailable`.

### Picking the TTS

| | Kitten TTS 2 (LocalRouter today) | Drama 3 on TensorFold 1.1.1 |
| --- | --- | --- |
| Memory | 10.6 GiB | 40 GiB |
| Speed | RTF 1.5 (slower than real time) | RTF 0.94 (just faster than real time) |
| First audio | whole reply only (10 s for a 7 s sentence) | 1.4 s streamed |
| Fits beside chat, Whisper, an image or a video job | yes, with 40+ GiB to spare | yes, on a margin of a few GiB for video (it can cost Kitten) |

Neither is a responsive live-voice TTS on this box today. Kitten cannot keep up with playback, and Drama 3's 1.4 s
first audio plus 0.94 RTF leaves almost no slack. The fastest measured part of the loop is the chat model: 0.13 s
TTFT on a short turn, and 0.61 s on the next turn of a cached 20k-token conversation.

### Kitten on GB10: why RTF 1.5 and 10.6 GiB (to confirm on the node)

Kitten has not been GPU-qualified. `docs/dev/KITTEN-TTS2.md` records the GB10 gate as one failed run (9 of 37
checks) and one run "pending … Until it passes, CUDA is not qualified". Its only speed figure is CPU on the dev box
(RTF about 10). So RTF 1.5 is the first GB10 speed number, not a regression.

**It did run on the GPU.** The evidence:
- The setup printed `torch 2.11.0+cu130`.
- The tool config sets `"device": "cuda"`.
- The worker logged `cuda peak reserved 4760 MiB` on every request.
- `nvidia-smi` listed the worker with 4.9 GiB.

**Likely cause of the speed:** each request is host-bound per codec token.
- The language model runs through stock transformers `generate()` (kittenml 0.9.4 `model.py:380`): eager PyTorch,
  batch 1, bf16, sdpa attention (FA2 when importable), and a LogitsProcessor chain at every step (repetition,
  token-run, temperature, top-k, top-p).
- Each codec token is one Python step launching every kernel of the 1.7B model, about 28 layers' worth, from
  GB10's Arm cores. The GPU waits on those launches.
- One second of audio needs 25 tokens. The run made about 170 tokens in about 10 s, roughly 20 tokens/s.
- The decoder (S3 flow plus HiFT) should be the small part.
- Not the cause: CPU fallback, or batch and warm-up (the voice conditioning is built at load, and three warm runs
  gave the same RTF).
- Torch's default of 20 CPU threads is a minor suspect. The probe also runs with 4.

**Likely cause of the memory:** 10.6 GiB is 4.8 GiB of CUDA reserve (the bf16 LM 3.5 GB, s3gen about 1 GB, KV)
plus 5.5 GiB of host RSS.
- That host RSS equals the CPU-only resident on the dev box (5.51 GB). The host copy of the weights is the likely
  reason.
- `lm_weights: packed` is unpacked to bf16 on the CPU and installed with `load_state_dict(assign=True)`, then
  copied by `.to("cuda")` (`model.py:179-188`).
- The CPU tensors are freed, but tensors under glibc's 32 MiB mmap threshold go back to malloc's arenas, not the
  OS. Torch and its CUDA libraries account for the rest.

**Confirm with one probe** (`tools/spark/jarvis/kitten_probe.py`). It runs the worker's own load and synthesis
three times, timed apart. It needs about 12 GiB and 3 minutes, and nothing else may run on the GPU:

```sh
scp tools/spark/jarvis/kitten_probe.py dgx01@192.168.1.116:jarvis/scripts/
ssh dgx01@192.168.1.116 'for t in 0 4; do systemd-run --user --scope -q -p MemoryMax=24G timeout 900 \
  ~/jarvis/venvs/kitten/bin/python -I ~/jarvis/scripts/kitten_probe.py ~/jarvis/src/localrouter ~/jarvis/models/kitten $t \
  | tee ~/jarvis/out/kitten-probe-t$t.jsonl; done'
```

| Hypothesis | Confirmed if |
| --- | --- |
| On the GPU | `env`: `lm_devices` and `decoder_devices` are `["cuda:0"]` |
| Host-bound decode | `run`: `lm_tokens_per_s` about 20, `lm_cpu_over_wall` about 1.0 (one core busy for the whole LM), `gpu_kernel_share` well under 0.5, `decoder_s` a small part of `wall_s` |
| Threads | the `t4` run's RTF is no better (if it is clearly better, set `threads` in the tool config) |
| Retained host weights | `memory`: `rss_after_trim_gib` about 3 GiB below `rss_gib` |

**Fixes, not made here:**
- **Memory:** a `malloc_trim(0)` after load, or loading the LM straight to the device. Either keeps the bits and
  should save about 3 GiB. Re-run the gate to show it.
- **Speed:** a static KV cache with CUDA graphs (`torch.compile(mode="reduce-overhead")`), or kittenml's own vLLM
  backend (`kittentts2/vllm/`). Both pick other kernels, so per `CONTRIBUTING.md` they need the bit gate re-run or
  an accepted new baseline.

## Recommendation for PR #13 (Drama 3 adapter)

**Do not switch #13 to TensorFold 1.1.1's own Drama 3 server yet.** Keep drama-tensorfold's launcher, but make two
changes in #13 first:

| | drama-tensorfold (its README, GB10; not measured here) | TensorFold 1.1.1 native (measured, DGX-01) |
| --- | --- | --- |
| First audio, streamed | 0.141 s | **1.40–1.41 s** (10x slower) |
| RTF, about 8 s of audio | 0.634 (0.682 after its patch 0040) | **0.938–0.946** |
| Whole reply, about 8 s of audio | 4.89 s | 7.45 s (7.94 s of audio) |
| Start to ready | 6.4 s (one start, warm caches) | 23.1–23.7 s warm cache, 29.7 s cold |
| Memory | 54 GiB cap; needs **88 GiB free at load**; aborts below 34 GiB | **39.7–40.0 GiB** measured |
| Degenerate replies | silence guard: patches 0039 and 0055 end a reply that stays silent | none: "The morning train leaves the station at seven fifteen, crosses the river…" gives 18.7 s of audio with an **8.39 s silent gap**, identically every time |
| Voices, references, long form | voice files, references, `--long-form` | default voice only ("only the default voice is supported") |
| Supply chain | 59 patches on TensorFold `459771e7`, built from source | signed release archive, sha256-pinned, no build |

The native server is simpler to ship and uses less memory. On every latency figure that matters for voice, though,
it is behind what drama-tensorfold reports, and it has a reproducible degenerate reply that drama-tensorfold guards
against. The two changes for #13:
1. **Its 88 GiB free-at-load admission can never pass on the Jarvis box.** With the 27B chat, Whisper and Kitten
   warm, 83.1 GiB was free. Measure drama-tensorfold's real footprint on DGX-01 (the native engine needs 40 GiB),
   then set the admission from that measurement, as #13's `qualify.sh` run would.
2. **Keep the adapter on the shared HTTP contract** (`POST /v1/audio/speech`, `stream: true` with `pcm`, 44.1 kHz
   s16le mono). Both servers speak it, so the later switch is a launcher change. Switch once a TensorFold release
   ships the silence guard and first audio within about 2x of drama-tensorfold's 0.14 s. Re-run
   `tools/spark/jarvis/drama_run.sh` and `drama_silence.sh` against each new release.

## Results by step

### 1. TensorFold v1.1.1

- **Installed** to `~/jarvis/opt/tensorfold-1.1.1-linux-aarch64/`. `tensorfold-native --version` reports
  `tensorfold-native 1.1.1`.
- **`capabilities --json` libraries:**
  - Found: cuBLAS 13, cuBLASLt 13 and libtfgrammar.
  - Not found: NCCL, libturbojpeg, libwebp and libtfvideo. None of these are needed here.

### 2. Chat

- **Pulls** (`tensorfold pull`):
  - `TensorFold/Qwen3.8-27B-MLX-4bit@22d8d538` (14.95 GiB). Memory floor per `tensorfold info`: 16.98 GiB.
  - `z-lab/Qwen3.8-27B-DFlash2@50307d4c` (3.58 GiB).
  - Both ran at about 190 MiB/s, with every file checked by `pull`.
- **Server's own startup report** (defaults): "model 15.40 GiB; 8 streams at once, 2.71 GiB each at a
  32768-token window … a 8.0 GiB prompt cache".
  - That is why the defaults take 38 GiB.
  - `--parallel 1` keeps one 2.71 GiB stream, and with a 2 GiB prompt cache the server sits at 19.2 GiB.
  - Decode speed is unchanged.
  - `/memory` reports 18.05 GiB of device memory (36.95 GiB with the defaults).
- **Smoke**, streamed and whole: "The capital of France is Paris."
  - Streamed TTFT 0.134 s, in 2 chunks.
  - Whole reply in 0.309 s.
  - `finish_reason` `stop`.
- **Decode** (greedy, 512 tokens, 3 runs per prompt, the prompts in `chat_bench.py`):
  - Chat, with the drafter: 38.1–41.2 tok/s.
  - Code, with the drafter: 72.0–77.8 tok/s. Code drafts better: about 430 of 1,100 drafted tokens accepted,
    against about 340 of 1,900 for chat.
  - Without the drafter: 13.3–13.4 tok/s.
  - The drafter is a 3.0x speed-up on chat and 5.6x on code.
  - TensorFold's recipe reports 50.0 (chat) and 57.8 (code) tok/s on GB10 with its own prompts.
- **Time to first token:**

  | Prompt | TTFT |
  | --- | --- |
  | 1,056–1,060 tokens | 1.51–1.55 s |
  | 20,058 tokens | 57.9 s |
  | 30,056 tokens | 120 s |

  - **Next turn** on a cached 20k-token conversation: **0.61 s**.
    - 19,968 of 20,087 tokens came from the prompt cache.
    - The state was 1.39 GiB, about 71 KiB per token.
  - A 30k-token conversation needs about 2.0 GiB of cache, so **use `--prompt-cache-gib 4`**. A cache miss on a
    long history costs a minute or two.

### 3. Drama 3, TensorFold native

- **Checkpoint:** `tensorfold pull fishaudio/drama-3@e0a9218aa48888270f06ef1ed9ea67622c0e17d2`, 38.63 GiB.
  - All 10 checksummed files match drama-tensorfold's `recipe/checksums/checkpoint.sha256` (`sha256sum -c`: codec,
    8 shards, tokenizer).
- **Serving:** TensorFold serves it with `tensorfold serve fishaudio/drama-3`. There is no separate speech command.
  - It logs "Drama PDL stays off: device or both image markers are unqualified".
  - Output is 44.1 kHz, mono, s16.
  - Flags: drama-tensorfold's launcher flags (`--context 8192 --parallel 1 --no-drafts --prompt-cache-gib 0
    --keep-warm 0`) and the defaults give the same speed. The launcher flags save 1.7 GiB.
- **Two runs on the same 125-character sentence**, *"Good evening. Your calendar is clear tomorrow morning, but you
  have a dentist appointment at two and dinner with Sam at seven."*:
  - Each produces 7.94 s of audio.
  - First audio: 1.404–1.410 s, which matches the server's log (`first_audio_ready_seconds=1.461` on the longer
    text).
  - Streamed RTF: 0.946.
  - Whole RTF: 0.938.
  - It contains one 0.43 s pause.
- **Saved WAV:** `~/jarvis/out/drama-native-8s/whole0.wav`, sha256 `4479e713…00cb7904`.
  - LocalRouter's Whisper transcribes it word for word.
- **Silence probe** (`drama_silence.sh`, 4 texts):

  | Text | Audio | Silent gap |
  | --- | --- | --- |
  | Short, 34 characters | 1.86 s | none; RTF 1.22 |
  | Calendar | 7.94 s | none |
  | Flights | 9.29 s | none |
  | Train | 18.7 s | 8.39 s |

  - The train sentence repeats its gap byte for byte, greedy and sampled alike.

### 4. LocalRouter

- **`zig build`** on DGX-01 at `eefae95`: exit 0, 2 min 14 s, 1.6 GB max RSS.
- **`zig build test`:** 108 of 109 tests pass.
  - The one failure is environmental. `pack.zig`'s "aligned reads" test writes to the fixed path
    `/var/tmp/stk-pack-direct-test`, and a root job left that directory root-owned on 2026-10-17. I left it in
    place because it is outside `~/jarvis`.
  - A test that writes to a fixed shared path collides across users. This branch gives every such test a directory
    unique to the run (`src/engines/qwen_image/testdir.zig`; the same random suffix in `audio.zig` and the
    integration harness): the pack, capture, memcap receipt, audio and integration-daemon directories.
  - The suite's selftest line reads "image 251 ms, video 205 ms". Idle daemon RSS was 14,144 KiB, and `/v1/models`
    p50 was 1.2 ms.
- **`tools/spark/install.sh` fails at origin/main** on a fresh machine with the default `IMAGE_MODELS` (which
  includes Turbo):
  - `prepare.sh` downloaded Turbo with `--include "transformer/*" "scheduler/*" model_index.json …`.
    `huggingface_hub` 1.33's `hf download` takes the extra words as file names and then ignores `--include`
    ("Ignoring `--include` since filenames have been explicitly set").
  - So no transformer weights were fetched, `.complete` was written anyway, and the Turbo pack crashed with
    `KeyError: 'img_in.weight'`.
  - This branch passes one `--include` per pattern and requires the transformer weights before writing `.complete`.
    An existing bad `.complete` is downloaded again.
  - With that fix the install ran to completion: exit 0, 2,129 s, on loopback (`LOCALROUTER_BIND=localhost`).
  - Its MCP checks passed: a 512x512 Turbo image in 6 s, and a 1 s 768x448 clip in 22 s, both including the load.
- **Speech:** the image built by `install.sh` has no Python, so the `cmd` speech workers cannot run in it.
  - For speech, a native `localrouter` was built on DGX-01 with `-Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121`
    (ReleaseSafe, 104 s).
  - Whisper and Kitten ran from `tools/speech/{whisper,kitten}` setups under `~/jarvis` (all pins and the VAD sha256
    checked).
  - The same daemon also served the image and video packs that `install.sh` made, for the co-residency runs.
  - Whisper transcribed Kitten's output and the Drama 3 WAV exactly.
  - Kitten's warm RTF of 1.5 was the same with nothing else running.

### 5. Co-residency

See [the verdict](#co-residency-verdict). The single-user chat config was used throughout. One run had Drama 3
loaded last; the other did not.
- **With Drama 3:**
  - Peak in use: 106.7 GiB (1 s video run) and 106.4 GiB (5 s video run).
  - Lowest `MemAvailable`: 7.1 GiB, with the image model idle-resident.
- **Without Drama 3:** peak 72.9 GiB.
- **Chat after both generation jobs:** answered normally (TTFT 0.135 s).

## Commands

Everything ran as user `dgx01` on DGX-01, under `~/jarvis`. `S=~/jarvis/scripts` is a copy of
`tools/spark/jarvis/`. Each server ran as a transient `systemd-run --user` unit with `RuntimeMaxSec` and
`MemoryMax`, and was stopped at the end of its script. Before each heavy step, these checks returned empty:
`ls ~/ai/lr-jobs`, `docker ps -q`, and `nvidia-smi --query-compute-apps=pid --format=csv,noheader`.

```sh
# tools (zig and uv checked against their published sha256; ffmpeg 9.0.2 static arm64 from BtbN, checked against its checksums.sha256)
curl -fsSL https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz -o zig.tar.xz   # sha256 9e8d1166…
curl -fsSL -o uv.tgz https://github.com/astral-sh/uv/releases/download/0.13.4/uv-aarch64-unknown-linux-gnu.tar.gz
curl -fsSL -o ffmpeg.tar.xz https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-linuxarm64-gpl-9.0.tar.xz

# 1. TensorFold
V=1.1.1
curl -fsSLO https://github.com/ashhart/TensorFold/releases/download/v$V/tensorfold-$V-linux-aarch64.tar.gz
curl -fsSLO https://github.com/ashhart/TensorFold/releases/download/v$V/tensorfold-$V-linux-aarch64.tar.gz.sha256
sha256sum -c tensorfold-$V-linux-aarch64.tar.gz.sha256        # OK; 2e0103a5…9cb39
tar -C ~/jarvis/opt -xzf tensorfold-$V-linux-aarch64.tar.gz && (cd ~/jarvis/opt/tensorfold-$V-linux-aarch64 && sha256sum -c SHA256SUMS)
ln -s ~/jarvis/opt/tensorfold-$V-linux-aarch64/bin/tensorfold-native ~/jarvis/bin/tensorfold
tensorfold capabilities --json

# 2. chat (HF_HOME=~/jarvis/hf throughout)
tensorfold pull TensorFold/Qwen3.8-27B-MLX-4bit
tensorfold pull z-lab/Qwen3.8-27B-DFlash2
PORT=28080 bash $S/chat_run.sh drafter cold --drafter z-lab/Qwen3.8-27B-DFlash2       # cold load, smoke, TTFT, decode
PORT=28080 bash $S/chat_run.sh drafter-warm warm --drafter z-lab/Qwen3.8-27B-DFlash2  # + the 30k fill
PORT=28080 bash $S/chat_run.sh plain-warm warm --no-drafts
PORT=28080 bash $S/chat_run.sh jarvis-p1 warm --drafter z-lab/Qwen3.8-27B-DFlash2 --parallel 1 --prompt-cache-gib 2
#   chat_run.sh serves: tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --name qwen27b --host 127.0.0.1 --port 28080
#                       --context 32768 --temperature 0 --no-thinking --no-update-check <flags above>
#   and the Jarvis config's cold load the same way after: sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
bash $S/chat_turns.sh 20000 2 8                                                     # next-turn TTFT, cache 2 and 8 GiB

# 3. Drama 3
tensorfold pull fishaudio/drama-3@e0a9218aa48888270f06ef1ed9ea67622c0e17d2
curl -fsSL -o drama-checkpoint.sha256 https://raw.githubusercontent.com/plotarmordev/drama-tensorfold/main/recipe/checksums/checkpoint.sha256
(cd ~/jarvis/hf/hub/models--fishaudio--drama-3/snapshots/e0a9218aa48888270f06ef1ed9ea67622c0e17d2 && sha256sum -c ~/jarvis/dl/drama-checkpoint.sha256)
bash $S/drama_run.sh native-cold cold                                   # tensorfold serve fishaudio/drama-3 --name fishaudio/drama-3 --port 28793
bash $S/drama_run.sh native-p1-warm warm --context 8192 --parallel 1 --no-drafts --prompt-cache-gib 0 --keep-warm 0
bash $S/drama_silence.sh greedy; bash $S/drama_silence.sh t08 --temperature 0.8 --top-p 0.8
TEXT="Good evening. Your calendar is clear tomorrow morning, but you have a dentist appointment at two and dinner with Sam at seven." \
  bash $S/drama_run.sh native-8s warm --context 8192 --parallel 1 --no-drafts --prompt-cache-gib 0 --keep-warm 0

# 4. LocalRouter (git archive of eefae95 in ~/jarvis/src/localrouter; ZIG_GLOBAL_CACHE_DIR=~/jarvis/zig-cache)
systemd-run --user --scope -p MemoryMax=24G -p CPUQuota=600% timeout 1100 zig build
systemd-run --user --scope -p MemoryMax=32G -p CPUQuota=800% timeout 1800 zig build test --summary all
docker tag localrouter:dev localrouter:preserved-5fa470cddc5e    # keep the machine's existing image (restored after)
LOCALROUTER_HOME=~/jarvis/lr LOCALROUTER_BIND=localhost LOCALROUTER_PORT=8190 MODELS="image video" \
  IMAGE_MODELS="qwen-image-2.1 qwen-image-2.1-turbo" bash tools/spark/install.sh        # run 1: the Turbo download bug
#   (same command again with this branch's prepare.sh: exit 0)
docker compose --env-file ~/jarvis/lr/.env -f docker/compose.yaml -p localrouter down
docker tag localrouter:dev localrouter:jarvis-bringup; docker tag localrouter:preserved-5fa470cddc5e localrouter:dev
bash tools/speech/kitten/setup.sh ~/jarvis/venvs/kitten && bash tools/speech/kitten/fetch.sh ~/jarvis/models/kitten
UV_PROJECT_ENVIRONMENT=~/jarvis/venvs/whisper uv sync --locked --python 3.12 --project tools/speech/whisper
python3 -I tools/speech/whisper/fetch_model.py --root ~/jarvis/models/whisper
curl -fsSL -o ~/jarvis/models/vad/silero_vad.jit https://raw.githubusercontent.com/snakers4/silero-vad/5cd7945676eb32225748052e2e6a0580e4686a08/src/silero_vad/data/silero_vad.jit   # sha256 e1122837… OK
zig build -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121          # native daemon, ~/jarvis/src/localrouter-cuda
bash $S/speech_daemon.sh ~/jarvis/out/drama-native-8s/whole0.wav    # localrouter serve --config (daemon_config.sh, KEEP=false) --port 28190

# 5. co-residency
bash $S/coresident.sh full with-drama
VIDEO_SECONDS=5 bash $S/coresident.sh full-5s with-drama
VIDEO_SECONDS=5 bash $S/coresident.sh nodrama-5s no-drama
```

## State left on DGX-01

- **Stopped:** every process and unit started here. No `jarvis-*` units, no GPU processes, and the `localrouter`
  compose project is down.
- **Restored:** `localrouter:dev` points at its original image `5fa470cd…` again. The image built here is tagged
  `localrouter:jarvis-bringup` (`62665cbf…`), together with the `nvidia/cuda:13.0.1-base-ubuntu24.04` base it
  pulled.
- **`~/jarvis` (215 GB)**, kept for the next design steps:
  - Checkpoints and packs: `hf/` 58 GB and `lr/` 144 GB.
  - Wheels: `uv-cache/` 8.3 GB.
  - Speech models, venvs, sources, scripts and every raw result under `out/`.
  - About 170 GB of that was downloaded.
  - Some files under `lr/` are root-owned, because `install.sh`'s container writes as root.
- **Written outside `~/jarvis` by these runs, not removed:**
  - `~/.cache/tensorfold/window-costs/` (2 files, 590 B each).
  - `~/.local/share/uv/` (empty except a credentials directory).
- **Found running and left alone:**
  - `~/ai/perf1100-w1-20261021/maintenance.py`, listening on 127.0.0.1:18080 since 2026-10-20.
  - A `gpuwatch.py` watcher.
