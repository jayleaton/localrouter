# deploy/strix-halo: LocalRouter on an AMD Strix Halo box

Profile: `deploy/profiles/strix-halo-gfx1151.json` (status: preview). One box with a Ryzen AI Max+ 395 / Radeon 8060S
(gfx1151): unified LPDDR5X memory of which the GPU may map the GTT pool (96 GiB with `amdgpu.gttsize=98304`). The
installer never changes kernel parameters, firmware or the bootloader; the GTT size is a prerequisite it only checks.

| Tool | Kind | Runs as | GTT |
| --- | --- | --- | --- |
| `qwen3.8-flash-next` | chat (default) | `openai_server` → `tools/gufo/serve.sh` → Gufo `serve llm` (UD-Q4_K_XL + MTP) on :8081 | 88 GiB |
| `qwen3.8-flash-next-q3` | chat (the voice loop's) | `openai_server` → llama.cpp `llama-server` (Vulkan, UD-Q3_K_XL + MTP) on :8080 | 62 GiB |
| `qwen3-tts-customvoice` | speech | `tools/speech/openai_audio` → Gufo `serve tts` on :8091 | 6.5 GiB |
| `qwen3-asr` | transcription (default) | `tools/speech/openai_audio` → Gufo `serve asr` on :8092 | 5.5 GiB |
| `whisper-large-v3-turbo` | transcription | `tools/speech/whisper/worker.py` in the ROCm venv | 2.5 GiB |

`budget_bytes` is the GTT pool less 2 GiB (`render.sh`, from `tools/install/lib.sh hw_gtt`): a GPU allocation past GTT
fails even when MemAvailable has room, so the scheduler must plan against GTT. Q3 + TTS + ASR fit together (the voice
loop); Q4 + TTS + ASR do not, so a `/v1` chat on the Q4 model unloads the speech models and the voice loop swaps back.

```sh
zig build -Doptimize=ReleaseSafe                 # no -Dnvcc: the CUDA engines build with empty kernel images
deploy/strix-halo/start.sh                       # render config.json, start the transient user unit, wait for /health
deploy/strix-halo/status.sh                      # unit, health, each tool's state, containers, GTT
deploy/strix-halo/stop.sh                        # stop the daemon (its workers stop their servers), then leftovers
SITE=path/to/site.env deploy/strix-halo/start.sh           # with a site file (paths, voice, which model does what)
```

`env.sh` holds every default (prefix `$J`, ports, model directories, measured memory); a site file sourced after it
overrides any of them. `tools/install/doctor.sh --profile strix-halo-gfx1151 --unit localrouter-amd` checks the box
(GPU, ROCm, GTT, Podman, the steps, the service); export the site file's variables first when it moves model paths.

## Measured (2026-11-15, Radeon 8060S, ROCm 7.2.4, kernel 7.2, Podman 6.1 with runc)

- `/v1/chat/completions` on `qwen3.8-flash-next` (cold load 17.5 s): 56 tokens in 1.69 s; streamed answers.
- `/v1/audio/speech` (`stream`: pcm, chunked): first audio 226 ms, about 2.4x real time; `wav` 4.75 s cold.
- `/v1/audio/transcriptions`: Qwen3-ASR 2.4 s cold, 0.58 s warm for a 5.9 s clip; Whisper ROCm 9.8 s cold, 0.52 s warm.
- Voice loop over `/v1/realtime` (`tools/spark/jarvis/voice_e2e.py --smoke --skip-barge`, chat on the Q3 tool):
  PASSED; first audio after the end of speech 1217 ms and 1376 ms on warm turns (25 s on the first, a cold chat load),
  a 42.6 s spoken reply with no underrun.

Not here: the compiled-in engines (Qwen-Image, MiniMax H3, Kandinsky-6) are CUDA kernels, and Drama 3 has GB10 and Metal
builds only. Gufo serves Qwen-Image 2.1 and MiniMax H3 on gfx1151; wrapping them is the next step (a `cmd` image/video
worker, or ComfyUI behind one).
