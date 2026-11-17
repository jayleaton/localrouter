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
| `minimax-h3` | video (text or image to video, with audio) | `tools/comfyui` → ComfyUI on :8188, `comfyui/minimax-h3-t2va.json` (int8 DiT + 4-step turbo LoRA) | 36 GiB (booked: all) |
| `minimax-h3-image` | image (text to image, edits with an optional mask) | `tools/comfyui` → ComfyUI on :8189, `comfyui/minimax-h3-image-t2i.json` and `-edit.json` (Q4_K_M GGUF, 6 steps; whole-scene edits: `-edit-unmasked.json`, Q8_0, ~2 MP, 30 steps) | 31 GiB t2i / 37 GiB whole-scene (booked: all) |

`budget_bytes` is the GTT pool less 2 GiB (`render.sh`, from `tools/install/lib.sh hw_gtt`): a GPU allocation past GTT
fails even when MemAvailable has room, so the scheduler must plan against GTT. Q3 + TTS + ASR fit together (the voice
loop); Q4 + TTS + ASR do not, so a `/v1` chat on the Q4 model unloads the speech models and the voice loop swaps back.
The MiniMax tools are booked at the whole budget (`H3_RESIDENT=budget`): ComfyUI reads the weights into host memory
(`--disable-mmap`) before the GPU takes them, and on this APU that is the same RAM the GTT pool comes from, which the GTT
budget does not see (host use peaked at 63 GiB for an image whose GTT peak was 30 GiB; beside the 62 GiB Q3 model it
would not fit). So a MiniMax request unloads every other model first, and the voice models (`keep_loaded`) come back
a couple of seconds after the job, unloading the MiniMax tool; the next MiniMax request starts ComfyUI again.

```sh
zig build -Doptimize=ReleaseSafe                 # no -Dnvcc: the CUDA engines build with empty kernel images
deploy/strix-halo/start.sh                       # render config.json, start the transient user unit, wait for /health
deploy/strix-halo/status.sh                      # unit, health, each tool's state, containers, GTT
deploy/strix-halo/stop.sh                        # stop the daemon (its workers stop their servers), then leftovers
SITE=path/to/site.env deploy/strix-halo/start.sh           # with a site file (paths, voice, which model does what)
deploy/strix-halo/comfyui/install.sh comfyui     # the MiniMax tools' ComfyUI (pinned; PyTorch ROCm venv); again: no-op
deploy/strix-halo/comfyui/install.sh models      # their weights (about 90 GB, including the H3-Image Q8_0 GGUF; COMFY_HF=DIR reuses a download)
```

Every download is pinned: the Gufo runtime by digest (`tools/gufo/IMAGE`; `start.sh` passes it into the unit, so the
workers run the image bootstrap pulled), the Hugging Face weights by commit with every file's size and sha256
(`pins/*.json`, fetched and checked by `tools/install/hf_pinned.py`), the Whisper venv by `uv.lock` hashes, the ComfyUI venv by a hashed requirements file, llama.cpp
and the host tools by sha256. `gufo diagnose` is a GPU step of its own (`--no-gpu` skips it).

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

> **Licence note (MiniMax).** The MiniMax H3 and MiniMax H3-Image Community Licenses grant nothing in the United
> States, the European Union, the United Kingdom or South Korea ("Excluded Territories"), and set further conditions
> elsewhere (Acceptable Use Policy, disclosure of machine-generated content, revenue thresholds). The `minimax-h3` and
> `minimax-h3-image` tools are opt-in (the profile's `minimax` model needs `--accept`): check the licences
> ([H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
> [H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)) before you deploy them, or remove
> them from `config.json.in`. LocalRouter shows each tool's licence and excluded regions and enforces nothing.

MiniMax through ComfyUI. `comfyui/install.sh` (the profile's `minimax` model: steps `comfyui` and `weights`) installs,
without sudo and idempotently, ComfyUI and ComfyUI-GGUF at pinned commits under `$COMFY_ROOT` (`$J/comfy`), a Python
3.13 venv with PyTorch 2.14.1+rocm7.2 from `comfyui/requirements-rocm7.2.txt` (every package by sha256, installed with
`--require-hashes`), `comfyui/nodes/minimax_h3_image_gguf.py` (loads the Unsloth GGUF with the PDD config ComfyUI's
detection needs), `comfyui/nodes/rocm_conv3d_patchify.py` (PyTorch ROCm's 3-D convolution segfaults ComfyUI on image to
video, in the vision tower's patch embedding; on ROCm only, a Conv3d whose kernel equals its stride, with no padding, unit
dilation and one group, runs as the matrix product it is instead: equal to PyTorch's up to the order of the sums, checked
by `comfyui/test_conv3d_patchify.py`; `LOCALROUTER_CONV3D_PATCHIFY=0` in the tool's `env` turns it off; it replaces any
hand patch of `comfy/ops.py`, which also stops `install.sh` moving the checkout; docs/dev/UPSTREAM.md, entries 13 and 14), and the weights at pinned revisions (`pins/minimax-*.json`: every file's size and sha256, checked in
full before a manifest is marked done), linked into `ComfyUI/models`. A step whose pins match is skipped; a checkout
with local changes is left alone. Edits auto-select by whether a mask is supplied (a request may still set size, steps or guidance). A masked or local
edit uses `comfyui/minimax-h3-image-edit.json`: Q4_K_M, the first input image scaled to about 1 MP on the 32-pixel grid
(unless the request names a size), 6 steps, cfg 1.0; further images (up to 5 in all) are references (`<Picture 2>` ...),
and the mask keeps the input's pixels wherever it is opaque (`ImageCompositeMasked`). A whole-scene edit (no mask) uses
`comfyui/minimax-h3-image-edit-unmasked.json`: Q8_0, the input scaled to about 2 MP on the same 32-pixel grid (1440x1440
when square, 2.07 MP), 30 steps, cfg 1.0, denoise 1.0. Image to video sends the first frame to `MiniMaxH3ImageToVideo`,
which stretches it to the clip (LocalRouter has already cut it to the clip's shape). Measured with Q3 + TTS + ASR loaded
before each request:

- `/v1/images/generations`, `minimax-h3-image`, 1024x1024: 47.4 s (the three models unloaded in 2 s, ComfyUI started,
  39.6 s in ComfyUI with cold weight loads); GTT peak 29.6 GiB, host memory in use at most 63 GiB.
- `/v1/videos`, `minimax-h3`, 512x288, 2 s (56 frames, the 17k+5 grid) with AAC audio: 56.0 s (50.2 s in ComfyUI);
  GTT peak 36.1 GiB, host memory in use at most 74 GiB. 1 s (39 frames): 47.2 s.
- `/v1/images/edits`, `minimax-h3-image`, a 1024x1024 input without a size. Masked (Q4_K_M, about 1 MP, 6 steps, cfg
  1.0): 104-111 s (98-100 s in ComfyUI: the reference is encoded by the vision encoder and the VAE); GTT peak 30.7 GiB,
  host use at most 63 GiB. With a mask over the figure ("a bright red umbrella"): the change lands inside the mask, and
  outside it the output equals the input (mean absolute difference 0.02 of 765). Whole-scene (Q8_0, about 2 MP, 30 steps,
  cfg 1.0): a rain-to-snow restyle was convincing (falling snow, a white street, snow on rooftops and signs, neon and
  the figure kept) in 384 s, GTT peak 37.2 GiB. At 1 MP, neither Q8_0 nor 30 steps was enough (the street stayed wet);
  raising cfg added some snow along the sidewalks but took more than 6 minutes.
- `/v1/videos` with `input_reference`, `minimax-h3`, 512x288, 2 s: 64.7 s (58.9 s in ComfyUI); GTT peak 41.0 GiB, host
  use at most 73 GiB; the first frame is the input's centre crop, then the scene moves as asked, with sound.
- The voice models were back 2-3 s after each job (a speech and transcription round trip right after: 1.4 s, 0.5 s).

Not here: the compiled-in engines (Qwen-Image, MiniMax H3, Kandinsky-6) are CUDA kernels, and Drama 3 has GB10 and Metal
builds only. Not wired: H3's last frame (`last_frame`), and H3-Image's prompt expansion.

Licences: both MiniMax licences grant nothing in the United States, the European Union, the United Kingdom or South
Korea (the tools' `license` lists them; LocalRouter shows it and enforces nothing). Check them before you deploy the
MiniMax tools.
