# Kitten TTS 2

[KittenML/kitten-tts-2](https://huggingface.co/KittenML/kitten-tts-2) is LocalRouter's text to speech model: a 1.7B
speech language model (Qwen3 architecture, ternary weights) that writes S3 codec tokens, and an S3 flow decoder plus
HiFT vocoder that turn them into 24 kHz audio. It has 47 built-in voices, nine of which are named after the language
they carry, and inline expression markup. It is served by a `cmd` worker, `tools/speech/kitten/kitten_worker.py`,
running KittenML's own Python package (the reference implementation) on pinned, hash-checked files.

Powered by Stellon Labs. See [Licences](#licences): the weights' commercial grant has conditions an operator must
settle before commercial use.

## What it serves

| | |
| --- | --- |
| Request | `speech`: `input` (1 to 4096 characters), `voice`, `preset`, `seed`; `speed` 1 only |
| Output | `speech.wav`: RIFF, mono, 16-bit PCM, 24000 Hz. The daemon encodes other response formats |
| Voices | 47; default Bruno. Arabic, Chinese, French, German, Hindi, Italian, Portuguese, Russian and Spanish are the way to those languages |
| Presets | `stable` (temperature 0.8, top-p 0.8, top-k 50; the default) and `expressive` (0.9, 0.9, 50) |
| Markup | a leading `[emotion]` (angry, contemplative, excited, joyful, mundane, nervous, sad, stern, surprised, tender), inline `<event>` (gasp, giggle, growl, gulp, laugh, pause, scoff, sigh, sob, um) and `(((emphasis)))`. The model's own controls, passed through as text. Beta upstream: they steer delivery, they do not guarantee it |
| Refused (400) | `speed` other than 1 (the model has no rate control; resampling would not be the model), `instructions` (use the markup instead), streaming, an unknown voice or preset, empty or over-long input |
| Not offered | voice cloning from a reference clip (it needs an audio input contract and the speaker and Whisper models); the lossy `emb4` packing and student decoders (see [Numerics](#numerics)) |

Long input is split as the package splits it: sentences grouped into chunks of up to 380 characters (130 minimum),
each chunk generated with its own token budget (25 codec tokens a second; a runaway is capped at 1.8x the expected
length), decoded, trimmed of edge silence and joined with a 0.16 s pause and 8 ms fades. The worker reports one
progress step per chunk. Text is normalized by KittenML's English normalizer ("$12.50" becomes "twelve dollars fifty
cents"); for the nine language voices it is switched off, as KittenML advises, since it would mangle numbers in other
languages. `normalize` in the options forces it either way.

The seed seeds torch's generator before the first chunk, exactly as the package does. With the same files, device,
thread count and seed, the same request gives the same bits, cold or warm ([the gate](#numerics) checks this).

## Setup

```sh
tools/speech/kitten/setup.sh /opt/localrouter/kitten-venv     # Python 3.12 from the hash-pinned lock (x86_64 CPU, aarch64 CUDA 13.0)
tools/speech/kitten/fetch.sh /data/models/kitten              # 2.08 GB: the packed language model, voices, tokenizer, s3gen
```

Then add the tools from `tools/speech/kitten/config.example.json` to the daemon's config. There are two, one model
each way:
- **`kitten-tts-2`:** `lm_decode: "eager"`, kittenml's own generate, bit for bit the reference. It is **not real
  time on GB10**: RTF 1.5–1.6, host-bound (see [GB10](#gb10)).
- **`kitten-tts-2-graphs`:** `lm_decode: "graphs"`, the decode step as CUDA graphs, made to be the realtime fallback.
  It is the `text_to_speech` default.

Each names its own `qualification` file. The daemon serves a tool only once `qualify.sh --marker` has written that
file on this machine (see [Qualification](#qualification)). `weights` is the directory `fetch.sh` filled; `voices`,
`default_voice`, `presets`, `max_input_chars` and the speed range are what the daemon advertises on `/v1/models` and
checks before loading. The worker checks them against the checkpoint when it loads, and refuses to load if the config
offers a speed range (`min_speed` and `max_speed` must both be 1), a voice or a preset the model lacks.

| Option | Default | |
| --- | --- | --- |
| `device` | `auto` | `cuda` or `cpu` |
| `threads` | torch's | CPU threads |
| `lm_weights` | `packed` | `packed` (ternary, 954 MB, bit-identical to bf16) or `full` (bf16, 3.47 GB); `emb4` needs `allow_lossy` |
| `decoder` | `default` | `student_w4` needs `allow_lossy` |
| `normalize` | `auto` | English normalizer: on, except for the language voices |
| `warm_voices` | the default voice | voices whose conditioning is built at load, not in the first request |
| `verify` | `size` | at load, check each pinned file's size, or its `sha256` (slower), or `none` |
| `keep_float` | false | also write `speech.f32` and `speech.chunks.json` (tokens per chunk), for the gate |
| `lm_decode` | `eager` | `eager`: kittenml's generate (the reference's bits); `graphs`: CUDA-graph decode (CUDA only; its own accepted baseline) |
| `graph_tokens` | 2048 | `graphs`: the static KV cache's length (112 KiB a token, 224 MiB at 2048); a chunk whose prompt and budget exceed it decodes eagerly |
| `graph_backend` | `inductor` | `graphs`: `inductor` (`torch.compile` reduce-overhead) or `cudagraphs` (the eager kernels captured, no Triton) |

The worker never reaches the network: it sets the Hugging Face libraries offline and answers the package's one
download (the s3gen weights from ResembleAI/chatterbox-turbo) with the pinned file.

## Pins

`tools/speech/kitten/pins.json` holds every file's size and sha256; `fetch.sh` and `verify: sha256` check them.

| What | Pin |
| --- | --- |
| Model | KittenML/kitten-tts-2 @ `f2eee74f4467855ad58fe16f1fb7b7df473263fd` |
| Language model | `lm/model-ternary.safetensors`, 954,467,780 B, sha256 `8dd6cc31…ae920054` |
| Voices | `voices/` (47 voices: `.npz` precomputed speaker embedding and reference codec tokens, `.wav` reference clip) |
| Decoder | ResembleAI/chatterbox-turbo @ `749d1c1a46eb10492095d68fbcf55691ccf137cd`, `s3gen_meanflow.safetensors`, sha256 in pins |
| Package | kittenml 0.9.4, wheel sha256 `74b72355…4c8b8c60aa5f`; the worker refuses another version |
| Runtime | Python 3.12, torch 2.11.0 and torchaudio 2.11.0 (`+cpu` on x86_64, `+cu130` on aarch64); every package by hash in `requirements-*.txt` |

## Numerics

The served path is lossless with respect to KittenML's release. The packed language model is the checkpoint's
ternary weights stored five trits a byte with each 128-weight group's scale at full precision, and unpacks to exactly
the bf16 tensors of `lm/model.safetensors`; the decoder is the default S3 flow. KittenML ships two lossy options, and
the worker refuses them unless `allow_lossy` is set: `emb4` quantizes the 324M-parameter embedding to 4 bits (L2
relative error 0.118), and the student decoders are distilled, 4- or 8-bit flows.

`tools/speech/kitten/qualify.py gate` checks the worker against the reference, which is the package's public
`KittenTTS2.generate` loaded with its defaults, bit for bit:

- **ops:** text normalization, chunks and prompt ids; the language model's prefill logits; the S3 flow's mel; the
  HiFT vocoder's waveform. One process loads the model as the reference does, another as the worker does.
- **chained:** per chunk, the sampled codec tokens and the decoded sample count.
- **full:** the joined float32 waveform; the 16-bit PCM in `speech.wav` against the reference waveform quantized the
  same way (clip to ±1, ×32767, round half to even); and a warm repeat of the first request against the cold one.

Fixtures: F1 a short sentence (Bruno), F2 numbers, money, a date and a percentage through the normalizer (Luna), F3
emotion, event and emphasis markup with the expressive preset (Kiki), F4 German with the normalizer off (German), F5
a 640-character paragraph that splits into chunks and joins (Bruno).

Results:

| Target | Fixtures | Result |
| --- | --- | --- |
| DEV: AMD Ryzen 7 6800U, CPU, torch 2.11.0+cpu, 8 threads | F1 | **pass**, 13 of 13 checks bitwise equal: normalized text and prompt ids, prefill logits (158,241 fp32), flow mel, vocoder waveform; tokens and chunk samples; float32 waveform (sha256 `2be41205…`), PCM16 (`c20521fa…`), cold = warm |
| DEV, through the daemon (`/v1/audio/speech` wav and pcm, MCP `synthesize_speech`) | F1 | the same PCM16 bytes (`c20521fa…`) on every path |
| GB10 (dgx02), CUDA, worker with `torch.use_deterministic_algorithms(True)` (36cd15a) | F1 to F5 | **fail**, 9 of 37 checks: op `hift_wav`; full `F1`, `F3`, `F4` and `F1-warm`, each `.f32` and `.wav_pcm16`. Equal: every other op (text, prompts, prefill logits, flow mel), every chained check (tokens, chunk samples), and full `F2` and `F5` |
| GB10 (dgx02), CUDA, worker with the reference's torch settings | F1 to F5 | pending: `tools/speech/kitten/qualify.sh` in an owner-granted window. Until it passes, CUDA is not qualified |

Why the first GB10 run failed, measured in one process per setting on dgx02 (torch 2.11.0+cu130): with torch's
defaults the reference reproduces itself exactly from process to process (HiFT at six token lengths and the whole of
F1). With `torch.use_deterministic_algorithms(True)` or `torch.backends.cudnn.deterministic = True` the flow mel is
unchanged, and the first HiFT module whose output differs is its first ConvTranspose1d upsample (`ups.0`); from there
the waveform differs, at every length tested but one. `CUBLAS_WORKSPACE_CONFIG` alone changes nothing. That the
deterministic setting selects a different cuDNN algorithm for that layer is an inference: the diagnostic recorded
module outputs, not kernel names. The worker therefore runs with the
reference's settings: it has no `deterministic` option, refuses one in the config, and refuses to load if torch is in
deterministic or cuDNN benchmark mode.

`tools/speech/kitten/qualify.sh ROOT VENV OUT` runs it on a target: unit tests, every pinned file's sha256, the gate,
then the bench, each under a time limit. `OUT/status.json` records each step's exit code and `"qualified"`, which is
true only when all four pass. A bench that times out or misses any measurement fails, so partial numbers are never
reported as a result. Memory is sampled every 0.5 s into `memory.log`; the sampler stops before `SHA256SUMS` is taken.

A GPU run is a separate baseline from a CPU run: the gate compares worker and reference on the same device, threads
and seed, never across devices.

### The graphs decode: a named exception

`lm_decode: "graphs"` changes how the language model's decode step runs, not the model. It uses:
- a static KV cache (one per worker, reset between chunks);
- the decode step compiled and captured as CUDA graphs, through transformers' `compile_config`;
- the speaker conditioning computed with the reference hook's own ops before `generate`;
- SDPA attention.

The prefill is not captured, but it also runs over the static cache (all `graph_tokens` slots, an explicit mask), and
the decode kernels differ from eager decode's. So its logits differ in the low bits and the sampled
codec tokens drift from the reference's after some step. That is a deliberate exception to `CONTRIBUTING.md`'s
bit-exact rule. It is made explicit here, so it is not made quietly, and it relaxes nothing else.

| Name | Scope | Still exact | Tolerances against the eager reference | Evidence |
| --- | --- | --- | --- | --- |
| `kitten-tts-2-graphs-decode` | `lm_decode: "graphs"` only: the language model under graphs (the prefill over the static cache with its explicit mask, and the decode step as CUDA graphs), so its logits, codec tokens and audio | **op:** text, prompts, eager prefill logits, flow mel and vocoder, via the shared eager load path. **self:** cold and warm in one process, and across two processes. **baseline:** the accepted bits for this GPU and runtime | each fixture's audio length within 0.75–1.33x of the reference; its longest silence (−45 dBFS, 20 ms frames) at most the reference's + 0.5 s, and at least 1 s allowed | pending: the first accepted GB10 run |

`qualify.py gate --lm-decode graphs` is in `qualify_graphs.py`. It checks all of that and records how many tokens
each fixture matched the reference for (information, not a gate).

**The baseline:**
- `tools/speech/kitten/baselines/graphs-<gpu>.json` holds each fixture's token and PCM sha256 under the runtime that
  made them (torch, transformers, kittenml, CUDA, GPU).
- A baseline that is missing, or recorded under another runtime, fails the gate.
- `--accept-baseline` replaces it with this run's bits. That is the operator's recorded decision, after listening to
  the run's WAVs.
- The new file is written to `OUT/gate/baseline/` to be committed, and the gate report and marker name it.

### Qualification

```sh
tools/speech/kitten/qualify.sh ROOT VENV OUT --lm-decode eager  --marker /data/models/kitten/qualification-eager.json
tools/speech/kitten/qualify.sh ROOT VENV OUT --lm-decode graphs --marker /data/models/kitten/qualification-graphs.json \
  [--graph-backend inductor|cudagraphs] [--graph-tokens 2048] [--accept-baseline]
```

A run writes the marker only when:
- every step passed (unit, sha256 of every pinned file, the gate for that `lm_decode`, a complete bench);
- the full fixture set ran;
- it was a CUDA run.

It writes the marker last, after `status.json` and a complete `SHA256SUMS`, and then takes the manifest again with
`marker.json` in it. A run cut short before that point leaves no marker, and a run whose second manifest fails
removes the one it wrote. A failed run leaves an existing marker alone.

The marker holds:
- `"qualified": true`;
- the `lm_decode`, and for graphs the `graph_backend` and `graph_tokens` it was qualified with (`--graph-backend`,
  `--graph-tokens`);
- the host, GPU and pins' sha256;
- the exceptions used and their evidence;
- the graphs baseline;
- the bench's cold load, warm RTF, warm first chunk and memory.

The daemon reads only `"qualified"` (#14's `qualification` gate). The worker also reads the marker its tool names,
and refuses to load if the marker qualified another `lm_decode` or other graph settings.

Under graphs, a decode step torch cannot capture is an error (`cudagraph_or_error`), and so is any capture skipped
during the load's warm-up. "graphs" never quietly runs uncaptured.

### The kitten-tts-2-cpp runtime

KittenML's [kitten-tts-2-cpp](https://github.com/KittenML/kitten-tts-2-cpp) (MIT, a llama.cpp fork) runs the
language model in GGML and the decoder as TorchScript in CPU LibTorch. Its GGUF stores the same ternary weights
losslessly (`TQ2_1`), but it quantizes activations (Q8_K, or Q8_0 with AMX) and samples with its own RNG, and its
README states that output is not bit-identical to Python and that a seed does not reproduce Python's tokens. It is
therefore not a LocalRouter backend: it cannot pass the gate, and no numerical baseline for it has been accepted. Its
measured speed, for the trade-off:

| Runtime (DEV, F1, Bruno, seed 7, 8 threads, shared box) | Wall | Audio | RTF | Peak RSS | Bits vs reference |
| --- | --- | --- | --- | --- | --- |
| kittenml / torch 2.11 CPU (the worker) | 37.2 s | 3.62 s | 10.3 | 5.5 GB | equal (the gate) |
| kitten-tts-2-cpp `1ce0bb50`, AVX2 (no AMX), default decoder | 13.3 s (LM 5.8 s at 25.3 tokens/s, decoder 4.0 s) | 3.78 s | 2.6 (3.5 incl. load) | 2.04 GiB | not equal: other tokens from the same seed; KittenML's own validation reports prefill logits max abs 0.21 against Python |

The C++ runtime is about 3x faster on this CPU and uses 40% of the memory. Using it would need an accepted numerical
baseline of its own (an explicit, documented decision, not taken here); on GB10 the reference itself runs on the GPU.

## GB10

Eager, measured before this change (DGX-01, 2026-10-21, `tools/spark/jarvis/kitten_probe.py`, the 125-character
calendar sentence, 161 codec tokens, 6.54 s of audio):

| | |
| --- | --- |
| Warm RTF | 1.52–1.64 (20 or 4 CPU threads alike) |
| Language model | 15.6–17.0 tokens/s (real time needs 25): 57–64 ms a token wall, 24.8 ms of it kernels, one CPU core at 100 % |
| Decoder (S3 flow + HiFT) | 0.19–0.31 s, 2–3 % of the wall |
| Memory after load | host RSS 4.44 GiB (3.95 anonymous, live: `malloc_trim` frees nothing), CUDA 4.33 GiB reserved; 10.6 GiB `MemAvailable` drop through the daemon |
| Load | 16–18 s |

So eager decode is host-bound. The graphs decode removes the per-token Python and launch overhead, and the estimate is
about 40 tokens/s (RTF 0.6–0.75). Moving the unpacked weights to the device during load should remove about 3.5 GiB of
host memory. Both are measured by `qualify.sh` (gate and bench, eager and graphs). This section gets the numbers then.

## Measurements

DEV (AMD Ryzen 7 6800U, 16 threads, 27 GB, shared with other jobs: load average 14 to 30), CPU, 8 threads:

| | |
| --- | --- |
| Worker load (process start to ready: imports, model build, normalizer, default voice) | 33 to 87 s (two runs) |
| Resident after load (reported to the daemon) | 5.51 GB (bf16 language model 3.5 GB, s3gen fp32 about 1 GB, torch) |
| F1 (3.62 s of speech), warm | 34.7 to 44 s: RTF about 10. Time to first audio is the whole chunk: audio arrives per chunk |
| F1 cold through the daemon (includes the load) | 90.9 s |
| Cancel (MCP `cancel_job` mid-generation) | the request ends 409 `cancelled` in 1.0 s; the worker process is gone 1.08 s after the cancel; MemAvailable returns 4.68 GB above its loaded level; no files left in the job |
| Unload (`POST /v1/tools/kitten-tts-2/unload`) | worker gone in 0.82 s |

CPU torch is a correctness baseline, not a serving target: at RTF 10 it is too slow to speak to anyone. The GB10 numbers
(cold load, first audio, warm RTF for F1 and the 640-character F5, peak RSS and CUDA memory, cancel reclaim) come
from `qualify.sh` (gate and bench) on a GB10: see [GB10](#gb10).

## Lifecycle

Load reads the pinned files, builds the model on the device, builds the normalizer's grammars and the default voice's
decoder conditioning, and replies with the measured resident bytes (host RSS plus torch's reserved GPU memory). A
generate writes `speech.wav` into the job directory; audio never travels in the protocol's JSON lines. Cancelling a
job or a deadline kills the worker's process group, as for every tool; the worker starts no processes of its own, and
leaves no file but the finished `speech.wav` (written once, at the end). Unload is process exit. The worker exits
when the daemon that started it is gone.

## Licences

| Part | Licence |
| --- | --- |
| Kitten TTS 2 weights, voices, config (KittenML / Stellon Labs) | [Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/f2eee74f4467855ad58fe16f1fb7b7df473263fd/LICENSE.md) |
| `speaker/` (pyannote speaker embedding; fetched, unused for built-in voices) | MIT |
| kittenml package | Apache 2.0 |
| s3gen decoder weights (ResembleAI chatterbox-turbo) | MIT (model card) |
| kitten-tts-2-cpp (measured only, not shipped) | MIT |

LocalRouter does not distribute the weights: `fetch.sh` downloads them from their publishers. The Stellon Labs
licence still binds whoever uses them. It allows research, non-commercial and commercial use free of charge, but:

- commercial use requires registering with Stellon Labs, and every licence under it ends once the user or its
  affiliates pass USD 1,000,000 in annual revenue or in total funding, after which a separate licence is needed;
- distributing the weights, a derivative, or a product or service that uses them requires a copy of the licence,
  the notice "This Stellon Labs Model is licensed under the Stellon Labs Community License, Copyright © Stellon Labs
  Inc. All Rights Reserved" in a NOTICE file, and "Powered by Stellon Labs" shown prominently in documentation;
- outputs may not be used to train or distill another model (other than fine-tunes and LoRAs of this one).

Whether a given deployment is commercial, is under the threshold, and is registered is the operator's decision and is
not settled by this code. Nothing here accepts those terms or registers on anyone's behalf.
