# Chat: Qwen3.8 27B on TensorFold

LocalRouter is one "Jarvis box" on a single GB10: a chat model that is always warm and does most of the work, with
image and video generation and speech to text and text to speech beside it. The chat model is
[TensorFold/Qwen3.8-27B-MLX-4bit](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit) with the
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) drafter, served by
[TensorFold](https://github.com/ashhart/TensorFold) 1.1.1's own OpenAI-compatible server (`tensorfold serve`, the
release binary `bin/tensorfold-native`). LocalRouter does not reimplement the model: it runs TensorFold's server as the
tool's worker and proxies requests to it.

**Status: qualified on DGX-01 (GB10)** by `tools/chat/qwen38-27b/qualify.sh` run2 at commit 685f517 on
2026-10-21: 273 of 273 checks, with one named exception (see [Exceptions](#exceptions)). TensorFold's own
qualification table lists the 27B only on Metal and an RTX 4090; this run establishes it on GB10 (aarch64 CUDA). Every
other machine runs qualify.sh itself: the tool is refused (503) on a host until its own marker says qualified.

| Run | Commit | Checks | Result |
| --- | --- | --- | --- |
| run1 | 6e83844 | 269 of 273 | not qualified: P1 and P2 stream vs whole `reasoning_content`, TensorFold's trailing `\n` (direct and through LocalRouter alike) |
| run2 | 685f517 | 273 of 273 | qualified, with `tensorfold-1.1.1-stream-reasoning-trailing-newline` used by exactly those four checks (599 vs 598 B, 352 vs 351 B, tail `\n`). Memory drop 20.0 GiB (configured 24 GiB); cold load 12.5 s; servers stopped on SIGTERM in 0.8 s |

Marker (sha256 `f64c573b5341c50d7d5fc4474c4b6e84df94d71eabe6c1d5ffd9f20748c5a61e`): `qualified: true`, host
`dgx-01`, GPU `NVIDIA GB10`, `pins_sha256` `3be3ea41…`, `tensorfold_sha256` `2e0103a5…`, 273 checks, `exceptions:
["tensorfold-1.1.1-stream-reasoning-trailing-newline"]`. Receipts on DGX-01 in `~/jarvis/qualify-chat/run2/` (manifest
`SHA256SUMS` `2c1f290d…`).

## What it serves

| | |
| --- | --- |
| Route | `POST /v1/chat/completions`, whole or streamed (`"stream": true`, server-sent events); MCP `chat` |
| Request | OpenAI's chat body. LocalRouter checks it is a JSON object with a non-empty `messages` array and a boolean `stream`, sets `model` to the tool id, and passes every other field through |
| Passed through | `messages` (system, user, assistant with `tool_calls`, tool), `tools`, `tool_choice`, `parallel_tool_calls`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `max_tokens`, `max_completion_tokens`, `stop`, `response_format` (JSON object or schema), `stream_options.include_usage`, `reasoning_effort`, `chat_template_kwargs` (`enable_thinking`), `thinking_budget` |
| Reasoning | on by default (the template's `xhigh`); `"reasoning_effort": "none"` or `"chat_template_kwargs": {"enable_thinking": false}` turns it off per request (MCP: `"reasoning": false`). The thinking arrives as `reasoning_content` |
| Reply | the server's own JSON or event stream, byte for byte (`data: [DONE]` ends a stream) |
| Refused by the server (400) | `n` other than 1, image/audio/video input, `logprobs`, malformed sampling fields, a bad `tool_choice`, a context overflow (`context_length_exceeded`) |

Errors keep LocalRouter's mapping: the server's 4xx (but 401/403) is a 400 with its message, its 503 (busy, draining)
a 503, anything else a 500. A server that is gone mid-request is a 503 and the next request starts it again. A stream
refused before its first event gets the plain JSON error; one that breaks off later ends with
`data: {"error": {...}}` and `data: [DONE]`.

## How it runs

The tool uses the compiled-in `openai_server` engine (`src/engines/openai_server.zig`; the protocol is in
[ARCHITECTURE.md](../ARCHITECTURE.md#a-models-own-server-openai_server)):

1. **Load.** The scheduler admits the model by `resident_bytes` against the budget, like every other model, and holds
   the host's GPU lock for LocalRouter as a whole ([The GPU lock](#the-gpu-lock)); the worker takes no lock. The
   worker picks a free loopback port, starts
   `tensorfold-native serve ... --host 127.0.0.1 --port {port} --name {id}` in its own process group with the server's
   output in the tool's log (`<data>/logs/qwen3.8-27b.log`), and polls `GET /health` every 250 ms until it answers 200
   (the server exiting first fails the load with its status). The load reply names the port.
2. **Requests.** The daemon POSTs each request to `127.0.0.1:{port}/v1/chat/completions` on its scheduler thread, one
   at a time like every LocalRouter job (the server runs `--parallel 1`), and writes the reply into the job directory
   as it arrives (`completion.json`, or `completion.sse` that the route forwards while it grows).
3. **Cancel.** A client that closes a stream, or `cancel_job`, closes that request's connection. The worker, the
   weights and the prompt cache stay loaded.
4. **Unload.** SIGTERM (TensorFold drains for `TF_DRAIN_S`, set to 5 s in the example), SIGKILL after
   `stop_timeout_s` (8 s). If the daemon dies, the worker kills the server before it exits.

The server reads the checkpoint and drafter from the Hugging Face cache that `fetch.sh` fills
(`HF_HUB_CACHE=/data/models/chat/hf`, `HF_HUB_OFFLINE=1`), pinned to their revisions by `refs/main`.

## Memory

| | |
| --- | --- |
| `resident_bytes` | **25,769,803,776 (24 GiB)**: the measured 21.1 GiB peak plus room for the 4 GiB prompt cache |
| `working_bytes` | 0: the server sizes its KV and prompt cache at start, inside the resident figure |
| Server flags | `--parallel 1 --prompt-cache-gib 4 --context 32768`, drafter on: one user at a time, as LocalRouter runs one job at a time |

Measured on DGX-01 (GB10) by the bring-up, `docs/dev/JARVIS-BRINGUP.md` (PR #16), TensorFold 1.1.1 at the pinned
archive:

| Setting | Resident | Speed |
| --- | --- | --- |
| `--parallel 1 --prompt-cache-gib 2`, drafter | 19.2 GiB; 21.1 GiB with a 30k-token prompt | 29.9 s cold load; about 40 tok/s chat, 75 tok/s code; 1.5 s to first token at 1k tokens |
| TensorFold's defaults (8 parallel, 8 GiB prompt cache), drafter | 38.4 GiB | the same speed |
| `--parallel 1`, no drafter | 35.6 GiB | 13.3 tok/s |

A 30k-token prompt takes 120 s to its first token; a cached 20k-token conversation's next turn 0.61 s, and a 30k
conversation needs about 2 GiB of the prompt cache, so 4 GiB keeps one long conversation and its neighbours warm.

The scheduler cannot see a model server's memory: it plans with `resident_bytes` while the tool is loaded, so the
figure must cover the server's peak. TensorFold's CUDA server also sizes its window from `MemAvailable` less a reserve;
`--context 32768`, `--parallel 1` and `--prompt-cache-gib 4` bound it. `qualify.sh` samples `MemAvailable` every 0.5 s
during both of its server runs and reports the largest drop in `report.json`.

## Expected speed

Measured on DGX-01 (GB10) on 2026-10-22 against the live Jarvis service, with `tools/spark/jarvis/chat_speed.py`.
It sends the web client's request shape (its system prompt, the image and video tools, thinking off, no temperature)
and reads TensorFold's own `tensorfold` and `speculative` figures. Each user turn has a fresh nonce, so only the shared
prefix comes from the prompt cache.

| State | Prose (client request) | Code | TTFT |
| --- | --- | --- | --- |
| Chat with Drama 3, Whisper and Kitten loaded but idle (the GPU is at 0% while they idle, so this is also chat alone) | 34–39 tok/s, 3.2–3.4 tokens a drafter round | 70–87 tok/s, 6.7–7.6 a round | 0.14–0.16 s |
| The same through the daemon (`/v1/chat/completions` on :8190), identical greedy requests | 39.0 vs 38.9 tok/s direct | 86.5 vs 86.9 | the same |
| While Drama 3 synthesizes | 14–19 tok/s, 2.8–3.7 a round | 34–38 tok/s | 0.32–0.55 s (1.4 s once) |

- **Prose against code.** A drafter round costs about 90 ms either way. DFlash2's drafts are accepted about 3.3
  tokens a round on prose and about 7.5 on code. Qualification's 83 tok/s came from short, predictable answers; open
  prose runs at about 35–39 tok/s, and that is the model and drafter's lossless speed on this GPU. Without the drafter it
  is 13 tok/s (JARVIS-BRINGUP.md). TensorFold's CUDA engine has no other drafting setting, and it refuses a bf16
  drafter (`--drafter-bits 0`).
- **Sampling.** The client sends no temperature, so the server samples (`"sampling": "exact"`): about 8% slower than
  greedy (`temperature: 0`), which accepts more drafts. The client keeps sampling.
- **The prompt cache works.** The client's system prompt and tools (384 of about 480 prompt tokens) and a voice
  conversation's history (512–640) come from the cache on every turn. The voice loop's one-token primes keep it warm.
- **The daemon costs nothing measurable.** Thinking is off by default in the client, which sets no `max_tokens` (the
  server's 4096). Typed chat never calls text to speech: the client speaks only in voice mode and for voice previews.
- **Contention.** Two processes on the GPU time-slice it; TensorFold exposes no priority between processes. While
  Drama 3 synthesizes, chat runs at about 40% of its speed (drafter acceptance is unchanged), and Drama 3 slows too:
  its real-time factor goes from 0.66 alone to 1.47–1.52 while chat decodes, that is, slower than playback. In a voice
  turn chat is still about four times faster than speech. What suffers is the audio, while the two overlap: a long reply
  decoding while the first sentences are spoken can starve playback. Typed chat is slowed only when something else
  (a voice turn, a test batch) is using the GPU at the same time.

## Priority and eviction

A chat tool defaults to `keep_loaded: true` and `priority: 100` (every other tool defaults to 0): it is loaded when the
daemon starts, never unloaded for idleness, unloaded last when a request needs memory, and loaded again on its own once
the memory is free. At its measured 24 GiB, the chat model and the FP8 image model (about 40 GiB) fit side by side
in GB10's 128 GB with the default 8 GiB reserve; a MiniMax H3 video job, which needs most of the machine, evicts the
image model first and then the chat model, and the chat model comes back after the video model idles out.

The box's voice models sit beside it the same way: chat (priority 100), text to speech and speech to text (give them
priorities below 100, e.g. 50 and 10, and `keep_loaded`) are loaded together whenever their `resident_bytes` fit the
budget, and a job that needs the machine evicts them lowest priority first: transcription, then speech, then chat.
They all come back by themselves once it is done. `tests/jarvis.zig` checks both: the three warm together under one
GPU lock, serving in turn without a reload, and the eviction order and return.

## The GPU lock

LocalRouter's scheduler is the one admission authority on the box: it decides which models are loaded together from
their `resident_bytes` against the budget, nothing else. `gpu_lock` only keeps GPU users outside LocalRouter out (they
take the file exclusively): the scheduler takes it once, `flock(LOCK_SH | LOCK_NB)`, when the first model loads, holds
it while any model is loaded or loading, and releases it when none is. If an outside job holds it, nothing loads; the
request (or explicit load, or `keep_loaded` start) fails with a 503 naming the lock, and `keep_loaded` models retry
after a minute.

No LocalRouter worker takes the host lock. Each worker's environment says so: `LOCALROUTER_GPU_LOCK_HELD=1`, the host
lock's path in `LOCALROUTER_HOST_GPU_LOCK`, and in `LOCALROUTER_GPU_LOCK` a lock file of its own,
`<data>/locks/<id>.lock` ('/' in the id becomes '_'), created and uncontended. A launcher that insists on taking a lock
gets that file: Drama 3's worker (PR #13) passes `LOCALROUTER_GPU_LOCK` to `pilot.py --gpu-lock` unchanged, so its
exclusive, non-blocking take succeeds and never keeps the chat model, or any other LocalRouter model, out. Run outside
LocalRouter, the same launcher is given the host lock and guards the machine as before.

## Setup

```sh
tools/chat/qwen38-27b/fetch.sh /opt/localrouter/chat --hf /data/models/chat/hf   # TensorFold 1.1.1, checkpoint, drafter (~20 GB)
tools/chat/qwen38-27b/setup.sh /opt/localrouter/chat --hf /data/models/chat/hf   # runs the binary here; prints the tool config
```

Then merge the printed tool (as in `tools/chat/qwen38-27b/config.example.json`) into the daemon's config, with
`gpu_lock` set to the lock every GPU job on the host already takes, stop the daemon, and run the qualify step on the
GB10 (it refuses another host, a server already running, or under 64 GiB `MemAvailable`):

```sh
tools/chat/qwen38-27b/qualify.sh /opt/localrouter/chat /opt/localrouter/bin/localrouter /tmp/qwen38-qualify-1 \
  /data/models/chat/qwen38-27b-qualification.json
```

Restart the daemon: the tool is served once the marker says `"qualified": true`. See `tools/chat/qwen38-27b/README.md`.

## Qualification

The tool's config names `qualification: /data/models/chat/qwen38-27b-qualification.json`. Until that file says
`"qualified": true`, `/v1/models` lists the model with `"qualified": false`, and every request, explicit load and
`keep_loaded` start is refused with a 503 naming the file. `qualify.sh` writes it only when every step passes:

1. **Pins:** the TensorFold archive and every checkpoint and drafter file against `pins.json` (sha256).
2. **Direct:** TensorFold's server started with exactly the tool's command line and environment; the fixed prompts
   (`prompts.json`: a short answer, reasoning on, reasoning off, a required tool call, a tool result round trip, a JSON
   schema, `stop` with `max_tokens`, a long multi-turn prompt) at `temperature: 0`, `seed: 7`, each whole twice and
   streamed once.
3. **Through LocalRouter:** a separate daemon (port 0, its own data directory, the tool without the gate) serving the
   same prompts whole and streamed through `/v1/chat/completions`.
4. **Compare:** per prompt, the two direct runs are equal (deterministic tokens at temperature 0), LocalRouter's
   reply equals TensorFold's (content, reasoning, tool calls, finish reason, completion tokens), each stream's joined
   deltas equal its whole reply, and each stream ends with `data: [DONE]`.
5. **Marker:** `{"qualified": true, "host", "gpu", "pins_sha256", "tensorfold_sha256", "checks", "report",
   "qualified_at"}`, written atomically, plus `exceptions` (the names of the exceptions below that the run used, `[]`
   when none) and `exception_evidence`.

The daemon checks only `"qualified": true`; the marker's other fields record what was qualified. A new pin (TensorFold,
checkpoint or drafter) needs a new run.

### Exceptions

Every comparison is exact, with one named exception, defined in `qualify.py` (`EXCEPTIONS`) and listed in the marker
when a run used it:

| Name | `tensorfold-1.1.1-stream-reasoning-trailing-newline` |
| --- | --- |
| Scope | `reasoning_content` only; a stream against the whole reply of the same server only (direct stream vs direct whole, LocalRouter stream vs LocalRouter whole); the joined stream may be the whole reply's text followed by whitespace, nothing else. Pinned TensorFold 1.1.1 only |
| Still exact | tokens (`completion_tokens`), `content`, `tool_calls`, `finish_reason`, determinism (two whole runs), LocalRouter against TensorFold, and every other field |
| Evidence | DGX-01 run1 (commit 6e83844): P1 stream 599 B vs whole 598 B, P2 352 B vs 351 B. The stream's last reasoning delta ends with `\n` (`"…Canberra.\"\n"`) that the whole reply strips; the bytes are the same direct and through LocalRouter, so it is TensorFold's, not the proxy's |
| Upstream | drafted, not posted: `tools/chat/qwen38-27b/upstream-issue-stream-reasoning-newline.md` |

LocalRouter passes both forms through unchanged: a streaming client gets the newline, a whole reply does not. Any
other mismatch, including any difference inside the reasoning text, fails the run.

## Pins

`tools/chat/qwen38-27b/pins.json` holds every hash; `fetch.sh` refuses anything else.

| What | Pin |
| --- | --- |
| TensorFold | v1.1.1, tag `92d71a3c`, commit `b10231b94f8adf3d4132acebdc0a1ac87e53a838`; `tensorfold-1.1.1-linux-aarch64.tar.gz` sha256 `2e0103a5…f9139cb39`, `-linux-x86_64` `e1785e32…6185fbdf` |
| Checkpoint | TensorFold/Qwen3.8-27B-MLX-4bit @ `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c` (4-bit affine, group 64; three shards, 16,054,541,349 B) |
| Drafter | z-lab/Qwen3.8-27B-DFlash2 @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (BF16, 3,848,817,896 B; served in 4 bits, `--drafter-bits` default) |

## Known interactions

- **PR #13 merge.** #13 sets `LOCALROUTER_GPU_LOCK` to the host lock in `process.zig`; this PR replaces that line
  with the environment above. Take this PR's version when the two meet. Drama 3's own memory checks (its launcher wants
  88 GiB of `MemAvailable` before it starts) still apply beside the warm chat model; its `resident_bytes` (80 GiB) makes
  the scheduler free that room first.
- **One request at a time.** Chat requests queue behind image, video and speech jobs like any other job, and vice
  versa; a long reply holds the queue for its length (`max_tokens`, 4096 by default).

## Licences

| Part | Licence |
| --- | --- |
| TensorFold 1.1.1 (binary, fetched) | Apache 2.0 |
| Qwen3.8-27B-MLX-4bit (TensorFold; base Qwen/Qwen3.8-27B) | Apache 2.0 |
| Qwen3.8-27B-DFlash2 (z-lab) | Apache 2.0 |

LocalRouter does not distribute the binary or the weights: `fetch.sh` downloads them from their publishers.
