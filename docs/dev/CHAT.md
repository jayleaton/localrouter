# Chat: Qwen3.8 27B on TensorFold

LocalRouter is one "Jarvis box" on a single GB10: a chat model that is always warm and does most of the work, with
image and video generation and speech to text and text to speech beside it. The chat model is
[TensorFold/Qwen3.8-27B-MLX-4bit](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit) with the
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) drafter, served by
[TensorFold](https://github.com/ashhart/TensorFold) 1.1.1's own OpenAI-compatible server (`tensorfold serve`, the
release binary `bin/tensorfold-native`). LocalRouter does not reimplement the model: it runs TensorFold's server as the
tool's worker and proxies requests to it.

**Status: qualified on DGX-01 (GB10)** by `tools/chat/qwen38-27b/qualify.sh`: TensorFold 1.1.1 by run2 at commit
685f517 on 2026-10-21, and 1.1.4 by the run at commit 37755e0 on 2026-10-25, each 273 of 273 checks with one named
exception (see [Exceptions](#exceptions)). The pin is 1.1.1: 1.1.4 was deployed on 2026-10-25 and taken back the same
day (see [Why not 1.1.4](#why-not-114-yet)). The deployed marker comes from the run on the deployed commit. Every
other machine runs qualify.sh itself: the tool is refused (503) on a host until its own marker says qualified.

| Run | Commit | Checks | Result |
| --- | --- | --- | --- |
| run1 | 6e83844 | 269 of 273 | not qualified: P1 and P2 stream vs whole `reasoning_content`, TensorFold's trailing `\n` (direct and through LocalRouter alike) |
| run2 | 685f517 | 273 of 273 | qualified, with `tensorfold-1.1.1-stream-reasoning-trailing-newline` used by exactly those four checks (599 vs 598 B, 352 vs 351 B, tail `\n`). Memory drop 20.0 GiB (configured 24 GiB); cold load 12.5 s; servers stopped on SIGTERM in 0.8 s |
| 1.1.4 | 37755e0 | 269 of 273, then 273 of 273 | TensorFold 1.1.4: the same four checks, the same bytes (599 vs 598 B, 352 vs 351 B, tail `\n`): the newline is still upstream's, so the exception (renamed `tensorfold-stream-reasoning-trailing-newline`) lists 1.1.4 beside 1.1.1, and compare passes with it. Every prompt's greedy reply (content, reasoning, tool calls, finish reason, token count) equals run2's on 1.1.1, 8 of 8. Memory drop 20.8 GiB |

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

Measured on DGX-01 (GB10) by the bring-up, `docs/dev/JARVIS-BRINGUP.md` (PR #16), TensorFold 1.1.1 (the pin then) at its
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

**TensorFold 1.1.4 against 1.1.1** on DGX-01 (GB10), 2026-10-25: the same checkpoint, drafter and flags (`--context
32768 --parallel 1 --prompt-cache-gib 4`, drafter on), one server at a time in the order 1.1.1, 1.1.4, 1.1.4, 1.1.1,
beside the live service with its image and decision models unloaded and no other GPU process. `chat_bench.py`: fresh
prompts (a new nonce each), greedy; the footprint is the server process's nvidia-smi memory plus RssAnon and RssShmem
at 0.5 s.

| | 1.1.1 | 1.1.4 | |
| --- | --- | --- | --- |
| TTFT, 2.1k-token prompt | 3.09–3.12 s (676 tok/s) | 1.90–1.91 s (1,102 tok/s) | 1.63x |
| TTFT, 8.1k | 15.04–15.08 s (538 tok/s) | 7.36–7.38 s (1,098 tok/s) | 2.04x |
| TTFT, 30.1k | 125.4–125.8 s (240 tok/s) | 29.38–29.39 s (1,024 tok/s) | 4.27x |
| Decode, 512 tokens (chat and code, 3 runs each a server start: 12) | 37.3–47.7 tok/s, mean 44.4 | 47.5–52.9 tok/s, mean 49.6 | +12% |
| Load (warm page cache) | 13.7–19.6 s | 14.3 s (92.5 s at its first start on the box) | |
| Peak footprint, with the 30k prompt | 22.7 GiB (18.2 GiB of it device) | 22.6 GiB (19.4 GiB device) | the same |

Both versions answer every fixed prompt of the qualification with the same bytes (above). A new executable misses
TensorFold's cache of measured decode-window costs (keyed by the executable) and measures them once, which is the
likely cause of 1.1.4's slow first start; later starts took 14 s.

### Why not 1.1.4 yet

The table above has only fresh prompts. A conversation's next turn is the voice loop's every turn, and there 1.1.4 is
slower: its prompt cache resumes only at its 4,096-row prompt chunks, so a conversation shorter than 4,096 tokens is
prefilled again from the start each turn. On DGX-01 (1.1.4 deployed, `chat_bench.py turn`): a 1,569-token
conversation's next turn had `cached_tokens` 0 and took 1.46 s to its first token (1.1.1 cut prompts in 128-row chunks, per
upstream's c50b59a2); at 5,084 tokens, `cached_tokens` was exactly 4,096. In the voice smoke the chat's first token came
1.45 to 1.70 s after each turn's request, against 0.1 to 0.3 s on 1.1.1, and every voice latency check failed (first
audio 2.3 to 2.5 s). The pin went back to 1.1.1 the same day. Before a later TensorFold moves it, `chat_bench.py turn`
(a cached next turn) must stay at 1.1.1's speed as well as the fresh prompts.

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

| Name | `tensorfold-stream-reasoning-trailing-newline` (until 2026-10-25: `tensorfold-1.1.1-stream-reasoning-trailing-newline`) |
| --- | --- |
| Scope | `reasoning_content` only; a stream against the whole reply of the same server only (direct stream vs direct whole, LocalRouter stream vs LocalRouter whole); the joined stream may be the whole reply's text followed by whitespace, nothing else. TensorFold 1.1.1 and 1.1.4 only (the versions it was observed on) |
| Still exact | tokens (`completion_tokens`), `content`, `tool_calls`, `finish_reason`, determinism (two whole runs), LocalRouter against TensorFold, and every other field |
| Evidence | DGX-01 run1 (commit 6e83844, 1.1.1) and the 1.1.4 run (commit 37755e0): P1 stream 599 B vs whole 598 B, P2 352 B vs 351 B, on both. The stream's last reasoning delta ends with `\n` (`"…Canberra.\"\n"`) that the whole reply strips; the bytes are the same direct and through LocalRouter, so it is TensorFold's, not the proxy's |
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

## macOS (Apple Silicon)

The same tool on a Mac: TensorFold **1.1.4**'s `macos-arm64` release (native Zig/Metal server, no Python or MLX
install) serving the same checkpoint and drafter revisions as the Spark, behind the same `openai_server` engine, with
the same flags (`--context 32768 --parallel 1 --prompt-cache-gib 4`, drafter on). Files:
`tools/chat/qwen38-27b/{pins-macos.json, fetch-macos.sh, setup-macos.sh, config.macos.example.json, bench_macos.py,
baseline-macos.json}`; the Spark files are unchanged and the Spark stays on 1.1.1. The Mac config has no
`qualification` marker: that gate and its receipts are the GB10's, and `bench_macos.py` is the Mac's qualification.

**Mac reference baseline.** Metal arithmetic is not CUDA arithmetic, so the Mac is a deliberate numerical change
(CONTRIBUTING.md): no bit parity with the Spark is claimed. `baseline-macos.json` holds this Mac's greedy replies to
prompts.json (temperature 0, seed 7: content, reasoning, tool calls, finish reason, token count), and every
`bench_macos.py` run on 1.1.4 compares against it. The replies with and without the drafter are identical, 8 of 8.
They also equal DGX-01's qualification replies (`direct.json` of run2 on 1.1.1 and of the 1.1.4 run), 8 of 8 on
every compared field: agreement on these short greedy replies, not a bit-parity claim for logits or longer outputs.

Measured 2026-10-30 on a Mac Studio, Apple M5 Ultra, 96 GiB, macOS 27.2, `bench_macos.py serve` (one server under
the host's GPU lock, nothing else loaded), greedy, reasoning off for speed, every speed prompt cold (a unique prefix).
Prompt sizes are the server's `prompt_tokens`. Decode tok/s is (tokens - 1) / (wall - TTFT) of a stream.

| | Drafter (DFlash2) | No drafter |
| --- | --- | --- |
| Load to `/health` | 15.2 s (first load after download); 3.0 s warm | 3.0 s |
| `phys_footprint` loaded | 20.8 GiB | 19.0 GiB |
| Peak `phys_footprint` (rusage lifetime maximum) | **25.6 GiB** | 24.1 GiB |
| Decode, code, 512 tokens | **196 tok/s** | 47.3 tok/s |
| Decode, document summary, 256 tokens at 2.5k / 9.5k / 31.2k-token prompts | 148 / 127 / 104 tok/s | 47.0 / 45.0 / 42.4 tok/s |
| TTFT at 2.5k / 9.5k / 31.2k tokens (prefill) | 2.05 / 7.75 / 27.3 s (1,222 / 1,226 / 1,143 tok/s) | 1.96 / 7.63 / 26.9 s (1,274 / 1,246 / 1,160 tok/s) |
| TTFT, short prompt (74 tokens) | 0.08 s | 0.08 s |
| Next turn of a 2.1k-token conversation | **0.066 s**, `cached_tokens` 2,086 of 2,115 (cold 2.6k: 2.06 s) | 0.065 s, the same |
| Next turn of a 9.7k-token conversation | 0.071 s, 9,627 of 9,656 cached (cold 11.4k: 9.35 s) | 0.069 s, the same |
| SIGTERM to exit | 0.69 s | 0.71 s |

- **Features:** prompts.json whole twice and streamed once: required tool call (`get_weather` with JSON arguments),
  tool result, `response_format` JSON schema (via `lib/libtfgrammar.dylib`), reasoning on and off, stop and
  `max_tokens`, SSE ending in `data: [DONE]`; whole = whole, stream = whole and = the baseline: 188 of 188 checks
  (the drafter run, before the baseline existed, 148 of 148). The `tensorfold-stream-reasoning-trailing-newline`
  exception is used on Metal by the same checks with the same bytes as on the GB10 (P1 599 vs 598 B, P2 352 vs 351 B).
- **The cached next turn works on Metal.** The 4,096-row resume of 1.1.4 on CUDA ([Why not 1.1.4](#why-not-114-yet))
  does not show here: a 2k-token conversation's next turn takes all but its last 29 tokens from the cache.
- **Speed against upstream.** TensorFold's recipe (M5 Ultra, macOS 27, 100 greedy tokens) reports plain decode 51.6 /
  49.4 / 46.2 tok/s at 2k / 8k / 32k and prefill 1,325 / 1,269 / 1,171 tok/s; this host is within 10%. The drafter is
  2.5 to 4.1 times plain decode here (4.1 on code, 2.5 at a 31k-token prompt), against about 3 on the GB10 (prose 35–39 against 13 tok/s).
- **Memory.** `resident_bytes` in `config.macos.example.json` is **30 GiB** (32,212,254,720): the 25.6 GiB peak plus
  room for the rest of the 4 GiB prompt cache; `working_bytes` 0 as on the Spark. Unified memory is counted once: the
  footprint includes the Metal buffers. The research estimate was 32 GiB; TensorFold's published Mac figure 26.5 GiB
  (M5 Max).
- **Through LocalRouter** (`bench_macos.py daemon`: `localrouter serve` built on this Mac from `feat/mac-foundation`,
  the tool alone): `/v1/models` lists it as chat; MCP `load_model` loads it in 3.9 s (warm page cache) and MCP `chat`
  answers in 1.9 s; through `/v1/chat/completions` 188 of 188 checks, the same replies as the baseline; decode 198 /
  153 / 127 / 105 tok/s (code; 2.5k / 9.5k / 31.2k prompts), TTFT 2.02 / 7.72 / 27.3 s, cached next turns 0.065 s
  (2.1k) and 0.105 s (9.7k); peak `phys_footprint` 26.0 GiB. The daemon adds nothing measurable. After SIGTERM the
  model server was gone, but that run's daemon did not exit within 15 s and was killed: on macOS, `shutdown()` of the
  listening socket does not wake a blocked `accept()`. The daemon now connects to itself on macOS instead
  (`feat/mac-foundation` 64bcb59, checked by `tests/orphans.zig`); the chat daemon run has not been repeated with it.
- **A smaller prompt cache.** `setup-macos.sh ROOT --prompt-cache-gib 1` writes the same tool with `--prompt-cache-gib
  1` and `resident_bytes` **27 GiB** (28,991,029,248), for a Mac that keeps chat, Drama 3 and Whisper warm together
  while image and decisions load on demand. 27 GiB is an **estimate**, not a measurement: the 4 GiB configuration
  peaked at 25.6 GiB with at most 4 GiB of cache. A 30k-token conversation needs about 2 GiB of cache (above), so 1
  GiB keeps short and mid-length conversations warm but not a long one, whose next turn may be prefilled again. The
  default stays 4 GiB / 30 GiB until `bench_macos.py serve` measures the 1 GiB footprint and cached-turn latency.

## Licences

| Part | Licence |
| --- | --- |
| TensorFold 1.1.1 (binary, fetched) | Apache 2.0 |
| Qwen3.8-27B-MLX-4bit (TensorFold; base Qwen/Qwen3.8-27B) | Apache 2.0 |
| Qwen3.8-27B-DFlash2 (z-lab) | Apache 2.0 |

LocalRouter does not distribute the binary or the weights: `fetch.sh` downloads them from their publishers.
