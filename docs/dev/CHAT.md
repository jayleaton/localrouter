# Chat: Qwen3.8 27B on TensorFold

LocalRouter is one "Jarvis box" on a single GB10: a chat model that is always warm and does most of the work, with
image and video generation and speech to text and text to speech beside it. The chat model is
[TensorFold/Qwen3.8-27B-MLX-4bit](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit) with the
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) drafter, served by
[TensorFold](https://github.com/ashhart/TensorFold) 1.1.1's own OpenAI-compatible server (`tensorfold serve`, the
release binary `bin/tensorfold-native`). LocalRouter does not reimplement the model: it runs TensorFold's server as the
tool's worker and proxies requests to it.

**Status: not qualified on GB10.** Nothing in this page has run on a GPU yet. The tool is refused (503) until
`tools/chat/qwen38-27b/qualify.sh` passes on the machine, and its memory figure is a placeholder (see
[Memory](#memory)). TensorFold's own qualification table lists the 27B on Metal and on an RTX 4090, not on GB10
(aarch64 CUDA); the qualify run is what establishes it there.

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

1. **Load.** The worker takes the host's GPU lock (`gpu_lock` in the config, `LOCALROUTER_GPU_LOCK` in its
   environment) with `flock(LOCK_EX | LOCK_NB)`: another holder fails the load. It picks a free loopback port, starts
   `tensorfold-native serve ... --host 127.0.0.1 --port {port} --name {id}` in its own process group with the server's
   output in the tool's log (`<data>/logs/qwen3.8-27b.log`), and polls `GET /health` every 250 ms until it answers 200
   (the server exiting first fails the load with its status). The load reply names the port.
2. **Requests.** The daemon POSTs each request to `127.0.0.1:{port}/v1/chat/completions` on its scheduler thread, one
   at a time like every LocalRouter job (the server runs `--parallel 1`), and writes the reply into the job directory
   as it arrives (`completion.json`, or `completion.sse` that the route forwards while it grows).
3. **Cancel.** A client that closes a stream, or `cancel_job`, closes that request's connection. The worker, the
   weights and the prompt cache stay loaded.
4. **Unload.** SIGTERM (TensorFold drains for `TF_DRAIN_S`, set to 5 s in the example), SIGKILL after
   `stop_timeout_s` (8 s), then the lock is released. If the daemon dies, the worker kills the server before it exits.

The server reads the checkpoint and drafter from the Hugging Face cache that `fetch.sh` fills
(`HF_HUB_CACHE=/data/models/chat/hf`, `HF_HUB_OFFLINE=1`), pinned to their revisions by `refs/main`.

## Memory

| | |
| --- | --- |
| `resident_bytes` | **51,539,607,552 (48 GiB): a placeholder, not a measurement.** 40 GiB estimated (weights 14.95 GiB, the 4-bit drafter, an 8 GiB prompt cache, the 32,768-token context's KV, the CUDA runtime) plus 8 GiB for unmeasured GB10 overhead |
| `working_bytes` | 0: the server sizes its KV and prompt cache at start, inside the resident figure |
| Source of the real figure | the DGX-01 bring-up (in progress). Replace `resident_bytes` with its measured peak (MemAvailable drop from before load to the end of the longest qualify prompt, plus margin) and drop the `memory` note in the tool's options |

The scheduler cannot see a model server's memory: it plans with `resident_bytes` while the tool is loaded, so the
figure must cover the server's peak. TensorFold's CUDA server also sizes its window from `MemAvailable` less a reserve;
`--context 32768` and `--prompt-cache-gib 8` bound it. `qualify.sh` samples `MemAvailable` every 0.5 s during both of
its server runs and reports the largest drop, which is the candidate for the measured figure.

## Priority and eviction

A chat tool defaults to `keep_loaded: true` and `priority: 100` (every other tool defaults to 0): it is loaded when the
daemon starts, never unloaded for idleness, unloaded last when a request needs memory, and loaded again on its own once
the memory is free. With the placeholder 48 GiB, the chat model and the FP8 image model (about 40 GiB) fit side by side
in GB10's 128 GB with the default 8 GiB reserve; a MiniMax H3 video job, which needs most of the machine, evicts the
image model first and then the chat model, and the chat model comes back after the video model idles out.

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
   "qualified_at"}`, written atomically.

The daemon checks only `"qualified": true`; the marker's other fields record what was qualified. A new pin (TensorFold,
checkpoint or drafter) needs a new run.

## Pins

`tools/chat/qwen38-27b/pins.json` holds every hash; `fetch.sh` refuses anything else.

| What | Pin |
| --- | --- |
| TensorFold | v1.1.1, tag `92d71a3c`, commit `b10231b94f8adf3d4132acebdc0a1ac87e53a838`; `tensorfold-1.1.1-linux-aarch64.tar.gz` sha256 `2e0103a5…f9139cb39`, `-linux-x86_64` `e1785e32…6185fbdf` |
| Checkpoint | TensorFold/Qwen3.8-27B-MLX-4bit @ `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c` (4-bit affine, group 64; three shards, 16,054,541,349 B) |
| Drafter | z-lab/Qwen3.8-27B-DFlash2 @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (BF16, 3,848,817,896 B; served in 4 bits, `--drafter-bits` default) |

## Known interactions

- **The GPU lock is exclusive while the chat model is loaded.** Since it is kept loaded, another worker that takes the
  same lock (Drama 3's launcher, PR #13) cannot load beside it, and the scheduler does not know to unload the chat model
  for it: that load fails with a 503 until an operator unloads the chat tool (`POST /v1/tools/qwen3.8-27b/unload`).
  Sharing the lock between LocalRouter's own workers is a follow-up.
- **One request at a time.** Chat requests queue behind image, video and speech jobs like any other job, and vice
  versa; a long reply holds the queue for its length (`max_tokens`, 4096 by default).

## Licences

| Part | Licence |
| --- | --- |
| TensorFold 1.1.1 (binary, fetched) | Apache 2.0 |
| Qwen3.8-27B-MLX-4bit (TensorFold; base Qwen/Qwen3.8-27B) | Apache 2.0 |
| Qwen3.8-27B-DFlash2 (z-lab) | Apache 2.0 |

LocalRouter does not distribute the binary or the weights: `fetch.sh` downloads them from their publishers.
