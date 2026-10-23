# LocalRouter architecture

LocalRouter is one small daemon that owns a machine's GPU memory on behalf of many models. Agents ask for a job
("an image of …", "a 5 s video from this picture"). The daemon picks a model that can do it, makes room, loads it,
runs the job, and unloads the model when it has been idle long enough. Models are adapters behind one interface, so
adding a model, or a new kind of model, does not touch the orchestration.

```
 agents ── MCP (/mcp, or `localrouter mcp-stdio`) ──┐
        ── OpenAI-style HTTP (/v1/...) ─────────────┤
                                                   ▼
                    ┌──────────── localrouter serve ────────────┐
                    │  API (routes.zig, mcp.zig)                │
                    │  jobs table ──► scheduler (one executor)  │
                    │     memory budget, priority, idle TTLs    │
                    └──────┬───────────────┬────────────────────┘
                           │ worker protocol (JSON lines over stdin/stdout)
                 ┌─────────┴───┐     ┌─────┴────────────┐
                 │ worker:     │     │ worker: any      │
                 │ compiled-in │     │ program (`cmd`)  │
                 │ Zig engine  │     │ Python, a binary │
                 └─────────────┘     └──────────────────┘
```

## The pieces

| Piece | Where | Does |
| --- | --- | --- |
| API | `src/api/routes.zig`, `src/api/speech.zig`, `src/api/chat.zig` | OpenAI's `/v1/models`, `/v1/images/generations`, `/v1/images/edits`, `/v1/videos`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/chat/completions`; output files; `/v1/tools` for operators |
| MCP | `src/api/mcp.zig`, `src/cli/mcp_stdio.zig` | the same jobs as MCP tools, Streamable HTTP (stateless) or a stdio bridge |
| Voice | `src/api/realtime.zig`, `src/voice/` | `/v1/realtime` (OpenAI Realtime API over a WebSocket) and the `/voice` page: VAD, STT, chat, sentence-chunked TTS, barge-in, tool jobs ([VOICE.md](dev/VOICE.md)) |
| Pairing | `src/pair.zig`, `src/api/pair.zig`, `src/cli/pair.zig` | with `pairing.required`, `/v1/realtime` and `/mcp` answer paired devices only (401 before the body or upgrade); `/v1/pair` trades a one-time code for a token; `localrouter pair` makes codes and tokens; hashes in `<data_dir>/devices.json`, read on every check |
| Jobs | `src/sched/jobs.zig` | one request's life: queued, in progress, completed / failed / cancelled; its directory of inputs and outputs |
| Scheduler | `src/sched/scheduler.zig` | one executor thread a lane (`main`; `realtime` for speech and transcription): admits a job, makes room (evicting idle tools), loads, runs, enforces deadlines and idle TTLs |
| Tools | `src/tool/` | a tool is a model as the scheduler sees it: `needs`, `load`, `generate`, `unload`, `abort`, `state`. Today every tool is a worker process (`process.zig`) |
| Engines | `src/engine/engine.zig`, `src/engines/` | the compiled-in model implementations a worker runs |

**One process a loaded model.** Each tool runs in its own worker process. Unloading is process exit, so every byte of
GPU and host memory returns to the machine, and a crashing model fails one job, not the daemon. The idle daemon is a
few MB and holds no GPU memory.

**Memory.** Every tool states what it needs before it is loaded (`needs(request)`: resident bytes while loaded, plus
working bytes for this request). The scheduler admits a job only if it fits under the budget (the machine's available
memory minus `reserve_bytes`, or `budget_bytes`). If it does not fit, idle tools are unloaded: lowest `priority` first,
then least recently used. Priority only orders the victims: a request is never refused while unloading idle tools would
make room. `keep_loaded` tools are loaded at start, are exempt from their idle TTL, and come back on their own as soon
as the scheduler is idle, unloading idle tools of lower priority if they must, so an operator can keep, say, an image
model and a voice model warm side by side and still let a video job take the whole machine when it needs it.

**Lanes.** Each lane runs one job at a time. Speech and transcription tools run on the `realtime` lane (a tool's `lane`
overrides it), so a sentence of TTS or a turn of STT never waits behind an image, a video or a chat reply. A tool is
loaded and run only by its own lane; admission and eviction are serialized across lanes, a tool in use on either lane
is never a victim, and memory admitted on one lane but not yet allocated is held back from the other's admission.

## Requests and capabilities

A request has a kind (`image`, `video`, `speech`, `transcription`, `chat`) and needs one capability:

| Capability | Kind | Request |
| --- | --- | --- |
| `text_to_image` | image | prompt |
| `image_edit` | image | prompt + 1 to 5 input images |
| `text_to_video` | video | prompt |
| `image_to_video` | video | prompt + 1 input image |
| `text_to_speech` | speech | text (+ voice, speed, preset) |
| `speech_to_text` | transcription | one audio file (+ language, prompt, timestamps) |
| `chat_completion` | chat | an OpenAI chat request (messages, tools, sampling, ...), proxied to the model's own server |

A tool's capabilities come from its engine (or from its config). A request names a model, or gets the config's default
for its capability (`defaults`, checked at start), else the first one of its kind that has it; a model that lacks it,
or is of the other kind, refuses with a message saying so and naming the models that can.

## Adding a model

### A compiled-in engine (fast path)

Implement `engine.Entry` in `src/engines/<name>_tool.zig` and list it in `engine.entries`:

```zig
pub const entry: engine.Entry = .{
    .name = "my_model",                                 // a tool's "engine" in the config
    .capabilities = &.{ .text_to_image, .image_edit },
    .needs = needs,     // fn (cfg, request) Needs: resident + working bytes, pure and cheap (no I/O)
    .create = create,   // fn (env, cfg) Engine: allocate the engine; nothing heavy yet
    .check = check,     // optional fn (cfg, request, why) bool: refuse what the model cannot serve, before it loads
};
```

The `Engine` vtable has three calls: `load` (weights and kernels; returns resident bytes), `generate(job, sink, arena)`
(writes output files into `job.dir`, reports progress through `sink`, returns the file names) and `unload`. Return
`engine.Refused.Refused` for a request the model cannot serve (wrong size, missing input): the API turns it into a 400.
`src/engines/testpattern.zig` is a complete, GPU-free example; `qwen_image_tool.zig` and `minimax_h3_tool.zig` are the
real ones.

### Any program (the `cmd` adapter)

A tool with `"cmd": ["python", "my_server.py"]` runs that program as its worker. It speaks the worker protocol: one JSON
object per line on stdin and stdout.

```
daemon -> worker   {"op":"load"}
worker -> daemon   {"ok":true,"resident":17179869184}
daemon -> worker   {"op":"generate","id":"img_…","dir":"/data/jobs/img_…","request":{"image":{"prompt":"…","width":1024,…}}}
worker -> daemon   {"progress":{"phase":"denoise","step":3,"of":25}}        (any number)
worker -> daemon   {"ok":true,"files":["0.png"],"seed":7,"ms":11800}
                   or {"ok":false,"error":"…","type":"invalid_request_error"}
daemon -> worker   {"op":"exit"}
```

A worker that can end a generate early answers its load with `"cancel":"signal"` beside `ok`: a cancel then sends it
SIGUSR1, and it replies `{"ok":false,"type":"cancelled"}` and stays loaded (no reload for the next request). Any other
worker, or one that has not answered within 5 s, is cancelled by killing its process group.

Input images arrive as files in the job directory (the request's `references` / `first_frame` name them); outputs go
there too. Set `resident_bytes` and `working_bytes` in the config so the scheduler can plan. Anything that can read and
write lines can be a model: a Python pipeline, a llama.cpp server wrapper, a TTS engine.

### Speech tools

Speech and transcription tools use the same worker protocol; the request is `{"speech": {...}}` or
`{"transcription": {...}}`, and the worker writes one file into the job directory. The daemon owns everything around
the model: input decoding, response encodings, and the limits checked before a tool loads.

```
{"op":"generate","id":"speech_…","dir":"…","request":{"speech":{"input":"Hello.","voice":"Rosie","speed":1,"preset":"","seed":7}}}
  -> {"ok":true,"files":["speech.wav"],"seed":7,"ms":420}      speech.wav: RIFF PCM 16-bit mono at the model's rate
{"op":"generate","id":"transcription_…","dir":"…","request":{"transcription":{"audio":"audio.wav","duration_ms":5860,
   "language":"","prompt":"","temperature":0,"timestamps":"segment","seed":7}}}
  -> {"ok":true,"files":["transcript.json"],"seed":7,"ms":900}
     transcript.json: {"text","language","duration","segments":[{"id","start","end","text",…}],"words":[{"word","start","end"}]}
```

- `audio.wav` is always 16 kHz mono 16-bit PCM: the daemon decodes every upload (WAV, MP3, FLAC, OGG/Opus, M4A/MP4,
  WebM, AAC) with ffmpeg, one at a time, under a deadline, with demuxers from a fixed list and local files only, and
  caps it at the tool's `max_audio_s` (at most 30 minutes). Its length is in `duration_ms`, so `needs` can price it.
  An adapter does no decoding.
- `speech.wav` is the only text to speech output. `/v1/audio/speech` returns it as `wav` or raw `pcm` (with
  `x-sample-rate`, `x-channels` and `x-sample-format: s16le` headers), or encodes `flac`, `mp3`, `opus` or `aac`.
- Fields a segment or word carries beyond the ones above (`avg_logprob`, `no_speech_prob`, …) are passed through
  to `verbose_json`.
- Refuse a request the model cannot honour with `{"ok":false,"type":"invalid_request_error",…}` (a 400); never
  change it quietly. Better: declare the limits in the tool's `options` so the daemon refuses before loading:
  `voices` and `default_voice` (an empty voice becomes the default before the worker sees it), `presets` (the
  model-native styles `preset` may name; none: every preset is refused), `min_speed` / `max_speed` (1 and 1 for a
  model without a speed control), `max_input_chars` (at most 4096) and, for transcription, `max_audio_s`. `/v1/models`
  and `list_models` show them.
- A reply may say why the audio ended and where long-form segments start: `{"ok":true,"files":["speech.wav"],
  "finish_reason":"length","segments":"2;220500"}`. `/v1/audio/speech` sends them as `x-finish-reason` and
  `x-long-form-segments` (printable ASCII only; anything else is dropped).
- Streaming: a tool whose options say `"stream": true` gets `"stream": true` in the request when a client asks for a
  stream, and writes `speech.wav` as the audio comes: a canonical 44-byte header first (its sizes may be placeholders),
  then the samples, then the real sizes before it replies. The daemon sends each new run of samples on as an HTTP/1.1
  chunk; when the job fails after the first chunk, the connection ends without the last chunk.
- References: a tool whose options say `"references": true` gets the request's voice references as
  `references.json` in the job directory (`"references": "references.json"`), a list of `{text, vq_codes}` whose shape
  the daemon has checked.
- A refusal's `type` is `invalid_request_error` (400), `service_unavailable` (503: the model's own server is busy or
  not ready, retry later) or anything else (500).
- Not offered by the API, and refused with a reason: translation, SSE streams, speech `instructions`.

### Models that wrap their own server

A `cmd` worker can run a model's own server rather than load the model itself (`tools/speech/drama3`). It starts the
server on load (on loopback; a launcher that insists on a lock gets the worker's private `LOCALROUTER_GPU_LOCK`, see
[the host's GPU lock](#the-hosts-gpu-lock)), is its client for each request, and stops it on `exit`, keeping the
daemon waiting with progress lines until the server is down, so nothing else loads into memory that is still held. A
launcher's own memory admission is turned off: the scheduler, from the tool's `resident_bytes`, is the only admission
authority. A server that dies takes the worker with it (no reply), so the job fails with a 503 and the next request
starts the model again. Its worker may check more than the [qualification](#qualification) file (Drama 3's binds it
to the host, its pins and its engine binary).

### A model's own server (`openai_server`)

Some models ship their own OpenAI-compatible server (TensorFold's `serve`, a llama.cpp server). The compiled-in
`openai_server` engine runs one as the worker: on `load` it starts `options.server` (an argv; `{port}`, `{id}` and
`{weights}` are filled in) on a free loopback port with `options.env`, waits until `options.health` (default `/health`)
answers 200, and replies `{"ok":true,"resident":0,"port":N}`. Any `cmd` worker may name a `port` the same way.

```
{"id": "qwen3.8-27b", "kind": "chat", "engine": "openai_server", "resident_bytes": 25769803776,
 "options": {"server": ["tensorfold-native", "serve", "TensorFold/Qwen3.8-27B-MLX-4bit", "--host", "127.0.0.1",
                        "--port", "{port}", "--name", "{id}"], "env": {"HF_HUB_OFFLINE": "1"}}}
```

The daemon sends each chat request to that port itself (`src/tool/upstream.zig`, a loopback HTTP/1.1 client) and
writes the reply into the job directory as it arrives: `completion.json`, or the event stream `completion.sse`, which
the route forwards while it grows. The server's 4xx is a 400, its 503 a 503, anything else a 500. Cancelling closes the
request's connection and leaves the worker loaded (the server keeps its weights and prompt cache warm); a server that is
gone, or silent past `generate_timeout_s`, kills the worker, so the next request starts it again. Unload stops the
server with SIGTERM (SIGKILL after `stop_timeout_s`, default 8); if the daemon dies, the worker stops the server first.
`resident_bytes` and `working_bytes` are the model's measured figures: a server reports none.

### The host's GPU lock

A machine whose GPU is shared with jobs outside LocalRouter names their lock file in `gpu_lock`; those jobs take it
exclusively. The scheduler is the one admission authority for LocalRouter's own models: it decides what is loaded
together from `resident_bytes` against the budget alone, and holds the host lock once for all of them (`flock`, shared,
without waiting) from the first load until nothing is loaded. While an outside job holds it, nothing loads: the request,
explicit load or `keep_loaded` start fails with a 503 naming the lock. No worker takes the host lock, so the chat model,
a voice model and a transcription model stay loaded side by side. A worker sees `LOCALROUTER_GPU_LOCK_HELD=1` and the
host lock's path in `LOCALROUTER_HOST_GPU_LOCK`; `LOCALROUTER_GPU_LOCK` names a lock file of its own,
`<data>/locks/<id>.lock` (created, uncontended), for a launcher that insists on taking a lock (Drama 3's `pilot.py`,
given `--gpu-lock "$LOCALROUTER_GPU_LOCK"`), so it can never keep another LocalRouter model out.

### Qualification

A model not yet verified on a machine's GPU names a `qualification` file in its config: it is listed (`"qualified":
false`) and every request, load or `keep_loaded` start is refused with a 503 until that file, written by the model's
qualify script on this machine, says `"qualified": true`. The gate is the same for every kind (`engine.unqualified`).

### A new kind of model (embeddings, …)

1. Add the kind to `request.Kind` and its request struct to `Request` (validated in `request.validate`).
2. Add its capabilities to `request.Capability`.
3. Add the HTTP route (OpenAI's shape where one exists, e.g. `/v1/audio/speech`) and an MCP tool in `mcp.zig`.
4. Write the engine or the `cmd` adapter.

The scheduler, memory accounting, priorities, idle unloading and the worker protocol need no change.

## Engines and correctness

The first engines (Qwen-Image 2.1, its Turbo checkpoint, and MiniMax H3) are written in Zig with their own CUDA kernels, built on
[TensorFold](https://github.com/ashhart/TensorFold)'s runtime and kernels. Each is checked against a Python reference
of the same model (`tools/twin`) bit for bit: every operation alone and chained, then prompt to pixels and audio
samples, on the GPU it ships for. Speed changes keep those bits, or they do not ship. `docs/dev/` has the measured
results and the development history.
