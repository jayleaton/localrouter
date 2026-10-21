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
| API | `src/api/routes.zig`, `src/api/speech.zig` | OpenAI's `/v1/models`, `/v1/images/generations`, `/v1/images/edits`, `/v1/videos`, `/v1/audio/speech`, `/v1/audio/transcriptions`; output files; `/v1/tools` for operators |
| MCP | `src/api/mcp.zig`, `src/cli/mcp_stdio.zig` | the same jobs as MCP tools, Streamable HTTP (stateless) or a stdio bridge |
| Voice | `src/api/realtime.zig`, `src/voice/` | `/v1/realtime` (OpenAI Realtime API over a WebSocket) and the `/voice` page: VAD, STT, chat, sentence-chunked TTS, barge-in, tool jobs ([VOICE.md](dev/VOICE.md)) |
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
make room. `keep_loaded` tools are loaded at start, are exempt from their idle TTL, and come back on their own once
the memory is free again, so an operator can keep, say, an image model and a voice model warm side by side and still
let a video job take the whole machine when it needs it.

**Lanes.** Each lane runs one job at a time. Speech and transcription tools run on the `realtime` lane (a tool's `lane`
overrides it), so a sentence of TTS or a turn of STT never waits behind an image, a video or a chat reply. A tool is
loaded and run only by its own lane; admission and eviction are serialized across lanes, a tool in use on either lane
is never a victim, and memory admitted on one lane but not yet allocated is held back from the other's admission.

## Requests and capabilities

A request has a kind (`image`, `video`, `speech`, `transcription`) and needs one capability:

| Capability | Kind | Request |
| --- | --- | --- |
| `text_to_image` | image | prompt |
| `image_edit` | image | prompt + 1 to 5 input images |
| `text_to_video` | video | prompt |
| `image_to_video` | video | prompt + 1 input image |
| `text_to_speech` | speech | text (+ voice, speed, preset) |
| `speech_to_text` | transcription | one audio file (+ language, prompt, timestamps) |

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
- Not offered by the API, and refused with a reason: translation, streamed responses, speech `instructions`.

### A new kind of model (text, …)

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
