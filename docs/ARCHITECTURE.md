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
| API | `src/api/routes.zig` | OpenAI's `/v1/models`, `/v1/images/generations`, `/v1/images/edits`, `/v1/videos`; output files; `/v1/tools` for operators |
| MCP | `src/api/mcp.zig`, `src/cli/mcp_stdio.zig` | the same jobs as MCP tools, Streamable HTTP (stateless) or a stdio bridge |
| Jobs | `src/sched/jobs.zig` | one request's life: queued, in progress, completed / failed / cancelled; its directory of inputs and outputs |
| Scheduler | `src/sched/scheduler.zig` | one executor thread: admits a job, makes room (evicting idle tools), loads, runs, enforces deadlines and idle TTLs |
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

## Requests and capabilities

A request has a kind (`image`, `video`) and needs one capability:

| Capability | Request |
| --- | --- |
| `text_to_image` | prompt |
| `image_edit` | prompt + 1 to 5 input images |
| `text_to_video` | prompt |
| `image_to_video` | prompt + 1 input image |

A tool's capabilities come from its engine (or from its config). A request names a model, or gets the first one of its
kind that has the capability; a model that lacks it refuses with a message saying so.

## Adding a model

### A compiled-in engine (fast path)

Implement `engine.Entry` in `src/engines/<name>_tool.zig` and list it in `engine.entries`:

```zig
pub const entry: engine.Entry = .{
    .name = "my_model",                                 // a tool's "engine" in the config
    .capabilities = &.{ .text_to_image, .image_edit },
    .needs = needs,     // fn (cfg, request) Needs: resident + working bytes, pure and cheap (no I/O)
    .create = create,   // fn (env, cfg) Engine: allocate the engine; nothing heavy yet
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

Input images arrive as files in the job directory (the request's `references` / `first_frame` name them); outputs go
there too. Set `resident_bytes` and `working_bytes` in the config so the scheduler can plan. Anything that can read and
write lines can be a model: a Python pipeline, a llama.cpp server wrapper, a TTS engine.

### A new kind of model (speech, text, …)

1. Add the kind to `request.Kind` and its request struct to `Request` (validated in `request.validate`).
2. Add its capabilities to `request.Capability`.
3. Add the HTTP route (OpenAI's shape where one exists, e.g. `/v1/audio/speech`) and an MCP tool in `mcp.zig`.
4. Write the engine or the `cmd` adapter.

The scheduler, memory accounting, priorities, idle unloading and the worker protocol need no change.

## Engines and correctness

The first engines (Qwen-Image 2.1 and MiniMax H3) are written in Zig with their own CUDA kernels, built on
[TensorFold](https://github.com/ashhart/TensorFold)'s runtime and kernels. Each is checked against a Python reference
of the same model (`tools/twin`) bit for bit: every operation alone and chained, then prompt to pixels and audio
samples, on the GPU it ships for. Speed changes keep those bits, or they do not ship. `docs/dev/` has the measured
results and the development history.
