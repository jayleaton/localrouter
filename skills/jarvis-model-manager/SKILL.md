---
name: jarvis-model-manager
description: Make room on a LocalRouter (Jarvis) box before a heavy job (a video) through its MCP tools, tell the user what pauses, run the job, and confirm the models came back. Use with the localrouter MCP server (docs/MCP.md) whenever a video or other large job may not fit beside the warm models.
---

# Managing the models of a LocalRouter box

One GPU (a GB10 with about 120 GiB for models) keeps several models warm. A heavy job may not fit beside them. The
server makes room by itself (it unloads idle models lowest priority first and reloads them after), but doing it
explicitly lets you tell the user first, and lets you check that everything came back.

## The tools (MCP, `/mcp` on the box)

| Tool | Use |
| --- | --- |
| `list_models` | each model's `state`, `resident_bytes`, `needs_bytes` (its default job), `displaced` (unloaded to make room, coming back), `held`; `memory.mem_available_bytes`, `memory.reserve_bytes`, `memory.voice_paused` |
| `reserve_for` | `model`, `seconds`/`size` (to price the job), `dry_run`, `hold_s` (default 600): what it would unload, or unloads it (never a chat model), loads the model and holds it for its next job. What it unloads comes back by itself when that job is done, or after `hold_s` |
| `unload_model` | `model`, `hold_s` (default 120): one model now; it comes back by itself after `hold_s`. A chat model is refused |
| `load_model` | `model`, `wait_s`: bring one back now (refused if it would need the chat model's memory) |

Voice sessions on the box are told before a voice model is unloaded (`localrouter.voice.paused`) and when they are back
(`localrouter.voice.resumed`). While voice is paused they keep the call open: replies are text only, the user's turns
wait for the ears to come back, and the job's "it is ready" is spoken once voice is back. Every unload is in the daemon's
journal (`unload_model`, `reserve_for ... unloading`, `evicting`).

## The procedure

1. Before the heavy job, call `reserve_for` with its model and `dry_run: true` (for a video, its `seconds`).
   - `unloads` empty: it fits. Run the job.
   - `fits` false (an error): it does not fit even without everything but the chat model. Say so; do not unload.
2. If it would unload models, tell the user in one short sentence what pauses and roughly for how long, for example:
   "I'll pause voice for about a minute while I make the video."
3. Call `reserve_for` with `dry_run: false`, then start the job (`generate_video`; with `wait_s` short, poll `get_job`).
4. When the job is done the unloaded models load again by themselves (about half a minute for the voice models). Call
   `list_models`: every model you saw unloaded should be `ready` (or loading). If one stays `unloaded` and not
   `displaced`, call `load_model` for it.
5. Tell the user the result and that everything is back.

Rules: never try to unload the chat model (the tools refuse); unload only what `reserve_for` names or what the user
asks for; prefer `reserve_for` to a run of `unload_model` calls, since it unloads exactly enough and restores after the job.

## On the Jarvis deploy (deploy/jarvis/config.json.in)

| Model | Resident | Priority | Kept warm |
| --- | --- | --- | --- |
| `qwen3.8-27b` (chat) | 24 GiB | 100 | yes, never unloaded by these tools |
| `fishaudio/drama-3` (voice) | 42 GiB | 95 | yes |
| `whisper-large-v3-turbo` (ears) | 3.6 GiB, plus 2 GiB a job | 90 | yes |
| `qwen-image-2.1-turbo` (image) | 35.2 GiB | 10 | yes |
| `minimax-h3` (video) | about 39 GiB, plus its frames | 0 | no (unloads 30 s after a clip) |

With everything warm about 14 GiB is free (6 GiB more is the reserve). A video then needs the image model's memory;
`reserve_for` unloads the image model and, only when memory is tighter (another job, a long chat context), Whisper and
then Drama 3. A 3-second clip takes about a minute; the voice models are back about half a minute after it.
