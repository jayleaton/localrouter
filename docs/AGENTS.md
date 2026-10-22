# Using LocalRouter from an agent

LocalRouter is an HTTP server (default port 8190), OpenAI-compatible where OpenAI has an API for the job. Tools load
on demand, so the first request to a tool waits for it to load. Later requests are fast until the tool has been idle
for its TTL.

Base URL: `http://<host>:8190/v1` (any API key, or none).

By default the daemon listens on `127.0.0.1` only, so `<host>` is `127.0.0.1` on the machine itself. To reach it from
other machines the operator starts it with `--host tailscale` (the tailnet address; `<host>` is then the machine's
tailnet name or IP), or `--host all` / an IP (`host` in the config takes the same values).

## MCP (the easy way)

The same server speaks the Model Context Protocol at `http://<host>:8190/mcp` (Streamable HTTP, stateless); [MCP.md](MCP.md) has
the client configs, every tool with an example, and the security note:

```bash
claude mcp add --transport http localrouter http://<host>:8190/mcp
```

| Tool | Does |
| --- | --- |
| `list_models` | the models: kind, capabilities, `default_for`, priority, `keep_loaded`, whether each is loaded, and `qualified` for a model gated on this machine's GPU |
| `generate_image` | `prompt`, `size`, `n`, `seed`, `steps`, `model`: waits, returns the PNG(s) inline and as URLs, and the seed. `images` (up to 5 base64 strings or http(s) URLs) makes it an edit of those images |
| `generate_video` | `prompt`, `size` (omit it for the model's own size), `seconds`, `seed`, `audio`, `wait_s`: the MP4's URL when done within `wait_s`, else a job id. `image` (one base64 string or URL) makes it image to video |
| `list_voices` | each speech model's voices, its default voice, whether it is loaded and whether it is the default |
| `transcribe_audio` | `audio` (base64, data URL or URL) or `path` (a file on the box inside the config's `local_files`), `language`, `prompt`, `timestamps` (`none`, `segment`, `word`), `wait_s`: the transcript's text, then its JSON (language, duration, segments, words); a longer job returns its id for `get_job` |
| `synthesize_speech` | `input`, `voice`, `speed`, `preset`, `format`, `seed`, `references`: the audio inline (WAV by default) and as a URL; the summary says why the audio ended (`finish_reason`) |
| `chat` | `prompt` and/or `messages`, `system`, `model`, `reasoning` (false: no thinking), `max_tokens`, `temperature`, `top_p`, `stop`, `seed`, `tools`, `tool_choice`, `response_format`: the reply's text (its tool calls as JSON when it called one instead), then the whole completion |
| `get_job` | `id`, `wait_s`: status, progress, output URLs |
| `cancel_job` | `id` |
| `release_models` | unloads every model now |

Clients that only speak MCP over stdio use the bridge: `localrouter mcp-stdio --url http://<host>:8190` (below).

Refusals (bad size, no memory free right now, a failed load) come back as tool errors with the reason. The skill for
agents is `skill/localrouter/SKILL.md`. The HTTP API below is the same jobs without MCP.

## Find the tools

```bash
curl -s $BASE/models | jq '.data[] | {id, kind, running}'
```

`kind` is `image`, `video`, `speech`, `transcription` or `chat`. `running: true` means it is loaded now. `capabilities` lists
what the model can do: `text_to_image`, `image_edit`, `text_to_video`, `image_to_video`, `text_to_speech`,
`speech_to_text`, `chat_completion`. Speech models also list `voices`, `default_voice`, `presets`, `speed` (the range they take),
`max_input_chars`, `stream` (whether `stream: true` works) and `references` (whether voice references do);
transcription models `max_audio_s`. `default_for` lists the capabilities the model serves
when a request names no model (the operator's `defaults` in the config, else the first model that has the capability).
Name a model with `model` to choose one yourself, per request. A request that needs a capability the model lacks is
refused with a 400 naming the model and the capability; a model of the other kind is a 400 and an unknown one a 404,
both listing the models that can do it. A model's own limits (a larger size than it was set up for, a step count its
schedule lacks) are refused before it loads. `localrouter models` prints the same table on the command line.

| Image model | Steps | Notes |
| --- | --- | --- |
| `qwen-image-2.1` | 25 (any 1 to 200) | the base model; FP8 (`-nvfp4`: the faster NVFP4) |
| `qwen-image-2.1-turbo` | 8 (only 8) | the Turbo checkpoint and its own schedule: about 3x fewer steps. Omit `steps` |

Both run without classifier-free guidance (CFG 1, the models' default): `guidance` and `negative_prompt` are not used.

## Images (OpenAI Images API)

```bash
curl -s -m 900 $BASE/images/generations -H 'content-type: application/json' -d '{
  "model": "qwen-image-2.1", "prompt": "a red fox in fresh snow", "size": "1024x1024", "n": 1, "seed": 42
}' | jq -r '.data[0].b64_json' | base64 -d > fox.png
```

| Field | Notes |
| --- | --- |
| `size` | `WIDTHxHEIGHT`, 256 to 2048 a side, multiples of 16 |
| `n` | 1 to 4 |
| `seed` | Extension. Omit it for a random one; the response's `seed` reproduces the image |
| `steps`, `guidance`, `negative_prompt` | Extensions; 0 or absent means the tool's default |
| `response_format` | `b64_json` (default) or `url` (served from `/v1/files/...`, kept 24 h) |

The request is synchronous: it returns when the image is done. Use a client timeout of at least 15 minutes, so a
cold load fits.

## Image edits (OpenAI Images API)

```bash
curl -s -m 900 $BASE/images/edits -F model=<edit-model> -F prompt='make it night' -F 'image[]=@fox.png' \
  | jq -r '.data[0].b64_json' | base64 -d > night.png
```

`POST /v1/images/edits` takes `image` or `image[]` (1 to 5 files) and the fields of a generation (`prompt`, `size`,
`n`, `seed`, `response_format`...). A JSON body works too: `"image": "<base64>"` or `"images": ["<base64>", "data:image/png;base64,...", "https://..."]`.
Inputs must be PNG, JPEG or WebP, at most 32 MB each. URLs are fetched by the server (http and https only, 32 MB,
30 s; see [Input URLs](#input-urls)). The model must list `image_edit` in its capabilities.

## Videos (OpenAI Videos API)

```bash
id=$(curl -s $BASE/videos -H 'content-type: application/json' \
  -d '{"model": "testpattern-video", "prompt": "waves at dusk", "size": "1344x768", "seconds": "5"}' | jq -r .id)
curl -s $BASE/videos/$id            # status: queued, in_progress, completed, failed or cancelled; progress 0-100
curl -s -o out.mp4 $BASE/videos/$id/content   # once completed
curl -s -X DELETE $BASE/videos/$id  # cancel
```

Extensions: `seed`, `steps`, `guidance`, `fps`, `audio` (true by default). Poll every few seconds.

Any job, of any kind, can be looked up by its id: `GET $BASE/jobs/{id}` returns `{id, kind, status, progress, phase,
urls, error}`. `urls` lists the `/v1/files/...` outputs once `status` is `completed`. The voice loop's
`localrouter.job.created` and `localrouter.job.done` events carry the same id.

Image to video: add `input_reference`, a file in a multipart body (`-F input_reference=@frame.png`, with the other
fields as form fields) or a base64 string, data URL or URL in a JSON body. The model must list `image_to_video`.

## Speech (OpenAI Audio API)

```bash
curl -s -m 900 $BASE/audio/speech -H 'content-type: application/json' \
  -d '{"input": "Hello from the GPU box.", "voice": "<a voice from /models>"}' -o hello.wav
```

| Field | Notes |
| --- | --- |
| `input` | up to 4096 characters (or the model's `max_input_chars`) |
| `voice` | one of the model's `voices`; omit for its `default_voice`. Without `model`, a voice the default speech model lacks goes to the speech model that lists it |
| `speed` | 0.25 to 4 within the model's `speed` range; a model without a speed control takes only 1 |
| `response_format` | `wav` (default), `pcm` (16-bit little-endian, mono; rate in the `x-sample-rate` header), `flac`, `mp3`, `opus`, `aac` |
| `preset`, `seed` | Extensions: a model-native style from its `presets`; a seed for models that sample |
| `stream` | Extension, for a model whose `stream` is true: `true` with `response_format: "pcm"` sends the audio as HTTP/1.1 chunks while it is made |
| `references` | Extension, for a model whose `references` is true: `[{"text": "...", "vq_codes": [[...], ...]}]`, a clip's exact transcript and its codec codes (one list a codebook), spoken into the prompt before the model's voice |

`instructions` and `stream_format: "sse"` are refused (400): the models take no style prompt. Anything a model cannot
do is refused before it loads, never approximated.

A model that says why its audio ended sets `x-finish-reason`: `stop` (it ended by itself), `length` (clipped at its
frame limit; the audio is cut short) or `silence` (its silence guard ended a reply that fell silent or into a steady hum
for 10 s; the audio before it is kept). A long-form reply also sets `x-long-form-segments: N;S2,S3,...`, the number of
segments and the sample at which each after the first starts.

A streamed reply (`stream: true`) is raw 16-bit mono PCM (`x-sample-rate`, `x-channels`, `x-sample-format` headers, as
for `pcm`) in chunks. Read it to its end:

- It ended by itself: the plain last chunk.
- It was cut short, or is long form: the last chunk carries the `x-finish-reason` trailer (and `x-long-form-segments`).
- It failed after it started: the connection closes without a last chunk. Treat the audio as incomplete.
- It failed before any audio: an ordinary error reply. Closing the connection cancels the job.

**Fish Audio Drama 3** (`fishaudio/drama-3`, when the operator has set it up; Built with Fish Audio) speaks 44.1 kHz
audio up to 47.6 s a synthesis by default (longer with the operator's `max_frames`, and whole long inputs in segments
with `long_form`). Its voices are fixed speakers (`voices` in `/v1/models`; LocalRouter ships six the model itself
invented, `Mae` by default): the same voice is the same speaker in every reply, so a conversation keeps one voice. The
model's built-in voice is not one speaker (each text lands on its own), so it is not offered unless the operator lists
`default`. There is no `speed` or `preset`, and `seed` is ignored (the engine seeds each synthesis from its prompt). Write
delivery cues in the text, such as `[sighs]` or `[laughs]` where they should land. Clone only a voice the model
invented, your own, or one whose owner has consented on record: the model's licence forbids imitating anyone else.
Until the operator has qualified it on the machine's GPU it is listed with `"qualified": false` and every request gets
a 503 saying it is not GPU-qualified on this host.

## Transcriptions (OpenAI Audio API)

```bash
curl -s -m 900 $BASE/audio/transcriptions -F file=@meeting.m4a -F response_format=verbose_json \
  -F 'timestamp_granularities[]=segment' | jq -r .text
```

| Field | Notes |
| --- | --- |
| `file` | WAV, MP3, FLAC, OGG/Opus, M4A/MP4, WebM or AAC; up to 32 MB and 30 minutes (or the model's `max_audio_s`). A JSON body may send it as base64, a data URL or a URL |
| `language` | ISO-639-1 (`en`, `de`, `ja`...); omit to detect it |
| `prompt` | text that guides spelling and style |
| `response_format` | `json` (default: `text`, `usage`), `text`, `verbose_json` (`language`, `duration`, `segments`), `srt`, `vtt` |
| `timestamp_granularities[]` | `segment` and/or `word`, with `verbose_json`; `word` only for models that support it (else 400) |
| `temperature` | 0 to 1 |

The text stays in the spoken language: there is no `/v1/audio/translations` (404) and `stream=true` is refused. A
model may drop what it heard where no voice is detected (the Whisper tool does: silence or noise alone returns empty
text, not a guess), and a WAV file cut short (its data chunk claims more bytes than were sent) is refused with a 400
rather than half transcribed. The
request waits for the transcript; for long files over MCP use `transcribe_audio` with `wait_s` and `get_job`, which
also lets you `cancel_job` it.

## Chat (OpenAI Chat Completions API)

```bash
curl -s $BASE/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Summarise the plan in three bullets."}], "temperature": 0
}' | jq -r '.choices[0].message.content'
curl -sN $BASE/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hi"}], "stream": true}'     # server-sent events, then data: [DONE]
```

The chat model is the machine's default and stays loaded (`keep_loaded`, priority 100), so a reply starts at once; it
gives way only when a job needs its memory (a video job that needs the whole machine) and comes back by itself after.
The body is the model's own: LocalRouter checks only that it is a JSON object with a non-empty `messages` array, sets
`model` to the model it routes to, and passes every other field through to the model's server, which refuses what it
does not take (that refusal is a 400 with the server's message). For the Qwen3.8 27B tool (TensorFold):

| Field | Notes |
| --- | --- |
| `messages` | `system`, `user`, `assistant` (with `tool_calls`) and `tool` (`tool_call_id`) messages; text only (images, audio are a 400) |
| `tools`, `tool_choice`, `parallel_tool_calls` | OpenAI function calling; `tool_choice` `none`, `auto`, `required` or a named function |
| `temperature`, `top_p`, `top_k`, `min_p`, `seed` | `temperature: 0` is greedy: the same request gives the same tokens |
| `max_tokens` / `max_completion_tokens`, `stop` | the reply's limit (the server's default is 4096) and stop strings |
| `response_format` | `json_object` or `json_schema` (structured output) |
| `reasoning_effort`, `chat_template_kwargs.enable_thinking` | reasoning on or off: `"reasoning_effort": "none"` or `"chat_template_kwargs": {"enable_thinking": false}` answers without thinking; the reasoning arrives in `reasoning_content` (`delta.reasoning_content` when streaming) |
| `stream`, `stream_options.include_usage` | server-sent events; with `include_usage` a last event carries `usage` (`"choices": []`) |
| `n` | 1 only (else 400) |

A streamed request the model refuses gets an ordinary JSON error, not a stream. If a stream breaks off after it began,
it ends with `data: {"error": {...}}` and `data: [DONE]`. Closing the connection cancels the request; the model stays
loaded. A model that is not yet qualified on this machine (`"qualified": false` in `/models`) is refused with a 503.

## Voice (OpenAI Realtime API)

`ws://<host>:8190/v1/realtime` is a full-duplex voice conversation: the OpenAI Realtime API's events over a WebSocket,
pcm16 at 24 kHz mono both ways (base64 in `input_audio_buffer.append` and `response.audio.delta`). Server VAD ends
each turn; the turn is transcribed, answered by the chat model with the session's history, and spoken back sentence
by sentence. Speaking over the assistant (`input_audio_buffer.speech_started`) stops its audio and cancels the reply.
`http://127.0.0.1:8190/voice` (or `/`) is a browser client for chat and voice (no install; [VOICE.md](dev/VOICE.md#the-web-client)).

| `session.update` field | Notes |
| --- | --- |
| `model` | the chat model (also `?model=` on the URL); blank: the chat server's default |
| `voice`, `speech_model` | a voice from any speech model's `voices` in `/v1/models` (without `speech_model`, the first configured speech model that lists it speaks; it stays the session's voice for every reply), and which speech model (extension) |
| `input_audio_transcription` | `{model, language, prompt}` for the transcription model |
| `turn_detection` | `{type: "server_vad", threshold, prefix_padding_ms, silence_duration_ms, create_response, interrupt_response}`, or null for push to talk (`input_audio_buffer.commit`, then `response.create`) |
| `modalities` | `["audio","text"]` (default) or `["text"]` |
| `instructions`, `tools`, `tool_choice`, `temperature`, `max_response_output_tokens` | as OpenAI's; `tools` are your functions (you return `function_call_output`) |
| `partial_interval_ms` | extension: a `localrouter.transcription.partial` per this much speech (0: none) |

The chat model can also call LocalRouter's own `generate_image` and `generate_video`: the job runs, its URLs arrive as
`localrouter.job.done` `{call_id, job_id, kind, status, urls, error}`, and the assistant says it is done. Only
`pcm16` and `server_vad` are offered; anything else is an `error` event naming the field, and the session stays open.
By default both endpoints answer this machine only (the operator sets `voice.remote` to open them; browsers need HTTPS
for the microphone off loopback). Details: `docs/dev/VOICE.md`.

## Input URLs

A URL in `image`, `images`, `input_reference` or `file` (HTTP) or `images` / `image` / `audio` (MCP) is fetched by the daemon, so by default
it may only name public hosts. The host is resolved by the daemon, and the request is refused with a 400 (a tool error
over MCP) if any address it resolves to is loopback, in RFC 1918 space (10/8, 172.16/12, 192.168/16), link-local
(169.254/16, which includes cloud metadata at 169.254.169.254; fe80::/10), carrier-grade NAT (100.64/10, where tailnet
addresses live), unique-local IPv6 (fc00::/7), multicast, reserved, or unspecified (`0.0.0.0`, `::`). IPv4 carried inside
IPv6 (`::ffff:a.b.c.d`, NAT64, 6to4) is judged as the IPv4 address. The connection goes to the address that was checked,
not to a second lookup, so DNS cannot swap it afterwards.

Redirects are not followed: a 3xx answer is an error, so a public URL cannot bounce the daemon to a private one.

Two exceptions. The daemon's own output URLs (`/v1/files/...`, as returned by `response_format: url` and the MCP tools)
work, because the URL's host and port are the address the request itself came to; the daemon fetches them from itself.
And the operator can set `"allow_private_urls": true` in the config to lift the check (a lab where inputs live on a
private image server); redirects stay off.

## Command line

```bash
localrouter gen image "a red fox in fresh snow" -o fox.png --size 1360x768 --seed 42
localrouter gen video "waves at dusk" -o waves.mp4 --seconds 5
localrouter gen image "a red fox in fresh snow" -o fox.png --model qwen-image-2.1   # a model of your choice
localrouter models                                                                  # ids, capabilities, defaults
```

The server address comes from `$LOCALROUTER_URL` (default `http://127.0.0.1:8190`) or `--url`.

## MCP over stdio

`localrouter mcp-stdio [--url http://host:8190]` reads JSON-RPC lines on stdin, POSTs each to the daemon's `/mcp`, and writes
the response as one line on stdout (notifications produce none; if the daemon cannot be reached, the request gets a
JSON-RPC error). It keeps no state. In a client's config: `{"command": "localrouter", "args": ["mcp-stdio", "--url", "http://<host>:8190"]}`.
The URL defaults to `$LOCALROUTER_URL`, else `http://127.0.0.1:8190`.

## Priority and keep_loaded (operators)

Per tool in the config: `priority` (integer, default 0, a chat model 100; higher is kept longer) and `keep_loaded`
(default false, a chat model true).
When a request needs memory, idle tools are unloaded lowest priority first, then least recently used, until it fits;
a tool that is loading or running is never unloaded. Priority only orders the victims: a request is never refused while
unloading idle tools, of any priority, would make room. A `keep_loaded` tool is loaded when the daemon starts if it
fits (best effort, logged), is not unloaded by its idle TTL, and is loaded again as soon as the scheduler has been
idle for a couple of seconds: if it does not fit, it unloads idle tools of lower priority, as a request would (never one
of its own priority or above), so a video job's model gives the room back right after the job rather than at its idle
TTL. It can still be evicted by a request that needs the room, and `POST /v1/tools/{id}/unload` (or `release`) keeps it down until it is next used or loaded.
`GET /v1/tools` shows `priority`, `keep_loaded` and `capabilities`.

## Errors

Errors use OpenAI's shape: `{"error": {"message", "type"}}`.

| Status | Meaning | Do |
| --- | --- | --- |
| 400 | Invalid request (size, n, prompt, input images or audio, a capability the model lacks, a model of another kind, a size, step count, voice, speed, preset or audio length the model does not take; a chat request the model's server refuses) | Fix the request |
| 404 | Unknown model (the message lists the models that can do it) or route | Re-check `GET /models` |
| 503 `insufficient_memory` | The machine has no room for this tool right now (another service may have priority) | Retry later; don't loop |
| 503 `server_error` | The tool failed to load, its worker or server died, its server is busy, or it is not yet qualified on this machine; the message says which | Retry once; report it if it repeats |
| 500 `server_error` | The model failed on this request | Report it |
| 504 | A deadline passed | Retry once, then report |

## Being considerate

- One job runs at a time. Send requests one after another, not in parallel bursts.
- Batch work for one tool together. Alternating tools can force reloads when memory is tight.
- `POST /v1/tools/release` unloads everything, and `GET /v1/tools` shows what is loaded. These are for operators;
  agents should not need them.
