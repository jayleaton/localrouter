# Using LocalRouter from an agent

LocalRouter is an HTTP server (default port 8190), OpenAI-compatible where OpenAI has an API for the job. Tools load
on demand, so the first request to a tool waits for it to load. Later requests are fast until the tool has been idle
for its TTL.

Base URL: `http://<host>:8190/v1` (any API key, or none). When the operator turns on device pairing
(`pairing.required`), every route but `/health` and `/v1/pair` needs a paired device's token as
`Authorization: Bearer <token>`, with the scope the route needs: `inference` (chat, responses, `/v1/realtime`), `media` (images,
videos, speech, transcription), `decisions` (`/v1/systemone`), `read` (models, voices, residency) or `manage` (load,
unload, release, cloning a voice). No token is a 401 `invalid_api_key`, a missing scope a 403 naming it; ask the operator for a token with
what you need (`localrouter pair token NAME --scopes inference,media,read` on the box; `GET /v1/pair` shows yours).
Your jobs and output files are then yours alone: fetch `/v1/jobs/...` and `/v1/files/...` with the same token. The
table of scopes and the legacy profile are in [MCP.md](MCP.md#scopes).

By default the daemon listens on `127.0.0.1` only, so `<host>` is `127.0.0.1` on the machine itself. To reach it from
other machines the operator starts it with `--host tailscale` (the tailnet address; `<host>` is then the machine's
tailnet name or IP), or `--host all` / an IP (`host` in the config takes the same values).

To install LocalRouter on a box (the whole Jarvis stack on a DGX Spark), follow [INSTALL.md](../INSTALL.md): one
deterministic sequence of commands, with what to check and what to ask the user at each step.

## MCP (the easy way)

The same server speaks the Model Context Protocol at `http://<host>:8190/mcp` (Streamable HTTP, stateless); [MCP.md](MCP.md) has
the client configs, every tool with an example, and the security note:

```bash
claude mcp add --transport http localrouter http://<host>:8190/mcp
claude mcp add --transport http localrouter http://<host>:8190/mcp --header "Authorization: Bearer <token>"   # with pairing
```

| Tool | Does |
| --- | --- |
| `list_models` | the models: kind, capabilities, `default_for`, priority, `keep_loaded`, whether each is loaded, and `qualified` for a model gated on this machine's GPU |
| `generate_image` | `prompt`, `size`, `n`, `seed`, `steps`, `model`: waits, returns the PNG(s) inline and as URLs, and the seed. `images` (up to 5 base64 strings or http(s) URLs) makes it an edit of those images |
| `edit_image` | `prompt` and `image` (one base64 string, data URL or URL) or `images` (1 to 5), optional `mask`, `size` (omit it for the input's shape), `n`, `seed`, `model`: the edited PNG(s) as `generate_image` returns them. Needs a model that lists `image_edit` |
| `generate_video` | `prompt`, `size` (omit it for the model's own size), `seconds`, `seed`, `audio`, `wait_s`: the MP4's URL when done within `wait_s`, else a job id. `image` (one base64 string or URL) makes it image to video |
| `list_voices` | each speech model's voices (cloned ones included), its default voice, whether it is loaded and whether it is the default |
| `transcribe_audio` | `audio` (base64, data URL or URL) or `path` (a file on the box inside the config's `local_files`), `language`, `prompt`, `timestamps` (`none`, `segment`, `word`), `wait_s`: the transcript's text, then its JSON (language, duration, segments, words); a longer job returns its id for `get_job` |
| `synthesize_speech` | `input`, `voice`, `speed`, `preset`, `format`, `seed`, `references`: the audio inline (WAV by default) and as a URL; the summary says why the audio ended (`finish_reason`) |
| `chat` | `prompt` and/or `messages`, `system`, `model`, `reasoning` (false: no thinking), `max_tokens`, `temperature`, `top_p`, `stop`, `seed`, `tools`, `tool_choice`, `response_format`: the reply's text (its tool calls as JSON when it called one instead), then the whole completion |
| `decide` | `state`, `questions` (choice, noul, score), `model`: one line a question (the choice and its probability, the noul's probability, the expected level), then the model's checked reply |
| `get_job` | `id`, `wait_s`: status, progress, output URLs |
| `cancel_job` | `id` |
| `release_models` | unloads every model now |
| `computer_control`, `computer_use` | the user's paired computer (the desktop helper): a fixed control, or a bounded task on its screen whose consequential steps wait for the user's tap; `computer_status`, `computer_stop`. Needs a paired device token ([dev/COMPUTER.md](dev/COMPUTER.md)) |

Clients that only speak MCP over stdio use the bridge: `localrouter mcp-stdio --url http://<host>:8190` (below).

Refusals (bad size, no memory free right now, a failed load) come back as tool errors with the reason. The skill for
agents is `skill/localrouter/SKILL.md`. The HTTP API below is the same jobs without MCP.

## Find the tools

```bash
curl -s $BASE/models | jq '.data[] | {id, kind, running, license: .license.name}'
curl -s $BASE/capabilities | jq '{contract, unsupported}'   # what this box offers a harness, and what it does not
```

`GET /v1/capabilities` is the harness contract's discovery document ([HARNESS.md](HARNESS.md)): the contract
version, each surface's profile and authorization, the MCP tools and the features not offered.

`kind` is `image`, `video`, `speech`, `transcription`, `chat` or `decision`. `running: true` means it is loaded now. On a
multi-node LocalRouter each model also names the `node` that runs it; the API is the same whichever node that is.
`capabilities` lists what the model can do: `text_to_image`, `image_edit`, `text_to_video`, `image_to_video`,
`text_to_speech`, `speech_to_text`, `chat_completion`, `typed_decision`, `voice_clone` (a speech model that clones a
voice from recordings: [Voices](#voices-cloning)). Speech models also list `voices`, `default_voice`, `presets`, `speed` (the range they take),
`max_input_chars`, `stream` (whether `stream: true` works), `references` (whether voice references do) and
`instructions` (whether `instructions` do);
transcription models `max_audio_s`. `default_for` lists the capabilities the model serves
when a request names no model (the operator's `defaults` in the config, else the first model that has the capability).
Name a model with `model` to choose one yourself, per request. A request that needs a capability the model lacks is
refused with a 400 naming the model and the capability; a model of the other kind is a 400 and an unknown one a 404,
both listing the models that can do it. A model's own limits (a larger size than it was set up for, a step count its
schedule lacks) are refused before it loads. `localrouter models` prints the same table on the command line.
`license`, where the operator's config or the model's engine states one, is the weights' licence: `name`, `url`,
`use`, `commercial` (omitted when not stated), `excluded_regions` and `notice` (what a product or its outputs must
show). A model without one has no `license`. See [Licences](#licences).

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

Every PNG is labelled as machine-generated in its metadata, whichever model made it (the pixels are untouched): tEXt
`Software` `LocalRouter`, tEXt `Comment` `AI-generated (machine-generated) by model <id> with LocalRouter`, and an
XMP packet (iTXt `XML:com.adobe.xmp`) with the IPTC digital source type `Iptc4xmpExt:DigitalSourceType`
`http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia`, the machine-readable AI-generated label.
Read it with `exiftool -Comment -XMP-iptcExt:DigitalSourceType fox.png`. No C2PA manifest is attached: one needs a
signing certificate and a manifest store, and an unsigned one proves nothing.

## Image edits (OpenAI Images API)

```bash
curl -s -m 900 $BASE/images/edits -F model=<edit-model> -F prompt='make it night' -F 'image[]=@fox.png' \
  | jq -r '.data[0].b64_json' | base64 -d > night.png
```

`POST /v1/images/edits` takes `image` or `image[]` (1 to 5 files) and the fields of a generation (`prompt`, `size`,
`n`, `seed`, `response_format`...). A JSON body works too: `"image": "<base64>"` or `"images": ["<base64>", "data:image/png;base64,...", "https://..."]`.
Inputs may be PNG, JPEG or WebP, at most 32 MB each; the model's own limit on how many may be lower (Qwen-Image: 3 by
default). URLs are fetched by the server (http and https only, 32 MB, 30 s; see [Input URLs](#input-urls)). The model
must list `image_edit` in its capabilities, which an image model does only once its edit pack is installed.

Without `size` (or with `auto`) the result keeps the input's shape (the last image's, for several), at a size the model
picks (Qwen-Image: about 1 megapixel, sides in multiples of 32); with `size` it is that size. `mask` (one file, or one
base64 string or URL in JSON) is a PNG of the first image's size whose alpha is 0 where the edit goes: wherever it is
opaque the first image's pixels are kept, blended at the mask's soft edges (as OpenAI's edits). With `size` equal to
the input's, the kept pixels are the input's exactly.

```bash
curl -s -m 900 $BASE/images/edits -F model=<edit-model> -F prompt='a lake at night' -F image=@fox.png -F mask=@mask.png \
  -F size=1024x1024 -F response_format=url | jq -r '.data[0].url'
```

An edit carries the [images' label](#images-openai-images-api) with `AI-edited (machine-generated) from an input
image` in its comment and the IPTC term for edits, `compositeWithTrainedAlgorithmicMedia`.

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
fields as form fields) or a base64 string, data URL or URL in a JSON body (PNG, JPEG or WebP). The model must list
`image_to_video`. The image becomes the clip's first frame: it is cut to the clip's shape (centred) and resized to its
size, so send one of that shape to keep all of it. Say what the image shows in the `prompt` too; a prompt about
something else pulls the clip away from it.

MiniMax H3 (`minimax-h3`) takes sizes up to its `max_width` x `max_height` and clips up to `max_seconds`, in multiples
of 32 pixels (1280x720 is not one: send 1280x704); a request past them is refused with the limits named, before
anything loads. Without `size` a clip gets the operator's `default_size`. On a GB10 (the Jarvis box: up to 1280x704
and 8 s, 768x448 by default) the engine takes 83 s for 864x480 at 5.17 s, 244 s for 1280x704 at 5.17 s and 397 s for
1280x704 at 8 s; through the API, a 2.3 s clip at 768x448 takes 33 to 39 s with the load.

Every MP4 is labelled as machine-generated in `moov/udta/meta/ilst`, whichever model made it (the coded video and
audio are untouched): `comment` (`©cmt`) `AI-generated (machine-generated) by model <id> with LocalRouter` (from a
first frame: `AI-edited ... from an input image`) and `description` (`desc`), the IPTC digital source type URI
(`.../digitalsourcetype/trainedAlgorithmicMedia`, or `compositeWithTrainedAlgorithmicMedia` from a first frame).
Read it with `ffprobe -v error -show_entries format_tags out.mp4`. `encoder` stays ffmpeg's (`Lavf...`): its CLI
overwrites that tag. No XMP box and no C2PA manifest (see [Images](#images-openai-images-api)).

## Speech (OpenAI Audio API)

```bash
curl -s -m 900 $BASE/audio/speech -H 'content-type: application/json' \
  -d '{"input": "Hello from the GPU box.", "voice": "<a voice from /models>"}' -o hello.wav
```

| Field | Notes |
| --- | --- |
| `input` | up to 4096 characters (or the model's `max_input_chars`) |
| `voice` | one of the model's `voices` (a voice cloned for it included); omit for its `default_voice`. Without `model`, a voice the default speech model lacks goes to the speech model that lists it |
| `speed` | 0.25 to 4 within the model's `speed` range; a model without a speed control takes only 1 |
| `response_format` | `wav` (default), `pcm` (16-bit little-endian, mono; rate in the `x-sample-rate` header), `flac`, `mp3`, `opus`, `aac` |
| `preset`, `seed` | Extensions: a model-native style from its `presets`; a seed for models that sample |
| `stream` | Extension, for a model whose `stream` is true: `true` with `response_format: "pcm"` sends the audio as HTTP/1.1 chunks while it is made |
| `references` | Extension, for a model whose `references` is true: `[{"text": "...", "vq_codes": [[...], ...]}]`, a clip's exact transcript and its codec codes (one list a codebook), spoken into the prompt before the model's voice |
| `instructions` | for a model whose `instructions` is true: how to say it (delivery, emotion, pace), up to 500 characters, without line breaks, speaker tags or the characters the model's direction is written with (Drama 3: `[` and `]`). It is written into the text as the model's own markup, and counts toward `max_input_chars` |

`stream_format: "sse"` is refused (400), and so are `instructions` for a model whose `instructions` is false (it takes
no style prompt). Anything a model cannot do is refused before it loads, never approximated.

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
delivery cues in the text, such as `[sighs]` or `[laughs]` where they should land. Drama 3 has no separate style
control (its system prompt is fixed), so `instructions` become such a bracketed direction before the line:
`{"input": "I'm fine.", "instructions": "whispering, nervous"}` speaks `[whispering, nervous] I'm fine.` (after a
leading `<|speaker:N|>` tag, when the text starts with one). It clones voices ([Voices](#voices-cloning)). Clone only a
voice the model invented, your own, or one whose owner has consented on record: the model's licence forbids imitating
anyone else.
Until the operator has qualified it on the machine's GPU it is listed with `"qualified": false` and every request gets
a 503 saying it is not GPU-qualified on this host.

## Voices (cloning)

```bash
curl -s $BASE/voices | jq '.data[] | {id, model, kind}'                     # every voice: named (configured) and cloned
curl -s -m 900 $BASE/voices -F consent=true -F name=Sam -F file=@take1.webm -F file=@take2.wav   # clone (manage scope)
curl -s -X PATCH $BASE/voices/draft-1a2b3c4d -H 'content-type: application/json' -d '{"name": "Sam"}'
curl -s -X DELETE $BASE/voices/Sam
```

A speech model that lists `voice_clone` makes a voice from recordings of one person: `POST /v1/voices` takes the clips
as the multipart field `file` (repeat it; WAV, MP3, FLAC, OGG/Opus, M4A, WebM; up to 8, 32 MB and 2 minutes each, 10 to
60 s of speech in all) and `consent=true`, the caller's statement that the voice is their own (`source=owner`, the
default) or that its owner consented on record (`source=consented` with a `note` saying where). Optional: `name`,
`transcript` (once per clip, in order: the clip's exact words; blank or absent, the transcription model writes them),
`model`, `language`, `force`. Each clip is heard whole by the transcription model (its voice activity detector finds
the speech), trimmed to its speech and cut at pauses into references of at most 30 s, each brought to the same level.
Only clear failures are refused, with a 400 saying which clip and why: no speech, less than 10 s of it in all, or a
background (noise, or music, named as such) less than 5 dB under the voice, measured between the words; that one has
code `noisy_background`, and `force=true` uses the clips anyway. Anything else that may make the voice worse (a
background under 30 dB, clipping, a clip cut inside a word, more than 60 s or 8 references, of which the first are
kept) is a warning in the voice's `clips[].warnings`. The model then encodes the references (a few seconds on the GPU)
and the clips are deleted.

Without `name` the voice is a draft (`"status": "draft"`, `id` `draft-...`, `expires_at` an hour on), usable as a
`voice` for a preview but not listed; `PATCH /v1/voices/{id}` with a `name` saves it (and renames a saved voice), and
`DELETE /v1/voices/{id}` removes it. Names are 1 to 64 of `A-Z a-z 0-9 . _ -`, led by a letter or digit, not `default`
or `draft-...`, and not a name already in use (ignoring case; 409). A voice is the model's: it is listed in that
model's `voices` (`/v1/models`, MCP `list_voices`) and spoken there in speech requests, voice sessions and
`synthesize_speech`; a request that names it without a `model` goes to that model. The voice object: `id`, `name`,
`model`, `kind` (`named` or `cloned`), `status`, `created_at`, `seconds`, `transcribed`, `source`, `clips` (per clip:
`seconds_in`, `seconds_kept`, `speech_left_out_s`, `references`, `snr_db` (null: not measured), `background` (`noise`
or `music`), `warnings`) and `references` (each `text`, `seconds`, `frames`, `level_dbfs`, `gain_db`, `snr_db`). Creating, renaming and deleting need `manage`; listing needs `read`.

## The assistant's name

```bash
curl -s $BASE/assistant      # {"object": "assistant", "name": "Jarvis", "default_name": "Jarvis", "custom": false, "voice": {"model", "voice"}}
curl -s -X PATCH $BASE/assistant -H 'content-type: application/json' -d '{"name": "Friday"}'   # manage scope
```

What the user calls their assistant, one name per box, chosen when they set up a client and changed in any client's
settings: every client shows it, and a voice session introduces itself with it. `voice.instructions` may hold `{name}`
for it (the default does: "You are {name}, a helpful voice assistant..."); instructions that do not mention the name,
the daemon's or a session's own, get "You are <name>." before them, from the next turn after a rename. Until one is
set it is the config's `assistant.name` ("Jarvis"); a set name is kept in `<data_dir>/assistant.json`, and
`{"name": ""}` goes back to the config's. A name is 1 to 32 characters: letters (any script), digits, spaces,
apostrophes, hyphens and full stops, with at least one letter or digit; spaces are trimmed and runs of them made one,
and anything else is a 400. `voice` is the voice a call speaks in when it names none (null without a speech model), a
hint for a client to show beside the name. Reading it needs any paired device; renaming needs `manage`.

## Transcriptions (OpenAI Audio API)

```bash
curl -s -m 900 $BASE/audio/transcriptions -F file=@meeting.m4a -F response_format=verbose_json \
  -F 'timestamp_granularities[]=segment' | jq -r .text
```

| Field | Notes |
| --- | --- |
| `file` | WAV, MP3, FLAC, OGG/Opus, M4A/MP4, WebM or AAC; up to 32 MB and 30 minutes (or the model's `max_audio_s`). A JSON body may send it as base64, a data URL or a URL |
| `language` | ISO-639-1 (`en`, `de`, `ja`...) or `auto` (detect it); omitted: the server's `stt.language` (`en` unless configured) |
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

**Small limits and thinking.** The model reasons before it answers, and the reasoning counts against `max_tokens`. So a
request whose `max_tokens` (or `max_completion_tokens`) is below the model's `min_thinking_tokens` (1024 by default; the
operator's tool config, `0` never) and that does not choose reasoning itself (`reasoning_effort`, `reasoning`,
`thinking` or `chat_template_kwargs.enable_thinking`) is answered with thinking off. Choose it, and it is as you asked. A
reply that still runs out while reasoning keeps `finish_reason: "length"` and its empty `content`. Either way the reply
says what happened in `warnings`, an extension (a streamed reply sends it in a last chunk with `choices: []`, before
`data: [DONE]`):

```json
"warnings": [{"code": "reasoning_used_budget", "message": "the reply reached max_tokens 16 while still reasoning (16 reasoning tokens), so it has no content: raise max_tokens, or switch thinking off (reasoning_effort: none)"}]
```

`thinking_off_for_small_budget` says thinking was switched off for a small limit. A reply without warnings has no
`warnings` member. The MCP `chat` tool's first text is the warning when there is no reply.

A streamed request the model refuses gets an ordinary JSON error, not a stream. If a stream breaks off after it began,
it ends with `data: {"error": {...}}` and `data: [DONE]`. Closing the connection cancels the request; the model stays
loaded. A model that is not yet qualified on this machine (`"qualified": false` in `/models`) is refused with a 503.

## Responses (OpenAI Responses API)

```bash
curl -s $BASE/responses -H 'content-type: application/json' -d '{
  "instructions": "Be brief.", "input": "Summarise the plan in three bullets.", "store": false
}' | jq -r '.output[] | select(.type == "message") | .content[0].text'
```

`POST /v1/responses` is the same chat model behind OpenAI's Responses API, for clients that speak only that (Codex).
It is stateless: send `store: false` and the whole conversation in `input` each turn, the previous `output` items
included as they came back; `previous_response_id` is refused. Function tools only (hosted tools are refused): a
`function_call` output item is yours to run, then send `{"type": "function_call_output", "call_id": ..., "output":
...}`. Reasoning comes back as a `reasoning` item (`reasoning_text`), never as message text; `"reasoning": {"effort":
"none"}` switches it off. The [small-limit rule](#chat-openai-chat-completions-api) is the same, with `max_output_tokens`
and `reasoning.effort`: below `min_thinking_tokens` and not chosen, thinking is off, and the response object carries
the `warnings`. `stream: true` sends the Responses events (`response.output_text.delta`...,
`response.completed`); closing the connection cancels. What is and is not taken, field by field:
[HARNESS.md](HARNESS.md#responses).

## Decisions (the System One shape)

A decision model answers typed questions about a state with probabilities, in one forward pass and without writing
text: route a request, classify, check a statement. `POST /v1/systemone` takes TypeSafe's System One request, so a
client written for it (SideBud's Jev client) only changes its base URL; a `model` that is not a decision model here
(`jev-latest`) goes to the default one.

```bash
curl -s $BASE/systemone -H 'content-type: application/json' -d '{
  "model": "jev-latest",
  "state": {"user_said": "draw me a fox in the snow"},
  "questions": {
    "route": {"type": "choice", "instructions": "Which action does the user want?",
              "criteria": {"generate_image": "make a picture", "generate_video": "make a clip", "reply": "just answer"}},
    "two": {"type": "noul", "instructions": "Does the user ask for more than one thing?"},
    "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["not at all", "somewhat", "very"]}}}'
```

| Question | `criteria` | Answer |
| --- | --- | --- |
| `choice` | option id -> what it means (or null), 2 to 255 | `choice` (an offered id), `confidence`, `probabilities` (every option) |
| `noul` | `{"true": ..., "false": ...}`, optional | `noul`: the probability that it is true |
| `score` | 2 to 10 levels, lowest first | `score` (the expected level, 0 = the first), `confidence`, `probabilities` keyed `"0"`... |

At most 64 questions and 16 KB a request. The questions are checked before anything loads (400), and the model's
answers against them before they are returned: every question answered with its own type, a choice that is one of the
offered options, probabilities in [0, 1] that sum to 1. An answer that fails is the model's error (500), never a
decision. A decision acts on nothing; what to do with it is your code. Calibration, option order and literal reading
are the model's own: validate it on your own task before acting on its probabilities.

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
| `input_audio_transcription` | `{model, language, prompt}` for the transcription model; `language` a code or `auto` (detect each turn); unset: the server's `stt.language` (`en`) |
| `turn_detection` | `{type: "server_vad", threshold, prefix_padding_ms, silence_duration_ms, create_response, interrupt_response}`, or null for push to talk (`input_audio_buffer.commit`, then `response.create`) |
| `modalities` | `["audio","text"]` (default) or `["text"]` |
| `instructions`, `tools`, `tool_choice`, `temperature`, `max_response_output_tokens` | as OpenAI's; `tools` are your functions (you return `function_call_output`) |
| `partial_interval_ms` | extension: a `localrouter.transcription.partial` per this much speech (0: none) |
| `localrouter` | extension: `{"tool_execution": "server"}` (default) or `"client"`; set it before the first response; with pairing it needs the `inference` scope |

The chat model can also call LocalRouter's own `generate_image` and `generate_video` (and `edit_image`, which edits the
session's last picture, when a model lists `image_edit`): the job runs, its URLs arrive as
`localrouter.job.done` `{call_id, job_id, kind, status, urls, error}`, and the assistant says it is done. With
`"tool_execution": "client"` only your `tools` exist (any names, built-in ones included) and none runs on the box: each
call arrives once, complete, as `response.function_call_arguments.done`; return its `function_call_output` and send
`response.create`. An output for an unknown call, a second output, or one that comes after the user spoke again or
another response began is refused, and `localrouter.function_call.cancelled` tells you when a call stops waiting
([VOICE.md](dev/VOICE.md#client-tool-execution), example in `examples/realtime-client-tools/`). Only
`pcm16` and `server_vad` are offered; anything else is an `error` event naming the field, and the session stays open.
Both endpoints answer other machines when `voice.remote` is on: unset, it is on when the daemon listens beyond loopback
(`host` `all`, `tailscale` or a LAN address) with `pairing.required`, so each call still needs a paired token, and off
otherwise (this machine only; `realtime.remote` in `/v1/capabilities` says which; browsers need HTTPS for the
microphone off loopback). With pairing on, `/v1/realtime` also needs a token: `Authorization: Bearer`, a
`Sec-WebSocket-Protocol` entry `openai-insecure-api-key.<token>` beside `realtime` (the OpenAI browser SDK's), or
`?api_key=<token>`, of a device with `inference`; without one the upgrade is refused with a 401 (403 without the
scope). The session's picture and video tools need `media`, its model tools `manage`. Details: `docs/dev/VOICE.md`.

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
Behind a TLS proxy (`tailscale serve`) output URLs are `https://` (from `X-Forwarded-Proto`), and those are not
recognized as the daemon's own: send such an output back as base64.
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
`spare_room_only` (default false) marks an on-demand model nothing waits on, such as the decision model behind the
voice loop's shadow decisions: it loads only into spare room, never unloading a `keep_loaded` tool (idle tools that are
not kept may still go) and leaving room for the kept tools' working bytes. A request it cannot serve so is refused
with a 503 `insufficient_memory` naming `spare_room_only`, before anything is unloaded; the shadow decisions then skip
the turns. An operator can still make room for it on purpose with MCP `reserve_for`.
While a realtime voice session is open (`voice_sessions` in `GET /v1/tools`), a request or a load for a tool outside
the voice loop (an image, a video, a decision) never unloads a voice-loop model (chat, speech, transcription) of a
higher priority: it is refused with a 503 `insufficient_memory` whose message starts "busy: a voice call is open",
before anything is unloaded, and the call keeps its voice. The chat model sees that message as the tool's result and
says the machine is busy. Without a call, priority only orders the victims, as above. MCP `reserve_for` (which tells
the session first and brings the voice back after the job) is unchanged.
`GET /v1/tools` shows `priority`, `keep_loaded`, `spare_room_only` and `capabilities`.

## Licences

Each model's weights keep their publisher's licence. `GET /v1/models` lists it per model as `license`, and
`GET /v1/capabilities` lists them all (`models.licenses`); an operator sets one per tool in the config (`license`,
the same shape), and the compiled-in image and video engines state their own when the config does not. LocalRouter
informs and enforces nothing: check the terms before you use or publish what a model makes. The Jarvis box's models
(a summary, not legal advice; the [README](../README.md#credits) has the links):

| Model | Licence | Use | Restrictions / notice |
| --- | --- | --- | --- |
| `qwen3.8-27b` | Apache 2.0 | any | |
| `knowline-4b` | Apache 2.0 | any | |
| `whisper-large-v3-turbo` | MIT | any | |
| `fishaudio/drama-3` | Fish Audio Community License v1.0 | research and non-commercial; commercial only under USD 500,000 annual revenue | no Hosted Service (third parties generating speech through an API) without a separate licence; "Built with Fish Audio" |
| `kitten-tts-2` (optional) | Stellon Labs Community License | commercial after registering, until USD 1,000,000 revenue or funding | outputs may not train another model; "Powered by Stellon Labs" |
| `qwen-image-2.1`, `qwen-image-2.1-turbo` | Qwen Research License | non-commercial research or evaluation only | commercial use needs a separate licence from Qwen |
| `minimax-h3` | MiniMax H3 Community License | outside the excluded territories; commercial included (over USD 20 million yearly revenue: MiniMax's written authorization) | not licensed in the US, EU, UK or South Korea; public content disclosed as machine-generated; "MiniMax H3" on a commercial product's interface |

## Errors

Errors use OpenAI's shape: `{"error": {"message", "type"}}`.

| Status | Meaning | Do |
| --- | --- | --- |
| 400 | Invalid request (size, n, prompt, input images or audio, a capability the model lacks, a model of another kind, a size, step count, voice, speed, preset or audio length the model does not take; a chat request the model's server refuses) | Fix the request |
| 401 `invalid_api_key` | Pairing is required and no paired token came (or it was revoked) | Ask the operator for a token |
| 403 | Pairing is required and the token lacks the scope this route or tool needs (the message names it) | Ask for that scope |
| 404 | Unknown model (the message lists the models that can do it) or route; with pairing, a job or file of another device | Re-check `GET /models` |
| 503 `insufficient_memory` | The machine has no room for this tool right now (another service may have priority) | Retry later; don't loop |
| 503 `server_error` | The tool failed to load, its worker or server died, its server is busy, or it is not yet qualified on this machine; the message says which | Retry once; report it if it repeats |
| 500 `server_error` | The model failed on this request | Report it |
| 504 | A deadline passed | Retry once, then report |

## Being considerate

- One job runs at a time. Send requests one after another, not in parallel bursts.
- Batch work for one tool together. Alternating tools can force reloads when memory is tight.
- `POST /v1/tools/release` unloads everything, and `GET /v1/tools` shows what is loaded. These are for operators;
  agents should not need them.
