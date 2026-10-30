# Voice: a full-duplex conversation on one box

The user talks to LocalRouter instead of typing. Microphone audio streams in over a WebSocket. Server-side voice activity
detection (VAD) finds the end of each turn. The turn is transcribed (STT, partial then final) and sent to the chat
model with the session's history. The reply streams back as text and, sentence by sentence, as synthesized speech (TTS).
When the user starts talking over the assistant, playback stops and the chat generation is cancelled at once.

```
 browser /voice (or any OpenAI Realtime client)
   │  WebSocket /v1/realtime: JSON events, PCM16 24 kHz mono base64 both ways
   ▼
 ┌───────────────────────── session (src/voice/session.zig) ─────────────────────────┐
 │ reader: client events ─► VAD ─► speech_started / speech_stopped ─► commit          │
 │                              └─ barge-in: cancel response (chat socket shut, TTS   │
 │                                 queue dropped, no more audio sent)                 │
 │ worker: final STT ─► chat stream (history, tools) ─► sentence chunker ─► speaker   │
 │ speaker: one TTS job a sentence ─► resample ─► response.audio.delta (100 ms each)  │
 │ tool waiters: image / video job ─► localrouter.job.done (URL) ─► spoken follow-up  │
 └──────┬──────────────────────────────┬────────────────────────────┬────────────────┘
        │ scheduler, realtime lane      │ OpenAI chat completions    │ scheduler, main lane
        ▼                               ▼ (SSE, this daemon or any)  ▼
   transcription + speech tools     chat model                   image / video tools
```

## Decisions

| Topic | Decision | Why |
| --- | --- | --- |
| Protocol | `GET /v1/realtime` upgraded to a WebSocket; the OpenAI Realtime API's beta event names and shapes | Existing clients and SDKs speak it; the shapes cover audio in and out, transcripts, responses, tool calls and cancellation. Extensions are namespaced `localrouter.*` and clients ignore them |
| Audio | `pcm16` only, 24 kHz mono little-endian both ways (OpenAI's default); g711 is refused | One format keeps the daemon free of codecs; the page and SDKs send pcm16 already |
| Subprotocol | When the client offers `realtime` in `Sec-WebSocket-Protocol` (browser SDKs do), the server selects it | Browsers fail the handshake if an offered protocol is not echoed |
| VAD | Energy based, in the daemon, per 20 ms frame against an adaptive noise floor. `threshold` (0 to 1) sets the margin over the floor (6 + 24 x threshold dB); `prefix_padding_ms` and `silence_duration_ms` as OpenAI's `server_vad`. Speech is `min_speech_ms` (60 ms) of loud frames in a row to start a turn. In a turn, mid-speech any loud frame keeps it going (speech has short dips); once the speaker has been quiet for 200 ms, only 120 ms or more of loud frames in a row is speech that restarts the wait, and a shorter burst (a click, a knock, a cough) counts as quiet (`vad.zig`: `pause_min_frames`, `resume_frames`). The turn ends on one unbroken run of `silence_duration_ms`, so pauses never add up. `turn_detection: null` is push-to-talk (`input_audio_buffer.commit`) | Runs on the reader thread in microseconds, so barge-in needs no model and no GPU. Whisper's own Silero gate still drops non-speech from transcripts. Echo is the browser's job (`echoCancellation`) |
| Barge-in | On `speech_started` during a response, on the reader thread: the response is marked cancelled, the chat stream's socket is shut down, queued TTS jobs are dropped, the speech job in progress is ended in its worker when the worker offers the soft cancel (`"cancel":"signal"` in its load reply: SIGUSR1, no reload), and no further audio is sent. `response.done` reports `cancelled` / `turn_detected` | Within the audio chunk that crossed the threshold. A TTS job already running finishes and is discarded: killing its worker would cost a reload on the next turn |
| History | Per session: the system `instructions`, then user, assistant, tool-call and tool-result messages. An interrupted reply keeps only the sentences whose audio was sent, and `conversation.item.truncate` cuts it to what was played. The oldest turns are dropped past `max_history_chars` | The model sees what the user actually heard |
| STT | The final transcript runs on the whole turn when VAD commits it. Nothing is transcribed before the turn ends: no draft and no early end, so the transcript the model gets is one pass over the whole utterance (`partial_interval_ms`, off by default, only shows interim words and is never the model's input). Each turn is transcribed in the session's `input_audio_transcription.language` (the `/voice` page and the companion send the person's **Language** setting, English by default), else the server's `stt.language` (`en`); `auto` detects it on every turn. Detection is opt-in because it is unreliable on the 1 to 2 s turns a call is made of (DGX-02, 2026-10-27: a clean "Stop." came back as Italian and as Korean), and a session no longer keeps the language its last turn was detected in: with that, one short turn misheard as another language held every later turn of the call in it. Voice turns decode in one greedy pass (`voice.greedy_transcripts`, the worker's request `decode: "greedy"`: the same model, precision, timestamps and Silero gate, without the temperature fallback's bookkeeping); the HTTP API keeps the model card's recipe. On DGX-01 the two gave identical transcripts on every qualify fixture and e2e prompt, in both auto and fixed language, and greedy was about 135 ms faster on a voice turn. The Whisper worker decodes a second of silence once after loading, so the first turn after a reload is not cold (1.1 s cold, 0.36 s warm); and when the session's STT has been idle for 20 s (another model's job may have run meanwhile: the first decode after a video took 1.7 s), the session transcribes a second of silence as soon as someone starts speaking, while they talk. While the user speaks, a partial can run every `partial_interval_ms` of new speech (default 0, off: on one Whisper a partial in flight holds the final back by a whole decode), never more than one at a time, reported as `localrouter.transcription.partial` | Whisper is not a streaming model; re-running it on the growing turn is the standard way to get partials. Partials replace each other, so they are not OpenAI's append-only `delta`. The draft hides the decode behind the hangover; the sentence-end rule shortens the hangover only where cutting in is unlikely, and a pause between two sentences of one request still has to outlast the decode |
| Chat | OpenAI `/v1/chat/completions` with `stream: true`, behind `voice.Chat` (`src/voice/chat.zig`). `voice.chat_url` defaults to this daemon's own `/v1`, so the chat tool works the day it lands; any OpenAI-compatible server works too. At the session's start and after each response the session primes the chat server (`voice.prime_chat`): the conversation as the next turn's request will begin it, with `max_tokens` 1 and the reply discarded, so that request finds it in the server's prefix cache | One contract, no coupling to how the chat model is served. Priming keeps the first token of the next turn from paying for the history after a new assistant message or a barge-in |
| TTS | A sentence chunker over the streamed text; one speech job a chunk; the first chunk may end at a clause (a comma after 24 characters). Within a sentence the audio is streamed: the session tails the job's `speech.wav` as the model writes it, resamples it piece by piece (`resample.Stream`, the same samples as the whole) and sends 40 to 100 ms deltas | First audio when the model's first audio exists, not after the whole reply or sentence |
| Leaked reasoning | `<think>…</think>` spans are dropped from the stream (`chunker.ThinkFilter`). A reply's start is also held (`chunker.Lead`) until text follows its first sentence end, or a clause mark 24 or more characters in (where the chunker may end the first chunk), that is not and cannot become `</think>` (about one token): a `</think>` that was never opened (a model writing its reasoning as plain text) drops everything before it, and later stray tags are removed. Send the server's own switch too (`voice.chat_extra`, e.g. `{"chat_template_kwargs": {"enable_thinking": false}, "reasoning_effort": "none"}` for TensorFold's Qwen) | Measured on DGX-01: with thinking off, Qwen3.8-27B still answered "The capital of France is Paris.\n</think>\n\nThe capital of France is Paris." once; the voice must neither say the tag nor the answer twice |
| Lanes | The scheduler gets more executors: `realtime` for speech and `listen` for transcription (a tool's `lane` overrides its kind's default; with the chat kind of #14, chat gets its own). Residency changes (admission, eviction, loads) are serialized across lanes; memory admitted but not yet visible in MemAvailable is held back | With one executor a streaming chat job would hold the GPU queue, and sentence 1's TTS would wait for the whole reply. A turn's transcript must not wait behind a sentence of speech either (measured on DGX-01: the barge-in turn's transcript 6.1 s after speech end on a shared lane, 2.3 s on its own) |
| Turn end | The turn ends when the speaker has been quiet for `silence_duration_ms` in one unbroken run: the page's "Wait before replying", 0.3 to 5 s, default 0.8 s. Speech within it continues the same turn and restarts the wait. Nothing considers the turn before it ends: no chat request, no speech and no tool call. Its words may be transcribed early: once the speaker has been quiet for `voice.early_stt_ms` (150), the turn so far is transcribed, and if the wait ends with nothing more said that transcript is the turn's (no second decode), so the answer starts as the wait ends. The journal logs each voice chat request with how long after its turn ended, and each `max_tokens` 1 prime of the conversation before the turn | The wait is the speaker's to think. A longer wait costs that much latency, and the user sets it |
| Tools | The chat model is offered `generate_image` and `generate_video` (when such models exist; `voice.tools`), and `edit_image` when a model lists `image_edit`. A call queues a normal LocalRouter job; the session waits for it in the background, sends `localrouter.job.done` with the file URLs, adds the result to the history and creates a response so the assistant confirms it aloud. The session remembers its last completed picture (made or edited): "edit that picture to make it night" is `edit_image` with only a `prompt`, which edits that file (or `image`, an earlier result's URL); with no picture yet the call's output says so and no job runs. When a model lists `image_to_video`, `generate_video` also takes `from_last_image: true`, which animates the last picture. Client-defined `tools` follow OpenAI's flow (the client returns `function_call_output`) | The conversation continues while the job runs; files are ordinary `/v1/files/...` outputs |
| Client | `/voice` (and `/`, which redirects to it): Jarvis, one HTML file with inline JS, embedded in the binary. See [The web client](#the-web-client) | No build step. `getUserMedia` needs a secure context: `http://127.0.0.1` is one; remote use needs HTTPS in front |
| Exposure | `/voice` and `/v1/realtime` answer non-loopback peers when `voice.remote` is true; unset (null), it is true when the daemon listens beyond loopback with `pairing.required` (every call still needs a paired token with `inference`), false otherwise | A microphone endpoint should not appear on the network because the image API was opened up |
| Memory | Chat, STT and TTS are `keep_loaded` with priorities above the image and video tools; the scheduler decides the rest. Voice job directories (user audio, TTS audio) are deleted within a minute of use | The loop never forces the chat model out; utterances do not sit on disk for 24 h |

## Session configuration

`session.update` takes OpenAI's fields: `modalities` (`["audio","text"]` or `["text"]`), `instructions`, `voice`,
`input_audio_format` / `output_audio_format` (`pcm16`), `input_audio_transcription` (`{model, language, prompt}`; the
transcription model), `turn_detection` (`{type: "server_vad", threshold, prefix_padding_ms, silence_duration_ms,
create_response, interrupt_response}` or null), `tools`, `tool_choice`, `temperature`,
`max_response_output_tokens`. Extensions: `model` (the chat model; also `?model=` on the URL), `speech_model` (the
TTS model), `partial_interval_ms`, `localrouter.tool_execution` ([Client tool execution](#client-tool-execution)). A name that is not a model of the right kind is an `error` event naming the models
that can do it, as the HTTP API's refusals do. Defaults come from the daemon config:

```json
{"voice": {"remote": null, "chat_url": "", "chat_api_key": "", "chat_model": "", "speech_model": "", "speech_models": [],
           "transcription_model": "", "voice": "", "instructions": "You are {name}, a helpful voice assistant...",
           "partial_interval_ms": 0, "prime_chat": true, "warm_speech": true, "greedy_transcripts": true, "speak_written": false, "max_history_chars": 24000, "tools": true, "tool_wait_s": 1800,
           "temperature": null}}
```

`speech_model` and `transcription_model` empty: the config's `text_to_speech` and `speech_to_text` defaults, so
swapping a voice model is one `defaults` entry. `speech_models` is an optional fallback chain for speech (in order, each
configured and qualified one tried; one that fails before any audio was sent falls through to the next); empty, the
`text_to_speech` default speaks alone.

`{name}` in `instructions` (the daemon's or a session's) is the assistant's name, `assistant.name` or the one a client
set over `/v1/assistant` ([AGENTS.md](../AGENTS.md#the-assistants-name)), read each turn; instructions that do not
mention the name get "You are <name>." before them, so the assistant knows what it is called whatever a client sent.

`voice.temperature` is a session's chat temperature until it sends its own (null: the chat server's). The Jarvis
deployment runs Qwen with thinking off at its non-thinking sampling, `temperature` 0.7 with `top_p` 0.8 in `chat_extra`:
at its server's 1.0 and 0.95 (the thinking settings), a spoken picture request went without the tool call in 4 to 7 of
20 samples of the same request, and in none at 0.7.

A Jarvis box keeps the three voice models warm above the image and video models, so a picture or a clip evicts
neither (the tool entries are the adapters' own, abbreviated):

```json
{"defaults": {"text_to_speech": "kitten-tts-2", "speech_to_text": "whisper-large-v3-turbo"},
 "voice": {"chat_model": "qwen3.8-27b"},
 "tools": [
  {"id": "qwen3.8-27b", "kind": "...", "priority": 100, "keep_loaded": true},
  {"id": "whisper-large-v3-turbo", "kind": "transcription", "priority": 90, "keep_loaded": true, "lane": "realtime"},
  {"id": "kitten-tts-2", "kind": "speech", "priority": 90, "keep_loaded": true, "lane": "realtime"},
  {"id": "qwen-image-2.1-turbo", "kind": "image", "priority": 10},
  {"id": "minimax-h3", "kind": "video", "priority": 0}]}
```

## Events

Client to server: `session.update`, `input_audio_buffer.append` / `.commit` / `.clear`, `conversation.item.create`
(`message` with `input_text`, `function_call` or `function_call_output`), `conversation.item.truncate`,
`conversation.item.delete`, `response.create`, `response.cancel`. Anything else is an `error` event; the session stays
open. A `function_call` item is history: it starts no job. The Jarvis page seeds a conversation's pictures and videos
as a `function_call` and its output. Seeded as text alone ("make me a cow", "here's your cow"), the model copied the
pattern and promised pictures without calling the tool.

Server to client, one voice turn:

```
session.created                                   (on connect)
input_audio_buffer.speech_started                 {audio_start_ms, item_id}
localrouter.transcription.partial                 {item_id, transcript}            (0 or more)
input_audio_buffer.speech_stopped                 {audio_end_ms, item_id}
input_audio_buffer.committed                      {previous_item_id, item_id}
conversation.item.created                         {item: user message, input_audio}
conversation.item.input_audio_transcription.completed {item_id, content_index, transcript}
response.created
response.output_item.added / conversation.item.created   (assistant message)
response.content_part.added                       {part: {type: audio, transcript: ""}}
response.audio_transcript.delta                   (text as the model writes it)
response.audio.delta                              (base64 pcm16, 100 ms each, in order)
response.audio.done, response.audio_transcript.done, response.content_part.done, response.output_item.done
response.done                                     {response: {status: completed | cancelled | failed, output}}
```

A tool call adds `response.output_item.added` (a `function_call` item), `response.function_call_arguments.delta` /
`.done`, `response.output_item.done`; then, when the job ends, `conversation.item.created` (the
`function_call_output`), `localrouter.job.done` `{call_id, job_id, kind, status, urls, error}` and a new response.
Errors are `error` `{error: {type, code, message, param, event_id}}`.

## Client tool execution

A harness that runs its own tools on the user's device sets `session.update {"session": {"localrouter":
{"tool_execution": "client"}, "tools": [...]}}` (default `"server"`, the built-in tools above). The audio pipeline is
the same in both modes: the wait timer, early transcription, chat priming, sentence speech and barge-in are unchanged.
What changes is who owns the tools:

- Only the client's `tools` are offered. The built-in image, video, model-management and computer tools, and the
  model-management procedure in the instructions, are not, and no call runs on the server. The built-in names are free
  for the client's own functions. Each name must be unique, 1 to 64 letters, digits, `_` or `-`; a named `tool_choice`
  must be one of them.
- `session.created` / `session.updated` echo `"localrouter": {"tool_execution": ...}`. Any other `localrouter` key or
  value is refused (`invalid_session`), so a client asking for a mode this server lacks finds out at once. The mode can
  change only before the session's first response. With pairing required, the session's device needs the `inference`
  scope for it (the legacy `voice` profile includes it); without it the request is refused and the session keeps server
  tools.
- Each call arrives once, complete, after the model has finished it: `response.output_item.added` (the `function_call`
  item), `response.function_call_arguments.delta` (all the arguments at once) and `.done` `{call_id, name, arguments}`,
  then `response.done`. A reply may make several calls (`output_index` 1, 2, ...). A `call_id` is unique in the session:
  when the model repeats one (a chat server's `call_0` each reply), the session gives the call its own. A cancelled or
  failed response makes no calls.
- Run each call, then send `conversation.item.create {"item": {"type": "function_call_output", "call_id", "output"}}`
  for it and `response.create` to hear the answer. Nothing starts a response by itself for an output.
- An output must answer a call that still waits for one. Refused as an `error`, never added to the conversation:
  `unknown_call_id` (no such call), `duplicate_call_output` (it already has its output) and `call_expired`.
- A waiting call expires when the conversation goes on without it: the user starts speaking (with `interrupt_response`,
  as barge-in does) or a response begins. The session sends `localrouter.function_call.cancelled` `{item_id, call_id,
  name, response_id, reason: turn_detected | response_started}` so the harness can stop the work, and the model reads
  the call as cancelled with no result. A late output is refused, so it can never be spoken in a later turn.
- A `function_call` item a client creates is history, as in server mode (a reconnecting harness rebuilds its
  conversation); it may take one output, and its `call_id` must be new.

`examples/realtime-client-tools/` is a small client that registers one local function, is called, returns the output
and hears the continuation.

## The chat contract

What the session needs from the chat server (`voice.chat_url`, default this daemon):

- `POST {chat_url}/chat/completions` with `messages`, `stream: true`, `stream_options.include_usage`, optional
  `model`, `tools`, `tool_choice`, `temperature`, `max_tokens`.
- Server-sent events `data: {chunk}` ending with `data: [DONE]`; `choices[0].delta.content` text and
  `choices[0].delta.tool_calls[]` (`index`, `id`, `function.name`, `function.arguments` fragments); `finish_reason`.
  `delta.reasoning_content` and `<think>…</think>` text are never spoken or kept.
- Closing the connection stops the generation without unloading the model (barge-in does this).
- Errors as OpenAI's envelope with a non-200 status; the session reports them as an `error` event and fails the
  response.

## Latency budget (speech end to first audio)

Expected on one GB10, warm, not yet measured (the end-to-end run on DGX-01 measures it):

| Stage | Expected | Lever |
| --- | --- | --- |
| VAD end of turn | `silence_duration_ms` (500) | the setting; 300 to 400 for snappy turns |
| Final STT, Whisper large-v3-turbo, a 3 s turn | 150 to 300 ms | partials keep the model warm; a shorter window |
| Chat time to first token, Qwen3.8-27B, 1 to 2 k tokens of history | 300 to 800 ms | prefix caching of the history |
| Chat tokens to the first clause, about 10 tokens at 15 to 20 tok/s | 500 to 700 ms | the clause break (24 characters) |
| TTS of the first clause, Kitten TTS 2 | 300 to 600 ms | its GPU RTF, unmeasured on GB10 |
| Daemon overhead (WAV, scheduler, resample, WebSocket) | under 20 ms | measured by the integration test's fakes |
| **Total** | **about 1.8 to 2.9 s** after the user stops (1.3 to 2.4 s after the VAD commits) | |

The daemon's own share is covered by `tests/voice.zig` with zero-latency fakes; the model stages are what DGX-01 measures.

### Measured on DGX-01 (2026-10-21)

`tools/spark/jarvis/voice_e2e.sh` on `integration/jarvis` `5ed7de6f` (this branch with #14, #13 and #17). The models
were Qwen3.8-27B (TensorFold, thinking off), Drama 3 streamed, and Whisper large-v3-turbo, with
`silence_duration_ms` 500. The prompts were spoken by Drama 3 and streamed in real time; times run from the last
voiced sample. The table covers seven turns, with the warm-up left out of the statistics (median, p90):

| From speech end to | before streaming (`24be911`) | streamed speech (`5ed7de6f`) |
| --- | ---: | ---: |
| final transcript | 1.66 s, 2.30 s | 1.37 s, 2.18 s |
| first reply text | 1.83 s, 2.91 s | 1.82 s, 2.37 s |
| first reply audio | 4.77 s, 8.20 s | **2.20 s, 2.95 s** |
| the barge-in turn's first audio | 8.20 s | **2.08 s** |

- **Where the time goes:**
  - VAD ends the turn 0.42–0.51 s after speech.
  - Whisper takes another 0.8–1.5 s; partials are off, so none holds the final back.
  - The chat model's first token comes 0.2–0.5 s later.
  - The first audio delta follows the first clause by 0.3–0.6 s.
- **Barge-in turn:** its first audio no longer waits for the interrupted sentence, which Drama 3's worker now ends on
  the soft cancel (SIGUSR1). The cancel came 4–40 ms after `speech_started`, with no audio of the old answer after it.
- **Tool turn:** the image was done 18.3 s after speech end, its PNG was fetched, and the confirmation was spoken.
- **Eviction case:** idle tools were unloaded and a 23 GiB ballast stood in for another tenant, leaving 30 GiB free.
  - The 1 s video then unloaded Drama 3 and Whisper; chat, at priority 100, stayed.
  - After the clip, Whisper was already back and Drama 3 was ready in 7.5 s.
- **Drama 3 behind another tenant:** its launcher keeps a 30 GiB headroom floor of its own, so it cannot start while
  another process holds the memory. LocalRouter's admission said yes, but the server exited with status 1. Reloads
  therefore back off from 5 s, not a flat 60 s.
- **Peak memory:** 13.1 GiB `MemAvailable` (of 114.4 idle) during the image job.
- **Leaked reasoning:** run `24be911`, with thinking off, once answered "The capital of France is Paris.\n</think>\n\nThe
  capital of France is Paris.".
  - **Fix:** the voice request now also sends `reasoning_effort: none`, and `chunker.Lead` holds each reply's start.
  - **Rerun (`f3dd471d`):** all 31 checks passed, including "no reply contains think>" and "no reply repeats its own
    first sentence". Median first audio was 2.05 s and the barge-in turn 2.01 s; Drama 3 reloaded in 8.5 s with a
    24 GiB ballast held.
  - **What the hold costs:** first text came 0.1–0.46 s later than in the previous run; first audio did not move.
  - **Did the server switch alone fix it?** Unknown. The leak did not reproduce in 20 + 20 streamed requests, with
    thinking off only and with `reasoning_effort: none` added. Those probe requests carried no tools, while the one
    observed leak happened in a voice request that did. The hold stays as the guard.

### First spoken word under 1 s (DGX-01, 2026-10-21)

Times run from the end of the user's speech to the first non-silent reply audio, as heard by the client. They are
the median over the six turns after the warm-up, and the p90 over the five turns without a tool call. The models,
precision and transcripts are unchanged from the baseline; every transcript in the after run was byte-identical
to the before run's.

| | before (`4011b7e0`) | after (`369808d8`) |
| --- | ---: | ---: |
| **first non-silent audio, median** | **2172 ms** | **961 ms** |
| **p90, turns without a tool call** | **2389 ms** | **981 ms** |
| checks | 31 of 33 (both latency checks failed) | 33 of 33 |

Stage medians in ms (measured before the wait became the end of the turn: the draft and the early end below are gone); a negative `stt_wait` means the transcript was started before the turn ended:

| stage | before | after | what changed |
| --- | ---: | ---: | --- |
| VAD end of turn | 484 | 349 | the draft's sentence end ends the turn after 300 ms of quiet, not 500 |
| STT wait | 0 | -263 | the draft starts after 100 ms of quiet and becomes the final transcript |
| STT | 888 | 252 | fixed or sticky language (no detection pass), 4 CPU threads, greedy decode, Silero run alongside generate(), memoized resampler |
| chat first token | 182 | 184 | (cold turns: the prime cut the warm-up's 716 to 164, the capital turn stayed 388) |
| to first text for TTS | 286 | 183 | the lead hold releases one token past the first clause or sentence |
| TTS first samples | 128 | 131 | |
| delivery and leading silence | 2 + 15 | 2 + 9 | |

Barge-in still cancelled the answer 41 ms after `speech_started`, with no audio after it. The eviction case
passed: Drama 3 was back 8.0 s after the clip, with the ballast held. The leak checks (no `think>`, no repeated
first sentence) passed.

Card recipe against greedy, on the same loaded model: identical transcripts on all 21 qualify fixtures that hold
speech or silence, including the codec variants, and on the 8 e2e prompts, each in auto and fixed language
(56 pairs; silence and noise in auto only). The median e2e prompt took 362 ms with the recipe and 265 ms greedy. drama-tensorfold's 0.14 s figure is
TTS alone: request start to the first streamed codec frame, warm (`bench/drama_bench.py --stream`). Here that stage
is `TTS first samples`, about 130 ms.

## macOS (Apple Silicon): hearing

On a Mac the transcription tool is `tools/speech/whisper/macos`
([README](../../tools/speech/whisper/macos/README.md)). It is the same model, large-v3-turbo, through mlx-whisper on
Metal (the FP16 MLX conversion, `mlx-community/whisper-large-v3-turbo@a4aaeec0`). It speaks the same worker protocol
and returns the same `transcript.json`, with the same Silero gate on the CPU. The session's side is unchanged:
`decode: "greedy"` for voice turns, the session's language or `auto`, the silence prime and optional partials all
work as above. Two differences matter to the voice loop:

- **A dropped transcription frees the realtime lane at once.** The worker offers soft cancel (`"cancel": "signal"`).
  A job the session drops, such as a superseded partial or an early transcript that the turn outgrew, ends within
  milliseconds and the model stays loaded. The Linux worker has no soft cancel, so it finishes that decode and its
  result is ignored (`Scheduler.drop`). A hard cancel also keeps this worker loaded instead of killing it.
- **It is faster than the turn-end budget above assumes.** Measured on an M5 Ultra, warm, greedy: a 5.9 s turn takes
  0.09 s from request to transcript (0.13 s with the HTTP API's recipe). Cold start, the warm-up included, is 1.5 s.
  The worker is resident at 3.2 GB and peaks at 4.35 GB on long-form audio with word timestamps.

Through the daemon, the hearing path alone (`tools/speech/whisper/hearing.py`: fixtures streamed over `/v1/realtime`
with server VAD and `silence_duration_ms` 500, no chat model) heard every turn in en, de, fr, ja and es. The final
transcript arrived a median 466 ms after the speech ended, with a fixed language, and 590 ms with `auto` and partials
every second. The early transcript was usually the final one, so it was ready when VAD ended the turn.

Its transcripts are a Mac reference of their own, not the Spark's bit for bit. On the qualification fixtures, 15 of
18 are identical to the transformers recipe, and greedy equals the recipe on all 17 speech fixtures. The README lists
the rest.

## The web client

`src/voice/voice.html` is the page at `/voice`. It is one file, with no build step and no libraries, embedded in the
binary, and it works in desktop Chrome and Safari and in phone browsers. Chat and voice share one conversation.

**Layout and look.**
- It uses Sidebud's visual language: charcoal, ivory and amber, system fonts, and soft 14 to 24 px radii.
- Light and dark themes follow the system setting.
- On a wide screen a sidebar lists the conversations, with new, rename and delete (delete can be undone). On a phone
  the sidebar slides in as a drawer.
- The main pane holds the conversation, an empty state with suggestions, and a composer:
  - attachments, shown disabled for now;
  - the reasoning toggle;
  - a large mic button and the send button, which becomes stop while a reply streams.
- On a phone the page is as tall as the visual viewport, so the composer stays above the keyboard. It respects
  safe-area insets, and its touch targets are at least 44 px.
- Focus rings and accessible names are on every control, and reduced motion is respected.

**Chat.**
- It streams `/v1/chat/completions`. Each message part (thinking, body, footer) is updated in place, so streaming does
  not jump.
- Markdown uses a small built-in renderer: the text is escaped first, and links must be http(s) or same-origin.
- Code blocks are highlighted by a small tokenizer (keywords, strings, numbers, comments, calls) and have a copy
  button.
- Messages show their role, a timestamp, copy, and on the last reply, regenerate.
- **Reasoning** turns thinking on (`chat_template_kwargs.enable_thinking`) or off (plus `reasoning_effort: none`).
  `delta.reasoning_content` and inline `<think>` text go into a collapsible "Thinking… / Thought for N s" section.
- When image or video models exist, the page offers `generate_image` and `generate_video` and runs the calls itself:
  `/v1/images/generations` with `response_format: url`, or `/v1/videos` polled until the clip is done.
- The results are cards. A picture shows a shimmer while it is made, enlarges on a click and has a download button;
  a video plays inline and downloads. The results go back to the model as tool messages.

**Voice mode.**
- The mic button opens a full-screen view and starts a `/v1/realtime` session from that click, the user gesture that
  lets a phone play audio.
- An amber orb and Sidebud's 44-bar wave show each state: connecting, listening, hearing you, thinking (a turning
  ring), speaking, muted, and push-to-talk ready. Both follow the live level: the reply's own audio through an
  `AnalyserNode` while it speaks, the microphone otherwise.
- The transcript runs underneath: the draft transcript (`localrouter.transcription.partial`), then the final one, and
  the reply as it streams.
- **Hands-free** uses server VAD, and the big button mutes. **Push to talk** sets `turn_detection: null`: hold the
  button (or Space), release, and the turn is committed plus 200 ms of tail.
- Barge-in: tap the orb while it thinks or speaks, just talk, or press push-to-talk. Each one stops the audio, sends
  `conversation.item.truncate` with what was heard, and (by tap or press) `response.cancel`.
- An echo guard sends quiet microphone chunks as silence while the reply plays, until three loud chunks in a row say
  someone is talking over it.
- Minimizing keeps the session on: a bar above the composer shows its state, and typed messages get spoken replies.
- **Generations reel.** Voice mode shows every picture and video in the conversation, typed or spoken, newest first,
  next to the orb.
  - **Layout:** on a wide screen it is a column docked on the right. On a phone it is a strip along the bottom edge,
    below the orb and the controls, and it can be collapsed.
  - **Same items as the chat:** the reel reads the same tool messages as the chat's inline cards, so it is a second
    view of them, not a separate store.
  - **Progress:** a card appears the moment the tool call starts ("Creating a picture…"), turns into its thumbnail
    when the job is done, and pulses once. A refused or failed job shows its reason and **Retry**, which runs the
    same prompt again over HTTP.
  - **Cards:** each card has download and copy-prompt actions.
  - **Lightbox:** a card opens over voice mode, with the image full size or the video playing. The call goes on:
    barge-in by voice works, a call pill in the lightbox can interrupt the reply, and closing it returns to the orb.
- **Generations never spin for ever.**
  - The client keeps each job's id from `localrouter.job.created`. If `localrouter.job.done` is late (for example the
    socket was replaced, or the tab slept), it asks `GET /v1/jobs/{id}`.
  - A job that never starts (45 s), or makes no progress (image 4 min, video 25 min), fails with its reason.
  - The voice tools' descriptions tell the model to say only that it is making the picture when it calls the tool,
    and to say it is ready, or that it failed, when the result comes back as the tool output. Measured on DGX-01:
    "Absolutely, let me make that for you." at the call, then "Done! Here's your penguin." 7.7 s later, after
    `job.done`. A 2048×2048 request the image model refuses (1664 px at most) comes back failed and is admitted aloud.
- **New version notice.** A tab left open across a deploy keeps running the old page. The page compares itself with
  the one served (at load, every 10 minutes and when shown again) and offers **Reload** when they differ, except
  during a call.
- **Reconnects.** When the connection closes without being asked to (the service restarting, a proxy dropping it),
  the page opens a new session after 1, 2, 4 and up to 15 s, sends the settings and the conversation again, and keeps
  the microphone and the view. Per-item state is reset, since a new session numbers its items from 1. Voice ends only
  after about a minute of failures.
- **Playback start.** A reply's audio is held until 200 ms of it is in hand, or 150 ms after its first piece came,
  then scheduled back to back on the AudioContext clock.
  - Drama 3 streams a 46 ms first piece, then a 186 ms burst about every 125 ms (1.49 times real time). The first
    burst comes 120 to 185 ms after the first piece.
  - Started on the first piece (the earlier client, 50 ms after it), playback ran dry after the first syllable. The
    restart lead made that a 130 to 190 ms stall: the stutter at the start of every reply.
  - The 10 recorded DGX-01 delta traces, replayed through both schedulers: the old one stalled in the first 2 s of all
    10 replies, the hold in none of the 6 that streamed at full rate. Playback starts 86 to 110 ms later.
  - In the browser (`client_check.py`, output recorded after the mixer): before, 4 of 5 replies had a 133 to 175 ms
    gap in their first 2 s; after, none of the full-rate ones did.
  - Later underruns restart with a lead that grows by their lateness (0.1 up to 0.5 s), so a slow stream pauses
    rarely rather than stuttering.
- **A reply speaks while it is written** (since 2026-10-27, after JJ's "it's not even streaming at all": the text was
  all on screen before the voice began). `voice.speak_written` is off by default, so each sentence is spoken as soon
  as it is complete. #123 also capped the page's hold on a reply still being written at 1 s, which left pauses where
  speech, sharing the GPU with the chat model, fell behind; the start below replaced that cap.
- **No pause inside a reply: the start waits for it** (JJ, 2026-10-27, on #124's short pauses: "rather than there
  being pauses, just wait another few fractions of a second"). The page (and the companion with it, and the mobile
  app's `replyHold.ts`) starts a held reply once the audio in hand plays the rest without running dry at the rate it
  comes in: `in hand >= rest x (1 - rate) + margin`.
  - `rest`: the audio still to come for the text written so far (the whole text after `text_done`), at the voice's
    audio per character (learned per voice in the browser from its last replies, with 15% to spare; 65 ms at least).
  - `rate`: an EWMA (alpha 0.7) of the arrival rate over 150 ms windows from the first piece, seeded with the voice's
    rate on its last replies (1.0 on a fresh page), less half its standard deviation; 1 once the text is complete with
    the GPU free and the server's free rate above real time.
  - `margin`: 300 ms, none for a written reply on a free GPU (a short one starts on 200 ms in hand, as before). After a
    pause the session's next replies need twice its length more (up to 2 s).
  - A slow stream starts later in proportion: at a quarter of real time, three quarters of the rest must be in hand.
    #123's 1 s cap on the hold is gone; the absolute cap is 30 s; `speech_done` starts a reply at once.
  - Fitted on 30 recorded DGX-02 replies with speech first (Mae and Scarlette; a story, an explainer, a history, two
    tips, one sentence; a fresh session each), parameters chosen among those with no piece over 50 ms late in any
    reply, validated on a third of them held out: long replies started 1.8 to 3.3 s after the request (the best start
    with hindsight was 1.1 to 2.3 s), two-tip replies 1.3 to 2.0 s, one sentence 0.15 s after its first audio.
  - Measured live on DGX-02 the next night (84 long replies, Mae and Scarlette, typed and spoken, ten prompts, a fresh
    session each; `voice_e2e.py`'s replay of the page on each), that rule left a pause in 16 of them (0.07 to 0.5 s
    each): in the first second a long reply's speech often comes at real time, and slows once the chat model writes
    its next sentences beside it. So a reply whose text is still being written is now watched for 1.6 s from its
    first piece before it may start (`OBSERVE_MS`): 3 of the 84 then pause (0.07 to 0.4 s, each after 35 s or more of
    speech), long replies start 3.5 s after the request on average instead of 3.2 (the best start with hindsight: 2.7),
    and short ones, written before the wait ends, start as before (0.9 s on average after the turn).
    `voice_e2e.py` replays the same rule on its long reply and fails on any underrun over 50 ms.
- **Speech first** (`voice.speech_first`, on by default; `src/sched/yielder.zig`). While a reply is written, the
  session pauses its chat model's server (SIGSTOP; SIGCONT resumes it) whenever the reply's speech is under 2 s ahead
  of real time (counted from its first audio) and a written sentence waits behind the one being spoken, until speech
  is 3.5 s ahead, none waits, or the text is complete. Two processes on GB10 time-slice the GPU with no priority
  between them (CHAT.md); beside a decoding chat model Drama 3 came at 0.55x real time on DGX-02, alone at 1.4x, and
  the chat model writes about ten times faster than speech is spoken, so it keeps ahead even paused most of the time.
  It is lossless: the server's decode resumes where it stopped. Only the server is stopped, not the worker that
  supervises it (a dying daemon still reaches the worker, which kills it), and only on a lease that each piece of
  speech renews (600 ms): a speaker that stalls lets it run again within that. It applies when the chat model is this
  daemon's own server (`chat_url` unset, the model loaded here); elsewhere sentences are spoken beside it as before.
  The server is shared by every session and request, so a pause is only ever taken while nothing else waits on the
  chat lane, and anything queued there (another session's reply, from JJ's second device; a computer-use step; a
  prime) ends it at once (`Scheduler.yieldChat`, and `push` interrupts it). #124 lacked this: on DGX-02, 2026-10-27,
  a second session's request waited 6 and 17 s for its first token behind long replies whose chat model was paused.
  The session sends `localrouter.response.speech_first` `{response_id, item_id}` before the reply's first audio; the
  page and the mobile app then start a reply still being written once the audio in hand covers the text written so
  far (MS_PER_CHAR a character, less what of it is in) at the measured rate, still within the 1 s cap. Without the
  sentence of lookahead, a sentence boundary early in the reply ran dry 0.1 to 0.6 s while the paused chat model
  wrote the next one. Measured on DGX-02 (Drama 3 NVFP4, Qwen3.8 27B), a typed six-sentence story, a fresh session
  each run, the page's start replayed on the events (`voice_e2e.py`'s long-reply check):

  | | page start | text complete | pauses a reply: count, total, longest |
  | --- | ---: | ---: | --- |
  | spoken once written (#122), Mae x5, Scarlette x4 | 5.0 to 7.0 s | 4.6 to 6.7 s | none |
  | beside the chat model (#123), Mae x5, Scarlette x4 | 2.0 to 2.3 s | 7.0 to 11.1 s | 13 to 26, 1.9 to 3.6 s, 0.43 s |
  | speech first, Mae x5 | 1.9 to 2.3 s | 20.9 to 25.7 s | 0 to 2, 0 to 0.28 s, 0.18 s |
  | speech first, Scarlette x4 | 2.0 to 2.8 s | 17.1 to 32.3 s | 0 to 7, 0 to 1.02 s, 0.19 s |

  The response as a whole ends about as soon (22.7 to 41.4 s for 25 to 45 s of audio).
- **With `voice.speak_written` on, a reply is spoken once its text is written** (#122, its default for a day). The speaker takes the reply's
  sentences only when its spoken text is complete (the chat stream ended, or a tool call began), still one speech job a
  sentence. Chat and speech share the GPU, so splitting it bought nothing: on DGX-02 a six-sentence story was written in
  9.7 to 11.6 s beside speech at 0.55x real time (the chat alone: 4.8 s), and the page held its start for the writing
  (below), or a client playing at once ran dry 3.8 to 4.7 s in. Written first, the chat has the GPU to itself, and
  speech after it comes at 1.4x, so the reply plays from its first audio to its end. Measured with `voice_e2e.py
  --smoke` (its long-reply check) and a typed story over `/v1/realtime`, on DGX-02 (Drama 3 NVFP4, Qwen3.8 27B):

  | | first audio | text complete | response done | audio | late, playing from first audio + 0.2 s |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | before (Mae, 5 runs) | 1.1 to 1.6 s | 9.7 to 11.6 s | 27.7 to 52.0 s | 29 to 62 s | 3.4 to 4.7 s |
  | after (Mae, 4 runs) | 6.7 to 7.5 s (12.5 s first after a restart) | 6.5 to 7.3 s (12.3 s) | 31.8 to 54.0 s | 35 to 65 s | none |
  | after (Scarlette, a cloned voice, 3 runs) | 6.1 to 8.3 s | 5.9 to 7.3 s | 34.5 to 43.9 s | 40 to 52 s | none |

  So a long reply starts 3 to 5 s sooner on the page, which held it to its text's end (and played it gap-free from
  there). Short replies are written before their first audio either way: the smoke's turns had their first audio 1.07
  to 1.45 s after the speech ended before and 1.07 to 1.36 s after. `false` speaks each sentence as soon as it is
  complete, beside the chat model; that suits a box where speech keeps up while the chat decodes, and clients without a
  start hold. The page's hold (below) stays: it still covers a preamble spoken while tool arguments are written, and a
  picture or video being made.
- **Known limit, with `voice.speak_written` false: a longer reply still pauses near its start.** The chat model decodes the rest of the reply on the
  same GPU while Drama 3 speaks the first sentence. Measured on DGX-01, with a 600-token chat decoding:
  - Drama 3 drops from 1.49 to 0.67 times real time;
  - the chat stream slows from 20.2 to 33.1 s.

  Until the chat stream ends, speech arrives slower than it plays. The session sends
  `localrouter.response.text_done` `{response_id, item_id, tool_calls, chars, speech_rate}` as soon as the spoken text is complete: the chat
  stream ended (`tool_calls` false: from then on speech has the GPU to itself), or a tool call began (`tool_calls` true:
  the model still writes arguments, which are not spoken). `localrouter.response.speech_done` `{response_id, item_id}`
  follows once every sentence of the reply has been spoken and sent. A picture or video call with nothing said before it
  gets "Making that now." (the tool description asks the model to say so; it often did not, and the turn was silent
  until the picture was done, 12 to 15 s).
  - **The start waits for the writing when speech falls behind** (default; the "smooth start" setting is gone). A reply
    starts with 200 ms in hand once `text_done` has come (500 ms while a picture or video is being made), or, while the
    chat model still writes, once speech has come at 1.05x real time or better over its first 250 ms; otherwise it waits
    for `text_done` (at most 10 s). A short reply is written before its first audio, so it starts as before. The priority
    is a reply that plays to its end without running dry over time to first audio: a 1 s buffer (smooth start) still ran
    dry in a long reply, because a reply's length is not known until it is written. On DGX-01, `client_check` (five
    replies a run, each checked to its end): 3 of 3 runs with no underrun in any reply; the long reply waited 2.9 to 5.9 s
    for its writing, at 0.51 to 0.64x real time.
  - **Once written, the rest comes at the free rate.** From `text_done` with `tool_calls` false, speech has the GPU to
    itself, so what is still to come arrives faster than it plays, whatever the rate so far (which shared the GPU with
    the chat model). `speech_rate` on `text_done` is how fast speech came the last time it had the GPU to itself, in
    this process (null until measured): the audio a reply sent from its `text_done` to its `speech_done`, over that
    time, once it holds 0.5 s (a sentence's start counts in it, so a short span reads low: the safe side). The page
    uses it (else the last reply's rate) in place of the rate measured so far, when it is 1.05x or better; so a written
    reply starts with 200 ms in hand, on a session's first turn as on later ones. Before this, a reply still being written when its first audio came was held
    on its rate so far once `text_done` came (0.88 to 0.92x on DGX-01): 0.75 to 6 s.
  - **The first session after a start warms speech** (`voice.warm_speech`, on by default): on its own thread, it speaks
    one word ("Hello.") when it connects, the audio discarded. The speech model's first job after a load took 0.84 s to
    its first audio on DGX-01 (0.1 s warm), and a 2 s sentence 4.7 s in all; the engine speaks one request at a time, so
    a reply waits behind it, hence one word. Its rate is not kept (a first job's: 1.03x beside the chat prime, 1.45x
    warm; a second, warm sentence collided with the first turn). The first reply after a start measures the rate.
  - Replayed over 10 recorded default-length replies (15 to 35 s of audio, chat decoding for 3 to 10 s), first audio
    measured from the request:

    | | replies with a gap in their first 2 s | first audio median |
    | --- | ---: | ---: |
    | #28's hold alone | 8/10 | 1.24 s |
    | smooth start (1 s cap) | 0/10 | 2.80 s |
    | the whole text first | 0/10 | 4.69 s |
  - So no lossless client or session choice removes the stutter within about 1.2 s. Cancelling the chat stream and
    resuming it later is not lossless either: 0 of 6 greedy replies came back the same.
  - CUDA MPS with the chat server at client priority 1 (below speech) does not help on DGX-01 (GB10). Measured with
    `ct_probe` (one 10.6 s seeded speech request while a greedy chat request decodes, 3 trials each): speech came at
    0.62 to 0.63x real time without MPS and at 0.62x with it (1.45x alone), with the same chat and speech output bytes.
    Both decoders are bound by the shared memory bandwidth, which client priority does not divide. Not shipped.
  - The 130 to 195 ms a short reply is held at its start on DGX-01 is Drama 3's decode cadence, not the cushion: its
    first piece is one 46 ms codec frame, and the next 185 ms (four frames) comes in one piece about 140 ms later.
    Started on the first piece, playback would run dry for about 95 ms. A cushion measured from the last reply's
    lateness (tried, between 100 and 200 ms) stayed at 200 ms for that reason. A shorter start is a speech engine
    change: a second decode of fewer frames.

**Voice picker.**
- The voices are exactly what `/v1/models` lists for the speech model in use (names, or `{id, name, description,
  preview_url}`). The engine choice appears only when more than one speech model is listed.
- Each voice has a preview button. It plays `preview_url`, or speaks a short line through `/v1/audio/speech` with
  that voice's model. While the model is loading (a 503), the preview waits up to 90 s.
- **Clone my voice**, for a speech model that lists `voice_clone`: a dialog that asks the speaker to confirm the voice
  is theirs (nothing records or uploads before), then records them reading a script (echo cancellation, noise
  suppression and gain control off; a level meter; a quiet second first, 10 to 60 s) or takes up to 8 files, makes a
  draft with `POST /v1/voices`, plays a line in it, and saves it under a name (`PATCH`). Clips refused for a loud
  background (`noisy_background`) offer **Use anyway** (`force=true`); a draft's clip warnings show under the preview.
  Its list deletes cloned
  voices. Recordings stay in the page's memory until they are uploaded and are never stored by the page; closing the
  dialog deletes an unsaved draft ([DRAMA3.md](DRAMA3.md#cloned-voices)).
- The choice is kept per engine and sent as the session's `voice`. Settings the daemon no longer lists, such as a
  removed engine or Drama 3's single "default" voice from before its named voices, are dropped when the models
  load. Every sentence would otherwise fail with "has no voice". A session that refuses a voice anyway (an `error`
  naming it) gets the default instead.

**Settings drawer.**
- Chat model, reasoning, tools, max tokens and instructions.
- Engine and voice.
- Hands-free sensitivity (VAD threshold 0.8 to 0.2), the pause before a reply (`silence_duration_ms`), and the echo
  guard.
- An API key (only for a proxy in front), the debug log and deleting all chats.

**Status.**
- A pill in the top bar sums up the models' state from `/v1/tools`, polled every 4 s: ready, working, warming up or a
  problem. It also shows the last voice reply's latency.
- Opening the pill lists each model, the free memory and the last turns. For voice, that is from the end of your
  speech (the last loud microphone chunk, not the VAD's decision) to the first audible reply sample as this browser
  schedules it, with the server's STT, chat and TTS stages from `localrouter.response.timing`. For chat, it is the
  first token and tokens per second.
- **Copy debug log** copies the last N events: chat requests and results, realtime events without audio payloads,
  tool jobs and page errors, with the settings and the key redacted.

**Local only.**
- Conversations and settings live in `localStorage` (`jarvis.*`).
- An API key comes from a prompt on the first 401, from the settings, or once from `#key=` in the URL, which is then
  removed from the address bar. It is sent as `Authorization: Bearer` and as `?api_key=` on the WebSocket. Media is
  then fetched with the key into blob URLs.
- `window.jarvis` exposes the conversation, the voice and UI state and the debug log, so a test does not need to
  scrape the page.

`tools/spark/jarvis/client_check.py` checks the page in headless Chrome over the DevTools protocol, using only the
Python standard library, on the GPU host. It works against any running daemon, and `--page` writes a branch's
`voice.html` over the served page before it parses, keeping the real URL and CSP. It checks that:
1. The page loads, with the status pill and its model list, and every visible control has an accessible name.
2. A chat reply streams with Markdown, a highlighted code block with a copy button, a timestamp and actions, and the
   light and dark themes follow the system.
3. A picture renders as a card that enlarges and downloads.
4. On a phone, nothing is wider than the screen and the touch targets are large enough.
5. The voice picker lists engines and voices, and a preview plays.
6. With `--wav`, Chrome's fake microphone plays the WAV once in hands-free voice mode. The check sees the transcript,
   a spoken reply and its latency, and photographs the listening, hearing, thinking, speaking, muted and
   push-to-talk views.

It saves screenshots and `results.json`.

## Tests

- `src/voice/*.zig` unit tests: WebSocket framing (masking, fragments, control frames, size limits), VAD boundaries,
  resampling, sentence chunking, the chat SSE parser, and the session state machine with in-process fake STT, chat and
  TTS (turns, history, barge-in cancelling chat and TTS, tool call to job, error paths).
- `tests/voice.zig`: the real daemon and real workers: `tests/fixtures/speech_worker.py` (transcripts chosen by the
  tone's pitch, slow synthesis on request) and `tests/fixtures/voice_chat_server.py` (an OpenAI-compatible streaming fake
  that records requests and disconnects), over a real WebSocket: a spoken turn end to end, barge-in, a tool call that
  makes a testpattern image, the page, refusals.
