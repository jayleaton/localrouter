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
| VAD | Energy based, in the daemon, per 20 ms frame against an adaptive noise floor. `threshold` (0 to 1) sets the margin over the floor (6 + 24 x threshold dB); `prefix_padding_ms` and `silence_duration_ms` as OpenAI's `server_vad`. `turn_detection: null` is push-to-talk (`input_audio_buffer.commit`) | Runs on the reader thread in microseconds, so barge-in needs no model and no GPU. Whisper's own Silero gate still drops non-speech from transcripts. Echo is the browser's job (`echoCancellation`) |
| Barge-in | On `speech_started` during a response, on the reader thread: the response is marked cancelled, the chat stream's socket is shut down, queued TTS jobs are dropped, the speech job in progress is ended in its worker when the worker offers the soft cancel (`"cancel":"signal"` in its load reply: SIGUSR1, no reload), and no further audio is sent. `response.done` reports `cancelled` / `turn_detected` | Within the audio chunk that crossed the threshold. A TTS job already running finishes and is discarded: killing its worker would cost a reload on the next turn |
| History | Per session: the system `instructions`, then user, assistant, tool-call and tool-result messages. An interrupted reply keeps only the sentences whose audio was sent, and `conversation.item.truncate` cuts it to what was played. The oldest turns are dropped past `max_history_chars` | The model sees what the user actually heard |
| STT | The final transcript runs on the whole turn when VAD commits it. While the user speaks, a partial can run every `partial_interval_ms` of new speech (default 0, off: on one Whisper a partial in flight holds the final back by a whole decode), never more than one at a time, reported as `localrouter.transcription.partial` | Whisper is not a streaming model; re-running it on the growing turn is the standard way to get partials. Partials replace each other, so they are not OpenAI's append-only `delta` |
| Chat | OpenAI `/v1/chat/completions` with `stream: true`, behind `voice.Chat` (`src/voice/chat.zig`). `voice.chat_url` defaults to this daemon's own `/v1`, so the chat tool works the day it lands; any OpenAI-compatible server works too | One contract, no coupling to how the chat model is served |
| TTS | A sentence chunker over the streamed text; one speech job a chunk; the first chunk may end at a clause (a comma after 24 characters). Within a sentence the audio is streamed: the session tails the job's `speech.wav` as the model writes it, resamples it piece by piece (`resample.Stream`, the same samples as the whole) and sends 40 to 100 ms deltas | First audio when the model's first audio exists, not after the whole reply or sentence |
| Leaked reasoning | `<think>…</think>` spans are dropped from the stream (`chunker.ThinkFilter`). A reply's start is also held (`chunker.Lead`) until its first sentence has 24 characters after it: a `</think>` that was never opened (a model writing its reasoning as plain text) drops everything before it, and later stray tags are removed. Send the server's own switch too (`voice.chat_extra`, e.g. `{"chat_template_kwargs": {"enable_thinking": false}, "reasoning_effort": "none"}` for TensorFold's Qwen) | Measured on DGX-01: with thinking off, Qwen3.8-27B still answered "The capital of France is Paris.\n</think>\n\nThe capital of France is Paris." once; the voice must neither say the tag nor the answer twice |
| Lanes | The scheduler gets more executors: `realtime` for speech and `listen` for transcription (a tool's `lane` overrides its kind's default; with the chat kind of #14, chat gets its own). Residency changes (admission, eviction, loads) are serialized across lanes; memory admitted but not yet visible in MemAvailable is held back | With one executor a streaming chat job would hold the GPU queue, and sentence 1's TTS would wait for the whole reply. A turn's transcript must not wait behind a sentence of speech either (measured on DGX-01: the barge-in turn's transcript 6.1 s after speech end on a shared lane, 2.3 s on its own) |
| Tools | The chat model is offered `generate_image` and `generate_video` (when such models exist; `voice.tools`). A call queues a normal LocalRouter job; the session waits for it in the background, sends `localrouter.job.done` with the file URLs, adds the result to the history and creates a response so the assistant confirms it aloud. Client-defined `tools` follow OpenAI's flow (the client returns `function_call_output`) | The conversation continues while the job runs; files are ordinary `/v1/files/...` outputs |
| Client | `/voice`: one HTML file with inline JS, embedded in the binary. Mic through an AudioWorklet at 24 kHz, playback scheduled gap-free, flushed on `speech_started` | No build step. `getUserMedia` needs a secure context: `http://127.0.0.1` is one; remote use needs HTTPS in front |
| Exposure | `/voice` and `/v1/realtime` answer loopback peers only unless `voice.remote` is true | A microphone endpoint should not appear on the network because the image API was opened up |
| Memory | Chat, STT and TTS are `keep_loaded` with priorities above the image and video tools; the scheduler decides the rest. Voice job directories (user audio, TTS audio) are deleted within a minute of use | The loop never forces the chat model out; utterances do not sit on disk for 24 h |

## Session configuration

`session.update` takes OpenAI's fields: `modalities` (`["audio","text"]` or `["text"]`), `instructions`, `voice`,
`input_audio_format` / `output_audio_format` (`pcm16`), `input_audio_transcription` (`{model, language, prompt}`; the
transcription model), `turn_detection` (`{type: "server_vad", threshold, prefix_padding_ms, silence_duration_ms,
create_response, interrupt_response}` or null), `tools`, `tool_choice`, `temperature`,
`max_response_output_tokens`. Extensions: `model` (the chat model; also `?model=` on the URL), `speech_model` (the
TTS model), `partial_interval_ms`. A name that is not a model of the right kind is an `error` event naming the models
that can do it, as the HTTP API's refusals do. Defaults come from the daemon config:

```json
{"voice": {"remote": false, "chat_url": "", "chat_api_key": "", "chat_model": "", "speech_model": "",
           "transcription_model": "", "voice": "", "instructions": "You are a helpful voice assistant...",
           "partial_interval_ms": 0, "max_history_chars": 24000, "tools": true, "tool_wait_s": 1800}}
```

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
(`message` with `input_text`, or `function_call_output`), `conversation.item.truncate`, `conversation.item.delete`,
`response.create`, `response.cancel`. Anything else is an `error` event; the session stays open.

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

## Tests

- `src/voice/*.zig` unit tests: WebSocket framing (masking, fragments, control frames, size limits), VAD boundaries,
  resampling, sentence chunking, the chat SSE parser, and the session state machine with in-process fake STT, chat and
  TTS (turns, history, barge-in cancelling chat and TTS, tool call to job, error paths).
- `tests/voice.zig`: the real daemon and real workers: `tests/fixtures/speech_worker.py` (transcripts chosen by the
  tone's pitch, slow synthesis on request) and `tests/fixtures/voice_chat_server.py` (an OpenAI-compatible streaming fake
  that records requests and disconnects), over a real WebSocket: a spoken turn end to end, barge-in, a tool call that
  makes a testpattern image, the page, refusals.
