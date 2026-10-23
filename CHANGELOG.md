# Changelog

## Unreleased

- Voice: a picture or video turn answered in several tool calls is confirmed aloud, and the chat model no longer manages
  models around a picture.
  - The session checks memory itself: before a video, the `reserve_for` dry run, and when a voice model would go the
    user hears that the voice pauses; after a job, its result lists any model still coming back (`coming_back`). The
    model management instructions (`voice.manage_models`) now keep its tools for when the user asks, so a picture is
    one `generate_image` call (on DGX-01 the chat model called `list_models` before and after it).
  - `voice_e2e` (the smoke's picture turn) judges the turn's calls as a set (`generate_image` alone), and takes the
    confirmation from the responses after the picture's output up to the first that makes no further call.
- Voice: a session's first turn, and any reply still being written when its audio starts, start like later short ones.
  - `localrouter.response.text_done` carries `speech_rate`: how fast speech came the last time it had the GPU to
    itself. Once a reply is written, the page holds its start for that rate (200 ms in hand), not for the slower rate
    it came at while the chat model was still writing it.
  - The first session after a start speaks one discarded sentence when it connects (`voice.warm_speech`), so the speech
    model's slow first job is paid before a reply needs it.
- `client_check`: a run that flags a click keeps the flagged replies' received audio (`received-*.wav` and `.json` in
  its output directory; at most 3 replies of 120 s), so the flag can be classified. `--save-received` keeps them
  unflagged too.
- Voice: the answer starts as the wait ends.
  - The turn's words are transcribed while the speaker is quiet (150 ms into a pause, `voice.early_stt_ms`). If the
    wait ends with nothing more said, that transcript is the turn's, so speech to text no longer runs after the wait.
    Nothing considers the words before the wait ends.
  - A reply already written, with no picture or video being made, starts at the arrival rate the previous reply
    measured (if it kept up), instead of measuring for 250 ms first.
- `client_check`: clicks are judged where a fault can be told from speech.
  - Playback: the output must equal each received piece once aligned on the audio clock; a sample off by over 0.05 is
    playback's.
  - Stream: only where received pieces join, a step at least 0.05 and 8x the steps around it.
  - Speech inside a piece is not judged: its plosives and sibilants have steps as large. The earlier measure, a fixed
    threshold on any step, counted them; it is still reported as `clicks_old`.
  - `window.jarvis.clickTest` checks this on each run's own audio. Recorded speech at a pretended seam
    (`click_fixtures.json`) and the clean output and seams must not count. Splices into the output, and a step or a
    dropped 10 ms at a received seam, must.
- Voice picture turns:
  - The spoken text ends where the model's tool call begins (`localrouter.response.text_done` with `tool_calls: true`); the
    arguments are never spoken. The page then waits for that short speech to be all sent
    (`localrouter.response.speech_done`, new), not for the arguments to be written.
  - A reply waits until its text is complete and enough audio is in hand to play all of it at the rate it is coming in
    (`text_done` carries the spoken length; about 65 ms of audio a character, 25% margin), or all its speech is in
    (`speech_done`). A fast stream starts at 200 ms in hand. Slow speech beside a tool call's arguments or a picture
    being made no longer starts too early.
  - A picture or video call with nothing said first gets "Making that now." (a video adds that it takes a few minutes),
    as the tool description asks. It is recorded as the assistant's text.
  - The page plays at the wire rate (a 24 kHz AudioContext) and schedules pieces on whole frames: one resampling of
    the mixed output, not one per piece (a seam click).
- Voice page: a reply plays to its end without running dry. While the chat model is still writing it and speech is not
  keeping up (under 1.05x real time over its first 250 ms), its start waits for the writing to end (`text_done`, at most
  10 s), then 200 ms in hand. Short replies are written before their first audio and start as before. The "smooth
  start" setting and the rate-sized hold of #51 are gone. `client_check` now fails on any underrun anywhere in a reply.
- Voice page: a reply no longer starts with only Drama 3's 46 ms first piece in hand.
  - While the chat model is still writing, its start waits for enough audio to play its first 3 s at the measured
    arrival rate: 3 s x (1 - rate), at least 200 ms, at most 1 s in hand, at most 1.5 s after the first piece. A stream
    that keeps up does not wait.
  - Once the chat model is done, the start still waits for 200 ms in hand.
  - On DGX-01, `client_check` passed with no underrun in any reply's first 2 s in 7 of 7 runs, against 1 of 3 before.
    First audio: short typed replies median 948 ms (1074 before); the long reply median 2.9 s (2.6 s before).
    "Smooth start" stays off by default.
- `client_check`'s playback test now fails on delivery underruns (the page scheduler's own measurement) and clicks,
  not on silence in the recorded output.
  - Drama 3's own audio has no exact-zero runs (24 replies probed). The 658 and 729 ms "gaps" came from the recording
    losing its alignment with the AudioContext clock across the test's reconnect.
  - Recorded silences are still reported, as delivery or render gaps.
  - The voice step reads the reply that answers the spoken turn: since #44, a tool turn can add more responses.
- Voice page: while a picture or video is being made, a reply starts once 500 ms are in hand (900 ms at most),
  not 200, so the denoise step already on the GPU cannot stall its first words. Smoke budgets now include the wait:
  warm turns 1.4 s from the end of speech (the 500 ms wait plus STT, chat and speech), 1.7 s after a picture or video.
  `client_check`'s gap test ends each window with the reply's own audio (the quiet before the next reply is not a
  gap).
- Speech first: while the daemon makes speech, image and video workers pause between their diffusion steps (and
  before the VAE), so a reply never stutters while a picture or clip generates. Each wait is at most 6 s and a job's
  waits at most 30 s in all. The gate is a flock on `<data>/locks/speech.gate`. On DGX-01, streamed Drama 3 replies
  played during a picture had a gap over 30 ms in the first 2 s in 8 of 8 trials before and 0 of 8 after (real-time
  factor 0.55 to 0.75 before, 1.15 to 1.27 after). A picture took 8.3 to 8.5 s (13.7 s alongside a 10 s reply).
- Voice: "Wait before replying" is the end of the turn: 0.3, 0.5, 0.8, 1, 1.5, 2, 3, 4 or 5 s, default 0.8 s, kept in
  settings. It replaces "Pause before I reply", and stored values start again at 0.8 s.
  - The turn ends on one unbroken quiet run that long. Speech restarts it, so pauses never add up. In a pause, a
    loud run shorter than 120 ms (a click, a knock) is not speech and does not restart it.
  - Nothing starts before the turn ends: no transcript, no chat request, no speech and no tool call. The whole
    utterance is transcribed once, after the wait.
  - Removed: the draft transcript at the first pause, the 300 ms early end on a finished sentence, and the
    speculative reply. Their options `draft_after_ms` and `early_end_ms` are gone: remove them from a config.
- Voice: a call in a conversation that already has pictures makes new ones. The page seeded the history as text
  only, so the model saw requests answered without a tool call. It then said "Making you one now!" and called nothing,
  and no card appeared. The page now seeds each generation as a `function_call` and its output, and the realtime
  session accepts `function_call` items (as OpenAI's does). `client_check.py --profile` keeps the profile's
  conversations and closes Chrome cleanly, so its storage is kept.
- Voice mode shows a reel of the conversation's pictures and videos beside the orb (a bottom strip on phones).
  - Cards: a placeholder as the tool call starts, the thumbnail when the job is done, and a failed state with Retry.
    Each card has download and copy-prompt actions.
  - The lightbox opens over the call without ending or muting it.
  - Generations can no longer spin for ever: the client asks the new `GET /v1/jobs/{id}` when a result event is
    missing, and fails a job that never starts or stalls.
  - The voice tools tell the model to say it is making the picture, not that it is ready, until the result arrives,
    and to admit a failure.
  - A voice the model does not list is never sent.
  - A tab running a page older than the deploy offers a reload.
  - It now checks as each call opens (and after a reconnect, which is how a deploy reaches an open call), and offers the
    reload in a call too. A tab that predates the reel cannot hear of it: reload it once.
  - The placeholder shows even when the page missed the call's item (it is made from `localrouter.job.created`). A
    picture the chat asked for over HTTP that a reload interrupted is marked failed instead of waiting for ever.
  - On phones the reel shows as a strip under the call. The chat sidebar's off-canvas `aside` rule had moved it off
    screen, so that rule is now scoped to `#side`. With the reel open, the stage keeps the orb, its state and the reply.
- A video request without a size gets the model's own size: 768x448 for MiniMax H3 by default, from its `max_width`
  and `max_height`. Before this, voice, the web client and `/v1/videos` asked for 1344x768. H3 evicted the image
  model to load, then failed with a bare "generate failed: Refused". H3 now refuses a size, length, frame rate or
  first frame it cannot serve before it loads, and says why. `engine.Entry.defaults` is the new hook for this.
- Voice: the session sends `localrouter.response.text_done` when the chat model stops writing a reply. The client's
  new "smooth start" setting (off by default) uses it to hold a slow reply's start, so long answers don't stutter
  while speech shares the GPU with chat decoding. It costs about 1.5 s of first audio on such replies.
- Jarvis client voice fixes:
  - A voice call survives the connection closing (a service restart): it reconnects with backoff and goes on. It used
    to show "Voice disconnected" and stop.
  - Saved voice settings the daemon no longer lists (Drama 3's "default" voice from before its named voices, a
    removed engine) are dropped. They had made every spoken reply fail with "has no voice" and made previews fail.
  - The voice picker shows exactly what `/v1/models` lists, with no engine choice when there is one engine.
  - Previews wait for a loading model.
- Jarvis client: a spoken reply no longer stutters as it starts.
  - Its audio is held until 200 ms is buffered, or 150 ms after the first piece, then scheduled back to back. Drama
    3's 46 ms first piece used to run dry before the next burst.
  - The cost is 86 to 110 ms on first audio.
  - `client_check.py` records the page's output and fails on a playback gap over 30 ms, or a click, in a reply's
    first 2 s.
- Drama 3 speaks in fixed voices: six named speakers the model itself invented (`Mae` by default, `Arlo`, `Ezra`,
  `Nora`, `Theo`, `Wren`), pinned and installed by `fetch.sh`, so every sentence of every reply is the same voice.
  The built-in voice was a different speaker for each sentence. A pinned engine patch keeps each voice's prefilled
  reference, so a sentence in a voice reaches first audio in 0.13 s, as the built-in voice did, with byte-identical
  output. A `voice` without a `model` (HTTP, MCP and the realtime session) goes to the speech model that lists it, so
  one picker offers every model's voices ([docs](docs/dev/DRAMA3.md#voices)).
- Jarvis web client, a UI pass in Sidebud's look, with system light and dark themes:
  - a sidebar of conversations (rename, delete with undo);
  - messages with roles, timestamps, copy and regenerate, highlighted code blocks with copy, and a collapsible
    thinking section;
  - image and video cards that enlarge and download;
  - a full-screen voice mode with an orb and wave for listening, hearing, thinking, speaking, muted and push-to-talk,
    a live transcript, tap-the-orb barge-in and big push-to-talk and hands-free controls;
  - a voice picker with previews, read from a voice-list API when the daemon has one and from `/v1/models` otherwise;
  - a settings drawer and a status pill in place of the debug strip;
  - a mobile-first layout with 44 px targets and a composer that stays above the keyboard;
  - focus states, accessible names and reduced-motion support.
- Web client (Jarvis) at `/voice`, with `/` redirecting to it: one embedded HTML file for chat and voice in one
  conversation.
  - Chat: streamed chat with Markdown, a reasoning switch, stop and regenerate.
  - Voice: hands-free or push to talk over `/v1/realtime`, with interim and final transcripts, streamed audio, and
    barge-in by voice or tap. You can choose the speech model and the voice.
  - Images and videos from tool calls show inline, from chat and from voice.
  - A status strip shows model state and per-turn latency.
  - Conversations are kept in `localStorage`, and a "copy log" button copies the debug log.
  - A voice session's draft transcript is now sent as `localrouter.transcription.partial`.
  - `tools/spark/jarvis/client_check.py` checks the page in headless Chrome.
- Chat: a new kind, `chat` (`chat_completion`), behind OpenAI's `POST /v1/chat/completions` (whole and server-sent
  events) and the MCP `chat` tool. The first model is Qwen3.8 27B on TensorFold 1.1.1's own server (MLX 4-bit with the
  DFlash2 drafter), run by the new `openai_server` worker engine, which supervises a model's OpenAI-compatible server
  on loopback while the daemon proxies requests to it. A chat tool is `keep_loaded` with
  priority 100 by default; cancelling a request leaves it loaded. Gated by a per-model `qualification` file until
  `tools/chat/qwen38-27b/qualify.sh` passes on the machine (docs/dev/CHAT.md).
- Host GPU lock (`gpu_lock`): the scheduler holds it once, shared, while any model is loaded, against GPU users
  outside LocalRouter; its own models are admitted by memory alone and never take it, so chat and voice stay warm
  together. Workers get `LOCALROUTER_GPU_LOCK_HELD=1`, the host lock in `LOCALROUTER_HOST_GPU_LOCK`, and a private
  `<data>/locks/<id>.lock` in `LOCALROUTER_GPU_LOCK` for a launcher that insists on taking one.

- Fish Audio Drama 3 text to speech (`fishaudio/drama-3`, `tools/speech/drama3`; Built with Fish Audio): a `cmd`
  worker that runs plotarmordev/drama-tensorfold's native server (TensorFold's Zig/CUDA engine with its Drama 3 patch
  series), built from pinned sources and started by that repository's own launcher on loopback. LocalRouter's scheduler admits
  it (`resident_bytes` 42 GiB: the measured 40.45 GiB peak plus 1.55, 1.5 times the spread of nine measured peaks;
  it was the peak plus 8, which booked 8.5 GiB Drama 3 never uses): a pinned patch adds `--external-admission` to the launcher,
  which turns off its own 88 GiB admission and 34 GiB abort and lowers the engine's 34 GiB reserve, so it stays warm beside chat and Whisper. Its worker declares soft cancel (`"cancel": "signal"`: SIGUSR1 ends a generate and keeps the model loaded). Whole and streamed replies, voice files and inline `references`, long form.
  It is listed but refused until GPU-qualified: the new per-tool `qualification` file gates any model with a 503
  saying it is not GPU-qualified on this host until the model's qualify script (here `qualify.sh`) says
  so. Qualified on DGX-01: first audio 0.142 s, warm RTF 0.663, beside the chat model too ([docs](docs/dev/DRAMA3.md)).
- Speech API: `stream: true` (with `response_format: "pcm"`) streams raw PCM as HTTP/1.1 chunks for models that
  offer it; `x-finish-reason` (`stop`, `length`, `silence`) and `x-long-form-segments` as headers, or as a stream's
  trailers; a stream that fails after it started ends without its last chunk. `references` (voice references) for
  models that take them, over HTTP and MCP. Workers may refuse with `service_unavailable` (503) and report
  `finish_reason` and `segments`.

- Voice: a full-duplex conversation on `GET /v1/realtime`, a WebSocket speaking the OpenAI Realtime API's events
  (pcm16 24 kHz in and out), and a built-in browser client at `/voice`. Server VAD ends each turn; the turn is
  transcribed (partials while speaking), sent with the session's history to the chat model (OpenAI streaming
  `/v1/chat/completions` at `voice.chat_url`, this daemon's own by default), and spoken back sentence by sentence.
  Talking over the assistant stops its audio within the chunk that crossed the threshold and cancels the chat stream.
  The chat model can call `generate_image` and `generate_video`; the job's file URLs come back as
  `localrouter.job.done` and the assistant confirms aloud. Loopback peers only unless `voice.remote`
  (docs/dev/VOICE.md).
- Scheduler lanes: a second executor, `realtime`, runs speech and transcription tools (by default; `lane` in a tool's
  config) beside image, video and chat jobs, so a TTS sentence never waits for a whole reply or a video. Residency
  changes stay serialized across lanes and admitted-but-unallocated memory is held back from MemAvailable.
  `/v1/tools` lists each tool's `lane`.
- Speech: two new kinds, `speech` (`text_to_speech`) and `transcription` (`speech_to_text`), behind OpenAI's
  `POST /v1/audio/speech` (wav, pcm, flac, mp3, opus, aac) and `POST /v1/audio/transcriptions` (json, text,
  verbose_json, srt, vtt; segment and word timestamps), and the MCP tools `synthesize_speech` and `transcribe_audio`.
  Uploads are decoded by the daemon to 16 kHz mono WAV with ffmpeg, bounded (fixed demuxer list, local files only,
  deadline, 32 MB and 30 minutes, one decode at a time), so adapters never decode and jobs are priced by duration.
  Per-tool `voices`, `default_voice`, `presets`, speed range, `max_input_chars` and `max_audio_s` are checked before a
  tool loads and listed in `/v1/models`. Translation, streaming and speech `instructions` are refused, not
  approximated. A WAV upload cut short (data chunk longer than the file) is refused, not half decoded. The testpattern
  engine serves both kinds for tests (`testpattern-speech`, `testpattern-transcription`).
- Whisper large-v3-turbo speech to text (`tools/speech/whisper`, a `cmd` worker on pinned transformers/torch): the
  model card's decoding, checked bit for bit against that recipe on GB10 (`twin.py`), and a voice activity gate
  (Silero VAD v6.2.3, pinned) that drops segments where no voice is heard, so silence and noise give empty text.

- Qwen-Image 2.1 Turbo (`qwen_image_turbo`, installed as `qwen-image-2.1-turbo`): the official Turbo checkpoint on
  the same native Zig engine, with its own 8-step schedule (the checkpoint's `sample_sigmas`, shift 1, CFG 1). On GB10
  its sigmas, context, latents and pixels equal the twin's; the bf16 twin's cosine gate against diffusers fails at
  sigma 0.2 for Turbo and for Qwen-Image 2.1 alike (`docs/dev/RESULTS.md`). Its FP8 pack is calibrated and digest-checked like the base model's; the text encoder and
  VAE packs are shared (the same tensors). The installer adds it beside Qwen-Image 2.1 and makes it the default image
  model (`IMAGE_MODELS`, `IMAGE_DEFAULT`).
- Model choice per capability: the config's `defaults` names the model a request without `model` gets (checked at
  start); `/v1/models` and `list_models` show each model's `default_for`; `localrouter models` prints them. Unknown and
  wrong-kind models are refused with the models that can serve the request; engines can refuse requests they cannot
  serve (sizes, step counts) before loading (`Entry.check`).

## v0.1.0

First public release.

- One daemon with per-model worker processes, memory budgeting, priority eviction, idle unloading and warm models.
- MCP over HTTP and a stdio bridge; image and video HTTP endpoints, job polling and cancellation.
- Qwen-Image 2.1 text-to-image and MiniMax H3 text-to-video with audio engines on NVIDIA DGX Spark.
- Local conversion of publisher checkpoints, container service installation and a GPU-free test engine.
- Input URL validation, public-address checks, pinned connections and no redirects.
- Listens on `127.0.0.1` by default. `host` and `--host` take an IP, `localhost`, `all` or `tailscale` (the machine's
  tailnet IPv4, found from the network interfaces). The container runs `--host all` behind a published port that
  defaults to `127.0.0.1`; the installer's `LOCALROUTER_BIND` takes `localhost`, `all`, `tailscale` or an IP, asks once on
  a terminal, keeps the choice in `.env`, and serves a tailnet-only install from the host network so it starts at boot
  whichever of Docker and Tailscale comes up first.
- Qwen-Image 2.1 defaults to FP8 (closer to the original model): 1024x1024 in 16.7 s warm, about 22 s from cold on a DGX
  Spark. NVFP4 is the faster option (12.7 s warm, 16.1 s cold): `PRECISIONS="fp8s nvfp4"` installs it as
  `qwen-image-2.1-nvfp4`.

Image edit and image-to-video inputs are accepted by the API but refused by both model engines until their ports
land. The full two-model installer still needs an end-to-end hardware check. There is no authentication.
