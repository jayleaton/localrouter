# Changelog

## Unreleased

- Voice: speech first no longer stalls other sessions. The chat model's server is shared, so it is paused only while
  nothing else waits on the chat lane, and a request queued there (another session or device, a computer-use step, a
  prime) resumes it at once. Before, a second device's request waited behind a reply's paused chat model (6 and 17 s
  for a first token on DGX-02).

- Voice: no pauses inside a reply. The `/voice` page, the companion and the mobile app start a reply once the audio in
  hand plays the rest without running dry at the rate it is coming in (`rest x (1 - rate) + 300 ms`; the rate an EWMA
  of the arrival, less half its spread, seeded per voice from its last replies; the rest from the text written so far
  at the voice's audio per character), so a slower stream starts later in proportion instead of pausing. #123's 1 s
  cap on the hold is gone. After a pause, the session's next replies need more in hand. On 30 recorded DGX-02 replies:
  no pause in any; long replies start 1.8 to 3.3 s after the request, two-sentence ones 1.3 to 2.0 s, one sentence as
  before. `voice_e2e.py`'s long-reply check replays the rule and fails on any underrun over 50 ms.

- Voice: speech first. While a reply is written, the daemon pauses its own chat model's server (SIGSTOP, lossless:
  its decode resumes where it stopped) whenever the reply's speech is under 2 s ahead of real time and a written
  sentence waits, until it is 3.5 s ahead (`voice.speech_first`, on by default; a lease each piece of speech renews, so
  a stalled speaker never leaves it paused). Beside a decoding chat model Drama 3 came at 0.55x real time on DGX-02,
  alone at 1.4x. A long reply still starts at about 2 s, while it is written, and its pauses fell from 13 to 26 a reply
  (1.9 to 3.6 s in all) to 0 to 7 (at most 1 s). The session announces it (`localrouter.response.speech_first`), and the
  `/voice` page, the companion and the mobile app then start a reply still being written once the audio in hand
  covers the text written so far at the measured rate.

- Voice: a reply speaks while it is written again. `voice.speak_written` is off by default (each sentence is spoken as
  soon as it is complete), and the `/voice` page, the companion and the mobile app hold a reply whose text is still
  being written at most 1 s, not until its text is complete. Held for its whole text, a long reply's voice began after
  all of it was on screen (5 to 7 s on DGX-02). Where speech, sharing the GPU with the chat model, falls behind, the
  reply pauses; `voice_e2e.py`'s long-reply check requires the start within 3 s and before the text is complete, and
  reports the pauses.

- Voice: a long reply streams again. Its speech starts once its text is written (`voice.speak_written`, on by
  default), not sentence by sentence beside the chat model: the two share the GPU, so on DGX-02 a six-sentence story
  took 10 to 12 s to write beside speech at 0.55x real time, and the `/voice` page and the companion held its start
  until then (or a player starting at once ran dry 4 s in). Written first, the chat has the GPU to itself (6 to 7 s)
  and speech then comes at 1.4x, so the reply plays from its first audio to its end without a gap, 3 to 5 s sooner.
  Short replies start as before. `voice_e2e.py` (and `--smoke`) checks a long reply streams (`--skip-long` to skip).

- Speech to text runs in a set language by default: `stt.language` (`en`) when a caller names none, for voice calls,
  `/v1/audio/transcriptions` and MCP `transcribe_audio`; `auto` (from the caller or the config) detects it. A voice
  session no longer carries the language its last turn was detected in: Whisper's detection on a 1 to 2 s turn is
  unreliable (on DGX-02 a clean "Stop." came back as Italian and as Korean), and one misheard turn held the rest of the
  call in that language, so clear English came back as other languages. The `/voice` page and the companion have a
  **Language** setting (English by default, Auto-detect, or any of Whisper's languages), kept per device and sent as
  each call's transcription language.
- Voice calls from other machines are on by default where they are gated: with `voice.remote` unset, a daemon that
  listens beyond loopback (`host` `all`, `tailscale` or an address) with `pairing.required` serves `/voice` and
  `/v1/realtime` to them (each call still needs a paired token with `inference`); without pairing, or on loopback,
  calls stay on the machine (behind `tailscale serve` every peer is loopback, as before). Discovery's
  `realtime.remote` now says whether a call that reaches the daemon is served (true behind a loopback-only listener).
  A daemon on the network with calls kept local refused the companion's socket with a 403 its WebSocket never showed,
  so a call said "voice dropped" and retried 8 times: the companion now reads the box's reason after a socket closes
  unopened and says it once (a box that is restarting is still retried), and refuses a call up front, and marks Voice
  calls in the Connect window, when discovery reports `realtime.remote: false` for a box that is not this computer.
- Voice cloning no longer refuses loud, mastered uploads as noisy. Each clip is heard whole by the transcription
  model, whose Silero detector says where the speech is, and the background is measured only between the words (not
  measured, and passed, when there is too little of it); upstream's quietest-tenth rule read loudness-normalised or
  compressed takes, whose quietest frames are soft syllables, as noise. Only clear failures are refused: no speech,
  under 10 s, or a background under 5 dB below the voice, which names music when it is tonal (code
  `noisy_background`; `force=true` or **Use anyway** in the `/voice` dialog keeps it). Everything else that was
  refused (a background under 30 dB, clipping, a clip cut inside a word, over 60 s or 8 references) is a warning in
  `clips[].warnings`, shown under the preview. A clip that makes one reference reuses that transcript.
- The MiniMax H3 pack build (`python -m stk_twin.h3.build`, `tools/spark/prepare.sh`) moves the DiT's block linears to
  the GPU once instead of holding a host and a device copy: 53.4 GiB at the peak instead of over 65 GiB (it was stopped
  at 5 GiB free beside the warm Jarvis set on DGX-02), and the pack is byte-identical to the pins
  (`tools/twin/packs/built/minimax-h3.json`, 6/6 files).
- `tools/twin/packs/built/qwen-image-2.1-turbo.json` pins the image packs as `prepare.sh` builds them since the edit pack
  (#78): `edit.json` and the te/ and vae/ packs with the vision tower and VAE encoder. The 2026-10-21 pins predate it,
  so the profile's `image/packs` check failed on every fresh build (4 of 10 files). The new pins come from DGX-02
  (2026-10-27), whose te/ and vae/ rebuilt without the edit step match the old pins byte for byte.
- The GPU Docker preflight (`tools/spark/install.sh`, `tools/package/install.sh`, the `jarvis-gb10` profile's `docker`
  prerequisite) also accepts Docker reaching the GPU through CDI (an `nvidia.com/gpu=all` spec from `nvidia-ctk`, no
  `nvidia` runtime registered), as on DGX-02: `--gpus all` works there, and the check refused it.
- Voice cloning for Drama 3: `POST /v1/voices` makes a voice from recordings of one person (10 to 60 s, uploaded or
  recorded in the `/voice` page and the companion, with a "this is my voice" confirmation), `GET /v1/voices` lists
  every voice, `PATCH`/`DELETE /v1/voices/{id}` name and remove one. Clips are trimmed, cut at pauses, levelled and
  refused with a reason when silent, clipped, cut inside a word or noisy (drama-tensorfold's gates); the transcription
  model writes missing transcripts; the engine's own `voice encode` (the pinned build, `codec.pth` already installed:
  no new weights) makes the codes. A cloned voice is spoken like a named one (speech, voice calls, MCP), with its
  references sent inline, so it hits the prefix cache and needs no restart. Voices are kept only in
  `data_dir/voices/` on the box; the clips are deleted once encoded. Creating and deleting need `manage`.
- The Drama 3 worker names the default voice in each request itself instead of starting the server with
  `--default-voice` (the same prompt rows, so the same audio), and checks at load that the server loaded it.
- The companion app's Connect window is a step flow in the `/voice` page's look: the Spark address (offered from the
  Mac's tailnet when Tailscale runs there, checked before a code is spent; a `localrouter://pair` link fills in both),
  the pairing code with what `pair.sh add` does, the connection (the box, the token's scopes, what it can use from
  `/v1/capabilities`), and the macOS permissions with the stale-grant card. Each failure is named with its fix: a host
  that does not resolve, answer or listen, a timeout, a certificate problem, something other than LocalRouter, a box
  too old, a wrong or expired code, a revoked token. **Re-pair** and **Disconnect** replace **Unpair**. The pairing
  protocol and auth are unchanged.
- `tools/jarvis/smoke.sh`'s `stt` check finds its pinned LibriSpeech fixture in `fetch.py`'s cache
  (`~/.cache/localrouter-speech/fixtures`) as well as under `~/jarvis`, and builds it with `fetch.py` (pinned, checked
  download) when it is nowhere. A cleanup of scratch directories under `~/jarvis` had removed the only copy, and every
  smoke failed `stt` with "no fixture" while speech to text worked.
- Job outputs no longer outlive a restart: on start the daemon removes job directories under `data_dir/jobs` older than
  `keep_outputs_s` (24 h). Expiry only knew the jobs of the running process, so every restart left the earlier ones on
  disk for good (on DGX-01: 6,483 job directories, 1.2 GiB, from three days of deploys).
- `tools/spark/jarvis/client_check.py` removes the Chrome profile it made on every way out (a failure before Chrome
  starts, Chrome exiting before its debugging port, `timeout`'s SIGTERM), after stopping Chrome's whole process group:
  the browser's helpers kept writing into it, and 16 profiles (160 MB) had piled up in `/tmp`.
- A voice call keeps its voice: while a realtime session is open, a request or an API load for a tool outside the
  voice loop (image, video, decision) never unloads a voice-loop model (chat, speech, transcription) of a higher
  priority. It is refused with 503 `insufficient_memory`, "busy: a voice call is open ...", which the chat model says.
  Before, a picture asked for in a call on DGX-01 loaded the cold image model and unloaded Whisper, then Drama 3 (two
  minutes without speech). `reserve_for`'s announced voice pause is unchanged. Each voice response's stage times
  (`localrouter.response.timing`) are also in the journal (`voice SESSION: timing {...}`).
- `tools/spark/window.sh`: a GPU window beside a live service (bench, test daemon, qualification) takes the box lock,
  waits for no voice session, unloads only the tools it names and loads them back, and is stopped (its process group
  and its labelled Docker containers) within 2 s of a voice session opening.
- Chat thinking budget (Chat Completions, `/v1/responses`, MCP `chat`): a request whose limit is below the chat
  model's `min_thinking_tokens` (tool config, default 1024; `0` never) and that does not choose reasoning itself is
  answered with thinking off (`thinking_off`, default `chat_template_kwargs.enable_thinking: false`). A reply still cut
  while reasoning keeps `finish_reason: "length"`. Both cases add an extension `warnings: [{code, message}]`
  (`thinking_off_for_small_budget`, `reasoning_used_budget`); a stream sends it in a last chunk with `choices: []` before
  `[DONE]`. Before this, "17 x 19" with `max_tokens: 16` came back as an empty `content`: all 16 tokens went to reasoning.
- Speech `instructions` (OpenAI's style field) on `/v1/audio/speech` and MCP `synthesize_speech`: a speech tool whose
  `instructions` option is a template takes them, written into the line. Drama 3 has no style slot of its own (its
  system prompt is fixed), so it takes them as its documented inline tag, `"[{s}] "`, at the start of the line. A tool
  without the option still refuses them (400). `/v1/models` lists `instructions` for each speech model.
- Model licences: tool config `license` (`name`, `url`, `use`, `commercial`, `excluded_regions`, `notice`), with engine
  defaults for Qwen-Image and MiniMax H3, listed in `/v1/models`, `/v1/tools`, MCP `list_models` and
  `/v1/capabilities` (`models.licenses`). One licence table in the README and docs/AGENTS.md. Informational:
  LocalRouter enforces none of it, and no model choice changed.
- Generated images and videos are labelled machine-generated in their metadata, whichever engine wrote them: PNG
  `tEXt` (`Software`, `Comment`) and XMP with the IPTC digital source type `trainedAlgorithmicMedia`
  (`compositeWithTrainedAlgorithmicMedia` for edits and image-to-video); MP4 `comment` and `description` tags. Pixels
  and coded streams are byte-identical. Files from `cmd` tools are labelled after the job. No C2PA (it needs a signing
  certificate).
- `tools/harness/nemo_check.sh`: a public review's test order as a client-side check anyone can run against a box
  (models, chat with thinking off, speech, Whisper on it, an image, an edit or its clear refusal, a video, chat again).
- Drama 3 on the NVFP4 pruned weights (plotarmordev/drama-3-NVFP4-Pruned: routed experts in NVFP4, 7,369 of 10,240
  kept; lossy, approved by JJ): the Drama 3 tool's `variant` option (`full`, the default, or `nvfp4`) with its own
  pins (`pins-nvfp4.json`, beside `pins.json`, whose marker it leaves valid), checkpoint, engine build
  (plotarmordev/TensorFold `drama-nvfp4` with LocalRouter's voice prefix cache ported) and marker
  (`qualify.sh --accept-lossy`). `fetch.sh --variant nvfp4` links files the full checkpoint already holds. On the
  Jarvis box (`DRAMA_VARIANT`, default `nvfp4`) it books 20.56 GiB against 39.25, and the KnowLine decision model is
  now kept warm (priority 15, above the image model). qualify's bench measures the server's own peak (nvidia-smi
  `used_memory` + RssAnon + RssShmem); `tools/speech/drama3/compare.py` compares speech models through a daemon
  (first audio, real-time factor alone and while chat writes, Whisper round-trip WER, WAVs side by side).
- `spare_room_only` (tool config): an on-demand model nothing waits on loads only into spare room. It never unloads a
  `keep_loaded` tool (idle unkept tools may still go), leaves room for the kept tools' working bytes, and is otherwise
  refused with a 503 `insufficient_memory` before anything is unloaded. The Jarvis decision model (KnowLine, behind the
  voice loop's shadow decisions) has it: a decision no longer evicts the warm image model (DGX-01 full smoke,
  2026-10-24), and without spare room shadow decisions skip the turns.
- Jarvis `wait_warm` (start.sh, deploy windows) waits until the box answers at warm speed, not only until every kept
  tool reports ready: primed rounds of a short chat turn, a sentence of speech and its transcription, three in a row
  within 1.0, 3.0 and 0.8 s. Right after a restart every tool was ready while a transcription took 1.7 s against 0.4 s
  warm, and the first full smoke's voice turns missed their budget.
- Responses API: `POST /v1/responses` serves OpenAI's Responses shape over the chat model, for harnesses that speak
  only that (Codex). Stateless (`store: false`, the conversation replayed each turn): message, `function_call`,
  `function_call_output` and `reasoning` items; `instructions` and `developer` messages as system messages in place;
  function tools with the model server's call ids; `reasoning.effort`; `text.format`; `max_output_tokens` and the
  `incomplete` status. Reasoning returns as a `reasoning` item, never as output text. Stored state
  (`previous_response_id`, `store: true`), hosted, custom and namespace tools, background mode, summaries, images and
  any unknown field are 400s naming the field, listed in `/v1/capabilities` (`responses`, `unsupported`). Needs the
  `inference` scope.
- Responses streaming and Codex: `stream: true` sends the Responses event sequence (`response.created`, per item
  `output_item.added`, text and reasoning deltas, `function_call_arguments.delta` / `.done`, `output_item.done`, then
  `response.completed`, `.incomplete` or `.failed`), with sequence numbers and the model server's call ids; closing
  the connection cancels. Codex CLI is qualified against the box ([docs/harnesses/CODEX.md](docs/harnesses/CODEX.md)),
  and what it sent is a contract fixture (`tests/fixtures/harness/codex.jsonl`).
- Voice picture turns: a reply cancelled by barge-in while it wrote a tool call no longer leaves the session's own
  "Making that now." in the history (the call is dropped, so the line promised a picture that never came, and the model
  copied it). `voice.temperature` sets a session's starting chat temperature; the Jarvis config runs Qwen with thinking off
  at 0.7 and `top_p` 0.8, its non-thinking sampling, instead of the server's 1.0 and 0.95, which left 1 in 4 spoken picture
  requests without the tool call.

- Realtime client tool execution: `session.update {"localrouter": {"tool_execution": "client"}}` makes a voice session's
  tools the client's alone. The built-in image, video, model and computer tools are neither offered nor run, and their
  names are free. Each call reaches the client once, complete; its `function_call_output` must answer a call still
  waiting for one: an unknown, second or late output is refused (`unknown_call_id`, `duplicate_call_output`,
  `call_expired`). A call stops waiting when the user speaks again or another response begins
  (`localrouter.function_call.cancelled`), so a late result is never spoken in a later turn. Server execution stays
  the default, and the voice pipeline (wait timer, early transcription, priming, speech) is the same in both modes.
  `examples/realtime-client-tools/` is a one-file harness that uses it. With pairing required, the mode needs the
  session device's `inference` scope.
- Service scopes: with `pairing.required`, every route but `/health`, `/v1/pair` and `/voice` needs a paired token
  with the scope it serves, over HTTP, the realtime socket and MCP alike (401 without a token, 403 without the scope,
  before the body or the upgrade). Pairing stays off by default.
  - `inference` (chat completions, `/v1/realtime`), `media` (images, videos, speech, transcription), `decisions`
    (`/v1/systemone`), `read` (models, voices, residency) and `manage` (load, unload, reserve, release). Changing
    models needs `manage`, whichever way it is asked. `localrouter pair token NAME --scopes inference,media,read` issues
    a harness a token; `GET /v1/pair` shows a token's scopes.
  - MCP `tools/list` shows only the tools a token may call; a voice session's picture and video tools need `media`,
    its model tools `manage`, checked at each call so a revoke or a scope change counts at once.
  - A job and its output files belong to the device that asked for them: `/v1/jobs`, `/v1/videos/{id}`, `/v1/files`,
    `get_job` and `cancel_job` answer only that device (others get 404).
  - Legacy tokens are unchanged: `voice` still opens `/v1/realtime`, `mcp` `/mcp`, and either (or none listed, as every
    device paired before scopes) grants every service scope. Computer control's scopes grant no service scope.
  - The Jarvis deploy scripts read `/v1/tools` with a `read` token install.sh makes once pairing is on
    (`$DEPLOY/operator.token`); `smoke.sh` sends `$LOCALROUTER_TOKEN` on every request.
- The harness contract (`docs/HARNESS.md`, contract `2026-10-23`): what a harness on another device may rely on
  across Chat Completions, the realtime beta profile, MCP, `/v1/systemone`, model management and pairing, and what
  is not offered. `GET /v1/capabilities` describes it per box (open, no secrets): profiles, which routes need which
  scope, the MCP tools and protocol versions, the realtime events, and `unsupported` features with their status
  (`coming` or `unsupported`). The requests our web page, mobile app and desktop helper actually sent the Jarvis box
  are fixtures (`tests/fixtures/harness`, recorded with `tools/harness/record_proxy.py`), replayed by the
  integration tests.
- Output URLs keep the client's scheme: behind `tailscale serve` (`X-Forwarded-Proto: https`) the URLs in MCP results,
  `response_format: url` images and `localrouter.job.done` are `https://`. They were `http://` on the TLS port, which
  answers 400, so an outside harness could not open what the MCP tools returned.
- Device pairing, off by default (`"pairing": {"required": true}`): `/v1/realtime` and `/mcp` then answer only a
  paired device's token, and refuse anything else with a 401 (`invalid_api_key`, `WWW-Authenticate: Bearer`) before
  the upgrade or the body. There is no loopback exemption (behind `tailscale serve` every peer is loopback); the other
  routes are unchanged.
  - A token is `Authorization: Bearer`, a `Sec-WebSocket-Protocol` entry `openai-insecure-api-key.<token>`, or
    `?api_key=` (what the page sends from Settings → API key).
  - `localrouter pair add NAME [--url URL]` makes a one-time code (XXXX-XXXX, 10 minutes) and the app's
    `localrouter://pair` link; `POST /v1/pair` trades it for a token, `GET /v1/pair` says whether a token is paired.
    `pair token NAME` mints one directly, `pair list` and `pair revoke ID|NAME` manage them; the daemon reads
    `<data_dir>/devices.json` (hashes only, 0600) on every check, so a revoke needs no restart. Five wrong codes
    within 10 minutes discard the pending codes.
  - `localrouter mcp-stdio --token T` (or `LOCALROUTER_TOKEN`); `deploy/jarvis/pair.sh`; `install.sh` keeps the
    deployed pairing setting (`PAIRING_REQUIRED=1|0` sets it); the voice and MCP check scripts send
    `$LOCALROUTER_TOKEN`.
  - A failed request's log line names its path without the query string, so a key never reaches the journal.
- Computer control: Jarvis can use the user's Mac through a paired desktop helper (`apps/desktop-helper`, ported from
  SideBud's companion; docs/dev/COMPUTER.md).
  - MCP tools `computer_control` (open or quit an app, volume, media, dark mode, lock the screen) and `computer_use` (a
    bounded on-screen task: 15 steps, 90 s of work), plus `computer_status` and `computer_stop`. The voice session's
    chat model gets the first two while a helper is connected, and says when a step waits for a tap.
  - The box grants every step (`src/computer/`): simple controls and benign steps run; sending, buying, deleting,
    posting, typing into a field that is not a search box, Enter after one, and locking the screen each wait for a tap
    in the web UI, bound to that one exact action for 60 s. Step decisions come from the chat model through a
    System One shaped decision interface, so the open Jev service can replace it.
  - Every task, grant, approval and outcome is in `<data>/computer/audit.jsonl` (`GET /v1/computer/audit`); typed text
    is kept as its length and a hash.
  - The helper's socket (`/v1/desktop`), the tools and the approvals need a paired device token with the right scope;
    computer control refuses without them even when `pairing.required` is off. Pairing gains per-device scopes
    (`localrouter pair token|add NAME --scopes LIST`, `localrouter pair scopes ID|NAME LIST`): a device with none keeps
    `voice` and `mcp`, and a desktop helper's token (`desktop_helper` only) reaches neither `/mcp` nor `/v1/realtime`.

- Typed decisions: a `decision` kind of model, `POST /v1/systemone` (TypeSafe's System One shape: choice, noul and
  score questions about a state) and the MCP `decide` tool.
  - Questions are checked before anything loads, and the model's answers against them before anyone reads them: each
    question answered with its own type, a choice among the offered options, probabilities in [0, 1] summing to 1.
    A System One client (SideBud's Jev client) only changes its base URL. At most 64 questions and 16 KB a request.
  - Decision tools run on their own scheduler lane (`decide`), so a decision never waits behind a chat reply.
  - Jarvis: KnowLine-4B Gen5.5 (`tools/decide/knowline`, Apache-2.0, bf16), the best open decision model that fits
    beside the warm set; on demand at priority 5, not kept warm (its booking would leave under 5 GiB for jobs).
  - Voice (`voice.shadow_decisions`, on in Jarvis): after each reply the decision model is asked how it would have
    routed the turn among the tools the chat model was offered (or a spoken reply), only when it is already loaded,
    one at a time, and the answer is logged beside what the chat model did (`voice shadow decision {...}`). Nothing
    acts on it.
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
