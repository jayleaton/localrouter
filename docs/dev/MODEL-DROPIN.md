# Model drop-in: audit and contract

A new model of a kind LocalRouter already serves should be one adapter plus one config entry. MCP, the OpenAI routes
and the voice loop should not need any change. This page records what the audit of 2026-10-28 (base `87ea48b`) found
and what was fixed. It also states the contract each kind follows and how the conformance tests prove it.

## Verdict

The orchestration was already shaped for drop-ins:

- the scheduler and the worker protocol know only kinds and capabilities;
- every request resolves its model through `routes.resolve` (a named model, else the config's `defaults`, else the
  first of the kind);
- a model that lacks a capability is refused before it loads, with a message naming the models that have it.

The leaks were in the layers around it. The voice loop named two speech models in code. Voice cloning assumed Fish
Audio's codec (10 codebooks, 44.1 kHz). The model-management prompt hard-coded the roster and its sizes. One MCP
description named Drama 3, and the decision provider was a type called `Qwen`. All of these are fixed here (see
[Gaps](#gaps-ranked)). After the fix, swapping a model of an existing kind is config only. A new compiled-in engine
adds one line to `engine.entries`.

## The contract

MCP tools, OpenAI endpoints and the voice loop talk to a kind, never to a model:

```
client ── MCP tool / OpenAI route / voice loop
              │  request (kind + one capability; model id optional)
              ▼
         routes.resolve ── config `defaults` / first of kind ── engine.check (declared limits) ── 400 before load
              │
         scheduler ── worker protocol ── adapter (compiled-in `engine.Entry`, or a `cmd` / `openai_server` worker)
```

A model is described entirely by its config entry. That entry gives its `kind`, its `engine` or `cmd`, its
`capabilities` (or the engine's own list), its memory (`resident_bytes`, `working_bytes`, `clone_working_bytes`, `reference_working_bytes`), its
licence, and its declared limits in `options`. `/v1/models`, MCP `list_models` / `list_voices`, `/v1/capabilities` and
the voice loop's tool offers are all built from these fields (`engine.audioInfo`, `engine.capabilityList`,
`routes.defaultFor`). No tool schema, description or model list names a model.

| Kind | Capabilities | Declared limits (tool `options` unless noted) | Who checks them |
| --- | --- | --- | --- |
| chat | `chat_completion` | `server`/`env`/`health` (its own server), `min_thinking_tokens`, `thinking_off` (top level) | `openai_server`, `api/budget.zig` |
| speech | `text_to_speech`, `voice_clone` | `voices`, `default_voice`, `presets`, `min_speed`/`max_speed`, `max_input_chars`, `instructions` (template), `stream`, `references`, `clone_rate`, `clone_codebooks`, `clone_working_bytes` | `engine.checkSpeech`, `engine.speechInstructions`, `api/voices.zig` |
| transcription | `speech_to_text` | `max_audio_s`, `languages` | `engine.checkTranscription`, `api/speech.zig` |
| image | `text_to_image`, `image_edit` | the engine's own (`max_side`, `max_pixels`, `edit_refs`, `precision` for Qwen-Image); `reference_working_bytes` (a `cmd` worker: what each edit reference adds to `working_bytes`, measured) | the engine's `Entry.check` |
| video | `text_to_video`, `image_to_video` | the engine's own (`max_width`, `max_height`, `max_seconds`, `default_size` for H3) | the engine's `Entry.check` |
| asset3d | `image_to_3d` | `formats`, `input_modes`, `max_steps`, `max_guidance`, `min_gaussians`/`max_gaussians`/`gaussian_multiple`/`default_gaussians`, `default_steps`, `default_guidance`, `working_bytes_by_gaussians` (measured; a count above every measurement is refused) | `engine.checkAsset3d`; the result's manifest and PLY by `media/asset3d.zig` |
| decision | `typed_decision` | `decision_path` | `api/decide.zig` (answers checked against the questions) |

Defaults come from config: `defaults.<capability>` for every route and MCP tool; `voice.chat_model`,
`voice.speech_model` and `voice.transcription_model` for the voice loop (empty: those defaults); and
`computer.decision_model` for computer decisions (empty: the chat default). `voice.voice` is optional; empty means the
speech model's `default_voice`.

### MCP: a stable kind-level tool list

`tools/list` is the same on every box: one tool per job kind (`generate_image`, `edit_image`, `generate_video`,
`generate_3d`, `synthesize_speech`, `transcribe_audio`, `chat`, `decide`, ...). A tool whose capability no configured model has stays
listed. Calling it is refused cleanly and the refusal names the models and what they can do (`requireCapability`:
"no model on this machine can do image_edit: its models are plain (text_to_image)"). This is the existing,
tested contract (`tests/mcp.zig`), and it keeps `tools/list` identical across a swap, which harnesses cache. A model's
own facts (voices, limits, `qualified`) come from `list_models`. The voice loop works the other way: a chat model can't
read a refusal before it promises a result, so the loop offers its chat model `generate_image`, `edit_image` and
`generate_video` (with `from_last_image` only when a model has `image_to_video`) only when a model can serve them
(`respond.offered`).

## Per kind

"Before" is what a swap needed at `87ea48b`; "after" is this change.

| Kind | Adapter interface | Hard-coded outside the adapter (before) | Swap: before → after |
| --- | --- | --- | --- |
| chat | `openai_server` engine (`src/engines/openai_server.zig`) or any `cmd` worker naming a `port`; requests proxied by `tool/upstream.zig` | the computer decision provider was the type `decide.Qwen`, logged as provider `"qwen"` (it already used the configured model); `api/budget.zig` defaults "thinking off" to `chat_template_kwargs.enable_thinking` (the Qwen-family switch), overridable per tool with `thinking_off` | config → config (provider renamed `ChatDecider`, logged as `"chat"`) |
| speech | `cmd` worker (`tools/speech/drama3`, `tools/speech/kitten`) or a compiled-in engine; worker protocol `{"speech": ...}` → `speech.wav` | `voice.speech_models` defaulted to `["fishaudio/drama-3", "kitten-tts-2"]` (`config.zig`): with Kitten configured, it outranked a newly configured `text_to_speech` default in the voice loop; `api/voices.zig` refused a clone unless `voice.json` had exactly 10 codebook rows; clone clips were always 44.1 kHz; the MCP `synthesize_speech` schema said "Drama 3 hears it as a bracketed direction" | config plus a code edit for clone-capable models → config |
| transcription | `cmd` worker (`tools/speech/whisper`) or a compiled-in engine; `{"transcription": ...}` → `transcript.json` | the `/voice` page and the companion offer Whisper large-v3's languages (no way to declare others); the `decode: greedy` hint (`request.Decode`) is Whisper-shaped, but a worker may ignore it | config (with a wrong language list) → config (`languages` declared, refused by the daemon, `/voice` narrows its picker) |
| image | compiled-in `engine.Entry` (`qwen_image`, `qwen_image_turbo`) or a `cmd` worker; `{"image": ...}` → PNGs | none in routing; prose hints only (MCP `size` says "about 1 megapixel is best"; the voice tool's size example) | one adapter + one `engine.entries` line (compiled-in), or config (`cmd`) |
| video | compiled-in `minimax_h3` or a `cmd` worker; `{"video": ...}` → MP4 | none in routing; the MCP schema defaults (`fps` 24, `seconds` 1–20) are API bounds, and the model's own limits are its options | as image |
| asset3d | a `cmd` worker (`tools/asset3d/idol`) or a compiled-in engine; `{"asset3d": ...}` → `asset.ply`, `asset.json` (the manifest), `input.png` | none: a new kind, so nothing came before it | config (`cmd`) |
| decision | `openai_server` (KnowLine) with `decision_path`; `api/decide.zig` checks the answers | none | config |
| speaker / voice-clone embedding | not a daemon kind: `voice_clone` is a capability of a speech tool, whose adapter encodes the voice; CAM++ (Chatterbox Turbo's encoder) is used only by offline qualification scripts on the box, never by `src/` | `vq_codes` (a codec-code grid) is the one wire shape for `references` and cloned voices | config for any codec-code model; a model that conditions on something else (a speaker embedding) needs a new reference field, which is a wire addition |

Model names that remain in `src/` belong to adapters (`src/engines/qwen_image*`, `src/engines/minimax_h3*`, their
`check` CLIs and licences), config and engine unit tests, and comments that record measurements. The `/voice` page's
`short()` drops a "Fish Audio" vendor prefix from display names; this is cosmetic and has no effect on other names.

## Gaps, ranked

| # | Risk | Gap | Fix |
| --- | --- | --- | --- |
| 1 | high | Voice cloning tied to Fish's codec: exactly 10 `vq_codes` rows, 44.1 kHz clips. Any other `voice_clone` model failed every clone with "the model wrote no valid voice.json" | `clone_codebooks` (absent: any count, the same for every reference) and `clone_rate` (absent: 44.1 kHz; clips are resampled only when they differ). The Jarvis template sets `clone_codebooks: 10`, so Drama 3 keeps the strict check |
| 2 | high | The voice loop's speech fallback chain named two models in code. Swapping the `text_to_speech` default left Kitten speaking calls on any box that still had Kitten configured | `voice.speech_models` defaults to empty: the `text_to_speech` default speaks alone. A box that wants a chain lists it (Jarvis and the live box already do) |
| 3 | medium | The model-management prompt (`voice/model_skill.md`) gave the chat model a fixed roster: "Drama 3, the voice (21 GiB); Whisper, the ears (4 GiB) … 120 GiB". After a swap it planned memory with wrong names and sizes | a `{models}` line rendered from config at each request (`skill.render`): each model's role, id and needed memory (as `list_models` prices it), within `budget_bytes`, else the machine's memory less `reserve_bytes`. Served the same way at `/v1/skills/model-manager` |
| 4 | medium | No way to declare a transcription model's languages: a model that hears only some got any code, and the UIs offered Whisper's list | `languages` option: listed in `/v1/models` and `list_models`, a `language` outside it is refused before load, and `/voice` and the companion read it from `/v1/models` (Whisper's list stays as their offline fallback; the mobile app sends no language, so the box's default applies) |
| 5 | low | The MCP `synthesize_speech.instructions` description named Drama 3 | model-neutral wording (the only `tools/list` text change) |
| 6 | low | `computer/decide.zig`'s provider was `Qwen` (named after a model, though it used the configured one) | `ChatDecider`, audit-log provider `"chat"` |
| 7 | low | MCP's no-capable-model refusal pointed at the image edit pack for every capability (`image_to_video`, `typed_decision`, ...) | the hint only for `image_edit` |

The conformance suite also found a latent bug. MCP `decide` with no `model` could fail with "state is required" after
the decision had already run. `api/decide.zig` wrote the model into a shallow copy of the caller's map, and the put
could move the entries out from under the caller. The model now goes into a clone.

Left as they are, with the reason:

- **Prose hints** in MCP schemas ("about 1 megapixel is best", `fps` 24) are API-level guidance. Each model's real
  limits are refused by its engine's `check` and listed by its options.
- **`budget.default_off`** (`chat_template_kwargs.enable_thinking`) is a default for the common open-model servers;
  a model with another switch sets `thinking_off`.

## Adding or swapping a model

1. **Same kind, existing adapter type** (another `cmd` speech worker, another `openai_server` chat model): add its
   config entry with `kind`, `cmd` or `engine`, memory figures, `capabilities`, and its limits in `options`. Point
   `defaults.<capability>` at it (and `voice.speech_models`, if the box uses a fallback chain). No code changes.
2. **A new compiled-in engine**: implement `engine.Entry` in `src/engines/<name>_tool.zig`, add it to `engine.entries`
   and to `main.zig`'s test imports ([ARCHITECTURE.md](../ARCHITECTURE.md#adding-a-model)). Nothing outside the
   adapter changes.
3. **A new kind** is the only case that touches the API: see
   [ARCHITECTURE.md](../ARCHITECTURE.md#a-new-kind-of-model-embeddings-).

Declare only what the model can do. The daemon refuses what it can't do before anything loads.

## Conformance tests

`tests/dropin.zig` runs the same assertions against two model sets for every kind. Each set uses different ids and,
where it can, a different adapter (the compiled-in test engine, a `cmd` worker, an `openai_server` stand-in). The suite
checks that:

- MCP `tools/list` and `/v1/capabilities`' tool list are identical across sets;
- `/v1/models` lists exactly the configured models;
- every OpenAI route and MCP job tool works with no model named and with the model named;
- the voice loop takes either set's speech model;
- a set whose image model can't edit and whose video model can't do image to video gets clean refusals;
- changing only `defaults` moves an unnamed request to the other model;
- a declared `languages` list is shown and enforced;
- `POST /v1/assets3d` and MCP `generate_3d` make a labelled PLY, its manifest and its input view on either set
  (testpattern, and a `cmd` worker with `working_bytes_by_gaussians`), served through `/v1/files` and `/content`;
  a `num_gaussians` outside the declared range, an undeclared `input_mode` and a missing image are refused before the
  tool loads.
