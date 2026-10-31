# Fish Audio Drama 3

[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) is Fish Audio's expressive text to speech model: a
40-layer hybrid (Gated DeltaNet and attention) text backbone, a 13-layer temporal codebook tower that writes ten codec
codebooks a frame, and Fish Speech's modded-DAC codec, at 44.1 kHz and about 21.5 frames a second. LocalRouter serves
it as `fishaudio/drama-3` through plotarmordev's [drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold),
which ports it to [TensorFold](https://github.com/ashhart/TensorFold)'s native Zig/CUDA engine for the GB10. The
worker, `tools/speech/drama3/drama3_worker.py`, runs that repository's own launcher and server, built from pinned,
hash-checked sources; it does not reimplement the model.

On Apple silicon it runs on drama-tensorfold's Metal path instead ([macOS](#macos-apple-silicon)).

**Built with Fish Audio.** See [Licences](#licences): the weights' licence limits commercial use and forbids a Hosted
Service without a separate licence, so this tool stays on loopback.

> [!NOTE]
> **The Jarvis box serves the NVFP4 pruned weights since 2026-10-24** ([The NVFP4 pruned weights](#the-nvfp4-pruned-weights)):
> plotarmordev/drama-3-NVFP4-Pruned, a lossy community quant approved by JJ, on its own engine build with LocalRouter's
> voice prefix cache ported to it. 20.56 GiB booked against 39.25. The full model below stays installed beside it and
> qualified (`variant: "full"`, the default of the tool option).
>
> **Qualified on DGX-01 (GB10), 2026-10-21** ([measured results](#measured-on-dgx-01)): first audio p50 0.142 s
> streamed, warm RTF p50 0.663, peak 40.4 GiB, beside the warm chat model as well. The marker is bound to that host,
> these pins and that engine binary: on any other machine the model is listed with `"qualified": false` and every
> request gets `503 server_error: model 'fishaudio/drama-3' is not GPU-qualified on this host` until
> `tools/speech/drama3/qualify.sh` passes there. The native engine is not bit-exact with the SGLang-Omni reference and
> upstream's own accuracy bars fail; that was accepted explicitly ([Numerics](#numerics)).

## What it serves

| | |
| --- | --- |
| Request | `speech`: `input` (1 to 4096 characters), `voice`, `references`, `stream` (with `response_format: "pcm"`) |
| Output | `speech.wav`: RIFF, mono, 16-bit PCM, 44100 Hz. The daemon encodes the other response formats; a stream is raw PCM chunks |
| Voices | the voice files in `voices_dir` (six shipped, [Voices](#voices)), the operator's `default_voice` among them; `default` (the model's built-in voice, a different speaker for each text) only when listed |
| References | `[{"text", "vq_codes"}]`: a clip's exact transcript and its codec codes (10 codebooks), spoken into the prompt before the voice's own references, in the reference implementation's order |
| Length | 1024 frames (47.55 s) a synthesis; `max_frames` raises it (4096: about 190 s in one synthesis); `long_form` speaks long inputs in joined segments |
| Finish | `x-finish-reason`: `stop`, `length` (clipped at the frame limit) or `silence` (10 s below -40 dBFS, which also ends a collapse into a steady hum); long form adds `x-long-form-segments: N;S2,S3,...` |
| Markup | point tags in the text, such as `[sighs]` and `[laughs]`, where the event should land; the model card lists no full vocabulary |
| Instructions | no native control: the system prompt is the fixed `convert the provided text to speech` (`drama_prompt.zig`), and SGLang-Omni's server refuses `instructions`. The tool's `"instructions": "[{s}] "` option has the daemon write them as a bracketed direction before the line (after a leading speaker tag): `[whispering, nervous] I'm fine.` |
| Cloning | `POST /v1/voices` (the tool lists `voice_clone`): recordings of one person become a voice of their own, spoken with the references the engine's `voice encode` wrote ([Cloned voices](#cloned-voices)) |
| Refused (400) | `speed` other than 1, `preset`, `instructions` holding `[`, `]`, a line break or a tag (the daemon, before loading); an unknown voice, references the model refuses, a prompt longer than the model takes (the server, in its own words) |
| Not offered | audio in `references` (send codes, or clone a voice); SSE streams; `seed` (the engine seeds each synthesis from its prompt, so the reply's seed is 0) |

A streamed reply ends with the plain last chunk when the audio ended by itself, with the `x-finish-reason` trailer
when it was cut short (and both trailers for long form), and without a last chunk when the synthesis failed after the
first one. The worker parses the server's chunks itself, because Python's `http.client` drops trailers, and writes them
into `speech.wav` as they arrive; the daemon sends each new run on as it lands ([streaming](../ARCHITECTURE.md#speech-tools)).

## How it runs

```
daemon ── worker protocol ──> drama3_worker.py ── HTTP on 127.0.0.1:port ──> tensorfold-native serve
                                   │                                              ▲
                                   └── launcher/pilot.py --external-admission (own session, parent-death SIGTERM) ── systemd-run --user unit
```

- **Admission is LocalRouter's.** The scheduler admits the model from its `resident_bytes` against the budget, as for
  every model, beside the chat model and Whisper. Upstream's launcher has its own admission (it refuses unless
  `MemAvailable` is at least 88 GiB, and kills the unit if `MemAvailable` falls below 34 GiB), which cannot pass on a
  box that keeps chat and Whisper warm (DGX-01 had 83 GiB free with chat, Whisper and Kitten loaded) and would fight
  the scheduler. The launcher has no option to turn it off, so LocalRouter runs it with a pinned patch,
  `tools/speech/drama3/patches/pilot-external-admission.patch` (sha256 `8b6304cd…`): `--external-admission` skips
  both checks, and gives the engine its least memory reserve (`TENSORFOLD_MEMORY_RESERVE_GIB=2`, not 34). The engine
  admits its own allocations against `MemAvailable` less that reserve, so with 34 it refused a load the scheduler had
  admitted whenever less than about 73 GiB was free: on DGX-01, with 70.5 GiB free (another process holding 22 GiB),
  every reload failed with `OutOfDeviceMemory`. Its footprint is set by its flags (`--parallel 1`,
  `--prompt-cache-gib 0`, `--max-frames`), not by the reserve. Everything else stays: the 54 GiB cgroup cap, the
  server's 54 GiB allocator cap and the lock.
  `fetch.sh` writes the patched launcher to `ROOT/launcher/pilot.py` (sha256 `e0ce7087…`, pinned) and leaves the
  upstream checkout clean. `options.admission: "launcher"` restores upstream's checks for a launch with no
  scheduler.
- **Load.** The worker checks the qualification marker, the pinned launcher and the checkpoint files (`verify`), then
  starts the launcher with `--external-admission` and the lock file the daemon names in `LOCALROUTER_GPU_LOCK`: this
  tool's own, uncontended `<data>/locks/fishaudio_drama-3.lock`, while the scheduler holds the host's `gpu_lock`
  shared for every LocalRouter model ([the host's GPU lock](../ARCHITECTURE.md#the-hosts-gpu-lock)). The launcher
  starts the server as a transient `systemd --user` unit capped at 54 GiB with swap off. The worker is ready when the
  launcher prints `DRAMA_PILOT_READY`; then every voice the config lists must be one the server loaded.
- **Memory.** `resident_bytes` is 45,097,156,608 bytes (42 GiB): the largest peak measured on DGX-01, 40.45 GiB
  (the drop in `MemAvailable` from before the load, `run3`), plus 1.55 GiB, 1.5 times the spread of the nine block-FP8
  peaks measured there (39.43 to 40.45 GiB: `run1` 39.84, `run2` 40.36 beside the warm chat model, `run3` 40.45, six
  direct server sessions 39.43 to 39.91). GB10's CUDA memory does not show in the server's cgroup (its own peak was
  1.1 GB), so the drop is the measure, and it already holds everything the process costs beyond its CUDA
  allocations: of a direct session's 39.84 GiB drop, 38.00 GiB are the engine's own allocations (weights 36.27, codec frames 1.18, KV
  caches and state 0.55), 0.18 GiB the CUDA context, 0.56 GiB host heap (the driver's module images and staging,
  the tokenizer), about 0.27 GiB kernel slab and shared memory, and about 0.8 GiB the driver keeps outside the
  `/proc/meminfo` categories. `qualify.sh` refuses a run whose peak exceeds `resident_bytes`. The worker reports
  no size, so the scheduler holds `resident_bytes` for it while it is loaded. With chat (24 GiB) and Whisper (about
  4 GiB) that is 70 of DGX-01's 106 usable GiB (114 free less the 8 reserved): the three stay warm together, an image
  job beside them evicts only Whisper, a video job evicts the image, Drama 3, then chat (lower priorities first), and
  they come back by themselves (`tests/drama3.zig`).

  What holds memory for Drama 3, and what is only booked (measured through the daemon on DGX-01, 2026-10-21):

  | Layer | Where | Real or booked | Before (`resident_bytes` 48.45 GiB) | Now (42 GiB) |
  |---|---|---|---|---|
  | The engine and its process | the `MemAvailable` drop | real | 39.97 GiB loaded and idle; no more while speaking | 39.45 GiB; the same |
  | `resident_bytes` | the tool's config; the worker answers `load` with `"resident": 0` (`drama3_worker.py:286`), so the scheduler keeps the config's figure (`src/sched/scheduler.zig:296-297`) | booked for as long as it is loaded: it counts against `budget_bytes` and against `MemAvailable` at the next admission | 48.45 GiB, 8.49 above the real drop | 42 GiB, 2.54 above |
  | `reserve_bytes` | daemon config, default 8 GiB (`src/config.zig:72`); `fits` (`src/sched/scheduler.zig:351-357`) | booked once for the whole machine, not per model: what LocalRouter leaves the OS | a load needs `MemAvailable` ≥ 56.45 GiB | ≥ 50 GiB |
  | The launcher's caps: unit `MemoryMax=54G` and `TENSORFOLD_CUDA_MEMORY_LIMIT_GB=54` (`launcher/pilot.py:81-82, 134`) | upstream's launcher | ceilings only: nothing is allocated up to them (the engine allocates its 38 GiB and stops) | - | - |
  | The engine's reserve, `TENSORFOLD_MEMORY_RESERVE_GIB=2` (`launcher/pilot.py:80`, the engine's `cuda_memory.zig` `room`) | the pinned launcher patch | a check at allocation time (free less 2 GiB), not held | - | - |

  An earlier node config of DGX-01's qualification booked 57,982,058,496 bytes (54 GiB, the launcher's cap, before the
  peak was measured): 14 GiB above the drop, 22 GiB with the reserve. It now books 42 GiB.
- **Requests.** One at a time, as for every LocalRouter tool (the server runs `--parallel 1`). The server's 400 is a
  400, its 503 (busy) a 503 and anything else a 500. A server that dies ends the worker without a reply: that job is
  a 503 with the log tail, and the next request starts it again.
- **Unload** (idle TTL, `POST /v1/tools/fishaudio/drama-3/unload`, eviction): the worker sends the launcher SIGTERM; it
  stops the unit, waits until systemd reports it inactive, then releases the lock. The worker keeps the daemon waiting
  with progress lines until then, so nothing else loads into memory the server still holds.
- **Soft cancel.** The worker's load reply declares `"cancel": "signal"`: SIGUSR1 during a generate closes its
  connection to the server, which cancels the synthesis, and replies `{"ok": false, "type": "cancelled"}` with the
  model still loaded, so the next request needs no reload. The server takes one synthesis at a time and refuses another
  with a 503 "busy" until the cancelled one is dropped (its `/health` does not count speech, so the worker cannot wait
  for that up front), so the worker retries a busy reply every 50 ms for up to 10 s: the daemon sends one request at a
  time, so busy can only mean a synthesis being dropped. Without the signal a generate is unchanged. A daemon that sends the signal (barge-in) and its hard-abort
  fallback are #15's.
- **Cancel and deadlines** kill the worker's process group. The launcher runs in its own session with a parent-death
  signal, so it still gets SIGTERM and stops the unit. A launcher that is itself killed outright leaves its unit's name
  in `runtime/localrouter-drama3.unit`, and the next load stops that unit first.

## Setup

On the GB10 (CUDA 13 with nvcc, Zig 0.17.0, user systemd, about 39 GiB of disk for the checkpoint and more for the
engine checkout, build and caches):

```sh
tools/speech/drama3/fetch.sh /data/models/drama3      # drama-tensorfold at the pin; the checkpoint, every file sha256-checked
ZIG=/opt/zig-0.17.0/zig NVCC=/usr/local/cuda/bin/nvcc tools/speech/drama3/build.sh /data/models/drama3
tools/speech/drama3/qualify.sh /data/models/drama3 /etc/localrouter/config.json /data/models/drama3/qualify-01 \
  --reference /data/models/drama3/reference --accept-native-numerics
```

The `--reference` directory is upstream's reference runtime, as its `recipe/README.md` describes: `venv/` with the
reference's dependencies and an `sglang-omni/` checkout at the revision in `drama-tensorfold/recipe/sources.json`
(`ab095ef9e779e56af33b6ed2c863645c9903372d`). `tools/speech/drama3/reference.sh RUNTIME [UV]` sets it up on an aarch64
GB10 (about 9.4 GB): every sglang-omni dependency and its `fishaudio` extra except `nemo_text_processing` and
`dots.tts`, which need `pynini` (no aarch64 wheel; it does not build without OpenFst) and are imported only by other
models. Upstream pins the checkout but not the venv's packages, so LocalRouter cannot pin them either; the qualify
report records the torch and SGLang versions (on DGX-01: torch 2.13.0+cu130, sglang 0.5.21).

`fetch.sh` prints the weights' licence before downloading them and accepts nothing on anyone's behalf. `build.sh`
runs upstream's `scripts/build-engine.sh`, checks that the patched TensorFold HEAD is the pinned one, applies
LocalRouter's engine patches and builds the server again, checks that HEAD too, and records the
engine binary's sha256 in `ROOT/engine.json`. Then add the tool from `tools/speech/drama3/config.example.json` to the
daemon's config, with `gpu_lock` set to the lock every GPU job on the host already takes (only on a host without one,
create it with `touch` and have the other GPU jobs take it too).

| Option | Default | |
| --- | --- | --- |
| `port` | 18793 | the server's loopback port |
| `max_frames` | the engine's 1024 | frames one synthesis may make (`--max-frames`) |
| `long_form`, `long_form_chars`, `long_form_history` | off, 500, 1 | `--long-form` and its settings |
| `voices_dir` | none | an absolute directory of voice files (`--voices`) |
| `voices`, `default_voice` | `["default"]`, `default` | what `/v1/models` lists and the daemon checks. The worker names a `default_voice` other than `default` in every request that names none (or `default`), as the server's `--default-voice` did, so a cloned voice can be spoken with its references alone; it must be one the server loaded. The example config lists the six shipped voices, `Mae` by default ([Voices](#voices)) |
| `clone_working_bytes` | 8 GiB | what the scheduler books for an encode beside the loaded model (`capabilities` must list `voice_clone`): the encoder's measured peak ([Cloned voices](#cloned-voices)) |
| `verify` | `size` | at load, each pinned checkpoint file's size, or its `sha256` (minutes), or `none` |
| `pilot`, `native_bin`, `checkpoint`, `runtime_dir` | under `weights` | where fetch.sh and build.sh put them |
| `ready_timeout_s` | 900 | the launcher's readiness deadline |

### Voices

The built-in voice is not one speaker: the engine conditions on nothing but the prompt, so each text lands on its own
model-invented speaker (CAM++ speaker-embedding cosine between its replies to ten voice-loop sentences: 0.52 on average,
0.27 at worst, no higher than between different voices). A voice loop that speaks a reply one sentence a request
therefore changed voice every sentence. A voice file pins the speaker: its reference (a clip's transcript and codec
codes) is spoken into every prompt before the text.

LocalRouter ships six, `Mae` (the default), `Arlo`, `Ezra`, `Nora`, `Theo` and `Wren`, all invented by Drama 3 itself
(`tools/speech/drama3/voices/SOURCES.md` says how each was made and picked). `fetch.sh` installs them by sha256 into
`ROOT/voices`; set `voices_dir` to it, list them in `voices`, and name the default in `default_voice` (as
`config.example.json` does). Measured on DGX-01, the same ten sentences per voice, streamed:

| | Within a voice: cosine mean (min) | To the other voices | First audio p50 | RTF |
| --- | --- | --- | --- | --- |
| Built-in voice | 0.52 (0.27) | 0.51 | 0.144 s | 0.70 |
| The six voices, before the prefix cache | 0.75 to 0.86 (0.47 to 0.71) | 0.44 to 0.53 | 0.67 to 1.16 s | 0.86 to 1.28 |
| The six voices, with it | 0.71 to 0.86 (0.47 to 0.71) | 0.43 to 0.53 | 0.127 to 0.134 s | 0.71 (warm) |

The same text in the same voice gives the same audio, byte for byte. The built-in voice's RTF on the same short
sentences is 0.70 (warm). A voice's first sentence after a load, or after the cache dropped it, fills the cache: it is
prefilled whole (0.71 to 1.01 s to first audio). With the cache, the server's peak was 39.33 GiB.

Verified on DGX-01 with the build pinned here:
- the 26 qualify texts in the built-in voice are byte-identical to `run3`'s;
- the 60 voice replies are byte-identical with the cache on (54 resumed, 6 filled) and off (`DRAMA_PREFIX_CACHE=0`).

Two replies per voice, and the built-in voice's for comparison, are in `~/jarvis/voices/` on DGX-01.

**The prefix cache.** A reference adds its rows (280 to 431 frames, plus its transcript) to every prompt, and the
prompt pass reads every expert for each 128-row chunk, so a sentence in a voice cost 0.5 to 1 s more than one in the
built-in voice. LocalRouter's engine patch (`patches/engine/`, applied by `build.sh` after upstream's series) prefills a
referenced prompt's system message, the instruction and the references, in chunks of its own. It then keeps the
backbone's and the tower's state after it, so a later prompt with the same references resumes there and prefills only
its own text. The entries are keyed by those rows in full, least recently used dropped. There are
`DRAMA_PREFIX_CACHE` of them (default 2, 0 off), each about 152 MiB plus 46 KiB a row: 174 MiB for the longest voice.
A prompt without references runs exactly as before. With references, the cached and uncached passes are the same
chunks, so a resumed reply equals one prefilled whole, byte for byte.

Clone only a voice the model invented, your own, or one whose owner has consented on record: the model licence's
Section IV(b) forbids imitating an identifiable person without consent. The six shipped files hold only voices the model
invented, made offline with upstream's tools (`drama_voice_clip.py`, `drama_voice.py`); a person's voice is cloned on
the box itself ([Cloned voices](#cloned-voices)), and its clips and voice stay out of source control.

### Cloned voices

`POST /v1/voices` ([AGENTS.md](../AGENTS.md#voices-cloning)) makes a voice from recordings of one person, as
plotarmordev's drama-tensorfold does: a voice is reference passages (each a clip's exact transcript and its codec
codes) spoken into the prompt before the text, not a speaker embedding. Nothing new is downloaded: the codes come
from the pinned engine's own `tensorfold-native voice encode` (patches 0051 and 0052; the NVFP4 build carries it too),
which reads `codec.pth`, the same file in both checkpoints, and encodes in BF16 on the GPU.

1. **Heard whole** by the transcription model: its Silero voice activity detector says where each clip's speech is,
   and its transcript is the reference's when the clip makes one reference.
2. **Prepared by the daemon** (`src/media/voiceprep.zig`, upstream's `drama_voice_clip.py` trim, cut and level numbers
   on 10 ms frames at 44.1 kHz): each clip decoded with ffmpeg; trimmed to 0.25 s before its first and after its last
   speech frame; cut inside pauses into references of at most 30 s; each levelled to -20 dBFS speech, peaks under
   -1 dBFS. All clips together: at least 10 s; the first 60 s and 8 references are kept. Only clear failures are
   refused: no speech (a clip with none is left out), under 10 s in all, or a background about as loud as the voice.
   The background is the mean power of the frames at least 0.1 s from any speech the detector heard (with under 0.3 s
   of them it is not measured, and passes), so a loudness-normalised, compressed or limited upload, whose quietest
   frames are soft syllables rather than noise, is not read as noisy (upstream's rule, the quietest tenth of all
   frames, read a heavily compressed take at 16 dB and music alone at 2.3 dB). Under 30 dB is a warning in the voice's `clips[].warnings`; under 5 dB is refused with
   code `noisy_background` unless `force=true` ("Use anyway"). A tonal background (spectral flatness under 0.06) is
   named as music or a hum. Clipping, a clip starting or ending inside a word, a 30 s stretch with no pause and an
   over-long clip with a transcript (set aside, transcribed) are warnings. Not checked: a second speaker.

   | Fixture (LibriSpeech speaker 1272) | Speech over background |
   |---|---|
   | as published; loudnorm -14 LUFS + limiter, MP3; dynaudnorm + limiter | 32.7; 32.2; 35.7 dB |
   | 1272-135031-0001 as published (a noisy room) | 28.6 dB: warned |
   | heavy compression (20:1, +20 dB makeup), its pauses removed | 6.7 dB: warned |
   | a jazz band 15, 25 dB under the speech, then mastered | 17.9, 26.5 dB: warned as music |
   | a chord bed 10 dB under the speech | 3.2 dB: refused as music |
3. **Transcribed** by the transcription model where the caller gave no transcript and the clip makes several
   references (a clip with one is never cut).
4. **Encoded** by the worker: one `voice encode` per reference, a job of its own on the realtime lane, booked at
   `clone_working_bytes` beside the loaded model. The job directory, clips included, is deleted once the codes are read,
   and so are the transcription jobs.
5. **Kept** in `data_dir/voices/NAME.json` (0600): the transcripts, codes, `source` and the consent statement (who, when);
   a draft (no name yet) lives in memory for an hour. `DELETE` removes the file.
5. **Spoken** with its references inline (`references` in the server request, after any the caller sent) and no
   `voice`, so the server needs no restart and no `--voices` file, and the prefix cache keys on the same rows. The
   worker names the default voice in other requests itself, as `--default-voice` did, so the built-in voice plus a
   clone's references is the clone and nothing else.

**Measured on a GB10 (DGX-02, 2026-10-25),** the NVFP4 build and checkpoint copied from DGX-01, chat not loaded:
LibriSpeech dev-clean speaker 1272 (a LibriVox recording, CC-BY-4.0) cloned through the API from four clips, 43.8 s kept,
no transcripts (Whisper's were word-perfect), then the 24 sentences of `compare-texts.txt` in each voice
(`compare.py`, streamed, Whisper large-v3-turbo):

| | First audio p50 | First audio, first sentence | RTF p50 | WER | CAM++ cosine to the real speaker, mean (min) |
| --- | --- | --- | --- | --- | --- |
| Mae (named) | 0.324 s | 0.81 s | 0.748 | 0.0 % | 0.22 (0.10) |
| Arlo (named) | 0.323 s | 0.79 s | 0.746 | 0.0 % | 0.27 (0.14) |
| LS1272 (cloned) | 0.344 s | 2.0 s | 0.758 | 0.9 % | **0.68** (0.48) |
| Speaker 1272, real | | | | | 0.85 (0.73) |

- **Similarity** is CAM++ (chatterbox-turbo's encoder, as used to pick the named voices) between each reply and five
  held-out 13 to 17 s passages of the same reader that the clone never heard; the real passages against each other
  score 0.85, the ceiling. Median F0: clone 117 Hz, reader 111 Hz. The clone's own replies agree at 0.79 (the named
  voices 0.86 to 0.88).
- **Latency.** A clone's references are longer than a named voice's (945 frames against 280 to 431), so its first
  sentence after a load, or after the cache dropped it, prefills them whole (2.0 s to first audio); every sentence
  after resumes from the cache, 20 ms slower than a named voice.
- **Making one** took 21.6 s through the API (four transcriptions and four encodes; about 4.7 s an encode of 30 s).
- **Refusals seen:** a fifth LibriSpeech clip (1272-135031-0001) measures 28.9 dB from its speech to its quietest
  frames (30.0 dB to its pause frames' median), and is refused for its background; 60.2 s of clips is refused as more
  than 60 s.

**Memory.** The encoder, for a 30 s reference (the longest): 1,064 MiB GPU + 519 MiB host, `MemAvailable` down at most
1,717 MiB over three runs, so `clone_working_bytes` is 2 GiB (the peak and 256 MiB, rounded up). A clone's longer
references also make its prefix-cache entry larger (152 MiB and 46 KiB a row): with two clones of 56 s cached (1,212
and 1,216 frames) the server held 20,375 MiB GPU + 0.62 GiB host, 20.52 GiB, against 19.69 + 0.62 qualified with the
named voices. Two clones of 60 s add 7 MiB more, so the NVFP4 tool's `resident_bytes` is now 22,309,502,976
(20.78 GiB: that peak and 256 MiB) instead of 20.56 GiB. The full weights' figure gets the same 224 MiB (the same
cache code), not measured on them.

**Several nodes.** When Drama 3 runs on another node (`docs/dev/TWO-SPARK.md`), the front prepares the clips, keeps the
voices and sends each encode and speech job to that node with its files, as for any job; the node's copy of an encode
job's directory (the prepared clips) stays there until its `keep_outputs_s` expiry, not deleted at once as the front's
is. Not run on two nodes.

**Not verified:** a real person's own recordings (no recording of JJ yet); a clone in a voice call on DGX-01 beyond the
smoke; voices of other languages or accents; the encoder's codes against the reference encoder's (upstream measured
2.5 to 3 % of semantic codes different, with equal similarity); the full weights.

## Pins

`tools/speech/drama3/pins.json` holds every hash. `fetch.sh` and `build.sh` refuse anything else.

| What | Pin |
| --- | --- |
| Upstream | plotarmordev/drama-tensorfold @ `1a53b8aea738a99d3630f949b6b2a1f9389f0795` (main, 2026-10-17, PRs #4 to #9); sha256 of `recipe/pilot.py`, `scripts/build-engine.sh`, `engine/UPSTREAM`, `recipe/checksums/checkpoint.sha256`, `recipe/sources.json`, `NOTICE`, `LICENSE`, `engine/patches/SHA256SUMS` |
| Engine | TensorFold @ `c7137bb31a7dad924fb541a7068422127b9a94b2` plus the 59 patches in `engine/patches/` (each by sha256; series sha256 `3fff8dd6…a910d247`), applied by `build-engine.sh`: HEAD `459771e7531431490b80d7a8149c9160e03c01ce`; then LocalRouter's `patches/engine/` (the voice prefix cache, by sha256), applied by `build.sh`: HEAD `efc6c2f82a4a12726e36dce66b0969c1d3aee3f7` |
| Voices | `voices/*.json`, the six named voices, each by sha256; `fetch.sh` installs exactly these into `ROOT/voices` |
| Checkpoint | fishaudio/drama-3 @ `e0a9218aa48888270f06ef1ed9ea67622c0e17d2`: `codec.pth`, eight weight shards (41,474,674,072 bytes together, matching upstream's `sources.json` and `checkpoint.sha256`), tokenizer, config, chat template and licence, each by size and sha256 |
| Toolchain | Zig 0.17.0, CUDA 13 |
| NVFP4 variant | `pins-nvfp4.json`: plotarmordev/drama-3-NVFP4-Pruned @ `494db08481db952f834b2241bce63fb3dfa72554` (17 files, 22,294,649,076 bytes of weights and codec, each by size and sha256: the same digests as drama-tensorfold's `recipe/checksums/nvfp4.sha256` at `844d0ea`, plus LICENSE and NOTICE); plotarmordev/TensorFold @ `b1f4b9b4830c9bd43bf7c432cffe7dea21313537` (`drama-nvfp4`, as drama-tensorfold's `engine/nvfp4/UPSTREAM`) plus `patches/engine-nvfp4/` (the prefix cache, ported): HEAD `f4a0eff0892bf02a2772f3b82ef53bbfe49d7f9d`; serve flags `--cfg-gamma 1.5 --cfg-batched` |

## Qualification

`qualify.sh ROOT CONFIG OUT --reference RUNTIME [--accept-native-numerics] [--coresident]` runs on the GB10 (DGX-01),
every step under a time limit, and writes `OUT/status.json` and a manifest as Kitten's does. Like the daemon, e2e and
bench hold the host's `gpu_lock` shared while the model is loaded and give the launcher a private lock; the worker
runs with LocalRouter's admission (`--external-admission`).

- **assets:** the checkout, every pinned file and patch, the patched launcher, the engine binary against
  `engine.json`, and the full sha256 of every checkpoint file;
- **e2e:** the real worker on the real GPU: a cold load (the private lock is held while loaded); a sentence whole (44.1
  kHz mono, `stop`), again warm (byte-identical), and streamed (the same bytes); a 300-character paragraph whole and
  streamed (equal, warm RTF recorded); each configured voice, and inline references from a voice file; an unknown
  voice refused with a 400; long form when it is on; then unload: the unit inactive, the lock free and `MemAvailable`
  back within 2 GiB;
- **bench:** upstream's own client, `bench/drama_bench.py`, against the server the worker started, with upstream's
  24-item corpus (sha256-checked) and schedule, as its `before-after.sh` runs it: whole WAV replies on the fresh
  process (cold, 2 warmup, 3 warm passes), then streamed PCM on the same process. Every 0.5 s it samples the unit's
  cgroup memory, `MemAvailable` and `nvidia-smi`. It passes when every request was valid audio, warm streamed first
  playable audio (`first_frame_s`) p50 is at most 0.5 s, warm RTF p50 is under 1.0 for both runs (real time: why
  Drama 3 is the Jarvis box's voice), and the peak (the larger of the cgroup's peak and the drop in `MemAvailable`
  from before the load, since GB10's CUDA memory does not show in the cgroup) fits `resident_bytes`. The comparison
  with upstream's claims (first audio 0.141 s, RTF 0.634, 0.682 after its patch 0040) is recorded as ratios, not
  gated. With `--coresident` (chat and Whisper already warm in a running daemon), the report records what was
  loaded and `MemAvailable`; such a run measures the Jarvis box but cannot qualify, because upstream's accuracy guard
  needs the host lock to itself;
- **accuracy:** upstream's own checks from `qualification/`, as its README runs them, each in its bounded GPU guard
  (`run-bounded-oracle.py`: the launcher's admission, 54 GiB cap, abort floor and the shared lock): the backbone
  reference check (the 40-layer prefill of its 182-token fixture, native chunks 1, 7 and 128 against the reference's
  extend: hidden states and semantic logits), the same native outputs rescored under upstream's registered worst-row
  bar (CPU), and the tower reference check (13 layers, teacher-forced, at 1, 7 and 128 rows). Their verdicts and
  metrics go into `OUT/accuracy/report.json` and the marker, pass or fail. Upstream reports that the backbone check
  fails some bars, so a failing verdict is the expected result today. Upstream's codec gate and its tower full-context
  replays are not reproducible from its repository (its `qualification/README.md` says so) and are not run.

Only when assets, e2e and bench pass, the accuracy checks produced valid evidence, and `--accept-native-numerics` is
given, does it write the tool's `qualification` file: `{"qualified": true, "host", "gpu", "pins_sha256",
"native_sha256", "upstream_commit", "checkpoint_revision", "numerics", "upstream_accuracy", "bench", "report",
"qualified_at"}`, where `bench` holds first audio, RTF, the peak memory, `resident_bytes` and the ratios to upstream's
claims, where
`upstream_accuracy` holds the reference revision, each check's verdict and metrics, and upstream's own
`native_drama_qualified`. The accuracy verdicts do not decide acceptance; the flag does. The daemon admits the model once that file says `"qualified": true`; the
worker also refuses to load unless its `host` is this machine, its `pins_sha256` the current `pins.json` and its
`native_sha256` the engine binary, so new pins or a rebuild need a new run.

### Numerics

This is a deliberate exception to [CONTRIBUTING](../../CONTRIBUTING.md)'s bit-exact rule, made explicit so it is
not made quietly: the native engine is not bit-identical with the SGLang-Omni reference (`ab095ef9`), and upstream
reports that it misses some of its own accuracy bars (backbone all-row cosine 0.9986 against 0.999, last-row 0.99809
against 0.999, hidden-state relative L2 0.0528 against 0.05; one-frame codec decode 33.1 dB against 40 dB; a tower
top-1 that differs where the reference's logits tie). Its evidence is in upstream's `results/`. The qualify script
checks LocalRouter's integration and the engine's self-consistency (cold and warm, whole and streamed), reruns
upstream's reproducible accuracy checks on this host and records their verdicts, but does not require them to pass;
`--accept-native-numerics` is the operator's recorded acceptance of the native build as the baseline, and the marker
says so next to those verdicts. Upstream's numbers come from upstream's hardware and corpus; none was measured here.

## Measured on DGX-01

DGX-01: NVIDIA GB10 (sm_121), 121.7 GiB `MemTotal`, driver 580.178.04, CUDA 13.0, Zig 0.17.0. 2026-10-21, from
LocalRouter at `6a15091`; engine `tensorfold-native` sha256 `46bfd0b6…` (TensorFold `459771e7`, the pinned patched
HEAD). The checkpoint files were the ones already on the host, each checked against the pins (17 files, no download).

**Qualification (`run1`: nothing else on the GPU).** assets 12/12, e2e 15/15, bench 14/14, accuracy evidence valid,
`--accept-native-numerics`: qualified. Marker sha256 `0b5399f672c3124afb100ae0823d722bc00237d738bddca4faab9c66d85ecbdf`;
the run's manifest (`SHA256SUMS`, 399 files) sha256 `090cf1bb…`. That marker was superseded by `run3` below, when the launcher
patch changed the pins.

| Upstream's own client and corpus (`bench/drama_bench.py`, 24 texts, 3 warm passes) | DGX-01 | Upstream's claim | Gate |
| --- | ---: | ---: | --- |
| First playable audio, streamed, warm p50 (p95) | **0.142 s** (0.180) | 0.141 s | ≤ 0.5 s |
| Real-time factor, whole replies, warm p50 (p95) | **0.663** (0.706) | 0.634; 0.682 after its patch 0040 | < 1.0 |
| Real-time factor, streamed, warm p50 | 0.686 | | < 1.0 |
| End to end, whole, warm p50 | 5.61 s | | |
| Launcher start to ready (warm page cache) | 6.45 s | 6.4 s | |
| Peak memory (`MemAvailable` drop; the cgroup's own peak 1.1 GB) | **39.84 GiB** (42.78 GB) | | ≤ `resident_bytes` |

e2e: a cold load, the same sentence twice (byte-identical), streamed (the same PCM as whole), the 300-character
paragraph whole and streamed (equal), references from a voice file, an unknown voice refused (400) with the server
still up, and unload with the unit inactive, the lock free and `MemAvailable` back.

**Beside the chat model (`run2`, `--coresident`).** The LocalRouter daemon of PR #14's chat qualification
(Qwen3.8-27B, its qualified marker, 24 GiB `resident_bytes`) warm, then the same qualify run; it cannot qualify (the
accuracy guard needs the host lock to itself), and did not touch the marker.

| | Drama 3 alone | Drama 3 with chat warm |
| --- | ---: | ---: |
| First audio p50 / RTF p50 whole / RTF p50 streamed | 0.142 s / 0.663 / 0.686 | 0.142 s / 0.665 / 0.689 |
| Peak memory | 39.84 GiB | 40.36 GiB |
| `MemAvailable`: idle, chat loaded, both loaded (lowest) | 114.4 GiB idle | 94.8 GiB chat; 54.4 GiB both |

Chat through the daemon (each of its 8 qualify prompts streamed twice; time to first token, decode tokens a second):
alone, P1 0.164 s and 83 tok/s, P2 0.164 s and 73 tok/s; with Drama 3 loaded, P1 0.165 s and 84 tok/s, P2 0.165 s and
75 tok/s; every prompt within run-to-run noise. Both meet their targets together: Drama 3 faster than real time with
first audio in 0.14 s, chat at its qualified speed, and 54 GiB still free with both loaded (60.0 GiB in use above idle).

**Upstream's accuracy checks** (`run1`, reference runtime from `reference.sh`: torch 2.13.0+cu130, sglang 0.5.21), all
recorded in the marker, none required:

- Backbone: all-row cosine 0.99860 (bar 0.999), relative L2 0.0528 (bar 0.05), last-row cosine 0.99809 (bar 0.999),
  worst row 0.9169 (registered bar 0.8713, passes), semantic top-1 equal, top-30 0.933, KL 0.0062: the same figures
  upstream publishes after its patch 0040, failing the same three bars.
- Tower, teacher-forced on upstream's default fixture of synthetic random hidden states: 1 row bit-exact; 7 rows
  cosine 0.659 and 128 rows 0.869 (bar 0.999). On this fixture the reference does not agree with itself either: its
  chunked against batched output has cosine 0.334 at 7 rows and 0.790 at 128, and the per-layer key and value cosines
  fall from 1.0 at layer 0 to 0.71 at layer 12, so random inputs amplify rounding differences through the 13 layers.
  Upstream's passing tower result was on real hidden states (`--input-hidden`, its R6c), which its repository does not
  ship. The failure is recorded as measured; nothing was loosened.

**Requalified (`run3`, LocalRouter `738f84d`: the launcher patch now lowers the engine's reserve, so the pins changed).**
assets, e2e, bench and accuracy all passed again: first audio p50 0.142 s, warm RTF p50 0.663, peak 40.45 GiB. Marker
sha256 `a860364dd152537e98f67f818e768444df84ce8c6cab4b23125889e5d2d55db4` (`pins_sha256` `e30d5396…`); manifest sha256
`37c314ca…`. The later worker change (`adbf05a`: a busy reply after a cancel is retried) changes no pin and leaves
that marker valid.

**Requalified at `resident_bytes` 42 GiB (`run4`, LocalRouter `889cf70`; same pins, same engine).** assets, e2e, bench
and accuracy passed: first audio p50 0.142 s, warm RTF p50 0.667, peak 39.88 GiB (42,818,375,680 bytes, within the
45,097,156,608 booked). Marker sha256 `90f9f4a36f7754737435a6522d6ffe28fb7b7591ad99bc70d6e3e48ea8b3962a`; manifest
sha256 `51434b8f…` (399 files). Through the daemon at the old and new figure (one streamed paragraph, three times):
loaded and idle the drop was 39.97 and 39.45 GiB, no higher while speaking, and back to within 0.2 GiB after unload;
the scheduler booked 48.45 GiB (8.49 more than the drop) before and 42 GiB (2.54 more) now.

**Requalified with the named voices and the prefix cache (`run6`, LocalRouter `74ee0d6`: new pins, engine
`efc6c2f8`, binary sha256 `4248109c…`).** The six voices served, `Mae` the default, so the bench spoke in `Mae`.
assets, e2e (each voice, and an unknown voice refused), bench and accuracy passed: first audio p50 0.137 s, RTF p50
0.675 whole and 0.696 streamed, peak 40.30 GiB (43,270,909,952 bytes, within the 45,097,156,608 booked). Marker sha256
`7ec4c40a572fb787bae493f56c71c4c94500b0ba6ee4f2a93484fe254a8aa908`; manifest sha256 `eb0740e1…` (406 files).
Upstream's accuracy oracle admits only with 88 GiB free, so the Jarvis service is stopped for a qualify run (`run5`,
with it up, had 52 GiB free and could not run it). Deployed to the Jarvis box: `smoke.sh --quick` 19 of 19.

**LocalRouter's admission is the only one (through the daemon, `verify/` on the node).** Ballasts held by a bounded
`systemd-run` unit, Drama 3 loaded on demand by a `localrouter serve` built from this branch:

| Free at load | Launcher | Result |
| --- | --- | --- |
| 70.7 GiB (44 GiB held) | before: engine reserve 34 GiB | load fails, `OutOfDeviceMemory` (as in the voice e2e run `eca23de7`), 503 |
| 70.7 GiB (44 GiB held) | now: reserve 2 GiB | loads (cold, 8.3 s), whole reply 200 `stop`; lowest `MemAvailable` 31.1 GiB |
| 93.4 GiB (22 GiB held) | now | loads (8.1 s), 200 `stop` |

**Soft cancel, through the daemon.** SIGUSR1 to the worker 1 s into a streamed reply: the stream ends without its
last chunk (curl exit 18), the worker logs the cancel, and the next request is served by the same worker and unit (no
reload) in 1.72 s, byte-identical to the first reply. Before the busy retry, that next request was refused 503 "the
engine is busy" 3.7 ms in, while the server was still dropping the cancelled synthesis.

## The NVFP4 pruned weights

[plotarmordev/drama-3-NVFP4-Pruned](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned) is an unofficial
community quant of fishaudio/drama-3 (at `e0a9218`, the revision the full model is pinned at), not made, reviewed or
endorsed by Fish Audio. It is lossy, and JJ approved it for the Jarvis box on 2026-10-24.

| | |
| --- | --- |
| Licence | Fish Audio Community License v1.0: its `LICENSE` is byte-identical to Drama 3's (the same sha256), with a `NOTICE`; [Licences](#licences) applies unchanged |
| Format | safetensors, BF16 and U8. The routed experts that were block FP8 are NVFP4 (GPTQ, calibrated on 120 English requests); every other weight, the codec included, is unchanged (`codec.pth`, the config, tokenizer and chat template have the full model's sha256, so fetch.sh links them rather than downloading them again) |
| Pruning | 7,369 of the 10,240 routed experts kept (40 layers of 256; 145 to 256 a layer): each layer keeps the experts that cover 90 % of router weight on English calibration requests and private voice-cloning requests. The kept list is `metadata.drama_experts` in `model.safetensors.index.json` |
| Size | 22,294,649,076 bytes of weights and codec (20.76 GiB; the full model's 41.5 GB), 22.3 GB downloaded |
| Engine | only TensorFold native loads it (not Transformers, not SGLang-Omni): plotarmordev/TensorFold `drama-nvfp4` @ `b1f4b9b4`, one squashed commit (`v1.0.11-193-g70dec469`, the build upstream measured) on `c7137bb3`, the TensorFold commit the full engine's 59 patches apply to. It carries their features (voice files, `--max-frames`, the silence guard, long form, stream trailers) and adds NVFP4 experts, compact checkpoints and guidance |
| Served with | `--cfg-gamma 1.5 --cfg-batched`, as upstream measured it: classifier-free guidance from a second backbone sequence holding the plain prompt (no references), decoded in one batch with the first every frame. It acts only on a prompt with audio rows, a voice's references: every Jarvis request has one (`Mae` by default) |

**Upstream's claims** (drama-tensorfold `results/full-vs-quant-01`: one GB10, 24 English texts, 4 seed salts, 96
takes an arm, Whisper large-v3 word error rate; "not proof of equal quality"): with a reference voice, WER 1.31 %
[0.04, 3.32] against the full model's 1.42 % [0.00, 3.48] (paired difference -0.11 pp [-0.35, +0.08], 0 failed takes
of 96 on either) and speaker cosine to the full model's takes 0.900 against the full model's own 0.901; in the
built-in voice 2.92 % against 6.23 %. Peak device memory 19.35 GiB against 36.95; cold start 26.3 s against 5.9;
first streamed audio p50 1.955 s against 2.428 (its 1,335-row reference prompt, prefilled whole: no prefix cache).

**How LocalRouter serves it.** The tool option `"variant": "nvfp4"` (default `"full"`) selects `pins-nvfp4.json` beside
`pins.json`: the checkpoint in `ROOT/drama-3-NVFP4-Pruned`, the engine in `ROOT/build/tensorfold-nvfp4` (recorded in
`ROOT/engine-nvfp4.json`), and the serve flags. `fetch.sh --variant nvfp4` and `build.sh --variant nvfp4` fill them;
the upstream checkout, the launcher and the voices are the full model's, unchanged. `"guidance": false` serves it
without the flags (unguided; not what upstream measured). A marker for it carries `"variant"` and
`variant_pins_sha256` beside `pins_sha256`, and the worker refuses one for the other weights, so the two installs
qualify independently and switching between them is the tool entry alone (`DRAMA_VARIANT` on the Jarvis box).

**The prefix cache, ported** (`patches/engine-nvfp4/`). LocalRouter's engine patch for the full build does not apply
to this engine: its backbone keeps a request's DeltaNet states, KV cache and position in a `Sequence` (the guidance
sequence is a second one), and the host's prompt pass is a function shared with its probe tool. The port takes the
same snapshot and restore from the Model's own sequence, and the prompt pass gains a resumer hook
(`prefillResuming`); `prefill`, which the probe calls, runs as before. The guidance sequence prefills the plain
prompt (28 rows for a short sentence) and is not cached. Without the port a sentence in a voice would prefill its
360 to 730 reference rows every time. On DGX-01 it resumes as before: `prefix cache: resumed after 385 prompt rows
(168 MiB)`.

**Qualified on DGX-01, 2026-10-24** (`~/jarvis/qualify-drama3/nvfp4/run1`, `--coresident`: chat, Whisper and the
image model warm in a LocalRouter daemon; `--accept-native-numerics --accept-lossy`; engine sha256 `31cf24d3…`, the
same binary from the manual build and from `build.sh`). assets, e2e (every voice, refusals, unload) and bench
passed; upstream's accuracy checks were recorded as not applicable. Marker `drama3-nvfp4-qualification.json`.

| Upstream's client and corpus, 3 warm passes, the `Mae` voice | NVFP4, beside the warm models | Full model (`run6`, alone) |
| --- | ---: | ---: |
| First playable audio, streamed, warm p50 (p95) | 0.198 s (0.263) | 0.137 s |
| Real-time factor, streamed, warm p50 (p95) | 0.714 (0.780) | 0.696 |
| Real-time factor, whole, warm p50 (p95) | 0.909 (1.649)¹ | 0.675 |
| Launcher start to ready (warm page cache) | 10.1 s | 6.45 s |
| Peak, per process: nvidia-smi `used_memory` + RssAnon + RssShmem | **20.31 GiB** (19.69 + 0.57 + 0.05) | 38.9 GiB (38.3 + 0.6) |
| `resident_bytes`: the peak plus 256 MiB | **22,075,092,992 (20.56 GiB)** | 42,144,366,592 (39.25 GiB) |

¹ Not the engine's speed: another project's GPU job (`mm31_kl_gate`, which pauses and resumes itself) ran through the
whole-reply passes, and the same text, the same audio to the sample, took 0.68 in one pass and 1.40 in the next. Each
text's fastest whole pass was 0.68 to 0.70, and the streamed passes after it, on the same server, held 0.70 to 0.72
for every long and medium text and 0.75 to 0.83 for the short ones.

**Against the full model** (`tools/speech/drama3/compare.py` through a LocalRouter daemon on DGX-01, 2026-10-24,
beside warm chat and Whisper; `~/jarvis/qualify-drama3/nvfp4/compare3`): the 24 sentences of
`compare-texts.txt` in each of the six voices (144 takes a model), then 6 of them in each voice (36 takes) while chat
streams a 600-token reply back to back; every take streamed. Interleaved (`--interleave`): both models loaded, each
text spoken by one and then the other, because another project's GPU job ran through every window (2.4 to 5.8 GiB of
`mm31_kl_gate`, which pauses and resumes itself) and slowed whichever model happened to be speaking. Whisper
large-v3-turbo transcribed every take; word error rate after NFKC, lowercase, punctuation dropped, digits spelled out
and UK/US spellings folded (Whisper writes "twenty" as "20" and "harbour" as "harbor", the same for both models).

| | Full model | NVFP4, γ 1.5 (served) | NVFP4 unguided (`guidance: false`)² |
| --- | ---: | ---: | ---: |
| First audio p50 (p90), alone | 0.264 s (0.567) | 0.324 s (0.710) | 0.243 s (0.246) |
| Real-time factor p50, alone | 0.713 | 0.741 | 0.692 |
| First audio p50 (p90), while chat writes | 0.594 s (2.109) | 0.728 s (1.880) | 0.546 s (1.710) |
| Real-time factor p50, while chat writes | 1.696 | 1.780 | 1.640 |
| Chat decode while speaking | 15.2 tok/s | 15.2 tok/s | 16.1 tok/s |
| Word error rate, pooled (144 takes, 1,938 words) | 0.05 % (1 word) | 0.05 % (1 word) | 0.05 % (1 word) |
| Failed takes, finish reasons | 0; stop 180 | 0; stop 180 | 0; stop 180 |
| Load through the daemon (page cache warm) | 8.1 s | 25.1 s | 24.3 s |
| GPU memory of the server, nvidia-smi | 38.5 GiB | 19.7 GiB | 19.6 GiB |

Per voice (first audio p50, RTF p50, WER), NVFP4 γ 1.5 against the full model: Mae 0.326 s 0.762 0.0 % (0.266 s
0.730 0.0 %), Arlo 0.324 0.754 0.0 % (0.264 0.722 0.0 %), Ezra 0.324 0.754 0.0 % (0.263 0.725 0.0 %), Nora 0.324
0.733 0.0 % (0.265 0.710 0.0 %), Theo 0.324 0.730 0.3 % (0.263 0.707 0.0 %), Wren 0.324 0.726 0.0 % (0.263 0.701
0.2 %). No voice degrades by compare.py's flags (WER mean 3 points worse with at least 2 takes worse, a failed take,
or RTF p50 above 1.0). The misheard words: the full model's "pasta" heard as "pastor" (Wren, in both runs), the
quant's "ring" as "wring" (Theo), and unguided "lights are off" as "lights her off" (Theo). Nobody has listened to
the samples yet. Every take of both models, side by side
per voice and text, with a page to play them: `compare3/wav/` and `compare3/listen.html` on DGX-01.

What it costs: guidance's second backbone sequence, decoded in one batch with the first every frame and prefilled
with the plain prompt first, adds 60 ms to first audio (0.264 to 0.324 s, every voice alike) and 4 % to the
real-time factor; while chat writes, both models fall below real time alike (the GPU is shared; the voice loop's
cushion covers it, [VOICE.md](VOICE.md)). Without guidance the quant is faster than the full model on every measure,
at the same word error rate here; upstream measured it only guided, and speaker similarity was not measured here.

² `compare2`: sequential passes (each model's 180 takes in turn) beside chat, Whisper and the image model, against
the full model in the same run (whose figures there were 0.282 s, 0.732, 0.587 s and 1.662, near the interleaved
ones); the foreign job was present then too.

**Back to the full model.** Its install, engine and marker are untouched by all of this (`pins.json` did not change):
set the tool's `variant` to `full` with its own marker and `resident_bytes` (on the Jarvis box: `DRAMA_VARIANT=full`
and `deploy/jarvis/install.sh`).

## macOS (Apple silicon)

On a Mac, Drama 3 runs on plotarmordev's drama-tensorfold **Mac path**: TensorFold's native Zig/Metal engine at that
repository's own Mac NVFP4 pin and patches, serving the same NVFP4 pruned weights as the GB10 (JJ's decision,
2026-10-30; not mlx-audio). The GB10 files (`pins.json`, `pins-nvfp4.json`, `fetch.sh`, `build.sh`, `qualify.sh`,
`drama3_worker.py`) are not used on macOS and behave as before on Linux.

```
daemon ── guard ── drama3_worker_macos.py ── HTTP on 127.0.0.1:port ──> tensorfold-native serve (Metal)
                         │                                                  ▲ own process group, holds the lock
                         └── launcher/pilot_macos.py --external-admission --parent-pid WORKER ──┘
```

### Mac pins and setup

`tools/speech/drama3/pins-macos.json`:

| What | Pin |
| --- | --- |
| Upstream | plotarmordev/drama-tensorfold @ `db6ba7da314b24b42b02978ba4401f54f6738311`; sha256 of `recipe/pilot_macos.py`, `recipe/weight_selection.py`, `scripts/build-engine.sh`, `engine/nvfp4/macos/UPSTREAM` and its `patches/SHA256SUMS`, `recipe/checksums/nvfp4.sha256`, `recipe/sources.json`, `recipe/README-macos.md`, `results/macos-nvfp4.json`, `LICENSE`, `NOTICE` |
| Engine | ashhart/TensorFold @ `3d219d2156d4c25dcfb74370c8703cd89e58aa49` (the 1.1.4 line, Metal) plus the five patches in `engine/nvfp4/macos/patches/` (Metal NVFP4 and CFG, a test fixture, the CLI extension point, sampling controls), each by sha256, applied by upstream's `scripts/build-engine.sh --weights nvfp4` with its fixed committer: HEAD `a2b8f03e2330e2eb974cfcaa15827a239dc6bef7`. No LocalRouter engine patch: the Metal base already keeps one reference-prefix snapshot (below) |
| Launcher | upstream's `recipe/pilot_macos.py` with `patches/pilot-macos-external-admission.patch` (sha256 `a4e701a1…`), written to `ROOT/launcher/pilot_macos.py` (sha256 `24e617ab…`) beside the module it imports |
| Voices | the six `voices/*.json` converted by `voices_macos.py` to `ROOT/voices-macos/`, each output by sha256 |
| Checkpoint | plotarmordev/drama-3-NVFP4-Pruned @ `494db08481db952f834b2241bce63fb3dfa72554`: the same 17 files, sizes and sha256 as `pins-nvfp4.json` (upstream's `recipe/checksums/nvfp4.sha256` at the pinned commit, plus LICENSE and NOTICE) |
| Toolchain | Zig 0.17.0, Xcode's Metal compiler; serve flags `--cfg-gamma 1.5 --cfg-batched`, as on the GB10 |

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer   # when xcode-select names the Command Line Tools
tools/speech/drama3/setup-macos.sh ~/models/drama3                  # checkout, launcher, voices, weights, engine
python3 -I tools/speech/drama3/qualify_macos.py CONFIG OUT --accept-native-numerics --first-audio-gate 0.55
```

`setup-macos.sh` verifies the checkout and every pinned file, writes the patched launcher, converts the voices, takes
the weights with the Hugging Face CLI into its shared cache (`ROOT/drama-3-NVFP4-Pruned` links to the snapshot;
without the CLI it downloads each pinned file with curl), checks every file's sha256, then builds the engine with
upstream's script and records the server's, `drama3-generate`'s and `drama3-voice`'s sha256 in
`ROOT/engine-macos.json` (2 min 21 s on the M5 Ultra). Without a usable Metal compiler (Xcode's licence not accepted,
or the Metal Toolchain missing) it stops with exit 3 after everything else is in place. The tool is
`config.example-macos.json`; `deploy/profiles/mac-apple-silicon.json` has it as `tts`.

### What differs from the GB10

| | GB10 (`drama3_worker.py`) | Mac (`drama3_worker_macos.py`) |
| --- | --- | --- |
| Launcher | `pilot.py` in a transient `systemd --user` unit, 54 GiB cgroup and allocator caps | `pilot_macos.py`: the server in its own process group holding the inherited lock, settled TERM then KILL; no memory cap exists on Metal |
| Admission | `--external-admission` skips the 88 GiB / 34 GiB checks | `--external-admission` skips the free+inactive floor (weights + 30 GiB) and the 34 GiB abort, which would otherwise kill the server beside a warm chat model |
| Worker killed | parent-death signal on the launcher; a stale unit is stopped by name | `--parent-pid`: the launcher stops the server once the worker is gone (the daemon's `guard` kills the worker's group; the launcher has its own session). A launcher killed outright leaves `runtime/localrouter-drama3.server` (the group), which the next load stops after checking that its leader is still the engine binary |
| Voices | `--voices DIR` of `drama3-voice/1` files, and references in a request | only the `--voices` catalog (`drama3-http-voice/1`), read at start; references in a request are refused (`references: false`) |
| Cloned voices | references inline in each request, no restart | a catalog voice: one server restart the first time a clone is spoken (below) |
| Prefix cache | LocalRouter's patch: 2 voices, LRU | upstream's: one reference-prefix snapshot; a change of voice prefills the new voice's references |
| Clone encode | `tensorfold-native voice encode` (BF16, CUDA) | `drama3-generate --encode` (Metal), beside the running server |
| Long form | `--long-form` | none in the Metal server: `long_form` is refused; `max_frames` up to 8190 |

### Voices on the Mac

A `drama3-voice/1` file holds `vq_codes` codebook-major (10 rows of `frames` codes); the Metal catalog holds `codes`
frame-major (`frames` rows of 10, the semantic code first), with an attested `source` (`synthetic`, `self` or
`consented`) and a note. Both come from the same codec (`codec.pth`, one sha256 in both checkpoints) in the same
codebook order, so `voices_macos.py` transposes the codes and does not re-encode. It first checks each file against
its own `codes_sha256` and frame count and against the catalog's limits (1 to 4 references, 1 to 646 frames each,
codebook 0 in 0..4095 and 1 to 9 in 0..1023, transcripts without the prompt's structural tokens, 256 KiB a file), and
writes `source.kind: "synthetic"` (the voices are model-invented, `voices/SOURCES.md`) with the file's provenance in the
note. The output is byte-stable, so its sha256 is pinned.

Each converted voice spoke on the M5 Ultra; Whisper large-v3-turbo (MLX) heard every first reply word for word ("Good
morning, the kettle is on and your first meeting starts in twenty minutes.", written "20"), and the median F0 of each
reply matches the voice's pitch range in `voices/SOURCES.md`: Arlo 110, Ezra 102, Theo 113, Mae 188, Nora 215, Wren
225 Hz.

### Cloned voices on the Mac

The Metal server takes no references in a request and reads its catalog only at start, so the worker publishes a
clone to the catalog itself (the simplest route the server allows: offline encode, catalog, reload):

1. **Encode** (`clone`): each prepared clip (canonical mono 16-bit 44.1 kHz WAV, at most 30 s; the worker refuses
   others) through `drama3-generate --codec codec.pth --ref-audio CLIP --encode OUT`, on Metal beside the running
   server, and written as the GB10's `voice.json` (`localrouter-voice-codes/1`, codebook-major), so a saved voice is
   the same file on either host. A clone whose references cannot fit the catalog (below) is refused here.
2. **Speak** (`voice_references`): the references become a catalog voice named for its content (`clone-` and 16 hex
   digits of the file's sha256), `source.kind: "consented"` (the daemon accepted the clone only with `consent=true`
   and keeps that statement). The clone's up to 8 references are packed into the catalog's 4 by joining the adjacent
   pair with the fewest frames (transcripts with a space, codes end to end) while a joined reference stays within 30 s.
3. **Reload**: a clone the running server has not loaded goes into `runtime/clones/` and the worker restarts the
   server with it, inside that request. The 16 most recently spoken clones are kept and put into every later load's
   catalog, so only a clone's first request after it falls out pays the restart.

Limits against the GB10: the first request in a new clone waits for a reload (5.3 s on the M5 Ultra, plus the clone's
first-sentence prefill: 11.45 s to its whole first reply); cloned references are capped at 4 of 30 s after packing;
a request's own `references` are not offered; while the server restarts nothing else is spoken. Cloning through
`POST /v1/voices` also needs the transcription model, so on the Mac it waits for the Whisper port; the measurement
below drives the worker's `clone` op directly.

### Qualification on the Mac

`qualify_macos.py` runs everything through the real worker on this machine and writes `report.json` and, with
`--accept-native-numerics`, the tool's marker: `{"qualified": true, "host", "platform", "chip", "pins_sha256" (of
pins-macos.json), "native_sha256", "upstream_commit", "patched_head", "checkpoint_revision", "numerics", "bench",
...}`. The worker refuses to load without it, on another host, with other pins or another engine binary. Steps:
assets (launcher and checkpoint by sha256, binaries against `engine-macos.json`), a cold load, e2e (one sentence whole
twice, byte-identical, and streamed, the same samples; an unknown voice and request references refused), each voice's
first N lines of `compare-texts.txt` streamed straight from the server (first audio, RTF, level, 44.1 kHz, `stop`), a
clone of a model-invented voice (an 11.4 s passage in the built-in voice, cloned with the worker's `clone`, spoken with
its restart, then measured as a voice), and unload (the server's group gone, the lock free). It uses its own runtime
directory, so the operator's clone cache is not touched.

**Numerics.** The Metal build is a new baseline, not the GB10's: NVFP4 expert codes unpacked in Metal kernels with
FP32 accumulation, CFG as two independent backbone states in one command buffer. It is deterministic on the Mac (the
same text in the same voice is byte-identical; streamed equals whole) but not compared with the CUDA build or the
reference; upstream's own M5 Ultra receipt (`results/macos-nvfp4.json`) is correctness-only.

**First-audio gate.** The GB10's bar is warm streamed first audio p50 at most 0.5 s. With guidance (`--cfg-gamma 1.5
--cfg-batched`, as on the GB10) the M5 Ultra measures 0.52 s: the server already sends its first chunk after 4 frames
(186 ms of audio); the rest is resuming the voice's snapshot and prefilling the text for both CFG states. Decided
2026-10-30: the Mac keeps guidance on, as the GB10 does, and qualifies at `--first-audio-gate 0.55` (the Mac profile's
qualify step).

**Unguided, for later.** `"guidance": false` (no CFG) measures 0.25 s warm first audio and RTF 0.54 on the M5 Ultra
(below), but its effect on quality with the named and cloned voices has not been measured (no WER, similarity or
listening comparison against guided). It is an option to evaluate, not a configuration to deploy.

### Measured on the Mac Studio (M5 Ultra)

Apple M5 Ultra, 96 GiB (103,079,215,104 bytes), macOS 27.2, Xcode 27.2 Metal Toolchain, Zig 0.17.0; 2026-10-30;
engine `tensorfold-native` 1.1.4 sha256 `fd3618ed…` (HEAD `a2b8f03e`); checkpoint in the page cache. Five lines of
`compare-texts.txt` per voice, streamed (the first fills the voice; warm figures are the other four); every reply
44.1 kHz mono, `stop`, -14 to -27 dBFS. Each run held the development machine's GPU lock alone.

| Voice | First sentence, first audio | Warm first audio p50 | Warm RTF p50 (range) |
| --- | --- | --- | --- |
| Mae (default; already filled by e2e) | 0.56 s | 0.522 s | 0.780 (0.771 to 0.790) |
| Arlo | 4.29 s | 0.520 s | 0.776 (0.770 to 0.784) |
| Ezra | 4.63 s | 0.518 s | 0.787 (0.785 to 0.796) |
| Nora | 5.02 s | 0.519 s | 0.758 (0.748 to 0.763) |
| Theo | 5.94 s | 0.524 s | 0.758 (0.744 to 0.763) |
| Wren | 5.84 s | 0.523 s | 0.738 (0.724 to 0.756) |
| Clone (11.4 s source, 1 reference) | 11.45 s with the restart | 0.516 s | 0.794 (0.789 to 0.802) |
| **All named voices** | | **0.522 s** | **0.766** |
| Unguided (`guidance: false`), all named voices | 4.0 to 5.7 s | 0.254 s | 0.536 |

The same figures came out of four guided runs within 0.02 s and 0.02 RTF.

- **Load:** 5.2 s to ready with the checkpoint in the page cache (the server reports 6.36 GB of BF16 and copied
  weights and 13.04 GB of packed experts read in place); 12.8 s for the first load after setup.
- **A change of voice** prefills the new voice's references (280 to 431 frames): 4.3 to 5.9 s to first audio, against
  0.7 to 1.0 s on the GB10. One snapshot is kept, so a call that stays in one voice pays it once; alternating voices
  pays it every time.
- **Memory.** The server keeps 27.5 GiB wired between requests (its log), its `phys_footprint` peak is 9.75 GB (the
  packed experts are file-backed and not counted there), its resident size 17.7 GB. What the machine loses is what
  the scheduler must book: free+inactive fell by 31.15 to 32.76 GB (29.0 to 30.5 GiB) across three guided runs with the
  server alone, and wired memory rose by 30.80 GB through the daemon (unguided: 31.72 GB). That drop overstates it:
  measured under load in the integration daemon with the full Mac set loaded (2026-10-31), Drama 3 held a 9.1 GiB
  footprint plus 19.1 GiB of weights wired while speaking, a 28.2 GiB peak. `resident_bytes` (and the profile's
  `DRAMA_RESIDENT` and `memory_bytes`) is therefore 30,870,077,440 (28.75 GiB: about 0.55 GiB above that peak; it was
  33,285,996,544, 31.0 GiB, from the drop). After unload, wired memory was back within 0.7 GB.
- **Clone encode:** 3.43 s for one 11.4 s reference through the worker; `drama3-generate --encode` peaks at 3.29 GB
  footprint (3.10 GB resident), so `clone_working_bytes` is 3,758,096,384 (3.5 GiB).
- **Through the daemon** (`localrouter serve` from `feat/mac-foundation`, the tool from `config.example-macos.json`):
  listed `qualified: true` with the six voices, `references: false`; kept loaded at start in 6.1 s; whole WAV and
  streamed PCM (`x-sample-rate: 44100`, `x-channels: 1`, chunked) in Mae, Arlo and Theo, first byte of a warm stream
  0.51 to 0.53 s, a voice's first request 4.5 to 5.9 s; `instructions` spoken as a bracketed direction; an unknown
  voice and request references refused with 400. SIGKILL of the daemon took the worker, launcher and server down
  within 1 s (the launcher's `--parent-pid`, its completion record `group_settled: true`).

**Not verified on the Mac:** cloning through `POST /v1/voices` (needs the transcription model); a real person's
clips; long inputs past 1024 frames; the silence guard on the real model; speaker similarity (only pitch and
transcripts were checked); co-residence with the chat model; a cold load from disk rather than the page cache; Macs
other than the M5 Ultra.

**Licence.** Unchanged by the platform: the weights are FACL v1.0 ([Licences](#licences)), an unofficial community
quant under it. No Hosted Service without a separate licence from Fish Audio: keep the Mac's LocalRouter on loopback
for this model, as on the GB10. Built with Fish Audio.

## What was taken from upstream

From plotarmordev/drama-tensorfold at `1a53b8a`, used unmodified from the fetched checkout (nothing is copied into
this repository):

| Upstream file | Used for |
| --- | --- |
| `recipe/pilot.py` | the launcher the worker runs, as `ROOT/launcher/pilot.py` with LocalRouter's `--external-admission` patch: its arguments, its stdout (`{"unit": …}`, `DRAMA_PILOT_READY`), its lock, cgroup cap and stop |
| `scripts/build-engine.sh`, `engine/UPSTREAM`, `engine/patches/*.patch` | the engine build `build.sh` runs, pinned by hash |
| `recipe/checksums/checkpoint.sha256`, `recipe/sources.json` | the checkpoint pins |
| `qualification/run-bounded-oracle.py`, `backbone-reference-check.py`, `rescore-native.py`, `tower-reference-check.py` | the accuracy step `qualify.sh` runs, unmodified, against the reference at `sources.json`'s revision |
| README ("HTTP contract", "Long replies", "Voice cloning", "License") and, in the patched TensorFold tree, `zig/src/server/speech.zig` and `docs/drama-voices.md` (patches 0007, 0015, 0018, 0031, 0032, 0035, 0039, 0041, 0042) | the HTTP contract the worker speaks and the test stand-in (`tests/fixtures/drama3_pilot.py`) follows: fields, WAV and PCM, chunked streams, `X-Finish-Reason` header and trailer, `X-Long-Form-Segments`, `/v1/voices`, 400/503/500 |

## Not verified

Through the daemon on the GPU: the qualify run drives the worker in-process, as the daemon's worker process does, so
the daemon's HTTP re-chunking of a stream (a 20 ms poll of `speech.wav`) is covered by CI only. Also not run on the GPU:
the silence guard and long form on the real model, voice files in `voices_dir` (only inline references were), and
eviction of the real model by an image or video job (CI covers the order with stand-ins). Since 2026-10-24
compare.py streams through the daemon on the GPU (both weights, every voice) and scores every take with Whisper
([The NVFP4 pruned weights](#the-nvfp4-pruned-weights)); audio quality has still not been judged by ear, nor speaker
similarity measured. For the NVFP4 weights also: long form, the silence guard and the engine's own host tests
(`zig build test-cuda-host test-server-cpu` in the engine tree) were not run.

## What upstream lacks that LocalRouter needs

- A test mode: there is no CPU or scripted engine to run the server without a GPU, so CI tests the worker against a
  stand-in of the HTTP contract, not the server.
- A way to let an outer scheduler admit the load: the launcher's admission (88 GiB free) and abort (34 GiB) are
  constants in `pilot.py` with no option to turn them off, hence LocalRouter's `--external-admission` patch (a
  candidate to offer upstream). It also reports no resident size, so the tool's figure comes from qualify's bench.
- A launcher that cleans up when killed: its unit outlives a SIGKILLed launcher, so the worker keeps the unit's name
  and uses a parent-death signal; a stop hook in the unit, or a unit name chosen by the caller, would close that gap
  upstream.
- A free port: the server takes a fixed `--port` and the launcher refuses one in use; the tool's `port` must be
  unique on the host.
- Control a speech API usually has: no `speed`, no `seed` (each synthesis is seeded from its prompt), no presets.
- Numerical qualification against the reference ([Numerics](#numerics)). Its tower check's real-input fixture is not in the repository (the default synthetic fixture does not let even the reference agree with itself). Its reference runtime (the `sglang-omni` venv its accuracy checks need) is pinned by checkout revision only, not by package versions.

## Licences

| Part | Licence |
| --- | --- |
| drama-tensorfold (launcher, scripts, engine patches, codec implementation) | Apache 2.0, Copyright 2026 plotarmordev |
| TensorFold | Apache 2.0 (code before 0.6.0: MIT) |
| flash-linear-attention kernels the patches adapt | MIT, Copyright (c) 2023-2025 Songlin Yang, Yu Zhang |
| Drama 3 weights and codec checkpoint | [Fish Audio Community License v1.0](https://huggingface.co/fishaudio/drama-3/blob/e0a9218aa48888270f06ef1ed9ea67622c0e17d2/LICENSE) |

LocalRouter does not distribute the weights: `fetch.sh` downloads them from Hugging Face. The upstream checkout keeps
its NOTICE, LICENSE and LICENSES/. Anyone who redistributes the weights must keep the notice "These Fish Audio
Materials are licensed under the Fish Audio Community License, Version 1.0. Copyright © 39 AI, INC. All Rights
Reserved." (IV(a)), and anything that uses them shows "Built with Fish Audio". A summary, not legal advice:

- Research and non-commercial use is free (II). Commercial use only while the user's and its affiliates' annual
  revenue is under USD 500,000 (III(a)), and a client the model is deployed or operated for must be under it too.
- **No Hosted Service without a separate licence, whatever the purpose or revenue (III(c)):** letting third parties
  generate speech from text or audio they supply, for example through an API. LocalRouter is an API, so keep it on
  loopback (its default) for this model, and do not give others access to it, through a tailnet or otherwise, unless
  one of III(c)'s exclusions applies or Fish Audio has granted a licence. The exclusions cover speech for content the
  operator creates, selects or controls, operator-directed production that delivers only finished output, and a free,
  non-commercial research or demonstration service that is not monetized or offered as a production API.
- Use restrictions that also cover outputs (IV(b)): no imitating an identifiable person's voice without consent, no
  impersonation to deceive, no removing watermark or provenance signals, binding third parties to terms as
  protective, and no training or distilling another speech or generative model on the outputs. A breach ends the
  licence (IV(f)).

Whether a deployment is commercial, under the threshold, or a Hosted Service is the operator's decision and is not
settled by this code. Nothing here accepts the licence on anyone's behalf.
