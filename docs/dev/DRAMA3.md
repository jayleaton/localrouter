# Fish Audio Drama 3

[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) is Fish Audio's expressive text to speech model: a
40-layer hybrid (Gated DeltaNet and attention) text backbone, a 13-layer temporal codebook tower that writes ten codec
codebooks a frame, and Fish Speech's modded-DAC codec, at 44.1 kHz and about 21.5 frames a second. LocalRouter serves
it as `fishaudio/drama-3` through plotarmordev's [drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold),
which ports it to [TensorFold](https://github.com/ashhart/TensorFold)'s native Zig/CUDA engine for the GB10. The
worker, `tools/speech/drama3/drama3_worker.py`, runs that repository's own launcher and server, built from pinned,
hash-checked sources; it does not reimplement the model.

**Built with Fish Audio.** See [Licences](#licences): the weights' licence limits commercial use and forbids a Hosted
Service without a separate licence, so this tool stays on loopback.

> [!NOTE]
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
| Refused (400) | `speed` other than 1, `preset`, `instructions` (the daemon, before loading); an unknown voice, references the model refuses, a prompt longer than the model takes (the server, in its own words) |
| Not offered | audio upload for cloning (voice files are encoded offline, below); SSE streams; `seed` (the engine seeds each synthesis from its prompt, so the reply's seed is 0) |

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
| `voices`, `default_voice` | `["default"]`, `default` | what `/v1/models` lists and the daemon checks; a `default_voice` other than `default` becomes `--default-voice`. The example config lists the six shipped voices, `Mae` by default ([Voices](#voices)) |
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
Section IV(b) forbids imitating an identifiable person without consent. Voice files are made offline with upstream's
tools (`drama_voice_clip.py` prepares a 10 to 30 s take, `drama_voice.py` encodes it under the GPU lock and records its
`source`, which the server requires), as its README describes. A clip of a person's voice, and its voice file, hold
that voice: keep them out of source control. The six shipped files hold only voices the model invented.

## Pins

`tools/speech/drama3/pins.json` holds every hash. `fetch.sh` and `build.sh` refuse anything else.

| What | Pin |
| --- | --- |
| Upstream | plotarmordev/drama-tensorfold @ `1a53b8aea738a99d3630f949b6b2a1f9389f0795` (main, 2026-10-17, PRs #4 to #9); sha256 of `recipe/pilot.py`, `scripts/build-engine.sh`, `engine/UPSTREAM`, `recipe/checksums/checkpoint.sha256`, `recipe/sources.json`, `NOTICE`, `LICENSE`, `engine/patches/SHA256SUMS` |
| Engine | TensorFold @ `c7137bb31a7dad924fb541a7068422127b9a94b2` plus the 59 patches in `engine/patches/` (each by sha256; series sha256 `3fff8dd6…a910d247`), applied by `build-engine.sh`: HEAD `459771e7531431490b80d7a8149c9160e03c01ce`; then LocalRouter's `patches/engine/` (the voice prefix cache, by sha256), applied by `build.sh`: HEAD `efc6c2f82a4a12726e36dce66b0969c1d3aee3f7` |
| Voices | `voices/*.json`, the six named voices, each by sha256; `fetch.sh` installs exactly these into `ROOT/voices` |
| Checkpoint | fishaudio/drama-3 @ `e0a9218aa48888270f06ef1ed9ea67622c0e17d2`: `codec.pth`, eight weight shards (41,474,674,072 bytes together, matching upstream's `sources.json` and `checkpoint.sha256`), tokenizer, config, chat template and licence, each by size and sha256 |
| Toolchain | Zig 0.17.0, CUDA 13 |

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
eviction of the real model by an image or video job (CI covers the order with stand-ins). Audio quality was not judged
by ear or by a speech recognizer here.

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
