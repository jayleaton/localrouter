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

> [!WARNING]
> **Listed but refused until GPU-qualified.** Nothing here has run on a GPU yet. Until an operator runs
> `tools/speech/drama3/qualify.sh` on a GB10 and it passes, `/v1/models` lists the model with `"qualified": false` and
> every request gets `503 server_error: model 'fishaudio/drama-3' is not GPU-qualified on this host`. Upstream's own
> status is "testing, not qualified" against the SGLang-Omni reference ([Numerics](#numerics)).

## What it serves

| | |
| --- | --- |
| Request | `speech`: `input` (1 to 4096 characters), `voice`, `references`, `stream` (with `response_format: "pcm"`) |
| Output | `speech.wav`: RIFF, mono, 16-bit PCM, 44100 Hz. The daemon encodes the other response formats; a stream is raw PCM chunks |
| Voices | `default` (the model's built-in voice, or the operator's `default_voice` file), and the voice files in `voices_dir` |
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
  `tools/speech/drama3/patches/pilot-external-admission.patch` (sha256 `57d1fdde…`): `--external-admission` skips
  both checks and keeps everything else, its 54 GiB cgroup cap, the server's 54 GiB allocator cap and its lock.
  `fetch.sh` writes the patched launcher to `ROOT/launcher/pilot.py` (sha256 `40efba1d…`, pinned) and leaves the
  upstream checkout clean. `options.admission: "launcher"` restores upstream's checks for a launch with no
  scheduler.
- **Load.** The worker checks the qualification marker, the pinned launcher and the checkpoint files (`verify`), then
  starts the launcher with `--external-admission` and the lock file the daemon names in `LOCALROUTER_GPU_LOCK`: this
  tool's own, uncontended `<data>/locks/fishaudio_drama-3.lock`, while the scheduler holds the host's `gpu_lock`
  shared for every LocalRouter model ([the host's GPU lock](../ARCHITECTURE.md#the-hosts-gpu-lock)). The launcher
  starts the server as a transient `systemd --user` unit capped at 54 GiB with swap off. The worker is ready when the
  launcher prints `DRAMA_PILOT_READY`; then every voice the config lists must be one the server loaded.
- **Memory.** `resident_bytes` is 54 GiB (57,982,058,496 bytes), the launcher's cgroup cap and the server's own
  allocator cap, until `qualify.sh`'s bench measures the real peak (`peak_memory_bytes` in the marker). The worker
  reports no measured size, so the scheduler holds 54 GiB for it while it is loaded. With chat (24 GiB) and Whisper
  (about 4 GiB) that is 82 GiB of DGX-01's 106 usable (114 GiB free less the 8 GiB reserve): the three stay warm
  together, and an image or video job beside them evicts Whisper, then Drama 3 (lower priorities first), and they come
  back by themselves (`tests/drama3.zig`). For reference, TensorFold 1.1.1's own Drama 3 server was measured at 40 GiB
  on DGX-01 (docs/dev/JARVIS-BRINGUP.md, PR #16); drama-tensorfold's own peak has not been measured.
- **Requests.** One at a time, as for every LocalRouter tool (the server runs `--parallel 1`). The server's 400 is a
  400, its 503 (busy) a 503 and anything else a 500. A server that dies ends the worker without a reply: that job is
  a 503 with the log tail, and the next request starts it again.
- **Unload** (idle TTL, `POST /v1/tools/fishaudio/drama-3/unload`, eviction): the worker sends the launcher SIGTERM; it
  stops the unit, waits until systemd reports it inactive, then releases the lock. The worker keeps the daemon waiting
  with progress lines until then, so nothing else loads into memory the server still holds.
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

The `--reference` directory is upstream's reference runtime, as its `recipe/README.md` describes: `venv/` with
`sgl-omni` and the reference's dependencies installed, and an `sglang-omni/` checkout at the revision in
`drama-tensorfold/recipe/sources.json` (`ab095ef9e779e56af33b6ed2c863645c9903372d`). Upstream pins that revision but
not the venv's packages, so LocalRouter cannot pin them either; the qualify report records the reference's torch and
SGLang versions.

`fetch.sh` prints the weights' licence before downloading them and accepts nothing on anyone's behalf. `build.sh`
runs upstream's `scripts/build-engine.sh`, checks that the patched TensorFold HEAD is the pinned one, and records the
engine binary's sha256 in `ROOT/engine.json`. Then add the tool from `tools/speech/drama3/config.example.json` to the
daemon's config, with `gpu_lock` set to the lock every GPU job on the host already takes (only on a host without one,
create it with `touch` and have the other GPU jobs take it too).

| Option | Default | |
| --- | --- | --- |
| `port` | 18793 | the server's loopback port |
| `max_frames` | the engine's 1024 | frames one synthesis may make (`--max-frames`) |
| `long_form`, `long_form_chars`, `long_form_history` | off, 500, 1 | `--long-form` and its settings |
| `voices_dir` | none | an absolute directory of voice files (`--voices`) |
| `voices`, `default_voice` | `["default"]`, `default` | what `/v1/models` lists and the daemon checks; a `default_voice` other than `default` becomes `--default-voice` |
| `verify` | `size` | at load, each pinned checkpoint file's size, or its `sha256` (minutes), or `none` |
| `pilot`, `native_bin`, `checkpoint`, `runtime_dir` | under `weights` | where fetch.sh and build.sh put them |
| `ready_timeout_s` | 900 | the launcher's readiness deadline |

### Voices

Clone only a voice the model invented, your own, or one whose owner has consented on record: the model licence's
Section IV(b) forbids imitating an identifiable person without consent. Voice files are made offline with upstream's
tools (`drama_voice_clip.py` prepares a 10 to 30 s take, `drama_voice.py` encodes it under the GPU lock and records its
`source`, which the server requires), as its README describes; list their names in `voices` and set `voices_dir`.
Clips and voice files hold a person's voice: keep them out of source control.

## Pins

`tools/speech/drama3/pins.json` holds every hash. `fetch.sh` and `build.sh` refuse anything else.

| What | Pin |
| --- | --- |
| Upstream | plotarmordev/drama-tensorfold @ `1a53b8aea738a99d3630f949b6b2a1f9389f0795` (main, 2026-10-17, PRs #4 to #9); sha256 of `recipe/pilot.py`, `scripts/build-engine.sh`, `engine/UPSTREAM`, `recipe/checksums/checkpoint.sha256`, `recipe/sources.json`, `NOTICE`, `LICENSE`, `engine/patches/SHA256SUMS` |
| Engine | TensorFold @ `c7137bb31a7dad924fb541a7068422127b9a94b2` plus the 59 patches in `engine/patches/` (each by sha256; series sha256 `3fff8dd6…a910d247`), applied by `build-engine.sh`: HEAD `459771e7531431490b80d7a8149c9160e03c01ce` |
| Checkpoint | fishaudio/drama-3 @ `e0a9218aa48888270f06ef1ed9ea67622c0e17d2`: `codec.pth`, eight weight shards (41,474,674,072 bytes together, matching upstream's `sources.json` and `checkpoint.sha256`), tokenizer, config, chat template and licence, each by size and sha256 |
| Toolchain | Zig 0.17.0, CUDA 13 |

## Qualification

`qualify.sh ROOT CONFIG OUT --reference RUNTIME [--accept-native-numerics]` runs on the GB10, every step under a time limit, and writes
`OUT/status.json` and a manifest as Kitten's does:

- **assets:** the checkout, every pinned file and patch, the engine binary against `engine.json`, and the full sha256
  of every checkpoint file;
- **e2e:** the real worker on the real GPU: a cold load (the shared lock is held while loaded); a sentence whole (44.1
  kHz mono, `stop`), again warm (byte-identical), and streamed (the same bytes); a 300-character paragraph whole and
  streamed (equal, warm RTF recorded); each configured voice, and inline references from a voice file; an unknown
  voice refused with a 400; long form when it is on; then unload: the unit inactive, the lock free and `MemAvailable`
  back within 2 GiB;
- **accuracy:** upstream's own checks from `qualification/`, as its README runs them, each in its bounded GPU guard
  (`run-bounded-oracle.py`: the launcher's admission, 54 GiB cap, abort floor and the shared lock): the backbone
  reference check (the 40-layer prefill of its 182-token fixture, native chunks 1, 7 and 128 against the reference's
  extend: hidden states and semantic logits), the same native outputs rescored under upstream's registered worst-row
  bar (CPU), and the tower reference check (13 layers, teacher-forced, at 1, 7 and 128 rows). Their verdicts and
  metrics go into `OUT/accuracy/report.json` and the marker, pass or fail. Upstream reports that the backbone check
  fails some bars, so a failing verdict is the expected result today. Upstream's codec gate and its tower full-context
  replays are not reproducible from its repository (its `qualification/README.md` says so) and are not run.

Only when assets and e2e pass, the accuracy checks produced valid evidence, and `--accept-native-numerics` is given,
does it write the tool's `qualification` file: `{"qualified": true, "host", "gpu", "pins_sha256", "native_sha256",
"upstream_commit", "checkpoint_revision", "numerics", "upstream_accuracy", "report", "qualified_at"}`, where
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

All GPU behaviour: the build, a load, any audio, the timings upstream reports (first audio 0.14 s streamed, RTF about
0.63 to 0.68, 6.4 s to ready with warm caches), memory under the 54 GiB cap, stream-equals-whole, cold-equals-warm,
the silence guard and long form on the real model, and unloading on a real systemd unit. CI checks the daemon side and
the worker against the stand-in only.

## What upstream lacks that LocalRouter needs

- A test mode: there is no CPU or scripted engine to run the server without a GPU, so CI tests the worker against a
  stand-in of the HTTP contract, not the server.
- A way to let an outer scheduler admit the load: the launcher's admission (88 GiB free) and abort (34 GiB) are
  constants in `pilot.py` with no option to turn them off, hence LocalRouter's `--external-admission` patch (a
  candidate to offer upstream). It also reports no resident size, so the tool holds its 54 GiB cap until measured.
- A launcher that cleans up when killed: its unit outlives a SIGKILLed launcher, so the worker keeps the unit's name
  and uses a parent-death signal; a stop hook in the unit, or a unit name chosen by the caller, would close that gap
  upstream.
- A free port: the server takes a fixed `--port` and the launcher refuses one in use; the tool's `port` must be
  unique on the host.
- Control a speech API usually has: no `speed`, no `seed` (each synthesis is seeded from its prompt), no presets.
- Numerical qualification against the reference ([Numerics](#numerics)). Its reference runtime (the `sglang-omni` venv its accuracy checks need) is pinned by checkout revision only, not by package versions.

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
