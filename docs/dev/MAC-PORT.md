# macOS Apple Silicon port

This page consolidates the research of 2026-10-30 (base `fe24726`) for LocalRouter on macOS: what runs where, what is
pinned, what is decided and what is still unmeasured. Target hardware: M5 Ultra, 96 GiB unified memory. Nothing here
is a Mac measurement unless it says so (the measurements are in [Measured on the Mac](#measured-on-the-mac)); every
size is a decimal-byte figure from HF metadata and every memory figure marked "estimate" is an admission starting
point, not a peak.

## Purpose and scope

Mac targets, decided by JJ:

- **Chat:** Qwen3.8 27B.
- **Speech:** Drama 3 through plotarmordev's drama-tensorfold Metal NVFP4 path. **Not** mlx-audio.
- **Decisions:** KnowLine-4B Gen5.5.
- **Transcription:** Whisper large-v3-turbo with the Silero VAD gate.
- **Images:** Qwen-Image 2.1 Turbo, and Qwen-Image 2.1 edit.

Out of scope and **absent on macOS** (not ported, listed as disabled): Kandinsky-6, Kitten TTS 2, MiniMax H3.

Spark/Linux behaviour does not change. Mac paths are added beside the Linux ones: comptime OS switches in Zig and
separate pins files (`pins-macos.json`), never a replacement.

## Installing on a Mac

The `mac-apple-silicon` profile ([deploy/profiles/mac-apple-silicon.json](../../deploy/profiles/mac-apple-silicon.json))
is a **preview**. Every model installs, qualifies and serves through the daemon on the reference M5 Ultra (96 GiB,
macOS 27). It has not run on other Macs yet, and there is no service. You need an Apple Silicon Mac with at least
64 GiB, Xcode with its Metal Toolchain, Zig 0.17.0 and python3 3.11 or newer. The plan checks each of these and says
how to fix what is missing.

```bash
git clone https://github.com/jayleaton/localrouter && cd localrouter
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer     # if xcode-select points at the Command Line Tools
tools/install/bootstrap.sh --profile mac-apple-silicon --dry-run    # the plan: steps, downloads, disk, memory, licences
tools/install/bootstrap.sh --profile mac-apple-silicon --accept     # after reading the licences the plan lists
```

The run builds LocalRouter and downloads every model, about 60 GiB, each file checked against its pin. It then benches
chat and runs Drama 3's Metal qualification, which writes the marker the daemon needs. `--models` picks a subset.

The run then ends: no launchd service is written yet. To serve, build one daemon config whose `tools` list has each
model's entry. `$J` is the `--prefix`, `~/jarvis` by default.

| Model | Its tool entry |
| --- | --- |
| Qwen3.8 27B chat | `$J/chat/config.json`, written by the `chat/setup` step |
| Drama 3 | `$J/qualify-drama3/config.json`, written by the `tts/config` step |
| Whisper | the entry in [tools/speech/whisper/macos/README.md](../../tools/speech/whisper/macos/README.md), with your paths |
| KnowLine-4B | [tools/decide/knowline/macos/config.example.json](../../tools/decide/knowline/macos/config.example.json) |
| Qwen-Image 2.1 Turbo | [tools/image/qwen-mlx/config.example.json](../../tools/image/qwen-mlx/config.example.json) |

Then run `zig-out/bin/localrouter serve --config FILE`. `tools/install/doctor.sh` still checks for the systemd
service, so its service checks do not apply on a Mac.

## Per-model engines

The engines below are the ones the per-model ports use; their measured numbers are in the next section. "Seam" is either the `openai_server` wrapper (the daemon spawns a child HTTP server and
proxies to it) or an external `cmd` worker (newline JSON, see [MODEL-DROPIN.md](MODEL-DROPIN.md)). `openai_server`
declares chat and decision capabilities only, so it is not a media bridge.

| Model | Mac engine / runtime | Seam | Pinned source | Weights | Owner |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8 27B chat | TensorFold native Zig/Metal, `qwen27` family, DFlash2 drafter; macos-arm64 release archive plus its `lib/` (grammar dylib) | `openai_server` | TensorFold [v1.1.4 `eab8582`](https://github.com/ashhart/TensorFold/tree/eab8582819688325e371907e4706e1c602b8a273) release archive; [`TensorFold/Qwen3.8-27B-MLX-4bit@22d8d53`](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit/tree/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c); drafter [`z-lab/Qwen3.8-27B-DFlash2@50307d4`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) | MLX affine 4-bit/g64, 16,054,541,349 B (14.95 GiB); drafter bf16 3,848,817,896 B (packed to 4-bit by the loader). Apache-2.0 | chat |
| Drama 3 speech | drama-tensorfold Mac stack: TensorFold base plus `engine/nvfp4/macos/patches/` (Metal NVFP4 + CFG), Metal DAC decoder and encoder | `cmd` worker around the native HTTP server (native server takes startup voice catalogs, not per-request references) | [drama-tensorfold `db6ba7d`](https://github.com/plotarmordev/drama-tensorfold/tree/db6ba7da314b24b42b02978ba4401f54f6738311), base TensorFold [`3d219d2`](https://github.com/ashhart/TensorFold/tree/3d219d2156d4c25dcfb74370c8703cd89e58aa49); weights [`plotarmordev/drama-3-NVFP4-Pruned@494db08`](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned/tree/494db08481db952f834b2241bce63fb3dfa72554) | NVFP4 (E2M1 experts, E4M3 block-16 scales), 22,294,649,076 B (20.76 GiB) weights+codec; repo 22,323,516,597 B. FACL v1.0 | drama3 |
| KnowLine-4B decisions | `mlx-lm` Qwen3.5 text model plus the author's scoring front end (`knowline_server.py`, `knowline_layout.py`); no TensorFold engine fits | server wrapper replacing `tools/decide/knowline/serve.py` (it hardcodes CUDA); `/v1/systemone` stays proxied | [`PelaAI/KnowLine-4B-Gen5.5@779944d`](https://huggingface.co/PelaAI/KnowLine-4B-Gen5.5/tree/779944d2cc8ffa19cef386aee83c83bcb967ade2); [`mlx-lm@736ab0d`](https://github.com/ml-explore/mlx-lm/tree/736ab0d8c1d0edc79c3d798044fe6904e2fa23be) | bf16 baseline 9,319,828,224 B (8.68 GiB). Q5 option [`Liamariel/...-mlx-q5@053081f`](https://huggingface.co/Liamariel/knowline-4b-gen5.5-mlx-q5/tree/053081f764ffeddf4b1278d4183db95b72180520), 2,892,837,529 B, only after a decision-quality comparison. Apache-2.0 | knowline |
| Whisper large-v3-turbo STT | `mlx-whisper` persistent worker; WhisperKit/Core ML is the alternate. No TensorFold Whisper | `cmd` worker | [`mlx-community/whisper-large-v3-turbo@a4aaeec`](https://huggingface.co/mlx-community/whisper-large-v3-turbo/tree/a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb); `mlx-whisper` 0.4.3, pinned in `tools/speech/whisper/macos/uv.lock` | FP16, 1,613,977,612 B (1.50 GiB). Optional q4 [`@660c343`](https://huggingface.co/mlx-community/whisper-large-v3-turbo-q4/tree/660c343bbf4e52ac257f0b7d952e5388e6f93bef) 463,664,664 B needs its own qualification. MIT | whisper |
| Silero VAD | existing TorchScript JIT on CPU, beside the Whisper worker | in the Whisper `cmd` worker | [`snakers4/silero-vad@5cd7945`](https://github.com/snakers4/silero-vad/tree/5cd7945676eb32225748052e2e6a0580e4686a08) (v6.2.3), `silero_vad.jit` SHA-256 `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720` | 2,272,526 B. MIT | whisper |
| Qwen-Image 2.1 Turbo | MLX-Serve >= 26.10.2 (alternative: mflux) | `cmd` worker (load/generate/exit, progress, cancel, output files) | [`ddalcu/Qwen-Image-2.1-Turbo-MLX-Serve-8bit@214ca34`](https://huggingface.co/ddalcu/Qwen-Image-2.1-Turbo-MLX-Serve-8bit/tree/214ca346520a59597b89bd77408afa49d526bde5); runtime [`ddalcu/mlx-serve@1fa96c6`](https://github.com/ddalcu/mlx-serve/tree/1fa96c6d212866f7c2d57f5d7fc80d28cfa08d76); upstream [`Qwen/Qwen-Image-2.1-Turbo@fc13ba1`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/fc13ba1986fcb905854dc8de2f52eee395f49b5b) | q8/g64 pack, 17.73 GB repo, vision tower included. Qwen Research License (non-commercial) | qwen-image |
| Qwen-Image 2.1 edit | mflux `QwenImage21Edit` (`mflux-generate-qwen-2.1-edit`) | `cmd` worker (same as Turbo) | [`Qwen/Qwen-Image-2.1@d26bb61`](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3) via [`mflux@5171ae1`](https://github.com/mflux-community/mflux/tree/5171ae15e84f81046f9685fd77bf0bafa3187b61) | 33.12 GB BF16 source, quantised to q8 at load/export; edit VAE stays FP32 | qwen-image |

## Measured on the Mac

From the merged per-model ports and the integration run on the reference M5 Ultra (96 GiB). Memory is the process
footprint as macOS reports it; GB figures are decimal as reported, GiB binary.

| Model | Memory | Speed |
| --- | --- | --- |
| Qwen3.8 27B chat | 26 GiB peak; 26.75 GiB booked (28,723,101,696 B) with the default 4 GiB cache | ~196 tok/s with DFlash2 |
| Drama 3 | 28.75 GiB booked (30,870,077,440 B) | 0.52 s to first audio, RTF 0.77; switching voice adds 4 s |
| KnowLine-4B | 10 GiB | 137 ms a decision |
| Whisper large-v3-turbo (`mlx-whisper` 0.4.3) | 3.2 GB resident, 4.35 GB peak; 3.25 GiB booked + 1.25 GiB working | |
| Qwen-Image 2.1 Turbo (q4) | 9.5 GiB resident + 6.5 GiB working + 2 GiB per reference image | 8.5 s a 1024² image |

Start times. "Cache warm" means the weights are still in the page cache from a recent read; "cold" means after a purge.

| Model | Load, cache warm | Load, cache cold | First request |
| --- | --- | --- | --- |
| Qwen3.8 27B chat | 4.9 s | 5.1 s | 0.1 s |
| Drama 3 | 12.5 s | 13.0 s | first audio 4.3-4.8 s (it builds the voice prefix) |
| Whisper large-v3-turbo | 2.3 s | 2.4 s | |
| KnowLine-4B | 2.1 s | 2.3 s | |
| Qwen-Image 2.1 Turbo (q4) | 2.0 s | 2.3 s | 8.5 s a 1024² image |

From daemon start to the warm set loaded takes 22 s. After that, a fresh call hears first audio in 0.65 s.

**Memory policy (JJ).** Chat, Drama 3 and Whisper stay warm (`keep_loaded`). KnowLine and Qwen-Image load on demand,
one at a time. The GPU's wired ceiling is about 77.8 GiB (79,626 MiB).

**How the scheduler counts it** (`runJob` and `fits` in `src/sched/scheduler.zig`):

- A loaded model holds its resident booking for as long as it is loaded.
- Working bytes are booked per job. They count only while that model runs a request, and are released when the request
  ends. Whisper's 1.25 GiB therefore counts only while it transcribes.
- A request to a cold model books its resident and its working bytes. An image edit adds 2 GiB for each reference
  image.
- One exception: when a `spare_room_only` tool loads cold, it also leaves room for the working bytes of every loaded
  kept tool. KnowLine's example config is `spare_room_only`, so its load also books Whisper's 1.25 GiB.

By the bookings, the idle warm set holds 58.75 GiB: chat 26.75, Drama 3 28.75, Whisper 3.25.

| On demand, beside the idle warm set | Booked | Total |
| --- | --- | --- |
| KnowLine | 10 + 1.25 Whisper headroom | 70.0 GiB: fits |
| Text to image | 9.5 + 6.5 = 16 | 74.75 GiB: fits |
| Image edit, 1 reference image | 18 | 76.75 GiB: fits |
| Image edit, 2 reference images | 20 | 78.75 GiB: over |

A transcription running at the same moment adds 1.25 GiB, and then a 1-reference edit does not fit. KnowLine and an
image job together never fit. `bootstrap.sh --dry-run` prints 60.0 GiB for the warm models, because it counts
Whisper's working bytes as always booked. That is more cautious than the scheduler.

Bookings are only the plan. To admit a request, the scheduler checks live memory: `memory.available()` less
`reserve_bytes` must cover the request, plus whatever earlier admissions have not allocated yet. `memory.available()`
is the GPU ceiling less what is wired now, or the VM's available memory if that is smaller. When `budget_bytes` is set,
the bookings must also stay under it. Loaded models count in the live check by what they actually wire, not by their
bookings. `reserve_bytes` defaults to 8 GiB, so with the default a 1-reference edit needs 26 GiB available, which means
the warm set may wire at most about 51.8 GiB. With a smaller reserve, the edit fits beside the voice set as long as
what the warm set wires still leaves the reserve plus 18 GiB.

**Eviction, as the integration run saw it.** An image request evicts, in order, KnowLine, then Whisper, then Drama 3.
Chat is never evicted. Speech is down for about 25 s an image: the image job, then Whisper and Drama 3 loading again.

Unverified or open, per the research (the table above supersedes its memory and speed estimates):

- **Chat pin.** The reviewed Mac baseline is TensorFold v1.1.4. The chat archive deployed on the Spark is v1.1.1
  (`b10231b`), rolled back for a GB10 cache-latency regression. Whether Metal shows it is unknown. v1.1.9 exists
  (`465e48b`); it is not qualified. Do not carry the GB10 qualification marker or booking across.
- **Memory and speed** are measured now (above). The research's estimates (chat 32 GiB, KnowLine 12-16 GiB, Whisper
  4-6 GiB, Qwen-Image 19.3 GB publisher peak) and upstream chat speeds are superseded. Whisper's speed and the
  Qwen-Image edit path are not measured yet.
- **Qwen-Image edit** at five references: latency, peak and quality are unverified; the Turbo pack's vision tower is
  card-stated. Text-only exports cannot serve `image_edit`. Verify keys before advertising it.
- **Licences travel with the weights.** Drama: FACL v1.0 (no Hosted Service without a separate licence; $500,000
  threshold; "Built with Fish Audio"). Qwen-Image: Research License, and ignore community cards that claim Apache-2.0.

## Decisions

- **Engine correctness.** The Mac backend is a deliberate numerical change. Each model documents an explicit **Mac
  reference baseline** (backend, checkpoint, precision) and its own acceptance gate. No bit-parity claim with the
  Spark; Spark/CUDA receipts do not qualify Metal. TensorFold and mflux agreeing (PSNR 43.35 dB on Qwen-Image) is not
  CUDA output identity.
- **Out-of-process first.** Chat uses the TensorFold Metal server behind `openai_server`; everything else is a `cmd`
  worker or external server. In-process Metal engines come later, if at all (contract below).
- **No CUDA translation.** CUDA pointers, fatbins, NVFP4 `mma.sync` assembly, memcap and the vendor loader do not map to
  Metal. Mac builds exclude them or provide a separate Mac scheme.
- **One unified-memory budget.** Count each backing once; no "96 GiB VRAM plus 96 GiB RAM". Admission needs measured
  resident/working bookings per model on this class of machine ([measured](#measured-on-the-mac)). Servers' allocations
  are invisible to the daemon, so trust only measured receipts and observed process footprint.
- **Shared GPU, co-tenancy.** The scheduler still runs one job at a time and unloads idle tools by priority. During
  development, every command that loads weights goes through the machine-wide GPU lock. Upstream reports of idle GPU
  residency after a request (#1647, #1572) are unverified here; confirm unload/kill releases every server buffer.
- **Drama on Mac is NVFP4 via drama-tensorfold.** Its Mac pin/patch stack is separate from the CUDA `drama-nvfp4`
  fork; do not build the CUDA fork on Mac. The six voices need explicit conversion to the native catalog schema, not a
  file copy. Cloning needs offline catalog publication or a new runtime reference contract.
- **KnowLine keeps the author's scorer.** bf16 first. Preserve choice/noul/score, 64 questions / 255 labels,
  read-twice up to 3,000 tokens, last-position logits for every label and the folded calibration. Top-k chat logprobs
  cannot do this.
- **STT.** Set `condition_on_previous_text=False` explicitly (MLX defaults to True), keep greedy decoding, the Silero
  gate (16 kHz, threshold 0.5, min speech 250 ms, padding 30 ms) and the segment-overlap drop.
- **Images.** Keep the fixed unshifted eight-node Turbo schedule. Apply the five-reference limit and mask convention;
  set mflux steps explicitly (its default of 40 differs from LocalRouter's 25).
- **Absent models.** Kandinsky-6, Kitten TTS 2 and MiniMax H3 are not registered on macOS and the Mac profile omits them.
- **Toolchain.** Zig 0.17.0 only (Homebrew ships it). Metal shader builds need full Xcode: the licence must be accepted
  (`sudo xcodebuild -license accept`, `sudo xcodebuild -runFirstLaunch`) and the Metal Toolchain component installed
  (`xcodebuild -downloadComponent MetalToolchain`). The reference Mac has Xcode 27.2 and Metal Toolchain 27B47.
  `xcode-select` there points at the Command Line Tools, so builds set
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`. Prebuilt release archives need none of this.

## Foundation design

What makes the daemon build, test and run on macOS arm64 (`zig build`, `zig build test`). Linux behaviour is unchanged.

**Build.** `build.zig` sets `build_config.cuda_engines` to whether the target is Linux. On macOS the CUDA modules
(`cuda`, `nvfp4_kernels`, `qwen_image`, `minimax_h3`, `kandinsky6`, the vendor loader) are not imported or tested.
`engine.entries` then holds only `testpattern` and `openai_server`. A tool whose `engine` is a CUDA one finds no
engine: its worker exits with "no engine named ...", and the tool fails to load. The `check` subcommands for those
engines are gone. `check gpu` prints the unified-memory budget instead of probing CUDA. The binary links Metal and
libobjc, but only to read the GPU's working-set ceiling.

**Workers die with the daemon.** Linux arms `PR_SET_PDEATHSIG`. macOS has no such call, so `src/engine/orphan.zig`
provides two replacements:

| Process | Linux | macOS |
| --- | --- | --- |
| compiled-in worker (`localrouter worker`) | `PR_SET_PDEATHSIG` SIGKILL | `orphan.arm(.KILL)`: a thread blocks on kqueue `EVFILT_PROC`/`NOTE_EXIT` for the parent, then signals itself |
| `openai_server` worker and its model server | `PR_SET_PDEATHSIG` SIGTERM, whose handler kills the server | the same `orphan.arm(.TERM)`, so the same handler runs |
| `cmd` worker (any program) | none: it sees stdin EOF | spawned as `localrouter guard <daemon pid> -- CMD...` (see below) |

`guard` forks a watchdog twice, so launchd reaps it and it is never a zombie of CMD's. It then `exec`s CMD in place,
so the daemon keeps the same pid, pipes, process group and signals (cancel's SIGUSR1, the yielder's SIGSTOP). The
watchdog redirects stdin and stdout to `/dev/null`, so it holds none of the worker's pipes. It watches the daemon and
the worker with kqueue. If the daemon exits first, it SIGKILLs the worker's process group: the worker, any server the
worker started, and the watchdog itself. If the worker exits first, the watchdog just exits.
`tests/orphans.zig` SIGKILLs a daemon that has a compiled-in worker, an `openai_server` worker with its server, and a
`cmd` worker loaded. It then checks that every process that was under the daemon is gone within 10 s.

**One unified-memory budget** (`src/sched/memory.zig`). The scheduler admits work against `memory.available()`, as
on the GB10. On macOS that value is computed from three sources:

- `host_statistics64(HOST_VM_INFO64)`: the VM's available memory is `hw.memsize` minus wired, compressor, and
  non-purgeable anonymous pages. File cache counts as free, as with Linux's MemAvailable.
- The GPU's headroom: its wired ceiling minus the memory wired now (Metal buffers in use are wired). The ceiling is
  `iogpu.wired_limit_mb` when an administrator has set it, else Metal's `recommendedMaxWorkingSetSize`, else 3/4 of
  memory.
- Pressure from `kern.memorystatus_vm_pressure_level`. Critical pressure admits nothing.

The budget is the smaller of the first two, or 0 under critical pressure. Memory is counted once: there is no
separate VRAM figure. `budget_bytes` and `reserve_bytes` work as they do on Linux. On this M5 Ultra (96 GiB) the
ceiling reads 79,626 MiB.

**Smaller portability fixes.**

- Loopback sockets set close-on-exec and non-blocking with `fcntl`, because macOS's `socket()` rejects
  `SOCK_CLOEXEC`/`SOCK_NONBLOCK`. They also set `SO_NOSIGPIPE`.
- The yielder's child and stopped-state probes use libproc instead of `/proc`.
- The tests' process probes are in `tests/proc.zig` (libproc and `KERN_PROCARGS2` on macOS).

**Stopping.** On SIGTERM or SIGINT, Linux wakes the blocked `accept` with `shutdown()` on the listening socket.
macOS returns ENOTCONN for that call and `accept` stays blocked, so there the daemon connects to its own address
once (`serve.zig` `watchStop`). It then exits 0 and its workers follow. `tests/orphans.zig` covers both this and
SIGKILL.

**Reading test output.** `zig build test-integration` may end with `failed command: ./.zig-cache/o/.../test ...` and
still pass. Zig 0.17's build runner prints that line, left over from the run, whenever a passing test step's child
processes wrote to the inherited stderr (the daemons' `info:` logs do). It is the same on Linux. The exit status and
`Build Summary` (`N/N tests passed`) are what count.

**Cancel reaches the worker.** A `cmd` worker that takes soft cancel gets SIGUSR1. The daemon used to send it at
once, so a cancel that came between "job running" and the `generate` line reaching the worker was lost (the worker
was not busy yet and ignored it). Now the job carries its cancel flag: a cancel before the write ends the job without
sending it, and after the write the daemon signals, then repeats every 200 ms until the reply arrives. A worker must
therefore ignore a SIGUSR1 that comes while no `generate` is running ([wire.zig](../../src/tool/wire.zig)).
`src/tool/process.zig` tests both windows. This is not Mac-specific; the Whisper port found it.

**Installer** (`tools/install`). It runs on macOS's bash 3.2 and BSD tools, and on a Spark it runs exactly the
commands it ran before. `lib.sh` holds the portable helpers:

| Helper | Linux | macOS fallback |
| --- | --- | --- |
| `readlines` | `mapfile` | read loop |
| `abspath` | `realpath -m` | by hand |
| `sha256` | `sha256sum` | `shasum -a 256` |
| `hw_mem`, `hw_avail` | `/proc/meminfo` | `sysctl` and `vm_stat` (the daemon's formula) |
| `hw_disk_free` | `df -B1` | `df -k` |
| `hw_os` | `/etc/os-release` | `sw_vers` |
| `hw_gpu` | `nvidia-smi` | the SoC name |

Other differences:

- On bash older than 4.4, nounset is relaxed, because `"${empty[@]}"` trips it there.
- `--peer-store` needs bash 4.
- Hard-link reuse (`cp -l`) falls back to downloading, because BSD `cp` has no `-l`.
- A `skeleton` profile with no models plans its host steps alone.
- A profile without `deploy` ends after its steps. The Mac service, launchd rather than systemd, is not written
  yet, and `doctor.sh`'s service checks are still the systemd ones.

`bootstrap.sh --profile mac-apple-silicon --dry-run` runs on this Mac, and CI dry-runs every profile as before. On
Linux, the `jarvis-gb10` plan is byte-identical before and after this change.

**CI.** No macOS job was added, because macOS runners are billed. The PR proposes one.

The Mac deploy profile is `deploy/profiles/mac-apple-silicon.json`, status `preview` ([Installing on a
Mac](#installing-on-a-mac)). It lists the host prerequisites (Zig 0.17.0, the Xcode Metal toolchain, python3) and the
five models. Its first host step
installs `uv` (0.13.4, `uv-aarch64-apple-darwin`, pinned by sha256 in `host.tools`) into `$J/bin`, because the Python
workers' venv steps (Whisper's `uv sync`) need it and macOS does not ship it.

## Proposed Metal engine contract

For a future in-process engine. It is the counterpart of the CUDA contract in
[TENSORFOLD-RUNTIME.md](TENSORFOLD-RUNTIME.md#contract-for-engines), **not an existing contract**. TensorFold v1.1.4
already ships a native Metal module with its own Objective-C bindings and no MLX
([`metal.zig`](https://github.com/ashhart/TensorFold/blob/eab8582819688325e371907e4706e1c602b8a273/zig/src/metal/metal.zig));
LocalRouter builds only the CUDA runtime and tokenizer, so a pin bump alone selects nothing.

1. **Ownership.** One Metal device and agreed command queue(s) per loaded engine; components borrow them. Explicit
   autorelease-pool boundaries, worker threads included
   ([`device.zig`](https://github.com/ashhart/TensorFold/blob/eab8582819688325e371907e4706e1c602b8a273/zig/src/metal/device.zig),
   [`command.zig`](https://github.com/ashhart/TensorFold/blob/eab8582819688325e371907e4706e1c602b8a273/zig/src/metal/command.zig)).
2. **Tensor handles.** `(MTLBuffer owner, byte offset, dtype, shape, strides)` with explicit borrowed views. Keep the
   owner and no-copy mmap pages alive until the GPU completes; a bare `u64` address is not portable.
3. **Ordering and staging.** Specify CPU write visibility, queue order, blit completion, event waits/signals and
   multi-queue barriers. Shared storage removes copies, not synchronisation. Unretained command buffers put lifetime on
   the caller.
4. **Kernels and replay.** Publish shader symbols, argument layouts, supported hardware, precision and threadgroup
   limits. ICB replay is recorded dispatches with declared resources, not inferred stream capture
   ([`icb.zig`](https://github.com/ashhart/TensorFold/blob/eab8582819688325e371907e4706e1c602b8a273/zig/src/metal/icb.zig)).
   Match references at operator seams and on full output.
5. **Memory and admission.** Reserve before allocating; count each unified-memory backing once, including mmap,
   private copies, cache/drafter state and runtime overhead; report resident and working peaks plus process footprint;
   honour system pressure and the recommended working set
   ([`memory_policy.zig`](https://github.com/ashhart/TensorFold/blob/eab8582819688325e371907e4706e1c602b8a273/zig/src/platform/memory_policy.zig),
   [`cache_fit.zig`](https://github.com/ashhart/TensorFold/blob/eab8582819688325e371907e4706e1c602b8a273/zig/src/native/cache_fit.zig)).
   TensorFold's 70% RAM policy is about 67.2 GiB on this machine before its margin; that is not a fit limit.
6. **Failure and unload.** Stop submissions, prove bounded completion on every relevant queue, drop residency
   references before freeing, release in dependency order. A faulted or timed-out operation never justifies freeing
   live resources; worker exit is the final boundary.
7. **Conformance.** Lifecycle, accounting, unavailable-capability, cancellation, server/worker death and exact-output
   gates on the real Mac backend and checkpoint, per [CONTRIBUTING.md](../../CONTRIBUTING.md).

Native Metal image/video building blocks (`tf-qwen-image`, `tf-h3-dit`) are test tools for a denoiser only: no prompt
encoder, decoder or server registration. They are not a ready engine.

## Sources

Pinned, immutable revisions (full hashes are in the links).

- TensorFold v1.1.4: [release](https://github.com/ashhart/TensorFold/releases/tag/v1.1.4), commit
  [`eab8582`](https://github.com/ashhart/TensorFold/tree/eab8582819688325e371907e4706e1c602b8a273); Qwen recipe
  [qwen3.8-27b](https://github.com/ashhart/TensorFold/blob/eab8582819688325e371907e4706e1c602b8a273/docs/recipes/qwen3.8-27b.md).
  Latest observed v1.1.9: [release](https://github.com/ashhart/TensorFold/releases/tag/v1.1.9) (`465e48b`).
- LocalRouter base [`fe24726`](https://github.com/jayleaton/localrouter/tree/fe24726cae6d0072e2970fc3b0027e353e129ad6);
  [chat pins](https://github.com/jayleaton/localrouter/blob/fe24726cae6d0072e2970fc3b0027e353e129ad6/tools/chat/qwen38-27b/pins.json).
- Chat: [Qwen3.8-27B-MLX-4bit](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit/tree/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c),
  [DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4).
- Drama: [drama-tensorfold `db6ba7d`](https://github.com/plotarmordev/drama-tensorfold/tree/db6ba7da314b24b42b02978ba4401f54f6738311),
  [Mac recipe](https://github.com/plotarmordev/drama-tensorfold/blob/db6ba7da314b24b42b02978ba4401f54f6738311/recipe/README-macos.md),
  [M5 Ultra receipt](https://github.com/plotarmordev/drama-tensorfold/blob/db6ba7da314b24b42b02978ba4401f54f6738311/results/macos-nvfp4.json),
  [NVFP4 weights `494db08`](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned/tree/494db08481db952f834b2241bce63fb3dfa72554),
  [fishaudio/drama-3 `e0a9218`](https://huggingface.co/fishaudio/drama-3/tree/e0a9218aa48888270f06ef1ed9ea67622c0e17d2)
  (reference, FACL licence).
- KnowLine: [original](https://huggingface.co/PelaAI/KnowLine-4B-Gen5.5/tree/779944d2cc8ffa19cef386aee83c83bcb967ade2),
  [Q5](https://huggingface.co/Liamariel/knowline-4b-gen5.5-mlx-q5/tree/053081f764ffeddf4b1278d4183db95b72180520),
  [mlx-lm](https://github.com/ml-explore/mlx-lm/tree/736ab0d8c1d0edc79c3d798044fe6904e2fa23be).
- STT: [Whisper MLX](https://huggingface.co/mlx-community/whisper-large-v3-turbo/tree/a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb),
  [Silero](https://github.com/snakers4/silero-vad/tree/5cd7945676eb32225748052e2e6a0580e4686a08).
- Images: [Turbo MLX-Serve q8](https://huggingface.co/ddalcu/Qwen-Image-2.1-Turbo-MLX-Serve-8bit/tree/214ca346520a59597b89bd77408afa49d526bde5),
  [mlx-serve](https://github.com/ddalcu/mlx-serve/tree/1fa96c6d212866f7c2d57f5d7fc80d28cfa08d76),
  [mflux](https://github.com/mflux-community/mflux/tree/5171ae15e84f81046f9685fd77bf0bafa3187b61),
  [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3),
  [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/fc13ba1986fcb905854dc8de2f52eee395f49b5b).
