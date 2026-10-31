# What LocalRouter carries against upstream engines

This file lists every engine-level change LocalRouter carries against the engines it runs, with each one's upstream
pull request. It counts patch files, scripted edits of upstream files, flags or workarounds that exist only because of an
upstream bug or a missing option, and vendored upstream code. Each entry gives where it lives here, what it
changes and why, whether upstream has it, and what happens next. [CONTRIBUTING.md](../../CONTRIBUTING.md#upstream-engine-changes)
requires an entry and an upstream pull request for every new one.

Audited 2026-10-31 against the latest upstream default branches:
- [ashhart/TensorFold](https://github.com/ashhart/TensorFold) `main` at `acea3b52` (after v1.2.0);
- [plotarmordev/drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold) `main` at `4d9cfb2`;
- [plotarmordev/TensorFold](https://github.com/plotarmordev/TensorFold) `drama-nvfp4` at `b1f4b9b4`.

## Status

| # | Change | Upstream | Upstream PR | Our side |
|---|---|---|---|---|
| 1 | Drama 3 voice prefix cache, CUDA full model | ashhart/TensorFold | [#1745](https://github.com/ashhart/TensorFold/pull/1745) | drop `patches/engine/` at the first full-model pin that includes it |
| 2 | Drama 3 voice prefix cache, CUDA NVFP4 | plotarmordev/TensorFold `drama-nvfp4` | already proposed: [plotarmordev/TensorFold#4](https://github.com/plotarmordev/TensorFold/pull/4) (not ours) | drop `patches/engine-nvfp4/` once #4 merges and drama-tensorfold repins NVFP4 |
| 3 | Launchers: `--external-admission` (+ Linux `--reserve-gib`) | plotarmordev/drama-tensorfold | [#35](https://github.com/plotarmordev/drama-tensorfold/pull/35) | replace both launcher patches with flags |
| 4 | macOS launcher: `--parent-pid` | plotarmordev/drama-tensorfold | [#36](https://github.com/plotarmordev/drama-tensorfold/pull/36) | as 3 |
| 5 | Launchers start under `python3 -I` | plotarmordev/drama-tensorfold | [#37](https://github.com/plotarmordev/drama-tensorfold/pull/37) | as 3 |
| 6 | Streamed `reasoning_content` keeps a trailing newline | ashhart/TensorFold | [#1743](https://github.com/ashhart/TensorFold/pull/1743) | drop the qualification exception once the chat pin includes it |
| 7 | `graph.endCapture` leaves the capture scope open on failure | ashhart/TensorFold | [#1744](https://github.com/ashhart/TensorFold/pull/1744) | drop the two `captureEnd` calls once the runtime pin includes it |
| 8 | Tokenizer has no `Lowercase` normalizer (CLIP) | ashhart/TensorFold | not opened: no upstream model needs it | keep `clip_tokenizer.zig`'s own pass |
| 9 | Chat pinned to 1.1.1 on GB10 (1.1.4 resumes only at 4,096-row chunks) | ashhart/TensorFold | none: a pin choice, no code carried | re-measure on each new release |
| 10 | cuBLASLt with a recorded algorithm; driver identity check | ashhart/TensorFold | not upstreamable: LocalRouter's attestation policy | keep |
| 11 | NVFP4 kernels copied from TensorFold v0.6.1 | ashhart/TensorFold v0.6.x | not upstreamable: unmodified copies of a frozen line | keep |

Not ours, and not listed above:
- The series pinned by hash in `tools/speech/drama3/pins.json` (59 patches, `engine/patches/` at drama-tensorfold
  `1a53b8ae`) and `pins-macos.json` (5 patches, `engine/nvfp4/macos/patches/` at `db6ba7da`) are drama-tensorfold's own.
  Every pinned sha256 equals that commit's file, checked on 2026-10-31.
- `apps/mobile`'s react-native patch.

## Entries

### 1. Drama 3 voice prefix cache, CUDA full model

- **Here:** `tools/speech/drama3/patches/engine/0001-drama-a-voice-s-reference-rows-are-prefilled-once-an.patch`,
  applied by `build.sh` after upstream's series. Added in `cd45432` (#25).
- **Upstream file:** TensorFold `zig/src/native/drama_host.zig`, `zig/src/families/drama/{backbone,tower}.zig`. Our
  patch targets drama-tensorfold's old engine base (TensorFold `c7137bb3` + 59 patches); the full model now pins
  TensorFold v1.1.9 with no patches.
- **What and why:** a voice's reference rows (280–431 frames plus their transcript) were prefilled again on every
  request. The patch snapshots backbone and tower state after them and restores it for a later prompt with the same
  rows. On DGX-01, streamed first audio p50 went from 0.67–1.16 s to 0.127–0.134 s, with replies byte-identical with
  the cache on and off ([DRAMA3.md](DRAMA3.md)).
- **Upstream today:** main has the Metal counterpart (`a06a4dff`, from #1237, "CUDA stays uncached") and no CUDA
  cache.
- **General?** Yes. The upstream port cuts on a prefill chunk boundary, as Metal does, so a cached reply is the same
  bytes as an uncached one. Ours cuts at the system message, which changes chunk cuts for referenced prompts.
- **PR:** [#1745](https://github.com/ashhart/TensorFold/pull/1745). Host tests run over a fake lane (cached cuts and state equal uncached). The Linux CUDA host build and the macOS Metal build both compile. CUDA wasn't run here: its PCM check, with the cache on and off on a GB10, is still to do.

### 2. Drama 3 voice prefix cache, CUDA NVFP4

- **Here:** `tools/speech/drama3/patches/engine-nvfp4/0001-…patch`, the same cache ported to plotarmordev's `drama-nvfp4`
  branch (`b1f4b9b4`). Added in `67eb7ed` (#77).
- **Upstream today:** an equivalent pull request by another contributor,
  [plotarmordev/TensorFold#4](https://github.com/plotarmordev/TensorFold/pull/4), is open. It is chunk-aligned and
  byte-identical, with GB10 receipts.
- **PR:** none of ours. That would duplicate #4.

### 3. Launchers: external admission

- **Here:** `tools/speech/drama3/patches/pilot-external-admission.patch` (Linux `recipe/pilot.py`, `68ca3bd`, #13) and
  part of `pilot-macos-external-admission.patch` (`recipe/pilot_macos.py`, `4bf4b70`, #136). `fetch.sh` and
  `setup-macos.sh` apply them with `git apply` to a copy under `ROOT/launcher/`.
- **What and why:** the launchers enforce their own admission and abort thresholds: 88 GiB `MemAvailable` on Linux,
  and weights + 30 GiB free+inactive on macOS, each with a 34 GiB abort. On Linux the engine also has a fixed 34 GiB
  `TENSORFOLD_MEMORY_RESERVE_GIB`. LocalRouter admits each load from its own budget, so these refused loads it had
  admitted.
- **General?** Yes: a flag for any supervisor that budgets memory itself. Upstream has no equivalent (macOS's
  `--min-available-gib` cannot go below 34 GiB).
- **PR:** [drama-tensorfold#35](https://github.com/plotarmordev/drama-tensorfold/pull/35) adds `--external-admission`
  to both launchers and `--reserve-gib` to the Linux one. Tested:
  - unit tests;
  - a macOS smoke test;
  - the real NVFP4 Metal engine on this Mac Studio: at 47.7 GiB free+inactive, main refused with "Need 50.76 GiB"; the
    branch was ready in 5.6 s and served a request.

  The Linux launcher was not run end to end (no systemd/CUDA here). Our patch has run on the GB10.

### 4. macOS launcher: `--parent-pid`

- **Here:** part of `pilot-macos-external-admission.patch` (`4bf4b70`, #136).
- **Why:** macOS has no parent-death signal. When the worker is killed, the launcher and server stay up, holding the
  lock and the memory.
- **PR:** [drama-tensorfold#36](https://github.com/plotarmordev/drama-tensorfold/pull/36). Tested with smoke tests
  using a real orphaning parent, and with the real Metal engine: the supervisor exited, then `stop_signal` was 15, the
  group settled and the server was gone.

### 5. Launchers start under `python3 -I`

- **Here:** part of `pilot-macos-external-admission.patch` (`4bf4b70`, #136). Both launchers now import
  `weight_selection`, which `-I` cannot find.
- **PR:** [drama-tensorfold#37](https://github.com/plotarmordev/drama-tensorfold/pull/37). Tested: a new test fails on
  main and passes on the branch, and the real engine served a request under `-I`.

### 6. Streamed `reasoning_content` trailing newline

- **Here:** no code change. It is a named exception in the chat qualification
  (`tools/chat/qwen38-27b/qualify.py`, `STREAM_REASONING_TRAILING_WS`, `685f517`, #14), also used by `bench_macos.py`.
  The bug report draft is `tools/chat/qwen38-27b/upstream-issue-stream-reasoning-newline.md`.
- **Upstream file:** `zig/src/server/chat.zig`. The whole reply `pyStrip`s the reasoning; the stream sends it as
  decoded. The bug is still on main.
- **PR:** [#1743](https://github.com/ashhart/TensorFold/pull/1743). Tested on Metal (Qwen3.8-27B, this Mac): the stream went from 599 to 598 bytes on main's repro, equal to the whole reply; tokens and content unchanged. CI runs the zig suites.

### 7. `graph.endCapture` leaves the capture scope open

- **Here:** `src/engines/qwen_image/dit.zig` and `src/engines/minimax_h3/dit.zig` call
  `cuda.copy_source.captureEnd` themselves when `endCapture` fails (`37755e0`, #92, the 1.1.4 runtime pin).
- **Why:** since 1.1.x, `endCapture` closes copy-source's capture scope only on success. After an invalidated
  capture, every later capture and `DeviceBuffer.upload`/`fromHost` in the process is refused. This is still on main.
- **PR:** [#1744](https://github.com/ashhart/TensorFold/pull/1744). A no-GPU fake-driver test fails on main; it was compiled on Linux and CI runs it. No CUDA run.

### 8. CLIP tokenizer Lowercase

- **Here:** `src/engines/kandinsky6/text/clip_tokenizer.zig` removes the `Lowercase` step from `tokenizer.json`
  before TensorFold parses it and lowercases the text itself. TensorFold's tokenizer has no `Lowercase` normalizer,
  and main still lacks one.
- **Not opened:** no TensorFold model uses a lowercasing tokenizer. Upstream's rules count code with no in-tree user as
  dead code. Revisit if TensorFold adds a CLIP or BERT-style encoder. Kandinsky-6 is GB10 only.

### 9. Chat stays on TensorFold 1.1.1 on GB10

- **Here:** `tools/chat/qwen38-27b/pins.json` (`251503f`, #97); the Mac uses 1.1.4. 1.1.4's prompt cache resumes only
  at 4,096-row prompt chunks, so a short conversation's next turn is prefilled again (first token 1.46 s, against
  0.1–0.3 s on 1.1.1; [CHAT.md](CHAT.md#why-not-114-yet)).
- **Not a patch:** we carry no code. Before the pin moves, `chat_bench.py turn` must match 1.1.1. It has not been
  measured on v1.2.0.

### 10. cuBLASLt recorded algorithm, driver identity (gaps G1, G3)

- **Here:** `src/engine/vendor/cublaslt.zig` and `src/engine/vendor/driver.zig`. They add cuBLASLt plans with a
  recorded algorithm from a pinned library, and a check of the libcuda identity. TensorFold is not changed.
- **Not upstreamable:** these implement LocalRouter's attested-library policy, not a TensorFold gap.

### 11. NVFP4 kernels from TensorFold v0.6.1

- **Here:** `kernels/cuda/nvfp4/*`, written by `tools/kernels/sync.py`. They are whole files, with only the torch
  includes and host wrappers cut.
- **Not upstreamable:** the kernels are unmodified, and the v0.6.x Python line takes no changes.
