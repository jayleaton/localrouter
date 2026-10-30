# Qwen3.8-27B chat on GB10 (TensorFold 1.1.1)

The `qwen3.8-27b` chat tool: TensorFold's native server (`bin/tensorfold-native serve`) running
`TensorFold/Qwen3.8-27B-MLX-4bit` with the `z-lab/Qwen3.8-27B-DFlash2` drafter, behind LocalRouter's `openai_server`
engine. Every file is pinned in `pins.json`.

TensorFold qualifies the 27B upstream only on Metal and on CUDA with an RTX 4090. On GB10 (linux-aarch64 CUDA) it is
unproven upstream, so the tool carries a `qualification` marker and LocalRouter refuses it (503 `server_error`, "not
GPU-qualified on this host") until `qualify.sh` passes on the machine and writes that marker.

| File | Does |
| --- | --- |
| `pins.json` | TensorFold v1.1.1 (tag, commit, both Linux archives by size and sha256), the checkpoint and drafter revisions, and every file needed to serve (LFS files by sha256, small files by git blob sha1) |
| `fetch.sh ROOT [--hf DIR]` | downloads the archive for this host to `ROOT/downloads`, extracts it to `ROOT/tensorfold-1.1.1` (checked against the archive's `SHA256SUMS`), and fetches the checkpoint and drafter into the Hugging Face cache layout under DIR (default `ROOT/hf`). Resumable and idempotent; `HF_TOKEN` is used if set |
| `setup.sh ROOT [--hf DIR] [--marker PATH]` | runs `--version`, `capabilities --json`, `models` and `info` with the cache offline, checks `lib/libtfgrammar.so`, and writes and prints `ROOT/config.json`, the tool config with this host's paths |
| `config.example.json` | the daemon config fragment (`gpu_lock`, `defaults.chat_completion`, the tool). `options.memory` says where `resident_bytes` (24 GiB, measured on DGX-01) comes from |
| `prompts.json` | the fixed prompt set P1 to P8: factual, reasoning on, reasoning off, a required tool call, a tool result, a JSON schema, stop + `max_tokens` 64, a ~2k-token multi-turn prompt. Temperature 0, seed 7 |
| `qualify.sh ROOT LOCALROUTER_BIN OUT MARKER` | the GB10 run: pins, direct, daemon, compare, then the marker. Exit codes in `OUT/status.json`, memory peak and timings in `OUT/report.json` |
| `qualify.py`, `qualify_lib.py` | the steps (standard library only, `python3 -I`) |

## Order

```sh
tools/chat/qwen38-27b/fetch.sh /opt/localrouter/chat --hf /data/models/chat/hf      # ~20 GB
tools/chat/qwen38-27b/setup.sh /opt/localrouter/chat --hf /data/models/chat/hf      # prints the tool config
# merge the printed gpu_lock, defaults and tools into the daemon's config; stop the daemon
tools/chat/qwen38-27b/qualify.sh /opt/localrouter/chat /opt/localrouter/bin/localrouter /tmp/qwen38-qualify-1 \
  /data/models/chat/qwen38-27b-qualification.json
# restart the daemon: the tool is served once the marker says "qualified": true
```

`qualify.sh` refuses to run unless the host is aarch64 with a GB10 (`nvidia-smi -L`) and MemAvailable is at least
64 GiB. `--any-gpu` skips the host check for a dry run elsewhere; a dry run never writes the marker. The steps:

1. **pins**: the archive's sha256, every file of the extracted tree against the archive's `SHA256SUMS` (no extra
   files), every pinned checkpoint and drafter file and `refs/main`, and that the config's `server` argv runs that
   binary on the pinned repos.
2. **direct**: starts the config's `options.server` argv itself (`{port}` a free port, `{id}` the tool id, the same
   `env`), takes the host's `gpu_lock`, waits for `/health`, sends each prompt whole twice and streamed once, then
   SIGTERMs the server and checks it is gone.
3. **daemon**: `LOCALROUTER_BIN serve` with a temporary config (port 0, data under OUT, `reserve_bytes` 8 GiB, the
   tool without `qualification` and with `keep_loaded: false`); checks `/v1/models` lists the tool as chat and sends
   each prompt through `/v1/chat/completions` whole and streamed; SIGTERM, and the server must be gone.
4. **compare**: per prompt, direct run 1 = run 2, daemon = direct, stream = whole (direct and daemon), each stream
   ends with `data: [DONE]`, on content, reasoning_content, tool_calls (name and arguments), finish_reason and
   completion_tokens; plus each prompt's structural expectations (P4 calls `get_weather` with JSON arguments, P6
   parses with the schema's keys, P7 stays within 64 tokens, ...).
5. **marker**: only when all four pass, written atomically (`.tmp` then rename) with host, GPU, the sha256 of
   `pins.json` and of the archive, the number of checks, OUT and the time.

MemAvailable is sampled every 0.5 s during direct and daemon; `report.json` gives the largest drop as the measured
candidate for `resident_bytes`; the config's 24 GiB comes from the DGX-01 bring-up (docs/dev/JARVIS-BRINGUP.md).

## macOS (Apple Silicon)

The same tool on a Mac: TensorFold 1.1.4's `macos-arm64` release (native Zig/Metal, no Python or MLX install) serving
the same checkpoint and drafter revisions, behind the same `openai_server` engine. The Linux files above are
unchanged; the Mac has its own beside them.

| File | Does |
| --- | --- |
| `pins-macos.json` | TensorFold v1.1.4 (tag, commit, the macos-arm64 archive by size and sha256, its published `.sha256` and `.sha256.sig` by sha256), and the same checkpoint and drafter files as `pins.json` |
| `fetch-macos.sh ROOT [--hf DIR]` | `fetch.sh` for macOS: the archive (checked against the pin and the release's `.sha256`), extracted to `ROOT/tensorfold-1.1.4` and checked against its `SHA256SUMS`, then the checkpoint and drafter into DIR (default the shared `~/.cache/huggingface/hub`) |
| `setup-macos.sh ROOT [--hf DIR]` | `--version`, `capabilities --json` (must list only the `metal` backend and every serve flag of the config), `models` and `info` offline, `lib/libtfgrammar.dylib`; writes and prints `ROOT/config.json` |
| `config.macos.example.json` | the Mac tool config: no `qualification` marker (that gate is the GB10's), `resident_bytes` 30 GiB from this Mac's measurement (docs/dev/CHAT.md) |
| `baseline-macos.json` | the Mac reference baseline: this Mac's greedy replies to prompts.json on 1.1.4, which every `bench_macos.py` run on 1.1.4 must equal (Metal is not CUDA arithmetic; see docs/dev/CHAT.md) |
| `bench_macos.py serve\|daemon\|port` | the Mac measurement and qualification: load time, footprint (rusage lifetime maximum and `footprint`), prompts.json's checks against `baseline-macos.json`, speed with and without the drafter, cached follow-ups; `daemon` runs the same suite through `localrouter serve` (plus MCP `load_model` and `chat`), `port` against any running endpoint |

The `.sha256.sig` is pinned by hash only: no key that verifies it is published at the pinned commit (see the notes in
`pins-macos.json`). The archive's sha256 and its `SHA256SUMS` are the trust inputs, as on Linux.

```sh
tools/chat/qwen38-27b/fetch-macos.sh ~/.local/share/localrouter/chat          # ~20 GB into ~/.cache/huggingface/hub
tools/chat/qwen38-27b/setup-macos.sh ~/.local/share/localrouter/chat          # prints the tool config
# each run loads the model: one at a time, nothing else large on the GPU
python3 -I tools/chat/qwen38-27b/bench_macos.py serve --config ~/.local/share/localrouter/chat/config.json \
  --out /tmp/qwen-mac-drafter
python3 -I tools/chat/qwen38-27b/bench_macos.py serve --config ~/.local/share/localrouter/chat/config.json \
  --out /tmp/qwen-mac-plain --no-drafter
# through LocalRouter (stop any running daemon that holds the tool first)
python3 -I tools/chat/qwen38-27b/bench_macos.py daemon --bin zig-out/bin/localrouter \
  --config ~/.local/share/localrouter/chat/config.json --out /tmp/qwen-mac-daemon
```
