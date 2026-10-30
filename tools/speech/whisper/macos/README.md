# Whisper large-v3-turbo worker on macOS (Apple Silicon, MLX)

The macOS counterpart of [the Spark worker](../README.md): the same `cmd` protocol, request fields, refusals and
`transcript.json`, with the same Silero gate. It runs Whisper through [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
on the GPU (Metal) instead of transformers on CUDA. It has its own uv project and lock. The Linux worker, its
`uv.lock` and its pins are unchanged.

| File | Does |
| --- | --- |
| `worker.py` | the worker: `load`, `generate` (writes `transcript.json`), `exit`, soft cancel on SIGUSR1 |
| `mlx_core.py` | the one transcription code path (`transcribe(model, model_dir, dtype, samples, opts)`) and the gate |
| `check.py` | the worker over its protocol, real weights: failed loads hold nothing, lifecycle, refusals, transcript shape, memory after each job, soft and hard cancellation, a daemon that goes away; exit 1 on any failure |
| `bench.py` | cold load, resident and peak footprint, per fixture text/WER/CER/RTF, word timestamps, greedy, warm runs; `--compare` against another engine's transcripts |
| `fetch_model.py` | downloads the pinned MLX weights and checks their sha256 (stdlib only) |
| `pyproject.toml`, `uv.lock` | the macOS arm64 environment (nothing else resolves in it) |

## Setup

```sh
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/localrouter-speech/venv-mlx   # outside the repository
uv sync --locked --python 3.12 --project tools/speech/whisper/macos
python3 -I tools/speech/whisper/macos/fetch_model.py   # -> ~/.cache/localrouter-speech/models/whisper-large-v3-turbo-mlx/<revision>/
curl -fL -o silero_vad.jit https://raw.githubusercontent.com/snakers4/silero-vad/5cd7945676eb32225748052e2e6a0580e4686a08/src/silero_vad/data/silero_vad.jit
shasum -a 256 silero_vad.jit   # e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720
```

A tool entry for the daemon:

```json
{"id": "whisper-large-v3-turbo", "kind": "transcription",
 "cmd": ["/Users/me/.cache/localrouter-speech/venv-mlx/bin/python", "/path/to/tools/speech/whisper/macos/worker.py"],
 "weights": "/Users/me/.cache/localrouter-speech/models/whisper-large-v3-turbo-mlx/a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
 "resident_bytes": 3489660928, "working_bytes": 1342177280,
 "options": {"dtype": "float16", "threads": 4, "vad": "silero", "vad_model": "/path/to/silero_vad.jit"}}
```

`options`: `dtype` auto|float16|float32 (auto is float16, the checkpoint's own precision), `threads` (torch CPU
threads, used by the gate), `max_new_tokens` (mlx-whisper's `sample_len`), `vad`, `vad_model` (as on Linux: the gate
must be configured or explicitly `"off"`). `device`/`attn` do not apply.

## Pins and why

| Pin | Why |
| --- | --- |
| Python 3.12 | as the Linux worker |
| mlx 0.32.4, mlx-whisper 0.4.3 | current releases, with macOS 14+ arm64 cp312 wheels; mlx-whisper is OpenAI's transcription algorithm (long-form, temperature fallback, DTW word timestamps) on MLX |
| torch 2.13.0 | the Silero gate only (TorchScript on the CPU), the Linux worker's release, from PyPI's macOS arm64 wheel |
| numpy 2.5.4 | as the Linux worker |

`[tool.uv] environments` restricts the lock to `sys_platform == 'darwin' and platform_machine == 'arm64'`.

Model: `mlx-community/whisper-large-v3-turbo` at revision `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`, FP16
(`weights.safetensors` 1,613,977,612 B, `config.json`; sha256 in `fetch_model.py`). It is OpenAI's MIT checkpoint
converted to MLX. The conversion card declares no licence of its own. mlx-whisper's tokenizer (tiktoken) ships in the
pinned package, so nothing else is fetched.

## Decoding: the Linux policy in mlx-whisper's terms

| Linux (transformers) | macOS (mlx-whisper) |
| --- | --- |
| `temperature` (0.0, 0.2, …, 1.0), or the one requested temperature | the same |
| `condition_on_prev_tokens=False` | `condition_on_previous_text=False` (mlx defaults True) |
| `compression_ratio_threshold=1.35` (zlib ratio in token space) | `2.4` (the same rule on text, OpenAI's default; transformers documents 1.35 as its token-space equivalent) |
| `logprob_threshold=-1.0`, `no_speech_threshold=0.6` | the same |
| `decode: "greedy"`: one pass at temperature 0, no thresholds | the same (all three thresholds `None`) |
| `language` "" detects on the first 30 s | the same (mlx-whisper's own detection) |
| `prompt` → `prompt_ids` | `initial_prompt` |
| `timestamps: "word"` → cross-attention DTW over the alignment heads | mlx-whisper's `word_timestamps` (DTW over the checkpoint's 6 alignment heads) |
| seed → `torch.manual_seed` | `mx.random.seed` |

Two details make the outputs line up:

* **Audio up to 30 s is padded with silence.** mlx-whisper (like openai-whisper, and transformers' long-form loop)
  cuts each window to the audio's frames and pads the log-mel with zeros. Zero is not the log-mel of silence. With
  that padding, large-v3-turbo drops casing and punctuation on short clips: "mr quilter is the apostle of the middle
  classes and we…" against the recipe's "Mr. Quilter is the apostle of the middle classes, and we…". The Linux
  recipe pads the *audio* of a short request to 30 s. `mlx_core._ShortForm` does the same in the mel domain: it pads
  with a silent frame of the same log-mel. Longer audio keeps the zero-padded last window, as transformers does.
* **The weights must fit the config.** mlx-whisper's `load_model` copies whatever the file holds into the model, so a
  config that does not match would run with random layers. `mlx_core.load` uses MLX's strict `load_weights`: every
  tensor name and shape must match. The checkpoint's own `alignment_heads` are kept.

The gate is the Linux one, unchanged: the same pinned `silero_vad.jit` and vendored `get_speech_timestamps`
(`../vendor`, both sha256-checked), threshold 0.5, 250 ms minimum speech, 30 ms padding. It runs on the CPU while
the GPU decodes. A segment that overlaps no speech interval is dropped. mlx-whisper leaves empty or instantaneous
segments in its output as blank text; they are not sent.

Differences from the Linux worker:

* **Soft cancel.** The load reply says `"cancel": "signal"`, so the daemon's cancel sends SIGUSR1. The decode stops at
  its next Python step and Silero at its next 32 ms chunk. The reply is `{"ok":false,"type":"cancelled"}` and the
  model stays loaded. Protocol lines are never cut mid-write.
* **A truncated WAV is refused.** If the data is shorter than its header says, the request is refused. The daemon
  already refuses such uploads.
* **A daemon that goes away.** macOS has no `PR_SET_PDEATHSIG`, so a thread checks `LOCALROUTER_PARENT` every second,
  also during a decode.
* **Warm-up.** After loading, the worker decodes a second of silence with word timestamps and the gate. That
  compiles the Metal kernels, numba's DTW and TorchScript's optimised Silero. The first word-timestamp request after
  a load took 2.4 s without it; with it, the first request is warm (0.18 s).
* **`resident` and memory.** `resident` is the process's physical footprint (`proc_pid_rusage`, what `footprint` and
  Activity Monitor report). On unified memory that includes the GPU buffers MLX holds. After every job the worker
  returns MLX's buffer cache to the system, so between jobs only the model stays resident.
* **`debug`** writes the segment tokens and times and the model's own transcript before the gate. The Linux worker
  also writes feature and score digests; those are twin.py's bit-level proof against transformers, which does not
  apply here.

## Mac reference baseline (M5 Ultra, 96 GiB, macOS 27.2, 2026-10-30)

This is a different engine from the Spark's, so its transcripts are a reference of their own (CONTRIBUTING.md,
engine correctness). They are not bit-identical to the Spark's. Measured with `bench.py` and `check.py` on
the fixtures from `tools/speech/fixtures` (LibriSpeech dev-clean and FLEURS, CC BY 4.0), under the machine's model
lock. The comparison engine is the Linux worker's own code path (`whisper_core.transcribe`: transformers 5.19.0,
`openai/whisper-large-v3-turbo` at its pinned revision, CPU float32, same gate) run on the same Mac. No Spark
transcript fixture exists to compare against.

| | |
| --- | --- |
| cold start (process start → load reply, warm-up included) | 1.51 s (weights in the page cache; the load itself 0.1–0.6 s) |
| resident after load (footprint) | 3.18 GB (2.96 GiB); MLX active 1.62 GB (the FP16 weights) |
| peak footprint over the whole run | 4.35 GB (4.06 GiB): the 280 s file with word timestamps |
| footprint after the run | 3.22 GB (MLX cache 0) |
| 5.9 s English turn, recipe / greedy (voice) / word timestamps | 0.134 s / 0.093 s / 0.174 s (warm, median of 3) |
| 10–23 s FLEURS clips (de, fr, es, ja, zh) | 0.14–0.24 s, RTF 0.008–0.017 |
| 60.8 s and 280 s long-form | 0.45 s (RTF 0.0074) and 1.76 s (RTF 0.0063) |
| silence 10 s / white noise 10 s | no text (the gate dropped 1 and 2 segments) |
| text identical to the transformers recipe | 15 of 18 fixtures; normalised WER 0 on 17, 0.016 on fr_fleurs_02 |
| greedy (voice) text identical to the recipe | 17 of 17 speech fixtures |
| WER against the human references | en 0–0.09 (LibriSpeech references are unpunctuated upper case; normalised), de/fr/es 0–0.035; CER ja 0, zh 0–0.048 |
| word timestamps | in order, inside the audio and their segments, one per text word, on every fixture |
| soft cancel (SIGUSR1 mid-decode on 280 s) | replied in 2 ms; the next request needed no reload |
| `check.py` | 44 of 44 |

The three that differ from the recipe: `en_librispeech_1272-128104-0001` (recipe "Nor is Mr. Quilter's manner…",
here lower case without punctuation; the same in float32, so not a precision effect), `speech_in_pink_noise_10db` (one
comma) and `fr_fleurs_02` ("pourraient être mieux servis", where the recipe has "pour être mieux servis"; the human
reference has "pourraient"). With word timestamps, the long-form text may differ from the same file without them. As
in openai-whisper, the next window then starts at the last word's end rather than at the last timestamp token.

## Checks

```sh
V=$HOME/.cache/localrouter-speech/venv-mlx/bin/python
LOCK="python3 -I /path/to/gpulock.py"   # on a shared machine: whatever serialises model loads
$LOCK $V tools/speech/whisper/macos/check.py --weights $W --vad-model silero_vad.jit --fixtures ~/.cache/localrouter-speech/fixtures
$LOCK $V tools/speech/whisper/macos/bench.py --weights $W --vad-model silero_vad.jit --fixtures ~/.cache/localrouter-speech/fixtures
```

Outputs go to `~/.cache/localrouter-speech/out/` (never the repository).
