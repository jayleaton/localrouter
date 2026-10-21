# Whisper large-v3-turbo worker

A `cmd` worker (see [the cmd adapter](../../../docs/ARCHITECTURE.md#any-program-the-cmd-adapter)) that transcribes
16 kHz mono s16le WAV files with `openai/whisper-large-v3-turbo` through transformers, plus its reference check and
benchmark.

| File | Does |
| --- | --- |
| `worker.py` | the worker: `load`, `generate` (writes `transcript.json`), `exit` |
| `whisper_core.py` | the one transcription code path (`transcribe(model, processor, samples, opts)`) |
| `twin.py` | the worker against the model card recipe written out inline: features, tokens, scores, output bit for bit; optional WER/CER |
| `loadcheck.py` | the load is all or nothing and the gate cannot be bypassed: bad VAD, unknown mode, a failed model load refuse load, a repeat load and generate, and hold no memory; debug with the gate off stays valid JSON |
| `wheel_tags.py` | after `pip install --require-hashes`, before an unmodified `pip check`: for the locked nvidia-cusparselt-cu13 0.8.1 wheel only (torch 2.13.0+cu130 pins it exactly), whose file name says `manylinux2014_aarch64` but whose WHEEL file is expected to say `manylinux2014_sbsa`, rewrites that Tag line and its RECORD row only after proving the wheel is the requirements-pinned file, its archive RECORD is sound, and the installed metadata and every installed file (each library an AArch64 ELF) are the archive's bytes. Anything else refuses; originals and an outcome record go to the job's output; a part-way write is reported UNPROVEN |
| `wheel_tags_selftest.py` | wheel_tags.py on scratch copies of the real wheel, authenticated and validated before anything is unpacked, then laid out by real pip (`--target`, so RECORD is pip's own form): the positive case and each refusal (wrong pin, changed metadata or library, duplicate, unsafe or extra RECORD rows, orphan or hashed bytecode rows, symlinked library, unexpected tag or version, already corrected) and an injected failure between the two writes (UNPROVEN); each copy is removed after its case and a cleanup failure fails the self-test |
| `bench.py` | cold load, first and warm generate, RTF, peak RSS / CUDA memory, SIGKILL cancellation and memory reclaim |
| `fetch_model.py` | downloads the pinned model files and checks their sha256 (stdlib only) |

## Setup

```sh
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/localrouter-speech/venv   # outside the repository
uv sync --locked --python 3.12 --project tools/speech/whisper
python3 tools/speech/whisper/fetch_model.py     # -> ~/.cache/localrouter-speech/models/whisper-large-v3-turbo/<revision>/
```

A tool entry for the daemon:

```json
{"id": "whisper-large-v3-turbo", "kind": "transcription",
 "cmd": ["/home/me/.cache/localrouter-speech/venv/bin/python", "/path/to/tools/speech/whisper/worker.py"],
 "weights": "/home/me/.cache/localrouter-speech/models/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
 "options": {"device": "auto", "dtype": "auto", "attn": "sdpa"}}
```

`options`: `device` auto|cpu|cuda (auto: CUDA when available), `dtype` auto|float16|bfloat16|float32 (auto: float16 on
CUDA, float32 on CPU), `attn` sdpa|eager, `threads` (torch CPU threads), `max_new_tokens` (default: none, see below).

## Pins and why

| Pin | Why |
| --- | --- |
| Python 3.12 | every pinned wheel exists for cp312 on x86_64 and aarch64; the system Python (3.14) is not used |
| torch 2.13.0 | official wheels for both targets: `+cpu` x86_64 (development) and `+cu130` aarch64 (GB10 / DGX Spark, CUDA 13.0), and the same torch release as the NGC 26.07 image the image engines' reference (`docs/dev/RESULTS.md`) was measured with |
| transformers 5.19.0 | the version the existing references use (`docs/dev/RESULTS.md`); has Whisper's sequential long-form decoding, temperature fallback and word timestamps from alignment heads |
| numpy 2.5.4 | current release with cp312 wheels for both targets |

`[tool.uv.sources]` sends torch to `download.pytorch.org/whl/cpu` on x86_64 and `/whl/cu130` on aarch64, so the one
`uv.lock` covers both machines (`[tool.uv] environments` restricts resolution to those two). Nothing else is a direct
dependency: no accelerate, librosa or ffmpeg bindings (the WhisperFeatureExtractor computes the log-mel in numpy, and
the daemon has already converted the upload to 16 kHz mono WAV). sm_121 (GB10) is served by the cu130 wheels'
sm_120 code; that is not yet measured on the target.

Model: `openai/whisper-large-v3-turbo` at revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`, license MIT. Only
the files transformers reads are fetched; `fetch_model.py` holds each file's size and sha256.

## Decoding

`whisper_core.transcribe` follows the model card (huggingface.co/openai/whisper-large-v3-turbo, "Transformers is
compatible with all Whisper decoding strategies, such as temperature fallback and condition on previous tokens"):

```python
gen_kwargs = {
    "max_new_tokens": 448,            # not used, see below
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
}
```

* Audio up to 30 s goes through the processor's default call (padded to 30 s, 3000 frames); longer audio through the
  model card's `truncation=False, padding="longest", return_attention_mask=True` call and transformers' sequential
  long-form algorithm (not the chunked pipeline). The model card's long-form call on short audio leaves the features
  unpadded and degrades the result (no casing or punctuation on the LibriSpeech sample), so it is not used for it.
* `max_new_tokens=448` plus the three start tokens exceeds `max_target_positions` (448) and transformers 5.19 raises
  on it; it is left unset so the generation config's `max_length` (448 tokens in all) bounds each window.
* `temperature` 0 (the default) uses the fallback tuple, as OpenAI's transcription API documents for 0; a request
  temperature above 0 samples at that one temperature. Sampling is seeded with `torch.manual_seed(seed)`.
* `decode: "greedy"` (opt-in; the voice loop sends it, the HTTP API does not) decodes once at temperature 0 without
  the fallback's thresholds, so no per-step scores are kept; the rest of the recipe, the timestamps and the gate are
  the same. On GB10 it gave the same transcripts as the recipe on every fixture here and on the voice e2e prompts,
  about 135 ms sooner on a voice turn. It refuses `temperature` above 0 and `debug`.
* After loading, the worker decodes a second of silence once and discards it: the first request would otherwise pay
  the first-call CUDA setup (on GB10, 1.1 s against 0.36 s warm).
* `return_timestamps=True` always (one decode whatever the response format), so `segments` are always written;
  `timestamps: "word"` adds `return_token_timestamps=True` (the ASR pipeline's `return_timestamps="word"` mechanism,
  cross-attention DTW over the generation config's `alignment_heads`) and groups tokens into words the way the
  pipeline does. transformers switches the model to eager attention for that call; the worker switches it back, so
  later requests run as configured. Word requests therefore decode with eager attention and may differ in the last
  bits from the same file without words.
* `language` "" detects the language on the first 30 s (`model.detect_language`, the call `generate` makes itself)
  and reports it; the task is always `transcribe`. `prompt` becomes `prompt_ids` (`processor.get_prompt_ids`).
* No speech, no text. The recipe writes text for audio without speech: on GB10 it decodes 10 s of digital silence as
  "Thank you." and 10 s of white noise as "so" (its no-speech rule skips a window only when no_speech_prob > 0.6 *and*
  avg_logprob < -1). The worker keeps the model's output exactly as the recipe makes it, then gates it: Silero VAD
  v6.2.3 (`get_speech_timestamps` with its defaults: threshold 0.5, 250 ms minimum speech, 30 ms padding) finds the
  speech intervals, and a segment that overlaps none of them is dropped. Nothing dropped: the transcript is returned
  unchanged, byte for byte; everything dropped: empty text and no segments. `transcript.json` carries the intervals
  and the dropped segment ids under `vad`. No phrase list is involved. `options.vad: "off"` returns the raw recipe
  output; it must be set explicitly, and the worker refuses to load with no `vad_model` otherwise.

## Checks

```sh
V=$HOME/.cache/localrouter-speech/venv/bin/python
$V tools/speech/whisper/twin.py a.wav b.wav --vad-model silero_vad.jit --timestamps segment,word --wer refs.json   # exit 1 on any mismatch
$V tools/speech/whisper/bench.py a.wav --vad-model silero_vad.jit --warm 2 --cancel long.wav
```

Outputs go to `~/.cache/localrouter-speech/out/` (never the repository).

## Third-party code and models

| What | Pin | License |
| --- | --- | --- |
| openai/whisper-large-v3-turbo | revision 41f01f3fe87f28c78e2fbf8b568835947dd65ed9, per-file sha256 in `fetch_model.py` | MIT |
| Silero VAD model `silero_vad.jit` | snakers4/silero-vad 5cd7945676eb32225748052e2e6a0580e4686a08 (v6.2.3), sha256 in `whisper_core.py` | MIT |
| `vendor/silero_utils_vad.py` | that commit's `src/silero_vad/utils_vad.py`, verbatim, sha256 in `whisper_core.py` | MIT (`vendor/SILERO-LICENSE`) |
| CPython 3.12.15 aarch64 (Spark jobs) | python-build-standalone 20261003, sha256 from its release SHA256SUMS in `tools/spark/whisper-job.sh` | PSF |
