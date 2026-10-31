"""Whisper large-v3-turbo transcription on Apple Silicon: the one code path the macOS worker runs. Audio is 16 kHz mono
s16le WAV; decoding is mlx-whisper's port of OpenAI's sequential long-form algorithm, on the MLX conversion
(mlx-community/whisper-large-v3-turbo, FP16) on the GPU.

This is a different engine from the Spark worker (../whisper_core.py, transformers on CUDA), so its transcripts are a
Mac reference of their own, not bit-identical to the Spark's (CONTRIBUTING.md, engine correctness). What it keeps the
same: the request fields and their refusals, the transcript JSON (text, language, duration, segments, words, vad), the
decoding policy (temperature fallback, condition_on_previous_text off, no-speech rule, greedy for voice turns) and the
voice activity gate: Silero VAD v6.2.3 as a TorchScript model on the CPU, the same pinned file and vendored
get_speech_timestamps with its defaults (threshold 0.5, 250 ms minimum speech, 30 ms padding); a segment that overlaps
no speech interval is dropped, a transcript with nothing dropped is returned unchanged."""
import hashlib
import importlib
import json
import math
import os
import sys
import threading
import types
import wave

import mlx.core as mx
import numpy as np
import torch
import mlx.nn as nn
from mlx_whisper import whisper
from mlx_whisper.tokenizer import LANGUAGES, get_tokenizer

mlx_transcribe = importlib.import_module("mlx_whisper.transcribe")  # the module: the package exports its function

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the Spark worker's vendor/
from vendor import silero_utils_vad  # noqa: E402

SAMPLE_RATE = 16000
N_SAMPLES = 30 * SAMPLE_RATE
FRAMES_PER_WINDOW = 3000  # mel frames in one 30 s window (hop 160)
WEIGHTS = ("config.json", "weights.safetensors")

# The policy of the Spark worker's recipe in mlx-whisper's (OpenAI's) terms. compression_ratio_threshold is 2.4 here:
# OpenAI's algorithm takes the gzip ratio of the text, and 2.4 is its default; the 1.35 of the transformers recipe is
# that threshold in token space (transformers' generate documentation).
TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CARD_KWARGS = {"compression_ratio_threshold": 2.4, "logprob_threshold": -1.0, "no_speech_threshold": 0.6}
# decode "greedy" (voice turns): one pass at temperature 0 without the fallback's thresholds or no-speech skip.
GREEDY_KWARGS = {"compression_ratio_threshold": None, "logprob_threshold": None, "no_speech_threshold": None}

# Silero VAD v6.2.3 (snakers4/silero-vad, MIT): the model file and the vendored algorithm, the Spark worker's pins.
VAD_NAME = "silero-vad v6.2.3"
VAD_JIT_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"  # src/silero_vad/data/silero_vad.jit
VAD_UTILS_SHA256 = "15d0f6b3ef9590e22ca2f146e1436f6901da356cb552d3183df4100862b333cb"  # src/silero_vad/utils_vad.py


class InvalidRequest(ValueError):
    """The request cannot be served (bad audio, unknown language, unsupported option): a 400, not a worker fault."""


def read_wav(path):
    """16 kHz mono PCM s16le WAV -> float32 samples in [-1, 1), converted exactly as openai-whisper does."""
    try:
        with wave.open(path, "rb") as w:
            if w.getcomptype() != "NONE":
                raise InvalidRequest(f"audio must be uncompressed PCM, got {w.getcomptype()}")
            if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SAMPLE_RATE:
                raise InvalidRequest(f"audio must be 16 kHz mono 16-bit PCM WAV, got {w.getframerate()} Hz, "
                                     f"{w.getnchannels()} channel(s), {8 * w.getsampwidth()}-bit")
            frames = w.readframes(w.getnframes())
            if len(frames) < 2 * w.getnframes():  # nothing is half heard: the daemon refuses a cut WAV too
                raise InvalidRequest(f"audio is truncated: its header says {w.getnframes()} samples, it holds "
                                     f"{len(frames) // 2}")
    except (wave.Error, EOFError) as e:
        raise InvalidRequest(f"audio is not a readable WAV file: {e or 'empty or truncated header'}") from e
    except FileNotFoundError as e:
        raise InvalidRequest(f"audio file not found: {os.path.basename(path)}") from e
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def resolve_dtype(dtype="auto"):
    if dtype in ("auto", "float16"):
        return mx.float16
    if dtype == "float32":
        return mx.float32
    raise RuntimeError(f"options.dtype must be auto, float16 or float32, got {dtype!r}")


def load(model_dir, dtype):
    """The model with its weights evaluated on the GPU, from local files only. mlx_whisper's own load_model, except
    that the weights must fit the config exactly (names and shapes, MLX's strict load_weights): load_model updates the
    model with whatever the file holds, so a config that does not fit it would run with random layers."""
    for name in WEIGHTS:
        if not os.path.isfile(os.path.join(model_dir, name)):
            raise RuntimeError(f"no {name} in weights dir {model_dir!r}")
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    config.pop("model_type", None)
    quantization = config.pop("quantization", None)
    weights = mx.load(os.path.join(model_dir, "weights.safetensors"))
    model = whisper.Whisper(whisper.ModelDimensions(**config), dtype)
    if quantization is not None:
        nn.quantize(model, **quantization, class_predicate=lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
                    and f"{p}.scales" in weights)
    if "alignment_heads" in weights:  # the checkpoint's own DTW heads (turbo: 6) replace the default (every upper head)
        model.alignment_heads = weights["alignment_heads"]
    try:
        model.load_weights(list(weights.items()), strict=True)
    except ValueError as e:
        raise RuntimeError(f"the weights in {model_dir!r} do not fit its config.json: {e}") from e
    del weights
    mx.eval(model.parameters())
    # mlx_whisper.transcribe looks its model up by path (ModelHolder); hand it this one so it never loads a second.
    mlx_transcribe.ModelHolder.model, mlx_transcribe.ModelHolder.model_path = model, model_dir
    return model


def release():
    """Drops mlx_whisper's reference to the model and returns MLX's cached buffers to the system."""
    mlx_transcribe.ModelHolder.model = mlx_transcribe.ModelHolder.model_path = None
    mx.clear_cache()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_vad(path):
    """Silero VAD's TorchScript model on the CPU, after checking both pinned digests."""
    utils = os.path.abspath(silero_utils_vad.__file__)
    for p, want in ((path, VAD_JIT_SHA256), (utils, VAD_UTILS_SHA256)):
        if not os.path.isfile(p) or file_sha256(p) != want:
            raise RuntimeError(f"{p}: missing, or its sha256 is not the pinned {want}")
    return silero_utils_vad.init_jit_model(path)


class _Stopped(Exception):
    pass


def speech_intervals(vad, samples, stop=None):
    """[[start_s, end_s], ...] where Silero VAD hears speech, with get_speech_timestamps' defaults. `stop` (an Event)
    ends it at the next 32 ms chunk (its progress callback), raising _Stopped."""
    def check(_percent):
        if stop is not None and stop.is_set():
            raise _Stopped()
    with torch.no_grad():
        ts = silero_utils_vad.get_speech_timestamps(torch.from_numpy(samples), vad, sampling_rate=SAMPLE_RATE,
                                                    progress_tracking_callback=check)
    return [[t["start"] / SAMPLE_RATE, t["end"] / SAMPLE_RATE] for t in ts]


def gate(result, seg_tokens, speech, tokenizer):
    """Drops the segments that overlap no speech interval; `seg_tokens` are each segment's token ids. Returns the
    result to send and the ids of the dropped segments (the Spark worker's rule, whisper_core.gate)."""
    def voiced(seg):
        return any(min(seg["end"], e) > max(seg["start"], s) for s, e in speech)

    keep = [i for i, seg in enumerate(result["segments"]) if voiced(seg)]
    dropped = [seg["id"] for i, seg in enumerate(result["segments"]) if i not in keep]
    if not dropped:
        return result, []
    out = dict(result)
    out["segments"] = [dict(result["segments"][i], id=n) for n, i in enumerate(keep)]
    tokens = [t for i in keep for t in seg_tokens[i]]
    out["text"] = tokenizer.decode(tokens).strip() if tokens else ""
    if "words" in result:
        spans = [(result["segments"][i]["start"], result["segments"][i]["end"]) for i in keep]
        out["words"] = [w for w in result["words"] if any(s <= w["start"] <= e for s, e in spans)]
    return out, dropped


def language_codes(model):
    """The ISO codes the model has a language token for (large-v3 and turbo: 100, with Cantonese)."""
    return list(LANGUAGES)[:model.num_languages]


class _Progress:
    """Stands in for tqdm inside mlx_whisper.transcribe: frames decoded -> {step, of} windows."""
    sink = None

    def __init__(self, total=0, **_):
        self.total, self.n = total, 0
        self.of = max(1, math.ceil(total / FRAMES_PER_WINDOW))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def update(self, n):
        self.n += n
        if _Progress.sink and self.total:
            _Progress.sink(min(self.of - 1, self.n * self.of // self.total), self.of)


mlx_transcribe.tqdm = types.SimpleNamespace(tqdm=_Progress)


class _ShortForm:
    """Audio up to 30 s is padded with silence, as the Spark recipe's features are. mlx_whisper (like openai-whisper
    and transformers' long-form loop) cuts a window to the audio's frames and pads the log-mel with zeros, which is
    not the log-mel of silence (that is the clamp floor, max - 8, rescaled); large-v3-turbo then drops casing and
    punctuation (M5 Ultra: "mr quilter is the apostle of the middle classes and we..." against the recipe's "Mr.
    Quilter is the apostle of the middle classes, and we..."). While `on`, a window shorter than 30 s is padded with
    the last frame of the whole log-mel, which mlx computes over the audio plus 30 s of digital zeros: a silent
    frame. Longer audio keeps the zero padding of its last window, as transformers' sequential long-form does."""
    on = False
    silence = None
    _mel = staticmethod(mlx_transcribe.log_mel_spectrogram)
    _pad = staticmethod(mlx_transcribe.pad_or_trim)

    @staticmethod
    def mel(*args, **kwargs):
        m = _ShortForm._mel(*args, **kwargs)
        _ShortForm.silence = m[-1:] if _ShortForm.on else None
        return m

    @staticmethod
    def pad(array, length=N_SAMPLES, *, axis=-1):
        short = _ShortForm.silence is not None and axis == -2 and array.ndim == 2 and array.shape[0] < length
        if not short:
            return _ShortForm._pad(array, length, axis=axis)
        fill = mx.broadcast_to(_ShortForm.silence, (length - array.shape[0], array.shape[1])).astype(array.dtype)
        return mx.concatenate([array, fill], axis=0)


mlx_transcribe.log_mel_spectrogram = _ShortForm.mel
mlx_transcribe.pad_or_trim = _ShortForm.pad


def join(thread, stop):
    """thread.join() that a signal handler's exception (the worker's soft cancel) cannot cut short: it stops the
    thread, waits for it to end, then raises what interrupted it."""
    interrupted = None
    while thread.is_alive():
        try:
            thread.join()
        except BaseException as e:
            interrupted = e
            stop.set()
    if interrupted is not None:
        raise interrupted


def transcribe(model, model_dir, dtype, samples, opts, progress=None, debug=False, vad=None):
    """One request: samples (float32, 16 kHz) and opts {language, prompt, temperature, timestamps, seed, decode,
    max_new_tokens} -> {"text", "language", "duration", "segments", "words"?, "vad"?}. With `vad` (load_vad), the
    voice activity gate runs on the model's transcript. With debug, also "_debug": the token ids and the model's own
    transcript before the gate. `model` must be the one loaded from `model_dir` with `dtype` (mlx_whisper looks it
    up by path and casts the features to the dtype its fp16 flag names)."""
    timestamps = opts.get("timestamps") or "none"
    if timestamps not in ("none", "segment", "word"):
        raise InvalidRequest(f"timestamps must be none, segment or word, got {timestamps!r}")
    temperature = float(opts.get("temperature") or 0.0)
    if not 0.0 <= temperature <= 1.0:
        raise InvalidRequest("temperature must be between 0 and 1")
    language = (opts.get("language") or "").lower()
    if language and language not in language_codes(model):
        raise InvalidRequest(f"unsupported language {language!r}")
    decode = opts.get("decode") or "card"
    if decode not in ("card", "greedy"):
        raise InvalidRequest(f"decode must be card or greedy, got {decode!r}")
    if decode == "greedy" and (temperature != 0.0 or debug):
        raise InvalidRequest("decode greedy is temperature 0 only, without debug")
    seed = int(opts.get("seed") or 0)

    kwargs = dict(GREEDY_KWARGS if decode == "greedy" else CARD_KWARGS)
    # OpenAI's transcription API: temperature 0 means fallback through higher temperatures on failed thresholds.
    kwargs["temperature"] = (0.0,) if decode == "greedy" else TEMPERATURE_FALLBACK if temperature == 0.0 else (temperature,)
    kwargs["condition_on_previous_text"] = False  # the Spark recipe's condition_on_prev_tokens=False; mlx defaults True
    kwargs["word_timestamps"] = timestamps == "word"
    kwargs["initial_prompt"] = opts.get("prompt") or None
    kwargs["language"] = language or None  # None: detected on the first 30 s, as the Spark worker does
    kwargs["task"] = "transcribe"
    kwargs["fp16"] = dtype == mx.float16
    if opts.get("max_new_tokens"):
        kwargs["sample_len"] = int(opts["max_new_tokens"])

    # The gate's speech intervals depend on the samples only: Silero runs on the CPU while the GPU decodes.
    found = {}
    finder = None
    stop = threading.Event()
    if vad is not None:
        def find_speech():
            try:
                found["speech"] = speech_intervals(vad, samples, stop)
            except BaseException as e:  # re-raised below, in the caller's thread
                found["error"] = e
        finder = threading.Thread(target=find_speech, name="silero", daemon=True)
        finder.start()

    mx.random.seed(seed)
    _Progress.sink = progress
    _ShortForm.on = len(samples) <= N_SAMPLES
    try:
        out = mlx_transcribe.transcribe(samples, path_or_hf_repo=model_dir, verbose=False, **kwargs)
    except BaseException:
        stop.set()  # the decode failed or was cancelled: the gate's intervals are not needed
        raise
    finally:
        _Progress.sink = None
        _ShortForm.on, _ShortForm.silence = False, None
        if finder is not None:
            join(finder, stop)  # never two Silero runs at once, even when the decode raised (or was cancelled)
    n_windows = max(1, math.ceil(len(samples) / N_SAMPLES))
    if progress:
        progress(n_windows, n_windows)

    tokenizer = get_tokenizer(True, num_languages=model.num_languages, language=out["language"], task="transcribe")
    segs = [s for s in out["segments"] if s["text"].strip()]  # mlx keeps instantaneous or empty segments, emptied
    result = {
        "text": out["text"].strip(),
        "language": out["language"],
        "duration": len(samples) / SAMPLE_RATE,
        "segments": [{"id": i, "start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3),
                      "text": s["text"].strip()} for i, s in enumerate(segs)],
    }
    if timestamps == "word":
        result["words"] = [{"word": w["word"].strip(), "start": round(float(w["start"]), 3),
                            "end": round(float(w["end"]), 3)} for s in segs for w in s.get("words", [])]
    seg_tokens = [[t for t in s["tokens"] if t < tokenizer.eot] for s in segs]
    model_result = {**result, "segments": [dict(s) for s in result["segments"]]}
    if finder is not None:
        if "error" in found:
            raise found["error"]
        speech = found["speech"]
        result, dropped = gate(result, seg_tokens, speech, tokenizer)
        result = dict(result)
        result["vad"] = {"model": VAD_NAME, "speech": [[round(s, 3), round(e, 3)] for s, e in speech],
                         "dropped_segments": dropped}
    if debug:
        result["_debug"] = {"segment_tokens": seg_tokens, "segment_times": [[s["start"], s["end"]] for s in segs],
                            "model_transcript": model_result}
    return result
