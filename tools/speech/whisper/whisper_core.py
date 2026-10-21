"""Whisper large-v3-turbo transcription: the one code path the worker runs (and `twin.py` checks against its own
inline reference). Audio is 16 kHz mono s16le WAV; decoding is transformers' sequential long-form algorithm with the
model card's generate kwargs. Import this only after the caller has set CUBLAS_WORKSPACE_CONFIG (worker.py does).

Voice activity gate. Whisper writes text for audio that holds no speech ("Thank you." for digital silence, "so" for
white noise: the model card recipe does it too, since its no-speech rule also needs avg_logprob < -1). The model's
output is left exactly as the recipe makes it; afterwards, a segment whose time span overlaps none of the speech
intervals Silero VAD finds (its own get_speech_timestamps with its defaults, vendor/silero_utils_vad.py at the pinned
commit) is dropped. A transcript with nothing dropped is returned unchanged, byte for byte; with every segment
dropped the text is empty. The model's own transcript and the VAD intervals are kept for twin.py and the response."""
import copy
import hashlib
import math
import os
import wave

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from vendor import silero_utils_vad

SAMPLE_RATE = 16000
N_SAMPLES = 30 * SAMPLE_RATE  # one 30 s window; longer audio is long-form
FRAMES_PER_WINDOW = 3000  # mel frames in one window (hop 160)
TIME_PRECISION = 0.02  # seconds per timestamp token (30 s / 1500 encoder positions)

# The model card's generate kwargs (huggingface.co/openai/whisper-large-v3-turbo, "temperature fallback and condition
# on previous tokens"), except max_new_tokens: 448 plus the 3 start tokens exceeds max_target_positions (448), which
# this transformers raises on; None lets generation_config.max_length (448 tokens in all) bound each window instead.
TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CARD_KWARGS = {
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
}
# decode "greedy" (opt-in, for voice turns): one pass at temperature 0 without the fallback's thresholds, so no
# per-step scores are kept. Same model, precision and timestamps; the gate still runs on its segments.
GREEDY_KWARGS = {"num_beams": 1, "condition_on_prev_tokens": False, "return_timestamps": True}


# Silero VAD v6.2.3 (snakers4/silero-vad, MIT): the model file and the vendored algorithm, both pinned.
VAD_NAME = "silero-vad v6.2.3"
VAD_COMMIT = "5cd7945676eb32225748052e2e6a0580e4686a08"
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
    except (wave.Error, EOFError) as e:
        raise InvalidRequest(f"audio is not a readable WAV file: {e or 'empty or truncated header'}") from e
    except FileNotFoundError as e:
        raise InvalidRequest(f"audio file not found: {os.path.basename(path)}") from e
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def resolve_device_dtype(device="auto", dtype="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device cuda requested but torch.cuda.is_available() is False")
    if dtype == "auto":
        dtype = "float16" if device == "cuda" else "float32"
    return device, getattr(torch, dtype)


def load(model_dir, device, dtype, attn="sdpa"):
    """The processor and the model in eval mode on `device`, from local files only."""
    processor = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, attn_implementation=attn,
                                                            local_files_only=True, use_safetensors=True)
    model.to(device).eval()
    return model, processor


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_vad(path):
    """Silero VAD's TorchScript model on the CPU, after checking both pinned digests."""
    utils = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "silero_utils_vad.py")
    for p, want in ((path, VAD_JIT_SHA256), (utils, VAD_UTILS_SHA256)):
        if file_sha256(p) != want:
            raise RuntimeError(f"{p}: sha256 is not the pinned {want}")
    return silero_utils_vad.init_jit_model(path)


def speech_intervals(vad, samples):
    """[[start_s, end_s], ...] where Silero VAD hears speech, with get_speech_timestamps' defaults."""
    with torch.no_grad():
        ts = silero_utils_vad.get_speech_timestamps(torch.from_numpy(samples), vad, sampling_rate=SAMPLE_RATE)
    return [[t["start"] / SAMPLE_RATE, t["end"] / SAMPLE_RATE] for t in ts]


def gate(result, seg_tokens, speech, tokenizer):
    """Drops the segments that overlap no speech interval (see the module docstring); `seg_tokens` are each
    segment's token ids. Returns the result to send and the ids of the dropped segments."""
    def voiced(seg):
        return any(min(seg["end"], e) > max(seg["start"], s) for s, e in speech)

    keep = [i for i, seg in enumerate(result["segments"]) if voiced(seg)]
    dropped = [seg["id"] for i, seg in enumerate(result["segments"]) if i not in keep]
    if not dropped:
        return result, []
    out = dict(result)
    out["segments"] = [dict(result["segments"][i], id=n) for n, i in enumerate(keep)]
    tokens = [t for i in keep for t in seg_tokens[i]]
    out["text"] = tokenizer.decode(tokens, skip_special_tokens=True).strip() if tokens else ""
    if "words" in result:
        spans = [(result["segments"][i]["start"], result["segments"][i]["end"]) for i in keep]
        out["words"] = [w for w in result["words"] if any(s <= w["start"] <= e for s, e in spans)]
    return out, dropped


def features(processor, samples):
    """Log-mel input features and attention mask. Up to 30 s: the processor's default short-form call (audio padded
    to 30 s, 3000 frames). Longer: the model card's long-form call (no truncation, padded to the longest)."""
    if len(samples) <= N_SAMPLES:
        return processor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
    return processor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt", truncation=False, padding="longest",
                     return_attention_mask=True)


def tensor_sha256(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def language_codes(model):
    """ISO code -> language token id, from the generation config ('<|en|>' -> 'en')."""
    return {k[2:-2]: v for k, v in model.generation_config.lang_to_id.items()}


def _set_attn(model, impl):
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(impl)
    else:
        model.config._attn_implementation = impl


def transcribe(model, processor, samples, opts, progress=None, debug=False, vad=None):
    """One request: samples (float32, 16 kHz) and opts {language, prompt, temperature, timestamps, seed, decode,
    max_new_tokens} -> {"text", "language", "duration", "segments", "words"?, "vad"?}. With `vad` (load_vad), the
    voice activity gate runs on the model's transcript. With debug, also "_debug": digests of the input features and
    per-window scores, the generated token ids and the model's own transcript before the gate (for twin.py)."""
    timestamps = opts.get("timestamps") or "none"
    if timestamps not in ("none", "segment", "word"):
        raise InvalidRequest(f"timestamps must be none, segment or word, got {timestamps!r}")
    temperature = float(opts.get("temperature") or 0.0)
    if not 0.0 <= temperature <= 1.0:
        raise InvalidRequest("temperature must be between 0 and 1")
    codes = language_codes(model)
    language = (opts.get("language") or "").lower()
    if language and language not in codes:
        raise InvalidRequest(f"unsupported language {language!r}")
    decode = opts.get("decode") or "card"
    if decode not in ("card", "greedy"):
        raise InvalidRequest(f"decode must be card or greedy, got {decode!r}")
    if decode == "greedy" and (temperature != 0.0 or debug):
        raise InvalidRequest("decode greedy is temperature 0 only, without debug")
    seed = int(opts.get("seed") or 0)
    device, dtype = model.device, model.dtype

    inputs = features(processor, samples)
    feats = inputs.input_features
    n_windows = max(1, math.ceil(feats.shape[-1] / FRAMES_PER_WINDOW)) if len(samples) > N_SAMPLES else 1
    inputs = inputs.to(device, dtype=dtype)

    if not language:  # the detection generate() itself runs on the first 30 s, done here so it can be reported
        with torch.no_grad():
            lang_id = model.detect_language(input_features=inputs.input_features,
                                            generation_config=model.generation_config,
                                            num_segment_frames=FRAMES_PER_WINDOW).item()
        language = {v: k for k, v in codes.items()}[lang_id]

    kwargs = dict(GREEDY_KWARGS if decode == "greedy" else CARD_KWARGS)
    kwargs["max_new_tokens"] = opts.get("max_new_tokens") or None
    # OpenAI's transcription API: temperature 0 means fallback through higher temperatures on failed thresholds.
    kwargs["temperature"] = (0.0,) if decode == "greedy" else TEMPERATURE_FALLBACK if temperature == 0.0 else (temperature,)
    kwargs["return_segments"] = True
    words = timestamps == "word"
    if words:
        kwargs["return_token_timestamps"] = True  # the pipeline's return_timestamps="word" mechanism
    prompt = opts.get("prompt") or ""
    if prompt:
        kwargs["prompt_ids"] = processor.get_prompt_ids(prompt, return_tensors="pt").to(device)

    def monitor(p):  # (batch, 2): seek frame, total frames
        seek, total = int(p[0, 0]), int(p[0, 1])
        if progress and total:
            progress(min(n_windows - 1, seek * n_windows // total), n_windows)

    attn = model.config._attn_implementation
    torch.manual_seed(seed)
    try:
        with torch.no_grad():
            out = model.generate(input_features=inputs.input_features, attention_mask=inputs.attention_mask,
                                 language=language, task="transcribe", monitor_progress=monitor, **kwargs)
    finally:
        if model.config._attn_implementation != attn:  # word timestamps switch the model to eager attention
            _set_attn(model, attn)
    if progress:
        progress(n_windows, n_windows)

    tok = processor.tokenizer
    sequences = out["sequences"]
    segs = out["segments"][0]
    result = {
        "text": tok.batch_decode(sequences, skip_special_tokens=True, decode_with_timestamps=False)[0].strip(),
        "language": language,
        "duration": len(samples) / SAMPLE_RATE,
        "segments": [{"id": i, "start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3),
                      "text": tok.decode(s["tokens"], skip_special_tokens=True).strip()}
                     for i, s in enumerate(segs)],
    }
    if words:
        result["words"] = []
        if segs:
            stamps = torch.cat([s["token_timestamps"] for s in segs])
            _, extra = tok._decode_asr([{"tokens": sequences, "token_timestamps": [stamps]}], return_timestamps="word",
                                       return_language=None, time_precision=TIME_PRECISION)
            result["words"] = [{"word": c["text"].strip(), "start": c["timestamp"][0], "end": c["timestamp"][1]}
                               for c in extra.get("chunks", [])]
    model_result = copy.deepcopy(result)  # the model's own transcript, independent of what the gate and debug add
    if vad is not None:
        speech = speech_intervals(vad, samples)
        result, dropped = gate(result, [s["tokens"].tolist() for s in segs], speech, tok)
        result = dict(result)  # model_result stays the model's own transcript
        result["vad"] = {"model": VAD_NAME, "speech": [[round(s, 3), round(e, 3)] for s, e in speech],
                         "dropped_segments": dropped}
    if debug:
        windows, seen = [], set()
        for s in segs:
            r = s["result"]
            if id(r) in seen:
                continue
            seen.add(id(r))
            windows.append(tensor_sha256(torch.stack([x.float() for x in r["scores"]])))
        result["_debug"] = {
            "features_sha256": tensor_sha256(feats),
            "features_shape": list(feats.shape),
            "tokens": sequences[0].tolist(),
            "segment_tokens": [s["tokens"].tolist() for s in segs],
            "segment_times": [[float(s["start"]), float(s["end"])] for s in segs],
            "scores_sha256": windows,
            "model_transcript": model_result,
        }
    return result
