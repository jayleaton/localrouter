#!/usr/bin/env python3
"""LocalRouter worker for Kitten TTS 2: text to 24 kHz speech with a built-in voice.

Speaks the worker protocol (docs/ARCHITECTURE.md): one JSON object a line on stdin and stdout, logs on stderr.

    {"op":"load"}                                   -> {"ok":true,"resident":<bytes>}
    {"op":"generate","id","dir","request":{"speech":{"input":"Hello.","voice":"Bruno","speed":1.0,"seed":7}}}
                                                    -> {"progress":{"phase":"speech","step":i,"of":n}}*
                                                       {"ok":true,"files":["speech.wav"],"seed":7,"ms":…}
    {"op":"exit"}

The reply's file is always speech.wav: RIFF, mono, signed 16-bit little-endian PCM at the model's native 24 kHz. The
daemon encodes the response formats a client asks for. Audio never travels inline in the JSON lines.

The model is the pinned KittenML/kitten-tts-2 checkpoint run by its own Python package (kittenml, the reference
implementation), offline, from the files `fetch.sh` downloaded and checked against `pins.json`. Only the lossless
packings are served: the ternary-packed language model (bit-identical to the bf16 file) and the default S3 decoder.
The lossy `emb4` embedding and the distilled student decoders need `"allow_lossy": true` in the tool's options.

Synthesis follows `KittenTTS2.generate` step for step (normalize, split into sentence chunks, language model to codec
tokens, decode, trim and join), so a progress line can follow each chunk; `gate.py` checks the bits against the
package's own `generate`.
"""

import json
import os
import random
import sys
import threading
import time
import wave

SAMPLE_RATE = 24000
OUTPUT = "speech.wav"
HERE = os.path.dirname(os.path.abspath(__file__))
PINS = os.path.join(HERE, "pins.json")
LOSSLESS_WEIGHTS = ("packed", "full")
LOSSLESS_DECODERS = ("default",)
# The voices named after a language: KittenML's guidance is to switch the English-tuned normalizer off for them.
LANGUAGE_VOICES = frozenset({"Arabic", "Chinese", "French", "German", "Hindi", "Italian", "Portuguese", "Russian",
                             "Spanish"})


class Refused(Exception):
    """A request this model cannot serve: the API's 400."""


class Options:
    """The tool's config (LOCALROUTER_TOOL): `weights` is the asset root `fetch.sh` filled; `options` tune the rest."""

    def __init__(self, tool):
        o = tool.get("options") or {}
        # The daemon reads some keys of the same object itself (the advertised `voices`, for one), so keys this worker
        # does not know are logged rather than refused.
        known = {"device", "threads", "lm_weights", "decoder", "allow_lossy", "max_input_chars",
                 "default_voice", "voices", "presets", "min_speed", "max_speed", "default_preset", "normalize",
                 "warm_voices", "verify", "keep_float", "lm_decode", "graph_tokens", "graph_backend"}
        unknown = sorted(set(o) - known)
        if unknown:
            log(f"options not used by the worker: {unknown}")
        self.voices = o.get("voices")
        self.presets = o.get("presets")
        # Kitten TTS 2 has no rate control, so the daemon must be told to accept speed 1 only.
        if (o.get("min_speed", 0.25), o.get("max_speed", 4)) != (1, 1):
            raise ValueError("this model has no speed control: set min_speed and max_speed to 1")
        self.root = tool.get("weights") or o.get("assets") or ""
        self.device = o.get("device", "auto")
        self.threads = int(o.get("threads", 0))
        if "deterministic" in o:
            # torch's deterministic algorithms change the output of HiFT's first ConvTranspose1d upsample (ups.0) on
            # CUDA (GB10, torch 2.11, measured; presumably another cuDNN algorithm): the audio would no longer be the
            # reference's. The reference's own defaults reproduce exactly from process to process, so the worker uses
            # them and offers no switch.
            raise ValueError("the deterministic option was removed: the worker runs with the reference's torch settings")
        self.lm_weights = o.get("lm_weights", "packed")
        self.decoder = o.get("decoder", "default")
        self.allow_lossy = bool(o.get("allow_lossy", False))
        self.max_chars = int(o.get("max_input_chars", 4096))
        self.default_voice = o.get("default_voice")
        self.preset = o.get("default_preset")
        self.normalize = o.get("normalize", "auto")
        self.warm_voices = o.get("warm_voices")
        self.verify = o.get("verify", "size")
        self.keep_float = bool(o.get("keep_float", False))
        # "eager": kittenml's own generate, bit-exact with the reference (and slower than real time on GB10: RTF about
        # 1.5, host-bound). "graphs": the decode step as CUDA graphs over a static KV cache (other kernels, so other
        # bits; qualify.sh --lm-decode graphs checks it against its accepted baseline). graph_tokens: the static
        # cache's length, prompt plus the longest chunk's budget.
        self.lm_decode = o.get("lm_decode", "eager")
        self.graph_tokens = int(o.get("graph_tokens", 2048))
        # "inductor": torch.compile reduce-overhead (fused Triton kernels in CUDA graphs); "cudagraphs": the eager
        # kernels captured as they are (no Triton), the fallback where inductor cannot build for the GPU.
        self.graph_backend = o.get("graph_backend", "inductor")
        self.qualification = tool.get("qualification") or ""  # the marker the daemon gates on; load() checks it
        if not self.allow_lossy and self.lm_weights not in LOSSLESS_WEIGHTS:
            raise ValueError(f"lm_weights {self.lm_weights!r} is lossy; set allow_lossy to serve it")
        if not self.allow_lossy and self.decoder not in LOSSLESS_DECODERS:
            raise ValueError(f"decoder {self.decoder!r} is lossy; set allow_lossy to serve it")
        if self.normalize not in ("auto", True, False):
            raise ValueError("normalize must be \"auto\", true or false")
        if self.verify not in ("size", "sha256", "none"):
            raise ValueError("verify must be \"size\", \"sha256\" or \"none\"")
        if self.max_chars < 1:
            raise ValueError("max_input_chars must be at least 1")
        if self.lm_decode not in ("eager", "graphs"):
            raise ValueError("lm_decode must be \"eager\" or \"graphs\"")
        if self.lm_decode == "graphs" and self.device not in ("cuda", "auto"):
            raise ValueError("lm_decode \"graphs\" needs device \"cuda\" (CUDA graphs)")
        if self.graph_backend not in ("inductor", "cudagraphs"):
            raise ValueError("graph_backend must be \"inductor\" or \"cudagraphs\"")
        if not 256 <= self.graph_tokens <= 8192:
            raise ValueError("graph_tokens must be from 256 to 8192")


class Speech:
    """One validated speech request (the `speech` arm of the daemon's request)."""

    def __init__(self, r, max_chars, voices, default_voice, presets):
        if not isinstance(r, dict):
            raise Refused("a speech request is required")
        text = r.get("input", r.get("text"))
        if not isinstance(text, str) or not text.strip():
            raise Refused("input is required")
        if len(text) > max_chars:
            raise Refused(f"input is too long: {len(text)} characters, at most {max_chars}")
        self.text = text
        self.voice = r.get("voice") or default_voice
        if self.voice not in voices:
            raise Refused(f"unknown voice {self.voice!r}; voices: {', '.join(voices)}")
        speed = r.get("speed")
        if speed is not None and float(speed) != 1.0:
            # Kitten TTS 2 has no rate control; resampling or time-stretching would be an approximation, not the model.
            raise Refused("speed is not supported by this model (1.0 only)")
        if r.get("instructions"):
            raise Refused("instructions are not supported by this model; use inline [emotion] tags in the input")
        if r.get("stream") or r.get("stream_format") not in (None, "", "audio"):
            raise Refused("streaming is not supported; the reply is one file")
        self.preset = r.get("preset") or None
        if self.preset is not None and self.preset not in presets:
            raise Refused(f"unknown preset {self.preset!r}; presets: {', '.join(presets)}")
        seed = r.get("seed")
        if seed is None:
            seed = random.getrandbits(63)
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 1 << 64:
            raise Refused("seed must be an integer from 0 to 2^64-1")
        self.seed = seed


def pcm16(wav):
    """Float samples to signed 16-bit, clipped, rounded half to even."""
    import numpy as np
    return np.rint(np.clip(np.asarray(wav, np.float32), -1.0, 1.0) * 32767.0).astype("<i2")


def write_wav(path, samples):
    """RIFF/WAVE, one channel, 16-bit PCM (little-endian, as RIFF defines it) at 24 kHz."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.tobytes())


def resident_bytes():
    """This process's resident host memory plus what torch holds on the GPU."""
    with open("/proc/self/statm") as f:
        rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        rss += torch.cuda.memory_reserved()
    return rss


def host_memory(torch):
    """After load: free what nothing references and report what host memory is left (RSS split, live CPU tensors)."""
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    with open("/proc/self/status") as f:
        rss = {k: int(v.split()[0]) >> 10 for k, v in (line.split(":", 1) for line in f) if k in ("RssAnon", "RssFile")}
    seen, sizes = set(), []
    for o in gc.get_objects():  # issubclass(type(o)): isinstance would touch dead weakref proxies (Dynamo's)
        try:
            if not issubclass(type(o), torch.Tensor) or o.device.type != "cpu":
                continue
            storage = o.untyped_storage()
            if storage.data_ptr() in seen:  # views and shared storages count once
                continue
            seen.add(storage.data_ptr())
            sizes.append(storage.nbytes())
        except Exception:  # fake, meta or lazy tensors, odd subclasses
            continue
    sizes.sort(reverse=True)
    return {"rss_anon_mib": rss.get("RssAnon"), "rss_file_mib": rss.get("RssFile"),
            "cpu_tensors_mib": sum(sizes) >> 20, "largest_cpu_tensors_mib": [s >> 20 for s in sizes[:5]]}


def check_marker(opts):
    """The qualification marker names the decode it qualified: refuse to serve another one under it (the daemon only
    reads "qualified"). No readable marker: nothing to check here, the daemon's gate decides."""
    if not opts.qualification:
        return
    try:
        with open(opts.qualification) as f:
            record = json.load(f)
    except (OSError, ValueError):
        return
    want = {"lm_decode": opts.lm_decode}
    if opts.lm_decode == "graphs":
        want |= {"graph_backend": opts.graph_backend, "graph_tokens": opts.graph_tokens}
    got = {k: record.get(k, "eager" if k == "lm_decode" else None) for k in want}
    if record.get("qualified") is True and got != want:
        raise RuntimeError(f"{opts.qualification} qualified {got}, but this tool is configured for {want}: run "
                           "tools/speech/kitten/qualify.sh for this configuration")


class KittenEngine:
    """The loaded model: kittenml's KittenTTS2 on the pinned, verified files."""

    def __init__(self, opts):
        self.opts = opts
        self.model = None

    def load(self):
        opts = self.opts
        pins = load_pins()
        kitten_dir, s3gen_path = asset_paths(opts.root, pins)
        verify_assets(opts.root, pins, opts.verify, opts.lm_weights, opts.decoder)
        check_marker(opts)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if opts.lm_decode == "graphs":
            # inductor's compiled kernels survive a restart here, so a reload compiles from cache, not from scratch
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                                  os.path.join(os.path.expanduser("~"), ".cache", "localrouter", "kitten-inductor"))
        import torch
        import kittenml
        if kittenml.__version__ != pins["runtime"]["kittenml"]:
            raise RuntimeError(f"kittenml {kittenml.__version__} is installed; pins.json wants "
                               f"{pins['runtime']['kittenml']}")
        import kittenml.kittentts2.vocoder as vocoder
        from kittenml.kittentts2 import text as text_utils
        from kittenml.kittentts2.model import KittenTTS2

        # The decoder's s3gen weights come from ResembleAI/chatterbox-turbo; serve the pinned, verified copy instead
        # of letting the package reach the Hub.
        def pinned_s3gen(repo_id, filename, **_):
            want = pins["s3gen"]
            if repo_id != want["repo"] or filename != want["file"]:
                raise RuntimeError(f"unpinned download requested: {repo_id}/{filename}")
            return s3gen_path

        vocoder.hf_hub_download = pinned_s3gen
        if opts.threads > 0:
            torch.set_num_threads(opts.threads)
        # The reference (kittenml's KittenTTS2 with torch's defaults) is what the audio must equal bit for bit; settings
        # that pick other kernels are refused rather than silently changing the output.
        if (torch.are_deterministic_algorithms_enabled() or torch.backends.cudnn.deterministic
                or torch.backends.cudnn.benchmark):
            raise RuntimeError("torch is set to deterministic algorithms or cuDNN deterministic/benchmark mode; the "
                               "reference runs with these off")
        device = opts.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(os.path.join(kitten_dir, "config.json")) as f:
            config = json.load(f)
        if device != "cpu" and not getattr(KittenTTS2, "_weights_to_device", False):
            # kittenml unpacks the language model into a CPU state dict, installs it with load_state_dict(assign=True)
            # and moves the model with .to(): on GB10 about 3.9 GiB of host memory stayed live after load. Moving each
            # unpacked tensor to the device first leaves no host copy for anything to hold. The values are the same
            # bf16 bits, so the output is unchanged (the gate checks it).
            unpack = KittenTTS2._load_lm_weights

            def weights_on_device(model, lm_dir):
                state = unpack(model, lm_dir)
                for name in list(state):
                    state[name] = state[name].to(model.device)
                return state

            KittenTTS2._load_lm_weights = weights_on_device
            KittenTTS2._weights_to_device = True
        self.model = KittenTTS2(kitten_dir, config, device=device, decoder=opts.decoder, weights=opts.lm_weights)
        self.text = text_utils
        self.torch = torch
        self.voices = self.model.available_voices
        self.default_voice = opts.default_voice or config.get("default_voice") or self.voices[0]
        if self.default_voice not in self.voices:
            raise ValueError(f"default_voice {self.default_voice!r} is not one of the model's voices")
        # The voices the daemon advertises (and validates before load) must all exist in the pinned checkpoint.
        missing = sorted(set(opts.voices or []) - set(self.voices))
        if missing:
            raise ValueError(f"options.voices names voices the model lacks: {missing}")
        missing = sorted(set(opts.presets or []) - set(self.model.decode_presets))
        if missing:
            raise ValueError(f"options.presets names presets the model lacks: {missing}")
        self.presets = list(opts.presets or self.model.decode_presets)
        self.default_preset = opts.preset or self.model.default_preset
        # Build the normalizer's grammars and the voices' conditioning now, not inside the first request.
        text_utils.warm_normalizer()
        for voice in opts.warm_voices if opts.warm_voices is not None else [self.default_voice]:
            self.model._voice_reference(voice)
        if opts.lm_decode == "graphs":
            if device != "cuda":
                raise RuntimeError(f"lm_decode \"graphs\" needs CUDA; the device is {device}")
            t = time.monotonic()
            import importlib.util  # by path: the worker runs with -I, so its own directory is not on sys.path
            spec = importlib.util.spec_from_file_location("kitten_graphs", os.path.join(HERE, "kitten_graphs.py"))
            graphs = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(graphs)
            cache_bytes = graphs.enable_graph_decode(self.model, torch, opts)
            graphs.warm_graph_decode(self.model, torch, self.default_voice)
            log(f"graphs: decode compiled and recorded in {time.monotonic() - t:.1f} s ({opts.graph_backend}, "
                f"static cache {opts.graph_tokens} tokens, {cache_bytes >> 20} MiB)")
        if device == "cuda":
            torch.cuda.synchronize()
        try:  # a diagnostic: it never fails the load
            log(f"loaded on {device}: host {json.dumps(host_memory(torch))}")
        except Exception as e:
            log(f"loaded on {device}: host memory not measured: {type(e).__name__}: {e}")
        return resident_bytes()

    def request(self, r):
        return Speech(r, self.opts.max_chars, self.voices, self.default_voice, self.presets)

    def synthesize(self, s, progress):
        """KittenTTS2.generate for `s`, reporting each chunk; returns float32 samples and the per-chunk record."""
        m, text_utils = self.model, self.text
        advanced = m._resolve_advanced({"seed": s.seed})
        normalize = self.opts.normalize
        if normalize == "auto":
            normalize = s.voice not in LANGUAGE_VOICES
        reference = m._speaker(s.voice, None, None)
        spoken = text_utils.normalize_text(s.text, normalize=normalize)
        if not spoken.strip():
            raise Refused("the input has nothing to speak")
        chunks = text_utils.split_for_synthesis(spoken, max_chars=advanced["chunk_chars"],
                                                min_chars=advanced["chunk_min_chars"])
        settings = m._resolve_preset(s.preset or self.default_preset, {})
        self.torch.manual_seed(int(s.seed))
        emotion_ids = None
        if m.emotion_control and text_utils.has_expression_tag(s.text):
            emotion_ids = m.tokenizer.encode(m.emotion_control, add_special_tokens=False)
        record, wavs = [], []
        for i, chunk in enumerate(chunks):
            budget = m._chunk_budget(chunk, 1000, single_chunk=len(chunks) == 1)
            ids = m._generate_tokens(reference, chunk, settings, budget, m.use_reference_prompt, emotion_ids,
                                     advanced)
            wav = m.codec.decode(reference["conditioning"], ids)
            if wav is not None:
                wavs.append(wav)
            record.append({"text": chunk, "tokens": ids, "samples": 0 if wav is None else len(wav)})
            progress(i + 1, len(chunks))
        joined = text_utils.join_chunks(wavs, m.sample_rate, gap_s=advanced["chunk_gap_s"])
        if joined is None:
            raise RuntimeError("the model produced no audio tokens")
        return joined, record

    def generate(self, s, out_dir, progress):
        start = time.monotonic()
        wav, record = self.synthesize(s, progress)
        write_wav(os.path.join(out_dir, OUTPUT), pcm16(wav))
        if self.opts.keep_float:
            import numpy as np
            np.asarray(wav, "<f4").tofile(os.path.join(out_dir, "speech.f32"))
            with open(os.path.join(out_dir, "speech.chunks.json"), "w") as f:
                json.dump(record, f)
        ms = int((time.monotonic() - start) * 1000)
        cuda = ""
        if self.torch.cuda.is_available() and self.torch.cuda.is_initialized():
            cuda = f", cuda peak reserved {self.torch.cuda.max_memory_reserved() >> 20} MiB"
        log(f"speech {len(wav) / SAMPLE_RATE:.2f} s in {ms} ms, {len(record)} chunk(s), voice {s.voice}, "
            f"resident {resident_bytes() >> 20} MiB{cuda}")
        return {"ok": True, "files": [OUTPUT], "seed": s.seed, "ms": ms}


def load_pins():
    with open(PINS) as f:
        return json.load(f)


def asset_paths(root, pins):
    if not root:
        raise ValueError("the tool's weights must name the asset root fetch.sh filled")
    k, s = pins["kitten"], pins["s3gen"]
    return (os.path.join(root, k["dir"], k["revision"]),
            os.path.join(root, s["dir"], s["revision"], s["file"]))


def needed_files(pins, lm_weights, decoder):
    """The pinned files this selection loads: the language model packing, its tokenizer, voices, and the decoder."""
    k = pins["kitten"]
    out = [("kitten", p) for p in k["common"]]
    out += [("kitten", p) for p in k["lm"][lm_weights]]
    out += [("kitten", p) for p in k["decoders"].get(decoder, [])]
    out.append(("s3gen", pins["s3gen"]["file"]))
    return out


def verify_assets(root, pins, mode, lm_weights, decoder):
    """Every file the load reads must be present at its pinned size (and hash, with verify "sha256")."""
    import hashlib
    if mode == "none":
        return
    kitten_dir, s3gen_path = asset_paths(root, pins)
    for repo, rel in needed_files(pins, lm_weights, decoder):
        path = os.path.join(kitten_dir, rel) if repo == "kitten" else s3gen_path
        size, digest = pins["files"][repo][rel]
        try:
            actual = os.path.getsize(path)
        except OSError:
            raise RuntimeError(f"missing pinned file {path}; run tools/speech/kitten/fetch.sh") from None
        if actual != size:
            raise RuntimeError(f"{path} is {actual} bytes, pinned {size}")
        if mode == "sha256":
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() != digest:
                raise RuntimeError(f"{path} sha256 {h.hexdigest()}, pinned {digest}")


def log(msg):
    print(f"kitten: {msg}", file=sys.stderr, flush=True)


def watch_parent():
    """Exit when the daemon that started us is gone (LOCALROUTER_PARENT; 0 or unset: no check)."""
    parent = int(os.environ.get("LOCALROUTER_PARENT", "0") or 0)
    if parent <= 1:
        return

    def run():
        while True:
            if os.getppid() != parent:
                os._exit(1)
            time.sleep(1)

    threading.Thread(target=run, daemon=True).start()


def serve(engine, inp, out):
    """The protocol loop. `engine` needs load() -> resident, request(dict) -> Speech, generate(speech, dir, progress)."""

    def send(obj):
        out.write(json.dumps(obj, separators=(",", ":")) + "\n")
        out.flush()

    resident = None
    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
            op = cmd.get("op")
        except (ValueError, AttributeError):
            send({"ok": False, "error": "malformed command", "type": "server_error"})
            continue
        if op == "exit":
            break
        try:
            if op == "load":
                if resident is None:
                    resident = engine.load()
                send({"ok": True, "resident": int(resident)})
            elif op == "generate":
                if resident is None:
                    raise RuntimeError("generate before load")
                req = (cmd.get("request") or {}).get("speech")
                if req is None:
                    raise Refused("this tool makes speech only")
                speech = engine.request(req)
                out_dir = cmd.get("dir") or ""
                if not os.path.isdir(out_dir):
                    raise RuntimeError(f"job directory {out_dir!r} does not exist")
                reply = engine.generate(speech, out_dir,
                                        lambda step, of: send({"progress": {"phase": "speech", "step": step,
                                                                            "of": of}}))
                send(reply)
            else:
                send({"ok": False, "error": f"unknown op {op!r}", "type": "server_error"})
        except Refused as e:
            send({"ok": False, "error": str(e), "type": "invalid_request_error"})
        except Exception as e:  # a failed load or generation is reported, not fatal to the protocol
            log(f"{op} failed: {type(e).__name__}: {e}")
            send({"ok": False, "error": f"{type(e).__name__}: {e}", "type": "server_error"})


def main():
    # Protocol lines own the real stdout; anything the libraries print goes to the log instead.
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    watch_parent()
    tool = json.loads(os.environ.get("LOCALROUTER_TOOL") or "{}")
    try:
        engine = KittenEngine(Options(tool))
    except ValueError as e:
        engine = None
        error = str(e)
    if engine is None:
        log(f"bad config: {error}")
        # A bad config fails every load with the reason, so the daemon reports it.
        class Broken:
            def load(self):
                raise RuntimeError(f"bad kitten config: {error}")

        engine = Broken()
    serve(engine, sys.stdin, out)


if __name__ == "__main__":
    main()
