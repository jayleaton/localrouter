"""python tools/speech/whisper/twin.py WAV... [--timestamps segment,word] [--wer refs.json] [--report out.json]

The worker against the model card's reference recipe, bit for bit. It drives worker.py as a subprocess over the
worker protocol (generate with "debug": true), then runs the reference inline in this process (processor + generate
written out here, not imported from whisper_core) on the same device, dtype and thread count, and compares:
  features   sha256 of the log-mel input features
  tokens     the generated token ids, whole and per segment
  scores     sha256 of each window's stacked per-step scores (the decoder's processed logits)
  output     the model's transcript (before the voice activity gate): text, language, segment times and texts, words
  gate       the worker's final transcript against the gate written out here: Silero VAD's own get_speech_timestamps
             (the vendored, pinned algorithm and model) and the rule "drop the segments that overlap no speech"
Exits 1 on any mismatch. --wer adds WER/CER of the worker's text against reference transcripts ({"file.wav": "text"}),
both sides through Whisper's text normalizer (English for en, basic otherwise)."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import wave

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch_model import DEFAULT_ROOT, REVISION  # noqa: E402  (stdlib-only constants)
from vendor import silero_utils_vad  # noqa: E402  (Silero's own code, pinned; not the worker's)

OUT = os.path.expanduser("~/.cache/localrouter-speech/out")


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def run_worker(cfg, jobs):
    """jobs: [(job_dir, transcription request)] -> [(reply, transcript, debug)] from one worker process."""
    env = dict(os.environ, LOCALROUTER_TOOL=json.dumps(cfg))
    env.pop("LOCALROUTER_PARENT", None)
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "worker.py")], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, env=env, text=True, bufsize=1)

    def call(cmd):
        p.stdin.write(json.dumps(cmd) + "\n")
        p.stdin.flush()
        progress = []
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError("worker exited")
            msg = json.loads(line)
            if "progress" in msg:
                progress.append(msg["progress"])
                continue
            return msg, progress

    load, _ = call({"op": "load"})
    if not load.get("ok"):
        raise RuntimeError(f"worker load failed: {load}")
    out = []
    for i, (job_dir, req) in enumerate(jobs):
        reply, progress = call({"op": "generate", "id": f"tr_twin{i}", "dir": job_dir, "debug": True,
                                "request": {"transcription": req}})
        tr = dbg = None
        if reply.get("ok"):
            tr = json.load(open(os.path.join(job_dir, "transcript.json")))
            dbg = json.load(open(os.path.join(job_dir, "whisper_debug.json")))
        out.append((reply, progress, tr, dbg))
    p.stdin.write(json.dumps({"op": "exit"}) + "\n")
    p.stdin.flush()
    code = p.wait(timeout=120)
    return load, out, code


def reference(model, processor, wav_path, req):
    """The model card's recipe for one file, written out."""
    with wave.open(wav_path, "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).flatten().astype(np.float32) / 32768.0
    if len(audio) <= 480000:  # short-form: the processor's default call pads to 30 s
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    else:  # long-form, exactly as on the model card
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", truncation=False, padding="longest",
                           return_attention_mask=True)
    features_sha = sha(inputs.input_features)
    inputs = inputs.to(model.device, dtype=model.dtype)
    gen_kwargs = {
        "num_beams": 1,
        "condition_on_prev_tokens": False,
        "compression_ratio_threshold": 1.35,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if not req.get("temperature") else (req["temperature"],),
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "return_timestamps": True,
        "return_segments": True,
        "task": "transcribe",
        "language": req.get("language") or None,
    }
    if req.get("timestamps") == "word":
        gen_kwargs["return_token_timestamps"] = True
    if req.get("prompt"):
        gen_kwargs["prompt_ids"] = processor.get_prompt_ids(req["prompt"], return_tensors="pt").to(model.device)
    attn = model.config._attn_implementation
    torch.manual_seed(int(req.get("seed") or 0))
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    if model.config._attn_implementation != attn:
        model.set_attn_implementation(attn)
    segs = out["segments"][0]
    lang_ids = {v: k[2:-2] for k, v in model.generation_config.lang_to_id.items()}
    if gen_kwargs["language"]:
        language = gen_kwargs["language"]
    elif segs:  # the ASR pipeline's return_language: the language token of the first window's full sequence
        language = next(lang_ids[t] for t in segs[0]["result"]["sequences"].tolist() if t in lang_ids)
    else:
        with torch.no_grad():
            language = lang_ids[model.detect_language(inputs.input_features, num_segment_frames=3000).item()]
    tok = processor.tokenizer
    scores, seen = [], set()
    for s in segs:
        if id(s["result"]) not in seen:
            seen.add(id(s["result"]))
            scores.append(sha(torch.stack([x.float() for x in s["result"]["scores"]])))
    ref = {
        "text": tok.batch_decode(out["sequences"], skip_special_tokens=True, decode_with_timestamps=False)[0].strip(),
        "language": language,
        "duration": len(audio) / 16000,
        "segments": [{"id": i, "start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3),
                      "text": tok.decode(s["tokens"], skip_special_tokens=True).strip()} for i, s in enumerate(segs)],
    }
    if req.get("timestamps") == "word":
        ref["words"] = []
        if segs:
            stamps = torch.cat([s["token_timestamps"] for s in segs])
            _, extra = tok._decode_asr([{"tokens": out["sequences"], "token_timestamps": [stamps]}],
                                       return_timestamps="word", return_language=None, time_precision=30 / 1500)
            ref["words"] = [{"word": c["text"].strip(), "start": c["timestamp"][0], "end": c["timestamp"][1]}
                            for c in extra.get("chunks", [])]
    dbg = {
        "features_sha256": features_sha,
        "tokens": out["sequences"][0].tolist(),
        "segment_tokens": [s["tokens"].tolist() for s in segs],
        "segment_times": [[float(s["start"]), float(s["end"])] for s in segs],
        "scores_sha256": scores,
    }
    return ref, dbg


def reference_gate(vad, wav_path, ref, tok):
    """The voice activity gate, written out: Silero's speech intervals, then the segments overlapping none dropped."""
    with wave.open(wav_path, "rb") as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).flatten().astype(np.float32) / 32768.0
    with torch.no_grad():
        ts = silero_utils_vad.get_speech_timestamps(torch.from_numpy(audio), vad, sampling_rate=16000)
    speech = [[t["start"] / 16000, t["end"] / 16000] for t in ts]
    kept = [seg for seg in ref["segments"] if any(min(seg["end"], e) > max(seg["start"], s) for s, e in speech)]
    out = dict(ref)
    if len(kept) != len(ref["segments"]):
        tokens = []
        for seg in kept:
            tokens += ref["_segment_tokens"][seg["id"]]
        out["segments"] = [dict(seg, id=n) for n, seg in enumerate(kept)]
        out["text"] = tok.decode(tokens, skip_special_tokens=True).strip() if tokens else ""
        if "words" in ref:
            out["words"] = [w for w in ref["words"] if any(seg["start"] <= w["start"] <= seg["end"] for seg in kept)]
    out.pop("_segment_tokens", None)
    out["vad"] = {"model": "silero-vad v6.2.3", "speech": [[round(s, 3), round(e, 3)] for s, e in speech],
                  "dropped_segments": [seg["id"] for seg in ref["segments"] if seg not in kept]}
    return out


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--weights", default=os.path.join(DEFAULT_ROOT, "whisper-large-v3-turbo", REVISION))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--vad-model", required=True, help="the pinned silero_vad.jit")
    ap.add_argument("--timestamps", default="segment", help="comma list of none,segment,word")
    ap.add_argument("--language", default="")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wer", help="JSON {wav basename: reference text}")
    ap.add_argument("--report", default=os.path.join(OUT, "twin-report.json"))
    args = ap.parse_args()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = args.dtype if args.dtype != "auto" else ("float16" if device == "cuda" else "float32")
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = {"id": "whisper-large-v3-turbo", "kind": "transcription", "weights": args.weights,
           "options": {"device": device, "dtype": dtype, "attn": args.attn, "threads": args.threads,
                       "vad": "silero", "vad_model": args.vad_model}}

    refs = {}  # read first: a bad file should fail before the long runs
    for k, v in (json.load(open(args.wer)) if args.wer else {}).items():  # {"a.wav": "text"} or {"a": {"text": ..}}
        text = v.get("text") if isinstance(v, dict) else v
        if isinstance(text, str) and text:
            refs[os.path.splitext(k)[0]] = text

    run_dir = os.path.join(OUT, "twin", time.strftime("%Y%m%d-%H%M%S"))
    jobs, cases = [], []
    for mode in args.timestamps.split(","):
        for wav in args.wavs:
            name = os.path.basename(wav)
            job_dir = os.path.join(run_dir, f"{mode}-{os.path.splitext(name)[0]}")
            os.makedirs(job_dir)
            shutil.copyfile(wav, os.path.join(job_dir, "audio.wav"))
            req = {"audio": "audio.wav", "language": args.language, "prompt": args.prompt,
                   "temperature": args.temperature, "timestamps": mode, "seed": args.seed}
            jobs.append((job_dir, req))
            cases.append((name, mode, wav))
    print(f"twin: worker on {len(jobs)} job(s) in {run_dir}", file=sys.stderr, flush=True)
    load, results, code = run_worker(cfg, jobs)

    print("twin: reference", file=sys.stderr, flush=True)
    processor = WhisperProcessor.from_pretrained(args.weights, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(args.weights, dtype=getattr(torch, dtype),
                                                            attn_implementation=args.attn, local_files_only=True)
    model.to(device).eval()
    vad = silero_utils_vad.init_jit_model(args.vad_model)

    files, all_ok = [], code == 0
    wer_tot = [0, 0, 0, 0]  # word edits, ref words, char edits, ref chars
    for (name, mode, wav), (job_dir, req), (reply, progress, tr, dbg) in zip(cases, jobs, results):
        entry = {"file": name, "timestamps": mode, "worker_ms": reply.get("ms"), "progress": len(progress)}
        if not reply.get("ok"):
            entry.update(ok=False, error=reply)
            files.append(entry)
            all_ok = False
            continue
        ref, rdbg = reference(model, processor, wav, req)
        gated = reference_gate(vad, wav, dict(ref, _segment_tokens=rdbg["segment_tokens"]), processor.tokenizer)
        checks = {
            "features": dbg["features_sha256"] == rdbg["features_sha256"],
            "tokens": dbg["tokens"] == rdbg["tokens"] and dbg["segment_tokens"] == rdbg["segment_tokens"],
            "scores": dbg["scores_sha256"] == rdbg["scores_sha256"],
            "segment_times": dbg["segment_times"] == rdbg["segment_times"],
            "transcript": dbg["model_transcript"] == ref,
            "gate": tr == gated,
        }
        entry.update(ok=all(checks.values()), checks=checks, windows=len(rdbg["scores_sha256"]),
                     features_sha256=rdbg["features_sha256"], scores_sha256=rdbg["scores_sha256"],
                     language=tr["language"], text=tr["text"], segments=len(tr["segments"]),
                     model_text=ref["text"], vad=tr.get("vad"))
        if mode == "word":
            entry["words"] = len(tr.get("words", []))
        if not entry["ok"]:
            entry["worker"], entry["reference"], entry["reference_gated"] = tr, ref, gated
        all_ok &= entry["ok"]
        stem = os.path.splitext(name)[0]
        if refs.get(stem) and mode == args.timestamps.split(",")[0]:
            tok = processor.tokenizer
            norm = tok.normalize if tr["language"] == "en" else tok.basic_normalize
            hyp, gold = norm(tr["text"]).split(), norm(refs[stem]).split()
            we, ce = edit_distance(hyp, gold), edit_distance(" ".join(hyp), " ".join(gold))
            wer_tot[0] += we
            wer_tot[1] += len(gold)
            wer_tot[2] += ce
            wer_tot[3] += len(" ".join(gold))
            entry["wer"] = round(we / max(1, len(gold)), 4)
            entry["cer"] = round(ce / max(1, len(" ".join(gold))), 4)
        files.append(entry)

    report = {
        "pass": all_ok,
        "device": device, "dtype": dtype, "attn": args.attn, "threads": args.threads,
        "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
        "python": sys.version.split()[0],
        "model": "openai/whisper-large-v3-turbo", "revision": REVISION, "weights": args.weights,
        "worker_resident": load.get("resident"), "worker_exit": code,
        "request": {k: v for k, v in jobs[0][1].items() if k not in ("audio", "timestamps")},
        "files": files,
    }
    if refs:
        report["wer"] = {"wer": round(wer_tot[0] / max(1, wer_tot[1]), 4),
                         "cer": round(wer_tot[2] / max(1, wer_tot[3]), 4), "ref_words": wer_tot[1],
                         "normalizer": "Whisper EnglishTextNormalizer for en, BasicTextNormalizer otherwise"}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
