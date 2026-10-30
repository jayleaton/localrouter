"""The voice loop's hearing path on a real daemon, alone: speech fixtures streamed over /v1/realtime as a microphone,
with no chat or speech model involved (`create_response: false`, text modality). Standard library only; reuses
tools/spark/jarvis/voice_e2e.py's WebSocket client and real-time microphone.

    python3 -I hearing.py --url http://127.0.0.1:8190 --fixtures ~/.cache/localrouter-speech/fixtures \\
        [--model whisper-large-v3-turbo] [--partial-ms 0] [--report out.json]

The fixtures are levelled to -24 dBFS and their pauses shortened to 0.3 s (mic()), as a person with a gain-controlled
microphone says one request. Per turn (each fixture in its own session, the session's `input_audio_transcription.language` set to the
fixture's language, or "auto" with --auto): speech end (the fixture's last voiced sample, in wall time) to
speech_stopped and to the final transcript (conversation.item.input_audio_transcription.completed); the transcript
against the human reference (WER, CER for ja/zh); partials when --partial-ms > 0. Silence: no turn may start.
Exits 1 when a turn is not heard, its transcript misses the reference by more than --max-wer (--max-cer), or the
silence starts a turn."""
import argparse
import array
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "spark", "jarvis"))
import voice_e2e as ve  # noqa: E402
from qualify import scores  # noqa: E402

TURNS = ("en_librispeech_1272-128104-0000", "en_librispeech_1272-141231-0000", "en_librispeech_1272-135031-0001",
         "de_fleurs_02", "fr_fleurs_01", "ja_fleurs_01")


class Hearing(ve.Session):
    """A session that only hears: text modality, server VAD without responses."""

    def __init__(self, url, events_file, model, language, partial_ms):
        import re
        host, port = re.match(r"http://([^:/]+):(\d+)", url).groups()
        self.ws = ve.Ws(host, int(port), "/v1/realtime")
        self.events, self.file, self.audio = [], events_file, {}
        self.queue = array.array("h")
        self.next_send = time.monotonic()
        self.sent_samples = 0
        self.wait_for("session.created", 30)
        session = {"modalities": ["text"],
                   "input_audio_transcription": {"model": model, "language": language},
                   "turn_detection": {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300,
                                      "silence_duration_ms": 500, "create_response": False, "interrupt_response": False}}
        if partial_ms:
            session["partial_interval_ms"] = partial_ms
        self.ws.send({"type": "session.update", "session": session})
        upd = self.wait_for("session.updated", 30)
        errs = [e for e in self.events if e.get("type") == "error"]
        if errs:
            raise RuntimeError(f"session.update refused: {errs[0]['error']}")
        self.session = upd.get("session", {})


def mic(path, rms_dbfs=-24.0):
    """The fixture as a microphone with gain control says it: at `rms_dbfs` (FLEURS has clips at -62 dBFS, below any
    room's noise floor), 24 kHz, with pauses over 0.3 s shortened (ve.squeeze: they would end the turn)."""
    s, rate = ve.wav_pcm(open(path, "rb").read())
    rms = (sum(x * x for x in s) / max(len(s), 1)) ** 0.5
    if rms > 0:
        g = 32768 * 10 ** (rms_dbfs / 20) / rms
        s = array.array("h", (max(-32768, min(32767, int(x * g))) for x in s))
    return ve.squeeze(ve.resample(s, rate, ve.RATE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8190")
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--model", default="whisper-large-v3-turbo")
    ap.add_argument("--turns", default=",".join(TURNS))
    ap.add_argument("--auto", action="store_true", help="language auto (detected per turn) instead of the fixture's")
    ap.add_argument("--partial-ms", type=int, default=0)
    ap.add_argument("--max-wer", type=float, default=0.15)
    ap.add_argument("--max-cer", type=float, default=0.15)
    ap.add_argument("--report", default=os.path.expanduser("~/.cache/localrouter-speech/out/hearing.json"))
    a = ap.parse_args()
    refs = json.load(open(os.path.join(a.fixtures, "refs.json")))
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    events = open(os.path.splitext(a.report)[0] + ".events.jsonl", "w")
    checks, turns = [], []

    def check(name, ok, **detail):
        checks.append({"check": name, "ok": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(detail, ensure_ascii=False)[:300]}", flush=True)

    for name in a.turns.split(","):
        ref = refs[name]
        lang = "auto" if a.auto else ref["language"]
        sess = Hearing(a.url, events, a.model, lang, a.partial_ms)
        samples = mic(os.path.join(a.fixtures, name + ".wav"))
        sess.pump(0.5)  # a little room tone first, as a microphone has
        mark = len(sess.events)
        t_end = sess.say(samples)
        try:
            stopped = sess.wait_for("input_audio_buffer.speech_stopped", 60, mark)
            heard = sess.wait_for("conversation.item.input_audio_transcription.completed", 60, mark,
                                  lambda e: e["item_id"] == stopped["item_id"])
        except TimeoutError as e:
            check(f"{name}: heard", False, error=str(e))
            sess.close()
            continue
        sess.pump(0.3)
        partials = [e["transcript"] for e in sess.events[mark:] if e.get("type") == "localrouter.transcription.partial"]
        started = [e for e in sess.events[mark:] if e.get("type") == "input_audio_buffer.speech_started"]
        sess.close()
        wer, cer = scores(heard["transcript"], ref["text"], ref["language"])
        row = {"fixture": name, "language": lang, "duration_s": ref["duration_s"], "transcript": heard["transcript"],
               "reference": ref["text"], "wer": wer, "cer": cer, "turns_started": len(started),
               "speech_end_to_vad_stop_ms": round((stopped["_t"] - t_end) * 1000),
               "speech_end_to_transcript_ms": round((heard["_t"] - t_end) * 1000),
               "vad_stop_to_transcript_ms": round((heard["_t"] - stopped["_t"]) * 1000), "partials": partials}
        turns.append(row)
        cjk = ref["language"] in ("ja", "zh")
        check(f"{name}: heard ({lang})", (cer <= a.max_cer) if cjk else (wer <= a.max_wer),
              transcript=heard["transcript"][:120], wer=wer, cer=cer, end_to_transcript_ms=row["speech_end_to_transcript_ms"],
              vad_to_transcript_ms=row["vad_stop_to_transcript_ms"], partials=len(partials))
        if a.partial_ms:
            check(f"{name}: partials while speaking", len(partials) > 0 or ref["duration_s"] * 1000 < 2 * a.partial_ms,
                  partials=len(partials))

    sess = Hearing(a.url, events, a.model, "en", a.partial_ms)
    mark = len(sess.events)
    sess.say(mic(os.path.join(a.fixtures, "silence_10s.wav")))
    sess.pump(11)
    started = [e for e in sess.events[mark:] if e.get("type") in ("input_audio_buffer.speech_started",
                                                                  "conversation.item.input_audio_transcription.completed")]
    sess.close()
    check("silence: no turn", not started, events=[e["type"] for e in started])
    events.close()

    lat = sorted(t["speech_end_to_transcript_ms"] for t in turns)
    vad = sorted(t["vad_stop_to_transcript_ms"] for t in turns)
    out = {"url": a.url, "model": a.model, "partial_ms": a.partial_ms, "auto": a.auto,
           "median_speech_end_to_transcript_ms": lat[len(lat) // 2] if lat else None,
           "median_vad_stop_to_transcript_ms": vad[len(vad) // 2] if vad else None,
           "turns": turns, "checks": checks, "failures": sum(not c["ok"] for c in checks)}
    json.dump(out, open(a.report, "w"), indent=1, ensure_ascii=False)
    print(f"median speech end -> transcript {out['median_speech_end_to_transcript_ms']} ms, "
          f"VAD stop -> transcript {out['median_vad_stop_to_transcript_ms']} ms")
    print(f"{len(checks) - out['failures']}/{len(checks)} checks passed; report {a.report}")
    sys.exit(1 if out["failures"] else 0)


if __name__ == "__main__":
    main()
