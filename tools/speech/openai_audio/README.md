# openai_audio: a `cmd` worker for any OpenAI-compatible audio server

`worker.py` lets LocalRouter use a model's own speech server (anything that serves OpenAI's
`POST /v1/audio/speech` and/or `POST /v1/audio/transcriptions`) as a `speech` or `transcription` tool, the way the
compiled-in `openai_server` engine does for chat. It is Python standard library only, so it runs under the system
`python3 -I` with no venv.

- **load**: starts `options.server` (an argv; `{port}`, `{id}` and `{weights}` are filled in) in its own process group
  on a loopback port (`options.port`, else a free one) and waits until `options.health` (default `/v1/models`) answers
  200. Without `options.server` it uses `options.url`, a server you run yourself.
- **speech**: asks the server for raw PCM (`response_format: "pcm"`) and writes `speech.wav` as it arrives: a
  placeholder header, the samples, then the real sizes. With `"stream": true` in the tool's options the daemon forwards
  the audio as it grows (time to first audio is the server's). The rate is the server's `x-sample-rate` header, else
  `options.sample_rate` (24000). Seeds are folded to 32 bits.
- **transcription**: posts `audio.wav` (16 kHz mono, prepared by the daemon) as multipart with `language` and `prompt`,
  and writes `transcript.json` (`text`, `language`, `duration`, `segments`; one segment when the server gives none).
- **exit / unload**: SIGTERM to the server's process group, progress lines while it stops (so the daemon does not load
  anything into memory that is still held), SIGKILL after `options.stop_timeout_s` (15).
- A server 4xx becomes `invalid_request_error` (400), a 503 or an unreachable server `service_unavailable` (503).
  Voice cloning and references are refused.

Declare the model's limits in `options` as for any speech tool (`voices`, `default_voice`, `min_speed`/`max_speed`,
`max_input_chars`, `max_audio_s`, `languages`), and its measured `resident_bytes`.

```json
{"id": "qwen3-tts-customvoice", "kind": "speech", "capabilities": ["text_to_speech"],
 "cmd": ["/usr/bin/python3", "-I", "tools/speech/openai_audio/worker.py"], "weights": "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
 "resident_bytes": 6979321856, "keep_loaded": true, "lane": "realtime",
 "options": {"server": ["tools/gufo/serve.sh", "tts", "{port}", "-v", "{weights}:/m:ro", "--",
                        "gufo", "serve", "tts", "--model", "/m", "--served-model-name", "{id}"],
             "port": 8091, "health": "/v1/audio/voices", "stream": true, "sample_rate": 24000,
             "voices": ["ryan", "serena"], "default_voice": "serena", "min_speed": 1, "max_speed": 1}}
```

Tests (no GPU, a fake server): `python3 -I tools/speech/openai_audio/test_worker.py`.
