# Whisper on AMD GPUs (PyTorch ROCm)

The same worker as the Spark (`../worker.py`, `../whisper_core.py`) with a venv locked to PyTorch's ROCm 7.2 wheels
(torch 2.13.0+rocm7.2, the release the other workers pin). On ROCm `torch.cuda` is the HIP device, so the tool's
`options.device` stays `"cuda"` (or `"auto"`). PyTorch's index publishes no digest for the torch wheel, so its sha256
(`2ad8066c…`, 6227849000 bytes, hashed from download-r2.pytorch.org on 2026-11-16) was added to `uv.lock` by hand;
`uv sync --locked` refuses any other bytes, and `tools/install/check_hashes.py` (CI) refuses a lock with an unhashed
artifact.

```sh
UV_PROJECT_ENVIRONMENT=$J/venvs/whisper-rocm uv sync --locked --python 3.12 --project tools/speech/whisper/rocm
python3 -I tools/speech/whisper/fetch_model.py --root $J/models/whisper
```

Tool: `"cmd": ["$J/venvs/whisper-rocm/bin/python", "tools/speech/whisper/worker.py"]`, options
`{"device": "cuda", "dtype": "float16", "attn": "sdpa", "vad": "silero", "vad_model": ...}` (or `"vad": "off"`).

Measured on a Radeon 8060S (gfx1151), 2026-11-15: 73 LibriSpeech dev-clean clips (481 s), WER 4.09% (honorifics
normalized), 17x real time, 373 ms median per clip, 1.6 GiB allocated (2.3 GiB GTT); 9.8 s cold load through the daemon,
0.5 s per short clip warm.
