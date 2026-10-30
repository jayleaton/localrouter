# Voice clip fixtures (`voiceprep.zig` tests)

Raw 16-bit little-endian mono PCM at 16 kHz, from LibriSpeech dev-clean speaker 1272 (CC BY 4.0,
[openslr.org/12](https://www.openslr.org/12), via `hf-internal-testing/librispeech_asr_dummy`):

- `voice-loud.s16`: utterance 1272-128104-0002 through ffmpeg
  `loudnorm=I=-14:TP=-1:LRA=7,alimiter=limit=0.89:level=false`, encoded as MP3 at 192 kb/s and decoded again: a
  loudness-normalised, limited upload.
- `voice-noisy.s16`: utterance 1272-135031-0001 as published, whose background the upstream gate read at 28.9 dB.

The tests carry each file's speech intervals as Silero VAD v6.2.3 (the Whisper worker's pinned detector) found them.
