# Speech fixtures

Licensed human-speech (plus synthetic and malformed) test inputs for qualifying the Whisper speech-to-text service.
Audio is never stored in the repository. `fetch.py` downloads each source from a pinned URL, checks its sha256,
converts speech to 16 kHz mono PCM s16le WAV and writes everything to `~/.cache/localrouter-speech/fixtures/`
together with `refs.json` (fixture name -> `text`, `language`, `source`, `license`, `duration_s`, plus a few extra
keys). `manifest.json` (this directory, also copied next to the audio) records every source pin and the sha256 of
every produced file.

```sh
python3 -I tools/speech/fixtures/fetch.py            # about 33 MB of downloads, 17.5 MB of fixtures, ~20 s
python3 -I tools/speech/fixtures/fetch.py --strict   # also fail if a produced sha256 differs from manifest.json
```

Requirements: python3 and ffmpeg (with libmp3lame and libopus for the codec variants). LibriSpeech is distributed as
parquet, so `extract_parquet.py` runs under `uv run --no-project --with pyarrow` (override the binary with `UV`),
or directly when pyarrow is importable. Both scripts run with `python3 -I`. Downloads are untrusted: each source
sits in its own directory under `_downloads/`, archives are parsed in memory only, and nothing from them is executed.
ffmpeg is run with `-map_metadata -1 -fflags +bitexact -flags:a +bitexact` in addition to the standard conversion
command so the output does not embed version strings; the produced hashes below were made with ffmpeg version n9.0.1 Copyright (c) 2000-2026 the FFmpeg developers and a rerun
reproduced them. A different ffmpeg build may resample differently, so a hash difference is a warning
(`--strict` makes it an error). Source sha256 values are the real pins.

## Sources

| Key | URL (pinned) | sha256 | Bytes | License |
|---|---|---|---|---|
| librispeech_parquet | https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/5be91486e11a2d616f4ec5db8d3fd248585ac07a/clean/validation-00000-of-00001.parquet | `4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603` | 9192059 | CC-BY-4.0 |
| fleurs_de_de_tar | https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/de_de/audio/test.tar.gz (Range bytes=0-4194303) | `819343c9eb6243ef830c85dd11ea548daddae2063988b3d9d09ecd52140bea7f` | 4194304 | CC-BY-4.0 |
| fleurs_de_de_tsv | https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/de_de/test.tsv | `82fab72c58a347345675c238f2492bb997105a4ab45d69e2a2b34ed29082fa97` | 559542 | CC-BY-4.0 |
| fleurs_fr_fr_tar | https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/fr_fr/audio/test.tar.gz (Range bytes=0-4194303) | `bba52dc993c30a8f6743de3cbbb2e39f4d5e2cf7558a5e6f707d9ee12ce11f88` | 4194304 | CC-BY-4.0 |
| fleurs_fr_fr_tsv | https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/fr_fr/test.tsv | `5d06d338b242e00786fcf12c4c92008b9f399d5a5c872c91dca90572e7869c0d` | 456972 | CC-BY-4.0 |
| fleurs_es_419_tar | https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/es_419/audio/test.tar.gz (Range bytes=0-4194303) | `c9462abfa84e89983234aef20591a35fcf5edf16845b64a441dcac9db1b87c95` | 4194304 | CC-BY-4.0 |
| fleurs_es_419_tsv | https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/es_419/test.tsv | `d107a93a4f54a18ac25cd470bb4cdadce14fb075b0c1d1542258e274d209ec09` | 599882 | CC-BY-4.0 |
| fleurs_ja_jp_tar | https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/ja_jp/audio/test.tar.gz (Range bytes=0-4194303) | `8d7db0d3ec590ac2f407b7fca0379cd5384682df17b2007a77495d9e7ec00cbe` | 4194304 | CC-BY-4.0 |
| fleurs_ja_jp_tsv | https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/ja_jp/test.tsv | `5dd9643511437414681ad3f23508596c621cdf78978724a09f1f06fefe9d300b` | 361174 | CC-BY-4.0 |
| fleurs_cmn_hans_cn_tar | https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/cmn_hans_cn/audio/test.tar.gz (Range bytes=0-4194303) | `03b964bba16b7f36d425b74f0ea052704ee3d8817e8c2a3d412de952384e581b` | 4194304 | CC-BY-4.0 |
| fleurs_cmn_hans_cn_tsv | https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/cmn_hans_cn/test.tsv | `5734461648f816181d7dab5fc79204b18c4b9bc2cd5138225b25c72d18385d21` | 491487 | CC-BY-4.0 |

Commits: `hf-internal-testing/librispeech_asr_dummy` at `5be91486e11a2d616f4ec5db8d3fd248585ac07a`, `google/fleurs` at
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`. The full FLEURS test shards are 290-580 MB each, so only the leading
4 MiB of each `test.tar.gz` is fetched (HTTP Range, sha256 of that exact prefix is pinned) and the first two complete
members listed in `test.tsv` are used. The FLEURS reference `text` is the raw (cased, punctuated) transcription;
`text_normalized` in `refs.json` is the lower-cased, punctuation-free column.

Note: the Hugging Face LibriSpeech dummy copy holds 73 utterances of the LibriSpeech **dev-clean** split (speaker 1272),
not test-clean (the test-clean archive is 346 MB). LibriSpeech references are the official upper-case, unpunctuated
transcripts.

## Fixtures

| Name | File | Duration (s) | Lang | Kind | License | wav/file sha256 | Size |
|---|---|---|---|---|---|---|---|
| en_librispeech_1272-128104-0000 | en_librispeech_1272-128104-0000.wav | 5.855 | en | LibriSpeech | CC-BY-4.0 | `799f78ed4beb4de7ceae3a809262d4ce242394342ccd1d58cef7d49dbc2def46` | 187404 |
| en_librispeech_1272-128104-0001 | en_librispeech_1272-128104-0001.wav | 4.815 | en | LibriSpeech | CC-BY-4.0 | `473b1bb573e8f7cdcd639c6e318036236190107b66bfe6892286121a92d464c1` | 154124 |
| en_librispeech_1272-135031-0001 | en_librispeech_1272-135031-0001.wav | 11.13 | en | LibriSpeech | CC-BY-4.0 | `f7b291643bbd4c319830bf6cb12bcc8511c8b52af2842cc7d33bde571aa3e1d6` | 356204 |
| en_librispeech_1272-141231-0000 | en_librispeech_1272-141231-0000.wav | 4.65 | en | LibriSpeech | CC-BY-4.0 | `7c205d4d1e9446847b15425be3180b139455ed18cc221461127e69792813bcac` | 148844 |
| long_en_librispeech_60s | long_en_librispeech_60s.wav | 60.82 | en | LibriSpeech | CC-BY-4.0 | `d4f4d9e2f19b201eabb7fbd5daf60b2ee8bf3c18f19084d2ce0912a1736e091f` | 1946284 |
| long_en_librispeech_270s | long_en_librispeech_270s.wav | 280.085 | en | LibriSpeech | CC-BY-4.0 | `eec2d5687d47aedb5038195bd854118c4aa8e40afbe4a7ea702e69e44a7fce9e` | 8962762 |
| de_fleurs_01 | de_fleurs_01.wav | 17.94 | de | FLEURS | CC-BY-4.0 | `cc46d12f9faf44a694e709c8bcf04028e686e6b4c1deefe3ac6dbd821a970630` | 574124 |
| de_fleurs_02 | de_fleurs_02.wav | 11.16 | de | FLEURS | CC-BY-4.0 | `97d77a9d323b903dcc10c608521fa07fb78766d4b65d016ba620caae442dece9` | 357164 |
| fr_fleurs_01 | fr_fleurs_01.wav | 10.2 | fr | FLEURS | CC-BY-4.0 | `f76c0a92ca39d2533aa105ac5794b08f004e15a3ff5ba32f4ba4995f45cd22fd` | 326444 |
| fr_fleurs_02 | fr_fleurs_02.wav | 23.4 | fr | FLEURS | CC-BY-4.0 | `96eaa990ca9b4a8fb0ed3420a395cfd6cc7714c434246d1ed05412d77cfe72c2` | 748844 |
| es_fleurs_01 | es_fleurs_01.wav | 12.84 | es | FLEURS | CC-BY-4.0 | `95df54f13e862069293afca618669316d194c1b4c6601eb808b501c86292b527` | 410924 |
| es_fleurs_02 | es_fleurs_02.wav | 21.36 | es | FLEURS | CC-BY-4.0 | `5ef951507d91ab3af0f4f7b346aa0a098feb98461dde2510b2fccacbe5a49405` | 683564 |
| ja_fleurs_01 | ja_fleurs_01.wav | 10.44 | ja | FLEURS | CC-BY-4.0 | `a4975b9fe15f9340f06e0acdf3b0cc810f522700bdec1b267f85d901987fa29a` | 334124 |
| ja_fleurs_02 | ja_fleurs_02.wav | 14.22 | ja | FLEURS | CC-BY-4.0 | `2d6ca8ff29055d2f885e3e3ea4c99fd7fbd28ebdd405e6dee6f6fd1f1258ed46` | 455084 |
| zh_fleurs_01 | zh_fleurs_01.wav | 10.38 | zh | FLEURS | CC-BY-4.0 | `407400c0a228817cfe9cc95feffe48261e005aaaea828412d6bcbb36915d8f7b` | 332204 |
| zh_fleurs_02 | zh_fleurs_02.wav | 8.28 | zh | FLEURS | CC-BY-4.0 | `d15ddc6cf1c0eaac6277cd7c57d061d06c71fb2e1ecdfb68fea9484a6eb35db8` | 265004 |
| silence_10s | silence_10s.wav | 10.0 | - | generated | CC0-1.0 | `ee7bea4232762775f8fce9b3e27e4d3948c8ac6a45a87ca769f703d6eed0b448` | 320044 |
| noise_10s | noise_10s.wav | 10.0 | - | generated | CC0-1.0 | `f801aab2ee1b90839ca2e63ff816ba834cac7ec1db8b1d2a0d1114d9c825beef` | 320044 |
| speech_in_pink_noise_10db | speech_in_pink_noise_10db.wav | 5.855 | en | LibriSpeech | CC-BY-4.0 | `653888b84aed30d6a0d0c146b1c039e0001d5409d4c44ce246b75a251036bfcc` | 187404 |
| en_44k1_stereo_mp3 | en_44k1_stereo.mp3 | 5.855 | en | LibriSpeech | CC-BY-4.0 | `f0468ea40eecb3ae9906dc6456f047a877ec437698e47b9ce4c734e556cdf65e` | 94895 |
| en_48k_stereo_opus_ogg | en_48k_stereo_opus.ogg | 5.855 | en | LibriSpeech | CC-BY-4.0 | `231653994e35acca1fbd6e852bab70dcfd83519950a464f6933d5269d6686c6a` | 49354 |
| en_44k1_stereo_flac | en_44k1_stereo.flac | 5.855 | en | LibriSpeech | CC-BY-4.0 | `8bcabd248aa96c53d24ce489a7218e48a4eba035624e7c179996c300edea60e2` | 175938 |
| malformed_truncated_wav | malformed_truncated.wav | - | - | malformed | CC0-1.0 | `a4134e418b5c52fcf394492ed64b5fee7d9913042b00e597fc0a1d48e43f45f6` | 93724 |
| malformed_random_wav | malformed_random.wav | - | - | malformed | CC0-1.0 | `9aad15f782db792f84e6d937cde4b4f791cca63f7681bcc844a2b1abf1595065` | 8192 |
| malformed_empty_wav | malformed_empty.wav | - | - | malformed | CC0-1.0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| malformed_notaudio_png | malformed_notaudio.png | - | - | malformed | CC0-1.0 | `bc9854f99dbe38c18f0ae3d55ad8fc7583c03b645fdc7be1ee68524a2888871e` | 463 |

Notes per group:

- `en_librispeech_*`: single utterances. `long_en_librispeech_60s` joins 1272-135031-0000.. until over 60 s;
  `long_en_librispeech_270s` joins 1272-128104-0000.. (speaker 1272, across chapters) until over 270 s. Both use the
  ffmpeg concat demuxer on the converted WAVs; reference = utterance texts joined with spaces (utterance ids are in
  `refs.json`).
- `de/fr/es/ja/zh_fleurs_0N`: FLEURS test utterances (`es` is the es_419 Latin American Spanish shard, `zh` is
  cmn_hans_cn Mandarin, simplified script).
- `silence_10s` (digital zero), `noise_10s` (ffmpeg `anoisesrc` white, amplitude 0.1, seed 42, about -24.8 dBFS RMS):
  reference text is empty, expect empty or near-empty output.
- `speech_in_pink_noise_10db`: 1272-128104-0000 plus `anoisesrc` pink noise (seed 7) scaled by RMS to 10 dB SNR
  (measured 10.03 dB, no clipping), mixed with `amix normalize=0`. The name says 10 dB; the file is about 5.9 s long.
- `en_44k1_stereo.mp3`, `en_48k_stereo_opus.ogg`, `en_44k1_stereo.flac`: 1272-128104-0000 re-encoded and kept in their
  native format (not converted to WAV) to exercise decoding and resampling. Same reference text. FLAC and MP3 are
  44.1 kHz stereo, Opus is 48 kHz stereo (the rate libopus decodes to).
- `malformed_*`: rejection tests only, `"malformed": true` in `refs.json`. Truncated WAV is the first utterance cut at
  half its data with the header unchanged; random bytes use `random.Random(20260101).randbytes(8192)`; the empty file
  has zero bytes; `malformed_notaudio.png` is a valid 16x16 PNG.

## Attribution (CC BY 4.0)

- LibriSpeech: V. Panayotov, G. Chen, D. Povey and S. Khudanpur, "Librispeech: an ASR corpus based on public domain
  audio books", ICASSP 2015. Licensed CC BY 4.0 (https://www.openslr.org/12/; underlying recordings are LibriVox
  public domain). Modified here: converted to 16 kHz mono WAV, concatenated, mixed with noise or re-encoded. The Hugging
  Face copy (`hf-internal-testing/librispeech_asr_dummy`) carries no license metadata of its own, so the license is
  inherited from LibriSpeech.
- FLEURS: A. Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech", 2022
  (arXiv:2205.12446), Google, https://huggingface.co/datasets/google/fleurs. Licensed CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/). Modified here: truncated tarball prefix, converted to 16 kHz mono WAV.
- Generated fixtures (silence, noise, malformed files) contain no third-party material and are CC0-1.0.

Do not redistribute the audio from this repository: it is fetched on demand and kept out of git.
