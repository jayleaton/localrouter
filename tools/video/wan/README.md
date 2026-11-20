# Wan worker (text or image to video)

A `cmd` worker for [Wan 2.2 TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers): text to video, and image
to video with the image as the first frame, at 24 fps, without audio. It runs Diffusers' `WanPipeline` and
`WanImageToVideoPipeline` (one set of modules) on CUDA and fits a 16 GB consumer card. The kind is `video`; the
capabilities are `text_to_video` and `image_to_video`. Setting up a desktop around it: [docs/dev/DESKTOP.md](../../../docs/dev/DESKTOP.md).

| File | Does |
| --- | --- |
| `worker.py` | the worker: `load`, `generate` (writes `video.mp4`), `exit`, soft cancel on SIGUSR1 |
| `fetch.sh` | downloads the pinned weights (no token) into `WEIGHTS.staging`, checks every digest (`pins.json`), then publishes them as `WEIGHTS`; `--venv` also builds the venv |
| `pins.json` | each file's repo, revision, size and digest (`tools/package/verify_assets.py` form) |
| `pyproject.toml`, `uv.lock` | the environment: x86_64 Linux and WSL2, PyTorch cu128 wheels (Blackwell sm_120 included) |
| `test_worker.py` | protocol and unit tests against a fake engine (no weights, no GPU) |
| `test_fetch.py` | `fetch.sh` against a stand-in curl: a file that fails its digest never reaches `WEIGHTS` |
| `config.example.json` | the tool, its limits and its measured bookings, on a desktop |

## Source and licence

`Wan-AI/Wan2.2-TI2V-5B-Diffusers` at `b8fff7315c768468a5333511427288870b2e9635`, ungated: Apache-2.0 (the model card).
The UMT5-XXL text encoder, the 5B transformer and the Wan 2.2 VAE (16x16 spatial, 4x temporal compression) come from
that one repo, 34.2 GB.

## What a request becomes

- **Size.** Width and height in multiples of 32. The example declares the model card's area, 1280 x 704 = 901,120
  pixels, at most 1280 a side, so portrait `704x1280` and landscape `1280x704` are both native; a request without a size
  gets `default_size`. The daemon refuses anything else before the worker loads (`engine.checkVideo`).
- **Length.** `seconds` x 24 frames, up to `max_seconds` (5: the model card's 121 frames). The model makes 4k+1 frames,
  so the worker makes one frame more than the clip and drops the last.
- **First frame.** Cut to the clip's shape (centred) and resized with Lanczos, never stretched. Pad it yourself to keep
  the whole picture.
- **Audio.** None: the tool declares `"audio": false`, so a request that leaves `audio` out gets none, and `audio: true`
  is refused.
- **Steps and guidance.** 50 and 5.0 (the model card's) unless the request or the options say otherwise. The negative
  prompt defaults to the official configs' `sample_neg_prompt`.
- **Output.** `video.mp4`, H.264 yuv420p (CRF 16), no audio track; the daemon adds its machine-generated label.

## Memory

On a 16 GB card the worker keeps one of the three large modules on the GPU at a time:

- The transformer is published in fp32 (20 GB). It loads shard by shard onto the GPU in bf16 and is stored in fp8 e4m3
  (`transformer_storage`, Diffusers' layerwise casting: each layer computes in bf16), 5 GB. Loading it on the host
  instead peaks at 28 GiB of RAM.
- The text encoder (UMT5-XXL, 11.4 GB) stays in bf16. Stored in fp8 its embeddings drift too far (one prompt's worst
  token at cosine 0.62 against bf16).
- `offload: model` moves each module to the GPU when it runs. Diffusers' own chain offloads only the module before
  it in its sequence, so the VAE, used first for the first frame, would stay through denoising, and on a 16 GB card
  WDDM then pages GPU memory and a step takes 3x as long. The worker moves the other two off before any of them runs.
- After every job (done, failed or cancelled) the worker replaces the scheduler, whose state holds the job's tensors,
  moves the modules off and empties the CUDA cache, then logs the GPU memory it still holds (`idle: GPU … allocated, …
  reserved`); it logs it again at exit (`unloaded:`).
- `from_pipe` is not used: it copies the text encoder (30 GiB of host memory).

Measured (RTX 5070 Ti, 704x1280, 5 s): 10.7 GiB peak GPU allocation, 21.6 GiB host memory resident, 26.5 GiB peak, about
25 s a step and 190 s for the VAE decode. `resident_bytes` and `working_bytes` in the example book 22 + 5 GiB.

## FP8 storage against the BF16 reference

Storing the transformer in fp8 is a precision change, so `transformer_storage: bf16` is its reference. fp8 is the
default because, against bf16 at the same seed, both text and image to video keep a mean frame PSNR of at least 20 dB
and a mean SSIM of at least 0.70, and image to video keeps its first frame at 30 dB or more. Measured on the RTX 5070
Ti (704x1280, 2 s, seed 7, 50 steps, guidance 5, the decoded H.264 frames, SSIM on BT.601 luma with an 11x11 Gaussian):

| | PSNR mean / min / first frame | SSIM mean / min / first frame |
| --- | --- | --- |
| Text to video | 22.17 / 20.50 / 22.77 dB | 0.904 / 0.884 / 0.925 |
| Image to video | 28.57 / 23.75 / 48.03 dB | 0.945 / 0.911 / 0.993 |

The scene and composition match; text to video differs in pose, image to video drifts slowly over the clip. fp8 is not
faster (5.0 s a step against 4.9 s for these 2 s clips) but holds 1.9 GB less of the card and 5 to 7 GiB less host memory:
bf16 peaked at 15.8 of the card's 16.3 GB and at 29.3 GiB of the 30 GiB WSL host.

## Options

`steps` (50), `guidance` (5.0), `negative_prompt`, `transformer_storage` (`fp8` or `bf16`; bf16 needs about 16 GB of free
GPU memory), `offload` (`model` or `none`: everything on the GPU, for a card with 24 GB or more), `vae_tiling` (true),
`crf` (16). The daemon's limits, also checked by the worker: `max_pixels`, `max_side`, `size_multiple`,
`default_size` (or `default_width` and `default_height`), `max_seconds`, `fps`, `audio`.

## Tests

```bash
~/wan-venv/bin/python -I tools/video/wan/test_worker.py      # the worker's venv (numpy, Pillow) and ffmpeg; no GPU
python3 -I tools/video/wan/test_fetch.py                     # bash; no network
```
