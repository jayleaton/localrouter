# Accepted graphs-decode baselines

`graphs-<gpu>.json`: each fixture's codec-token and PCM sha256 for `lm_decode: "graphs"`. `qualify.py gate --lm-decode
graphs` must reproduce these bits exactly. Each file also records the runtime that made them: torch, transformers,
kittenml, CUDA and the GPU. A file appears here only once an operator accepts a run with `--accept-baseline` after
listening to its audio or passing `asr_check.py` against the eager audio; that run's marker names the file. See
`docs/dev/KITTEN-TTS2.md`, Numerics and GB10.
