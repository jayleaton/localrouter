# KnowLine-4B: typed decisions (System One shape)

The decision model of the Jarvis box: [PelaAI/KnowLine-4B-Gen5.5](https://huggingface.co/PelaAI/KnowLine-4B-Gen5.5)
(Apache-2.0, a fine-tune of Qwen3.5-4B) at revision `779944d2cc8ffa19cef386aee83c83bcb967ade2`, served by its authors'
own `/v1/systemone` front end through `serve.py`, as a `decision` tool of the `openai_server` engine
([AGENTS.md](../../../docs/AGENTS.md#decisions-the-system-one-shape)). In Jarvis it runs in shadow and on demand: while it
is loaded, each voice reply logs how it would have routed the turn beside what the chat model did, and nothing acts on
it. It is not kept warm, since its booking would leave under 5 GiB for jobs beside the warm set.

## Why this model

TypeSafe has published no Jev weights. The open "Jev" models are independent reproductions; none is made by TypeSafe.
Checked 2026-10-23 (Decision Index 0.4.1, [jev-decision-index](https://huggingface.co/spaces/multimodalart/jev-decision-index):
"Full" is the maintainers' private run, the measure least open to training on the public sets):

| Model | Maker, relation to Jev | Licence | Format, size | Full / public | Serves here? |
| --- | --- | --- | --- | --- | --- |
| TypeSafe Jev 1.13 (hosted; SideBud's `jev-latest`) | TypeSafe | service terms | API | 59.2 / 58.0 | hosted only |
| autotrust/JEV-27B | AutoTrust: LoRA r16 + 24-slot head distilled from Jev 1.13's outputs | Apache-2.0 | adapter 416 MB on bf16 Qwen3.8-27B (54 GB) | 57.4 / 53.6 | no: TensorFold 1.1.1 has no LoRA, decision head or Qwen scoring hook; a merge is a second 27B copy (lossy at 4 bit, and over the 14 GiB) |
| autotrust/AutoTrust-JEV-27B-v1.1 | AutoTrust, full fine-tune (own data) | Apache-2.0 | 27B with vision, 54 GB | – / 67.3 | no: too large |
| autotrust/JEV-Gemma4-26B-A4B | AutoTrust, LoRA on Gemma 4 26B | Apache-2.0 | 52 GB base | – | no: too large |
| autotrust/JEV-9B | AutoTrust, distilled from Jev 1.13 | Apache-2.0 | 18 GB bf16 | – / 43.3 | no: over 14 GiB at its own precision |
| **PelaAI/KnowLine-4B-Gen5.5** | PelaAI, independent | Apache-2.0 | 9.3 GB bf16 | **56.0** / 66.8 | **yes** (this) |
| tattrongvu/Jeffan-4B-2010 | independent | Apache-2.0 | 8.4 GB | 52.2 / 64.9 | yes (bidirectional mask) |
| StartLux-Decision v1.1 4B | StartLux | CC-BY-NC | 9.3 GB | 54.8 / 67.8 | not for a general product |
| LiquidAI d1-3B | Liquid | LFM licence | 6.2 GB | 31.1 / 49.0 | TensorFold native |
| chaoliangUNSW Jev-Style 2B v3 | independent | Apache-2.0 | 3.8 GB | – / 34.1 | yes |

JEV-27B is the closest copy of hosted Jev, but cannot be served without a second 27B. KnowLine-4B is the best model
that fits beside the warm set, is licensed for general use, and answers the whole System One shape (choice, noul,
score, up to 255 options and 64 questions). Its training data was its authors' own; it states no relation to TypeSafe.

## Serving

`serve.py` runs `knowline_server.py` and `knowline_layout.py` from the weights directory (their SHA-256 are checked
first against the reviewed files) with the transformers backend in bf16, the shipped precision, so prompts, read-twice
and the label readout are the authors'. Two changes, both measured on DGX-01:

- **Only the last position's logits**, which is the one read: the stock backend computes the whole vocabulary at every
  prompt position.
- **Its own loader**: the model is built on the GPU and each checkpoint tensor is read and copied in alone.
  transformers' loader held up to 7 GB of host copies until the load ended (14.3 GiB at load against 11.5 GiB serving;
  MemAvailable down to 5.9 GiB beside the warm set). The answers on the 80-request probe are byte-identical
  (`KNOWLINE_STOCK_LOAD=1` runs the stock loader for that comparison), and the load takes 17 s instead of 58 s.

Linear attention runs transformers' reference PyTorch path (no flash-linear-attention build for this stack): correct,
and what makes a long state slow (below).

## Qualification on DGX-01 (2026-10-23, beside the warm Jarvis set)

| Measure | Result |
| --- | --- |
| Voice routing (1 choice of 3 to 7 options, a short state) | 192 ms median, 207 ms p90 (60 requests) |
| Three questions (choice, noul, score) | 504 ms median |
| A 14 KB state, read twice | 2.0 s median |
| A 16 KB state (the request cap; over 3,000 tokens, read once) | 1.15 s median |
| SideBud replay, HEAD (83 cases) | 71/83 correct, 0 wrong direct actions, 3.45 s a tree request |
| SideBud replay, c6c6140 (71 cases, nearest ADR 0018's set) | 61/71 correct, 0 wrong direct actions |
| Hosted Jev, ADR 0018 (63 / 69 cases) | 62/63 tree, 68/69 with screen phrasings; about 330 ms a tree request |
| Load | 17 s; peak 11.9 GiB (GPU and host) |
| Serving peak | 12.1 GiB (13.02 GB) on a 14 KB state read twice (about 6,000 tokens: the largest prompt, since a longer state is read once); booked 12.4 GiB (peak + 256 MiB) |

The replay is SideBud's own (`jev-replay.ts`'s fixed state, tree questions and act rules, ported to Python; the
installed-apps list of the Mac is a fixed stand-in): the 63- and 69-case sets were never committed, so the nearest
recoverable sets stand in. KnowLine routes 85% of them right against hosted Jev's 98%, never to a wrong direct action,
at about ten times the latency on SideBud's 15-question tree. Good enough to watch in shadow; not to act on.

## Setup

```sh
tools/decide/knowline/fetch.sh    # weights at the pinned revision (SHA256SUMS checked) and the venv from uv.lock
```

## macOS

On a Mac (Apple silicon) the same `openai_server` decision tool runs [`macos/serve_mlx.py`](macos/serve_mlx.py), not
`serve.py`. It uses the authors' unchanged front end, the same pinned revision and files, the same hashes and the same
read-twice and label readout. The forward pass is mlx-lm's Qwen3.5 text model on Metal instead of transformers on CUDA,
again with only the last position's logits and a float32 log-softmax. This is a deliberate numerical change
([CONTRIBUTING](../../../CONTRIBUTING.md#engine-correctness)), so its reference baseline is recorded below; it does
not claim parity with the Spark's bits. Text only, as on the Spark: a request with pictures gets a 422.

- **Weights:** the shipped bf16. mlx-lm folds Qwen3.5's `1 + w` RMSNorm gains into bf16 weights, a step of 1/128
  near 1. That doubled the probability error against float32, so the launcher keeps those gains in float32 and casts
  the norm's output back, as transformers does.
- **Setup:** `tools/decide/knowline/macos/fetch.sh` puts the weights in the Spark's directory (checked against
  SHA256SUMS) and builds the venv `$J/venvs/decide-mlx` from [`macos/uv.lock`](macos/uv.lock) (mlx 0.32.4, mlx-lm
  0.32.2, transformers 5.19.0 for the tokenizer, no torch). The pins are in [`macos/pins-macos.json`](macos/pins-macos.json).
- **Load:** Python's `HTTPServer` looks up the name of 127.0.0.1 when it binds. On macOS that lookup goes to mDNS and
  took 35 of the 37 seconds the load first measured, so the launcher skips it. One forward pass at load builds the
  Metal kernels; without it the first decision took 2.4 s.

### Baseline (M5 Ultra, 96 GiB, macOS 27.2, 2026-10-30)

The corpus is [`macos/corpus.json`](macos/corpus.json): 50 fixed requests and 54 questions, with 53 expected answers.
It covers voice routing over the Jarvis voice tools, follow-ups that need context, the smoke test's route,
desktop-helper steps, nouls, scores, multi-question requests, a chat-turn state, 40 options, Chinese, and long states
on each side of the 3,000-token read-twice cut. [`macos/bench.py`](macos/bench.py) runs it and compares two runs.

No Spark output for this corpus is committed (the replay sets above never were), so the reference was built on the same
Mac. It is the authors' **unmodified** `knowline_server.py --backend hf` on PyTorch 2.13 (MPS), in its default bf16
and with the dtype set to float32 as a precision anchor. Every run answered identically on each pass (deterministic).

| Compared with float32 (transformers) | Same top answer | max \|Δp\| median / p90 / max | KL mean / max | Expected answers right |
| --- | --- | --- | --- | --- |
| transformers bf16 (the Spark's stack, on MPS) | 54/54 | 0.0003 / 0.0022 / 0.0132 | 0.00004 / 0.00047 | 52/53 |
| **MLX bf16 (served)** | **54/54** | **0.0002 / 0.0044 / 0.0132** | **0.00005 / 0.00047** | **52/53** |
| MLX bf16, mlx-lm's bf16 norm fold | 54/54 | 0.0004 / 0.0044 / 0.0261 | 0.00009 / 0.00143 | 52/53 |
| MLX 5-bit (`Liamariel/knowline-4b-gen5.5-mlx-q5@053081f7`) | 54/54 | 0.0017 / 0.0223 / 0.0902 | 0.00220 / 0.02238 | 52/53 |

MLX bf16 against transformers bf16: 54/54 same answers, max |Δp| median 0.0002, p90 0.0026, max 0.0101. The 5-bit
quant gives the same top answers, but its probabilities move up to 0.09 (a noul from 0.562 to 0.651, a score's top
level from 0.578 to 0.509). That is about 20 times bf16's error, enough to cross a caller's threshold. It is no faster,
and it saves 5 GiB that a 96 GiB Mac does not need. **bf16 is served; the quant is not.** The one wrong expected answer
is the same in every run: "buy milk sometime this week" scores "somewhat urgent" (0.50) against "not urgent".

| bf16 on Metal, warm (the third pass) | Median latency (a request, all its questions) |
| --- | --- |
| Voice route, 9 options, about 520 tokens read twice (22 requests) | 73 ms |
| One noul / one score / one choice | 40 / 42 / 45 ms |
| Three questions (choice, noul, score) | 119 ms |
| 40 options | 64 ms |
| A long state read twice (4,638 tokens) / read once (3,464 tokens) | 488 / 369 ms |
| First request after load | 76 ms |
| The 5-bit quant, same corpus | 74 ms voice route, 539 ms long read twice |
| transformers bf16 on MPS (reference only, one pass) | 1,211 ms voice route, 9.9 s long read twice |

| Memory and load (physical footprint, `proc_pid_rusage`) | bf16 | 5-bit |
| --- | --- | --- |
| Load to healthy (weights in the file cache) | 1.3 s | 1.2 s |
| Loaded | 8.45 GiB | 3.36 GiB |
| Peak over the corpus | 9.74 GiB | 4.60 GiB |
| Peak on the longest prompt a request can make (5,990 tokens: a state just inside the read-twice cut) | 9.70 GiB | – |
| **Booking (`resident_bytes`: peak + 256 MiB)** | **10,724,056,240 B (9.99 GiB)** | – |

Across repeated requests the footprint stays flat (the test checks that it grows by less than 256 MiB). These are
measurements on this Mac only. The Spark figures above come from different hardware and a different stack, so the two
sets are not comparable.

```sh
tools/decide/knowline/macos/fetch.sh
# the server's tests: model-free, plus the memory test on the real weights (KNOWLINE_WEIGHTS)
KNOWLINE_WEIGHTS=~/jarvis/models/decide/knowline-4b-gen5.5 ~/jarvis/venvs/decide-mlx/bin/python -I -m unittest \
  discover -s tools/decide/knowline/macos -p 'test_*.py'
# the baseline: a run (starts and stops the server), then a comparison
~/jarvis/venvs/decide-mlx/bin/python -I tools/decide/knowline/macos/bench.py run /tmp/bf16.json \
  --front-end ~/jarvis/models/decide/knowline-4b-gen5.5
python3 -I tools/decide/knowline/macos/bench.py compare REF.json /tmp/bf16.json
# a Spark reference for the same corpus, from its running server:
python3 -I tools/decide/knowline/macos/bench.py run /tmp/spark.json --url http://127.0.0.1:PORT --model knowline-4b
```
