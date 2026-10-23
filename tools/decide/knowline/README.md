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
