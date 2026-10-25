# Two Sparks: one LocalRouter on two GB10s (design, for decision)

**Status: design only. Nothing here is built.** Written 2026-10-25 against `main` `bd233f9`. Every number is tagged:
**M** is measured (the source is named), **E** is estimated (the method is shown). Nothing new ran on a GPU for this
page. On DGX-01 it read only `/proc/meminfo`, `/v1/tools` and the NIC state. dgx-02 was not touched.

JJ's sketch is this. Spark 1 runs a larger chat model, perhaps with its KV cache spilled to Spark 2. Spark 2 runs
everything smaller: voice (Drama 3, Whisper), image, video and the decision model (KnowLine).

## The answer in short

| Question | Answer |
| --- | --- |
| Best placement | **(a): the chat model alone on Spark 1, everything else on Spark 2.** It removes the measured chat/voice GPU contention. A video job no longer evicts chat, and voice no longer loses the GPU to it. The link only carries HTTP, so it needs no RDMA |
| Tensor/pipeline split (b) | Only worth it for a model too big for one Spark (GLM-5.3-Flash, about 164 GiB). It then fills both Sparks: about 88 GiB on rank 0 and as much on rank 1. That leaves no room for image, video or the decision model on Spark 2, and it puts chat decode back on the voice GPU. TensorFold has no pipeline mode |
| KV cache on Spark 2 (c) | **Not for decoding.** The link is about 1/11 of local memory bandwidth at best, and a 30k-token context would cap decode near 8 tok/s. It could work as a cold tier for parked conversations, but TensorFold has nothing for it and Spark 1 has room for a big local prompt cache in (a). Not planned |
| Voice | Crossing the link costs under 1 ms a turn (E). Losing the contention is worth far more than that: Drama 3 runs at RTF 0.66 alone against 1.47–1.52 while chat decodes (M) |
| Daemons | **One daemon per Spark, with one front** (Spark 2) that serves the single OpenAI, MCP and Realtime surface. Memory admission, the GPU lock and worker processes are node-local in the code today, and they should stay that way |
| First step | Move today's qualified Qwen3.8-27B to Spark 1 unchanged (no model risk), measure, then choose a bigger model |
| Blocker | **A second Spark.** Only DGX-01 is available; dgx-02 is off-limits |

## 1. What we start from

| | Value | Tag |
| --- | --- | --- |
| GB10 memory | 128 GB LPDDR5x, unified (CPU and GPU), 273 GB/s | public spec |
| DGX-01 `MemTotal` | 121.7 GiB (127,598,748 kB) | M, 2026-10-25 |
| DGX-01 free at idle | about 114 GiB | M, [DRAMA3.md](DRAMA3.md), 2026-10-21 |
| Daemon `reserve_bytes` | 3 GiB (`deploy/jarvis/config.json.in`) | config |
| Admission budget per Spark | about **111 GiB** (114 free less the 3 GiB reserve) | E from the two figures above |
| Unmeasured overhead | 8 GiB per process until that process's peak is measured, then dropped (the GB10 memory policy) | policy |
| Link | 2 × ConnectX-7 QSFP at 200,000 Mb/s each, MTU 9000, on `192.168.100.10/24` and `192.168.101.10/24`, with RoCE devices `rocep1s0f1` and `roceP2p1s0f1` up | M, DGX-01 sysfs, 2026-10-25 |
| Link ceiling | 200 Gb/s in total, not per port: the SoC gives the NIC two PCIe 5.0 x4 links (about 25 GB/s line rate) | public (StorageReview, NVIDIA forum) |
| Link as measured by others | NCCL bus bandwidth 10–27 GB/s depending on tuning; MTU 1500 drops it to about 3 GB/s | public user claims; not measured here |
| LAN | `enP7s7` at 2.5 Gb/s | M, DGX-01 sysfs |

### Today's memory bookings on DGX-01 (one Spark)

These are the `resident_bytes` the scheduler admits against. The table is live `/v1/tools` on 2026-10-25 (M), and
each booking is the model's measured peak plus a margin.

| Model | Role | Booked | Priority | Kept warm |
| --- | --- | ---: | ---: | --- |
| `qwen3.8-27b` (MLX 4-bit + DFlash2) | chat | 22.13 GiB | 100 | yes |
| `fishaudio/drama-3` (NVFP4 pruned) | speech | 20.56 GiB | 95 | yes |
| `whisper-large-v3-turbo` | transcription | 3.75 GiB booked (3.59 reported) | 90 | yes |
| `qwen-image-2.1-turbo` (FP8) | image | 41.55 GiB | 10 | yes |
| `knowline-4b` (bf16) | decision | 12.38 GiB | 15 | yes, `spare_room_only` |
| `minimax-h3` (NVFP4) | video | 52.31 GiB when loaded | 0 | no |
| **Warm set** | | **100.4 GiB** | | live `MemAvailable` 19.4 GiB |

The full Drama 3 checkpoint books 39.25 GiB (M, `deploy/jarvis/env.sh`). Jarvis switched to the lossy NVFP4 weights
on 2026-10-24 to make room for KnowLine. On two Sparks that reason goes away (see section 3).

### Measured contention on one Spark (why a second Spark helps)

| Measured on DGX-01 | Alone | While the other runs | Source |
| --- | --- | --- | --- |
| Chat decode, prose | 34–39 tok/s | 14–19 tok/s while Drama 3 speaks | [CHAT.md](CHAT.md) |
| Chat decode, code | 70–87 tok/s | 34–38 tok/s | [CHAT.md](CHAT.md) |
| Chat TTFT (cached prefix) | 0.14–0.16 s | 0.32–0.55 s | [CHAT.md](CHAT.md) |
| Drama 3 real-time factor | 0.66 | 1.47–1.52 while chat decodes (slower than playback) | [CHAT.md](CHAT.md), [VOICE.md](VOICE.md) |
| A long reply's start | held 200 ms | waits 2.9–5.9 s for chat to finish writing (the start hold) | [VOICE.md](VOICE.md) `client_check` |
| A video job | | evicts image, then chat; chat reloads afterwards (12.5–29.9 s cold load) | [CHAT.md](CHAT.md) |

The GPU time-slices between processes, and TensorFold has no priority between them. A second GPU is the only way to
remove this contention without changing models.

## 2. Candidate big chat models

**Lossless** here means the publisher's own release precision (BF16 or FP8, or MXFP4 where that is how it ships).
Anything quantised or pruned by someone else is **lossy**, and by the standing rule it needs JJ's sign-off. Note that
today's Qwen3.8-27B is itself a 4-bit MLX checkpoint, which JJ accepted.

The decode estimates use a measured efficiency. Qwen3.8-27B without the drafter decodes at 13.3 tok/s (M) and reads
16.05 GB of weights per token, so it uses 213 GB/s of the 273: **η ≈ 0.78**. That gives:

- **Undrafted estimate:** η × 273 GB/s ÷ bytes read per token.
- **Drafted estimate:** the undrafted figure × 2.6–2.9, the measured DFlash2 gain on prose (13.3 → 35–39).

All figures are for one stream at short context.

| Model | Size and format | Fits | Decode (one stream) | TTFT / prefill | Engine on GB10 |
| --- | --- | --- | --- | --- | --- |
| **Qwen3.8-27B** (today) | 16.05 GB MLX 4-bit + 3.85 GB DFlash2 drafter. **Lossy, accepted** | One Spark: 22.13 GiB booked (M) | 34–39 prose, 70–87 code (M) | 0.14–0.16 s cached (M); a 30k-token cold prompt takes 120 s (M) | TensorFold 1.1.1 native, **qualified on DGX-01** |
| Qwen3.8-27B BF16 | about 54 GB. **Lossless** | One Spark | about 4 undrafted, 10–11 drafted (E). **Too slow for voice** | prefill (E) unknown | native CUDA support for BF16 unverified |
| Qwen3.8-27B FP8 (if published) | about 27 GB. **Lossless if official** | One Spark | about 8 undrafted, 20–23 drafted (E) | | FP8 path unverified |
| Qwen3.6-35B-A3B | 20.9 GB MLX 4-bit (lossy) or about 70 GB BF16 (lossless, E) | One Spark either way | 4-bit: 86–88 tok/s serial (M, TensorFold recipe, one Spark). BF16: about 35 undrafted (E) | | **Python engine 0.6.6 only**; the native binary has no family. MoE with about 3B active: wider but not obviously smarter than the dense 27B |
| **Qwen3.8-Flash-Next** | 81 GB MLX 4-bit, or 52 GB EXL3 3.05 bpw. **Lossy**. BF16 would be about 300 GB (E) and does not fit even two Sparks | One Spark. KV window about 74k tokens (4-bit) or 262k (EXL3) | 4-bit: 62.7 chat, 76.5 code. EXL3: 59.4 / 80.8 (M, TensorFold recipe, one Spark, sampled) | prefill 2,300–2,500 tok/s for 4-bit, 930–970 for EXL3 (M, same recipe) | TensorFold. **Which engine serves it on CUDA must be checked against our pin.** On vLLM with TP=2 across two Sparks: 50 against 37 tok/s (public, blazux #52) |
| **GLM-5.3-Flash** | about 164 GiB EXL3 4 bpw (experts) with BF16 elsewhere, or about 170 GiB MLX 4-bit. **Lossy** | **Two Sparks only** (TP=2): about 80 GiB of weights on each; rank 0 estimates 88 GiB and wants 110 GiB free at start | EXL3: 60.4 prose, 114.7 structured (M, prior art below). MLX: 50.9 at 32k depth (M, TensorFold recipe) | 170 ms TTFT; prefill about 1,950 tok/s up to 65k (M, prior art) | TensorFold **v0.5.0 + 53 patches**, Python engine in NVIDIA's PyTorch container. Not our native pin |
| Qwen3-235B-A22B | about 120–130 GB NVFP4 (E). **Lossy** | Two Sparks (TP=2) | 15–26 tok/s (public user claims, TRT-LLM and vLLM) | | not TensorFold: a new engine stack for us |
| DeepSeek-V4-Flash | about 151 GiB 4-bit | | | | TensorFold serves it only on a 256 GB Mac: "CUDA on two DGX Sparks … not in this family yet". Out |

### Prior art: GLM-5.3-Flash EXL3 on two Sparks

The "Spark Local" agent profile (in the T3 settings and `~/.config/opencode/agent/spark-local.md`) points at
`https://dgx-01.prawn-penny.ts.net/v1`, model `GLM-5.3-Flash-EXL3`. It is served by
`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold` (local clone `~/.cache/mia-ro/…`, `1f3d909`):

- **Topology.** `tensorfold serve … --tp 2 --rank 0|1 --master <head> --master-port 29551`. Rank 0 and the API run on
  the head; rank 1 runs on the worker, started over ssh.
- **Link.** A direct ConnectX-7 cable with a private subnet and RoCE v2. Small all-gathers (up to 512 KiB) are one-shot
  RDMA writes (patch 0006); NCCL carries the rest.
- **Measured effects** (its README): RoCE gathers add 6% to decode (prose 44.2 → 46.9 tok/s), and two cabled rails make
  a prompt chunk's gather about 1.8× faster.
- **Memory.** About 80 GiB of weights per Spark. `MemAvailable` at idle is 7.8 GiB on rank 0 and 10.8 GiB on rank 1.
  The server uses about 10 GiB beyond its own estimate at peak.
- **Operations.** Loads take 2–6 minutes. The worker keeps its own 166 GiB copy of the checkpoint, copied over the link.
- **Status.** That deployment used DGX-01 plus a second Spark. DGX-01 now runs Jarvis, so the profile is stale (its
  reachability was not checked).

What it shows: **TP=2 over CX-7 works and is fast, but a GLM-class model owns both Sparks.**

## 3. Placement options and per-node memory

All three options start from the same budget per Spark: about 111 GiB (E, from section 1). Spark 2 is assumed to
match DGX-01; it has not been measured.

### (a) Chat on Spark 1, everything else on Spark 2 (recommended)

**Spark 2** keeps the same bookings as today (M), less chat:

| State | Loaded | Booked | Headroom |
| --- | --- | ---: | ---: |
| Warm (NVFP4 Drama 3) | Drama 3 20.56, Whisper 3.75, image 41.55, KnowLine 12.38 | 78.2 GiB | 32.8 GiB |
| A video job | image evicted (priority 10), then H3 52.31 | 89.0 GiB | 22.0 GiB. **Voice and decisions stay warm** |
| Spark 1 down: fallback chat | image evicted, 27B 22.13 loaded on Spark 2 | 58.8 GiB | 52 GiB |
| Warm with the **full Drama 3** (39.25) | as above | 96.9 GiB | 14.1 GiB |
| Full Drama 3 plus a video job | image evicted, H3 loaded | 107.7 GiB | 3.3 GiB. Fits, but tight; no fallback chat during video |

**Spark 1** holds the chat model alone: 111 GiB, less 8 GiB until the new model's peak is measured, so about
**103 GiB** for weights, KV and prompt cache.

| Spark 1 model | Booked (E) | Left for KV / prompt cache (E) |
| --- | ---: | ---: |
| Qwen3.8-27B as today | 22.13 (M) | about 89 GiB. Raise `--prompt-cache-gib` from 3 to, say, 32 GiB: about 440k tokens at 76 KiB a token (M, per-token figure), roughly 14 parked 30k conversations |
| Qwen3.8-Flash-Next MLX 4-bit | about 76 GiB of weights | about 27 GiB |
| Qwen3.8-Flash-Next EXL3 3.05 bpw | about 48 GiB | about 55 GiB, the full 262k window |
| Qwen3.6-35B-A3B BF16 | about 65 GiB | about 38 GiB |

A variant puts image and video on Spark 1 beside a small chat model, leaving Spark 2 to voice and decisions. It frees
Spark 2 entirely for voice, but a video job evicts chat again, which is today's problem. Only consider it if the chat
model stays at 27B.

### (b) The chat model split across both Sparks

- **Tensor parallel (TP=2)** is what TensorFold has: two ranks, NCCL plus optional RoCE gathers. The native CUDA build
  serves one GPU; two ranks are the Python engine's (M, TensorFold `docs/recipes/cuda.md`).
- **Pipeline parallel** does not exist in TensorFold ("needs a pipeline mode", M, its docs). Pipeline mode would allow
  an uneven split. However, it does not speed up a single stream: stages run one after another, so it only adds
  capacity.
- **TP needs equal shards.** Spark 2's leftover memory therefore sets the model size:

| TP=2 model | Per Spark (E) | Spark 2 left after its shard (of 111) | What else fits on Spark 2 |
| --- | ---: | ---: | --- |
| GLM-5.3-Flash EXL3 | about 88 (rank 0 estimate, M) | about 23 GiB, or about 15 under the 8 GiB rule | Barely Drama 3 NVFP4 + Whisper (24.3), and only by trimming the KV pool. **No image, video or decision model** |
| Flash Next MLX 4-bit | about 38 + about 8 of KV | about 65 GiB, or about 57 under the 8 GiB rule | Voice + KnowLine (36.7); **image (41.55) does not fit with them**; video means stopping chat |

The costs of (b), beyond memory:

- **Chat decode is back on Spark 2's GPU every token.** Today's contention (chat at about 40%, Drama 3 slower than
  playback) returns on Spark 2.
- **TP ranks run in lockstep.** An image job on Spark 2 would slow chat on Spark 1 too.
- **One more failure domain.** Either Spark down means no big chat.
- **A different engine stack.** It runs TensorFold 0.5/0.6 Python in a container, or vLLM, not the native pin
  LocalRouter qualifies.

The gain is measured by others: TP=2 gives Flash Next 1.38× single-stream decode and 2.6× KV pool on vLLM (public).

**Use (b) only if JJ wants a GLM-class model more than image and video on the same box.** In that case Spark 2 becomes
"rank 1 + voice", and image/video run on demand by stopping chat.

### (c) KV cache on Spark 2 over RDMA

| | Figure | Tag |
| --- | --- | --- |
| Local memory | 273 GB/s | spec |
| Link | 25 GB/s line rate; 10–27 GB/s NCCL bus bandwidth | spec / public |
| Ratio | the link is 1/11 to 1/27 of local | E |
| 27B prompt cache per token | about 76 KiB (a 30k conversation is about 2 GiB) | M, [CHAT.md](CHAT.md) |

- **Decode against a remote KV.** Each new token's attention reads the whole KV. At 30k tokens that is about 2.2 GiB a
  token over about 20 GB/s: roughly 0.11 s a token, about **8 tok/s** at best (E). At 8k tokens it adds about 30 ms
  a token, which halves today's 26 ms decode step (E). **Not realistic** on 200 Gb/s.
- **As a cold tier.** Parking an idle conversation's prefix on Spark 2 and fetching it on its next turn costs about
  0.1 s for 2 GiB (E), against 120 s to prefill 30k tokens again (M). That is real value, but:
  - TensorFold has no remote KV, disaggregated prefill or KV offload (M, repository search), so it would be engine work.
  - In (a), Spark 1 has 27–89 GiB of its own for a larger local prompt cache.
  - Spark 2's free memory is the video job's headroom.

Revisit (c) only if the chosen Spark 1 model leaves under about 10 GiB for its prompt cache.

## 4. Voice latency across the link

Today's figures (M, [VOICE.md](VOICE.md), DGX-01 2026-10-21, `silence_duration_ms` 500):

| Stage | Today |
| --- | --- |
| First audio after speech ends | **2.20 s median, 2.95 s p90** |
| VAD closes the turn | 0.42–0.51 s |
| Whisper | 0.8–1.5 s more |
| Chat's first token | 0.2–0.5 s more |
| First audio after the first clause | 0.3–0.6 s more |
| Drama 3 first audio, warm | p50 0.142 s |

The wait timer is the user's: 0.3–5 s, 0.8 s by default in the web client. Early STT starts after 150 ms of quiet.

In (a) the front is Spark 2, so VAD, early STT, Whisper, Drama 3, the sentence chunker and the WebSocket stay local.
Only the chat request crosses to Spark 1:

| Added by the link | Estimate |
| --- | --- |
| TCP round trip over CX-7 | 20–50 µs (E; to measure) |
| TCP round trip over the 2.5 GbE LAN | 0.2–0.5 ms (E) |
| Per turn: a prime, the request and its first SSE event, at most about 3 round trips before the first token | **under 1–2 ms** on either path (E) |
| One more proxy hop (front daemon → Spark 1 daemon → TensorFold) | today's daemon hop costs nothing measurable: 39.0 against 38.9 tok/s direct (M, [CHAT.md](CHAT.md)) |
| Bandwidth: SSE at about 200 B a token × 60–90 tok/s | about 20 KB/s. Even the LAN carries it; **(a) needs no RDMA** |

What it saves (E, from the measured contention in section 1):

- Drama 3 runs at its alone rate (RTF 0.66) during chat decode, so a long reply's start hold releases at 200 ms
  instead of waiting 2.9–5.9 s.
- Chat decodes at its alone rate during speech.
- A video or image job no longer touches chat.

Image jobs on Spark 2 still share its GPU with Drama 3, as they do today.

**Expected:** first audio no worse than today's 2.20 s median, and much better on long replies. The wait timer and
Whisper still dominate the short ones.

A faster Spark 1 model shortens the chat stages further: about 10 tokens to the first clause takes about 0.13 s at 76
tok/s against about 0.26 s at 39 (E). This holds only if its prompt cache keeps the voice history warm. Uncached, a
600-token history at 2,400 tok/s costs 0.25 s (E).

In (b) every chat token also needs Spark 2's GPU, and the ranks wait for each other. Voice gets today's contention
back, and possibly worse. In (c) nothing on the voice path changes except that decode gets slower.

## 5. Scheduler, routing and health

### One daemon or two?

**Two node daemons, one front.** The daemon's core invariants are node-local in today's code:

- **Admission** reads this machine's `/proc/meminfo` (`src/sched/memory.zig:8`, `MemAvailable` less `reserve_bytes`).
- **The host GPU lock** is a local `flock` (`gpu_lock`, [ARCHITECTURE.md](../ARCHITECTURE.md#the-hosts-gpu-lock)).
- **Each model is a local worker process.** Unloading is process exit.
- **A model's own server** is reached on loopback only (`src/tool/upstream.zig:1`).

One daemon driving workers on two hosts would need a remote meminfo, remote spawn, a remote lock and remote process
supervision. That is a node agent, and a LocalRouter daemon already is one. Memory on a GB10 is unified per node, so
admission per node is also the correct model, not merely the convenient one.

```
 clients ─ OpenAI /v1, /mcp, /v1/realtime, /voice ─► front daemon (Spark 2) ── HTTP over CX-7 (LAN as 2nd path) ──► node daemon (Spark 1)
                                                     │ local: Drama 3, Whisper,                                       │ local: the chat model
                                                     │ image, video, KnowLine,                                        │ (and its prompt cache)
                                                     │ fallback chat                                                  │
                                                     └ remote tools: "node": "spark-1" ──────────────────────────────┘
```

### What changes

| Area | Change |
| --- | --- |
| Tool config | A tool may name a `node` (or use a new compiled-in `remote` engine). The front books **0 bytes** for it locally, because the peer admits it against its own memory. The front forwards the job to the peer's same route and streams the reply into the job directory as `upstream.zig` does now. `upstream.zig` gains a non-loopback host |
| Error mapping | The peer's 4xx is a 400 and its 503 is a 503, as today. An unreachable peer is a 503 naming the node |
| Lanes | The front's `chat` lane carries remote chat. The peer runs its own lanes. Cancelling closes the forwarded connection, so the peer cancels as it does now |
| Discovery | The front merges the peer's `/v1/models`, `/v1/tools` and `/v1/capabilities` and adds `node` to each entry. MCP `list_models` does the same. `load_model`, `unload_model`, `reserve_for` and `release_models` are forwarded to the owning node |
| Defaults and fallback | `defaults.chat_completion` is the remote model. A new `fallbacks` map (for example `chat_completion: ["qwen3.8-27b@spark-2"]`) is used when the remote tool is unavailable |
| Voice protection | The rule "while a realtime session is open, nothing outside the voice loop unloads a voice-loop model" must reach the peer. The front sends a session-open hold, so the peer does not evict the voice chat model for a non-voice job |
| Node auth | The peer binds only to the CX-7 subnet (and loopback). The front calls it with a node token, a new pairing scope `node`. Clients pair with the front only; `devices.json` stays on the front |
| Version skew | The front reads the peer's `/v1/capabilities` contract version and refuses an incompatible peer with a clear 503 |

The single surface does not change: clients keep one base URL (the front's tailnet name), one pairing and one MCP
server. A harness sees `node` only as an informational field ([HARNESS.md](../HARNESS.md)).

### Health and failover

| Event | Behaviour |
| --- | --- |
| Peer health check | The front polls `GET /health` and `/v1/tools` every 2 s over CX-7, and over the LAN when CX-7 fails. After 3 misses the peer is down and its tools are listed `unavailable` with the reason |
| **Spark 1 down** | Chat goes to the fallback 27B on Spark 2. It loads on demand at a priority below voice and above image, and evicts image if needed (52 GiB headroom, section 3). It takes 12.5–29.9 s to load (M). The voice session says it is "recovering", as model reloads do today. Everything else keeps working |
| A chat stream in flight when Spark 1 dies | Ends as a broken upstream stream does today: `data: {"error"}` then `[DONE]`. The voice turn fails like a 503 |
| Spark 1 back | Health passes, and the remote model becomes the default again. The fallback idles out on its TTL |
| **Spark 2 (front) down** | The box is down, as today. To survive it, the same binary can also hold the front role on Spark 1, with clients pairing with both names. That is later work, and the phone and web clients currently store one box |
| CX-7 cable down | Control and chat fall back to the LAN path (bandwidth is enough for (a)). (b) and (c) would stop |

### Install and deploy

- **Per-node tool sets.** `deploy/jarvis/install.sh` already takes a `TOOLS` subset (lines 33–35), so each node
  installs only its own models. A two-node profile (`deploy/profiles/jarvis-gb10x2.json`) names each node's role,
  address and tools. Each node's config sets its own `machine`.
- **Qualification stays per host.** Markers are bound to their host. If DGX-01 becomes Spark 2, every voice, image,
  video and decision marker stays valid, and only the chat model qualifies on the new Spark 1. That makes **DGX-01 the
  front** and the new box the chat node. In (b), a TP model needs a marker that binds both hosts.
- **Weights.** Never download them again. They move from DGX-01 over CX-7 (about 25 GB/s line rate, so 20 GB of 27B
  files take seconds, E) into the new node's model store, checked against the same pins.
- **Deploy order.** Peer first, then the front. The busy-box deploy guard applies to both nodes: no deploy while a
  smoke, qualify or check runs on either. The final smoke runs a voice turn whose chat crosses the link, plus
  `client_check`.

## 6. Staged plan

The estimates are focused engineer-days (E).

| Stage | Work | Effort | Gate |
| --- | --- | ---: | --- |
| S0 | **JJ:** which second Spark (dgx-02 back, or a new box). Check it is a 128 GB unit: a 64 GB variant was reported on 2026-10-23 (unverified) and would not hold the Spark 2 warm set (78 GiB) or Flash Next | none | decision |
| S1 | Hardware measurements (list below), in a window, no production impact | 1–2 | numbers in this doc, M |
| S2 | Node federation: `node`/`remote` tool, forwarded jobs and streams, merged discovery and MCP, node token scope, peer bound to CX-7 | 4–6 | GPU-free integration tests with two daemons on one host (fake peer) |
| S3 | Health, fallback chat, voice hold across nodes, version check | 2–3 | kill-the-peer test, fallback answers, recovery |
| S4 | Two-node deploy: profile, per-node `TOOLS`, weights copied over the link, smoke on both | 2–3 | full smoke, voice turn across the link |
| S5 | **First release:** today's 27B moved to Spark 1, qualified there; Spark 2 = DGX-01 as is | 1–2 | `qualify.sh` on Spark 1; voice e2e and `client_check` against today's 2.20 s / 2.95 s |
| S6 | Optional, **JJ:** the full Drama 3 back on Spark 2 (already qualified on DGX-01) | 0.5–1 | memory table in section 3 holds |
| S7 | Bigger chat model on Spark 1: pick, fetch, a `qualify.sh` for its family like `tools/chat/qwen38-27b`, memory peak measured. **JJ signs off any lossy one** | 3–5 per model | qualification on Spark 1, chat_speed, voice e2e |
| S8 | Optional (b): TP GLM-class. A second engine stack, two-host qualification, Spark 2 reduced to rank 1 + voice | 10–15 | contention measured with Drama 3 on rank 1 |
| S9 | (c) KV tier: not planned | — | see section 3 |

**S2–S5 total about 10–16 days** for a working two-Spark Jarvis with today's models.

### Measure on hardware before committing

1. **Link.** `ib_write_bw` and `ib_write_lat` on one and both rails, TCP round trip over CX-7 and over the LAN, at
   MTU 9000.
2. **The proxy hop.** Front → peer → TensorFold against direct: tok/s and TTFT equal within noise, using
   `tools/spark/jarvis/chat_speed.py`.
3. **Voice end to end.** First audio p50/p90 with chat remote, against 2.20 / 2.95 s (`voice_e2e.sh`), and the long
   reply's start hold (`client_check`).
4. **Drama 3 on Spark 2.** RTF with image warm and idle, and while an image job runs.
5. **Each candidate model.** Peak memory (`MemAvailable` drop, plus `nvidia-smi` used memory, plus `RssAnon` +
   `RssShmem`), and tok/s and TTFT through the daemon on Spark 1.
6. **Failover.** Time from killing the peer to the fallback's first token, and the recovery path.
7. **Only for (b).** TP decode while Drama 3 synthesises on rank 1, and whether an image job on Spark 2 slows chat.

## Decisions for JJ

| Decision | Default here |
| --- | --- |
| Second Spark: which machine, and when | none; this blocks S1 onwards |
| Placement | (a) |
| Spark 1 chat model after S5 | measure Flash Next (lossy, 4-bit or EXL3) and a lossless option against the 27B, then decide on numbers |
| Full Drama 3 back on Spark 2 | yes, if the S1 memory numbers hold. The NVFP4 switch was for room, and room comes back |
| Must the box survive the front going down? | not in the first release |
