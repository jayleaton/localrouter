# Qwen3.8-Flash-Next TP1 (chat on a GB10 of its own)

`qwen3.8-flash-next` in `deploy/jarvis/config.json.in`: Qwen3.8-Flash-Next served by TensorFold's native CUDA engine
on one DGX Spark, with the recipe's TP1 profile **qwen38-flash-next-tp1-tool-full**, so the compute lane and the
in-model runner stay on. The engine and its profile come from
[jayleaton/qwen4-tensorfold-spark](https://github.com/jayleaton/qwen4-tensorfold-spark) at `f0002f9`: upstream
TensorFold `aaac299d1` plus `patches/series.tp1` (0001-0083), built into a Docker image by `scripts/build.sh tp1`.
LocalRouter carries no engine change of its own for this model.

The model fills a GB10: it is marked `own_spark`, so a one-box install leaves it out. It is the chat model of the
two-Spark placement (`deploy/jarvis/topologies/flashnext-2spark.json`, [MULTI-SPARK.md](../../../docs/dev/MULTI-SPARK.md#placement-with-qwen38-flash-next-tp1-2026-11-14)).

| File | What it is |
| --- | --- |
| `pins.json` | recipe commit, engine base and tree, the image's labels, the checkpoint's repo and revision |
| `tf.env` | the profile's `TF_*` lines, unchanged: the engine settings the measured boot passed to the container |
| `serve.sh` | the server LocalRouter's `openai_server` engine runs: the recipe's `docker run` line in the foreground |
| `check.sh` | install.sh's preflight: docker with the NVIDIA container toolkit (an `nvidia` runtime, or its hook or CDI spec under runc), the pinned image, the checkpoint complete |

## How it runs

`serve.sh PORT NAME MODEL_DIR` runs the recipe's TP1 boot line (its README, "Serve on one Spark"):

```sh
docker run --rm --name localrouter-qwen3.8-flash-next --init --gpus all --ipc=host --network host --ulimit memlock=-1 \
  --env-file tf.env -e TENSORFOLD_CUDA_MEMORY_LIMIT_GB=103.5 -e TENSORFOLD_NO_UPDATE_CHECK=1 -v MODEL_DIR:/model:ro IMAGE \
  serve /model --host 127.0.0.1 --port PORT --name qwen3.8-flash-next --context 65536 --thinking-budget 8192 --no-update-check
```

Beyond the recipe's line, it adds the name LocalRouter sends requests under (`--name`), no update check, and a cap on
the engine's CUDA budget (below). None of these changes what the model computes.

- **Exact names only.** The container is `localrouter-<tool id>`, labelled `org.localrouter.tool` (the tool) and
  `org.localrouter.owner` (the install's `DEPLOY`). A container of that name with another tool's label is never touched.
- **Stopping.** Unloading sends `serve.sh` SIGTERM. It stops the container it started, by the id `docker run
  --cidfile` wrote (20 s, then SIGKILL), and exits; the tool's `stop_timeout_s` is 40. When the daemon dies, its
  worker SIGKILLs `serve.sh`, and a watchdog that `serve.sh` started stops the container. A container left behind
  (a host crash) is removed by the next start, and `deploy/jarvis/leftovers.sh` stops the install's labelled
  containers after the service stops.
- **Readiness.** `/health` answers once the model is loaded. The recipe's boot smoke on this image was ready in
  181 s; `load_timeout_s` is 900.

## Memory

The engine sizes itself to the free memory. After loading the weights, it gives streams and the prompt cache what is
left after a reserve of a tenth of the memory (12.2 GiB on a GB10). The two boots of this image on node01 (the
2026-11-13/14 confirmation runs, `arm-T1` and `arm2-T1`) logged:

> model 77.15 GiB; 8 streams at once, 2.28 GiB each at a 65536-token window, of 26.3 GiB left after a 12.2 GiB
> reserve; prompts in passes of up to 4096 rows, a 8.0 GiB prompt cache

`TENSORFOLD_CUDA_MEMORY_LIMIT_GB=103.5` (77.15 + 26.3, rounded up) caps the engine at that measured layout. A node
with more free memory then gets the same 8 streams and 8 GiB prompt cache; a node with less gets fewer streams.

| | GiB | Tag |
| --- | ---: | --- |
| Weights on the GPU | 77.15 | M, the engine's boot line |
| Streams (8 × 2.28 at 65,536 tokens) and prompt cache (8.0) | 26.3 | M, the same line |
| CUDA budget cap | 103.5 | config (`serve.sh`) |
| Host side (the server process, the lane's Python, docker) | 8 | policy: the GB10 rule for an unmeasured process |
| **Booked (`resident_bytes` 119,722,213,376)** | **111.5** | |

The node loads it only with `MemAvailable` at least 114.5 GiB: the booking plus the daemon's 3 GiB reserve. An idle
Spark has about 116 GiB, so nothing else may hold GPU memory on the chat node (another project's benchmark there
refuses the load). The per-process peak has not been measured: nvidia-smi `used_memory` plus `RssAnon` and `RssShmem`
of the server, through a long reply. Once it is, the 8 GiB policy margin becomes that peak plus a small margin.

## Speed (the recipe's figures, one Spark)

Single-stream decode was 48.8 / 47.8 / 53.8 tokens/s at 841 / 6k / 24k-token prompts on `c8e930375`, and 53.6 / 49.5
/ 55.3 with the CUDA compat driver, which this image uses. The `arm2-T1` boot's chat replies decoded at 48–110
tokens/s, with time to first token 0.05–0.06 s on a cached prefix and 0.37–0.47 s cold. Through LocalRouter it has
not been measured yet.

## Getting it onto a node

- **The image.** Build it from the recipe at the pinned commit (`scripts/build.sh tp1` on a Spark; the native compile
  needs about 5 GiB at `-j1`). Or copy a built one from another node over ConnectX-7, as root or a docker user:
  `docker save qwen4-tensorfold-f0002f9:tp1 | ssh USER@192.168.100.11 docker load`. `check.sh` accepts any tag whose
  labels match `pins.json`. `FLASHNEXT_IMAGE` (deploy/jarvis/env.sh) names it.
- **The checkpoint.** `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` at revision `2b170fa6`, 22 shards, about 105 GiB
  on disk. `FLASHNEXT_MODEL_DIR` (default `~/ai/models/qwen38-flash-next-mlx4-mtp`) names it. Reuse a copy that is
  already there, or copy one over the link; download it only when no node has it:
  `hf download TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP --revision 2b170fa6309d5d1ee380b35636075fac7945f286 --local-dir DIR`.
- **Docker** for the service's user, with the NVIDIA container toolkit, as `docker run --gpus all` needs it: an `nvidia` runtime, or the toolkit's hook or CDI spec under runc (`check.sh` checks both).

Then `check.sh "$FLASHNEXT_IMAGE" "$FLASHNEXT_MODEL_DIR"`. `deploy/jarvis/install.sh` runs the same check for a node
that runs this model.

## Not done yet

- **No LocalRouter qualification.** The recipe's release gates passed on this image (2026-11-13, `187e28a03`: replies
  byte-identical to `b62876185`'s). The 27B's `qualify.sh`, run through the daemon and with a marker bound to the host,
  has no counterpart here yet, so the tool has no `qualification` marker.
- **Voice turns.** The voice loop sends `chat_template_kwargs.enable_thinking: false`; that this engine answers it with
  thinking off (its log's `thinking=False`) is to be checked in the first window, with the cached next-turn check of
  `tools/spark/jarvis/chat_bench.py`.

## Licence

Qwen Community License 1.0 (the checkpoint's `LICENSE`). Use, hosting and sale are allowed. A business that runs a
Model as a Service or an AI Work Assistant needs a separate licence from Qwen for commercial use; internal use that
offers nothing to third parties is excepted. A commercial product above 100,000,000 monthly active users or
US$ 20,000,000 monthly revenue must show the model's name prominently in its UI. TensorFold and the recipe are
Apache-2.0.
