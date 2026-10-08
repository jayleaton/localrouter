# M5 runbook: Qwen-Image on a DGX Spark

What it proves on GB10 (sm_121), which no rented GPU can: LocalRouter's image tool gives the same pixels as its
Python reference on the Spark itself, how fast it is there (warm and cold, 1 MP and 0.3 MP, NVFP4 and FP8), how much
memory it takes, and that the Docker image builds and runs on DGX OS. Two scripts do the work; the owner starts them.

## Before the window

- The window must not overlap the co-tenant service's own windows (they are coordinated elsewhere). The scripts
  need the whole Spark: about 40 GB of memory at the peak (38 GiB measured on 2026-10-07, in the text encoder's packing; the engine itself 30 GiB NVFP4, 32 GiB FP8) and the GPU to themselves.
- Pick one Spark. The owner stops the co-tenant service there first; the scripts never touch it, and they refuse
  nothing on their own, so check `nvidia-smi` shows no other GPU process before starting.
- Disk: about 120 GB free under `/data/localrouter` (the checkpoint 31 GB, packs 27 GB, captures up to 15 GB at a time,
  build caches).
- Internet from the Spark: Hugging Face (the checkpoint, about 31 GB), GitHub (TensorFold, tfimage), ziglang.org,
  nvcr.io (the NGC PyTorch image, about 20 GB if not cached), pypi.

Time: about 1 h 45 min, most of it downloads and the twin's captures. The GPU-heavy part is about 45 minutes.

## In the window

On the Spark, as the owner:

```bash
# 1. LocalRouter's tree (a git archive from the dev box; no remote)
mkdir -p /data/localrouter && cd /data/localrouter
#    on the dev box: git -C localrouter archive --prefix=repo/ HEAD | ssh <spark> 'tar -x -C /data/localrouter'
mkdir -p twin kernels && cp -r repo/tools/twin/. twin/ && cp -r repo/kernels/. kernels/

# 2. the reference half: inside the NGC container (the twin's PyTorch), LocalRouter's tree at /workspace/localrouter
docker run --rm -it --gpus all --ipc host --ulimit memlock=-1 \
  -v /data/localrouter:/workspace/localrouter nvcr.io/nvidia/pytorch:26.07-py3 \
  bash /workspace/localrouter/repo/tools/spark/m5.sh 2>&1 | tee /data/localrouter/m5.out

# 3. the shipping half: the Docker image, built natively, serving the packs step 2 made
bash /data/localrouter/repo/tools/spark/m5-host.sh 2>&1 | tee /data/localrouter/m5-host.out

# 4. hand the results back (no checkpoint, no packs, no captures: a few MB)
tar -czf /data/localrouter/m5-results.tgz -C /data/localrouter results
```

Then restart the co-tenant service. Step 2 leaves no container behind (`--rm`); step 3 removes its container at the end. To free
the disk afterwards: `rm -rf /data/localrouter/{models,weights,captures,zig-cache,torch-ext,venv,src}` (keep `results`).

## What the scripts check

| Step | Pass when |
| --- | --- |
| packs | each NVFP4 and FP8 tensor's sha256 equals the reference (`tools/twin/packs/digests-*.json`) |
| replays | DiT, text encoder and VAE ops bit-exact, alone and chained, on sm_121 |
| SASS | the NVFP4 kernels' sm_121 SASS equals TensorFold's own build |
| e2e | sigmas, text context, latents and pixels byte-equal to the twin on sm_121 for the 6 gate prompts at 512 and 1024 (4 steps), and at 1024 and 576 with 25 steps for both precisions; with and without step graphs |
| times | e2e warm times per phase; API cold (page cache dropped) and warm wall times; peak memory (`memavail.log`, `docker-mem.log`) |
| attention | `attnbench` on GB10: which staging and arithmetic variant is fastest there |

If a step fails, the others still run; each has a log next to `steps.jsonl`. Nothing in either script changes the
system outside `/data/localrouter` and the two containers, except dropping the page cache once (`sudo`) for the cold load.
