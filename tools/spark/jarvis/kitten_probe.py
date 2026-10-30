"""Kitten TTS 2 on GB10, taken apart through the worker's own code path (run with the kitten venv's python -I).

  kitten_probe.py REPO WEIGHTS [THREADS]

One JSON line each:
  env     torch and CUDA, where the language model's and the decoder's parameters live, their dtypes, the attention
          implementation, torch's CPU threads, whether flash_attn imports, load seconds
  memory  after load: host RSS, CUDA allocated and reserved, the resident the worker reports to the daemon; RSS again
          after glibc's malloc_trim(0) (a large drop: freed host copies of the weights were kept by the allocator)
  run     the same ~7 s sentence three times: language model seconds and codec tokens (25 tokens are 1 s of audio),
          decoder seconds, process CPU seconds against wall seconds in the language model; the third run under
          torch.profiler adds the share of the language model's wall time the GPU spent in kernels
"""
import ctypes, json, os, sys, time

repo, weights = sys.argv[1], sys.argv[2]
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 0
sys.path.insert(0, os.path.join(repo, "tools", "speech", "kitten"))
import kitten_worker as kw  # noqa: E402

TEXT = ("Good evening. Your calendar is clear tomorrow morning, but you have a dentist appointment at two and dinner "
        "with Sam at seven.")


def rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def rss_split():
    """RssAnon (heap, tensors) and RssFile (mapped libraries and files) in GiB."""
    with open("/proc/self/status") as f:
        kib = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in f) if k in ("RssAnon", "RssFile", "RssShmem")}
    return {k: round(v / 2**20, 2) for k, v in kib.items()}


def out(**fields):
    print(json.dumps(fields), flush=True)


opts = {"device": "cuda", "min_speed": 1, "max_speed": 1}
if threads:
    opts["threads"] = threads
engine = kw.KittenEngine(kw.Options({"weights": weights, "options": opts}))
t0 = time.monotonic()
resident = engine.load()
load_s = time.monotonic() - t0
torch = engine.torch
m = engine.model
lm = m.model
try:
    import flash_attn  # noqa: F401
    has_fa = True
except Exception:
    has_fa = False
out(probe="env", torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    lm_devices=sorted({str(p.device) for p in lm.parameters()}), lm_dtypes=sorted({str(p.dtype) for p in lm.parameters()}),
    decoder_devices=sorted({str(p.device) for p in m.codec.s3gen.parameters()}),
    attn=getattr(lm.config, "_attn_implementation", None), torch_threads=torch.get_num_threads(), flash_attn=has_fa,
    load_s=round(load_s, 2))
before = rss()
ctypes.CDLL("libc.so.6").malloc_trim(0)
out(probe="memory", reported_resident_gib=round(resident / 2**30, 2), rss_gib=round(before / 2**30, 2),
    rss_after_trim_gib=round(rss() / 2**30, 2), rss_split_gib=rss_split(), cuda_allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 2),
    cuda_reserved_gib=round(torch.cuda.memory_reserved() / 2**30, 2))

timing = {}
generate_tokens, decode = m._generate_tokens, m.codec.decode


def timed(key, fn):
    def run(*args, **kwargs):
        torch.cuda.synchronize()
        wall, cpu = time.monotonic(), time.process_time()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        timing[key + "_s"] = timing.get(key + "_s", 0.0) + time.monotonic() - wall
        timing[key + "_cpu_s"] = timing.get(key + "_cpu_s", 0.0) + time.process_time() - cpu
        if key == "lm":
            timing["tokens"] = timing.get("tokens", 0) + len(result)
        return result
    return run


m._generate_tokens = timed("lm", generate_tokens)
m.codec.decode = timed("decoder", decode)
speech = engine.request({"input": TEXT, "voice": "Bruno", "seed": 7})
for i in range(3):
    timing.clear()
    gpu_share = None
    wall = time.monotonic()
    if i < 2:
        wav, _ = engine.synthesize(speech, lambda done, of: None)
    else:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            wav, _ = engine.synthesize(speech, lambda done, of: None)
        from torch.autograd import DeviceType
        # the kernels themselves only (an op's self device time would count each of its kernels a second time)
        events = prof.events()
        kernel_us = sum(e.time_range.elapsed_us() for e in events if e.device_type == DeviceType.CUDA)
        if kernel_us == 0:  # this torch lists kernels only under the ops that launched them
            kernel_us = sum(k.duration for e in events for k in getattr(e, "kernels", []))
        gpu_share = round(kernel_us / 1e6 / (timing["lm_s"] + timing["decoder_s"]), 3)
        timing["kernel_ms_per_token"] = kernel_us / 1e3 / timing["tokens"]
    wall = time.monotonic() - wall
    audio_s = len(wav) / m.sample_rate
    out(probe="run", run=i, profiled=i == 2, audio_s=round(audio_s, 2), wall_s=round(wall, 2), rtf=round(wall / audio_s, 3),
        lm_s=round(timing["lm_s"], 2), tokens=timing["tokens"], lm_tokens_per_s=round(timing["tokens"] / timing["lm_s"], 1),
        lm_cpu_over_wall=round(timing["lm_cpu_s"] / timing["lm_s"], 2), decoder_s=round(timing["decoder_s"], 2),
        gpu_kernel_share=gpu_share, kernel_ms_per_token=round(timing.get("kernel_ms_per_token", 0), 2) or None)
