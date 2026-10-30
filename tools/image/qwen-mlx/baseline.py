#!/usr/bin/env python3
"""The macOS Qwen-Image baseline: the real model through the worker, as the daemon drives it, with times and memory.

    ROOT/venv/bin/python -I tools/image/qwen-mlx/baseline.py ROOT OUT [--only t2i,edit1,edit2,edit5,mask] [--w8a8]

ROOT is fetch.sh's; OUT a directory outside the repository (the images are not committed). The worker loads (timed),
then makes the fixed prompts and seeds below: 1024x1024 text to image, and edits with 1, 2 and 5 input images (the
inputs are themselves made here, so the baseline needs nothing from outside) and one with a mask. Every case runs
twice with two seeds. A sampler reads the server's resident set every 100 ms (unified memory: MLX's Metal buffers are
in it). OUT/baseline.json holds the numbers and OUT/<case>-<seed>.png the images. Hold the machine's GPU lock while it
runs. docs/dev/IMAGE-EDIT.md, "macOS", has the results.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# The inputs, made first: (name, prompt, width, height, seed). Sizes differ so the edits see mixed shapes.
INPUTS = [
    ("woman", "a photo of a smiling woman with short black hair in a plain grey sweater, studio portrait, soft light", 1024, 1024, 11),
    ("hat", "a red knitted wool beanie with a white pompom, product photo on a white background", 768, 768, 12),
    ("cup", "a turquoise ceramic coffee cup with steam rising, product photo on a white background", 1024, 768, 13),
    ("dog", "a golden retriever sitting on grass, looking at the camera, daylight photo", 768, 1024, 14),
    ("room", "a cosy reading room with a green armchair, a bookshelf and a big window, warm evening light", 1344, 768, 15),
]
SEEDS = (7, 8)
# (case, prompt, inputs, size or None (the last input's shape), mask)
CASES = [
    ("t2i", "a red fox standing in fresh snow at golden hour, a wooden sign in the background that reads \"LocalRouter\"", [], (1024, 1024), False),
    ("edit1", "make it night time: dark blue sky through the window light, moonlight on her face", ["woman"], None, False),
    ("edit2", "the woman from image 1 wearing the red beanie from image 2, same portrait and lighting", ["woman", "hat"], None, False),
    ("edit5", "the woman from image 1 wearing the beanie from image 2, sitting in the armchair of the room in image 5, "
              "holding the cup from image 3, with the dog from image 4 at her feet", ["woman", "hat", "cup", "dog", "room"], None, False),
    ("mask", "give her a bright yellow sweater", ["woman"], (1024, 1024), True),
]


class Sampler:
    """The peak resident set of the worker's mlx-serve, sampled every 100 ms; `mark` starts a new window."""

    def __init__(self, worker_pid):
        self.worker_pid, self.pid = worker_pid, 0
        self.peak = self.window = 0
        self.stop = False
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while not self.stop:
            if not self.pid:
                out = subprocess.run(["pgrep", "-P", str(self.worker_pid), "mlx-serve"], capture_output=True, text=True).stdout.split()
                self.pid = int(out[0]) if out else 0
            if self.pid:
                out = subprocess.run(["ps", "-o", "rss=", "-p", str(self.pid)], capture_output=True, text=True).stdout.strip()
                rss = int(out or 0) * 1024
                self.peak, self.window = max(self.peak, rss), max(self.window, rss)
            time.sleep(0.1)

    def mark(self):
        w, self.window = self.window, 0
        return w


class Worker:
    def __init__(self, root, w8a8):
        tool = {"id": "qwen-image-2.1-turbo-mlx", "kind": "image", "weights": os.path.join(root, "qwen-image-2.1-turbo-mlx-q8"),
                "options": {"w8a8": w8a8}}
        self.p = subprocess.Popen([os.path.join(root, "venv", "bin", "python"), "-I", os.path.join(HERE, "qwen_mlx_worker.py")],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                  env=dict(os.environ, LOCALROUTER_TOOL=json.dumps(tool)))

    def call(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()
        while True:
            m = json.loads(self.p.stdout.readline())
            if "progress" not in m:
                return m


def gb(n):
    return round(n / 1e9, 2)


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("root")
    a.add_argument("out")
    a.add_argument("--only", default="t2i,edit1,edit2,edit5,mask")
    a.add_argument("--w8a8", action="store_true", help="MLX-Serve's lossy int8 DiT linears (not the baseline)")
    args = a.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    only = set(args.only.split(","))
    report = {"machine": f"{platform.machine()} {platform.mac_ver()[0]}", "w8a8": args.w8a8, "cases": []}

    w = Worker(args.root, args.w8a8)
    s = Sampler(w.p.pid)
    t = time.monotonic()
    r = w.call({"op": "load"})
    if not r.get("ok"):
        sys.exit(f"load failed: {r}")
    report["load_s"] = round(time.monotonic() - t, 2)
    report["resident_bytes"] = r["resident"]
    report["load_peak_rss"] = s.mark()
    print(f"load {report['load_s']} s, resident {gb(r['resident'])} GB", flush=True)

    def run(case, prompt, refs, size, seed, mask=None):
        job = os.path.join(out, "jobs", f"{case}-{seed}")
        shutil.rmtree(job, ignore_errors=True)
        os.makedirs(job)
        names = []
        for i, ref in enumerate(refs):
            shutil.copy(os.path.join(out, f"input-{ref}.png"), os.path.join(job, f"ref_{i}.png"))
            names.append(f"ref_{i}.png")
        img = {"prompt": prompt, "n": 1, "seed": seed, "references": names,
               "width": size[0] if size else 0, "height": size[1] if size else 0}
        if mask:
            shutil.copy(mask, os.path.join(job, "mask_0.png"))
            img["mask"] = "mask_0.png"
        s.mark()
        t = time.monotonic()
        r = w.call({"op": "generate", "id": f"{case}-{seed}", "dir": job, "request": {"image": img}})
        secs = round(time.monotonic() - t, 2)
        peak = s.mark()
        if not r.get("ok"):
            sys.exit(f"{case} seed {seed}: {r}")
        return os.path.join(job, r["files"][0]), secs, peak

    if only - {"t2i"}:
        for name, prompt, width, height, seed in INPUTS:
            path, secs, peak = run(f"input-{name}", prompt, [], (width, height), seed)
            shutil.copy(path, os.path.join(out, f"input-{name}.png"))
            report["cases"].append({"case": f"input-{name}", "size": f"{width}x{height}", "seed": seed, "s": secs, "peak_rss": peak})
            print(f"input {name} {width}x{height}: {secs} s, peak {gb(peak)} GB", flush=True)
    for case, prompt, refs, size, masked in CASES:
        if case not in only:
            continue
        mask = None
        if masked:
            from PIL import Image  # the worker's venv has it
            m = Image.new("RGBA", (1024, 1024), (0, 0, 0, 255))
            m.paste((0, 0, 0, 0), (256, 560, 768, 1024))  # clear over the sweater: only there may it change
            mask = os.path.join(out, "mask.png")
            m.save(mask)
        for seed in SEEDS:
            path, secs, peak = run(case, prompt, refs, size, seed, mask)
            dest = os.path.join(out, f"{case}-{seed}.png")
            shutil.copy(path, dest)
            report["cases"].append({"case": case, "inputs": refs, "seed": seed, "s": secs, "peak_rss": peak, "file": dest})
            print(f"{case} seed {seed}: {secs} s, peak {gb(peak)} GB -> {dest}", flush=True)

    t = time.monotonic()
    w.p.stdin.write(json.dumps({"op": "exit"}) + "\n")  # no reply: the worker stops the server and exits
    w.p.stdin.flush()
    w.p.wait(timeout=120)
    report["exit_s"] = round(time.monotonic() - t, 2)
    s.stop = True
    report["peak_rss"] = s.peak
    shutil.rmtree(os.path.join(out, "jobs"), ignore_errors=True)
    with open(os.path.join(out, "baseline.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"peak {gb(s.peak)} GB; {os.path.join(out, 'baseline.json')}")


if __name__ == "__main__":
    main()
