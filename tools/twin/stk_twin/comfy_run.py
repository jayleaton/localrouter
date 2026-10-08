"""python -m stk_twin.comfy_run --out DIR [--width 768 --height 448 --seconds 2 --steps 20 --seed 42 --lora] ...

MiniMax H3 text to video + audio through a headless ComfyUI 0.37.0 server (the reference the video twin is held to):
the "Image to Video (MiniMax H3)" blueprint's graph without a first frame (res_multistep, simple, BasicGuider, no CFG,
24 fps), the models from $VMODELS, every ComfyUI directory under --out. Prints one JSON line: wall seconds per run,
the outputs. `--extra-nodes DIR` loads custom nodes (the twin's recorder) into the server.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROMPT = ("Cinematic handheld shot in a rainy neon-lit night market. A street cook in a white apron flips noodles in a "
          "flaming wok, steam and sparks rising, customers laughing in the background. Rain drips from red paper "
          "lanterns. Sound: sizzling wok, crackling fire, rain on tarp, distant chatter. The cook says: \"Two more, "
          "extra spicy!\"")
DIT = "minimax_h3_fl2va_pruned_bf16.safetensors"
TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = "minimax_h3_video_vae_fp16.safetensors"
AVAE = "minimax_h3_audio_vae_fp32.safetensors"
LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"


def frames_for(seconds: float) -> int:
    """The blueprint's duration -> frame count (snapped to the 17k + 5 grid)."""
    n = max(5, round(seconds * 24))
    return n + (5 - n % 17) % 17


def graph(a, seed: int, prefix: str) -> dict:
    model = ["1", 0]
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": a.dit, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AVAE}},
        "5": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": a.prompt,
                                                                "width": a.width, "height": a.height,
                                                                "length": frames_for(a.seconds)}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "audio": ["12", 0], "fps": 24.0}},
        "14": {"class_type": "SaveVideo", "inputs": {"video": ["13", 0], "filename_prefix": prefix, "format": "mp4",
                                                     "format.codec": "h264", "format.codec.encoding": "re-encode",
                                                     "format.codec.encoding.crf": 12.0}},
    }
    if a.lora:
        g["21"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": LORA, "strength_model": 1.0}}
        model = ["21", 0]
    g["8"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": a.steps, "denoise": 1.0}}
    g["9"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["5", 0]}}
    g["10"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0],
                                                                 "sigmas": ["8", 0], "latent_image": ["5", 1]}}
    return g


def post(port: int, path: str, body: dict | None = None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"{}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--seconds", type=float, default=2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", type=int, default=2, help="the same graph again with seed + 1 ... (warm times)")
    ap.add_argument("--lora", action="store_true", help="the Turbo 8-step LoRA")
    ap.add_argument("--dit", default=DIT)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--extra-nodes", default=None)
    ap.add_argument("--comfy-args", default="", help="extra ComfyUI flags, e.g. --use-ck-attention")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    for d in ("input", "output", "temp", "user"):
        (out / d).mkdir(parents=True, exist_ok=True)
    models = Path(os.environ["VMODELS"])
    (out / "extra_model_paths.yaml").write_text(
        "stk:\n" + f"  base_path: {models}\n" + "".join(f"  {k}: {k}/\n" for k in
                                                      ("diffusion_models", "text_encoders", "vae", "loras")))
    comfy = Path(os.environ["COMFY"])
    cmd = [sys.executable, str(comfy / "main.py"), "--listen", "127.0.0.1", "--port", str(a.port), "--disable-auto-launch",
           "--output-directory", str(out / "output"), "--input-directory", str(out / "input"),
           "--temp-directory", str(out / "temp"), "--user-directory", str(out / "user"),
           "--extra-model-paths-config", str(out / "extra_model_paths.yaml"), *a.comfy_args.split()]
    if not a.extra_nodes:
        cmd.append("--disable-all-custom-nodes")
    env = dict(os.environ)
    if a.extra_nodes:
        env["STK_COMFY_NODES"] = a.extra_nodes
    log = open(out / "comfy.log", "w")
    srv = subprocess.Popen(cmd, cwd=comfy, stdout=log, stderr=subprocess.STDOUT, env=env)
    runs = []
    try:
        for _ in range(600):
            try:
                post(a.port, "/system_stats")
                break
            except Exception:
                if srv.poll() is not None:
                    raise SystemExit(f"ComfyUI exited; see {out / 'comfy.log'}")
                time.sleep(1)
        for r in range(a.runs):
            t0 = time.perf_counter()
            pid = post(a.port, "/prompt", {"prompt": graph(a, a.seed + r, f"run{r}")})["prompt_id"]
            while True:
                h = post(a.port, f"/history/{pid}")
                if pid in h and h[pid].get("status", {}).get("completed") is not None:
                    st = h[pid]["status"]
                    break
                if srv.poll() is not None:
                    raise SystemExit(f"ComfyUI exited; see {out / 'comfy.log'}")
                time.sleep(1)
            runs.append({"seed": a.seed + r, "wall_s": round(time.perf_counter() - t0, 1), "ok": st.get("status_str") == "success",
                         "messages": [m[0] for m in st.get("messages", [])]})
    finally:
        srv.terminate()
        srv.wait(timeout=60)
    files = sorted(str(p.relative_to(out)) for p in (out / "output").rglob("*") if p.is_file())
    print(json.dumps({"comfy_run": {"size": f"{a.width}x{a.height}", "frames": frames_for(a.seconds), "steps": a.steps,
                                    "lora": a.lora, "dit": a.dit, "runs": runs, "files": files}}))


if __name__ == "__main__":
    main()
