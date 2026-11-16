#!/usr/bin/env python3
"""Tests worker.py against a fake ComfyUI (no GPU): python3 -I tools/comfyui/test_worker.py"""
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = r'''
import json, os, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
args = sys.argv[1:]
port = int(args[args.index("--port") + 1]); out = args[args.index("--output-directory") + 1]
inp = args[args.index("--input-directory") + 1]
hist, seen = {}, []
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path == "/system_stats": return self.reply(200, {"system": {}})
        if self.path.startswith("/history/"):
            pid = self.path.split("/")[-1]; return self.reply(200, {pid: hist[pid]} if pid in hist else {})
        if self.path.startswith("/view?"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            p = os.path.join(out, q["subfolder"][0], q["filename"][0])
            return self.reply(200, open(p, "rb").read(), "application/octet-stream")
        if self.path == "/seen": return self.reply(200, seen)
        self.reply(404, {})
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.loads(raw) if raw[:1] == b"{" else {}
        if self.path == "/prompt":
            wf = body["prompt"]; pid = uuid.uuid4().hex; seen.append(wf)
            slow = "SLOW" in json.dumps(wf)
            def run():
                time.sleep(30 if slow else 0.3)
                if pid in hist: return
                video = any(n["class_type"] == "SaveVideo" for n in wf.values())
                name = "clip_00001_.mp4" if video else "img_00001_.png"
                os.makedirs(os.path.join(out, "sub"), exist_ok=True)
                open(os.path.join(out, "sub", name), "wb").write(b"FAKE:" + json.dumps(wf).encode())
                hist[pid] = {"status": {"status_str": "success", "completed": True, "messages": []},
                             "outputs": {"9": {"images": [{"filename": name, "subfolder": "sub", "type": "output"}]}}}
            threading.Thread(target=run, daemon=True).start()
            return self.reply(200, {"prompt_id": pid, "node_errors": {}})
        if self.path == "/upload/image":
            from email.parser import BytesParser
            msg = BytesParser().parsebytes(b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw)
            f = {p.get_param("name", header="content-disposition"): p for p in msg.get_payload()}
            sub = f["subfolder"].get_payload(decode=True).decode(); name = f["image"].get_filename()
            os.makedirs(os.path.join(inp, sub), exist_ok=True)
            open(os.path.join(inp, sub, name), "wb").write(f["image"].get_payload(decode=True))
            return self.reply(200, {"name": name, "subfolder": sub, "type": "input"})
        if self.path == "/interrupt":
            pid = body.get("prompt_id")
            hist[pid] = {"status": {"status_str": "error", "completed": False, "messages": [["execution_interrupted", {}]]}, "outputs": {}}
            return self.reply(200, {})
        if self.path == "/queue": return self.reply(200, {})
        self.reply(404, {})
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''

IMAGE_WF = {"1": {"class_type": "Cond", "inputs": {"prompt": "x", "width": 0, "height": 0}},
            "2": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 6}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}}}
VIDEO_WF = {"1": {"class_type": "Cond", "inputs": {"prompt": "x", "width": 0, "height": 0, "length": 0}},
            "2": {"class_type": "Noise", "inputs": {"noise_seed": 0}},
            "3": {"class_type": "CreateVideo", "inputs": {"images": ["1", 0], "audio": ["1", 1], "fps": 24}},
            "9": {"class_type": "SaveVideo", "inputs": {"video": ["3", 0]}}}

fails = 0


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name)
    fails += not ok


class Proc:
    def __init__(self, cfg):
        env = dict(os.environ, LOCALROUTER_TOOL=json.dumps(cfg))
        self.p = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "worker.py")], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1)

    def ask(self, obj, timeout=20):
        if obj is not None:
            self.p.stdin.write(json.dumps(obj) + "\n")
            self.p.stdin.flush()
        progress = []
        end = time.time() + timeout
        while time.time() < end:
            if select.select([self.p.stdout], [], [], 0.2)[0]:
                m = json.loads(self.p.stdout.readline())
                if "progress" in m:
                    progress.append(m["progress"])
                    continue
                m["_progress"] = progress
                return m
        raise TimeoutError(obj)


def main():
    d = tempfile.mkdtemp()
    comfy = os.path.join(d, "comfy")
    os.makedirs(comfy)
    open(os.path.join(comfy, "main.py"), "w").write(FAKE)
    for name, wf in (("img.json", IMAGE_WF), ("vid.json", VIDEO_WF)):
        json.dump(wf, open(os.path.join(d, name), "w"))
    base = {"comfy_dir": comfy, "python": sys.executable, "args": ["--fake"], "size_multiple": 32, "max_side": 2048}
    img = Proc({"id": "img", "resident_bytes": 123, "options": dict(base, workflow=os.path.join(d, "img.json"), bindings={
        "prompt": "1.prompt", "width": "1.width", "height": "1.height", "seed": "2.seed", "steps": "2.steps"})})
    r = img.ask({"op": "load"})
    check("load answers resident and signal cancel", r.get("ok") and r.get("resident") == 123 and r.get("cancel") == "signal")
    job = os.path.join(d, "job1")
    os.makedirs(job)
    r = img.ask({"op": "generate", "id": "img_1", "dir": job,
                 "request": {"image": {"prompt": "a fox", "width": 1024, "height": 512, "n": 2, "seed": 7, "steps": 4}}})
    check("image n=2 writes 0.png and 1.png", r.get("ok") and r.get("files") == ["0.png", "1.png"] and r.get("seed") == 7)
    wf1 = json.loads(open(os.path.join(job, "1.png"), "rb").read()[5:])
    check("bindings filled (prompt, size, seed+1, steps)", wf1["1"]["inputs"] == {"prompt": "a fox", "width": 1024, "height": 512}
          and wf1["2"]["inputs"] == {"seed": 8, "steps": 4})
    r = img.ask({"op": "generate", "id": "img_2", "dir": job, "request": {"image": {"prompt": "a", "width": 1008, "height": 1024}}})
    check("size off the 32 grid is refused", r.get("type") == "invalid_request_error" and "multiples of 32" in r["error"])
    r = img.ask({"op": "generate", "id": "img_3", "dir": job, "request": {"image": {"prompt": "a", "negative_prompt": "b"}}})
    check("unbound negative prompt is refused, not ignored", r.get("type") == "invalid_request_error")
    r = img.ask({"op": "generate", "id": "img_4", "dir": job, "request": {"image": {"prompt": "a", "references": ["r.png"]}}})
    check("edits are refused by a tool without an image_edit mode", r.get("type") == "invalid_request_error")
    img.p.stdin.write(json.dumps({"op": "generate", "id": "img_5", "dir": job, "request": {"image": {"prompt": "SLOW"}}}) + "\n")
    img.p.stdin.flush()
    time.sleep(1.5)
    img.p.send_signal(signal.SIGUSR1)
    r = img.ask(None, timeout=10)
    check("SIGUSR1 cancels the running job", r.get("type") == "cancelled")
    r = img.ask({"op": "generate", "id": "img_6", "dir": job, "request": {"image": {"prompt": "after"}}})
    check("worker stays loaded after a cancel", r.get("ok") and r.get("files") == ["0.png"])
    out_dirs = [l for l in os.listdir(tempfile.gettempdir()) if l.startswith("lr-comfyui-")]
    img.p.stdin.write(json.dumps({"op": "exit"}) + "\n")
    img.p.stdin.flush()
    img.p.wait(timeout=20)
    left = [l for l in os.listdir(tempfile.gettempdir()) if l.startswith("lr-comfyui-")]
    check("exit stops ComfyUI and removes its output directory", img.p.returncode == 0 and len(left) < len(out_dirs))

    vid = Proc({"id": "vid", "options": dict(base, workflow=os.path.join(d, "vid.json"), frame_multiple=17, frame_offset=5,
                                             fps=[24], bindings={"prompt": "1.prompt", "width": "1.width", "height": "1.height",
                                                                 "seed": "2.noise_seed", "frames": "1.length", "fps": "3.fps",
                                                                 "audio": "3.audio"})})
    check("video tool loads", vid.ask({"op": "load"}).get("ok"))
    job = os.path.join(d, "job2")
    os.makedirs(job)
    r = vid.ask({"op": "generate", "id": "v_1", "dir": job,
                 "request": {"video": {"prompt": "waves", "width": 512, "height": 288, "seconds": 2, "fps": 24, "seed": 3, "audio": False}}})
    wf = json.loads(open(os.path.join(job, "video.mp4"), "rb").read()[5:]) if r.get("ok") else {}
    check("video writes video.mp4", r.get("files") == ["video.mp4"])
    check("frames snap up to the 17k+5 grid (48 -> 56)", wf.get("1", {}).get("inputs", {}).get("length") == 56)
    check("audio off removes the bound input", "audio" not in wf.get("3", {}).get("inputs", {"audio": 1}))
    r = vid.ask({"op": "generate", "id": "v_2", "dir": job, "request": {"video": {"prompt": "w", "width": 512, "height": 288, "fps": 30}}})
    check("an fps the model does not take is refused", r.get("type") == "invalid_request_error")
    vid.p.stdin.close()
    vid.p.wait(timeout=20)
    check("EOF stops the worker", vid.p.returncode == 0)

    edit_wf = dict(IMAGE_WF, **{"5": {"class_type": "Scale", "inputs": {"image": None}},
                                "6": {"class_type": "Composite", "inputs": {"destination": ["5", 0]}}})
    json.dump(edit_wf, open(os.path.join(d, "edit.json"), "w"))
    ed = Proc({"id": "ed", "options": dict(base, workflow=os.path.join(d, "img.json"), bindings={"prompt": "1.prompt", "seed": "2.seed"},
                                            modes={"image_edit": {"workflow": os.path.join(d, "edit.json"), "bindings": {
                                                "references": ["5.image", "1.images.image_2"], "mask": "6.mask"}}})})
    check("edit tool loads", ed.ask({"op": "load"}).get("ok"))
    job = os.path.join(d, "job3")
    os.makedirs(job)
    for n in ("a.png", "b.png", "c.png", "m.png"):
        open(os.path.join(job, n), "wb").write(b"PNG" + n.encode())
    r = ed.ask({"op": "generate", "id": "img_e1", "dir": job,
                "request": {"image": {"prompt": "make it night", "references": ["a.png", "b.png"], "mask": "m.png", "seed": 5}}})
    wf = json.loads(open(os.path.join(job, "0.png"), "rb").read()[5:]) if r.get("ok") else {}
    loads = {k: v["inputs"]["image"] for k, v in wf.items() if v["class_type"] == "LoadImage"}
    ref1, ref2, mask = wf.get("5", {}).get("inputs", {}).get("image"), wf.get("1", {}).get("inputs", {}).get("images.image_2"), wf.get("6", {}).get("inputs", {}).get("mask")
    check("edit: each input image becomes a LoadImage linked where bound (mask: its MASK output)",
          r.get("ok") and len(loads) == 3 and loads.get(ref1[0], "").endswith("a.png") and ref1[1] == 0
          and loads.get(ref2[0], "").endswith("b.png") and loads.get(mask[0], "").endswith("m.png") and mask[1] == 1)
    check("edit: inputs uploaded under the job's subfolder", all(v.startswith("localrouter-") for v in loads.values()))
    r = ed.ask({"op": "generate", "id": "img_e2", "dir": job, "request": {"image": {"prompt": "x", "references": ["a.png", "b.png", "c.png"]}}})
    check("more input images than bound slots are refused", r.get("type") == "invalid_request_error" and "at most 2" in r["error"])
    r = ed.ask({"op": "generate", "id": "img_e3", "dir": job, "request": {"image": {"prompt": "x", "references": ["../x.png"]}}})
    check("an input outside the job directory is refused", r.get("type") == "invalid_request_error")
    r = ed.ask({"op": "generate", "id": "img_e4", "dir": job, "request": {"image": {"prompt": "x", "references": ["a.png"]}}})
    check("edit without a mask still runs when there is no unmasked overlay", r.get("ok") and r.get("files") == ["0.png"])
    ed.p.stdin.close()
    ed.p.wait(timeout=20)

    masked_wf = json.loads(json.dumps(edit_wf))
    unmasked_wf = json.loads(json.dumps(edit_wf))
    unmasked_wf["2"]["inputs"]["steps"] = 30
    unmasked_wf.pop("6")  # no composite: this path is for whole-scene edits
    json.dump(masked_wf, open(os.path.join(d, "edit-masked.json"), "w"))
    json.dump(unmasked_wf, open(os.path.join(d, "edit-unmasked.json"), "w"))
    ed2 = Proc({"id": "ed2", "options": dict(base, workflow=os.path.join(d, "img.json"),
                                             bindings={"prompt": "1.prompt", "seed": "2.seed", "steps": "2.steps"},
                                             modes={"image_edit": {"workflow": os.path.join(d, "edit-masked.json"),
                                                                   "bindings": {"references": "5.image", "mask": "6.mask"},
                                                                   "defaults": {"steps": 6},
                                                                   "unmasked": {"workflow": os.path.join(d, "edit-unmasked.json"),
                                                                                "bindings": {"mask": None},
                                                                                "defaults": {"steps": 30}}}})})
    check("edit tool with an unmasked overlay loads", ed2.ask({"op": "load"}).get("ok"))
    r = ed2.ask({"op": "generate", "id": "img_u1", "dir": job, "request": {"image": {"prompt": "make it snow", "references": ["a.png"], "seed": 1}}})
    wf = json.loads(open(os.path.join(job, "0.png"), "rb").read()[5:]) if r.get("ok") else {}
    check("whole-scene edit (no mask) uses the unmasked workflow and default steps",
          r.get("ok") and "6" not in wf and wf.get("2", {}).get("inputs", {}).get("steps") == 30)
    r = ed2.ask({"op": "generate", "id": "img_u2", "dir": job,
                 "request": {"image": {"prompt": "red umbrella", "references": ["a.png"], "mask": "m.png", "seed": 1}}})
    wf = json.loads(open(os.path.join(job, "0.png"), "rb").read()[5:]) if r.get("ok") else {}
    mask = wf.get("6", {}).get("inputs", {}).get("mask")
    loads = {k: v["inputs"]["image"] for k, v in wf.items() if v.get("class_type") == "LoadImage"}
    check("masked edit keeps the fast workflow and default steps",
          r.get("ok") and wf.get("6", {}).get("class_type") == "Composite" and wf.get("2", {}).get("inputs", {}).get("steps") == 6
          and mask and loads.get(mask[0], "").endswith("m.png") and mask[1] == 1)
    r = ed2.ask({"op": "generate", "id": "img_u3", "dir": job,
                 "request": {"image": {"prompt": "make it snow", "references": ["a.png"], "steps": 8}}})
    wf = json.loads(open(os.path.join(job, "0.png"), "rb").read()[5:]) if r.get("ok") else {}
    check("request steps override the unmasked default; the unmasked workflow is still used",
          r.get("ok") and "6" not in wf and wf.get("2", {}).get("inputs", {}).get("steps") == 8)
    ed2.p.stdin.close()
    ed2.p.wait(timeout=20)

    i2v = Proc({"id": "i2v", "options": dict(base, workflow=os.path.join(d, "vid.json"), bindings={"prompt": "1.prompt", "frames": "1.length"},
                                             modes={"image_to_video": {"bindings": {"first_frame": "1.first_frame"}}})})
    check("i2v tool loads", i2v.ask({"op": "load"}).get("ok"))
    r = i2v.ask({"op": "generate", "id": "v_i1", "dir": job, "request": {"video": {"prompt": "it moves", "seconds": 1, "fps": 24, "first_frame": "a.png"}}})
    wf = json.loads(open(os.path.join(job, "video.mp4"), "rb").read()[5:]) if r.get("ok") else {}
    ff = wf.get("1", {}).get("inputs", {}).get("first_frame")
    check("image to video: the first frame is a LoadImage on the bound input", r.get("ok") and ff and wf[ff[0]]["class_type"] == "LoadImage"
          and wf["1"]["inputs"]["length"] == 24)
    r = i2v.ask({"op": "generate", "id": "v_i2", "dir": job, "request": {"video": {"prompt": "t2v", "seconds": 1, "fps": 24}}})
    check("the same tool still makes text to video without the first frame", r.get("ok"))
    i2v.p.stdin.close()
    i2v.p.wait(timeout=20)

    bad = Proc({"id": "bad", "options": dict(base, workflow=os.path.join(d, "img.json"), bindings={"prompt": "7.prompt"})})
    r = bad.ask({"op": "load"})
    check("a binding to a missing node fails the load", r.get("ok") is False and "binding" in r.get("error", ""))
    bad.p.wait(timeout=10)
    print("all passed" if not fails else f"{fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
