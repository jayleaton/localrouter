#!/usr/bin/env python3
"""Tests of hf_pinned.py verify/download with no network (standard library only): python3 -I tools/install/test_hf_pinned.py
A stub `hf` stands in for huggingface_hub's CLI and writes the files it is asked for."""
import hashlib, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "hf_pinned.py")
fails = 0
def check(name, cond, got=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f": {got}")); fails += not cond

def run(*args):
    r = subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr

t = tempfile.mkdtemp()
files = {"a/model.gguf": b"weights" * 1000, "config.json": b"{}"}
m = {"repo": "org/model", "revision": "f" * 40, "files": {k: [len(v), hashlib.sha256(v).hexdigest()] for k, v in files.items()}}
man = os.path.join(t, "pin.json"); json.dump(m, open(man, "w"))
# the stub: `hf download REPO FILES... --revision REV --local-dir DIR`; logs its argv, writes FILES (one corrupted on request)
stub = os.path.join(t, "hf")
open(stub, "w").write(f"""#!{sys.executable}
import json, os, sys
a = sys.argv[1:]; open({os.path.join(t, 'argv')!r}, 'w').write(json.dumps(a))
d = a[a.index('--local-dir') + 1]; names = a[2:a.index('--revision')]
data = {json.dumps({k: v.decode() for k, v in files.items()})}
for n in names:
    os.makedirs(os.path.dirname(os.path.join(d, n)), exist_ok=True)
    open(os.path.join(d, n), 'w').write(data[n] if not os.environ.get('CORRUPT') else data[n].upper())
""")
os.chmod(stub, 0o755)

d = os.path.join(t, "dl")
code, out = run("download", man, d, "--hf", stub)
argv = json.load(open(os.path.join(t, "argv")))
check("download asks hf for the pinned revision and only the manifest's files",
      argv[:2] == ["download", "org/model"] and argv[argv.index("--revision") + 1] == "f" * 40 and sorted(argv[2:4]) == sorted(files), argv)
check("download verifies what it got", code == 0 and "2 file(s) verified" in out, out)
check("verify: a good tree passes", run("verify", man, d)[0] == 0)
open(os.path.join(d, "config.json"), "w").write("[]")
code, out = run("verify", man, d)
check("verify: same size, other bytes: refused by sha256", code == 1 and "config.json: sha256 differs" in out, out)
check("verify --sizes: the cheap check does not hash", run("verify", man, d, "--sizes")[0] == 0)
os.remove(os.path.join(d, "a/model.gguf"))
code, out = run("verify", man, d, "--sizes")
check("verify --sizes: a missing file is refused", code == 1 and "a/model.gguf: missing" in out, out)
r = subprocess.run([sys.executable, "-I", TOOL, "download", man, os.path.join(t, "bad"), "--hf", stub], capture_output=True, text=True, env={**os.environ, "CORRUPT": "1"})
check("download: bytes that differ from the pin fail the step", r.returncode == 1 and "sha256 differs" in r.stderr, r.stderr)
json.dump({**m, "revision": "main"}, open(man, "w"))
check("a manifest without a 40-hex revision is refused", run("verify", man, d)[0] != 0)
for p in sorted(os.listdir(os.path.join(HERE, "..", "..", "deploy", "strix-halo", "pins"))) if os.path.isdir(os.path.join(HERE, "..", "..", "deploy", "strix-halo", "pins")) else []:
    q = json.load(open(os.path.join(HERE, "..", "..", "deploy", "strix-halo", "pins", p)))
    check(f"pins/{p}: a commit and a sha256 for every file", len(q["revision"]) == 40 and all(len(h) == 64 and n > 0 for n, h in q["files"].values()), q["revision"])
print(f"{'FAILED' if fails else 'ok'}: {fails} failure(s)")
sys.exit(1 if fails else 0)
