#!/usr/bin/env python3
"""Tests of check_hashes.py on small lock files (standard library only): python3 -I tools/install/test_check_hashes.py"""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
H = "sha256:" + "ab" * 32
LOCK = '''version = 1
[[package]]
name = "app"
version = "0.1.0"
source = {{ virtual = "." }}

[[package]]
name = "torch"
version = "2.13.0+rocm7.2"
source = {{ registry = "https://download.pytorch.org/whl/rocm7.2" }}
wheels = [
    {{ url = "https://example.invalid/torch-2.13.0-cp312-cp312-linux_x86_64.whl"{torch} }},
]

[[package]]
name = "numpy"
version = "2.5.4"
source = {{ registry = "https://pypi.org/simple" }}
sdist = {{ url = "https://example.invalid/numpy-2.5.4.tar.gz", hash = "{H}" }}
wheels = [
    {{ url = "https://example.invalid/numpy-2.5.4-cp312-cp312-linux_x86_64.whl", hash = "{H}" }},
]
'''
fails = 0
def check(name, cond, got=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f": {got}")); fails += not cond

def run(name, text):
    d = tempfile.mkdtemp(); f = os.path.join(d, name)
    open(f, "w").write(text)
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "check_hashes.py"), f], capture_output=True, text=True)
    return r.returncode, r.stdout

code, out = run("uv.lock", LOCK.format(torch=f', hash = "{H}"', H=H))
check("uv.lock: every artifact hashed passes", code == 0, out)
code, out = run("uv.lock", LOCK.format(torch="", H=H))
check("uv.lock: an unhashed wheel is refused, by name", code == 1 and "torch 2.13.0+rocm7.2" in out and "numpy" not in out, out)
code, out = run("uv.lock", LOCK.format(torch=', hash = "sha256:1234"', H=H))
check("uv.lock: a malformed sha256 is refused", code == 1 and "torch" in out, out)
code, out = run("uv.lock", LOCK.format(torch=f', hash = "md5:{"ab" * 16}"', H=H))
check("uv.lock: another digest is refused", code == 1, out)
code, out = run("requirements-x.txt", f"# comment\n--index-url https://pypi.org/simple\nnumpy==2.5.4 \\\n    --hash={H} \\\n    --hash={H}\n    # via app\n")
check("requirements: hashed passes", code == 0, out)
code, out = run("requirements-x.txt", f"numpy==2.5.4 \\\n    --hash={H}\ntorch==2.13.0\n")
check("requirements: an unhashed requirement is refused", code == 1 and "torch==2.13.0" in out and "numpy" not in out, out)
r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "check_hashes.py")], capture_output=True, text=True)
check("this checkout's lock files all pass", r.returncode == 0, r.stdout)
print(f"{'FAILED' if fails else 'ok'}: {fails} failure(s)")
sys.exit(1 if fails else 0)
