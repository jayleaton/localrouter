"""Export the audio harness dumps of jobs A and B as a capture in the twin's rec.py format, for central's K6 gate.

  python -I -B export_rec.py --out-a OUT_A --out-b OUT_B --dest DEST [--which native|kernel_acc0|kernel_acc1]
                             [--sample-rate HZ] [--allow-smoke]

Not one of the six sealed kit files (jobs.py records HARNESS_FILES); it only reads a finished OUT_A / OUT_B.

Input: each OUT holds results-<A|B>.json, whose "dumps" list {name, file, shape, sha256, case} for raw little-endian
fp32 files under tensors/ (jobs.py Recorder.dump). Every dump used is size-checked and sha256-verified against that
list before anything is written.

Output: DEST in the format of tools/twin/stk_twin/rec.py, as read by stk_twin.kandinsky6 (read_capture / compare) and
src/engines/qwen_image/capture.zig:
  DEST/ops.jsonl      one record a line, "i" in order: op {i, name, kind, attrs, ins, outs, launches} or
                      note {i, note, ...}
  DEST/blobs/<sha>    raw C-contiguous little-endian bytes; the file name is the sha256 of the bytes
  ref                 {"sha": hex, "dtype": "float32" | "int16", "shape": [...]}

Ops, with central's C1 roles (attrs whose name starts with note_ are informational and never gated):
  k6.audio.latent_in  outs.latent f32 [1,218,40]              decoder/latent_rows (job A): the sampler's final state
                                                              before the transpose and the scaling division
  k6.audio.mel        ins.latent, outs.mel  f32 [1,128,436]   decoder/native_mel or decoder/kernel_mel_accN (job A)
  k6.audio.wave       ins.mel, outs.wave    f32 [1,1,223232]  vocoder/native_pcm or vocoder/kernel_pcm_accN (job B);
                                                              ins.mel is job B's vocoder input vocoder/mel
  k6.audio.pcm16      ins.wave, outs.pcm    i16 [223232]      trunc(clip(wave, -1, 1) * 32767) in float32, numpy
                                                              astype(int16): upstream postprocess_audio, mode "clip"
Every op carries the gated attr "chained": true only when each op's input is bit-identical to the previous op's output
(sha256), so the capture is one run of the chain. jobs.py seeds job B's mel itself, so normally wave's ins.mel is not
k6.audio.mel's output: then every op says chained=false, which central treats as standalone operator checks only, never
chain qualification. The "k6a_export" note records every source (job, dump name, file, sha256, case and its status) and
the results JSONs' sha256.

Refuses: an existing DEST, a missing or duplicated dump, a size or sha256 mismatch, a non-finite wave or one outside
[-1, 1], and (without --allow-smoke) dry-run / synthetic-weight results or shapes other than the preset's.
Standard library and numpy only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

FORMAT = "stk_twin/rec.py ops.jsonl + blobs/<sha256>"
EXPORTER = "kernels/cuda/kandinsky6/audio/harness/export_rec.py"
PRESET = {"latent_in": [1, 218, 40], "mel": [1, 128, 436], "wave": [1, 1, 223232]}
ACCS = (0, 1)  # jobs.py ACCS (k6a_cuda ACC_FMA, ACC_MUL_ADD)


class Refuse(Exception):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


class Results:
    """One job's OUT dir: results-<job>.json and its verified dumps."""

    def __init__(self, out: str, job: str):
        self.out, self.job = os.path.realpath(out), job
        self.path = os.path.join(self.out, f"results-{job}.json")
        if not os.path.isfile(self.path):
            raise Refuse(f"{self.path}: missing")
        self.sha256 = sha256_file(self.path)
        with open(self.path) as f:
            self.doc = json.load(f)
        if self.doc.get("job") != job:
            raise Refuse(f"{self.path}: job {self.doc.get('job')!r}, expected {job!r}")
        self.dumps: dict[str, dict] = {}
        for rec in self.doc.get("dumps", []):
            if rec["name"] in self.dumps:
                raise Refuse(f"{self.path}: dump {rec['name']!r} listed twice")
            self.dumps[rec["name"]] = rec
        self.cases = {c.get("name"): c.get("status") for c in self.doc.get("cases", [])}

    def smoke(self) -> list[str]:
        why = []
        if self.doc.get("dry_run_cpu"):
            why.append(f"job {self.job}: dry_run_cpu")
        if self.doc.get("weights") != "real":
            why.append(f"job {self.job}: weights {self.doc.get('weights')!r}")
        return why

    def load(self, name: str) -> tuple[np.ndarray, dict]:
        """The dump as a float32 array of its recorded shape, after checking size and sha256."""
        rec = self.dumps.get(name)
        if rec is None:
            raise Refuse(f"{self.path}: no dump {name!r} (have {sorted(self.dumps)})")
        rel = rec["file"]
        path = os.path.realpath(os.path.join(self.out, rel))
        if os.path.isabs(rel) or not path.startswith(self.out + os.sep):
            raise Refuse(f"{self.path}: dump {name!r} file {rel!r} is outside {self.out}")
        with open(path, "rb") as f:
            b = f.read()
        shape = [int(d) for d in rec["shape"]]
        if len(b) != numel(shape) * 4:
            raise Refuse(f"{path}: {len(b)} bytes for float32 {shape}")
        got = sha256_bytes(b)
        if got != rec["sha256"]:
            raise Refuse(f"{path}: sha256 {got}, results-{self.job}.json says {rec['sha256']}")
        src = {"job": self.job, "dump": name, "file": rel, "sha256": got, "shape": shape, "case": rec.get("case"),
               "case_status": self.cases.get(rec.get("case"))}
        return np.frombuffer(b, dtype="<f4").reshape(shape), src


def pcm16_from_wave(wave: np.ndarray) -> np.ndarray:
    """upstream postprocess_audio, mode "clip", batch 1: (np.clip(w, -1.0, 1.0) * 32767).astype(np.int16)."""
    if wave.ndim != 3 or wave.shape[0] != 1 or wave.shape[1] != 1:
        raise Refuse(f"wave shape {list(wave.shape)}: expected [1, 1, samples]")
    w = np.ascontiguousarray(wave.reshape(-1), dtype=np.float32)
    scaled = np.clip(w, -1.0, 1.0) * 32767
    if scaled.dtype != np.float32:
        raise Refuse(f"clip(wave) * 32767 is {scaled.dtype}, not float32 (numpy {np.__version__})")
    return scaled.astype(np.int16)


class Writer:
    def __init__(self, dest: str):
        self.dest, self.lines, self.i = dest, [], 0
        os.mkdir(dest)  # refuses an existing DEST (FileExistsError), also under a race
        os.mkdir(os.path.join(dest, "blobs"))

    def ref(self, a: np.ndarray, dtype: str) -> dict:
        le = {"float32": "<f4", "int16": "<i2"}[dtype]
        b = np.ascontiguousarray(a, dtype=le).tobytes()
        sha = sha256_bytes(b)
        p = os.path.join(self.dest, "blobs", sha)
        if not os.path.exists(p):
            with open(p, "wb") as f:
                f.write(b)
        return {"sha": sha, "dtype": dtype, "shape": [int(d) for d in a.shape]}

    def note(self, name: str, **data) -> None:
        self.lines.append(json.dumps({"i": self.i, "note": name, **data}))
        self.i += 1

    def op(self, name: str, attrs: dict, ins: dict, **outs: dict) -> None:
        self.lines.append(json.dumps({"i": self.i, "name": name, "kind": "audio", "attrs": attrs, "ins": ins,
                                      "outs": outs, "launches": []}))
        self.i += 1

    def finish(self) -> None:
        """ops.jsonl last, by rename: a DEST without it is an interrupted export."""
        tmp = os.path.join(self.dest, "ops.jsonl.tmp")
        with open(tmp, "w") as f:
            f.write("".join(line + "\n" for line in self.lines))
        os.replace(tmp, os.path.join(self.dest, "ops.jsonl"))


def export(a) -> dict:
    if os.path.lexists(a.dest):
        raise Refuse(f"{a.dest}: exists; refusing to overwrite")
    if a.which == "native":
        mel_name, wave_name = "decoder/native_mel", "vocoder/native_pcm"
    elif a.which in tuple(f"kernel_acc{n}" for n in ACCS):
        acc = a.which.removeprefix("kernel_acc")
        mel_name, wave_name = f"decoder/kernel_mel_acc{acc}", f"vocoder/kernel_pcm_acc{acc}"
    else:
        raise Refuse(f"--which {a.which!r}: native or kernel_acc{{{','.join(map(str, ACCS))}}}")

    ra, rb = Results(a.out_a, "A"), Results(a.out_b, "B")
    smoke = ra.smoke() + rb.smoke()
    latent, s_latent = ra.load("decoder/latent_rows")
    mel, s_mel = ra.load(mel_name)
    voc_in, s_voc_in = rb.load("vocoder/mel")
    wave, s_wave = rb.load(wave_name)
    for key, arr in (("latent_in", latent), ("mel", mel), ("wave", wave)):
        if list(arr.shape) != PRESET[key]:
            smoke.append(f"{key} shape {list(arr.shape)}, preset {PRESET[key]}")
    if list(voc_in.shape) != list(mel.shape):
        smoke.append(f"vocoder input mel shape {list(voc_in.shape)} != k6.audio.mel shape {list(mel.shape)}")
    if smoke and not a.allow_smoke:
        raise Refuse("not a preset real-weight run (" + "; ".join(smoke) + "); pass --allow-smoke to export anyway")
    if not np.isfinite(wave).all():
        raise Refuse(f"{wave_name}: non-finite samples")
    if float(np.abs(wave).max(initial=0.0)) > 1.0:
        raise Refuse(f"{wave_name}: samples outside [-1, 1] (max |x| {float(np.abs(wave).max())})")
    pcm = pcm16_from_wave(wave)

    chained = s_mel["sha256"] == s_voc_in["sha256"]
    consistency = {
        "k6.audio.mel_sha256": s_mel["sha256"], "vocoder_input_mel_sha256": s_voc_in["sha256"], "chained": chained,
        "statement": ("job B's vocoder input is bit-identical to k6.audio.mel: k6.audio.wave = vocoder(k6.audio.mel)"
                      if chained else
                      "job B's vocoder input differs from k6.audio.mel (jobs.py case_vocoder_chain seeds its own mel): "
                      "k6.audio.wave's ins.mel is job B's own mel, NOT k6.audio.mel's output; every op is chained=false, "
                      "a standalone operator check, never chain qualification"),
    }
    sr = {} if a.sample_rate is None else {"sample_rate": a.sample_rate}

    dest = os.path.abspath(a.dest)
    w = Writer(dest)
    try:
        refs = {"latent": w.ref(latent, "float32"), "mel": w.ref(mel, "float32"),
                "voc_in": w.ref(voc_in, "float32"), "wave": w.ref(wave, "float32"), "pcm": w.ref(pcm, "int16")}
        meta = {
            "format": FORMAT, "exporter": EXPORTER, "exporter_sha256": sha256_file(os.path.abspath(__file__)),
            "numpy": np.__version__, "which": a.which, "smoke": bool(smoke), "smoke_reasons": smoke,
            "results": {j: {"path": r.path, "sha256": r.sha256, "finished": r.doc.get("finished"),
                            "dry_run_cpu": r.doc.get("dry_run_cpu"), "weights": r.doc.get("weights"),
                            "started_utc": r.doc.get("started_utc"), "exit_reason": r.doc.get("exit_reason")}
                        for j, r in (("A", ra), ("B", rb))},
            "sources": {"k6.audio.latent_in": s_latent, "k6.audio.mel": s_mel,
                        "k6.audio.wave.ins.mel": s_voc_in, "k6.audio.wave": s_wave,
                        "k6.audio.pcm16": {"derived_from": "k6.audio.wave", "wave_sha256": s_wave["sha256"],
                                           "formula": "(np.clip(wave.reshape(-1) as float32, -1.0, 1.0) * 32767)"
                                                      ".astype(np.int16)"}},
            "mel_consistency": consistency,
        }
        w.note("k6a_export", **meta)
        c = {"chained": chained}
        w.op("k6.audio.latent_in", {**c, "note_source": "A:decoder/latent_rows", "note_smoke": bool(smoke)}, {},
             latent=refs["latent"])
        w.op("k6.audio.mel", {**c, "note_source": f"A:{mel_name}", "note_smoke": bool(smoke)},
             {"latent": refs["latent"]}, mel=refs["mel"])
        w.op("k6.audio.wave", {**sr, **c, "note_source": f"B:{wave_name}", "note_input": "B:vocoder/mel",
                               "note_smoke": bool(smoke)}, {"mel": refs["voc_in"]}, wave=refs["wave"])
        w.op("k6.audio.pcm16", {**sr, **c, "normalization": "clip", "note_source": "derived from k6.audio.wave",
                                "note_smoke": bool(smoke)}, {"wave": refs["wave"]}, pcm=refs["pcm"])
        w.finish()
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)  # only ever our own fresh directory (mkdir above)
        raise
    return {"dest": dest, "which": a.which, "smoke": bool(smoke), "chained": chained,
            "ops": {"k6.audio.latent_in": refs["latent"], "k6.audio.mel": refs["mel"],
                    "k6.audio.wave.ins.mel": refs["voc_in"], "k6.audio.wave": refs["wave"],
                    "k6.audio.pcm16": refs["pcm"]}}


def main() -> int:
    ap = argparse.ArgumentParser(description="Export K6 audio harness dumps (jobs A, B) as a rec.py capture.")
    ap.add_argument("--out-a", required=True, help="job A's OUT (results-A.json, tensors/)")
    ap.add_argument("--out-b", required=True, help="job B's OUT (results-B.json, tensors/)")
    ap.add_argument("--dest", required=True, help="new capture directory; must not exist")
    ap.add_argument("--which", default="native", help="native (default) or kernel_acc0 / kernel_acc1")
    ap.add_argument("--sample-rate", type=int, default=None,
                    help="recorded as the wave/pcm16 sample_rate attr when given (central's capture records it)")
    ap.add_argument("--allow-smoke", action="store_true",
                    help="export dry-run / synthetic-weight results or non-preset shapes, labelled smoke")
    a = ap.parse_args()
    try:
        rep = export(a)
    except (Refuse, FileExistsError) as e:
        print(json.dumps({"k6a_export": {"ok": False, "error": f"{type(e).__name__}: {e}"}}))
        return 1
    print(json.dumps({"k6a_export": {"ok": True, **rep}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
