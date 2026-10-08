"""The op recorder: every GPU op of the twin runs inside `rec.op(...)`, which (when recording) stores its inputs before
and its outputs after as content-addressed blobs, plus each Triton launch's cubin, grid and arguments. The capture is
the program the Zig engine replays op by op (alone: from the captured inputs; chained: from its own outputs).

Layout of a capture directory:
  ops.jsonl      one op a line: {i, name, kind, attrs, ins: {k: ref}, outs: {k: ref}, launches: [...]}
  blobs/<sha>    raw tensor bytes (contiguous); a ref is {sha, dtype, shape}
  cubins/<sha>   Triton cubins; a launch is {kernel, cubin, grid, num_warps, shared, args: [ref | scalar]}
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch


def _ref(t: torch.Tensor, blobs: Path) -> dict:
    data = t.detach().contiguous().view(torch.uint8).reshape(-1).cpu().numpy().tobytes() if t.numel() else b""
    sha = hashlib.sha256(data).hexdigest()
    p = blobs / sha
    if not p.exists():
        p.write_bytes(data)
    return {"sha": sha, "dtype": str(t.dtype).removeprefix("torch."), "shape": list(t.shape)}


class Op:
    """One op being recorded: `out(**tensors)` after it ran, `launch(compiled, grid, args)` per Triton launch."""

    def __init__(self, rec: "Rec", name: str, kind: str, attrs: dict, ins: dict):
        self.rec, self.name, self.kind, self.attrs = rec, name, kind, attrs
        self.ins = {k: _ref(v, rec.blobs) for k, v in ins.items()} if rec.on else {}
        self.outs: dict = {}
        self.launches: list = []

    def out(self, **tensors: torch.Tensor) -> None:
        if self.rec.on:
            torch.cuda.synchronize()
            self.outs.update({k: _ref(v, self.rec.blobs) for k, v in tensors.items()})

    def launch(self, compiled, grid, args) -> None:
        """A Triton launch: `compiled` is what `kernel[grid](...)` returned; tensors in `args` become refs."""
        if not self.rec.on:
            return
        cubin = compiled.asm["cubin"]
        sha = hashlib.sha256(cubin).hexdigest()
        p = self.rec.root / "cubins" / sha
        if not p.exists():
            p.write_bytes(cubin)
        md = compiled.metadata
        meta = {k: v for k, v in (md._asdict() if hasattr(md, "_asdict") else vars(md)).items()
                if isinstance(v, (int, float, str, bool)) or v is None}
        src = getattr(compiled, "src", None)
        sig = {str(k): str(v) for k, v in (getattr(src, "signature", None) or {}).items()}
        consts = {str(k): (v if isinstance(v, (int, float, bool, str)) else str(v))
                  for k, v in (getattr(src, "constants", None) or {}).items()}
        attrs = {str(k): str(v) for k, v in (getattr(src, "attrs", None) or {}).items()}
        enc = [{"ptr": _ref(a, self.rec.blobs)} if isinstance(a, torch.Tensor) else a for a in args]
        self.launches.append({"kernel": md.name, "cubin": sha, "grid": list(grid), "num_warps": md.num_warps,
                              "shared": md.shared, "meta": meta, "signature": sig, "constants": consts,
                              "attrs": attrs, "cubins_by_arch": self.rec.other_archs(compiled, sha), "args": enc})


class Rec:
    """Off by default (ops run at full speed); `start(dir)` records every op until `stop()`."""

    def __init__(self):
        self.on = False
        self.root: Path | None = None
        self.blobs: Path | None = None
        self.n = 0
        self._f = None
        self._aot: dict = {}

    def other_archs(self, compiled, sha: str) -> dict:
        """The same kernel (same source and specialization) compiled for each SM in $STK_AOT_ARCHS (e.g. "121", the
        Spark), so one capture on a PRO 6000 yields the cubins the GB10 build embeds. Compiled once per kernel."""
        out = {}
        for arch in filter(None, os.environ.get("STK_AOT_ARCHS", "").split(",")):
            key = (sha, arch)
            if key not in self._aot:
                import triton
                from triton.backends.compiler import GPUTarget

                ck = triton.compile(compiled.src, target=GPUTarget("cuda", int(arch), 32),
                                    options={"num_warps": compiled.metadata.num_warps,
                                             "num_stages": compiled.metadata.num_stages})
                data = ck.asm["cubin"]
                h = hashlib.sha256(data).hexdigest()
                (self.root / "cubins" / h).write_bytes(data)
                self._aot[key] = {"cubin": h, "shared": ck.metadata.shared, "name": ck.metadata.name}
            out[arch] = self._aot[key]
        return out

    def start(self, root: str | Path) -> None:
        self.root = Path(root)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        (self.root / "cubins").mkdir(exist_ok=True)
        self._f = open(self.root / "ops.jsonl", "a")
        self.on = True

    def stop(self) -> None:
        self.on = False
        if self._f:
            self._f.close()
            self._f = None

    def note(self, name: str, **data) -> None:
        """A non-op record (sigmas, shapes, the prompt) in the same stream."""
        if self.on:
            self._f.write(json.dumps({"i": self.n, "note": name, **data}) + "\n")
            self.n += 1

    @contextmanager
    def op(self, name: str, kind: str, attrs: dict | None = None, **ins: torch.Tensor):
        if self.on:
            torch.cuda.synchronize()
        o = Op(self, name, kind, attrs or {}, ins)
        yield o
        if self.on:
            self._f.write(json.dumps({"i": self.n, "name": name, "kind": kind, "attrs": o.attrs, "ins": o.ins,
                                      "outs": o.outs, "launches": o.launches}) + "\n")
            self.n += 1


REC = Rec()
