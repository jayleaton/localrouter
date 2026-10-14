"""Kandinsky-6 Pro-distill native reference ("oracle") capture and the bit gate against LocalRouter's native runtime.

Development and validation only: upstream kandinsky-6 eac06e39 is imported at run time from the pinned source tree
(never vendored), production stays native Zig, no ComfyUI. `capture` records the oracle in the twin's own capture
format (stk_twin/rec.py: ops.jsonl + content-addressed blobs/, read by Zig through src/engines/qwen_image/capture.zig);
`compare` gates two such directories op by op and role by role. Op names and roles are listed in README.md.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PIN_COMMIT = "eac06e39b904940baff2c408e925cdb0c4529269"  # kandinskylab/kandinsky-6
TORCH_PIN = "2.14.1+cu130"
# The full oracle prefix (receipts in k6-staging/oracle-full-v2-export). The minimal prefix lacks the upstream
# pipeline's import dependencies. The full closure preserves its Python, torch and numerical library bytes. It is a corrected
# install: one wheel's installed WHEEL tag and RECORD entry were rewritten (metadata-correction receipt), library
# bytes unchanged. It is not byte-identical official metadata and is never labelled official.
ORACLE = {
    "prefix": "/opt/k6-oracle",
    "python": "3.13.16",  # python-build-standalone
    "torch": TORCH_PIN,
    "cuda_major": "13",
    "numpy": "2.5.4",
    "archive": "k6-oracle-full-v2-prefix.tar.gz",
    "archive_bytes": 3629088624,
    "archive_sha256": "106e608bf85c4195bd64e83017d918c35dcefe29d6149e2591ebcc916d39bc85",
    "manifest_sha256": "e670043958ba3823ef157085f342c3a1c04cefb9efcada07b673c1f057acd395",
    "freeze_sha256": "18f5e02f48be9515ec817d2660829276fac3148a4d2f6bdfaa39524d8f4f4451",
    "metadata_correction_sha256": "38cd6454e0340808b30ec2733cb57c0096a5e24d211d471ad76c14f6af0580a9",
    "install": "corrected (metadata-correction receipt); not byte-identical official metadata",
}
QUALIFICATION = "unqualified"  # until a T0 GPU result exists; a CPU import or relocation pass qualifies nothing
AUX = "k6.aux."  # diagnostic ops: gated only when both captures carry them
# element size per dtype name as rec.py writes it (str(torch dtype) without "torch."); blobs are little-endian
ELEM = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8, "uint8": 1, "int8": 1, "bool": 1, "int16": 2,
        "int32": 4, "int64": 8}

if sys.byteorder != "little":
    raise ImportError("capture blobs are little-endian; this host is not")


def sha256_file(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def read_capture(root: str | Path) -> tuple[list[dict], list[dict]]:
    """(ops, notes) of a capture directory in run order; every ref is checked against its blob's size."""
    root = Path(root)
    ops, notes = [], []
    for ln, line in enumerate((root / "ops.jsonl").read_text().splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        if "note" in rec:
            notes.append(rec)
            continue
        for role in ("ins", "outs"):
            for key, ref in rec.get(role, {}).items():
                if ref.get("dtype") not in ELEM:
                    raise ValueError(f"{root}:{ln} {rec.get('name')} {role}.{key}: dtype {ref.get('dtype')!r}")
                size = (root / "blobs" / ref["sha"]).stat().st_size
                if size != numel(ref["shape"]) * ELEM[ref["dtype"]]:
                    raise ValueError(f"{root}:{ln} {rec.get('name')} {role}.{key}: blob has {size} bytes for "
                                     f"{ref['shape']} {ref['dtype']}")
        ops.append(rec)
    return ops, notes


def blob(root: str | Path, ref: dict) -> bytes:
    return (Path(root) / "blobs" / ref["sha"]).read_bytes()
