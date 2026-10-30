"""LocalRouter's launcher for KnowLine-4B's /v1/systemone server (typed decisions: choice, noul, score).

It runs the model repository's own front end, `knowline_server.py` with its `knowline_layout.py` (Apache-2.0), so the
prompt layout, the read-twice rule, the label tokens and the readout are exactly the authors'. Only the backend's
forward pass changes: the stock transformers backend computes the whole vocabulary's logits at every prompt position
(a working-memory peak that grows with the prompt, about 1 GB at 2,000 tokens); here only the last position's, which
is the one it reads. Weights are the shipped bf16, text only, on CUDA, copied in one tensor at a time (`load`).

Both front-end files are checked against the hashes reviewed for this launcher before anything is imported from the
weights directory, so a changed file there is refused rather than run.

    serve.py WEIGHTS_DIR --port N [--served-model-name ID]   (the openai_server engine's `options.server`)
"""

import argparse
import hashlib
import os
import sys

# The front end reviewed for this launcher: PelaAI/KnowLine-4B-Gen5.5 at 779944d2cc8ffa19cef386aee83c83bcb967ade2.
REVIEWED = {
    "knowline_server.py": "1b4c73419ce597cbf5635a8171b2fe981b577a9cf4856fb9387ce261e81a3729",
    "knowline_layout.py": "aa0fb35619f4883edd6c3bd1cc6dc259f41f082cfe0471e586c60d783da2af9d",
}


def verify(weights):
    for name, want in REVIEWED.items():
        with open(os.path.join(weights, name), "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != want:
            sys.exit(f"{name} in {weights} is not the reviewed file (sha256 {got}); refusing to run it")


def tensors(path, wanted):
    """The tensors of a safetensors file named in `wanted`, one host copy at a time (read, not mapped: a mapped shard
    kept every tensor read from it in the process until the file closed, up to 4 GB on DGX-01)."""
    import json
    import mmap
    import struct
    import torch
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    with open(path, "rb", buffering=0) as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for k, meta in header.items():
            if k == "__metadata__" or k not in wanted:
                continue
            start, end = meta["data_offsets"]
            if end == start:
                yield k, torch.empty(meta["shape"], dtype=dtypes[meta["dtype"]])
                continue
            # Its own anonymous mapping, unmapped after the copy: a heap buffer would stay in the process once freed
            # (glibc raises its mmap threshold after each large free), 0.65 GB more host memory while serving.
            buf = mmap.mmap(-1, end - start)
            f.seek(base + start)
            f.readinto(buf)
            t = torch.frombuffer(buf, dtype=dtypes[meta["dtype"]]).view(meta["shape"])
            yield k, t
            del t
            buf.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--served-model-name", default="knowline-4b")
    a = p.parse_args()
    weights = os.path.abspath(a.weights)
    verify(weights)
    sys.path.insert(0, weights)

    import torch
    import knowline_server as ks

    def score(self, text, media, ids):
        if media:
            raise ValueError("text only: pictures need the SGLang backend")
        inputs = torch.tensor([self.tok.encode(text, add_special_tokens=False)], device=self.model.device)
        with self.lock, torch.inference_mode():
            logits = self.model(input_ids=inputs, logits_to_keep=1).logits[0, -1].float().log_softmax(-1)
        return [float(logits[i]) for i in ids], int(inputs.shape[1])

    def load(self, model, device=None, dtype="bfloat16"):
        # The model is built on the GPU (its buffers, such as the rotary frequencies, computed there as usual), then
        # every checkpoint tensor is copied in, one at a time. transformers' own loader held up to 7 GB of host copies
        # until the end of the load (15.7 GiB at load against 11.5 GiB serving on DGX-01); this keeps the load within
        # the serving footprint. Same weights: the MTP head, which no decision reads, is not loaded (as by
        # transformers), and lm_head is tied to the embeddings (as in the config).
        import json
        import transformers
        from transformers import AutoConfig, AutoTokenizer
        self.torch = torch
        cfg = AutoConfig.from_pretrained(model)
        cls = getattr(transformers, "AutoModelForMultimodalLM", transformers.AutoModelForImageTextToText)
        with torch.device(device or "cuda"):
            m = cls.from_config(cfg, dtype=getattr(torch, dtype))
        params = m.state_dict()
        index = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
        missing = set(params) - set(index) - {"lm_head.weight"}
        if missing:
            raise RuntimeError(f"the checkpoint lacks {len(missing)} tensors the model has, e.g. {sorted(missing)[:3]}")
        with torch.no_grad():
            for shard in sorted(set(index.values())):
                for k, t in tensors(os.path.join(model, shard), set(params)):
                    params[k].copy_(t)
                    del t
        m.tie_weights()
        torch.cuda.empty_cache()  # the random initialisation's temporaries: not part of the serving footprint
        self.model = m.eval()
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lock = __import__("threading").Lock()

    if os.environ.get("KNOWLINE_STOCK_LOAD") != "1":  # 1: transformers' own loader (for comparing the two)
        ks.HF.__init__ = load
    ks.HF.score = score
    sys.argv = ["knowline_server.py", "--model", weights, "--backend", "hf", "--device", "cuda", "--host", "127.0.0.1",
                "--port", str(a.port), "--served-model-name", a.served_model_name, "--workers", "1"]
    ks.main()


if __name__ == "__main__":
    main()
