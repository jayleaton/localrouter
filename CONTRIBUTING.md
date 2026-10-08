# Contributing

Use Zig 0.17.0 and install ffmpeg for video output. From a checkout:

```sh
zig build
zig build test
```

No GPU or CUDA compiler is needed for these checks. Without `-Dnvcc`, CUDA fatbins are omitted; tests use the
GPU-free test engine and host checks. A hardware build uses:

```sh
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121,120
```

Keep changes focused, describe the resulting behavior and include the checks you ran. Never commit credentials,
machine identifiers, model weights, generated outputs or local configuration.

## New adapters

See [the architecture](docs/ARCHITECTURE.md#adding-a-model) for the engine interface and complete worker protocol.
A compiled-in adapter implements `engine.Entry` and is registered in `engine.entries`; declare only capabilities
that work. `needs` must cheaply predict resident and working memory without I/O. Load weights in `load`, write
outputs inside the job directory in `generate`, and release resources in `unload`.

An external adapter uses a `cmd` array and reads one JSON object per line from stdin. Reserve stdout for protocol
messages and log to stderr. Reply to `load` with `{"ok":true,"resident":<bytes>}`. For `generate`, read the supplied
request and job directory, optionally emit progress objects, then return `{"ok":true,"files":["0.png"],"seed":7,"ms":11800}`
or `{"ok":false,"error":"reason","type":"invalid_request_error"}`. Handle `exit` by releasing resources and exiting.
Input references are files in the job directory. Set `resident_bytes`, `working_bytes` and capabilities in the config.
Test lifecycle, memory accounting, unsupported requests, cancellation, worker failure and output handling.

## Engine correctness

Engine changes must preserve bit-exact results against the Python reference in `tools/twin`: test individual
operations, chained operations and full generation, including decoded pixels and audio samples, on the target GPU.
A successful compile or a visually similar output is insufficient. Performance changes ship only when those bits
are preserved. Include the hardware, precision, seeds, commands and comparison results with the change. A deliberate
numerical or precision change requires an explicit documented reference baseline and acceptance before release.
