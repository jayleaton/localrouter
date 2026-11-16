# comfyui: a `cmd` worker that serves image and video tools through ComfyUI

`worker.py` turns any ComfyUI workflow into a LocalRouter `image` (`text_to_image`) or `video` (`text_to_video`) tool,
so a model that ComfyUI runs (and LocalRouter's compiled-in engines do not, or not on this GPU) is served by
`/v1/images/generations`, `/v1/videos`, MCP `generate_image` / `generate_video` and the voice loop's picture and video
tools, scheduled and labelled like any other model. Standard library only: it runs under the system `python3 -I`.

- **load**: starts ComfyUI from `options.comfy_dir` with `options.python` (its own venv) on a loopback port
  (`options.port`, else a free one) with a private `--output-directory`, `options.args` and `options.env`, and waits
  for `/system_stats`. Or `options.url`: a ComfyUI you run yourself.
- **generate**: copies the API-format workflow (`options.workflow`, "Export (API)" in ComfyUI), writes the request into
  the inputs `options.bindings` names, queues it on `/prompt`, reports progress from `/ws` (denoise steps, then each
  node), waits on `/history`, and fetches the result through `/view`: `0.png`, `1.png`, ... (one run per image, seed
  +1 each) or `video.mp4`. A field the bindings do not name (negative prompt, steps, guidance, input images) is
  refused, never ignored.
- **input images**: an edit's `references` and `mask` and a video's `first_frame` are uploaded to ComfyUI
  (`/upload/image`, into the worker's private input directory) and wired in through a LoadImage node the worker adds:
  its IMAGE output, or for a mask its MASK output (1 where the mask is transparent, where an edit goes).
- **modes**: `options.modes.image_edit` / `image_to_video` give a request with input images its own workflow and
  bindings (merged over the tool's; a JSON `null` drops a binding): `{"image_edit": {"workflow": "edit.json",
  "bindings": {"references": [...]}}}`. `defaults` fills `steps`, `guidance` or `negative_prompt` when the request
  leaves them unset (0 or absent). `image_edit` may also set `unmasked` (the same fields), used when a request has
  input images but no mask, so a whole-scene restyle can use heavier defaults than a masked local edit. A request
  may still set size, steps or guidance. A tool without the mode refuses such requests. An edit without a size
  keeps whatever size its workflow derives from the input image.
- **cancel**: answers its load with `"cancel": "signal"`, so a cancel is SIGUSR1: the prompt is dropped from the queue,
  ComfyUI is interrupted, and the worker stays loaded.
- **exit / unload**: SIGTERM to ComfyUI's process group, progress lines while it stops, SIGKILL after
  `stop_timeout_s` (30); the private output directory is removed.

Bindings map a request field to `"node.input"` (or a list, every target set): `prompt` (required), `negative_prompt`,
`width`, `height`, `seed`, `steps`, `guidance`, the file inputs `references` (a list: the n-th target takes the n-th
image, so its length is the most a request may send), `mask` and `first_frame`, and for video `frames` (seconds x fps, snapped up to the model's grid with
`frame_multiple` / `frame_offset`), `fps` and `audio` (the input removed when a request asks for no sound). Limits:
`size_multiple` (8), `min_side`, `max_side`, `max_pixels`, `fps` (the rates it takes). A video tool's own size for
requests without one: `default_width` and `default_height` (read by the daemon). Memory is the tool's measured
`resident_bytes` / `working_bytes`.

```json
{"id": "minimax-h3-image", "kind": "image", "capabilities": ["text_to_image"],
 "cmd": ["/usr/bin/python3", "-I", "tools/comfyui/worker.py"], "resident_bytes": 100931731456,
 "options": {"comfy_dir": "/opt/ComfyUI", "python": "/opt/ComfyUI/venv/bin/python", "port": 8189,
             "args": ["--disable-mmap", "--cache-none"],
             "workflow": "/opt/workflows/minimax-h3-image-t2i.json",
             "bindings": {"prompt": "4.prompt", "width": "4.width", "height": "4.height", "seed": "6.seed", "steps": "6.steps"},
             "size_multiple": 32, "min_side": 256, "max_side": 2048}}
```

`comfy_events.py` is the worker's /ws progress client. Each tool runs its own ComfyUI (one model per process, unloaded as a whole by the scheduler). Tests (no GPU, a fake
ComfyUI, 29 checks): `python3 -I tools/comfyui/test_worker.py`.
