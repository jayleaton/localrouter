CUTLASS headers pinned to e05f953 from the existing standalone proof.
Only the nvcc dependency closure is included. License: cutlass/LICENSE.txt.
The first residual convolution uses SM80 PTX and the explicit caller scratch ABI.
Standalone and GPU layout-wrapper fixture: 1,785,856 bytes exact, SHA256
6c6d275cefdf70ada5a378f91abfeda369abbd7e8e5c6d2d1b30edc51bc33573.
This does not qualify the full decoder or other convolution shapes.
The up1 block0 shortcut (1x1, 2048 to 1024, T 218) uses the same closure's GemmSplitKParallel and ReduceSplitK:
standalone fixture 892,928 bytes exact, default and profiled, SHA256
dbb05ec4b8ee78db01fc0c82329af79fe3362b30fda1a7bcedc6dded89d325df. Default off; fixture shape only.
