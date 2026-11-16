# MiniMax H3-Image GGUF loader for ComfyUI (installed into custom_nodes by deploy/strix-halo/comfyui/install.sh).
# The unsloth/MiniMax-H3-Image-GGUF files carry the model's 6 PDD output heads but not the safetensors "config"
# metadata ComfyUI's model detection reads (pdd_steps / pdd_sigmas). This node loads the GGUF through ComfyUI-GGUF
# and supplies the config Comfy-Org's minimax_h3_image_pruned_* safetensors carry. It modifies no other files.
import json
_PDD = {"transformer": {"pdd_steps": 6, "pdd_sigmas": [1.0, 0.9675236940383911, 0.8875485062599182, 0.75,
                                                        0.5327739119529724, 0.23200887441635132, 0.0]}}

def _gguf_loader_cls():
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
    if cls is None:
        raise RuntimeError("ComfyUI-GGUF is not loaded")
    return cls

class UnetLoaderGGUFMiniMaxH3Image:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        names = [x for x in folder_paths.get_filename_list("unet_gguf")] if "unet_gguf" in folder_paths.folder_names_and_paths else folder_paths.get_filename_list("unet")
        return {"required": {"unet_name": (names,)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"
    TITLE = "Unet Loader (GGUF, MiniMax-H3-Image config)"

    def load(self, unet_name):
        import comfy.sd, folder_paths
        g = _gguf_loader_cls()
        orig = comfy.sd.load_diffusion_model_state_dict
        def patched(sd, model_options={}, metadata=None, **kw):
            md = dict(metadata or {})
            md.setdefault("config", json.dumps(_PDD))
            return orig(sd, model_options=model_options, metadata=md, **kw)
        comfy.sd.load_diffusion_model_state_dict = patched
        try:
            return g().load_unet(unet_name)
        finally:
            comfy.sd.load_diffusion_model_state_dict = orig

NODE_CLASS_MAPPINGS = {"UnetLoaderGGUFMiniMaxH3Image": UnetLoaderGGUFMiniMaxH3Image}
NODE_DISPLAY_NAME_MAPPINGS = {"UnetLoaderGGUFMiniMaxH3Image": UnetLoaderGGUFMiniMaxH3Image.TITLE}
