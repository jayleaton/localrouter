"""The Kitten worker's CUDA-graph decode (`lm_decode: "graphs"`), loaded by kitten_worker.py when a tool asks for it.

Kept apart so the eager worker never imports transformers' compile machinery. See kitten_worker.py and
docs/dev/KITTEN-TTS2.md (Numerics: the graphs decode).
"""

import sys


def log(msg):
    print(f"kitten: {msg}", file=sys.stderr, flush=True)


def enable_graph_decode(tts, torch, opts):
    """lm_decode "graphs": the language model's decode step as CUDA graphs over one static KV cache.

    Eager decode is host-bound on GB10 (about 59 ms a token of which 25 ms are kernels: Python and launches). Here
    transformers' generate compiles the decode step (`compile_config`) because the cache is a StaticCache, and:
    - one cache serves every call (`reset()` zeroes it in place), so the graph's addresses never change and it is
      recorded once, not per request;
    - the speaker conditioning that kittenml's forward pre-hook adds at prefill is computed here, before generate,
      with the hook's own ops, and the hook is left a no-op: its prefill test reads the cache length, a GPU tensor
      with a StaticCache, which would sync and break the graph at every step;
    - attention is SDPA (flash-attention's varlen path does not capture);
    - the prefill stays eager (generate compiles only the decode step), so prompt lengths never recompile.
    A chunk whose prompt plus budget exceeds the cache (`graph_tokens`) takes the eager path. Returns the cache bytes.
    """
    import types
    from kittenml.kittentts2.logits import build_logits_processors
    from kittenml.kittentts2.prompt import build_generation_prompt
    from transformers import CompileConfig, StaticCache
    m = tts.model
    if getattr(m.config, "_attn_implementation", "sdpa") != "sdpa":
        m.set_attn_implementation("sdpa")
    cfg, length = m.config, opts.graph_tokens
    cache = StaticCache(config=cfg, max_cache_len=length)
    if hasattr(cache, "early_initialization"):  # the tensors now, so the first capture already sees their addresses
        cache.early_initialization(batch_size=1, num_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
                                   dtype=tts.dtype, device=m.device)
    # A step torch cannot capture would otherwise only log "skipping cudagraphs" and run uncaptured: make it an error
    # (and warm_graph_decode refuses the load on any recorded skip), so "graphs" never quietly means eager speed.
    from torch._dynamo import config as dynamo_config
    from torch._inductor import config as inductor_config
    inductor_config.triton.cudagraph_or_error = True
    if opts.graph_backend == "inductor":
        compile_config = CompileConfig(fullgraph=True, dynamic=False, mode="reduce-overhead")
    else:
        # the static cache's layers update their own tensors in place, which this backend captures only when allowed
        dynamo_config.cudagraph_backend_support_input_mutation = True
        compile_config = CompileConfig(backend="cudagraphs", mode="default", fullgraph=True, dynamic=False)
    eager = tts._generate_tokens

    def generate_tokens(self, reference, chunk, settings, max_new_tokens, use_reference_prompt, emotion_ids,
                        advanced):
        prompt = build_generation_prompt(
            self.token_map, self.tokenizer.encode(chunk, add_special_tokens=False),
            reference_prefix=reference["prefix"] if use_reference_prompt else None, emotion_ids=emotion_ids)
        if len(prompt) + int(max_new_tokens) > length:
            log(f"graphs: a {len(prompt)}-token prompt with a {max_new_tokens}-token budget exceeds graph_tokens "
                f"{length}; this chunk decodes eagerly")
            return eager(reference, chunk, settings, max_new_tokens, use_reference_prompt, emotion_ids, advanced)
        # kittenml's model.py _generate_tokens, step for step, but for the speaker embedding and the cache
        processors = build_logits_processors(
            self.token_map, temperature=settings["temperature"], top_k=settings["top_k"], top_p=settings["top_p"],
            min_p=settings.get("min_p", 0.0), repetition_penalty=advanced["repetition_penalty"],
            repetition_window=advanced["repetition_window"], token_run_penalty=advanced["token_run_penalty"],
            token_run_grace=advanced["token_run_grace"], prompt_length=len(prompt))
        input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        with torch.no_grad():
            embeds = m.get_input_embeddings()(input_ids)  # speaker.py's hook: position 0 carries the speaker
            spk = m.spk_proj(reference["embedding"].to(embeds.dtype)).unsqueeze(1)
            embeds = torch.cat([spk, embeds[:, 1:, :]], dim=1)
            m._pending_speaker_embedding = None
            cache.reset()
            output = m.generate(
                input_ids, inputs_embeds=embeds, past_key_values=cache, compile_config=compile_config,
                max_new_tokens=int(max_new_tokens), do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                repetition_penalty=1.0, logits_processor=processors,
                eos_token_id=[self.token_map.speech_end_id, self.token_map.stop_id],
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None) or 0)
        low, high = self.token_map.audio_id_base, self.token_map.audio_id_end
        return [t - low for t in output[0].tolist()[len(prompt):] if low <= t < high]

    tts._generate_tokens = types.MethodType(generate_tokens, tts)
    item = torch.tensor([], dtype=tts.dtype).element_size()
    return length * 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * item


def warm_graph_decode(tts, torch, voice):
    """Compiles and records the decode graph at load (cudagraph trees: warm-up, record, replay), leaving the random
    state as it was, so the first request is neither slow nor seeded differently."""
    reference = tts._voice_reference(voice)
    advanced = tts._resolve_advanced(None)
    settings = tts._resolve_preset(tts.default_preset, {})
    cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
    for i in range(3):
        tts._generate_tokens(reference, f"Warming up the decoder, take {i + 1}.", settings, 24,
                             tts.use_reference_prompt, None, advanced)
    torch.cuda.synchronize()
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state(gpu)
    from torch._dynamo.utils import counters
    skips = counters["inductor"].get("cudagraph_skips", 0)
    if skips:
        raise RuntimeError(f"graphs: torch skipped CUDA-graph capture {skips} time(s) during warm-up (see the log); "
                           "the decode would run uncaptured")


