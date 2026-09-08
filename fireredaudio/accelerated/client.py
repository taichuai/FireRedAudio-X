"""HTTP transport for the engines' precomputed prompt embedding interfaces."""

import base64
import io
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch


def generate(base_url, backend, embeds, *, eos_id, max_tokens=300,
             temperature=0.0, repetition_penalty=1.1, seed=42, timeout=300):
    if embeds.ndim != 2 or not embeds.shape[0] or not torch.isfinite(embeds).all():
        raise ValueError("Expected finite, nonempty [sequence, hidden_size] embeddings")
    sampling = dict(temperature=temperature, top_p=0.8 if temperature else 1.0,
                    top_k=20 if temperature else -1, repetition_penalty=repetition_penalty,
                    stop_token_ids=[eos_id], skip_special_tokens=True)
    if backend == "vllm":
        buffer = io.BytesIO()
        torch.save(embeds.cpu(), buffer)
        payload = dict(model="fireredaudio", prompt_embeds=base64.b64encode(buffer.getvalue()).decode("ascii"),
                       max_tokens=max_tokens, seed=seed, **sampling)
        endpoint = "/v1/completions"
    elif backend == "sglang":
        # Supplying input_ids too makes SGLang discard input_embeds.
        payload = dict(input_embeds=embeds.float().tolist(),
                       sampling_params=dict(max_new_tokens=max_tokens, **sampling))
        endpoint = "/generate"
    else:
        raise ValueError(f"Unknown backend: {backend}")
    request = Request(base_url.rstrip("/") + endpoint,
                      data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"{backend} HTTP {error.code}: {error.read().decode(errors='replace')}") from error
    if backend == "vllm":
        choice = result["choices"][0]
        return {"text": choice["text"], "finish_reason": choice.get("finish_reason"),
                "usage": result.get("usage")}
    return {"text": result["text"], "finish_reason": result.get("meta_info", {}).get("finish_reason"),
            "usage": result.get("meta_info")}
