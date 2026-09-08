"""Export the text backbone without loading the full FireRedAudio model."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "backbone_llm."
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "tokenizer.model", "vocab.json", "merges.txt", "chat_template.jinja",
)


def backbone_key(key: str) -> str:
    return key.removeprefix(PREFIX).replace("model.language_model.", "model.", 1)


def text_rope_config(backbone: dict) -> dict:
    config = deepcopy(backbone)
    # Audio/text positions have identical T/H/W axes. MRoPE therefore reduces
    # exactly to ordinary RoPE; vLLM 0.18's MRoPE scheduler requires token IDs,
    # which are absent for prompt_embeds requests.
    for key in ("rope_parameters", "rope_scaling"):
        if isinstance(config.get(key), dict):
            config[key].pop("mrope_section", None)
            config[key].pop("mrope_interleaved", None)
    return config


def weight_map(source: Path) -> dict[str, str]:
    index = source / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text())["weight_map"]
    with safe_open(source / "model.safetensors", framework="pt") as weights:
        return {key: "model.safetensors" for key in weights.keys()}


def export_backbone(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "firered_audio":
        raise ValueError("Expected a FireRedAudio checkpoint")
    backbone = dict(config["backbone_config"])
    if backbone.get("model_type") != "qwen3_5_text":
        raise ValueError("Only the dense Qwen3.5 backbone is supported")
    mapping = weight_map(source)
    selected = {k: v for k, v in mapping.items() if k.startswith(PREFIX)}
    for required in ("model.embed_tokens.weight", "lm_head.weight"):
        if required not in {backbone_key(k) for k in selected}:
            raise ValueError(f"Missing backbone weight: {required}")
    if len({backbone_key(k) for k in selected}) != len(selected):
        raise ValueError("Conflicting backbone key namespaces")
    if output.exists():
        raise FileExistsError(f"Output must be a new directory: {output}")
    for shard in set(selected.values()):
        if not (source / shard).is_file():
            raise FileNotFoundError(source / shard)
    output.mkdir(parents=True)
    backbone["architectures"] = ["Qwen3_5ForCausalLM"]
    backbone["dtype"] = backbone.get("dtype", config.get("dtype", "bfloat16"))
    (output / "config.json").write_text(json.dumps(text_rope_config(backbone), indent=2) + "\n")
    output_map, total_size = {}, 0
    shards = sorted(set(selected.values()))
    # Keep at most one original shard in CPU memory; tensors retain their dtype.
    for i, shard in enumerate(shards, 1):
        with safe_open(source / shard, framework="pt", device="cpu") as reader:
            tensors = {
                backbone_key(k): reader.get_tensor(k)
                for k, filename in selected.items() if filename == shard
            }
            filename = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            save_file(tensors, output / filename, metadata={"format": "pt"})
            output_map.update({k: filename for k in tensors})
            total_size += sum(t.numel() * t.element_size() for t in tensors.values())
            del tensors
        print(f"Exported {filename}", flush=True)
    (output / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_size}, "weight_map": output_map,
    }, indent=2) + "\n")
    for filename in TOKENIZER_FILES:
        if (source / filename).is_file():
            shutil.copy2(source / filename, output / filename)

    # SGLang recognizes hybrid state allocation by the upstream architecture name.
    # The external model package supplies a text-only implementation of that name.
    sglang_view = output / "sglang"
    sglang_view.mkdir()
    sglang_config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": backbone,
        "dtype": backbone["dtype"],
        "tie_word_embeddings": backbone.get("tie_word_embeddings", False),
    }
    (sglang_view / "config.json").write_text(json.dumps(sglang_config, indent=2) + "\n")
    for path in output.iterdir():
        if path.is_file() and path.name != "config.json":
            (sglang_view / path.name).symlink_to(Path("..") / path.name)
    manifest = {"source_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
                "backbone_bytes": total_size,
                "weight_count": len(output_map), "format_version": 1}
    (output / "fireredaudio_export.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_backbone(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
