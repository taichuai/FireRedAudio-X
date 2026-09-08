"""Portable training state and streaming export into a complete HF checkpoint."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT_VERSION = 1
AUXILIARY_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "tokenizer.model", "vocab.json", "merges.txt", "chat_template.jinja",
    "processor_config.json", "preprocessor_config.json", "generation_config.json",
)


def canonical_key(key):
    return key.replace("backbone_llm.model.language_model.", "backbone_llm.model.", 1)


def weight_map(source):
    source = Path(source)
    index = source / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(source / "model.safetensors", framework="pt") as reader:
            mapping = {key: "model.safetensors" for key in reader.keys()}
    if not mapping or len({canonical_key(k) for k in mapping}) != len(mapping):
        raise ValueError("Empty or ambiguous checkpoint weight map")
    return mapping


def fingerprint(source):
    source = Path(source)
    mapping = weight_map(source)
    names = {"config.json", *mapping.values()}
    names.update(name for name in AUXILIARY_FILES if (source / name).is_file())
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode())
        digest.update(b"\0")
        with (source / name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def manifest_fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(model, optimizer, scheduler, sampler, step, directory, metadata, rng):
    directory = Path(directory)
    state = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    path = directory / f"step-{step:08d}.pt"
    payload = {"format_version": FORMAT_VERSION, "step": step,
               "trainable_state_dict": state, "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "sampler": sampler.state_dict(),
               "metadata": metadata, "rng": rng}
    atomic_save(payload, path)
    # A relative link avoids duplicating a multi-GB optimizer checkpoint.
    descriptor, temporary = tempfile.mkstemp(prefix=".latest-", dir=directory)
    os.close(descriptor)
    os.unlink(temporary)
    try:
        os.symlink(path.name, temporary)
        os.replace(temporary, directory / "latest.pt")
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    return path


def read_checkpoint(path):
    result = torch.load(path, map_location="cpu", weights_only=True)
    if result.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format; legacy prototype checkpoints cannot be resumed")
    required = {"step", "trainable_state_dict", "metadata", "optimizer", "scheduler", "sampler", "rng"}
    if not required.issubset(result):
        raise ValueError("Incomplete training checkpoint")
    return result


def restore_checkpoint(checkpoint, model, optimizer, scheduler, sampler, metadata):
    if checkpoint["metadata"] != metadata:
        raise ValueError("Resume requires the same base weights, manifests and training settings (including total steps)")
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    state = checkpoint["trainable_state_dict"]
    if set(parameters) != set(state):
        raise ValueError("Checkpoint trainable parameter set does not match")
    for name, value in state.items():
        if value.shape != parameters[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint parameter: {name}")
    model.load_state_dict(state, strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    sampler.load_state_dict(checkpoint["sampler"])
    return int(checkpoint["step"])


def clean_metadata(value):
    if isinstance(value, dict):
        return {k: clean_metadata(v) for k, v in value.items() if k not in {"_name_or_path", "name_or_path"}}
    if isinstance(value, list):
        return [clean_metadata(v) for v in value]
    return value


def export_checkpoint(source, checkpoint_path, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Export output must be a new directory")
    checkpoint = read_checkpoint(checkpoint_path)
    if fingerprint(source) != checkpoint["metadata"]["base_fingerprint"]:
        raise ValueError("Base checkpoint does not match the training checkpoint")
    mapping = weight_map(source)
    delta = checkpoint["trainable_state_dict"]
    canonical = {canonical_key(k): k for k in mapping}
    if not delta or set(delta) - canonical.keys():
        raise ValueError("Unknown or empty trainable weight set")
    # Check every replacement before creating output files.
    for shard in sorted({mapping[canonical[k]] for k in delta}):
        with safe_open(source / shard, framework="pt") as reader:
            for key, value in delta.items():
                original = canonical[key]
                if mapping[original] == shard:
                    if tuple(reader.get_slice(original).get_shape()) != tuple(value.shape) or not torch.isfinite(value).all():
                        raise ValueError(f"Invalid replacement weight: {key}")
    output.mkdir(parents=True)
    config = clean_metadata(json.loads((source / "config.json").read_text()))
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    exported, total_size = {}, 0
    shards = sorted(set(mapping.values()))
    for i, shard in enumerate(shards, 1):
        with safe_open(source / shard, framework="pt", device="cpu") as reader:
            tensors = {}
            for key, filename in mapping.items():
                if filename != shard:
                    continue
                value = reader.get_tensor(key)
                name = canonical_key(key)
                tensors[name] = delta[name].to(value.dtype).contiguous() if name in delta else value
            filename = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            save_file(tensors, output / filename, metadata={"format": "pt"})
            exported.update({k: filename for k in tensors})
            total_size += sum(t.numel() * t.element_size() for t in tensors.values())
            del tensors
        print(f"Exported {filename}", flush=True)
    (output / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_size}, "weight_map": exported,
    }, indent=2) + "\n")
    import shutil
    for name in AUXILIARY_FILES:
        if not (source / name).is_file():
            continue
        if name.endswith(".json"):
            content = clean_metadata(json.loads((source / name).read_text()))
            (output / name).write_text(json.dumps(content, ensure_ascii=False) + "\n")
        else:
            shutil.copyfile(source / name, output / name)
    (output / "finetuning.json").write_text(json.dumps({
        "base_fingerprint": checkpoint["metadata"]["base_fingerprint"],
        "step": checkpoint["step"], "trainable_tensors": len(delta),
    }, indent=2) + "\n")
    return output
