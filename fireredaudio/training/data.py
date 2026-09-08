"""Training manifests and assistant-only supervision."""

import json
from pathlib import Path

import torch

from fireredaudio.data.prompt_encoder import FEAT_TYPE_GENERATION, FEAT_TYPE_UNDERSTAND
from fireredaudio.redae.encoder import PATCH_ENCODER_DOWNSAMPLE_RATE, pad_to_multiple_of
from fireredaudio.utils.audio import GENERATION_SAMPLE_RATE, UNDERSTAND_SAMPLE_RATE, read_audio


def audio_paths(row):
    value = row["audio"]
    return [value] if isinstance(value, str) else value


def resolve_audio(path, manifest_dir):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = manifest_dir / candidate
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def load_manifest(path, kind):
    path = Path(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError("Each row requires a string text field")
            paths = audio_paths(row)
            if not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not p for p in paths):
                raise ValueError("audio must be a path or a nonempty list of paths")
            if kind == "tts":
                if not isinstance(row["audio"], str) or not row["text"].strip():
                    raise ValueError("TTS requires one target audio and nonempty text")
                if row.get("task", "tts") != "tts":
                    raise ValueError("The TTS trainer accepts only task=tts")
                if bool(row.get("prompt_audio")) != bool(row.get("prompt_text")):
                    raise ValueError("prompt_audio and prompt_text must be supplied together")
                if row.get("language", "zh") not in {"zh", "en"}:
                    raise ValueError("language must be zh or en")
                if row.get("prompt_audio"):
                    if not isinstance(row["prompt_audio"], str) or not isinstance(row["prompt_text"], str):
                        raise ValueError("Reference audio and text must be strings")
                    if not row["prompt_text"].strip():
                        raise ValueError("Reference text cannot contain only whitespace")
                    paths = paths + [row["prompt_audio"]]
            else:
                task = row.get("task", "asr")
                if task not in {"asr", "understand"}:
                    raise ValueError("Expected task=asr or understand")
                if "prompt" in row and not isinstance(row["prompt"], str):
                    raise ValueError("prompt must be a string")
                if task == "understand" and not row.get("prompt", "").strip():
                    raise ValueError("understand requires a prompt")
                if row.get("reasoning") is not None:
                    if task == "asr" or not isinstance(row["reasoning"], str):
                        raise ValueError("String reasoning is supported only for understand")
            for audio_path in paths:
                resolve_audio(audio_path, path.resolve().parent)
        except (KeyError, ValueError, TypeError, FileNotFoundError) as error:
            raise ValueError(f"{path.name}:{line_number}: {error}") from error
        rows.append(row)
    if not rows:
        raise ValueError("Training manifest is empty")
    return rows


def read_checked_audio(path, rate, max_seconds):
    waveform = read_audio(str(path), rate)
    if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Empty, non-finite or non-mono audio: {path.name}")
    if max_seconds is not None and waveform.numel() > max_seconds * rate:
        raise ValueError(f"Audio exceeds {max_seconds}s: {path.name}; segment audio and text together offline")
    return waveform


def prepare_sample(row, manifest_dir, encoder, model, kind, max_audio_seconds=30.0,
                   max_sequence_length=4096):
    from inference import (
        DEFAULT_ASR_PROMPT, GENERIC_SYSTEM_PROMPT, _chatml, build_understand_prompt,
    )

    if kind == "understanding":
        waveforms = [read_checked_audio(resolve_audio(p, manifest_dir), UNDERSTAND_SAMPLE_RATE,
                                       max_audio_seconds) for p in audio_paths(row)]
        if max_audio_seconds is not None and sum(w.numel() for w in waveforms) > max_audio_seconds * UNDERSTAND_SAMPLE_RATE:
            raise ValueError("Combined audio duration exceeds the per-sample limit")
        reasoning = row.get("reasoning")
        prefix = build_understand_prompt(
            row.get("prompt") or DEFAULT_ASR_PROMPT, len(waveforms),
            model.config.audio_special_token, enable_thinking=reasoning is not None,
        )
        response = (f"{reasoning}</think>\n\n" if reasoning is not None else "") + row["text"] + "<|im_end|>"
        audios = [{"feat_type": FEAT_TYPE_UNDERSTAND, "audio_understand": w.numpy(),
                   "audio_generation": None, "role": "user"} for w in waveforms]
        batch = encoder.encode(prefix + response, audios)
        response_ids = encoder.tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"]
        length = response_ids.shape[1]
        if not torch.equal(batch["input_ids"][:, -length:], response_ids):
            raise ValueError("Response tokenization changed at the prompt boundary")
        labels = torch.full_like(batch["input_ids"], -100)
        labels[:, -length:] = response_ids
    else:
        target = read_checked_audio(resolve_audio(row["audio"], manifest_dir), GENERATION_SAMPLE_RATE,
                                    max_audio_seconds)
        reference = None
        if row.get("prompt_audio"):
            reference = read_checked_audio(resolve_audio(row["prompt_audio"], manifest_dir),
                                            GENERATION_SAMPLE_RATE, max_audio_seconds)
        if max_audio_seconds is not None and target.numel() + (reference.numel() if reference is not None else 0) > max_audio_seconds * GENERATION_SAMPLE_RATE:
            raise ValueError("Reference plus target audio exceeds the per-sample limit")
        target = pad_to_multiple_of(target)
        start_patch = 0
        if reference is not None:
            reference = pad_to_multiple_of(reference)
            start_patch = reference.numel() // PATCH_ENCODER_DOWNSAMPLE_RATE
            target = torch.cat([reference, target])
        prompt_text = row.get("prompt_text", "")
        separator = " " if row.get("language", "zh") == "en" and prompt_text else ""
        prompt = _chatml(GENERIC_SYSTEM_PROMPT, f"Convert text to speech.\n{prompt_text}{separator}{row['text']}")
        chatml = prompt + f"<|sosp|>{model.config.audio_special_token_no_latent}<|eosp|><|im_end|>"
        batch = encoder.encode(chatml, [{"feat_type": FEAT_TYPE_GENERATION,
                                       "audio_understand": None, "audio_generation": target,
                                       "role": "assistant"}])
        ids = batch["input_ids"]
        positions = (ids[0] == model.config.audio_special_no_latent_id).nonzero(as_tuple=True)[0]
        if start_patch >= positions.numel():
            raise ValueError("Reference leaves no target audio patches")
        labels = torch.full_like(ids, -100)
        start = int(positions[start_patch]) if reference is not None else int((ids[0] == model.config.sosp_idx).nonzero(as_tuple=True)[0].item())
        labels[0, start:] = ids[0, start:]
        weights = torch.ones_like(ids, dtype=torch.float32)
        weights[labels == model.config.audio_special_no_latent_id] = 0.01
        batch["label_weights"] = weights
        batch["generation_target_start_patches"] = torch.tensor([start_patch], dtype=torch.long)
    if batch["input_ids"].shape[1] > max_sequence_length:
        raise ValueError("Serialized example exceeds --max-sequence-length; no truncation was applied")
    batch["labels"] = labels
    batch.pop("vae_is_assistant", None)
    # The original prompt encoder always returns both pathways. Empty generation
    # tensors must not be passed as supervised targets for understanding examples.
    return batch
