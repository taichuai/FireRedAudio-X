"""Single-device SFT with FP32 parameters, BF16 compute and resumable state."""

import argparse
from contextlib import nullcontext
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

from .checkpoints import (
    fingerprint, manifest_fingerprint, read_checkpoint, restore_checkpoint, save_checkpoint,
)
from .data import load_manifest, prepare_sample
from .losses import normalized_objective, supervision_counts


class ShuffledSampler:
    def __init__(self, size, seed):
        if size < 1:
            raise ValueError("Cannot sample an empty dataset")
        self.rng = random.Random(seed)
        self.order = list(range(size))
        self.rng.shuffle(self.order)
        self.cursor = 0

    def next(self):
        if self.cursor == len(self.order):
            self.rng.shuffle(self.order)
            self.cursor = 0
        index = self.order[self.cursor]
        self.cursor += 1
        return index

    def state_dict(self):
        return {"order": list(self.order), "cursor": self.cursor, "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        if sorted(state["order"]) != list(range(len(self.order))) or not 0 <= state["cursor"] <= len(self.order):
            raise ValueError("Invalid saved sampler state")
        self.order = list(state["order"])
        self.cursor = state["cursor"]
        self.rng.setstate(state["rng"])


def capture_rng(device):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if device.type == "cuda":
        if state["cuda"] is None:
            raise ValueError("CUDA random state is missing")
        torch.cuda.set_rng_state(state["cuda"], device)


def module_map(model):
    return {"backbone": model.backbone_llm, "audio_encoder": model.audio_encoder,
            "audio_adapter": model.audio_encoder.adapter,
            "patch_encoder": model.patch_encoder, "dit": model.dit}


def set_training_modes(model, names):
    model.eval()
    modules = module_map(model)
    for name in names:
        modules[name].train()
    model.red_vae.eval()


def configure_modules(model, names, gradient_checkpointing=True):
    modules = module_map(model)
    if not names or names - modules.keys():
        raise ValueError("Invalid trainable module selection")
    model.requires_grad_(False)
    for name in names:
        for parameter in modules[name].parameters():
            parameter.requires_grad_(True)
            parameter.data = parameter.data.float()
    if gradient_checkpointing:
        # Backbone and DiT are checkpointed at their memory-heavy calls in forward.
        for name in names & {"audio_encoder", "patch_encoder"}:
            modules[name].gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    set_training_modes(model, names)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_optimizer(model, learning_rate, backbone_learning_rate, weight_decay, device):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dtype != torch.float32:
            raise ValueError("Trainable parameters must be FP32")
        backbone = name.startswith("backbone_llm.")
        decay = weight_decay if parameter.ndim >= 2 else 0.0
        groups.setdefault((backbone, decay), []).append(parameter)
    return torch.optim.AdamW([
        {"params": parameters, "lr": backbone_learning_rate if backbone else learning_rate,
         "weight_decay": decay}
        for (backbone, decay), parameters in groups.items()
    ], fused=device.type == "cuda")


def make_scheduler(optimizer, total_steps, warmup_steps, schedule):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def move_batch(batch, device):
    return {key: value.to(device=device, dtype=torch.float32 if device.type == "cpu" and value.is_floating_point() else value.dtype)
            for key, value in batch.items()}


def autocast_context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def backward_window(model, batches, device, loss_chunk_size, flow_chunk_size, checkpoint_backbone):
    counts = [supervision_counts(batch) for batch in batches]
    text_count = sum(c[0] for c in counts)
    flow_count = sum(c[1] for c in counts)
    sums = {"text": 0.0, "flow": 0.0}
    for batch in batches:
        with autocast_context(device):
            outputs = model(**move_batch(batch, device), loss_chunk_size=loss_chunk_size,
                            flow_chunk_size=flow_chunk_size, checkpoint_backbone=checkpoint_backbone)
            loss = normalized_objective(outputs, text_count, flow_count)
        if not torch.isfinite(loss) or not loss.requires_grad:
            raise FloatingPointError("Non-finite loss or no gradient path to the selected modules")
        loss.backward()
        for name in sums:
            value = outputs[f"{name}_loss_sum"]
            if value is not None:
                sums[name] += float(value.detach())
    text = sums["text"] / text_count if text_count else 0.0
    flow = sums["flow"] / flow_count if flow_count else 0.0
    return {"loss": text + flow, "text_loss": text, "flow_loss": flow,
            "text_weight": text_count, "flow_count": flow_count}


def evaluate(model, rows, prepare, device, names, args):
    # Validation uses fixed noise/dropout draws without advancing training RNG.
    rng = capture_rng(device)
    model.eval()
    totals = {"text": 0.0, "flow": 0.0}
    text_count, flow_count = 0.0, 0
    try:
        torch.manual_seed(args.seed)
        with torch.no_grad():
            for row in rows:
                batch = prepare(row)
                counts = supervision_counts(batch)
                with autocast_context(device):
                    outputs = model(**move_batch(batch, device), loss_chunk_size=args.loss_chunk_size,
                                    flow_chunk_size=args.flow_chunk_size)
                for name in totals:
                    value = outputs[f"{name}_loss_sum"]
                    if value is not None:
                        if not torch.isfinite(value):
                            raise FloatingPointError("Non-finite validation loss")
                        totals[name] += float(value)
                text_count += counts[0]
                flow_count += counts[1]
    finally:
        restore_rng(rng, device)
        set_training_modes(model, names)
    text = totals["text"] / text_count if text_count else 0.0
    flow = totals["flow"] / flow_count if flow_count else 0.0
    return {"loss": text + flow, "text_loss": text, "flow_loss": flow}


def parse_args(kind):
    parser = argparse.ArgumentParser(description=f"FireRedAudio {kind} single-device SFT")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path)
    parser.add_argument("--model", default="pretrained_models/FireRedAudio")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path(f"demo_outputs/{kind}_training"))
    parser.add_argument("--steps", type=int, default=1000, help="total optimizer steps, including resumed steps")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--scheduler", choices=("linear", "constant"), default="linear")
    parser.add_argument("--max-audio-seconds", type=float, default=30.0, help="total audio duration per sample; overlong samples are rejected")
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--loss-chunk-size", type=int, default=64)
    parser.add_argument("--flow-chunk-size", type=int, default=32)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-modules", default="patch_encoder,dit" if kind == "tts" else "audio_adapter")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-save", action="store_true", help="do not write checkpoints or metrics files")
    args = parser.parse_args()
    for name in ("steps", "gradient_accumulation", "max_sequence_length", "loss_chunk_size", "flow_chunk_size", "save_every", "eval_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    for name in ("learning_rate", "backbone_learning_rate", "max_grad_norm", "max_audio_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight_decay must be finite and nonnegative")
    if not 0 <= args.warmup_steps < args.steps:
        parser.error("warmup_steps must be between zero and steps - 1")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    allowed = {"backbone", "patch_encoder", "dit"} if kind == "tts" else {"backbone", "audio_encoder", "audio_adapter"}
    names = {name.strip() for name in args.train_modules.split(",") if name.strip()}
    if not names or names - allowed:
        parser.error(f"train-modules must be a subset of {sorted(allowed)}")
    if "audio_encoder" in names:
        names.discard("audio_adapter")
    args.train_modules = sorted(names)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This trainer is single-device; distributed launch is not supported")
    return args


def main(kind):
    from transformers import AutoTokenizer
    from fireredaudio.audio_encoder.processor import FireRedAudioProcessor
    from fireredaudio.data.prompt_encoder import AudioPromptEncoder
    from fireredaudio.loading import load_fireredaudio

    args = parse_args(kind)
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA training requires BF16 support")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    names = set(args.train_modules)
    base_config = json.loads((Path(args.model) / "config.json").read_text())
    if base_config.get("model_type") != "firered_audio":
        raise ValueError("--model must point to the complete FireRedAudio checkpoint")
    rows = load_manifest(args.manifest, kind)
    val_rows = load_manifest(args.eval_manifest, kind) if args.eval_manifest else []
    if not args.resume and not args.no_save and any((args.output_dir / name).exists() for name in ("latest.pt", "metrics.jsonl")):
        raise FileExistsError("Output already contains a training run; use --resume or a new directory")
    excluded = {"manifest", "eval_manifest", "model", "device", "output_dir", "resume", "no_save", "save_every", "eval_every"}
    metadata = {"task": kind, "settings": {k: v for k, v in vars(args).items() if k not in excluded},
                "manifest_fingerprint": manifest_fingerprint(args.manifest),
                "eval_manifest_fingerprint": manifest_fingerprint(args.eval_manifest) if args.eval_manifest else None,
                "compute_dtype": "bfloat16" if device.type == "cuda" else "float32",
                "runtime": {"torch": str(torch.__version__), "transformers": version("transformers")}}
    print("Checking base checkpoint identity...", flush=True)
    metadata["base_fingerprint"] = fingerprint(args.model) if not args.no_save or args.resume else "unsaved"
    restored = read_checkpoint(args.resume) if args.resume else None
    if restored is not None and restored["metadata"] != metadata:
        raise ValueError("Resume settings, base checkpoint or manifests do not match")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    processor = FireRedAudioProcessor.from_pretrained(args.model)
    model = load_fireredaudio(args.model, device=device, attention=args.attention, use_liger=False,
                             dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    encoder = AudioPromptEncoder(tokenizer, processor, model.config.audio_special_token,
                                 model.config.audio_special_token_no_latent)
    trainable = configure_modules(model, names, args.gradient_checkpointing)
    print(f"Trainable parameters: {trainable:,}; FP32 parameters / optimizer state", flush=True)
    optimizer = make_optimizer(model, args.learning_rate, args.backbone_learning_rate, args.weight_decay, device)
    scheduler = make_scheduler(optimizer, args.steps, args.warmup_steps, args.scheduler)
    sampler = ShuffledSampler(len(rows), args.seed)
    start_step = 0
    if restored is not None:
        start_step = restore_checkpoint(restored, model, optimizer, scheduler, sampler, metadata)
        restore_rng(restored["rng"], device)
        del restored
    if start_step >= args.steps:
        raise ValueError("Checkpoint already reached --steps")

    def prepare(row, manifest=args.manifest):
        return prepare_sample(row, manifest.resolve().parent, encoder, model, kind,
                              args.max_audio_seconds, args.max_sequence_length)

    def report(record):
        line = json.dumps(record, ensure_ascii=False)
        print(line, flush=True)
        if not args.no_save:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    def validate(step):
        if val_rows:
            metrics = evaluate(model, val_rows, lambda row: prepare(row, args.eval_manifest), device, names, args)
            report({"step": step, "split": "validation", **metrics})

    validate(start_step)
    parameters = [p for p in model.parameters() if p.requires_grad]
    for step in range(start_step + 1, args.steps + 1):
        batches = [prepare(rows[sampler.next()]) for _ in range(args.gradient_accumulation)]
        optimizer.zero_grad(set_to_none=True)
        metrics = backward_window(model, batches, device, args.loss_chunk_size, args.flow_chunk_size,
                                  args.gradient_checkpointing)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
        rates = scheduler.get_last_lr()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        report({"step": step, "split": "train", **metrics, "grad_norm": float(grad_norm), "learning_rates": rates})
        if step % args.eval_every == 0 or step == args.steps:
            validate(step)
        if not args.no_save and (step % args.save_every == 0 or step == args.steps):
            path = save_checkpoint(model, optimizer, scheduler, sampler, step, args.output_dir, metadata, capture_rng(device))
            print(f"Saved {path.name}", flush=True)
