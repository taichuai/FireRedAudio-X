#!/usr/bin/env python3
"""One full-audio FireRedAudio transcription followed by shared-window alignment."""

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf

from scripts.fireredaudio_timeline import parse, write_json

DEFAULT_PROMPT = """请对整段音频做逐句的说话人归属转写，并把听到的副语言事件插入实际发生的位置。
每行格式只能是：[MM:SS.mmm-MM:SS.mmm] spk_N: 带副语言标签的逐字台词
时间戳必须相对整段音频起点，按实际发生时间顺序排列。同一实际发音人全程使用同一 spk_N 编号，不同发音人使用不同编号。
完整保留每句语音和语气词，不要总结或改写。每行尽量不超过15秒，按自然语句停顿分行。
副语言事件使用简洁的尖括号标签，例如<呼吸声>、<叹气声>、<咳嗽声>、<哭声>、<大笑>、<清嗓声>，插入实际发生的位置。
无法归属说话人的独立声音事件使用：[MM:SS.mmm-MM:SS.mmm] <事件标签>
不要输出性别、声线描述、角色姓名、总结、解释、代码块或 Markdown。覆盖完整音频，只输出结果行。"""


def check_budget(prompt_tokens, max_new_tokens, max_model_len):
    if prompt_tokens + max_new_tokens > max_model_len:
        raise ValueError(f"Prompt {prompt_tokens} + requested output {max_new_tokens} exceeds context {max_model_len}")


def is_truncated(result):
    reason = result.get("finish_reason")
    return reason == "length" or (isinstance(reason, dict) and reason.get("type") == "length")


def recognize_native(args, prompt, duration):
    import torch
    from transformers import GenerationConfig
    from inference import FireRedAudioInference, build_understand_prompt, set_seed
    from fireredaudio.data.prompt_encoder import FEAT_TYPE_UNDERSTAND
    from fireredaudio.utils.audio import read_audio, UNDERSTAND_SAMPLE_RATE

    # The original SDPA audio mask is quadratic in the complete recording length.
    if duration > 60 and importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError("Native recordings over 60s require the optional FlashAttention installation")
    set_seed(args.seed)
    engine = FireRedAudioInference(model_path=args.model, device=args.device)
    try:
        waveform = read_audio(str(args.audio), UNDERSTAND_SAMPLE_RATE)
        if not waveform.numel() or not torch.isfinite(waveform).all():
            raise ValueError("Audio must be nonempty and finite")
        batch = engine.encoder.encode(build_understand_prompt(prompt, 1, engine.model.config.audio_special_token), [{
            "feat_type": FEAT_TYPE_UNDERSTAND, "audio_understand": waveform.numpy(),
            "audio_generation": None, "role": "user",
        }])
        length = batch["input_ids"].shape[1]
        context = min(args.max_model_len, engine.model.config.backbone_config.max_position_embeddings)
        check_budget(length, args.max_new_tokens, context)
        started = time.monotonic()
        with torch.inference_mode():
            output = engine.model.generate(
                input_ids=batch["input_ids"].to(engine.device),
                attention_mask=batch["attention_mask"].to(engine.device),
                audio_features=batch["audio_features"].to(engine.device),
                audio_feature_attention_mask=batch["audio_feature_attention_mask"].to(engine.device),
                generation_config=GenerationConfig(do_sample=False, num_beams=1,
                    repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
                    eos_token_id=engine._eos_id, pad_token_id=engine._pad_id),
            )
        ids = output[0].tolist()
        truncated = len(ids) >= args.max_new_tokens and ids[-1] != engine._eos_id
        return {"text": engine.tokenizer.decode(ids, skip_special_tokens=True),
                "finish_reason": "length" if truncated else "stop",
                "usage": {"prompt_tokens": length, "completion_tokens": len(ids)},
                "generation_seconds": time.monotonic() - started}
    finally:
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def recognize_server(args, prompt, duration):
    import torch
    from fireredaudio.accelerated.frontend import AudioEmbeddingFrontend
    from fireredaudio.accelerated.client import generate

    if duration > 60 and args.attention != "flash_attention_2":
        raise ValueError("For recordings over 60s use --attention flash_attention_2")
    frontend = AudioEmbeddingFrontend(args.model, args.device, args.attention)
    try:
        embeds, _ = frontend.encode([str(args.audio)], prompt)
        check_budget(embeds.shape[0], args.max_new_tokens, args.max_model_len)
        return generate(args.base_url, args.backend, embeds, eos_id=frontend.eos_id,
                        max_tokens=args.max_new_tokens, temperature=0.0, repetition_penalty=1.0,
                        seed=args.seed, timeout=args.timeout)
    finally:
        del frontend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_pipeline(args, recognize=None):
    info = sf.info(args.audio)
    if not info.frames or not 0 < info.duration <= args.max_audio_seconds:
        raise ValueError("Audio is empty or exceeds --max-audio-seconds")
    prompt = args.prompt_file.read_text(encoding="utf-8").strip() if args.prompt_file else DEFAULT_PROMPT
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    # Catch missing alignment dependencies before performing a long recognition.
    alignment_python = args.aligner_python or sys.executable
    if not args.coarse_only:
        subprocess.run([alignment_python, "-c", "import soundfile; from qwen_asr import Qwen3ForcedAligner"], check=True)
    args.output.mkdir(parents=True, exist_ok=False)
    raw_path = args.output / "raw_response.txt"
    if args.firered_text:
        result = {"text": args.firered_text.read_text(encoding="utf-8"), "finish_reason": "provided"}
    else:
        print(f"Recognizing the complete {info.duration:.2f}s recording with {args.backend}...", flush=True)
        recognize = recognize or (recognize_native if args.backend == "native" else recognize_server)
        started = time.monotonic()
        result = recognize(args, prompt, info.duration)
        result["elapsed_seconds"] = time.monotonic() - started
    raw_path.write_text(result["text"], encoding="utf-8")
    write_json(args.output / "recognition.json", {**result, "backend": args.backend if not args.firered_text else "provided",
                                                "audio_seconds": info.duration})
    if is_truncated(result):
        raise RuntimeError("Transcription reached the output limit; raw response saved. Increase --max-new-tokens and context before slicing")
    parse(result["text"], info.duration)
    print("Transcription saved; aligning and exporting clips...", flush=True)
    command = [alignment_python if not args.coarse_only else sys.executable,
               str(ROOT / "scripts" / "fireredaudio_timeline.py"),
               "--audio", str(args.audio.resolve()), "--firered-text", str(raw_path.resolve()),
               "--output", str((args.output / "clips").resolve()),
               "--window-min-seconds", str(args.window_min_seconds),
               "--window-max-seconds", str(args.window_max_seconds),
               "--max-segment-seconds", str(args.max_segment_seconds),
               "--merge-gap-ms", str(args.merge_gap_ms), "--head-padding-ms", str(args.head_padding_ms),
               "--tail-padding-ms", str(args.tail_padding_ms), "--language", args.language]
    if not args.coarse_only:
        command += ["--aligner-model", args.aligner_model, "--device", args.aligner_device]
    subprocess.run(command, check=True)
    print(f"Finished: {args.output / 'clips' / 'review.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--model", default="pretrained_models/FireRedAudio")
    parser.add_argument("--backend", choices=("native", "vllm", "sglang"), default="native")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=("flash_attention_2", "sdpa", "eager"), default="flash_attention_2",
                        help="audio frontend attention for server backends; native auto-selects FlashAttention")
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-audio-seconds", type=float, default=3600)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--firered-text", type=Path, help="reuse a previously generated timestamped transcript")
    parser.add_argument("--aligner-model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--aligner-python", help="Python in the separate Qwen alignment environment")
    parser.add_argument("--aligner-device", default="cuda:0")
    parser.add_argument("--coarse-only", action="store_true", help="skip forced alignment; all coarse segments require review")
    parser.add_argument("--language", default="Chinese", help="ForcedAligner language")
    parser.add_argument("--window-min-seconds", type=float, default=90)
    parser.add_argument("--window-max-seconds", type=float, default=150)
    parser.add_argument("--max-segment-seconds", type=float, default=15)
    parser.add_argument("--merge-gap-ms", type=float, default=800)
    parser.add_argument("--head-padding-ms", type=float, default=150)
    parser.add_argument("--tail-padding-ms", type=float, default=500)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    if args.max_new_tokens < 1 or args.max_model_len <= args.max_new_tokens:
        parser.error("Require 0 < max-new-tokens < max-model-len")
    if not 0 < args.max_audio_seconds <= 3600 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Require positive timeout and 0 < max-audio-seconds <= 3600")
    if not 0 < args.window_min_seconds <= args.window_max_seconds <= 180 or not 0 < args.max_segment_seconds <= args.window_max_seconds:
        parser.error("Invalid segment/window duration limits")
    for name in ("merge_gap_ms", "head_padding_ms", "tail_padding_ms"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    run_pipeline(args)


if __name__ == "__main__":
    main()
