"""ASR/audio understanding through a separate vLLM or SGLang text server."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from fireredaudio.accelerated.client import generate
from fireredaudio.accelerated.frontend import AudioEmbeddingFrontend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default="pretrained_models/FireRedAudio")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio", nargs="+")
    inputs.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--task", choices=("asr", "understand"), default="asr")
    parser.add_argument("--prompt")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.temperature is not None and args.temperature < 0:
        parser.error("--temperature must be nonnegative")
    if args.input_jsonl:
        requests = [json.loads(line) for line in args.input_jsonl.read_text().splitlines() if line.strip()]
    else:
        requests = [{"audio": args.audio}]
    for item in requests:
        item.setdefault("task", args.task)
        item.setdefault("enable_thinking", args.enable_thinking)
        item.setdefault("prompt", args.prompt)
        if item["task"] not in ("asr", "understand"):
            parser.error("Only asr and understand tasks are supported")
        if item["task"] == "asr" and item["enable_thinking"]:
            parser.error("Thinking is supported only for understand")
        if not item["prompt"]:
            if item["task"] == "understand":
                parser.error("understand requires a prompt")
            item["prompt"] = "Transcribe speech to text."
        if isinstance(item.get("audio"), str):
            item["audio"] = [item["audio"]]
        if not isinstance(item.get("audio"), list) or not item["audio"]:
            parser.error("Each request needs audio: a path or a nonempty list of paths")
        for path in item["audio"]:
            if not Path(path).is_file():
                parser.error(f"Audio does not exist: {path}")

    frontend = AudioEmbeddingFrontend(args.model, args.device, args.attention)
    from inference import split_thinking

    def run_request(embeds, item, start, max_tokens):
        asr = item["task"] == "asr"
        result = generate(
            args.base_url, args.backend, embeds, eos_id=frontend.eos_id,
            max_tokens=max_tokens,
            temperature=args.temperature if args.temperature is not None else (0.0 if asr else 0.7),
            repetition_penalty=1.1 if asr else 1.0, seed=args.seed, timeout=args.timeout,
        )
        result["elapsed_seconds"] = time.monotonic() - start
        return result

    def finish(future, item):
        result = future.result()
        reasoning, answer = split_thinking(result["text"])
        result.update(id=item.get("id"), audio=item["audio"], backend=args.backend,
                      answer=answer, reasoning=reasoning)
        return result

    # Bound queued embeddings; only HTTP work runs in threads, the GPU encoder is serial.
    output = args.output.open("w", encoding="utf-8") if args.output else None
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            pending = []
            for item in requests:
                start = time.monotonic()
                embeds, _ = frontend.encode(item["audio"], item["prompt"], item["enable_thinking"])
                max_tokens = args.max_new_tokens or (1024 if item["enable_thinking"] else 300)
                if embeds.shape[0] + max_tokens > args.max_model_len:
                    raise ValueError(f"Prompt ({embeds.shape[0]}) + output ({max_tokens}) exceeds --max-model-len")
                future = pool.submit(run_request, embeds, item, start, max_tokens)
                pending.append((future, item))
                if len(pending) >= args.concurrency:
                    line = json.dumps(finish(*pending.pop(0)), ensure_ascii=False)
                    print(line, flush=True)
                    if output:
                        output.write(line + "\n")
                        output.flush()
            for job in pending:
                line = json.dumps(finish(*job), ensure_ascii=False)
                print(line, flush=True)
                if output:
                    output.write(line + "\n")
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    main()
