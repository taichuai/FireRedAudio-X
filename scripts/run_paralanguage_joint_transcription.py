#!/usr/bin/env python3
"""Transcribe audio with inline paralinguistic tags, without evaluation dependencies."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT = """请逐字转写这段中文音频，并把听到的副语言事件插入到它在语句中实际发生的位置。
副语言事件只能使用以下完整标签：<呼吸声>、<叹气声>、<疑问声-欸？>、<咳嗽声>、<疑问声-咦？>、<哭声>、<大笑>、<犹豫声-嗯。。。>、<惊讶声-啊！>、<应答声-嗯>、<惊讶声-哦！>、<疑问声-啊？>、<惊讶声-哇！>、<不满声-哼？>、<疑问声-嗯？>、<疑问声-哦？>、<惊讶声-哟！>、<清嗓声>
要求：
1. 保留音频中的全部语音内容，不要只输出标签。
2. 标签必须使用尖括号，并插入事件实际发生的位置。
3. 没有副语言事件时，只输出普通逐字转写。
4. 不要创造候选列表之外的标签，不要解释，不要输出 Markdown。
5. 只输出一行带标签的完整转写。"""


def normalize_one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_manifest(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8")
    if content.lstrip().startswith("["):
        records = json.loads(content)
    else:
        records = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain a nonempty JSON array or JSONL records")
    ids = set()
    result = []
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object")
        audio = record.get("path", record.get("audio"))
        sample_id = record.get("id", index)
        if not isinstance(audio, str) or not audio.strip():
            raise ValueError(f"Record {index} requires a nonempty path or audio string")
        if type(sample_id) not in (str, int) or str(sample_id) in ids:
            raise ValueError(f"Record {index} has an invalid or duplicate id")
        ids.add(str(sample_id))
        result.append({"id": sample_id, "path": audio})
    return result


def transcribe(engine, record: dict, base: Path, prompt: str, max_new_tokens: int) -> dict:
    path = Path(record["path"])
    audio_path = path if path.is_absolute() else base / path
    result = {"id": record["id"], "source_audio": record["path"],
              "joint_transcription": "", "joint_transcription_raw": "",
              "raw_angle_tags": [], "error": None}
    try:
        if not audio_path.is_file():
            raise FileNotFoundError("Audio file does not exist")
        raw = engine.understand(str(audio_path), prompt, task="understand",
                                max_new_tokens=max_new_tokens).answer
        if not raw.strip():
            raise ValueError("Model returned an empty transcription")
        result.update(joint_transcription=normalize_one_line(raw),
                      joint_transcription_raw=raw,
                      raw_angle_tags=re.findall(r"<[^>\n]+>", raw))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio", type=Path, help="single audio, relative to the current directory")
    inputs.add_argument("--manifest", type=Path, help="JSON array or JSONL containing id and path/audio")
    parser.add_argument("--data-root", type=Path, help="manifest audio root; defaults to the manifest directory")
    parser.add_argument("--model", default="pretrained_models/FireRedAudio")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("demo_outputs/paralanguage"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--prompt-file", type=Path, help="optional UTF-8 prompt overriding the default tags")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.audio and args.data_root:
        parser.error("--data-root applies only to --manifest")
    prompt = args.prompt_file.read_text(encoding="utf-8").strip() if args.prompt_file else PROMPT
    if not prompt:
        parser.error("prompt must not be empty")
    if args.manifest:
        records = load_manifest(args.manifest)
        base = args.data_root or args.manifest.resolve().parent
    else:
        records = [{"id": args.audio.stem, "path": str(args.audio)}]
        base = Path.cwd()
    output_path = args.output_dir / "joint_transcriptions.jsonl"
    if output_path.exists():
        parser.error(f"Output already exists: {output_path}; choose a new --output-dir")

    from inference import FireRedAudioInference, set_seed

    engine = FireRedAudioInference(model_path=args.model, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    with output_path.open("x", encoding="utf-8") as stream:
        for index, record in enumerate(records, 1):
            set_seed(args.seed + index)
            result = transcribe(engine, record, base, prompt, args.max_new_tokens)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            failures += result["error"] is not None
            print(f"[{index}/{len(records)}] {record['id']}: "
                  f"{result['error'] or result['joint_transcription']}", flush=True)
    print(f"Wrote {len(records)} records ({failures} failed) to {output_path}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
