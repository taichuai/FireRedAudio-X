#!/usr/bin/env python3
"""Run FireRedAudio's default ASR prompt on a fixed paralanguage manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import DEFAULT_ASR_PROMPT, FireRedAudioInference


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("demo_outputs/paralanguage/manifest.json"),
    )
    parser.add_argument("--model", default="pretrained_models/FireRedAudio")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("demo_outputs/paralanguage")
    )
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    engine = FireRedAudioInference(model_path=args.model, device=args.device)
    results = []
    for index, record in enumerate(records, 1):
        audio_path = args.data_root / record["path"]
        print(f"[{index:02d}/{len(records)}] id={record['id']}", flush=True)
        try:
            raw = engine.understand(
                str(audio_path),
                DEFAULT_ASR_PROMPT,
                task="asr",
                max_new_tokens=300,
            ).answer.strip()
            error = None
        except Exception as exc:
            raw = ""
            error = f"{type(exc).__name__}: {exc}"
        transcript = one_line(raw)
        angle_tags = re.findall(r"<[^>\n]+>", raw)
        result = {
            "id": record["id"],
            "source_audio": record["path"],
            "gold_transcription": record.get("text", ""),
            "default_asr_prompt": DEFAULT_ASR_PROMPT,
            "asr_transcription": transcript,
            "asr_transcription_raw": raw,
            "angle_bracket_tags": angle_tags,
            "error": error,
        }
        results.append(result)
        print(f"    {transcript}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "asr_baseline.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "asr_baseline.txt").write_text(
        "\n\n".join(
            f"ID: {row['id']}\n人工: {row['gold_transcription']}\nASR: {row['asr_transcription']}"
            for row in results
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "num_samples": len(results),
        "num_errors": sum(row["error"] is not None for row in results),
        "num_outputs_with_angle_bracket_tags": sum(
            bool(row["angle_bracket_tags"]) for row in results
        ),
        "prompt": DEFAULT_ASR_PROMPT,
    }
    (args.output_dir / "asr_baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
