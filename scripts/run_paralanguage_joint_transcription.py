#!/usr/bin/env python3
"""Run tagged verbatim transcription on an existing paralanguage sample manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import FireRedAudioInference, set_seed
from scripts.eval_paralanguage import (
    PROMPT_LABELS,
    evaluate,
    export_review_assets,
    extract_output_labels,
    normalize_prediction_text,
    parse_prediction,
)


PROMPT = """请逐字转写这段中文音频，并把听到的副语言事件插入到它在语句中实际发生的位置。
副语言事件只能使用以下完整标签：{labels}
要求：
1. 保留音频中的全部语音内容，不要只输出标签。
2. 标签必须使用尖括号，并插入事件实际发生的位置。
3. 没有副语言事件时，只输出普通逐字转写。
4. 不要创造候选列表之外的标签，不要解释，不要输出 Markdown。
5. 只输出一行带标签的完整转写。""".format(labels="、".join(PROMPT_LABELS))


def normalize_one_line(text: str) -> str:
    """Collapse model-generated layout whitespace without changing its words/tags."""
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("demo_outputs/paralanguage/manifest.json"))
    parser.add_argument("--classification-results", type=Path, default=Path("demo_outputs/paralanguage/predictions.jsonl"))
    parser.add_argument("--model", default="pretrained_models/FireRedAudio")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("demo_outputs/paralanguage"))
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="optional UTF-8 prompt file; defaults to the closed-label prompt",
    )
    args = parser.parse_args()

    prompt = (
        args.prompt_file.read_text(encoding="utf-8").strip()
        if args.prompt_file is not None
        else PROMPT
    )
    if not prompt:
        raise ValueError("prompt must not be empty")

    selected = json.loads(args.manifest.read_text(encoding="utf-8"))
    classification_rows = [
        json.loads(line)
        for line in args.classification_results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    classification_by_id = {row["id"]: row for row in classification_rows}
    if {row["id"] for row in selected} != set(classification_by_id):
        raise ValueError("manifest and classification result IDs do not match")

    engine = FireRedAudioInference(model_path=args.model, device=args.device)
    results = []
    for index, record in enumerate(selected, 1):
        sample_id = record["id"]
        audio_path = args.data_root / record["path"]
        set_seed(args.seed + index)
        print(f"[{index:02d}/{len(selected)}] id={sample_id}", flush=True)
        try:
            raw_response = engine.understand(
                str(audio_path),
                prompt,
                task="understand",
                max_new_tokens=args.max_new_tokens,
            ).answer.strip()
            response = normalize_one_line(raw_response)
            output_labels = extract_output_labels(raw_response)
            predicted_labels = parse_prediction(raw_response)
            error = None
        except Exception as exc:
            raw_response = ""
            response = ""
            output_labels = []
            predicted_labels = []
            error = f"{type(exc).__name__}: {exc}"
        gold_labels = sorted({item["text"] for item in record.get("timestamps", [])})
        result = {
            "id": sample_id,
            "source_audio": record["path"],
            "copied_audio": (Path("audio") / f"{sample_id}.wav").as_posix(),
            "gold_transcription": record.get("text", ""),
            "gold_labels": gold_labels,
            "classification_labels": classification_by_id[sample_id]["predicted_labels"],
            "joint_transcription": response,
            "joint_transcription_raw": raw_response,
            "raw_angle_tags": re.findall(r"<[^>\n]+>", raw_response),
            "normalized_transcription": normalize_prediction_text(response),
            "joint_output_labels": output_labels,
            "joint_predicted_labels": predicted_labels,
            "joint_label_exact_match": set(gold_labels) == set(predicted_labels),
            "error": error,
        }
        results.append(result)
        classification_by_id[sample_id].update(
            {
                "joint_transcript": response,
                "joint_transcript_raw": raw_response,
                "joint_predicted_labels": predicted_labels,
                "joint_error": error,
            }
        )
        print(f"    {response}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (args.output_dir / "joint_transcriptions.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "joint_transcriptions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "joint_transcriptions.txt").write_text(
        "\n\n".join(
            f"ID: {row['id']}\n人工: {row['gold_transcription']}\n模型: {row['joint_transcription']}"
            for row in results
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = evaluate(
        [
            {
                "gold_labels": row["gold_labels"],
                "predicted_labels": row["joint_predicted_labels"],
            }
            for row in results
        ]
    )
    (args.output_dir / "joint_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    enriched_rows = [classification_by_id[record["id"]] for record in selected]
    args.classification_results.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in enriched_rows) + "\n",
        encoding="utf-8",
    )
    export_review_assets(args.data_root, selected, enriched_rows, args.output_dir)
    print(
        f"wrote {len(results)} results to {args.output_dir / 'joint_transcriptions.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
