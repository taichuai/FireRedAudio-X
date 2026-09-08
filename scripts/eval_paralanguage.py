#!/usr/bin/env python3
"""Evaluate FireRedAudio on the Paralanguage multi-label annotations."""

from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import random
import re
import shutil
import sys
from pathlib import Path

# The script is launched from ``scripts/`` but inference.py lives at repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import FireRedAudioInference, set_seed


LABELS = (
    "<呼吸声>",
    "<叹气声>",
    "<疑问声-欸？>",
    "<咳嗽声>",
    "<疑问声-咦？>",
    "<哭声>",
    "<大笑>",
    "<犹豫声-嗯。。。>",
    "<惊讶声-啊！>",
    "<应答声-嗯>",
    "<惊讶声-哦！>",
    "<疑问声-啊？>",
    "<惊讶声-哇！>",
    "<不满声-哼？>",
    "<疑问声-嗯？>",
    "<疑问声-哦？>",
    "<惊讶声-哟！>",
)

# Optional protocol aliases, not an official model or benchmark label mapping.
# Keep the original output alongside normalized labels when interpreting results.
PROMPT_LABELS = LABELS + ("<清嗓声>",)
LABEL_ALIASES = {
    "<轻笑>": "<大笑>",
    "<笑声>": "<大笑>",
    "<清嗓声>": "<咳嗽声>",
    "<清嗓子>": "<咳嗽声>",
}
LAUGHTER_TEXT_PATTERN = re.compile(r"(?:哈){2,}|(?:嘿){2,}|(?:呵){2,}")

PROMPT = """你是一个严格的中文副语言事件检测器。请听音频，识别其中实际出现的所有副语言声音或语气词，不要转写台词。
只允许从下面的标签中选择：{labels}
请只输出 JSON 数组，数组元素必须是上述完整标签；没有检测到副语言事件时输出 []。不要输出解释、置信度或其它文字。""".format(
    labels="、".join(PROMPT_LABELS)
)


def annotation_labels(record: dict) -> set[str]:
    return {item["text"] for item in record.get("timestamps", [])}


def select_records(records: list[dict], total: int, seed: int) -> list[dict]:
    """Cover each label, include negatives, then fill the remaining budget randomly."""
    if total < len(LABELS):
        raise ValueError(f"--num-samples must be at least {len(LABELS)}")

    rng = random.Random(seed)
    labels_by_id = {r["id"]: annotation_labels(r) for r in records}
    positives = [r for r in records if labels_by_id[r["id"]]]
    negatives = [r for r in records if not labels_by_id[r["id"]]]
    chosen: list[dict] = []
    used: set[int] = set()

    # Rare labels get first choice, and short label sets are preferred to reduce
    # overlap while still preserving the dataset's naturally multi-label setting.
    support = collections.Counter(label for ls in labels_by_id.values() for label in ls)
    negative_count = min(10, len(negatives), max(1, total // 6))
    positive_budget = total - negative_count
    for label in sorted(LABELS, key=lambda x: (support[x], x)):
        candidates = [r for r in positives if label in labels_by_id[r["id"]]]
        candidates.sort(key=lambda r: (len(labels_by_id[r["id"]]), r["id"]))
        covered = 0
        for record in candidates:
            if record["id"] in used:
                continue
            chosen.append(record)
            used.add(record["id"])
            covered += 1
            if covered >= 3 or len(chosen) >= positive_budget:
                break

    # Reserve up to ten clean negatives as a useful false-positive control.
    chosen_negatives = rng.sample(negatives, negative_count)
    chosen.extend(chosen_negatives)
    used.update(r["id"] for r in chosen_negatives)

    # Fill the remainder with positives first so the negative control stays fixed.
    remaining = [r for r in positives if r["id"] not in used]
    remaining += [r for r in negatives if r["id"] not in used]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, total - len(chosen))])
    if len(chosen) < total:
        raise ValueError(f"dataset only provided {len(chosen)} selectable records")
    return sorted(chosen[:total], key=lambda r: r["id"])


def extract_output_labels(answer: str) -> list[str]:
    """Extract exact allowed surface labels without changing their taxonomy."""
    found = []
    recognized_labels = PROMPT_LABELS + tuple(
        label for label in LABEL_ALIASES if label not in PROMPT_LABELS
    )
    for label in recognized_labels:
        short = label[1:-1]
        if label in answer or short in answer:
            found.append(label)
    return found


def parse_prediction(answer: str) -> list[str]:
    """Extract labels and normalize prompt aliases to the dataset taxonomy."""
    normalized = []
    for label in extract_output_labels(answer):
        canonical = LABEL_ALIASES.get(label, label)
        if canonical not in normalized:
            normalized.append(canonical)
    if LAUGHTER_TEXT_PATTERN.search(answer) and "<大笑>" not in normalized:
        normalized.append("<大笑>")
    return normalized


def normalize_prediction_text(answer: str) -> str:
    """Map model-native laughter forms into the dataset's canonical inline tag."""
    for alias, canonical in LABEL_ALIASES.items():
        answer = answer.replace(alias, canonical)
    return LAUGHTER_TEXT_PATTERN.sub("<大笑>", answer)


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate(rows: list[dict]) -> dict:
    tp = fp = fn = 0
    exact = presence_correct = 0
    per_label = {}
    for label in LABELS:
        ltp = lfp = lfn = 0
        for row in rows:
            gold, pred = set(row["gold_labels"]), set(row["predicted_labels"])
            ltp += int(label in gold and label in pred)
            lfp += int(label not in gold and label in pred)
            lfn += int(label in gold and label not in pred)
        p = ltp / (ltp + lfp) if ltp + lfp else 0.0
        r = ltp / (ltp + lfn) if ltp + lfn else 0.0
        per_label[label] = {
            "support": sum(label in set(x["gold_labels"]) for x in rows),
            "tp": ltp,
            "fp": lfp,
            "fn": lfn,
            "precision": p,
            "recall": r,
            "f1": f1(p, r),
        }
        tp += ltp
        fp += lfp
        fn += lfn
    for row in rows:
        gold, pred = set(row["gold_labels"]), set(row["predicted_labels"])
        exact += int(gold == pred)
        presence_correct += int(bool(gold) == bool(pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "num_samples": len(rows),
        "exact_set_accuracy": exact / len(rows),
        "event_presence_accuracy": presence_correct / len(rows),
        "micro": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
        },
        "macro_f1": sum(x["f1"] for x in per_label.values()) / len(LABELS),
        "per_label": per_label,
    }


def export_review_assets(
    data_root: Path, selected: list[dict], rows: list[dict], output_dir: Path
) -> None:
    """Copy evaluated audio and write portable review artifacts."""
    records_by_id = {record["id"]: record for record in selected}
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    playlist = []
    review_rows = []
    for row in rows:
        record = records_by_id[row["id"]]
        source = data_root / record["path"]
        destination = audio_dir / f"{row['id']}.wav"
        shutil.copy2(source, destination)
        relative_audio = destination.relative_to(output_dir).as_posix()
        playlist.append(relative_audio)
        review_rows.append(
            {
                "id": row["id"],
                "audio": relative_audio,
                "exact_match": set(row["gold_labels"]) == set(row["predicted_labels"]),
                "gold_labels": row["gold_labels"],
                "predicted_labels": row["predicted_labels"],
                "annotation_text": record.get("text", ""),
                "joint_transcript": row.get("joint_transcript", ""),
                "normalized_transcription": row.get("normalized_transcription", ""),
                "joint_predicted_labels": row.get("joint_predicted_labels", []),
                "answer": row["answer"],
            }
        )

    (output_dir / "playlist.m3u").write_text(
        "\n".join(playlist) + "\n", encoding="utf-8"
    )
    with (output_dir / "review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "id",
                "audio",
                "exact_match",
                "gold_labels",
                "predicted_labels",
                "annotation_text",
                "joint_transcript",
                "normalized_transcription",
                "joint_predicted_labels",
                "answer",
            ),
        )
        writer.writeheader()
        for row in review_rows:
            writer.writerow(
                {
                    **row,
                    "gold_labels": " | ".join(row["gold_labels"]),
                    "predicted_labels": " | ".join(row["predicted_labels"]),
                    "joint_predicted_labels": " | ".join(row["joint_predicted_labels"]),
                }
            )

    table_rows = []
    for row in review_rows:
        status = "正确" if row["exact_match"] else "不一致"
        status_class = "match" if row["exact_match"] else "mismatch"
        gold = " ".join(row["gold_labels"]) or "[]"
        predicted = " ".join(row["predicted_labels"]) or "[]"
        joint_predicted = " ".join(row["joint_predicted_labels"]) or "[]"
        table_rows.append(
            f"""<tr data-status="{status_class}">
<td><strong>{row['id']}</strong><span class="status {status_class}">{status}</span></td>
<td><audio controls preload="none" src="{html.escape(row['audio'])}"></audio></td>
<td class="labels gold">{html.escape(gold)}</td>
<td class="labels predicted">{html.escape(predicted)}</td>
<td>{html.escape(row['annotation_text'])}</td>
<td>{html.escape(row['joint_transcript'])}</td>
<td>{html.escape(row['normalized_transcription'])}</td>
<td class="labels predicted">{html.escape(joint_predicted)}</td>
<td><code>{html.escape(row['answer'])}</code></td>
</tr>"""
        )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FireRedAudio 副语言评测试听</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, sans-serif; color: #202124; background: #f7f8fa; }}
body {{ margin: 0; }}
header {{ position: sticky; top: 0; z-index: 2; padding: 14px 20px; background: #fff; border-bottom: 1px solid #dfe3e8; }}
h1 {{ margin: 0 0 10px; font-size: 20px; letter-spacing: 0; }}
.controls {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; font-size: 14px; }}
main {{ padding: 0 20px 28px; overflow-x: auto; }}
table {{ width: 100%; min-width: 1120px; border-collapse: collapse; background: #fff; }}
th {{ position: sticky; top: 81px; z-index: 1; background: #eef1f4; text-align: left; font-size: 13px; }}
th, td {{ padding: 10px; border-bottom: 1px solid #e6e9ed; vertical-align: top; }}
tr:hover {{ background: #f5f8fb; }}
audio {{ width: 260px; height: 36px; }}
.status {{ display: block; width: fit-content; margin-top: 5px; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
.match {{ color: #116329; background: #dafbe1; }}
.mismatch {{ color: #9a3412; background: #ffedd5; }}
.labels {{ min-width: 130px; font-weight: 600; }}
.gold {{ color: #1f6f43; }}
.predicted {{ color: #8a3c13; }}
code {{ white-space: pre-wrap; font-size: 12px; }}
body.only-mismatch tr[data-status="match"] {{ display: none; }}
</style>
</head>
<body>
<header>
<h1>FireRedAudio 副语言评测试听</h1>
<div class="controls">
<span>共 {len(review_rows)} 条，完全正确 {sum(row['exact_match'] for row in review_rows)} 条</span>
<label><input id="mismatch-only" type="checkbox"> 只看不一致</label>
</div>
</header>
<main>
<table>
<thead><tr><th>ID</th><th>音频</th><th>人工标签</th><th>分类预测</th><th>人工完整标注</th><th>模型联合转写</th><th>归一化转写</th><th>联合转写标签</th><th>分类原始回答</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
</main>
<script>
document.getElementById('mismatch-only').addEventListener('change', event => {{
  document.body.classList.toggle('only-mismatch', event.target.checked);
}});
</script>
</body>
</html>
"""
    (output_dir / "review.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="directory containing user-provided evaluation data")
    parser.add_argument("--annotation", default=None)
    parser.add_argument("--model", default="pretrained_models/FireRedAudio")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output-dir", default="demo_outputs/paralanguage")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    annotation_path = Path(args.annotation or data_root / "para_annotation.json")
    records = json.loads(annotation_path.read_text(encoding="utf-8"))
    selected = select_records(records, args.num_samples, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"selected {len(selected)} samples; loading model on {args.device}", flush=True)

    set_seed(args.seed)
    engine = FireRedAudioInference(model_path=args.model, device=args.device)
    rows = []
    for index, record in enumerate(selected, 1):
        audio_path = data_root / record["path"]
        gold = sorted(annotation_labels(record))
        try:
            result = engine.understand(
                str(audio_path), PROMPT, task="understand", max_new_tokens=args.max_new_tokens
            )
            answer = result.answer.strip()
            predicted = parse_prediction(answer)
            error = None
        except Exception as exc:  # keep one bad file from discarding the whole run
            answer = ""
            predicted = []
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "id": record["id"],
            "path": record["path"],
            "gold_labels": gold,
            "predicted_labels": predicted,
            "answer": answer,
            "error": error,
        }
        rows.append(row)
        print(
            f"[{index:02d}/{len(selected)}] id={record['id']} "
            f"gold={','.join(gold) or '[]'} pred={','.join(predicted) or '[]'}",
            flush=True,
        )

    metrics = evaluate(rows)
    (output_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    export_review_assets(data_root, selected, rows, output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
