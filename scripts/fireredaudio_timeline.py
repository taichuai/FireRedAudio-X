#!/usr/bin/env python3
"""Export chronological FireRedAudio speaker clips with optional alignment."""
import argparse, copy, html, json, math, re
from pathlib import Path
import numpy as np
import soundfile as sf

LINE = re.compile(r"^\s*\[([\d:.]+)-([\d:.]+)\]\s*(?:(spk_\d+):\s*)?(.*)$")

def tc(value):
    out = 0.0
    for part in value.split(":"):
        out = out * 60 + float(part)
    return out

def parse(text):
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip(): continue
        match = LINE.fullmatch(line.strip())
        if not match: raise ValueError(f"Cannot parse line {number}: {line}")
        start, end, speaker, content = match.groups()
        rows.append({"start": tc(start), "end": tc(end), "speaker_id": speaker or "event",
                     "text": content, "source_lines": [number], "risks": []})
    return sorted(rows, key=lambda row: row["start"])

def merge(rows, gap, limit):
    result = []
    for source in rows:
        row = copy.deepcopy(source)
        row["coarse_start"], row["coarse_end"] = row["start"], row["end"]
        if result and row["speaker_id"] != "event" and result[-1]["speaker_id"] == row["speaker_id"]:
            previous = result[-1]
            if row["start"] - previous["end"] <= gap and max(row["end"], previous["end"]) - previous["start"] <= limit:
                previous["end"] = max(previous["end"], row["end"])
                previous["coarse_end"] = max(previous["coarse_end"], row["coarse_end"])
                previous["text"] += row["text"]
                previous["source_lines"] += row["source_lines"]
                continue
        result.append(row)
    return result

def build_windows(rows, minimum=90, maximum=150):
    result, current = [], []
    for index, row in enumerate(rows):
        if current and row["coarse_end"] - rows[current[0]]["coarse_start"] >= minimum and row["coarse_start"] >= rows[current[0]]["coarse_start"] + maximum:
            result.append(current); current = []
        current.append(index)
    if current: result.append(current)
    for owner in result:
        if rows[owner[-1]]["coarse_end"] - rows[owner[0]]["coarse_start"] > maximum:
            for index in owner:
                rows[index]["risks"].append("owner_window_over_max")
    return result

def energy_boundary(audio, sr, target, left, right):
    frame, hop = max(1, round(sr*.02)), max(1, round(sr*.01))
    starts = np.arange(max(0, round(left*sr)), min(len(audio)-frame, round(right*sr)), hop)
    if not len(starts): return target
    rms = np.sqrt(np.convolve(audio.astype(np.float64)**2, np.ones(frame)/frame, mode="valid")[starts])
    return float((starts[int(np.argmin(rms + abs((starts+frame/2)/sr-target)*.02))] + frame/2)/sr)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--firered-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aligner-model", default=None, help="Qwen model name or local directory")
    parser.add_argument("--merge-gap-ms", type=float, default=800)
    parser.add_argument("--max-segment-seconds", type=float, default=15)
    parser.add_argument("--window-min-seconds", type=float, default=90)
    parser.add_argument("--window-max-seconds", type=float, default=150)
    parser.add_argument("--head-padding-ms", type=float, default=150)
    parser.add_argument("--tail-padding-ms", type=float, default=500)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1: audio = audio.mean(axis=1)
    rows = merge(parse(args.firered_text.read_text(encoding="utf-8")), args.merge_gap_ms/1000, args.max_segment_seconds)
    owner_windows = build_windows(rows, args.window_min_seconds, args.window_max_seconds)
    aligner = None
    if args.aligner_model:
        import torch
        from qwen_asr import Qwen3ForcedAligner
        aligner = Qwen3ForcedAligner.from_pretrained(args.aligner_model, dtype=torch.bfloat16, device_map="cuda:0")
    for owner in owner_windows:
        for index in owner:
            row = rows[index]; text = re.sub(r"<[^<>]+>", "", row["text"]).strip()
            if not aligner or not text or row["speaker_id"] == "event":
                row["risks"].append("coarse_time_only"); continue
            left, right = max(0, row["coarse_start"]-1.2), min(len(audio)/sr, row["coarse_end"]+1.2)
            try:
                aligned = aligner.align(audio=(audio[round(left*sr):round(right*sr)], sr), text=text, language="Chinese")[0]
                valid = [item for item in aligned if item.end_time > item.start_time]
                if valid: row["start"], row["end"] = left+valid[0].start_time, left+valid[-1].end_time
                else: row["risks"].append("zero_duration_alignment")
            except Exception as exc: row["risks"].append(f"alignment_failed:{type(exc).__name__}")
    duration = len(audio)/sr
    for row in rows:
        row["start"] = max(0, row["start"]-args.head_padding_ms/1000)
        row["end"] = min(duration, row["end"]+args.tail_padding_ms/1000)
    for previous, current in zip(rows, rows[1:]):
        if previous["speaker_id"] == current["speaker_id"] or previous["end"] <= current["start"]: continue
        boundary = energy_boundary(audio, sr, (previous["coarse_end"]+current["coarse_start"])/2, previous["coarse_end"], current["coarse_start"])
        previous["end"], current["start"] = min(previous["end"], boundary-.06), max(current["start"], boundary+.06)
        previous["risks"].append("speaker_boundary_guard"); current["risks"].append("speaker_boundary_guard")
    manifest=[]
    for index, row in enumerate(rows, 1):
        if row["end"] <= row["start"]: row["start"], row["end"] = row["coarse_start"], row["coarse_end"]; row["risks"].append("invalid_interval_fallback")
        filename=f"{index:05d}.wav"; sf.write(args.output/filename, audio[round(row["start"]*sr):round(row["end"]*sr)], sr, subtype="PCM_16")
        manifest.append({**row, "id": index, "audio": filename, "duration": row["end"]-row["start"]})
    (args.output/"segments.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in manifest), encoding="utf-8")
    (args.output/"transcript.txt").write_text("\n".join(f"{row['id']:05d} [{row['start']:.3f}-{row['end']:.3f}] {row['speaker_id']}: {row['text']}" for row in manifest), encoding="utf-8")
    body="".join(f"<tr><td>{row['id']}</td><td>{html.escape(row['speaker_id'])}</td><td>{row['start']:.3f}-{row['end']:.3f}</td><td><audio controls src='{row['audio']}'></audio></td><td>{html.escape(row['text'])}</td><td>{html.escape(', '.join(row['risks']))}</td></tr>" for row in manifest)
    (args.output/"review.html").write_text("<!doctype html><meta charset='utf-8'><style>td{padding:8px;border-bottom:1px solid #ddd}audio{width:240px}</style><table><tr><th>ID</th><th>speaker</th><th>time</th><th>audio</th><th>text</th><th>risks</th></tr>"+body+"</table>", encoding="utf-8")
    (args.output/"run.json").write_text(json.dumps({"audio":str(args.audio),"firered_text":str(args.firered_text),"aligner_model":args.aligner_model,"clips":len(manifest),"windows":len(owner_windows),"risks":sum(bool(row['risks']) for row in manifest)}, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
