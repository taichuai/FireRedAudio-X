#!/usr/bin/env python3
"""Align a chronological FireRedAudio transcript in shared windows and export clips."""

import argparse
import copy
import html
import json
import math
from pathlib import Path
import re

import numpy as np
import soundfile as sf

LINE = re.compile(r"^\s*\[([\d:.]+)\s*-\s*([\d:.]+)\]\s*(?:(spk_\d+)\s*[:：]\s*)?(.*)$")
TAG = re.compile(r"<[^<>\n]+>")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tc(value):
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid timestamp: {value}")
    numbers = [float(part) for part in parts]
    if any(not math.isfinite(n) or n < 0 for n in numbers):
        raise ValueError(f"Invalid timestamp: {value}")
    if any(n >= 60 for n in numbers[1:]) or any(n != int(n) for n in numbers[:-1]):
        raise ValueError(f"Invalid timestamp: {value}")
    out = 0.0
    for number in numbers:
        out = out * 60 + number
    return out


def parse(text, duration=None):
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = LINE.fullmatch(line.strip())
        if not match:
            raise ValueError(f"Cannot parse transcript line {number}; raw output has been retained")
        start, end, speaker, content = match.groups()
        start, end = tc(start), tc(end)
        if end <= start or not content.strip():
            raise ValueError(f"Invalid interval or empty transcript at line {number}")
        if speaker is None and not TAG.fullmatch(content.strip()):
            raise ValueError(f"Speech without spk_N at line {number}")
        risks = []
        if duration is not None:
            if start >= duration or end > duration + 0.5:
                raise ValueError(f"Timestamp outside audio at line {number}")
            if end > duration:
                end = duration
                risks.append("coarse_end_clamped")
        rows.append(dict(start=start, end=end, coarse_start=start, coarse_end=end,
                         speaker_id=speaker or "event", text=content.strip(),
                         source_lines=[number], risks=risks))
    if not rows:
        raise ValueError("No timeline entries; refusing to export an empty transcription")
    return sorted(rows, key=lambda row: (row["start"], row["end"]))


def build_windows(rows, minimum=90.0, maximum=150.0):
    if not 0 < minimum <= maximum:
        raise ValueError("Require 0 < window minimum <= maximum")
    windows, current = [], []
    end = 0.0
    for index, row in enumerate(rows):
        if current:
            start = rows[current[0]]["coarse_start"]
            span = max(end, row["coarse_end"]) - start
            quiet_break = end - start >= minimum and row["coarse_start"] - end >= 0.4
            if span > maximum or quiet_break:
                windows.append(current)
                current = []
        current.append(index)
        end = max(end, row["coarse_end"]) if len(current) > 1 else row["coarse_end"]
    if current:
        windows.append(current)
    return windows


def normalized(text):
    return "".join(c.lower() for c in text if c.isalnum())


def clean_text(text):
    return TAG.sub("", text).strip()


def map_words(words, texts):
    expected = "".join(normalized(text) for text in texts)
    if "".join(normalized(word["text"]) for word in words) != expected:
        raise ValueError("alignment_text_mismatch")
    groups, cursor, begin = [], 0, 0
    spans = []
    for word in words:
        stop = cursor + len(normalized(word["text"]))
        spans.append((cursor, stop, word))
        cursor = stop
    for text in texts:
        end = begin + len(normalized(text))
        selected = []
        for left, right, word in spans:
            if right > begin and left < end:
                if left < begin or right > end:
                    raise ValueError("word_crosses_utterance_boundary")
                selected.append(word)
        groups.append(selected)
        begin = end
    return groups


def apply_word_alignment(row, group, left, right):
    if not group or any(not math.isfinite(w[k]) for w in group for k in ("start", "end")):
        return "invalid_alignment"
    if any(w["start"] < left - .03 or w["end"] > right + .03 or w["end"] < w["start"] for w in group):
        return "invalid_word_interval"
    zero_count = sum(w["end"] == w["start"] for w in group)
    if group[0]["end"] <= group[0]["start"] or group[-1]["end"] <= group[-1]["start"] or zero_count / len(group) > .35:
        return "invalid_word_interval"
    if any(a["end"] > b["start"] + .001 for a, b in zip(group, group[1:])):
        return "nonmonotonic_alignment"
    group = [dict(w, start=max(left, w["start"]), end=min(right, w["end"])) for w in group]
    if group[0]["end"] <= group[0]["start"] or group[-1]["end"] <= group[-1]["start"]:
        return "invalid_word_interval"
    row["words"] = group
    row["start"], row["end"] = group[0]["start"], group[-1]["end"]
    if zero_count:
        row["risks"].append("zero_duration_internal_word")
    if max(abs(row["start"] - row["coarse_start"]), abs(row["end"] - row["coarse_end"])) > 2:
        row["risks"].append("large_alignment_shift")
    if TAG.search(row["text"]):
        row["start"] = min(row["start"], row["coarse_start"])
        row["end"] = max(row["end"], row["coarse_end"])
        row["risks"].append("inline_event_timing_review")
    return None


def align_rows(rows, audio, sr, aligner, minimum=90, maximum=150, language="Chinese"):
    windows = build_windows(rows, minimum, maximum)
    duration = len(audio) / sr
    audits = []
    for window_id, owner in enumerate(windows):
        left = min(rows[i]["coarse_start"] for i in owner)
        right = max(rows[i]["coarse_end"] for i in owner)
        # Context padding also fits the hard maximum. Each aligner call receives
        # ALL speech text and continuous audio in this window, not individual rows.
        allowance = max(0.0, maximum - (right - left)) / 2
        left, right = max(0, left - min(1.0, allowance)), min(duration, right + min(1.0, allowance))
        speech = [i for i in owner if rows[i]["speaker_id"] != "event" and normalized(clean_text(rows[i]["text"]))]
        for i in owner:
            rows[i]["owner_window"] = window_id
            if i not in speech:
                rows[i]["risks"].append("event_coarse_timing")
        audit = dict(id=window_id, kind="owner", start=left, end=right, source_lines=[rows[i]["source_lines"] for i in owner],
                     status="coarse_only", words=[])
        if right - left > maximum + 1e-6:
            for i in owner:
                rows[i]["risks"].append("owner_window_over_max")
            audit["status"] = "owner_window_over_max"
        elif aligner is None:
            for i in speech:
                rows[i]["risks"].append("coarse_time_only")
        elif speech:
            texts = [clean_text(rows[i]["text"]) for i in speech]
            try:
                output = aligner.align(audio=(audio[round(left * sr):round(right * sr)], sr),
                                       text="\n".join(texts), language=language)[0]
                words = [dict(text=w.text, start=left + float(w.start_time), end=left + float(w.end_time))
                         for w in output]
                audit["words"] = words
                if not words or any(not math.isfinite(w[k]) for w in words for k in ("start", "end")):
                    raise ValueError("invalid_alignment")
                groups = map_words(words, texts)
                statuses = []
                for i, group in zip(speech, groups):
                    row = rows[i]
                    reason = apply_word_alignment(row, group, left, right)
                    row["alignment_status"] = reason or "aligned"
                    if reason:
                        row["risks"].append("alignment_failed")
                    statuses.append({"source_lines": row["source_lines"], "status": reason or "aligned"})
                successful = sum(s["status"] == "aligned" for s in statuses)
                audit.update(status="aligned" if successful == len(speech) else "partial" if successful else "alignment_failed",
                             words=words, texts=texts, utterances=statuses)
            except Exception as error:
                reason = str(error) if isinstance(error, ValueError) else type(error).__name__
                audit.update(status="alignment_failed", reason=reason)
                for i in speech:
                    rows[i]["risks"].append("alignment_failed")
        audits.append(audit)
    # Retry only failed owners using neighbouring text; successful large-window
    # alignments are kept. Coarse-only runs never invoke this fallback.
    if aligner is not None:
        for index, row in enumerate(rows):
            if "alignment_failed" not in row["risks"]:
                continue
            context = [i for i in range(max(0, index - 1), min(len(rows), index + 2))
                       if rows[i]["speaker_id"] != "event" and normalized(clean_text(rows[i]["text"]))]
            left = max(0, min(rows[i]["coarse_start"] for i in context) - 1.2)
            right = min(duration, max(rows[i]["coarse_end"] for i in context) + 1.2)
            retry = dict(kind="retry", source_lines=row["source_lines"], start=left, end=right, status="failed", words=[])
            if right - left > maximum:
                retry["reason"] = "retry_window_over_max"
                audits.append(retry)
                continue
            try:
                texts = [clean_text(rows[i]["text"]) for i in context]
                output = aligner.align(audio=(audio[round(left * sr):round(right * sr)], sr),
                                       text="\n".join(texts), language=language)[0]
                words = [dict(text=w.text, start=left + float(w.start_time), end=left + float(w.end_time)) for w in output]
                retry["words"] = words
                group = map_words(words, texts)[context.index(index)]
                reason = apply_word_alignment(row, group, left, right)
                if reason:
                    raise ValueError(reason)
                row["risks"].remove("alignment_failed")
                row["alignment_status"] = "retry_aligned"
                retry["status"] = "aligned"
            except Exception as error:
                retry["reason"] = str(error) if isinstance(error, ValueError) else type(error).__name__
            audits.append(retry)
    return audits


def split_long_rows(rows, limit):
    result = []
    for row in rows:
        words = row.get("words", [])
        if row["end"] - row["start"] <= limit:
            result.append(row)
            continue
        # No reliable token boundary exists for an unaligned or nonverbal span.
        if not words or TAG.search(row["text"]) or any(w["end"] <= w["start"] for w in words):
            row["risks"].append("segment_over_max_review")
            result.append(row)
            continue
        groups, current = [], []
        for word in words:
            if current and word["end"] - current[0]["start"] > limit:
                groups.append(current)
                current = []
            current.append(word)
        if current:
            groups.append(current)
        # Map token counts to original characters, retaining punctuation exactly.
        positions = [i for i, c in enumerate(row["text"]) if c.isalnum()]
        consumed, offset = 0, 0
        for index, group in enumerate(groups):
            consumed += sum(len(normalized(w["text"])) for w in group)
            stop = len(row["text"]) if index == len(groups) - 1 else positions[consumed]
            item = copy.deepcopy(row)
            item.update(start=group[0]["start"], end=group[-1]["end"], words=group,
                        text=row["text"][offset:stop], part=index + 1)
            if item["end"] - item["start"] > limit:
                item["risks"].append("segment_over_max_review")
            result.append(item)
            offset = stop
    return result


def preserve_order(rows):
    # Independent windows and retained coarse event times can disagree. Restore
    # conflicting owners to the original chronological timeline instead of
    # silently reordering their transcript text after alignment.
    for _ in range(len(rows)):
        changed = False
        for previous, current in zip(rows, rows[1:]):
            if previous["start"] <= current["start"]:
                continue
            for row in (previous, current):
                row["start"], row["end"] = row["coarse_start"], row["coarse_end"]
                row.pop("words", None)
                row["risks"].append("order_conflict_fallback")
                row["alignment_status"] = "coarse_order_fallback"
            changed = True
        if not changed:
            return
    raise ValueError("Cannot restore chronological order")


def merge(rows, gap, limit):
    result = []
    for source in rows:
        row = copy.deepcopy(source)
        if result and row["speaker_id"] != "event" and result[-1]["speaker_id"] == row["speaker_id"]:
            previous = result[-1]
            delta = row["start"] - previous["end"]
            if 0 <= delta <= gap and row["end"] - previous["start"] <= limit and previous["owner_window"] == row["owner_window"]:
                previous["end"] = row["end"]
                previous["coarse_end"] = max(previous["coarse_end"], row["coarse_end"])
                previous["text"] += row["text"]
                previous["source_lines"] = sorted(set(previous["source_lines"] + row["source_lines"]))
                previous["risks"] += row["risks"]
                previous.setdefault("words", []).extend(row.get("words", []))
                continue
        result.append(row)
    return result


def frame_energy(audio, sr):
    frame, hop = max(1, round(sr * .02)), max(1, round(sr * .01))
    starts = np.arange(0, max(0, len(audio) - frame + 1), hop)
    sums = np.r_[0.0, np.cumsum(audio.astype(np.float64) ** 2)]
    energy = np.sqrt((sums[starts + frame] - sums[starts]) / frame)
    return (starts + frame / 2) / sr, energy


def pad_rows(rows, audio, sr, head, tail, limit):
    duration = len(audio) / sr
    times, energy = frame_energy(audio, sr)
    for row in rows:
        row["aligned_start"], row["aligned_end"] = row["start"], row["end"]
        available = max(0, limit - (row["end"] - row["start"]))
        before = min(head, available / 2)
        after = min(tail, available - before)
        row["start"] = max(0, row["start"] - before)
        row["end"] = min(duration, row["end"] + after)
    # Only trim added padding. Real/uncertain overlapping speech is preserved and
    # marked for review rather than being cut at an invented midpoint.
    for index, current in enumerate(rows):
        for previous in reversed(rows[:index]):
            if previous["end"] <= current["start"]:
                continue
            left, right = previous["aligned_end"], current["aligned_start"]
            if left > right:
                previous["risks"].append("overlap_review")
                current["risks"].append("overlap_review")
                continue
            indices = np.flatnonzero((times >= left) & (times <= right))
            boundary = float(times[indices[np.argmin(energy[indices])]]) if indices.size else (left + right) / 2
            boundary = min(max(boundary, left), right)
            previous["end"] = min(previous["end"], boundary)
            current["start"] = max(current["start"], boundary)
    return times, energy


def coverage_audit(rows, times, energy, duration):
    if not energy.size:
        return []
    active = energy > max(.005, float(np.quantile(energy, .75)) * .2)
    for row in rows:
        active[(times >= row["start"] - .01) & (times <= row["end"] + .01)] = False
    intervals, start = [], None
    for index, flag in enumerate(np.r_[active, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if (index - start) * .01 >= .16:
                intervals.append(dict(start=max(0, float(times[start]) - .01),
                                      end=min(duration, float(times[index - 1]) + .01),
                                      reason="uncovered_energy_candidate"))
            start = None
    return intervals


def export_timeline(audio_path, text_path, output, *, aligner=None, minimum=90, maximum=150,
                    gap=.8, limit=15, head=.15, tail=.5, language="Chinese"):
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must be nonempty and finite")
    duration = len(audio) / sr
    raw = text_path.read_text(encoding="utf-8")
    rows = parse(raw, duration)
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw_response.txt").write_text(raw, encoding="utf-8")
    audits = align_rows(rows, audio, sr, aligner, minimum, maximum, language)
    preserve_order(rows)
    rows = split_long_rows(rows, limit)
    rows.sort(key=lambda r: (r["start"], r["end"]))
    rows = merge(rows, gap, limit)
    times, energy = pad_rows(rows, audio, sr, head, tail, limit)
    for index, row in enumerate(rows, 1):
        first, last = round(row["start"] * sr), round(row["end"] * sr)
        if not 0 <= first < last <= len(audio):
            raise ValueError(f"Invalid export interval at segment {index}; alignment audit required")
        row.update(id=index, audio=f"{index:05d}.wav", start=first / sr, end=last / sr,
                   duration=(last - first) / sr, risks=sorted(set(row["risks"])))
        sf.write(output / row["audio"], audio[first:last], sr, subtype="PCM_16")
    for name, records in (("segments", rows), ("review_required", [r for r in rows if r["risks"]]),
                          ("candidates", [r for r in rows if not r["risks"]])):
        (output / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    gaps = coverage_audit(rows, times, energy, duration)
    write_json(output / "alignment.json", audits)
    write_json(output / "uncovered_energy.json", gaps)
    (output / "transcript.txt").write_text("\n".join(
        f"{r['id']:05d} [{r['start']:.3f}-{r['end']:.3f}] {r['speaker_id']}: {r['text']}" for r in rows) + "\n", encoding="utf-8")
    body = "".join(f"<tr><td>{r['id']}</td><td>{html.escape(r['speaker_id'])}</td>"
                   f"<td>{r['start']:.3f}-{r['end']:.3f}</td><td><audio controls preload='none' src='{r['audio']}'></audio></td>"
                   f"<td>{html.escape(r['text'])}</td><td>{html.escape(', '.join(r['risks']))}</td></tr>" for r in rows)
    (output / "review.html").write_text("<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        "<title>FireRedAudio timeline review</title><style>td,th{padding:8px;border-bottom:1px solid #ddd}"
        "audio{width:240px}table{border-collapse:collapse}</style><table><tr><th>ID</th><th>Speaker</th>"
        "<th>Time</th><th>Audio</th><th>Text</th><th>Review reasons</th></tr>" + body + "</table>", encoding="utf-8")
    summary = dict(audio_duration=duration, clips=len(rows), windows=sum(w["kind"] == "owner" for w in audits),
                   aligned_windows=sum(w["status"] == "aligned" and w["kind"] == "owner" for w in audits),
                   partially_aligned_windows=sum(w["status"] == "partial" and w["kind"] == "owner" for w in audits),
                   retry_windows=sum(w["kind"] == "retry" for w in audits),
                   clips_with_word_alignment=sum(bool(r.get("words")) for r in rows),
                   review_required=sum(bool(r["risks"]) for r in rows), uncovered_energy_candidates=len(gaps),
                   speaker_ids=sorted({r["speaker_id"] for r in rows if r["speaker_id"] != "event"}),
                   speaker_source="FireRedAudio generated IDs, not verified voice clusters")
    write_json(output / "run.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--firered-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aligner-model", help="optional Qwen ForcedAligner model or local directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--merge-gap-ms", type=float, default=800)
    parser.add_argument("--max-segment-seconds", type=float, default=15)
    parser.add_argument("--window-min-seconds", type=float, default=90)
    parser.add_argument("--window-max-seconds", type=float, default=150)
    parser.add_argument("--head-padding-ms", type=float, default=150)
    parser.add_argument("--tail-padding-ms", type=float, default=500)
    args = parser.parse_args()
    for name in ("merge_gap_ms", "head_padding_ms", "tail_padding_ms"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if not 0 < args.max_segment_seconds <= args.window_max_seconds <= 180 or not 0 < args.window_min_seconds <= args.window_max_seconds:
        parser.error("Require positive segment/window limits, min <= max, max <= 180 seconds")
    if args.output.exists():
        parser.error("Output must be a new directory")
    # Validate source files before spending GPU memory on the aligner.
    info = sf.info(args.audio)
    parse(args.firered_text.read_text(encoding="utf-8"), info.duration)
    aligner = None
    if args.aligner_model:
        import torch
        from qwen_asr import Qwen3ForcedAligner
        aligner = Qwen3ForcedAligner.from_pretrained(args.aligner_model, dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
                                                    device_map=args.device)
    summary = export_timeline(args.audio, args.firered_text, args.output, aligner=aligner,
                              minimum=args.window_min_seconds, maximum=args.window_max_seconds,
                              gap=args.merge_gap_ms / 1000, limit=args.max_segment_seconds,
                              head=args.head_padding_ms / 1000, tail=args.tail_padding_ms / 1000,
                              language=args.language)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
