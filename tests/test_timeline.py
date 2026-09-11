"""Timeline ownership, full-window alignment and long-audio orchestration tests."""

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

if importlib.util.find_spec("soundfile") is None:
    raise unittest.SkipTest("Install the timeline extra to test audio slicing")

import numpy as np
import soundfile as sf

from scripts.fireredaudio_timeline import (
    align_rows, apply_word_alignment, build_windows, export_timeline, map_words, merge, pad_rows, parse, preserve_order, split_long_rows,
)
from scripts.run_fireredaudio_long_audio import check_budget, is_truncated, run_pipeline


def aligned_word(text, start, end):
    return SimpleNamespace(text=text, start_time=start, end_time=end)


class TimelineTests(unittest.TestCase):
    def test_invalid_times_and_missing_speakers_are_rejected(self):
        for text in ("[9-3] spk_1: backwards", "[0-0] spk_1: empty",
                     "[00:70-01:30] spk_1: invalid", "[1-2] unknown speaker", ""):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse(text, duration=100)
        with self.assertRaisesRegex(ValueError, "outside audio"):
            parse("[10-20] spk_1: outside", duration=5)

    def test_timestamps_and_standalone_events(self):
        rows = parse("[01:02.500-01:03.000] spk_2: hello\n[00:01-00:02] <笑声>", 64)
        self.assertEqual(rows[0]["speaker_id"], "event")
        self.assertEqual(rows[1]["start"], 62.5)

    def test_windows_are_bounded_and_cover_every_row_once(self):
        rows = parse("\n".join(f"[{i}-{i + 10}] spk_1: text" for i in range(0, 330, 10)))
        windows = build_windows(rows, 90, 150)
        self.assertEqual([i for window in windows for i in window], list(range(len(rows))))
        self.assertEqual(len(windows), 3)
        for window in windows:
            self.assertLessEqual(max(rows[i]["coarse_end"] for i in window) - rows[window[0]]["coarse_start"], 150)

    def test_aligner_receives_one_joint_window_and_preserves_speakers(self):
        rows = parse("[1-2] spk_1: 你好。\n[10-11] spk_2: 再见。")
        aligner = Mock()
        aligner.align.return_value = [[aligned_word("你好", 1, 2), aligned_word("再见", 10, 11)]]
        audits = align_rows(rows, np.zeros(20 * 100), 100, aligner)
        self.assertEqual(aligner.align.call_count, 1)
        call = aligner.align.call_args.kwargs
        self.assertEqual(call["text"], "你好。\n再见。")
        self.assertEqual(len(call["audio"][0]), 1200)
        self.assertEqual(audits[0]["status"], "aligned")
        self.assertEqual([r["speaker_id"] for r in rows], ["spk_1", "spk_2"])
        self.assertEqual([r["text"] for r in rows], ["你好。", "再见。"])

    def test_oversized_owner_never_reaches_aligner(self):
        rows = parse("[0-200] spk_1: long")
        aligner = Mock()
        audits = align_rows(rows, np.zeros(201 * 100), 100, aligner)
        aligner.align.assert_not_called()
        self.assertEqual(audits[0]["status"], "owner_window_over_max")
        self.assertIn("owner_window_over_max", rows[0]["risks"])

    def test_mismatched_or_zero_duration_alignment_falls_back(self):
        for words in ([aligned_word("错误", 1, 2)], [aligned_word("你好", 1, 1)]):
            rows = parse("[1-2] spk_1: 你好")
            aligner = Mock()
            aligner.align.return_value = [words]
            audit = align_rows(rows, np.zeros(1000), 100, aligner)
            self.assertEqual(audit[0]["status"], "alignment_failed")
            self.assertEqual((rows[0]["start"], rows[0]["end"]), (1, 2))
            self.assertIn("alignment_failed", rows[0]["risks"])

    def test_mapping_rejects_word_crossing_speaker_boundary(self):
        with self.assertRaisesRegex(ValueError, "crosses"):
            map_words([dict(text="你好再见", start=0, end=1)], ["你好", "再见"])

    def test_zero_duration_internal_word_keeps_valid_edges_with_review_flag(self):
        row = parse("[1-3] spk_1: 你好吗")[0]
        words = [dict(text="你", start=1, end=2), dict(text="好", start=2, end=2),
                 dict(text="吗", start=2, end=3)]
        self.assertIsNone(apply_word_alignment(row, words, 0, 4))
        self.assertEqual((row["start"], row["end"]), (1, 3))
        self.assertIn("zero_duration_internal_word", row["risks"])

    def test_only_failed_sentence_is_retried(self):
        rows = parse("[1-2] spk_1: 你好\n[3-4] spk_2: 再见")
        aligner = Mock()
        aligner.align.side_effect = [
            [[aligned_word("你好", 1, 2), aligned_word("再见", 3, 3)]],
            [[aligned_word("你好", 1, 2), aligned_word("再见", 3, 4)]],
        ]
        audits = align_rows(rows, np.zeros(1000), 100, aligner)
        self.assertEqual(aligner.align.call_count, 2)
        self.assertEqual(rows[0]["alignment_status"], "aligned")
        self.assertEqual(rows[1]["alignment_status"], "retry_aligned")
        self.assertNotIn("alignment_failed", rows[1]["risks"])
        self.assertEqual([a["kind"] for a in audits], ["owner", "retry"])

    def test_long_segment_splits_at_words_without_changing_text(self):
        row = parse("[0-30] spk_1: 你好，再见。谢谢！")[0]
        row["words"] = [dict(text="你好", start=0, end=9), dict(text="再见", start=10, end=19),
                        dict(text="谢谢", start=20, end=30)]
        result = split_long_rows([row], 15)
        self.assertEqual(len(result), 3)
        self.assertEqual("".join(r["text"] for r in result), row["text"])
        self.assertTrue(all(r["end"] - r["start"] <= 15 for r in result))
        self.assertTrue(all(r["speaker_id"] == "spk_1" for r in result))

    def test_coarse_long_segment_requires_review_instead_of_arbitrary_text_split(self):
        row = parse("[0-30] spk_1: unaligned")[0]
        result = split_long_rows([row], 15)
        self.assertEqual(len(result), 1)
        self.assertIn("segment_over_max_review", result[0]["risks"])

    def test_padding_does_not_cut_overlapping_speech(self):
        rows = parse("[1-3] spk_1: first\n[2-4] spk_2: second")
        pad_rows(rows, np.zeros(500), 100, .15, .5, 15)
        self.assertLessEqual(rows[0]["start"], 1)
        self.assertGreaterEqual(rows[0]["end"], 3)
        self.assertLessEqual(rows[1]["start"], 2)
        self.assertTrue(all("overlap_review" in r["risks"] for r in rows))

    def test_different_speakers_are_never_merged(self):
        rows = parse("[1-2] spk_1: first\n[2.1-3] spk_2: second")
        for row in rows:
            row["owner_window"] = 0
        self.assertEqual(len(merge(rows, .8, 15)), 2)

    def test_alignment_does_not_silently_reorder_source_transcript(self):
        rows = parse("[1-2] spk_1: first\n[3-4] spk_2: second\n[5-6] <事件>")
        rows[0]["start"] = 5.5
        rows[0]["end"] = 6
        rows[1]["start"] = 5.2
        rows[1]["end"] = 5.5
        preserve_order(rows)
        self.assertEqual([r["start"] for r in rows], [1, 3, 5])
        self.assertTrue(all("order_conflict_fallback" in r["risks"] for r in rows[:2]))

    def test_export_has_valid_samples_and_no_absolute_source_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            waveform = np.sin(np.arange(32000) * .1).astype(np.float32)
            sf.write(root / "audio.wav", waveform, 8000)
            (root / "text.txt").write_text("[0.5-1.5] spk_1: 测试。", encoding="utf-8")
            output = root / "out"
            summary = export_timeline(root / "audio.wav", root / "text.txt", output)
            row = json.loads((output / "segments.jsonl").read_text().strip())
            clip, sr = sf.read(output / row["audio"])
            self.assertAlmostEqual(len(clip) / sr, row["duration"])
            self.assertGreater(len(clip), 0)
            self.assertEqual(summary["review_required"], 1)
            self.assertEqual((output / "candidates.jsonl").read_text(), "")
            self.assertNotIn(str(root), (output / "run.json").read_text())
            with self.assertRaises(FileExistsError):
                export_timeline(root / "audio.wav", root / "text.txt", output)


class PipelineTests(unittest.TestCase):
    def arguments(self, root):
        audio = root / "audio.wav"
        sf.write(audio, np.zeros(16000), 8000)
        return SimpleNamespace(audio=audio, output=root / "out", prompt_file=None,
                               aligner_python=None, coarse_only=True, firered_text=None,
                               backend="native", max_audio_seconds=3600, window_min_seconds=90,
                               window_max_seconds=150, max_segment_seconds=15, merge_gap_ms=800,
                               head_padding_ms=150, tail_padding_ms=500, language="Chinese")

    def test_one_recognition_call_connects_to_real_clip_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root)
            recognizer = Mock(return_value={"text": "[0.2-1.5] spk_1: 测试。", "finish_reason": "stop"})
            run_pipeline(args, recognizer)
            recognizer.assert_called_once()
            self.assertTrue((args.output / "clips" / "00001.wav").is_file())
            self.assertTrue((args.output / "recognition.json").is_file())

    def test_truncated_transcription_is_saved_but_never_sliced(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            recognizer = Mock(return_value={"text": "partial", "finish_reason": "length"})
            with self.assertRaisesRegex(RuntimeError, "output limit"):
                run_pipeline(args, recognizer)
            self.assertEqual((args.output / "raw_response.txt").read_text(), "partial")
            self.assertFalse((args.output / "clips").exists())

    def test_context_and_sglang_truncation_checks(self):
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            check_budget(100, 100, 150)
        self.assertTrue(is_truncated({"finish_reason": {"type": "length", "length": 100}}))
        self.assertFalse(is_truncated({"finish_reason": "stop"}))


if __name__ == "__main__":
    unittest.main()
