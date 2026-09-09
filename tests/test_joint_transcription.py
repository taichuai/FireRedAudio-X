"""Joint transcription without model weights or evaluation dependencies."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts.run_paralanguage_joint_transcription import load_manifest, transcribe


class JointTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_json_and_jsonl_require_no_annotations(self):
        rows = [{"id": "one", "path": "audio.wav"}, {"id": "two", "audio": "other.wav"}]
        for content in (json.dumps(rows), "\n".join(json.dumps(row) for row in rows)):
            manifest = self.root / "inputs.json"
            manifest.write_text(content)
            self.assertEqual(load_manifest(manifest), [
                {"id": "one", "path": "audio.wav"}, {"id": "two", "path": "other.wav"},
            ])

    def test_invalid_manifest(self):
        for rows in ([], [{"path": []}], [{"id": "same", "path": "a"}, {"id": "same", "path": "b"}]):
            manifest = self.root / "inputs.json"
            manifest.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                load_manifest(manifest)

    def test_transcription_preserves_tags_and_words(self):
        (self.root / "audio.wav").touch()
        raw = "  没有咳嗽声。\n哈哈镜 <轻笑> <自定义标签>  "
        engine = Mock()
        engine.understand.return_value = SimpleNamespace(answer=raw)
        result = transcribe(engine, {"id": 1, "path": "audio.wav"}, self.root, "prompt", 1024)
        self.assertEqual(result["joint_transcription_raw"], raw)
        self.assertEqual(result["joint_transcription"], "没有咳嗽声。 哈哈镜 <轻笑> <自定义标签>")
        self.assertEqual(result["raw_angle_tags"], ["<轻笑>", "<自定义标签>"])
        self.assertIsNone(result["error"])
        engine.understand.assert_called_once_with(str(self.root / "audio.wav"), "prompt", task="understand", max_new_tokens=1024)

    def test_failed_record_does_not_block_next_record(self):
        engine = Mock()
        engine.understand.return_value = SimpleNamespace(answer="正常转写")
        missing = transcribe(engine, {"id": 1, "path": "missing.wav"}, self.root, "prompt", 1024)
        self.assertIn("FileNotFoundError", missing["error"])
        engine.understand.assert_not_called()
        (self.root / "audio.wav").touch()
        good = transcribe(engine, {"id": 2, "path": "audio.wav"}, self.root, "prompt", 1024)
        self.assertIsNone(good["error"])
        engine.understand.return_value = SimpleNamespace(answer="  ")
        empty = transcribe(engine, {"id": 3, "path": "audio.wav"}, self.root, "prompt", 1024)
        self.assertIn("empty transcription", empty["error"])


if __name__ == "__main__":
    unittest.main()
