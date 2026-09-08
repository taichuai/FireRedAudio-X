"""CPU regression tests; run with python -m unittest discover -s tests."""

import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from fireredaudio.accelerated.client import generate
from fireredaudio.accelerated.prepare import export_backbone, text_rope_config


class LauncherTests(unittest.TestCase):
    def test_vllm_preserves_qwen35_fp32_ssm_state(self):
        launcher = (Path(__file__).resolve().parents[1] / "scripts" / "serve_accelerated.sh")
        self.assertIn("--mamba-ssm-cache-dtype float32", launcher.read_text())


class ExportTests(unittest.TestCase):
    def test_text_rope_preserves_frequency_and_source_config(self):
        original = {"rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25,
                                        "mrope_section": [11, 11, 10], "mrope_interleaved": True}}
        config = text_rope_config(original)
        self.assertEqual(config["rope_parameters"], {"rope_theta": 10000000, "partial_rotary_factor": 0.25})
        self.assertEqual(original["rope_parameters"]["mrope_section"], [11, 11, 10])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.source.mkdir()
        self.output = Path(self.temp.name) / "export"
        (self.source / "config.json").write_text(json.dumps({
            "model_type": "firered_audio", "dtype": "bfloat16",
            "backbone_config": {"model_type": "qwen3_5_text", "hidden_size": 4},
        }))

    def checkpoint(self, namespace="model.language_model", sharded=False):
        weights = {
            f"backbone_llm.{namespace}.embed_tokens.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            "backbone_llm.lm_head.weight": torch.ones(3, 4, dtype=torch.bfloat16),
            "audio_encoder.conv1.weight": torch.zeros(2, 3),
            "dit.weight": torch.zeros(1),
        }
        save_file(weights, self.source / "model.safetensors")
        if sharded:
            (self.source / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {k: "model.safetensors" for k in weights},
            }))
        return weights

    def test_export_both_namespaces_and_index_formats(self):
        for namespace, sharded in (("model.language_model", True), ("model", False)):
            with self.subTest(namespace=namespace):
                weights = self.checkpoint(namespace, sharded)
                output = self.output / namespace
                manifest = export_backbone(self.source, output)
                index = json.loads((output / "model.safetensors.index.json").read_text())
                self.assertEqual(set(index["weight_map"]), {"model.embed_tokens.weight", "lm_head.weight"})
                result = load_file(output / index["weight_map"]["model.embed_tokens.weight"])
                self.assertTrue(torch.equal(result["model.embed_tokens.weight"], weights[f"backbone_llm.{namespace}.embed_tokens.weight"]))
                self.assertEqual(manifest["backbone_bytes"], 48)
                self.assertNotIn("source", manifest)
                self.assertNotIn(str(self.source), json.dumps(manifest))
                self.assertEqual(len(manifest["source_config_sha256"]), 64)
                self.assertTrue((output / "sglang" / "model.safetensors.index.json").is_file())
                self.assertEqual(set(load_file(self.source / "model.safetensors")), set(weights))
                if sharded:
                    (self.source / "model.safetensors.index.json").unlink()

    def test_refuses_existing_output_and_source(self):
        self.checkpoint()
        before = (self.source / "config.json").read_bytes()
        with self.assertRaises(FileExistsError):
            export_backbone(self.source, self.source)
        self.assertEqual((self.source / "config.json").read_bytes(), before)

    def test_missing_weight_fails_before_creating_output(self):
        save_file({"backbone_llm.lm_head.weight": torch.zeros(3, 4)}, self.source / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "embed_tokens"):
            export_backbone(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_shard_fails_before_creating_output(self):
        self.checkpoint(sharded=True)
        (self.source / "model.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            export_backbone(self.source, self.output)
        self.assertFalse(self.output.exists())


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.embeds = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)

    def test_vllm_serialization_preserves_embeddings_and_reasoning(self):
        def respond(request, timeout):
            self.assertEqual(request.full_url, "http://localhost:8010/v1/completions")
            payload = json.loads(request.data)
            decoded = torch.load(io.BytesIO(base64.b64decode(payload["prompt_embeds"])), weights_only=True)
            self.assertTrue(torch.equal(decoded, self.embeds))
            self.assertEqual(payload["stop_token_ids"], [7])
            self.assertNotIn("prompt", payload)
            return io.BytesIO(json.dumps({"choices": [{"text": "reason</think>answer", "finish_reason": "stop"}]}).encode())
        with patch("fireredaudio.accelerated.client.urlopen", side_effect=respond):
            result = generate("http://localhost:8010/", "vllm", self.embeds, eos_id=7)
        self.assertEqual(result["text"], "reason</think>answer")

    def test_sglang_never_sends_ids_which_discard_audio(self):
        def respond(request, timeout):
            self.assertTrue(request.full_url.endswith("/generate"))
            payload = json.loads(request.data)
            self.assertNotIn("input_ids", payload)
            self.assertNotIn("text", payload)
            self.assertEqual(payload["input_embeds"], self.embeds.float().tolist())
            self.assertEqual(payload["sampling_params"]["max_new_tokens"], 23)
            return io.BytesIO(b'{"text": "answer", "meta_info": {}}')
        with patch("fireredaudio.accelerated.client.urlopen", side_effect=respond):
            generate("http://localhost", "sglang", self.embeds, eos_id=7, max_tokens=23)

    def test_invalid_embeddings_fail_before_network(self):
        for embeds in (torch.zeros(4), torch.empty(0, 4), torch.full((2, 4), float("nan"))):
            with self.subTest(shape=embeds.shape), self.assertRaises(ValueError):
                generate("http://localhost", "vllm", embeds, eos_id=7)


if __name__ == "__main__":
    unittest.main()
