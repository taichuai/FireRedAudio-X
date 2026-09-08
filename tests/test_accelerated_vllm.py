"""Optional numerical test: run in .venv-vllm after installing its dependencies."""

import importlib.util
import unittest

import torch

from fireredaudio.accelerated.prepare import text_rope_config


@unittest.skipUnless(importlib.util.find_spec("vllm"), "requires the vLLM environment")
class RotaryEquivalenceTests(unittest.TestCase):
    def test_equal_mrope_axes_match_text_rope(self):
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.rotary_embedding import get_rope

        original = {"rope_parameters": {
            "rope_type": "default", "rope_theta": 10000000,
            "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        }}
        normalized = text_rope_config(original)
        positions = torch.tensor([0, 1, 7, 63, 127, 255])
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), set_current_vllm_config(VllmConfig()):
                mrope = get_rope(256, 512, rope_parameters=original["rope_parameters"], dtype=dtype)
                rope = get_rope(256, 512, rope_parameters=normalized["rope_parameters"], dtype=dtype)
                query = torch.randn(6, 16 * 256, dtype=dtype)
                key = torch.randn(6, 4 * 256, dtype=dtype)
                expected = mrope.forward_native(positions.expand(3, -1), query.clone(), key.clone())
                actual = rope.forward_native(positions, query.clone(), key.clone())
                for a, b in zip(expected, actual):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
