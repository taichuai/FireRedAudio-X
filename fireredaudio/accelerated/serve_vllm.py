"""Register the text architecture in both the server and spawned workers."""

import runpy

from transformers import AutoConfig
from vllm import ModelRegistry
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig

AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)
ModelRegistry.register_model(
    "Qwen3_5ForCausalLM", "fireredaudio.accelerated.vllm_model:FireRedAudioBackbone",
)

if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
