"""Expose vLLM 0.18's Qwen3.5 text model with its hybrid state interface."""

from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)


class FireRedAudioBackbone(Qwen3_5ForCausalLM, IsHybrid):
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()
