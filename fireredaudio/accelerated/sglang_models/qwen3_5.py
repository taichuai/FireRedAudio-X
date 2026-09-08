"""Text-only wrapper around SGLang's Qwen3.5 implementation (0.5.9)."""

import torch
from torch import nn
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForCausalLM as Qwen3_5Backbone,
    Qwen3_5ForConditionalGeneration as UpstreamQwen3_5,
)


class Qwen3_5ForConditionalGeneration(UpstreamQwen3_5):
    def __init__(self, config, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        if quant_config is not None:
            raise ValueError("The FireRedAudio adapter supports unquantized BF16 weights")
        self.pp_group = get_pp_group()
        if self.pp_group.world_size != 1:
            raise ValueError("Use tensor parallelism; pipeline parallelism is unsupported")
        self.config = config.text_config
        self.model = Qwen3_5Backbone(self.config, prefix=f"{prefix}.model".lstrip("."))
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size,
                                     prefix=f"{prefix}.lm_head".lstrip("."))
        self.logits_processor = LogitsProcessor(self.config)
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        # All positions are one-dimensional audio/text sequence positions. Replicate
        # them across MRoPE axes, just as HF does for inputs_embeds-only generation.
        if self.is_mrope_enabled and positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        hidden = self.model(input_ids, positions, forward_batch, input_embeds=input_embeds)
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        missing = set(dict(self.named_parameters())) - loaded
        if missing:
            raise ValueError(f"Incomplete FireRedAudio backbone weights: {sorted(missing)}")
        return loaded


EntryClass = Qwen3_5ForConditionalGeneration
