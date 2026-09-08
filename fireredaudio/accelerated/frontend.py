"""Build exactly the embeddings used by the original understanding inference."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import SinusoidsPositionEmbedding

from fireredaudio.audio_encoder.configuration_audio_encoder import FireRedAudioEncoderConfig
from fireredaudio.audio_encoder.modeling_audio_encoder import FireRedAudioEncoder
from fireredaudio.audio_encoder.processor import FireRedAudioProcessor
from fireredaudio.data.prompt_encoder import AudioPromptEncoder, FEAT_TYPE_UNDERSTAND
from fireredaudio.utils.audio import read_audio, UNDERSTAND_SAMPLE_RATE
from .prepare import weight_map, backbone_key


class AudioEmbeddingFrontend:
    def __init__(self, model_path: str, device: str = "cuda:0", attention: str = "sdpa"):
        source = Path(model_path)
        self.config = json.loads((source / "config.json").read_text())
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.tokenizer.padding_side = "left"
        self.processor = FireRedAudioProcessor.from_pretrained(source)
        self.prompt_encoder = AudioPromptEncoder(
            self.tokenizer, self.processor, self.config["audio_special_token"],
            self.config["audio_special_token_no_latent"],
        )
        config = FireRedAudioEncoderConfig(**self.config["audio_encoder_config"])
        config._attn_implementation = attention
        with torch.device("meta"):
            self.audio_encoder = FireRedAudioEncoder(config)
        # This deterministic nonpersistent buffer is absent from the checkpoint.
        self.audio_encoder.positional_embedding = SinusoidsPositionEmbedding(
            config.max_source_positions, config.d_model,
        )
        mapping = weight_map(source)
        state = {}
        embedding_keys = [k for k in mapping if k.startswith("backbone_llm.")
                          and backbone_key(k) == "model.embed_tokens.weight"]
        if len(embedding_keys) != 1:
            raise ValueError("Expected exactly one backbone embedding table")
        embedding_key = embedding_keys[0]
        selected = {k: v for k, v in mapping.items()
                    if k.startswith("audio_encoder.") or k == embedding_key}
        for shard in sorted(set(selected.values())):
            with safe_open(source / shard, framework="pt", device="cpu") as reader:
                for key, filename in selected.items():
                    if filename != shard:
                        continue
                    tensor = reader.get_tensor(key)
                    if key == embedding_key:
                        # A CPU embedding table avoids spending ~2 GB of encoder GPU RAM.
                        self.embedding = torch.nn.Embedding.from_pretrained(tensor, freeze=True)
                    else:
                        state[key.removeprefix("audio_encoder.")] = tensor
        self.audio_encoder.load_state_dict(state, strict=True, assign=True)
        self.audio_encoder = self.audio_encoder.to(self.device, dtype=torch.bfloat16).eval()
        if not hasattr(self, "embedding"):
            raise ValueError(f"Missing {embedding_key}")
        self.eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    @torch.inference_mode()
    def encode(self, audio_paths: list[str], prompt: str, enable_thinking: bool = False):
        # Reuse the repository's training-aligned prompt, including the think prefix.
        from inference import build_understand_prompt

        if not audio_paths:
            raise ValueError("At least one audio is required")
        audios = []
        for path in audio_paths:
            waveform = read_audio(path, UNDERSTAND_SAMPLE_RATE)
            if not waveform.numel() or not torch.isfinite(waveform).all():
                raise ValueError(f"Empty or non-finite audio: {path}")
            audios.append({"feat_type": FEAT_TYPE_UNDERSTAND,
                           "audio_understand": waveform.numpy(),
                           "audio_generation": None, "role": "user"})
        batch = self.prompt_encoder.encode(build_understand_prompt(
            prompt, len(audios), self.config["audio_special_token"], enable_thinking,
        ), audios)
        features = batch["audio_features"].to(self.device)
        mask = batch["audio_feature_attention_mask"].to(self.device).bool()
        lengths = mask.sum(-1)
        cnn_lengths, output_lengths = self.audio_encoder._get_feat_extract_output_lengths(lengths)
        packed = features.permute(0, 2, 1)[mask].permute(1, 0)
        audio_embeds = self.audio_encoder(
            packed, feature_lens=lengths, aftercnn_lens=cnn_lengths,
        ).cpu()
        ids = batch["input_ids"][0]
        audio_mask = ids == self.config["audio_special_token_id"]
        if audio_embeds.shape[0] != int(audio_mask.sum()) or int(output_lengths.sum()) != int(audio_mask.sum()):
            raise ValueError("Audio embedding length does not match prompt placeholders")
        embeds = self.embedding(ids).to(torch.bfloat16)
        embeds[audio_mask] = audio_embeds.to(embeds.dtype)
        return embeds.contiguous(), ids
