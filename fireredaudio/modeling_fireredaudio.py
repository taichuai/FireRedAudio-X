import warnings

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import GenerationConfig, PreTrainedModel
from transformers.generation.logits_process import LogitsProcessorList
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack

from .configuration_fireredaudio import FireRedAudioConfig
from .audio_encoder.modeling_audio_encoder import FireRedAudioEncoder
from .redae.encoder import RedAEAudioEncoderV1
from .flow.estimator import RedDiT
from .flow.patch_encoder import RedPatchEncoder
from .training.losses import weighted_text_loss


class FireRedAudioForCausalLM(PreTrainedModel):
    config_class = FireRedAudioConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: FireRedAudioConfig):
        super().__init__(config)

        # backbone llm
        self.backbone_llm = Qwen3_5ForCausalLM(config=self.config.backbone_config)

        # audio encoder
        self.audio_encoder = FireRedAudioEncoder(self.config.audio_encoder_config)

        # red_vae
        self.red_vae = RedAEAudioEncoderV1(self.config.red_vae_config)
        self.red_vae.eval()

        # patch encoder
        self.patch_encoder = RedPatchEncoder(self.config.patch_encoder_config)

        # dit head
        self.dit = RedDiT(self.config.dit_config)

        self.post_init()

    def train(self, mode: bool = True):
        super().train(mode)
        self.red_vae.eval()
        return self

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.LongTensor | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[
                feature_attention_mask.bool()
            ].permute(1, 0)
        else:
            audio_feature_lengths = None

        audio_feat_lengths, audio_output_lengths = (
            self.audio_encoder._get_feat_extract_output_lengths(
                audio_feature_lengths
                if audio_feature_lengths is not None
                else feature_attention_mask.sum(-1)
            )
        )
        feature_lens = (
            audio_feature_lengths
            if audio_feature_lengths is not None
            else feature_attention_mask.sum(-1)
        )
        audio_outputs = self.audio_encoder(
            input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
            return_dict=True,
            **kwargs,
        )
        if audio_outputs.shape[0] != sum(audio_output_lengths.tolist()):
            raise ValueError(
                "length of audio_features should match audio_output_lengths"
            )

        return audio_outputs

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        audio_features: torch.Tensor | None = None,
        audio_feature_attention_mask: torch.Tensor | None = None,
        vae_audios: torch.Tensor | None = None,
        patch_encoder_output_attention_mask: torch.Tensor | None = None,
        generation_target_start_patches: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        label_weights: torch.Tensor | None = None,
        return_logits: bool = False,
        loss_chunk_size: int = 64,
        flow_chunk_size: int = 32,
        checkpoint_backbone: bool = False,
    ) -> dict[str, torch.Tensor | int | None]:
        """Teacher-forced understanding and continuous-latent training forward.

        Generation training currently accepts one serialized conversation per
        replica. ``generation_target_start_patches[i]`` is the first supervised
        6.25 Hz patch in generation audio segment ``i``; ``-1`` makes the whole
        segment conditioning-only. The frozen RedAE still produces native 25 Hz
        targets, while Patch Encoder outputs are inserted into the LLM sequence.
        """
        if flow_chunk_size <= 0:
            raise ValueError("flow_chunk_size must be positive")
        input_embeds = self.backbone_llm.model.get_input_embeddings()(input_ids)

        if audio_features is not None and audio_features.numel() > 0:
            understanding_output = self.get_audio_features(
                input_features=audio_features,
                feature_attention_mask=audio_feature_attention_mask,
            )
            understanding_mask = input_ids == self.config.audio_special_token_id
            if understanding_output.shape[0] != int(understanding_mask.sum().item()):
                raise ValueError("understanding features do not match <|AUDIO|> positions")
            input_embeds = input_embeds.masked_scatter(
                understanding_mask.unsqueeze(-1).expand_as(input_embeds),
                understanding_output.to(input_embeds.dtype),
            )

        vae_latents = None
        generation_positions: list[torch.Tensor] = []
        if vae_audios is not None and vae_audios.numel() > 0:
            if input_ids.shape[0] != 1:
                raise NotImplementedError(
                    "generation training currently supports one conversation per replica"
                )
            if patch_encoder_output_attention_mask is None:
                raise ValueError("generation audio requires patch attention masks")
            with torch.no_grad():
                vae_latents = self.red_vae.encode(vae_audios).transpose(1, 2)
            patch_embeddings = self.patch_encoder(vae_latents)
            ragged_patch_embeddings = patch_embeddings[
                patch_encoder_output_attention_mask.bool()
            ]
            generation_mask = input_ids == self.config.audio_special_no_latent_id
            flat_positions = generation_mask[0].nonzero(as_tuple=True)[0]
            if ragged_patch_embeddings.shape[0] != flat_positions.numel():
                raise ValueError("generation patches do not match <|AUDIO_NO_LATENT|> positions")
            input_embeds = input_embeds.masked_scatter(
                generation_mask.unsqueeze(-1).expand_as(input_embeds),
                ragged_patch_embeddings.to(input_embeds.dtype),
            )
            cursor = 0
            for valid in patch_encoder_output_attention_mask.sum(dim=1).tolist():
                generation_positions.append(flat_positions[cursor : cursor + int(valid)])
                cursor += int(valid)

        def run_backbone(embeds):
            return self.backbone_llm.model(
                inputs_embeds=embeds, attention_mask=attention_mask,
                use_cache=False, return_dict=True,
            ).last_hidden_state

        # This also checkpoints a frozen backbone in eval mode, so gradients can
        # reach an upstream trainable audio/patch adapter without enabling dropout.
        if checkpoint_backbone and torch.is_grad_enabled():
            hidden_states = checkpoint(run_backbone, input_embeds, use_reentrant=False)
        else:
            hidden_states = run_backbone(input_embeds)

        text_loss = None
        text_loss_sum = None
        text_weight = hidden_states.new_zeros((), dtype=torch.float32)
        logits = self.backbone_llm.lm_head(hidden_states) if return_logits else None
        if labels is not None:
            text_loss_sum, text_weight = weighted_text_loss(
                hidden_states, self.backbone_llm.lm_head, labels,
                label_weights, loss_chunk_size,
            )
            text_loss = text_loss_sum / text_weight.clamp_min(torch.finfo(torch.float32).tiny)

        flow_loss = None
        flow_loss_sum = None
        flow_count = 0
        if generation_target_start_patches is not None:
            if vae_latents is None:
                raise ValueError("flow targets require generation audio")
            starts = generation_target_start_patches.tolist()
            if len(starts) != len(generation_positions):
                raise ValueError("one target start is required per generation audio segment")

            all_conditions = []
            all_histories = []
            all_targets = []
            history_frames = self.dit.history_length
            history_steps = self.dit.history_patches
            patch_size = self.dit.patch_size
            hidden_size = hidden_states.shape[-1]
            for audio_idx, (positions, raw_start) in enumerate(
                zip(generation_positions, starts)
            ):
                start = int(raw_start)
                num_patches = positions.numel()
                if start == -1:
                    continue
                if start < -1 or start >= num_patches:
                    raise ValueError(
                        f"target start {start} is outside audio segment with {num_patches} patches"
                    )
                if bool((positions <= 0).any()):
                    raise ValueError("every generation patch needs a preceding LLM position")
                step_conditions = hidden_states[0, positions - 1]
                valid_latents = vae_latents[audio_idx, : num_patches * patch_size]
                for patch_idx in range(start, num_patches):
                    cond_start = max(0, patch_idx - history_steps)
                    cond = step_conditions[cond_start : patch_idx + 1]
                    if cond.shape[0] < history_steps + 1:
                        cond = F.pad(
                            cond,
                            (0, 0, history_steps + 1 - cond.shape[0], 0),
                        )
                    frame_start = patch_idx * patch_size
                    history = valid_latents[max(0, frame_start - history_frames) : frame_start]
                    if history.shape[0] < history_frames:
                        history = F.pad(
                            history,
                            (0, 0, history_frames - history.shape[0], 0),
                        )
                    target = valid_latents[frame_start : frame_start + patch_size]
                    if target.shape != (patch_size, self.dit.config.vae_channels):
                        raise ValueError("incomplete RedAE target patch")
                    if cond.shape != (history_steps + 1, hidden_size):
                        raise ValueError("invalid LLM conditioning window")
                    all_conditions.append(cond)
                    all_histories.append(history)
                    all_targets.append(target)

            if all_targets:
                conditions = torch.stack(all_conditions)
                targets = torch.stack(all_targets)
                histories = torch.stack(all_histories)
                flow_count = len(all_targets)
                flow_loss_sum = hidden_states.new_zeros((), dtype=torch.float32)
                for start in range(0, flow_count, flow_chunk_size):
                    args = (conditions[start:start + flow_chunk_size],
                            targets[start:start + flow_chunk_size],
                            histories[start:start + flow_chunk_size])
                    if torch.is_grad_enabled():
                        chunk_loss = checkpoint(self.dit.compute_loss, *args, use_reentrant=False)
                    else:
                        chunk_loss = self.dit.compute_loss(*args)
                    flow_loss_sum = flow_loss_sum + chunk_loss * args[0].shape[0]
                flow_loss = flow_loss_sum / flow_count

        losses = [loss for loss in (text_loss, flow_loss) if loss is not None]
        loss = torch.stack(losses).sum() if losses else None
        return {
            "loss": loss,
            "text_loss": text_loss,
            "flow_loss": flow_loss,
            "text_loss_sum": text_loss_sum,
            "text_weight": text_weight,
            "flow_loss_sum": flow_loss_sum,
            "flow_count": flow_count,
            "logits": logits if return_logits else None,
        }

    @torch.inference_mode()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        audio_features=None,
        audio_feature_attention_mask=None,
        **kwargs,
    ):
        # Step1. token embeddings
        input_embeds = self.backbone_llm.model.get_input_embeddings()(input_ids)

        # Step2. audio encoder features
        audio_encoder_output = self.get_audio_features(
            input_features=audio_features,
            feature_attention_mask=audio_feature_attention_mask,
        )

        audio_placeholder_mask_2d = input_ids == self.config.audio_special_token_id
        num_placeholders = audio_placeholder_mask_2d.sum().item()
        if audio_encoder_output.shape[0] != num_placeholders:
            raise ValueError(
                f"audio_encoder_output length ({audio_encoder_output.shape[0]}) "
                f"!= num_placeholders ({num_placeholders})"
            )
        audio_placeholder_mask_2d = audio_placeholder_mask_2d.unsqueeze(-1).expand_as(input_embeds)
        input_embeds = input_embeds.masked_scatter(audio_placeholder_mask_2d, audio_encoder_output)

        # Step3. delegate to backbone LLM generate
        return self.backbone_llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
    
    # --- Flow one-step generation
    def _flow_onestep_generate(
        self, 
        backbone_output: torch.Tensor,
        history_vae_latents: torch.Tensor,
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
    ):
        """

        Args:
            backbone_output(torch.Tensor): full length backbone output, shape (b=1, t', c)
            history_vae_latents(torch.Tensor): history generated vae latents, shape (b=1, t'', c)
            n_timesteps(int): default to 10
        Return:
            one_vae_latents(torch.Tensor): this step generated vae latents
            next_backbone_input_embeds(torch.Tensor): next backbone step patched latents
        """
        one_vae_latents = self.dit.generate(
            backbone_output=backbone_output,
            history_vae_latents=history_vae_latents,
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
        )
        next_backbone_input_embeds = self.patch_encoder(one_vae_latents)
        return one_vae_latents, next_backbone_input_embeds        

    @torch.inference_mode()
    def generate_tts(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        vae_audios: torch.Tensor | None = None,
        vae_is_assistant: torch.Tensor | None = None,
        patch_encoder_output_attention_mask: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        max_new_audio_steps: int = 750,
        min_new_audio_steps: int = 0,
        max_new_text_tokens: int = 32,
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
    ):
        """Hybrid AR generation: text token sampling plus audio chunk DiT denoising.

        Text steps sample from lm_head through HF LogitsProcessor; audio steps call
        _flow_onestep_generate for 4 consecutive AE latents per chunk. <|sosp|> and
        <|eosp|> switch between the two modes. batch_size=1 only.

        Audio on both the user and assistant side goes through the generation path
        (VAE + patch_encoder), so audio_features / audio_feature_attention_mask are
        not exposed here.

        Args:
            input_ids: (1, T_p) chatml token ids with placeholders already expanded.
            attention_mask: (1, T_p) left padded, same shape as input_ids.
            vae_audios / vae_is_assistant / patch_encoder_output_attention_mask:
                Optional generation-side reference audio. vae_is_assistant marks which
                of them sit in the assistant turn; generation continues from the last
                such one, which is how ICL voice cloning works. Audio given in the user
                turn leaves it False and serves as context only.
            generation_config: Controls text sampling (do_sample / temperature / top_p /
                top_k / repetition_penalty / eos_token_id).
            max_new_audio_steps: Cap on audio chunks; each step is 4 AE latents (~160 ms).
            min_new_audio_steps: Suppress <|eosp|> for the first N steps so generation
                cannot stop immediately. N=6 is ~0.96 s; 0 disables it.
            max_new_text_tokens: Cap on text steps, guarding against a text-mode loop.
            n_timesteps / inference_cfg: Passed through to _flow_onestep_generate.

        Returns:
            text_token_ids: (1, T_gen) newly generated text tokens, including
                sosp / eosp / im_end.
            vae_latents: (1, N*4, 64) AE latents in generation order, ready for an
                external VAE decoder.
        """
        # ---- Phase 0: argument resolution ----
        if input_ids.shape[0] != 1:
            raise NotImplementedError("generate_tts only supports batch_size=1 for now")

        gen_cfg = generation_config if generation_config is not None else GenerationConfig()

        raw = gen_cfg.eos_token_id
        if raw is None:
            raise ValueError("generation_config.eos_token_id is required")
        eos_set = {int(x) for x in raw} if isinstance(raw, (list, tuple)) else {int(raw)}

        sosp_id = self.config.sosp_idx
        eosp_id = self.config.eosp_idx
        audio_no_latent_id = self.config.audio_special_no_latent_id
        device = input_ids.device

        # Text-side LogitsProcessor (repetition_penalty / temperature / top_k / top_p)
        from transformers.generation.logits_process import (
            RepetitionPenaltyLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )
        text_logits_processor = LogitsProcessorList()
        if gen_cfg.repetition_penalty is not None and float(gen_cfg.repetition_penalty) != 1.0:
            text_logits_processor.append(
                RepetitionPenaltyLogitsProcessor(penalty=float(gen_cfg.repetition_penalty))
            )
        if gen_cfg.do_sample:
            if gen_cfg.temperature is not None and float(gen_cfg.temperature) != 1.0:
                text_logits_processor.append(
                    TemperatureLogitsWarper(temperature=float(gen_cfg.temperature))
                )
            if gen_cfg.top_k is not None and int(gen_cfg.top_k) > 0:
                text_logits_processor.append(TopKLogitsWarper(top_k=int(gen_cfg.top_k)))
            if gen_cfg.top_p is not None and float(gen_cfg.top_p) < 1.0:
                text_logits_processor.append(TopPLogitsWarper(top_p=float(gen_cfg.top_p)))

        # ---- Phase 1: Prefill ----
        # Step1. token embeddings
        input_embeds = self.backbone_llm.model.get_input_embeddings()(input_ids)

        # Step2. Scatter generation-side ref audio (VAE + patch_encoder) into <|AUDIO_NO_LATENT|>
        ref_vae_latents = None
        if vae_audios is not None and vae_audios.shape[0] > 0:
            ref_vae_latents = self.red_vae.encode(vae_audios).transpose(1, 2)  # (N_gen, T_vae_max, 64)
            patch_enc_latent = self.patch_encoder(ref_vae_latents, None)        # (N_gen, T_vae_max//4, H)
            patch_enc_raged = patch_enc_latent[patch_encoder_output_attention_mask]
            mask_gen = (input_ids == audio_no_latent_id)
            num_gen_ph = int(mask_gen.sum().item())
            if patch_enc_raged.shape[0] != num_gen_ph:
                raise ValueError(
                    f"patch_encoder_output length ({patch_enc_raged.shape[0]}) "
                    f"!= num_placeholders ({num_gen_ph})"
                )
            input_embeds = input_embeds.masked_scatter(
                mask_gen.unsqueeze(-1).expand_as(input_embeds),
                patch_enc_raged,
            )

        # Prefill backbone forward, building the KV cache
        out = self.backbone_llm.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past_kv = out.past_key_values
        prefill_h = out.last_hidden_state           # (1, T_p, H), kept whole for Phase 1.5
        current_h = prefill_h[:, -1:]               # (1, 1, H)
        attn_mask = attention_mask                  # one column appended per step

        embed_tokens = self.backbone_llm.model.get_input_embeddings()

        # Pick the initial mode and preload backbone_audio_hiddens
        ids_1d = input_ids[0]
        sosp_positions = (ids_1d == sosp_id).nonzero(as_tuple=True)[0]
        eosp_positions = (ids_1d == eosp_id).nonzero(as_tuple=True)[0]
        last_sosp_pos = int(sosp_positions[-1].item()) if sosp_positions.numel() > 0 else -1
        last_eosp_pos = int(eosp_positions[-1].item()) if eosp_positions.numel() > 0 else -1
        initial_mode = "audio" if last_sosp_pos > last_eosp_pos else "text"

        H = current_h.shape[-1]
        vae_channels = self.dit.config.vae_channels
        empty_h = torch.empty((1, 0, H), dtype=current_h.dtype, device=device)

        if initial_mode == "audio":
            backbone_audio_hiddens = prefill_h[:, last_sosp_pos:-1].contiguous()  # (1, K-1, H)
        else:
            backbone_audio_hiddens = empty_h  # (1, 0, H)

        # Release the full prefill_h
        prefill_h = None

        # ---- Phase 2: Hybrid AR loop ----
        mode = initial_mode
        generated_text_ids: list[int] = []
        generated_vae_latents = torch.empty(
            (1, 0, vae_channels), dtype=current_h.dtype, device=device
        )

        if (
            initial_mode == "audio"
            and ref_vae_latents is not None
            and patch_encoder_output_attention_mask is not None
            and vae_is_assistant is not None
            and bool(vae_is_assistant.any())
        ):
            assistant_idx = vae_is_assistant.nonzero(as_tuple=True)[0]
            last_idx = int(assistant_idx[-1].item())
            valid_patch_len = int(patch_encoder_output_attention_mask[last_idx].sum().item())
            valid_vae_len = valid_patch_len * self.dit.patch_size
            if valid_vae_len > 0:
                history_vae_latents = ref_vae_latents[last_idx : last_idx + 1, :valid_vae_len].contiguous()
            else:
                history_vae_latents = None
        else:
            history_vae_latents = None

        n_text_tokens_emitted = 0
        n_audio_steps_emitted = 0

        while True:
            if mode == "text":
                # Sample a text token
                logits = self.backbone_llm.lm_head(current_h[:, -1, :])  # (1, V)
                if generated_text_ids:
                    input_ids_so_far = torch.tensor(
                        [generated_text_ids], dtype=torch.long, device=device
                    )
                else:
                    input_ids_so_far = torch.empty((1, 0), dtype=torch.long, device=device)
                scores = text_logits_processor(input_ids_so_far, logits)
                if gen_cfg.do_sample:
                    probs = torch.softmax(scores.float(), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)[:, 0]  # (1,)
                else:
                    next_token = scores.argmax(dim=-1)                          # (1,)

                tok = int(next_token.item())
                generated_text_ids.append(tok)
                n_text_tokens_emitted += 1

                # Stop
                if tok in eos_set:
                    break

                # Mode switch
                next_mode = "audio" if tok == sosp_id else "text"

                # Feed next_token's embedding to the backbone, mode switch or not
                next_embed = embed_tokens(next_token.view(1, 1))  # (1, 1, H)
                attn_mask = torch.cat(
                    [attn_mask, torch.ones((1, 1), dtype=attn_mask.dtype, device=device)],
                    dim=1,
                )
                out = self.backbone_llm.model(
                    inputs_embeds=next_embed,
                    attention_mask=attn_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = out.past_key_values
                current_h = out.last_hidden_state  # (1, 1, H)
                mode = next_mode

                # Guard against a text-mode loop
                if mode == "text" and n_text_tokens_emitted >= max_new_text_tokens:
                    warnings.warn(
                        f"generate_tts: text mode hit max_new_text_tokens={max_new_text_tokens} "
                        "without producing <|sosp|>; returning early."
                    )
                    break

            else:  # mode == "audio"
                # Collect backbone hiddens for the audio span; DiT.generate conditions
                # on only the last 3 frames.
                backbone_audio_hiddens = torch.cat(
                    [backbone_audio_hiddens, current_h[:, -1:]], dim=1
                )  # (1, K+1, H)

                # One patch = 4 AE latents
                one_vae_latents, next_input_embeds = self._flow_onestep_generate(
                    backbone_output=backbone_audio_hiddens,
                    history_vae_latents=history_vae_latents,
                    n_timesteps=n_timesteps,
                    inference_cfg=inference_cfg,
                )
                # one_vae_latents: (1, 4, 64); next_input_embeds: (1, 1, H)

                generated_vae_latents = torch.cat(
                    [generated_vae_latents, one_vae_latents], dim=1
                )
                history_vae_latents = (
                    one_vae_latents
                    if history_vae_latents is None
                    else torch.cat([history_vae_latents, one_vae_latents], dim=1)
                )
                n_audio_steps_emitted += 1

                # Feed patch_encoder output back in as continuous embeddings, not token ids
                attn_mask = torch.cat(
                    [attn_mask, torch.ones((1, 1), dtype=attn_mask.dtype, device=device)],
                    dim=1,
                )
                out = self.backbone_llm.model(
                    inputs_embeds=next_input_embeds,
                    attention_mask=attn_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = out.past_key_values
                current_h = out.last_hidden_state  # (1, 1, H)

                # <|eosp|> is the only exit signal. The other placeholder positions do
                # carry a next-token loss, but at ce_weight 0.01 against 1.0 elsewhere,
                # so lm_head is only weakly supervised there and its argmax means little.
                audio_exit_logits = self.backbone_llm.lm_head(current_h[:, -1, :])  # (1, V)
                if n_audio_steps_emitted < min_new_audio_steps:
                    audio_exit_logits[:, eosp_id] = float("-inf")
                next_argmax = int(audio_exit_logits.argmax(dim=-1).item())

                if next_argmax == eosp_id:
                    generated_text_ids.append(eosp_id)
                    eosp_embed = embed_tokens(
                        torch.tensor([[eosp_id]], dtype=torch.long, device=device)
                    )
                    attn_mask = torch.cat(
                        [attn_mask, torch.ones((1, 1), dtype=attn_mask.dtype, device=device)],
                        dim=1,
                    )
                    out = self.backbone_llm.model(
                        inputs_embeds=eosp_embed,
                        attention_mask=attn_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_kv = out.past_key_values
                    current_h = out.last_hidden_state
                    mode = "text"
                    backbone_audio_hiddens = empty_h  # reset to (1, 0, H)
                elif n_audio_steps_emitted >= max_new_audio_steps:
                    warnings.warn(
                        f"generate_tts: hit max_new_audio_steps={max_new_audio_steps} "
                        "without producing <|eosp|>; returning truncated latents."
                    )
                    break

        # ---- Phase 3: Return ----
        if generated_text_ids:
            text_token_ids = torch.tensor([generated_text_ids], dtype=torch.long, device=device)
        else:
            text_token_ids = torch.empty((1, 0), dtype=torch.long, device=device)

        return text_token_ids, generated_vae_latents  # (1, N*4, 64) or (1, 0, 64)



from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("firered_audio", FireRedAudioConfig)
AutoModelForCausalLM.register(FireRedAudioConfig, FireRedAudioForCausalLM)
