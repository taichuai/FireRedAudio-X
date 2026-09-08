"""CPU tests for SFT objectives, data boundaries, resume and checkpoint export."""

import copy
import json
from pathlib import Path
import random
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from fireredaudio.modeling_fireredaudio import FireRedAudioForCausalLM
from fireredaudio.flow.estimator import RedDiT
from fireredaudio.training.checkpoints import (
    atomic_save, export_checkpoint, fingerprint, read_checkpoint, restore_checkpoint, save_checkpoint,
)
from fireredaudio.training.data import load_manifest, prepare_sample, read_checked_audio
from fireredaudio.training.losses import supervision_counts, weighted_text_loss
from fireredaudio.training.trainer import (
    ShuffledSampler, backward_window, capture_rng, configure_modules, evaluate,
    make_optimizer, make_scheduler, restore_rng,
)

torch.set_num_threads(1)


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.proj = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.2)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, inputs_embeds, **kwargs):
        return SimpleNamespace(last_hidden_state=self.dropout(torch.tanh(self.proj(inputs_embeds).cumsum(1))))


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def encode(self, waveform):
        return waveform.view(waveform.shape[0], -1, 4).transpose(1, 2) * self.scale


class TinyPatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 8)

    def forward(self, latents):
        return self.proj(latents.reshape(latents.shape[0], -1, 4, 4).mean(2))


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vae_channels=4)
        self.patch_size, self.history_patches, self.history_length = 4, 2, 8
        self.proj = nn.Linear(8, 4)

    def compute_loss(self, conditions, targets, histories):
        self.conditions = conditions.detach().clone()
        self.histories = histories.detach().clone()
        prediction = self.proj(conditions.mean(1)).unsqueeze(1).expand_as(targets)
        return F.mse_loss(prediction, targets)


def tiny_model():
    model = FireRedAudioForCausalLM.__new__(FireRedAudioForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(audio_special_token_id=7, audio_special_no_latent_id=8)
    model.backbone_llm = nn.Module()
    model.backbone_llm.model = TinyLanguageModel()
    model.backbone_llm.lm_head = nn.Linear(8, 16, bias=False)
    model.audio_encoder = nn.Module()
    model.audio_encoder.adapter = nn.Linear(8, 8)
    model.red_vae = TinyVAE()
    model.patch_encoder = TinyPatchEncoder()
    model.dit = TinyDiT()
    return model


def text_batch(ids):
    ids = torch.tensor([ids])
    labels = ids.clone()
    labels[:, :2] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


class LossTests(unittest.TestCase):
    def test_chunked_ce_matches_dense_loss_and_gradients(self):
        torch.manual_seed(3)
        states = torch.randn(2, 7, 5, requires_grad=True)
        dense_states = states.detach().clone().requires_grad_()
        head = nn.Linear(5, 11)
        dense_head = copy.deepcopy(head)
        labels = torch.randint(0, 11, (2, 7))
        labels[:, :3] = -100
        weights = torch.ones(2, 7)
        weights[:, 3:5] = 0.01
        numerator, denominator = weighted_text_loss(states, head, labels, weights, chunk_size=2)
        valid_weights = weights[:, 1:] * (labels[:, 1:] != -100)
        dense = F.cross_entropy(dense_head(dense_states[:, :-1]).transpose(1, 2), labels[:, 1:],
                                reduction="none", ignore_index=-100)
        expected = (dense * valid_weights).sum() / valid_weights.sum()
        (numerator / denominator).backward()
        expected.backward()
        torch.testing.assert_close(numerator / denominator, expected)
        torch.testing.assert_close(states.grad, dense_states.grad)
        for actual, reference in zip(head.parameters(), dense_head.parameters()):
            torch.testing.assert_close(actual.grad, reference.grad)

    def test_fractional_denominator_is_not_clamped_to_one(self):
        model = tiny_model().eval()
        batch = text_batch([1, 2, 3])
        batch["label_weights"] = torch.tensor([[1.0, 1.0, 0.01]])
        result = model(**batch, return_logits=True)
        expected = F.cross_entropy(result["logits"][:, 1].float(), torch.tensor([3]))
        torch.testing.assert_close(result["text_loss"], expected)

    def test_accumulation_matches_global_weighted_loss(self):
        model = tiny_model().eval()
        reference = copy.deepcopy(model)
        batches = [text_batch([1, 2, 3]), text_batch([1, 2, 4, 5, 6, 9])]
        backward_window(model, batches, torch.device("cpu"), 2, 2, True)
        outputs = [reference(**batch, loss_chunk_size=64) for batch in batches]
        denominator = sum(supervision_counts(batch)[0] for batch in batches)
        (sum(o["text_loss_sum"] for o in outputs) / denominator).backward()
        for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
            if actual.grad is not None:
                torch.testing.assert_close(actual.grad, expected.grad, msg=name)

    def test_nonfinite_loss_aborts_backward(self):
        model = tiny_model()
        with torch.no_grad():
            model.backbone_llm.lm_head.weight.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            backward_window(model, [text_batch([1, 2, 3])], torch.device("cpu"), 2, 2, True)

    def test_tts_accumulation_matches_global_patch_average(self):
        model = tiny_model().eval()
        reference = copy.deepcopy(model)
        batches = []
        for count in (1, 3):
            batch = text_batch([1, 6] + [8] * count + [9, 2])
            batch["label_weights"] = torch.where(batch["labels"] == 8, 0.01, 1.0)
            batch["vae_audios"] = torch.randn(1, count * 16)
            batch["patch_encoder_output_attention_mask"] = torch.ones(1, count, dtype=torch.bool)
            batch["generation_target_start_patches"] = torch.tensor([0])
            batches.append(batch)
        backward_window(model, batches, torch.device("cpu"), 2, 1, True)
        outputs = [reference(**batch, flow_chunk_size=32) for batch in batches]
        text_denominator = sum(supervision_counts(batch)[0] for batch in batches)
        expected = sum(o["text_loss_sum"] for o in outputs) / text_denominator + sum(o["flow_loss_sum"] for o in outputs) / 4
        expected.backward()
        for (name, actual), (_, target) in zip(model.named_parameters(), reference.named_parameters()):
            if actual.grad is not None:
                torch.testing.assert_close(actual.grad, target.grad, rtol=1e-5, atol=1e-6, msg=name)

    def test_fp32_trainable_parameters_preserve_small_updates(self):
        model = tiny_model().bfloat16()
        configure_modules(model, {"backbone"}, False)
        parameter = model.backbone_llm.lm_head.weight
        with torch.no_grad():
            parameter.fill_(1.0)
        optimizer = make_optimizer(model, 2e-4, 2e-4, 0.0, torch.device("cpu"))
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertEqual(parameter.dtype, torch.float32)
        self.assertTrue((parameter < 1.0).all())
        self.assertEqual(optimizer.state[parameter]["exp_avg"].dtype, torch.float32)
        self.assertEqual(model.dit.proj.weight.dtype, torch.bfloat16)

    def test_cfg_drops_projected_bias_and_preserves_acoustic_history(self):
        model = RedDiT.__new__(RedDiT)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(vae_channels=4, backbone_hidden_size=8)
        model.patch_size, model.history_patches, model.history_length = 4, 2, 8
        model.backbone_input_proj = nn.Linear(8, 8)
        nn.init.constant_(model.backbone_input_proj.bias, 2.0)
        captured = []

        def estimate(x, t):
            captured.append(x)
            return x[:, :, :4]

        model._forward_estimator = estimate
        history = torch.randn(2, 8, 4)
        model.compute_loss(torch.randn(2, 3, 8), torch.randn(2, 4, 4), history, cfg_drop_rate=1.0)
        self.assertEqual(int(torch.count_nonzero(captured[0][:, :, 4:])), 0)
        torch.testing.assert_close(captured[0][:, :8, :4], history)

    def test_target_patch_does_not_leak_into_its_condition(self):
        model = tiny_model().eval()
        ids = torch.tensor([[1, 6, 8, 8, 8, 9, 2]])
        audio = torch.arange(48.0).reshape(1, -1)
        kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                      patch_encoder_output_attention_mask=torch.ones(1, 3, dtype=torch.bool),
                      generation_target_start_patches=torch.tensor([1]))
        with torch.no_grad():
            output = model(vae_audios=audio, **kwargs)
            condition = model.dit.conditions[0].clone()
            history = model.dit.histories[0].clone()
            changed = audio.clone()
            changed[:, 16:] += 100
            model(vae_audios=changed, **kwargs)
        self.assertEqual(output["flow_count"], 2)
        torch.testing.assert_close(condition, model.dit.conditions[0])
        torch.testing.assert_close(history, model.dit.histories[0])
        self.assertEqual(int(torch.count_nonzero(history[:4])), 0)

    def test_frozen_modules_do_not_receive_gradients(self):
        model = tiny_model()
        configure_modules(model, {"dit"}, gradient_checkpointing=False)
        ids = torch.tensor([[1, 6, 8, 8, 9, 2]])
        result = model(input_ids=ids, attention_mask=torch.ones_like(ids), vae_audios=torch.randn(1, 32),
                       patch_encoder_output_attention_mask=torch.ones(1, 2, dtype=torch.bool),
                       generation_target_start_patches=torch.tensor([0]), flow_chunk_size=1,
                       checkpoint_backbone=True)
        result["loss"].backward()
        self.assertIsNotNone(model.dit.proj.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.red_vae.parameters()))
        self.assertTrue(all(p.grad is None for p in model.backbone_llm.parameters()))
        self.assertTrue(all(p.grad is None for p in model.patch_encoder.parameters()))


class FakeTokenizer:
    tokens = {"<|AUDIO|>": 7, "<|AUDIO_NO_LATENT|>": 8, "<|sosp|>": 6, "<|eosp|>": 9}

    def __call__(self, text, **kwargs):
        chunks = re.findall(r"<\|[^>]+\|>|.", text, flags=re.DOTALL)
        return {"input_ids": torch.tensor([[self.tokens.get(c, 10 + ord(c[0])) for c in chunks]])}


class FakeEncoder:
    tokenizer = FakeTokenizer()

    def encode(self, chatml, audios):
        result = {}
        if audios[0]["audio_generation"] is not None:
            audio = audios[0]["audio_generation"]
            count = audio.numel() // 3840
            chatml = chatml.replace("<|AUDIO_NO_LATENT|>", "<|AUDIO_NO_LATENT|>" * count)
            result.update(vae_audios=audio.unsqueeze(0),
                          patch_encoder_output_attention_mask=torch.ones(1, count, dtype=torch.bool))
        ids = self.tokenizer(chatml)["input_ids"]
        result.update(input_ids=ids, attention_mask=torch.ones_like(ids))
        return result


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "audio.wav").touch()
        self.model = SimpleNamespace(config=SimpleNamespace(audio_special_token="<|AUDIO|>",
                                      audio_special_token_no_latent="<|AUDIO_NO_LATENT|>",
                                      audio_special_no_latent_id=8, sosp_idx=6))

    def test_duration_limit_rejects_instead_of_truncating(self):
        audio = torch.ones(32000)
        with patch("fireredaudio.training.data.read_audio", return_value=audio):
            with self.assertRaisesRegex(ValueError, "segment audio and text together"):
                read_checked_audio(self.root / "audio.wav", 16000, 1.0)
        self.assertEqual(audio.numel(), 32000)

    def test_reference_fields_must_be_paired(self):
        path = self.root / "train.jsonl"
        path.write_text(json.dumps({"audio": "audio.wav", "text": "hello", "prompt_audio": "audio.wav"}))
        with self.assertRaisesRegex(ValueError, "supplied together"):
            load_manifest(path, "tts")

    def test_reference_audio_is_masked_and_target_control_tokens_are_supervised(self):
        row = {"audio": "audio.wav", "text": "target", "prompt_audio": "audio.wav", "prompt_text": "reference"}
        with patch("fireredaudio.training.data.read_audio", return_value=torch.ones(7680)):
            batch = prepare_sample(row, self.root, FakeEncoder(), self.model, "tts")
        positions = (batch["input_ids"][0] == 8).nonzero(as_tuple=True)[0]
        self.assertTrue((batch["labels"][0, positions[:2]] == -100).all())
        self.assertTrue((batch["labels"][0, positions[2:]] == 8).all())
        self.assertEqual(int(batch["generation_target_start_patches"][0]), 2)
        self.assertEqual(supervision_counts(batch)[1], 2)
        torch.testing.assert_close(batch["label_weights"][0, positions[2:]], torch.full((2,), 0.01))

    def test_understanding_masks_prompt_and_supervises_reasoning_and_answer(self):
        row = {"task": "understand", "audio": ["audio.wav", "audio.wav"], "prompt": "Compare.",
               "reasoning": "Because.", "text": "Answer."}
        with patch("fireredaudio.training.data.read_audio", return_value=torch.ones(1600)):
            batch = prepare_sample(row, self.root, FakeEncoder(), self.model, "understanding")
        expected = FakeTokenizer()("Because.</think>\n\nAnswer.<|im_end|>")["input_ids"][0]
        torch.testing.assert_close(batch["labels"][batch["labels"] != -100], expected)
        self.assertEqual(int((batch["input_ids"] == 7).sum()), 2)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_resume_matches_uninterrupted_training(self):
        device = torch.device("cpu")
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = tiny_model()
        initial = copy.deepcopy(model)
        configure_modules(model, {"backbone"}, False)
        optimizer = make_optimizer(model, 0.01, 0.01, 0.0, device)
        scheduler = make_scheduler(optimizer, 4, 1, "linear")
        sampler = ShuffledSampler(3, 42)
        batches = [text_batch([1, 2, 3]), text_batch([1, 2, 4, 5]), text_batch([1, 2, 6, 7, 9])]
        metadata = {"base_fingerprint": "test", "settings": {"steps": 4}}

        def step(model, optimizer, scheduler, sampler):
            optimizer.zero_grad(set_to_none=True)
            backward_window(model, [batches[sampler.next()] for _ in range(2)], device, 2, 2, True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()

        for i in range(1, 5):
            step(model, optimizer, scheduler, sampler)
            if i == 2:
                path = save_checkpoint(model, optimizer, scheduler, sampler, i, self.root, metadata, capture_rng(device))
        saved = read_checkpoint(path)
        resumed = initial
        configure_modules(resumed, {"backbone"}, False)
        resumed_optimizer = make_optimizer(resumed, 0.01, 0.01, 0.0, device)
        resumed_scheduler = make_scheduler(resumed_optimizer, 4, 1, "linear")
        resumed_sampler = ShuffledSampler(3, 42)
        self.assertEqual(restore_checkpoint(saved, resumed, resumed_optimizer, resumed_scheduler, resumed_sampler, metadata), 2)
        restore_rng(saved["rng"], device)
        for _ in range(2):
            step(resumed, resumed_optimizer, resumed_scheduler, resumed_sampler)
        for (name, value), (_, expected) in zip(resumed.named_parameters(), model.named_parameters()):
            torch.testing.assert_close(value, expected, rtol=0, atol=0, msg=name)
        self.assertEqual(resumed_sampler.state_dict(), sampler.state_dict())
        self.assertEqual(resumed_scheduler.state_dict(), scheduler.state_dict())
        for actual, expected in zip(resumed_optimizer.state.values(), optimizer.state.values()):
            self.assertEqual(actual["exp_avg"].dtype, torch.float32)
            torch.testing.assert_close(actual["exp_avg"], expected["exp_avg"], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "same base"):
            restore_checkpoint(saved, resumed, resumed_optimizer, resumed_scheduler, resumed_sampler, {})

    def test_validation_preserves_random_state_and_modes(self):
        model = tiny_model()
        configure_modules(model, {"backbone"}, False)
        device = torch.device("cpu")
        rng = capture_rng(device)
        args = SimpleNamespace(seed=123, loss_chunk_size=2, flow_chunk_size=2)
        first = evaluate(model, [0], lambda _: text_batch([1, 2, 3]), device, {"backbone"}, args)
        second = evaluate(model, [0], lambda _: text_batch([1, 2, 3]), device, {"backbone"}, args)
        self.assertEqual(first, second)
        torch.testing.assert_close(torch.get_rng_state(), rng["torch"])
        self.assertEqual(random.getstate(), rng["python"])
        self.assertTrue(model.backbone_llm.training)
        self.assertFalse(model.red_vae.training)

    def test_export_merges_and_checks_base_identity(self):
        source = self.root / "base"
        source.mkdir()
        (source / "config.json").write_text(json.dumps({"model_type": "firered_audio", "_name_or_path": "local-only"}))
        base = {"backbone_llm.model.language_model.embed_tokens.weight": torch.ones(4, 3),
                "audio_encoder.weight": torch.arange(3.0)}
        save_file(base, source / "model.safetensors")
        delta = {"backbone_llm.model.embed_tokens.weight": torch.full((4, 3), 2.0)}
        payload = {"format_version": 1, "step": 3, "trainable_state_dict": delta,
                   "metadata": {"base_fingerprint": fingerprint(source)},
                   "optimizer": {}, "scheduler": {}, "sampler": {}, "rng": {}}
        path = self.root / "training.pt"
        atomic_save(payload, path)
        output = export_checkpoint(source, path, self.root / "export")
        result = load_file(output / "model-00001-of-00001.safetensors")
        torch.testing.assert_close(result["backbone_llm.model.embed_tokens.weight"], delta["backbone_llm.model.embed_tokens.weight"])
        torch.testing.assert_close(result["audio_encoder.weight"], base["audio_encoder.weight"])
        self.assertNotIn("_name_or_path", json.loads((output / "config.json").read_text()))
        with self.assertRaises(FileExistsError):
            export_checkpoint(source, path, source)
        base["audio_encoder.weight"] += 1
        save_file(base, source / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "Base checkpoint"):
            export_checkpoint(source, path, self.root / "wrong-base")
        self.assertFalse((self.root / "wrong-base").exists())

    def test_legacy_checkpoint_is_rejected(self):
        path = self.root / "legacy.pt"
        atomic_save({"step": 1, "trainable_state_dict": {}}, path)
        with self.assertRaisesRegex(ValueError, "legacy"):
            read_checkpoint(path)

    def test_exported_model_loads_and_preserves_audio_logits(self):
        from fireredaudio.configuration_fireredaudio import FireRedAudioConfig
        config = FireRedAudioConfig(
            audio_special_token_id=7, audio_special_no_latent_id=8, sosp_idx=6, eosp_idx=9,
            backbone_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                 head_dim=8, layer_types=["full_attention"], eos_token_id=2,
                                 rope_parameters={"rope_type": "default", "rope_theta": 10000,
                                                  "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]}),
            audio_encoder_config=dict(num_mel_bins=4, d_model=16, encoder_layers=1,
                                      encoder_attention_heads=2, encoder_ffn_dim=32,
                                      max_source_positions=32, n_window=16, output_dim=16),
            red_vae_config=dict(out_dim=4, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                downsample_num_hidden_layers=1),
            patch_encoder_config=dict(vae_dim=4, out_dim=16, hidden_size=16, depth=1, num_heads=2),
            dit_config=dict(vae_channels=4, backbone_hidden_size=16, hidden_size=16, depth=1, num_heads=2),
        )
        config.backbone_config._attn_implementation = "sdpa"
        config.audio_encoder_config._attn_implementation = "sdpa"
        model = FireRedAudioForCausalLM(config).float().eval()
        source = self.root / "tiny-base"
        model.save_pretrained(source)
        with torch.no_grad():
            model.backbone_llm.lm_head.weight[3].add_(0.25)
        delta = {"backbone_llm.lm_head.weight": model.backbone_llm.lm_head.weight.detach().clone()}
        path = self.root / "tiny-training.pt"
        atomic_save({"format_version": 1, "step": 1, "trainable_state_dict": delta,
                     "metadata": {"base_fingerprint": fingerprint(source)},
                     "optimizer": {}, "scheduler": {}, "sampler": {}, "rng": {}}, path)
        output = export_checkpoint(source, path, self.root / "tiny-export")
        restored = FireRedAudioForCausalLM.from_pretrained(output, dtype=torch.float32).eval()
        inputs = dict(input_ids=torch.tensor([[1, 7, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long),
                      audio_features=torch.randn(1, 4, 8), audio_feature_attention_mask=torch.ones(1, 8, dtype=torch.long),
                      return_logits=True)
        with torch.no_grad():
            torch.testing.assert_close(model(**inputs)["logits"], restored(**inputs)["logits"])


if __name__ == "__main__":
    unittest.main()
