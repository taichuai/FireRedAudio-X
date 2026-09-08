"""Memory-bounded losses with explicit normalization statistics."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def text_weights(labels, label_weights=None):
    weights = torch.ones_like(labels, dtype=torch.float32) if label_weights is None else label_weights.float()
    if weights.shape != labels.shape or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Label weights must be finite, nonnegative and match labels")
    return weights[:, 1:] * (labels[:, 1:] != -100)


def weighted_text_loss(hidden_states, lm_head, labels, label_weights=None, chunk_size=64):
    if chunk_size <= 0:
        raise ValueError("loss chunk size must be positive")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("Labels must match hidden-state sequence dimensions")
    weights = text_weights(labels, label_weights)
    valid = weights > 0
    states = hidden_states[:, :-1][valid]
    targets = labels[:, 1:][valid]
    selected_weights = weights[valid]
    denominator = selected_weights.sum()
    numerator = states.sum() * 0.0

    def chunk_loss(states, targets, weights):
        logits = lm_head(states).float()
        return (F.cross_entropy(logits, targets, reduction="none") * weights).sum()

    for start in range(0, targets.numel(), chunk_size):
        args = (states[start:start + chunk_size], targets[start:start + chunk_size],
                selected_weights[start:start + chunk_size])
        if torch.is_grad_enabled():
            # Recompute logits in backward instead of retaining a full vocabulary
            # activation for every supervised token in the conversation.
            numerator = numerator + checkpoint(chunk_loss, *args, use_reentrant=False)
        else:
            numerator = numerator + chunk_loss(*args)
    return numerator, denominator


def supervision_counts(batch):
    text_count = float(text_weights(batch["labels"], batch.get("label_weights")).sum())
    flow_count = 0
    if "generation_target_start_patches" in batch:
        lengths = batch["patch_encoder_output_attention_mask"].sum(-1).tolist()
        starts = batch["generation_target_start_patches"].tolist()
        if len(lengths) != len(starts):
            raise ValueError("One target start is required per generation segment")
        for length, start in zip(lengths, starts):
            if start < -1 or start >= length:
                raise ValueError("Target start is outside the audio segment")
            if start >= 0:
                flow_count += length - start
    return text_count, flow_count


def normalized_objective(outputs, text_count, flow_count):
    parts = []
    if text_count > 0 and outputs["text_loss_sum"] is not None:
        parts.append(outputs["text_loss_sum"] / text_count)
    if flow_count > 0 and outputs["flow_loss_sum"] is not None:
        parts.append(outputs["flow_loss_sum"] / flow_count)
    if not parts:
        raise ValueError("No supervised targets in the accumulation window")
    return sum(parts)
