"""Explicit causal-LM quality measurements; no universal accuracy assertion."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
from torch.nn import functional as F


def _encoded_texts(
    tokenizer, texts: Sequence[str], max_length: int, device
) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
    encoded = []
    for text_index, text in enumerate(texts):
        value = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        if value["input_ids"].shape[1] >= 2:
            encoded.append(
                (text_index, {key: tensor.to(device) for key, tensor in value.items()})
            )
    if not encoded:
        raise ValueError("Evaluation needs at least one text with two tokenizer tokens")
    return encoded


@torch.inference_mode()
def compare_causal_lms(
    reference,
    candidate,
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int = 512,
    device="cpu",
) -> Dict[str, object]:
    """Compare next-token NLL, perplexity, top-1 agreement, and KL on given texts.

    This is evidence only for the supplied texts and decoding setup. It is not a
    replacement for a held-out benchmark or human chat evaluation.
    """

    reference.eval()
    candidate.eval()
    batches = _encoded_texts(tokenizer, texts, max_length, device)
    totals = {
        "reference_nll": 0.0,
        "candidate_nll": 0.0,
        "kl_reference_candidate": 0.0,
        "tokens": 0,
        "top1_agreements": 0,
    }
    per_text = []
    for text_index, batch in batches:
        labels = batch["input_ids"][:, 1:]
        valid = batch.get("attention_mask", torch.ones_like(batch["input_ids"]))[:, 1:].bool()
        if not valid.any():
            continue
        reference_logits = reference(**batch).logits[:, :-1].float()
        candidate_logits = candidate(**batch).logits[:, :-1].float()
        reference_log_probs = F.log_softmax(reference_logits, dim=-1)
        candidate_log_probs = F.log_softmax(candidate_logits, dim=-1)
        target = labels.unsqueeze(-1)
        reference_token_nll = -reference_log_probs.gather(-1, target).squeeze(-1)
        candidate_token_nll = -candidate_log_probs.gather(-1, target).squeeze(-1)
        count = int(valid.sum().item())
        reference_sum = reference_token_nll[valid].sum().item()
        candidate_sum = candidate_token_nll[valid].sum().item()
        kl = (reference_log_probs.exp() * (reference_log_probs - candidate_log_probs)).sum(-1)
        kl_sum = kl[valid].sum().item()
        agreements = int(
            (reference_logits.argmax(-1)[valid] == candidate_logits.argmax(-1)[valid])
            .sum()
            .item()
        )
        totals["reference_nll"] += reference_sum
        totals["candidate_nll"] += candidate_sum
        totals["kl_reference_candidate"] += kl_sum
        totals["top1_agreements"] += agreements
        totals["tokens"] += count
        reference_text_mean = reference_sum / count
        candidate_text_mean = candidate_sum / count
        per_text.append({
            "text_index": text_index,
            "text": texts[text_index],
            "supervised_tokens": count,
            "reference_mean_nll": reference_text_mean,
            "candidate_mean_nll": candidate_text_mean,
            "mean_nll_delta": candidate_text_mean - reference_text_mean,
            "reference_perplexity": (
                math.exp(reference_text_mean) if reference_text_mean < 80 else math.inf
            ),
            "candidate_perplexity": (
                math.exp(candidate_text_mean) if candidate_text_mean < 80 else math.inf
            ),
            "mean_kl_reference_candidate": kl_sum / count,
            "top1_agreement": agreements / count,
        })
    if not totals["tokens"]:
        raise ValueError("No supervised next-token positions were available")
    tokens = totals["tokens"]
    reference_mean = totals["reference_nll"] / tokens
    candidate_mean = totals["candidate_nll"] / tokens
    return {
        "schema_version": 1,
        "texts": len(batches),
        "supervised_tokens": tokens,
        "reference_mean_nll": reference_mean,
        "candidate_mean_nll": candidate_mean,
        "mean_nll_delta": candidate_mean - reference_mean,
        "reference_perplexity": math.exp(reference_mean) if reference_mean < 80 else math.inf,
        "candidate_perplexity": math.exp(candidate_mean) if candidate_mean < 80 else math.inf,
        "mean_kl_reference_candidate": totals["kl_reference_candidate"] / tokens,
        "top1_agreement": totals["top1_agreements"] / tokens,
        "per_text": per_text,
        "scope": (
            "Metrics only for supplied texts; not a universal model-accuracy, "
            "language-quality, or downstream-task guarantee"
        ),
    }
