"""
BioReason-Pro reward functions for GRPO RL training.

Primary reward: weighted F-score (F_w) on predicted GO terms vs ground truth,
matching the BioReason-Pro paper's reward definition.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# GO term regex: matches GO:XXXXXXX
_GO_TERM_RE = re.compile(r"GO:\d{7}")

# Namespace prefixes used in GO term sets
_NS_PREFIXES = {"biological_process", "molecular_function", "cellular_component"}


def extract_go_terms(text: str) -> set[str]:
    """Extract all GO:XXXXXXX terms from a generated reasoning trace."""
    return set(_GO_TERM_RE.findall(text))


def weighted_f_score(
    predicted: set[str],
    ground_truth: set[str],
    beta: float = 1.0,
) -> float:
    """
    Compute F_beta score between predicted and ground truth GO term sets.

    F_1 (beta=1) is the default, matching BioReason-Pro evaluation.
    Returns 0.0 if both sets are empty (model predicts nothing, GT is nothing).
    """
    if not ground_truth and not predicted:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    tp = len(predicted & ground_truth)
    precision = tp / len(predicted)
    recall = tp / len(ground_truth)
    if precision + recall == 0:
        return 0.0
    beta2 = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def bioreason_reward_fn(
    completions: list[str],
    answers: list[str],
    beta: float = 1.0,
    format_penalty: float = 0.1,
    return_diagnostics: bool = False,
):
    """
    Compute per-completion rewards for BioReason GRPO.

    Args:
        completions: Generated reasoning traces (one per rollout)
        answers: Ground truth GO term strings — comma-separated GO:XXXXXXX terms
                 (one per prompt, repeated G times for G rollouts per prompt)
        beta: F-score beta (default 1.0 = F1)
        format_penalty: Penalty subtracted when completion contains no GO terms
        return_diagnostics: When True, also return per-completion counts useful
            for telemetry (predicted-term count, GT count, true-positive count).

    Returns:
        rewards: [N] float tensor of F_beta scores in [0, 1]
        successes: [N] bool tensor (reward > 0.5)
        diagnostics (optional): dict with int32 tensors
            - pred_count: [N] number of GO terms predicted
            - gt_count:   [N] number of GO terms in ground truth
            - tp_count:   [N] number of true positives (pred ∩ gt)
            - has_pred:   [N] bool, predicted at least one GO term
    """
    rewards = []
    pred_counts: list[int] = []
    gt_counts: list[int] = []
    tp_counts: list[int] = []
    for completion, answer in zip(completions, answers):
        predicted = extract_go_terms(completion)
        gt = extract_go_terms(answer)

        score = weighted_f_score(predicted, gt, beta=beta)

        # Penalize outputs with no GO terms at all (format failure)
        if not predicted:
            score = max(0.0, score - format_penalty)

        rewards.append(score)
        if return_diagnostics:
            pred_counts.append(len(predicted))
            gt_counts.append(len(gt))
            tp_counts.append(len(predicted & gt))

    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    successes_t = rewards_t > 0.5
    if not return_diagnostics:
        return rewards_t, successes_t

    diagnostics = {
        "pred_count": torch.tensor(pred_counts, dtype=torch.int32),
        "gt_count": torch.tensor(gt_counts, dtype=torch.int32),
        "tp_count": torch.tensor(tp_counts, dtype=torch.int32),
        "has_pred": torch.tensor(pred_counts, dtype=torch.int32) > 0,
    }
    return rewards_t, successes_t, diagnostics


def batch_level_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute advantages using batch-level normalization (BioReason-Pro's fix for
    low-variance rewards from highly similar rollouts).

    Standard GRPO normalizes per-prompt-group; this normalizes over the full batch,
    giving a non-zero learning signal even when all rollouts for a single prompt
    receive identical rewards.

    Args:
        rewards: [B * G] flat rewards tensor
        group_size: G rollouts per prompt
        eps: numerical stability

    Returns:
        advantages: [B * G] normalized advantages
    """
    mean = rewards.mean()
    std = rewards.std() + eps
    return (rewards - mean) / std
