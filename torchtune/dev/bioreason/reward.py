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


# ── GO ontology hierarchy propagation (matches cafa_eval / the real CAFA metric) ──
# The eval metric (cafaeval) credits a predicted SPECIFIC term for all its is_a
# ancestors before scoring. Our reward originally did EXACT term-set matching, which
# leaves the signal flat (~50% zeros, mean F1 0.04 on real rollouts) even against the
# correct target — RL can't learn from it. Propagating both predicted and GT terms to
# their ancestor-closure makes the reward dense (mean 0.23, 13% zeros) AND aligns it
# with the eval metric, so reward gains translate to F_max gains. The DAG is loaded
# once and cached (go-basic.obo ships in the repo + each ckpt dir).
_GO_DAG = None
_GO_DAG_PATH = None
_GO_ANCESTOR_CACHE: dict[str, frozenset[str]] = {}


def _resolve_obo_path(obo_path: Optional[str]) -> Optional[str]:
    """Resolve to an existing go-basic.obo FILE.

    obo_path may be the file itself OR a directory (e.g. base_model_path — each ckpt
    ships go-basic.obo). A directory passed straight to GODag fails ("COULD NOT READ"),
    so when given a dir we look for go-basic.obo inside it.
    """
    import os

    def _as_obo_file(p):
        if not p:
            return None
        if os.path.isfile(p):
            return p
        if os.path.isdir(p):
            cand = os.path.join(p, "go-basic.obo")
            return cand if os.path.isfile(cand) else None
        return None

    hit = _as_obo_file(obo_path)
    if hit:
        return hit
    # Fallbacks: the ontology ships alongside the BioReason source + via env override.
    src = os.environ.get("BIOREASON_SRC", "/lus/flare/projects/ModCon/ngetty/BioReason-Pro")
    for cand in (
        os.environ.get("BIOREASON_GO_OBO", ""),
        os.path.join(src, "bioreason2", "dataset", "go-basic.obo"),
    ):
        if cand and os.path.isfile(cand):
            return cand
    return None


def load_go_dag(obo_path: Optional[str] = None):
    """Load + cache the GO DAG (goatools). Returns None if unavailable (callers then
    fall back to exact matching)."""
    global _GO_DAG, _GO_DAG_PATH
    resolved = _resolve_obo_path(obo_path)
    if resolved is None:
        logger.warning("GO ontology .obo not found; reward falls back to exact match.")
        return None
    if _GO_DAG is not None and _GO_DAG_PATH == resolved:
        return _GO_DAG
    try:
        from goatools.obo_parser import GODag
        _GO_DAG = GODag(resolved, prt=None)
        _GO_DAG_PATH = resolved
        _GO_ANCESTOR_CACHE.clear()
        logger.info(f"Loaded GO DAG for reward propagation from {resolved}")
    except Exception as e:  # pragma: no cover - env-dependent
        logger.warning(f"Failed to load GO DAG ({e}); reward falls back to exact match.")
        _GO_DAG = None
    return _GO_DAG


def propagate_go_terms(terms: set[str], dag=None) -> set[str]:
    """Expand a GO term set to include all is_a ancestors (ontology closure).

    Mirrors cafaeval's propagation. Self-included. Unknown terms pass through
    unchanged. Per-term ancestor sets are memoized. If no DAG, returns terms as-is.
    """
    if dag is None:
        dag = _GO_DAG
    if dag is None or not terms:
        return set(terms)
    out: set[str] = set()
    for t in terms:
        cached = _GO_ANCESTOR_CACHE.get(t)
        if cached is None:
            if t in dag:
                cached = frozenset({t} | dag[t].get_all_parents())
            else:
                cached = frozenset({t})
            _GO_ANCESTOR_CACHE[t] = cached
        out |= cached
    return out


def weighted_f_score(
    predicted: set[str],
    ground_truth: set[str],
    beta: float = 1.0,
    propagate: bool = False,
    dag=None,
) -> float:
    """
    Compute F_beta score between predicted and ground truth GO term sets.

    F_1 (beta=1) is the default, matching BioReason-Pro evaluation.
    Returns 0.0 if both sets are empty (model predicts nothing, GT is nothing).

    When ``propagate=True``, both sets are expanded to their GO is_a ancestor closure
    before scoring (matches the cafaeval metric; turns a flat exact-match signal into a
    dense, learnable one). Falls back to exact matching if no DAG is available.
    """
    if propagate:
        predicted = propagate_go_terms(predicted, dag)
        ground_truth = propagate_go_terms(ground_truth, dag)
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
    propagate_hierarchy: bool = False,
    obo_path: Optional[str] = None,
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
    # Load the GO DAG once per call when hierarchy propagation is requested (cached
    # across calls). If unavailable, weighted_f_score silently falls back to exact.
    dag = load_go_dag(obo_path) if propagate_hierarchy else None

    rewards = []
    pred_counts: list[int] = []
    gt_counts: list[int] = []
    tp_counts: list[int] = []
    for completion, answer in zip(completions, answers):
        predicted = extract_go_terms(completion)
        gt = extract_go_terms(answer)

        score = weighted_f_score(
            predicted, gt, beta=beta,
            propagate=propagate_hierarchy and dag is not None, dag=dag,
        )

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


# Re-export: the canonical location is torchtune.dev.rl.rewards. Kept here for
# back-compat so the BioReason recipe import path (and any external callers)
# continue to work without churn.
from torchtune.dev.rl.rewards import batch_level_advantages  # noqa: E402,F401
