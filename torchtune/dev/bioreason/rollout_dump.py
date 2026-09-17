# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Persist BioReason GRPO rollout groups to JSONL for offline analysis.

WHY THIS EXISTS (2026-09-15): the GRPO recipe decodes its ``B*G`` completions to
compute the reward and then **discards them**. Only a single 200-character
truncated ``SAMPLE_RESPONSE`` per step survives in the log. That made a decisive
test impossible: F_max is a *ranked* metric, but the model emits no confidences,
so every predicted GO term scores 1.0 and the threshold sweep is inert. Ranking
terms by how many of the G samples in a group emitted them recovered +0.0349
F_max offline — but only against a *proxy* built from four repeated evaluations
of one checkpoint, because no real rollout group had ever been written to disk.

With this dump the same analysis runs on genuine temperature-sampled rollouts.
It is also the only way to do *any* post-hoc analysis of what the policy actually
generated during a run; today that data is unrecoverable once the step ends.

Deliberately cheap and off by default: gated on ``TORCHTUNE_DUMP_ROLLOUTS``,
rank-0 only, append-only JSONL, and every failure is swallowed with a warning.
A diagnostic dump must never be able to kill a training run.
"""

import json
import logging
import os
from typing import Optional, Sequence

log = logging.getLogger(__name__)

__all__ = ["rollout_dump_path", "dump_rollout_groups"]

_ENV_FLAG = "TORCHTUNE_DUMP_ROLLOUTS"
_ENV_PATH = "TORCHTUNE_ROLLOUT_DUMP_PATH"


def rollout_dump_path(output_dir: Optional[str] = None) -> Optional[str]:
    """Resolve the dump target, or None when dumping is disabled.

    Enabled by ``TORCHTUNE_DUMP_ROLLOUTS=1``. The path is
    ``TORCHTUNE_ROLLOUT_DUMP_PATH`` if set, else ``<output_dir>/rollouts.jsonl``.
    Returns None (rather than raising) when enabled without any usable path, so a
    misconfiguration degrades to "no dump" instead of crashing the run.
    """
    if os.environ.get(_ENV_FLAG, "0") != "1":
        return None
    explicit = os.environ.get(_ENV_PATH, "").strip()
    if explicit:
        return explicit
    if output_dir:
        return os.path.join(output_dir, "rollouts.jsonl")
    log.warning(
        "%s=1 but neither %s nor output_dir is set; rollout dump disabled",
        _ENV_FLAG,
        _ENV_PATH,
    )
    return None


def dump_rollout_groups(
    path: Optional[str],
    *,
    step: int,
    batch_size: int,
    grpo_size: int,
    decoded: Sequence[str],
    answers: Sequence[str],
    rewards: Optional[Sequence[float]] = None,
    successes: Optional[Sequence[float]] = None,
    proteins: Optional[Sequence[str]] = None,
    go_aspects: Optional[Sequence[str]] = None,
    max_protein_chars: int = 64,
) -> bool:
    """Append one JSONL record per prompt group. Returns True if anything was written.

    ``decoded``/``rewards``/``successes`` are flat, length ``batch_size*grpo_size``,
    laid out group-major (``b*grpo_size + g``) — matching the recipe's own
    ``for _b: for _g:`` decode loop. ``answers``/``proteins``/``go_aspects`` are
    per-prompt, length ``batch_size``.

    One record per *group*, not per sample: group frequency is the quantity the
    downstream ranking analysis needs, and keeping a group intact means the
    consumer cannot accidentally mix samples across prompts.

    The protein sequence is truncated to ``max_protein_chars`` — it is written as a
    join key for the eval set, not as data, and full sequences would dominate the
    file.

    Never raises. A diagnostic must not be able to fail a training step.
    """
    if not path:
        return False
    try:
        expected = batch_size * grpo_size
        if len(decoded) != expected:
            log.warning(
                "rollout dump skipped: decoded has %d entries, expected B*G=%d",
                len(decoded),
                expected,
            )
            return False

        def _flat(seq, b, g):
            if seq is None:
                return None
            i = b * grpo_size + g
            if i >= len(seq):
                return None
            v = seq[i]
            return float(v) if v is not None else None

        def _per_prompt(seq, b):
            if seq is None or b >= len(seq):
                return None
            return seq[b]

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for b in range(batch_size):
                prot = _per_prompt(proteins, b)
                rec = {
                    "step": int(step),
                    "prompt_idx": int(b),
                    "answer": _per_prompt(answers, b),
                    "go_aspect": _per_prompt(go_aspects, b),
                    "protein_prefix": (prot[:max_protein_chars] if prot else None),
                    "protein_len": (len(prot) if prot else None),
                    "samples": [
                        {
                            "g": g,
                            "text": decoded[b * grpo_size + g],
                            "reward": _flat(rewards, b, g),
                            "success": _flat(successes, b, g),
                        }
                        for g in range(grpo_size)
                    ],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True
    except Exception as e:  # noqa: BLE001 - diagnostics must never kill a run
        log.warning("rollout dump failed (continuing): %s", e)
        return False
