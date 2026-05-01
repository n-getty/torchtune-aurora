# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from torchtune.modules.transforms.tokenizers import (
    HuggingFaceModelTokenizer,
    ModelTokenizer,
)


@dataclass
class RewardOutput:
    """
    This class is used to store the reward and other statistics for a given reward function.

    Attributes:
        reward_base_name (str): the base name of the reward function, e.g. "math_correctness" or "formatting"
        total_reward (torch.Tensor): the total reward for the given reward function, shape ``[b]``
        successes (torch.Tensor): the number of successes for the given reward function, shape ``[b]``
        rewards (Optional[dict[str, torch.Tensor]]): an optional dictionary of sub-rewards for the given reward function,
           which are only used for logging purposes. e.g:
           ``{"soft_format_reward": torch.Tensor, "strict_format_reward": torch.Tensor}``
    """

    reward_base_name: str
    total_reward: torch.Tensor
    successes: torch.Tensor
    rewards: Optional[dict[str, torch.Tensor]] = field(default_factory=dict)

    def log(self, prefix: str = "") -> dict[str, float]:
        """
        Logs the reward and other statistics for the given reward function.

        Args:
            prefix (str): an optional prefix to add to the log keys

        Returns:
            A dictionary of the logged statistics.
        Example:
            >>> reward_output = RewardOutput(
                reward_base_name="math_correctness",
                total_reward=torch.tensor([1.0, 2.0, 3.0]),
                successes=torch.tensor([1.0, 0.0, 1.0]),
                rewards={"soft_format_reward": torch.tensor([1.0, 0.0, 1.0]), "strict_format_reward": torch.tensor([1.0, 0.0, 1.0])}
            )
            >>> reward_output.log(prefix="train")
            {
                "train/math_correctness": 2.0,
                "train/math_correctness/successes": 0.6666666666666666,
                "train/math_correctness/soft_format_reward": 1.0,
                "train/math_correctness/strict_format_reward": 1.0
            }
        """
        log_dict = {}
        prefix = (
            f"{prefix}/{self.reward_base_name}" if prefix else self.reward_base_name
        )

        for reward_name, reward in self.rewards.items():
            log_dict[f"{prefix}/{reward_name}"] = reward.mean().item()

        log_dict[f"{prefix}"] = self.total_reward.mean().item()
        log_dict[f"{prefix}/successes"] = self.successes.mean().item()
        return log_dict


class Reward(ABC):
    """
    This is an abstract base class for rewards which are used in GRPO.
    """

    @abstractmethod
    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: list[str],
        answers: list[str],
    ) -> RewardOutput:
        """
        This method is called to compute the reward for a given completion and answer.

        Args:
            completion_ids (torch.Tensor): the token ids of the completion, shape ``[b, seq_len]``
            completions (list[str]): the completions, shape ``[b, seq_len]``
            answers (list[str]): the answers, shape ``[b, seq_len]``

        Returns:
            A ``RewardOutput`` object containing the total reward to be used in advantage estimation,
                alongside additional metadata useful for logging.
        """
        pass


class FormattedMathCorrectnessReward(Reward):
    """
    This reward encourages the model to correctly answer a math problem, and requires
    the model to repond in an XML-style format to extract answers.

    Args:
        answer_tag (str): the tag for the answer section. The answer will be extracted from <{answer_tag}>{answer}</{answer_tag}>
        positive_reward (float): the reward provided for correctly formatted completions
        negative_reward (float): the reward provided for incorrectly formatted completions
    """

    def __init__(
        self, answer_tag: str, positive_reward: float, negative_reward: float = 0.0
    ):
        self.answer_tag = answer_tag
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: list[str],
        answers: list[str],
    ) -> RewardOutput:
        rewards = []
        import math_verify

        for completion, answer in zip(completions, answers):
            gold = math_verify.parse(answer)
            attempt = math_verify.parse(completion)
            if math_verify.verify(gold, attempt):
                reward = self.positive_reward
            elif answer in completion:
                reward = self.positive_reward / 2
            else:
                reward = self.negative_reward
            rewards.append(reward)

        rewards = torch.tensor(rewards)
        return RewardOutput(
            reward_base_name="math_correctness",
            total_reward=rewards,
            successes=(rewards == self.positive_reward).float(),
        )


class ThinkingAnswerFormattingReward(Reward):
    """
    This reward encourages the model to respond in a reasoning-style format. It applies
    both a soft and strict formatting reward.

    The "soft" formatting reward rewards the model for using the tags, even if the tags do not
    have newlines.

    The "strict" formatting reward rewards the model for using the tags, and having newlines.

    Taken from https://github.com/huggingface/open-r1/blob/06bdd503341f5375bf93c3720df13f8588d47712/src/open_r1/rewards.py

    Args:
        think_tag (str): the tag for the think section. The tag will be XML-formatted as <{think_tag}>...</{think_tag}>
        answer_tag (str): the tag for the answer section. The tag will be XML-formatted as <{answer_tag}>...</{answer_tag}>
        positive_reward (float): the reward provided for correctly formatted completions
        negative_reward (float): the reward provided for incorrectly formatted completions
    """

    def __init__(
        self,
        think_tag: str,
        answer_tag: str,
        positive_reward: float,
        negative_reward: float = 0.0,
    ):
        self.think_tag = think_tag
        self.answer_tag = answer_tag
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: list[str],
        answers: list[str],
    ) -> RewardOutput:
        # soft format reward pattern
        think_pattern = rf"<{self.think_tag}>.*?</{self.think_tag}>"
        answer_pattern = rf"<{self.answer_tag}>.*?</{self.answer_tag}>"

        # strict format reward pattern
        strict_pattern = rf"^<{self.think_tag}>\n.*?\n</{self.think_tag}>\n<{self.answer_tag}>\n.*?\n</{self.answer_tag}>\n$"
        soft_format_rewards = []
        strict_format_rewards = []
        for completion in completions:
            strict_format_rewards.append(
                self.positive_reward
                if re.match(strict_pattern, completion, re.DOTALL | re.MULTILINE)
                else self.negative_reward
            )

            think_matches = re.findall(think_pattern, completion, re.DOTALL)
            answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
            if len(think_matches) == 1 and len(answer_matches) == 1:
                think_index = completion.find(think_matches[0])
                answer_index = completion.find(answer_matches[0])
                if think_index < answer_index:
                    soft_format_rewards.append(self.positive_reward)
                    continue
            soft_format_rewards.append(self.negative_reward)

        soft_format_rewards = torch.tensor(soft_format_rewards)
        strict_format_rewards = torch.tensor(strict_format_rewards)
        rewards = soft_format_rewards + strict_format_rewards
        successes = (rewards >= self.positive_reward).float()
        return RewardOutput(
            reward_base_name="formatting",
            total_reward=rewards,
            rewards={
                "soft_format_reward": soft_format_rewards,
                "strict_format_reward": strict_format_rewards,
            },
            successes=successes,
        )


def at_least_one_space_between_think_tags(
    cot: str, answer: str, potential_answer: str
) -> tuple[float, float]:
    """Did the model at least try to think?"""
    if len(cot) > 0:
        return 1.0, 1.0  # (reward, success)
    else:
        return 0.0, 0.0


def math_response_correct(
    cot: str, answer: str, potential_answer: str
) -> tuple[float, float]:
    """Did it get the right answer?"""
    import math_verify

    if potential_answer is None:
        return 0.0, 0.0  # (reward, success)
    gold = math_verify.parse(answer)
    attempt = math_verify.parse(potential_answer)

    if math_verify.verify(gold, attempt):
        return 100.0, 1.0
    if answer in potential_answer:
        return 50.0, 0.0
    if len(potential_answer) > 0:
        return 1.0, 0.0
    return 0.0, 0.0


class TaggedMathCorrectnessReward(Reward):
    """Tag-extracting math correctness reward — config-honoring wrapper around
    the hardcoded ``math_response_correct``. Use this when the model is
    expected to emit ``<answer>...</answer>`` tags (the standard recipe
    prompt format). Differs from ``FormattedMathCorrectnessReward`` in that
    it parses the tag first, then runs math_verify on the tag contents only.
    """

    def __init__(
        self,
        answer_tag: str = "answer",
        positive_reward: float = 100.0,
        partial_reward: float = 50.0,
        format_only_reward: float = 1.0,
        negative_reward: float = 0.0,
    ):
        self.answer_tag = answer_tag
        self.positive_reward = positive_reward
        self.partial_reward = partial_reward
        self.format_only_reward = format_only_reward
        self.negative_reward = negative_reward

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: list[str],
        answers: list[str],
    ) -> RewardOutput:
        import math_verify

        rewards = []
        successes = []
        for completion, answer in zip(completions, answers):
            _, potential_answer = extract_tags(f"<think>{completion}")
            if not potential_answer:
                rewards.append(self.negative_reward)
                successes.append(0.0)
                continue
            gold = math_verify.parse(answer)
            attempt = math_verify.parse(potential_answer)
            if math_verify.verify(gold, attempt):
                rewards.append(self.positive_reward)
                successes.append(1.0)
            elif answer in potential_answer:
                rewards.append(self.partial_reward)
                successes.append(0.0)
            else:
                rewards.append(self.format_only_reward)
                successes.append(0.0)
        return RewardOutput(
            reward_base_name="math_correctness",
            total_reward=torch.tensor(rewards),
            successes=torch.tensor(successes),
        )


class ThinkingTagPresenceReward(Reward):
    """Lightweight format reward — config-honoring wrapper around the
    hardcoded ``at_least_one_space_between_think_tags``. Returns
    ``positive_reward`` when the completion has a non-empty ``<think>``
    section; ``negative_reward`` otherwise. Less strict than
    ``ThinkingAnswerFormattingReward``, which also requires the answer tag
    and format ordering.
    """

    def __init__(
        self,
        think_tag: str = "think",
        positive_reward: float = 1.0,
        negative_reward: float = 0.0,
    ):
        self.think_tag = think_tag
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: list[str],
        answers: list[str],
    ) -> RewardOutput:
        rewards = []
        for completion in completions:
            cot, _ = extract_tags(f"<{self.think_tag}>{completion}")
            rewards.append(self.positive_reward if cot else self.negative_reward)
        rewards_t = torch.tensor(rewards)
        return RewardOutput(
            reward_base_name="format",
            total_reward=rewards_t,
            successes=(rewards_t == self.positive_reward).float(),
        )


def extract_tags(text: str) -> tuple[str, str]:
    """
    Parse XML-like tags from text. Returns a dictionary with keys 'think' and 'answer'.
    The values are lists of strings, with each string being the content of a tag.
    """
    think_pattern = r"<think>(.*?)</think>"
    answer_pattern = r"<answer>(.*?)</answer>"
    think_match = re.search(think_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    cot = think_match.group(1).strip() if think_match else ""
    potential_answer = answer_match.group(1).strip() if answer_match else ""
    return cot, potential_answer


_GENE_STOP_WORDS = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
    "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM",
    "HIS", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE", "TWO",
    "WAY", "WHO", "BOY", "DID", "HIT", "LET", "MEN", "PUT", "SAY",
    "SHE", "TOO", "USE", "LIST", "CORE", "GENE", "GENES", "INVOLVED",
    "FOLLOWING", "NARRATIVE", "INCLUDE", "INCLUDING", "SUCH", "AS",
    "IN", "OF", "TO", "IS", "IT", "BE", "AT", "BY", "AN", "OR",
    "IF", "NO", "UP", "SO", "DO", "GO", "ME", "MY", "ON", "WE",
    "DNA", "RNA", "ATP", "ADP", "ECM", "ROS", "TGF", "EGF",
}
_GENE_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")
_GENES_TAG_RE = re.compile(r"<genes>(.*?)</genes>", re.DOTALL | re.IGNORECASE)


def _extract_genes(text: str) -> set:
    """Extract gene symbols from text.

    If the text contains a <genes>...</genes> tag (from the reasoning prompt
    format), parse only the tag contents. Falls back to scanning the full
    text during early training when the model hasn't learned the format yet.
    """
    m = _GENES_TAG_RE.search(text)
    source = m.group(1) if m else text
    candidates = _GENE_PATTERN.findall(source)
    return {g for g in candidates if g not in _GENE_STOP_WORDS}


def _gene_f1(predicted: set, reference: set) -> float:
    if not predicted and not reference:
        return 1.0
    if not predicted or not reference:
        return 0.0
    tp = len(predicted & reference)
    precision = tp / len(predicted)
    recall = tp / len(reference)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class GeneRecallReward(Reward):
    """
    Reward for gene recall tasks: F1 score between predicted and reference HGNC gene sets.

    The model's completion is scanned for uppercase gene-symbol tokens (e.g. TP53, BRCA1).
    The ``answer`` field contains the reference comma-separated gene list.

    Args:
        reward_metric (str): "f1" (default), "recall", or "jaccard"
    """

    def __init__(self, reward_metric: str = "f1"):
        self.reward_metric = reward_metric

    def _score(self, predicted: set, reference: set) -> float:
        if self.reward_metric == "recall":
            if not reference:
                return 1.0
            return len(predicted & reference) / len(reference)
        if self.reward_metric == "jaccard":
            union = predicted | reference
            if not union:
                return 1.0
            return len(predicted & reference) / len(union)
        return _gene_f1(predicted, reference)

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: list[str],
        answers: list[str],
    ) -> RewardOutput:
        rewards = []
        for completion, answer in zip(completions, answers):
            predicted = _extract_genes(completion)
            reference = _extract_genes(answer)
            rewards.append(self._score(predicted, reference))
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        return RewardOutput(
            reward_base_name="gene_recall",
            total_reward=rewards_t,
            successes=(rewards_t >= 0.5).float(),
            rewards={"f1": rewards_t},
        )


def gene_recall_batched_rewards(
    tokenizer: Union[ModelTokenizer, HuggingFaceModelTokenizer],
    completions: torch.Tensor,
    answers: list[str],
    device: torch.device,
    reward_metric: str = "f1",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Batched gene recall rewards for use in ``generate_trajectory``.

    Returns tensors shaped [batch_size, grpo_size, 1] matching the interface
    of ``batched_rewards``.
    """
    reward_fn = GeneRecallReward(reward_metric=reward_metric)

    batch_size, grpo_size, _ = completions.shape

    rewards_tensor = torch.zeros(
        batch_size, grpo_size, 1, dtype=torch.float32, device=device
    )
    successes_tensor = torch.zeros(
        batch_size, grpo_size, 1, dtype=torch.float32, device=device
    )

    for b in range(batch_size):
        for g in range(grpo_size):
            answer = answers[b]
            text_completion = tokenizer.decode(completions[b, g].tolist())
            out = reward_fn(
                completion_ids=completions[b, g],
                completions=[text_completion],
                answers=[answer],
            )
            rewards_tensor[b, g, 0] = out.total_reward[0]
            successes_tensor[b, g, 0] = out.successes[0]

    metadata = {"func_names": ["gene_recall"]}
    return rewards_tensor, successes_tensor, metadata


def batched_rewards(
    tokenizer: Union[ModelTokenizer, HuggingFaceModelTokenizer],
    completions: torch.Tensor,
    answers: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:

    reward_funcs = [
        at_least_one_space_between_think_tags,
        math_response_correct,
    ]

    num_reward_funcs = len(reward_funcs)

    batch_size, grpo_size, _ = completions.shape

    # TODO: should this be bfloat16?

    rewards_tensor = torch.zeros(
        batch_size, grpo_size, num_reward_funcs, dtype=torch.float32, device=device
    )

    successes_tensor = torch.zeros(
        batch_size, grpo_size, num_reward_funcs, dtype=torch.float32, device=device
    )

    metadata = {"func_names": [f.__name__ for f in reward_funcs]}

    for b in range(batch_size):

        for g in range(grpo_size):
            # print(answers)
            answer = answers[b]

            text_completion = tokenizer.decode(completions[b, g].tolist())

            cot, potential_answer = extract_tags(f"<think>{text_completion}")

            for rw_idx, reward_func in enumerate(reward_funcs):

                reward, success = reward_func(cot, answer, potential_answer)

                rewards_tensor[b, g, rw_idx] += reward

                successes_tensor[b, g, rw_idx] += success

    return rewards_tensor, successes_tensor, metadata


# ---------------------------------------------------------------------------
# Sum-of-digits rewards (torchtitan/ezpz comparison task)
# ---------------------------------------------------------------------------

def _extract_numeric_answer(text: str) -> str | None:
    """Extract a numeric answer using cascading patterns matching torchtitan/ezpz."""
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:the\s+)?answer\s+is\s+(\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"=\s*(\d+)", text)
    if m:
        return m.group(1).strip()
    numbers = re.findall(r"\b(\d+)\b", text)
    if numbers:
        return numbers[-1]
    return None


class SumDigitsReward(Reward):
    """
    Two-component reward for sum-of-digits arithmetic, matching torchtitan/ezpz:
    accuracy (1.0/0.0) + format (0.5/0.0).
    """

    def __init__(
        self,
        accuracy_reward: float = 1.0,
        format_reward: float = 0.5,
    ):
        self.accuracy_reward = accuracy_reward
        self.format_reward = format_reward

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: list[str],
        answers: list[str],
    ) -> RewardOutput:
        acc_rewards = []
        fmt_rewards = []
        for completion, answer in zip(completions, answers):
            extracted = _extract_numeric_answer(completion)
            acc_rewards.append(
                self.accuracy_reward if extracted == answer else 0.0
            )
            fmt_rewards.append(
                self.format_reward
                if re.search(r"\d+\s*\+\s*\d+", completion)
                else 0.0
            )

        acc_t = torch.tensor(acc_rewards)
        fmt_t = torch.tensor(fmt_rewards)
        total = acc_t + fmt_t
        return RewardOutput(
            reward_base_name="sum_digits",
            total_reward=total,
            successes=(acc_t == self.accuracy_reward).float(),
            rewards={"accuracy": acc_t, "format": fmt_t},
        )


def sum_digits_batched_rewards(
    tokenizer: Union[ModelTokenizer, HuggingFaceModelTokenizer],
    completions: torch.Tensor,
    answers: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Batched sum-of-digits rewards shaped ``[B, G, 2]``."""
    reward_fn = SumDigitsReward()

    batch_size, grpo_size, _ = completions.shape
    num_funcs = 2

    rewards_tensor = torch.zeros(
        batch_size, grpo_size, num_funcs, dtype=torch.float32, device=device
    )
    successes_tensor = torch.zeros(
        batch_size, grpo_size, num_funcs, dtype=torch.float32, device=device
    )

    for b in range(batch_size):
        for g in range(grpo_size):
            answer = answers[b]
            text_completion = tokenizer.decode(completions[b, g].tolist())
            out = reward_fn(
                completion_ids=completions[b, g],
                completions=[text_completion],
                answers=[answer],
            )
            rewards_tensor[b, g, 0] = out.rewards["accuracy"][0]
            rewards_tensor[b, g, 1] = out.rewards["format"][0]
            successes_tensor[b, g, 0] = out.successes[0]
            successes_tensor[b, g, 1] = (out.rewards["format"][0] > 0).float()

    metadata = {"func_names": ["accuracy", "format"]}
    return rewards_tensor, successes_tensor, metadata
