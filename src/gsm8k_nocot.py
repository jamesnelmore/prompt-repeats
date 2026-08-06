"""Prompts, grammar, scorer, and dataset loading.
"""

import re
from dataclasses import dataclass

import torch
from datasets import load_dataset

from task import (
    DATASET_PATH,
    GSM8K_DATASET_REVISION,
    NO_COT_PROMPT_TEMPLATE,
    template_with_copies,
)

# Constrained decoding, hardcoded for `task.py`'s ANSWER_SCHEMA: an object with a
# single integer "answer". The prefix is force-fed, then only digits (and a
# leading "-") are decodable, so no token sequence can reason first -- the same
# thing the eval's response_schema buys from the provider. This is a grammar for
# that one schema, not a general JSON-schema enforcer: change ANSWER_SCHEMA and
# this has to change with it.
ANSWER_PREFIX = '{"answer": '
ANSWER_CLOSE = "}"
# The eval's own scorer, so a completion here is read exactly as the eval read it.
ANSWER_PATTERN = re.compile(r'"answer":\s*(-?\d+)')


def questions(count: int | None) -> list[tuple[str, str]]:
    """(question, reference answer) for the first `count` pinned GSM8K test rows.

    `count=None` is the whole test set.
    """
    dataset = load_dataset(
        DATASET_PATH, "main", split="test", revision=GSM8K_DATASET_REVISION
    )
    return [
        (dataset[i]["question"], dataset[i]["answer"].split("####")[-1].strip())
        for i in range(len(dataset) if count is None else count)
    ]


def build_prompt(tokenizer, question: str, copies: int) -> str:
    """The eval's user message with `copies` copies of the question (1 = once)."""
    user_text = template_with_copies(NO_COT_PROMPT_TEMPLATE, copies).format(
        prompt=question
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Ignored by templates that have no thinking mode.
    )


@dataclass
class AnswerGrammar:
    """Token ids the ANSWER_SCHEMA grammar can emit, resolved for one tokenizer."""

    prefix: torch.Tensor  # Force-fed: '{"answer": '
    digits: torch.Tensor  # Every all-digit token in the vocabulary.
    close: int  # '}', which ends the integer.
    minus: int | None  # Legal only as the first character; None if multi-token.

    def allowed(self, emitted: int) -> torch.Tensor:
        """Ids decodable after `emitted` digit tokens."""
        if emitted == 0:
            extra = [] if self.minus is None else [self.minus]
        else:
            extra = [self.close]  # Closing early is the model's choice of length.
        if not extra:
            return self.digits
        tail = torch.tensor(extra, device=self.digits.device)
        return torch.cat([self.digits, tail])


def build_grammar(tokenizer, device: str | torch.device) -> AnswerGrammar:
    """Resolve the schema's literals and the digit allowlist against the vocabulary.

    Derived from the tokenizer rather than hardcoded ids, so pointing --model at
    a different family (gemma) needs no edit here.
    """

    def single(text: str) -> int | None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return int(ids[0]) if len(ids) == 1 else None

    # Multi-digit tokens are allowed and count as one step; "01" style leading
    # zeros are legal JSON-wise here but the scorer's regex reads them fine.
    digits = [
        token_id
        for token_id in range(tokenizer.vocab_size)
        if (piece := tokenizer.decode([token_id])) != "" and piece.isdigit()
    ]
    close = single(ANSWER_CLOSE)
    assert close is not None, f"{ANSWER_CLOSE!r} is not a single token"
    return AnswerGrammar(
        prefix=torch.tensor(
            [tokenizer(ANSWER_PREFIX, add_special_tokens=False)["input_ids"]],
            device=device,
        ),
        digits=torch.tensor(digits, device=device),
        close=close,
        minus=single("-"),
    )


def scored(completion: str, reference: str) -> bool:
    """The eval's no-CoT scorer, applied to a constrained completion."""
    found = ANSWER_PATTERN.search(completion)
    return found is not None and found.group(1) == reference
