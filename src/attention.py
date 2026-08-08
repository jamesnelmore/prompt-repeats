"""Token spans, 4D attention masks, and local constrained decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from gsm8k import ANSWER_CLOSE, ANSWER_PREFIX, no_cot_prompt

type Mask = Literal["answer", "copy2", "strictly_past", "past_or_aligned"] | None


@dataclass
class AnswerGrammar:
    """Token ids admitted by the shared integer JSON response format."""

    prefix: torch.Tensor
    digits: torch.Tensor
    close: int
    minus: int | None

    def allowed(self, emitted: int) -> torch.Tensor:
        extra = ([self.minus] if self.minus is not None else []) if emitted == 0 else [self.close]
        if not extra:
            return self.digits
        return torch.cat([self.digits, torch.tensor(extra, device=self.digits.device)])


def build_grammar(
    tokenizer: PreTrainedTokenizerBase, device: str | torch.device
) -> AnswerGrammar:
    """Resolve the answer-format grammar against one tokenizer.

    Forces answer to look like `{"answer": <NUMBER>}`.
    """

    def single(text: str) -> int | None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return int(ids[0]) if len(ids) == 1 else None

    digits = [
        token_id
        for token_id in range(tokenizer.vocab_size)
        if cast(str, tokenizer.decode([token_id])).isdigit()
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


@dataclass
class Prompt:
    """Tokenized prompt and question-token positions used by attention masks.

    `copy1` and `copy2` include every token whose character span overlaps the
    respective question. Rectangular masks use these conservative sets so a
    boundary token carrying any question text cannot leak information.

    `aligned1` and `aligned2` contain only tokens strictly inside each question
    span. Causal cross-copy masks pair them by ordinal position: aligned2[i]
    corresponds to aligned1[i]. They are derived from the tokenizer's natural
    full-prompt output rather than tokenizing each copy separately and stitching
    ids together, since tokenization is boundary-sensitive and doing so would
    change the model input. If the two strict spans differ in length, alignment
    is truncated to their shared prefix.
    """

    ids: torch.Tensor
    copy1: np.ndarray
    copy2: np.ndarray
    aligned1: np.ndarray
    aligned2: np.ndarray

    def blocked_queries(self, mask: Mask, total_length: int) -> np.ndarray:
        if mask is None or mask in {"strictly_past", "past_or_aligned"}:
            return np.array([], dtype=int)
        if mask == "copy2":
            return self.copy2
        assert mask == "answer"
        return np.arange(int(self.copy2.max()) + 1, total_length)


def tokenize(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    copies: int,
    device: str | torch.device,
) -> Prompt:
    """Apply the chat template and locate the repeated questions in token space."""
    user_text = no_cot_prompt(question, copies)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert isinstance(text, str)

    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)

    spans: list[tuple[int, int]] = []
    search_from = 0
    for _ in range(copies):
        start = text.index(question, search_from)
        spans.append((start, start + len(question)))
        search_from = start + len(question)

    def positions(span: tuple[int, int], strict: bool) -> np.ndarray:
        span_start, span_end = span
        return np.array(
            [
                index
                for index, (start, end) in enumerate(encoding["offset_mapping"])
                if end > start
                and (
                    start >= span_start and end <= span_end
                    if strict
                    else start < span_end and end > span_start
                )
            ],
            dtype=int,
        )

    copy1 = positions(spans[0], strict=False)
    copy2 = positions(spans[1], strict=False) if copies == 2 else np.array([], dtype=int)
    strict1 = positions(spans[0], strict=True)
    strict2 = positions(spans[1], strict=True) if copies == 2 else np.array([], dtype=int)
    width = min(len(strict1), len(strict2)) if copies == 2 else 0
    return Prompt(
        ids=torch.tensor([encoding["input_ids"]], device=device),
        copy1=copy1,
        copy2=np.setdiff1d(copy2, copy1),
        aligned1=strict1[:width],
        aligned2=strict2[:width],
    )


def blocked_bias(
    query_positions: torch.Tensor,
    kv_length: int,
    blocked_queries: np.ndarray,
    blocked_keys: np.ndarray,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return an additive causal attention bias with one rectangle blocked.

    A zero bias leaves an attention edge unchanged; `dtype.min` makes it
    effectively zero after softmax. The returned shape is `[1, 1, Q, K]`,
    broadcast by Hugging Face over the batch and attention-head dimensions.
    """
    keys = torch.arange(kv_length, device=query_positions.device)
    allowed = keys[None, :] <= query_positions[:, None]
    if len(blocked_queries):
        blocked_query_ids = torch.tensor(
            blocked_queries, device=query_positions.device
        )
        blocked_key_ids = torch.tensor(
            blocked_keys, device=query_positions.device
        )
        rows = torch.isin(query_positions, blocked_query_ids)
        columns = torch.isin(keys, blocked_key_ids)
        allowed &= ~(rows[:, None] & columns[None, :])
    assert bool(allowed.any(dim=1).all()), "a query position was fully masked"
    bias = torch.where(allowed, 0.0, torch.finfo(dtype).min).to(dtype)
    return bias[None, None] # Add batch and head axes


def causal_cross_bias(
    query_positions: torch.Tensor,
    kv_length: int,
    aligned1: np.ndarray,
    aligned2: np.ndarray,
    include_aligned: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return causal bias where copy2[i] reads copy1[j < i] or copy1[j <= i].

    Unlike `blocked_bias`, this blocks a triangle inside the copy2 × copy1
    rectangle: every copy-2 token gets a different cutoff in copy 1.
    """
    keys = torch.arange(kv_length, device=query_positions.device)
    allowed = keys[None, :] <= query_positions[:, None]
    if len(aligned1):
        # Map full-sequence token positions to their ordinal index inside each
        # aligned question span. -1 means "not an aligned question token".
        indices = torch.arange(len(aligned1), device=query_positions.device)
        query_index = torch.full((kv_length,), -1, device=query_positions.device)
        key_index = torch.full((kv_length,), -1, device=query_positions.device)
        query_index[torch.tensor(aligned2, device=query_positions.device)] = indices
        key_index[torch.tensor(aligned1, device=query_positions.device)] = indices
        q_i = query_index[query_positions]
        k_j = key_index
        cross_copy = (q_i[:, None] >= 0) & (k_j[None, :] >= 0)
        # Strict-past blocks j >= i; past-or-aligned blocks only j > i.
        blocked = (
            k_j[None, :] > q_i[:, None]
            if include_aligned
            else k_j[None, :] >= q_i[:, None]
        )
        allowed &= ~(cross_copy & blocked)
    assert bool(allowed.any(dim=1).all()), "a query position was fully masked"
    return torch.where(allowed, 0.0, torch.finfo(dtype).min).to(dtype)[None, None]


def attention_bias(
    prompt: Prompt,
    query_positions: torch.Tensor,
    kv_length: int,
    mask: Mask,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the additive 4D attention bias for one local decoding pass."""
    # These arms need token-by-token cutoffs, not one rectangular block.
    if mask in {"strictly_past", "past_or_aligned"}:
        return causal_cross_bias(
            query_positions,
            kv_length,
            prompt.aligned1,
            prompt.aligned2,
            include_aligned=mask == "past_or_aligned",
            dtype=dtype,
        )
    # No mask, copy2, and answer arms all use one rectangular block.
    return blocked_bias(
        query_positions,
        kv_length,
        prompt.blocked_queries(mask, kv_length),
        prompt.copy1,
        dtype,
    )


def check_window(model: PreTrainedModel, length: int) -> None:
    """Reject prompts that would replace a Gemma sliding window with full attention."""
    window = getattr(model.config.get_text_config(), "sliding_window", None)
    assert window is None or length <= window, (
        f"sequence is {length} tokens but the sliding window is {window}; "
        "the 4D mask would replace it with full attention"
    )


@torch.inference_mode()
def constrained_answer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: Prompt,
    grammar: AnswerGrammar,
    mask: Mask,
    max_digits: int,
) -> str:
    """Greedily produce one JSON integer answer under a custom attention mask."""
    check_window(model, prompt.ids.shape[1] + grammar.prefix.shape[1] + max_digits)
    past: Cache | None = None
    seen = 0

    def advance(new_ids: torch.Tensor) -> torch.Tensor:
        nonlocal past, seen
        total = seen + new_ids.shape[1]
        outputs = cast(
            CausalLMOutputWithPast,
            model(
                input_ids=new_ids,
                attention_mask=attention_bias(
                    prompt,
                    torch.arange(seen, total, device=prompt.ids.device),
                    total,
                    mask,
                    model.dtype,
                ),
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            ),
        )
        past, seen = outputs.past_key_values, total
        assert outputs.logits is not None
        return outputs.logits[0, -1].float()

    advance(prompt.ids)
    logits = advance(grammar.prefix)
    emitted_ids: list[int] = []
    for emitted in range(max_digits):
        allowed = grammar.allowed(emitted)
        allowed_logits = logits[allowed]
        best_allowed_index = int(allowed_logits.argmax())
        next_id = int(allowed[best_allowed_index])
        emitted_ids.append(next_id)
        if next_id == grammar.close:
            break
        logits = advance(torch.tensor([[next_id]], device=prompt.ids.device))
    return cast(str, tokenizer.decode(grammar.prefix[0].tolist() + emitted_ids))


@torch.inference_mode()
def verify_mask(
    model: PreTrainedModel, prompt: Prompt, mask: Mask
) -> dict[str, str | int | float]:
    """Return maximum blocked attention mass before and after applying one mask."""
    assert mask is not None, "mask verification requires an active mask"
    length = prompt.ids.shape[1]
    positions = torch.arange(length, device=prompt.ids.device)
    masses: dict[str, float] = {}
    checks: tuple[tuple[str, Mask], ...] = (("unmasked", None), ("masked", mask))
    for label, active in checks:
        outputs = cast(
            CausalLMOutputWithPast,
            model(
                input_ids=prompt.ids,
                attention_mask=attention_bias(
                    prompt, positions, length, active, model.dtype
                ),
                use_cache=False,
                output_attentions=True,
            ),
        )
        assert outputs.attentions is not None
        patterns = torch.stack([layer[0] for layer in outputs.attentions]).float()
        if mask in {"strictly_past", "past_or_aligned"}:
            block = patterns[:, :, prompt.aligned2][:, :, :, prompt.aligned1]
            diagonal = 1 if mask == "past_or_aligned" else 0
            blocked = torch.triu(
                torch.ones(
                    len(prompt.aligned1),
                    len(prompt.aligned1),
                    device=patterns.device,
                    dtype=torch.bool,
                ),
                diagonal=diagonal,
            )
            masses[label] = round(float(block[:, :, blocked].max()), 6)
            count = int(blocked.sum())
        else:
            queries = prompt.blocked_queries(mask, length)
            quadrant = patterns[:, :, queries][:, :, :, prompt.copy1]
            masses[label] = round(float(quadrant.sum(dim=-1).max()), 6)
            count = len(queries)
    return {"mask": mask, "blocked_queries": count, **masses}


def masked_query_count(prompt: Prompt, mask: Mask) -> int:
    """Return the number of query positions affected by the mask."""
    if mask in {"strictly_past", "past_or_aligned"}:
        return len(prompt.aligned2)
    return len(prompt.blocked_queries(mask, prompt.ids.shape[1]))
