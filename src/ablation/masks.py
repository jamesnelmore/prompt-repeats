"""Tokenise prompts, build 4D attention biases, and decode under them.

HF transformers rather than transformer-lens: aimed at gemma-3 12b/27b, which
load from the HF cache in bf16 onto one A100 as-is. The mask goes in as a 4D
`attention_mask`, which is HF's documented escape hatch --
`masking_utils._preprocess_mask_arguments` returns any 4D mask as-is, so ours
replaces the causal mask in every layer. See `blocked_bias` for what that means
on gemma-3's sliding-window layers, and run --verify-mask to confirm on the
actual model that the masked cells of the attention pattern really are zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gsm8k_nocot import AnswerGrammar, build_prompt


BASELINE = "2copies"  # The condition the masked ones are compared against.


@dataclass(frozen=True)
class Condition:
    """One arm: how many question copies, and who is blinded to copy 1."""

    name: str
    copies: int  # 1 = question once; 2 = twice; …
    # None | "copy2" | "answer" | "causal_copy2"
    scope: str | None


@dataclass
class Prompt:
    """A tokenised prompt plus the token positions the mask is defined over."""

    ids: torch.Tensor  # [1, prompt_len]
    copy1: np.ndarray  # Token positions carrying any of copy 1's characters.
    copy2: np.ndarray  # ... any of copy 2's, minus anything already in copy1.
    # Strict-inside, equal-length spans: copy2[i] aligns with copy1[i]. Used by
    # the causal_copy2 mask; unused by the rectangular pathway cuts.
    aligned1: np.ndarray
    aligned2: np.ndarray

    def blocked_queries(self, scope: str | None, total_length: int) -> np.ndarray:
        """Positions that must not see copy 1, over a sequence of `total_length`.

        "copy2" blinds only the second copy, leaving the answer free to read
        copy 1 directly (c1 → answer). "answer" blinds everything after copy 2
        -- the trailing scaffold and the decoded answer tokens as they appear --
        so copy 1 can only reach the answer through copy 2 (c1 → c2).
        "causal_copy2" is not a rectangular block; use `attention_bias` instead.
        """
        if scope is None or scope == "causal_copy2" or len(self.copy1) == 0:
            return np.array([], dtype=int)
        if scope == "copy2":
            return self.copy2
        assert scope == "answer", f"unknown mask scope {scope!r}"
        assert len(self.copy2) > 0, "answer mask needs a second copy"
        return np.arange(int(self.copy2.max()) + 1, total_length)


def tokenize(tokenizer, question: str, copies: int, device) -> Prompt:
    """Tokenise the prompt and locate each copy of the question in it.

    A token joins a copy if its character span *overlaps* that copy, which is the
    conservative choice for an intervention: a token straddling the boundary
    carries some of copy 1's characters, so it counts as copy 1 (hidden as a key,
    and dropped from copy 2's queries -- nothing is ever masked from itself).

    `aligned1` / `aligned2` instead keep only tokens lying strictly inside each
    copy so the two copies stay aligned token-for-token for causal_copy2.
    """
    prompt = build_prompt(tokenizer, question, copies)
    encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)

    spans: list[tuple[int, int]] = []
    search_from = 0
    for _ in range(copies):
        start = prompt.index(question, search_from)
        spans.append((start, start + len(question)))
        search_from = start + len(question)

    def overlapping(span: tuple[int, int]) -> np.ndarray:
        span_start, span_end = span
        return np.array(
            [
                index
                for index, (start, end) in enumerate(encoding["offset_mapping"])
                if end > start and start < span_end and end > span_start
            ],
            dtype=int,
        )

    def inside(span: tuple[int, int]) -> np.ndarray:
        span_start, span_end = span
        return np.array(
            [
                index
                for index, (start, end) in enumerate(encoding["offset_mapping"])
                if end > start and start >= span_start and end <= span_end
            ],
            dtype=int,
        )

    copy1 = overlapping(spans[0])
    copy2 = overlapping(spans[1]) if copies >= 2 else np.array([], dtype=int)
    if copies >= 2:
        strict = [inside(span) for span in spans[:2]]
        width = min(len(part) for part in strict)
        aligned1, aligned2 = strict[0][:width], strict[1][:width]
    else:
        aligned1 = inside(spans[0])
        aligned2 = np.array([], dtype=int)
    return Prompt(
        ids=torch.tensor([encoding["input_ids"]], device=device),
        copy1=copy1,
        copy2=np.setdiff1d(copy2, copy1),
        aligned1=aligned1,
        aligned2=aligned2,
    )


def blocked_bias(
    query_positions: torch.Tensor,
    kv_length: int,
    blocked_queries: np.ndarray,
    blocked_keys: np.ndarray,
    dtype: torch.dtype,
) -> torch.Tensor:
    """The additive attention bias for one forward pass: causal, minus the block.

    Shape [1, 1, queries, kv_length], 0 where attention is allowed and dtype-min
    where it is not. Additive rather than boolean (HF's eager path accepts either).

    Because HF takes a 4D mask as-is, this bias is also what gemma-3's
    sliding-window layers get, in place of the window they would have built.
    That is only harmless while the sequence fits inside the window, which
    `check_window` asserts rather than assumes.
    """
    device = query_positions.device
    keys = torch.arange(kv_length, device=device)
    allowed = keys[None, :] <= query_positions[:, None]

    if len(blocked_queries) and len(blocked_keys):
        rows = torch.isin(query_positions, torch.tensor(blocked_queries, device=device))
        columns = torch.isin(keys, torch.tensor(blocked_keys, device=device))
        allowed &= ~(rows[:, None] & columns[None, :])

    # A fully masked row would come back as NaN rather than as an error.
    assert bool(allowed.any(dim=1).all()), "a query position was left with no keys"
    return torch.where(allowed, 0.0, torch.finfo(dtype).min).to(dtype)[None, None]


def causal_cross_bias(
    query_positions: torch.Tensor,
    kv_length: int,
    aligned1: np.ndarray,
    aligned2: np.ndarray,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Causal mask, plus copy2[i] cannot attend to copy1[j] for j >= i.

    Keeps the second question copy but stops it from reading its own "future" in
    copy 1: the first copy-2 token is blind to all of copy 1, the next may only
    see copy1[0], and so on. Answer tokens and non-copy keys are untouched.
    """
    device = query_positions.device
    keys = torch.arange(kv_length, device=device)
    allowed = keys[None, :] <= query_positions[:, None]

    if len(aligned1) and len(aligned2):
        assert len(aligned1) == len(aligned2), "aligned copies must be equal length"
        query_index = torch.full((kv_length,), -1, device=device, dtype=torch.long)
        key_index = torch.full((kv_length,), -1, device=device, dtype=torch.long)
        order = torch.arange(len(aligned2), device=device)
        query_index[torch.tensor(aligned2, device=device)] = order
        key_index[torch.tensor(aligned1, device=device)] = order

        q_i = query_index[query_positions]
        k_j = key_index
        cross = (q_i[:, None] >= 0) & (k_j[None, :] >= 0)
        # Block aligned and future copy-1 keys (j >= i); past keys (j < i) stay.
        allowed &= ~(cross & (k_j[None, :] >= q_i[:, None]))

    assert bool(allowed.any(dim=1).all()), "a query position was left with no keys"
    return torch.where(allowed, 0.0, torch.finfo(dtype).min).to(dtype)[None, None]


def attention_bias(
    prompt: Prompt,
    query_positions: torch.Tensor,
    kv_length: int,
    scope: str | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the 4D additive mask for `scope` (rectangular or causal-cross)."""
    if scope == "causal_copy2":
        return causal_cross_bias(
            query_positions, kv_length, prompt.aligned1, prompt.aligned2, dtype
        )
    return blocked_bias(
        query_positions,
        kv_length,
        prompt.blocked_queries(scope, kv_length),
        prompt.copy1,
        dtype,
    )


def check_window(model, length: int) -> None:
    window = getattr(model.config.get_text_config(), "sliding_window", None)
    assert window is None or length <= window, (
        f"sequence is {length} tokens but the sliding window is {window}: a 4D mask "
        "replaces the window, so this run would hand sliding layers full attention. "
        "Build the mask per layer type before running prompts this long."
    )


@torch.inference_mode()
def constrained_answer(
    model,
    tokenizer,
    prompt: Prompt,
    grammar: AnswerGrammar,
    scope: str | None,
    max_digits: int,
) -> str:
    """Greedy-decode one ANSWER_SCHEMA object under the mask, with a KV cache.

    The prompt and the force-fed prefix go through in two batched passes, then
    one pass per digit. The bias is rebuilt every pass because the key length
    grows; under "answer" that is also what keeps the new answer tokens masked.
    """
    device = prompt.ids.device
    check_window(model, prompt.ids.shape[1] + grammar.prefix.shape[1] + max_digits)

    past, seen = None, 0

    def advance(new_ids: torch.Tensor) -> torch.Tensor:
        """Run `new_ids` on top of the cache; return the last position's logits."""
        nonlocal past, seen
        total = seen + new_ids.shape[1]
        positions = torch.arange(seen, total, device=device)
        outputs = model(
            input_ids=new_ids,
            attention_mask=attention_bias(
                prompt, positions, total, scope, model.dtype
            ),
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        past, seen = outputs.past_key_values, total
        return outputs.logits[0, -1].float()

    advance(prompt.ids)
    logits = advance(grammar.prefix)

    emitted_ids: list[int] = []
    for emitted in range(max_digits):
        allowed = grammar.allowed(emitted)
        next_id = int(allowed[int(logits[allowed].argmax())])
        emitted_ids.append(next_id)
        if next_id == grammar.close:
            break
        logits = advance(torch.tensor([[next_id]], device=device))

    # Decoded with the prefix, so the eval's scorer sees the whole JSON object.
    return tokenizer.decode(grammar.prefix[0].tolist() + emitted_ids)


@torch.inference_mode()
def verify_mask(model, prompt: Prompt, scope: str) -> dict:
    """Read the attention patterns back and confirm the blocked cells are zero.

    Reports the same quantity unmasked too, so a zero means the mask worked
    rather than that the model was never looking at those keys anyway.
    """
    length = prompt.ids.shape[1]
    positions = torch.arange(length, device=prompt.ids.device)
    masses = {}
    for label, active in (("unmasked", None), ("masked", scope)):
        outputs = model(
            input_ids=prompt.ids,
            attention_mask=attention_bias(
                prompt, positions, length, active, model.dtype
            ),
            use_cache=False,
            output_attentions=True,
        )
        patterns = torch.stack([layer[0] for layer in outputs.attentions]).float()
        if scope == "causal_copy2":
            # Upper triangle of the aligned copy2×copy1 block, including diagonal.
            block = patterns[:, :, prompt.aligned2][:, :, :, prompt.aligned1]
            n = len(prompt.aligned1)
            future = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=0)
            masses[label] = round(float(block[:, :, future].max()), 6)
            n_blocked = int(future.sum())
        else:
            queries = prompt.blocked_queries(scope, length)
            # [layer, head, query, key], sliced to the blocked quadrant: the largest
            # share of any blocked query's attention that still lands on copy 1.
            quadrant = patterns[:, :, queries][:, :, :, prompt.copy1]
            masses[label] = round(float(quadrant.sum(dim=-1).max()), 6)
            n_blocked = len(queries)
    return {"scope": scope, "blocked_queries": n_blocked, **masses}


def masked_query_count(prompt: Prompt, scope: str | None) -> int:
    """How many query positions the intervention touches (for the sample log)."""
    if scope == "causal_copy2":
        return len(prompt.aligned2)
    return len(prompt.blocked_queries(scope, prompt.ids.shape[1]))


def summarise(records: list[dict], conditions: list[Condition]) -> dict:
    """Accuracy per condition, plus how each masked arm moves against `BASELINE`."""
    by_condition: dict[str, dict[int, bool]] = {}
    for record in records:
        by_condition.setdefault(record["condition"], {})[record["index"]] = record[
            "correct"
        ]

    summary: dict[str, dict] = {}
    baseline = by_condition.get(BASELINE, {})
    for condition in (c.name for c in conditions):
        scores = by_condition.get(condition)
        if not scores:
            continue
        shared = sorted(set(scores) & set(baseline))
        summary[condition] = {
            "n": len(scores),
            "accuracy": round(sum(scores.values()) / len(scores), 4),
            # Paired against the unmasked 2-copy arm on the same questions, so
            # a small sample can still show whether answers actually changed.
            "vs_baseline": {
                "broken": sum(1 for i in shared if baseline[i] and not scores[i]),
                "fixed": sum(1 for i in shared if not baseline[i] and scores[i]),
            },
        }
    return summary


def load_done(path: Path, model: str) -> tuple[list[dict], set[tuple[int, str]]]:
    """Records already on disk for `model`, so a run can be resumed after a crash.

    Filtered by model so that pointing two models at one --out resumes and
    summarises each on its own records instead of crediting one with the other's.
    """
    if not path.exists():
        return [], set()
    records = [
        record
        for line in path.read_text().splitlines()
        if line and (record := json.loads(line)).get("model") == model
    ]
    return records, {(r["index"], r["condition"]) for r in records}
