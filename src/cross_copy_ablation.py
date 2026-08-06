"""Which path from copy 1 to the answer does the repeat need? (attention ablation)

Copy 1 can reach the answer two ways: the answer tokens attend to it directly, or
copy 2 attends to it and the answer then reads copy 2. `cross_copy_vis.py` shows
cross-copy attention exists; this script severs each path on its own and re-scores
the eval's no-CoT arm. If the repeat helps because of one path, accuracy should
fall when that path is cut and hold when the other is; if both (or neither)
matter, the two cuts will look the same.

Prompts, grammar and scorer are `nocot_eval`'s, i.e. the ones the eval used.

HF transformers rather than transformer-lens: this is aimed at gemma-3 12b/27b,
which load from the HF cache in bf16 onto one A100 as-is, while transformer-lens
would need its own weight conversion of a model it does not properly support. The
mask goes in as a 4D `attention_mask`, which is HF's documented escape hatch --
`masking_utils._preprocess_mask_arguments` returns any 4D mask as-is, so ours
replaces the causal mask in every layer. See `blocked_bias` for what that means
on gemma-3's sliding-window layers, and run --verify-mask to confirm on the
actual model that the masked quadrant of the attention pattern really is zero.

Conditions, all run per question:

  repeat1               the 1-repeat floor
  repeat2               the 2-repeat ceiling, unmasked
  repeat2_mask_copy2    copy 2 cannot attend to copy 1; the answer still can
                        (c1 → answer only)
  repeat2_mask_answer   the answer (and everything after copy 2) cannot attend
                        to copy 1; copy 2 still can (c1 → c2 only)

Under both masks the answer can still see copy 2, so a severed path only removes
copy 1's *direct* contribution along that edge -- residual info already mixed
into copy 2 remains readable.

Run with:
  python src/cross_copy_ablation.py --model google/gemma-3-12b-it --questions 10 \
      --out results/cross_copy_ablation_pathways_12b
  python src/cross_copy_ablation.py --model google/gemma-3-27b-it --questions -1 \
      --out results/cross_copy_ablation_pathways_27b
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nocot_eval import (
    AnswerGrammar,
    build_grammar,
    build_prompt,
    questions,
    scored,
)

MODEL_NAME = "google/gemma-3-12b-it"
BASELINE = "repeat2"  # The condition the masked ones are compared against.
# Distinguishes this run from earlier ablations that used mask_after (blinding
# everything after copy 1) instead of the pathway split below.
INTERVENTION = "pathways"
INTERVENTION_DESC = (
    "Sever each copy-1→answer path alone: mask_copy2 = c1→answer only "
    "(copy 2 blind to copy 1); mask_answer = c1→c2 only (answer blind to copy 1)."
)


@dataclass(frozen=True)
class Condition:
    """One arm: how many copies of the question, and who is blinded to copy 1."""

    name: str
    repeats: int
    scope: str | None  # None | "copy2" | "answer"


CONDITIONS = [
    Condition("repeat1", 1, None),
    Condition(BASELINE, 2, None),
    Condition("repeat2_mask_copy2", 2, "copy2"),
    Condition("repeat2_mask_answer", 2, "answer"),
]


@dataclass
class Prompt:
    """A tokenised prompt plus the token positions the mask is defined over."""

    ids: torch.Tensor  # [1, prompt_len]
    copy1: np.ndarray  # Token positions carrying any of copy 1's characters.
    copy2: np.ndarray  # ... any of copy 2's, minus anything already in copy1.

    def blocked_queries(self, scope: str | None, total_length: int) -> np.ndarray:
        """Positions that must not see copy 1, over a sequence of `total_length`.

        "copy2" blinds only the second copy, leaving the answer free to read
        copy 1 directly (c1 → answer). "answer" blinds everything after copy 2
        -- the trailing scaffold and the decoded answer tokens as they appear --
        so copy 1 can only reach the answer through copy 2 (c1 → c2).
        """
        if scope is None or len(self.copy1) == 0:
            return np.array([], dtype=int)
        if scope == "copy2":
            return self.copy2
        assert scope == "answer", f"unknown mask scope {scope!r}"
        assert len(self.copy2) > 0, "answer mask needs a second copy"
        return np.arange(int(self.copy2.max()) + 1, total_length)


def tokenize(tokenizer, question: str, repeats: int, device) -> Prompt:
    """Tokenise the prompt and locate each copy of the question in it.

    A token joins a copy if its character span *overlaps* that copy, which is the
    conservative choice for an intervention: a token straddling the boundary
    carries some of copy 1's characters, so it counts as copy 1 (hidden as a key,
    and dropped from copy 2's queries -- nothing is ever masked from itself).

    This differs from `cross_copy_vis.copy_positions`, which takes only tokens
    lying strictly inside a copy so the two copies stay aligned token-for-token
    for the visualisation. Here alignment does not matter and a leaked boundary
    token would.
    """
    prompt = build_prompt(tokenizer, question, repeats)
    encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)

    spans: list[tuple[int, int]] = []
    search_from = 0
    for _ in range(repeats):
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

    copy1 = overlapping(spans[0])
    copy2 = overlapping(spans[1]) if repeats >= 2 else np.array([], dtype=int)
    return Prompt(
        ids=torch.tensor([encoding["input_ids"]], device=device),
        copy1=copy1,
        copy2=np.setdiff1d(copy2, copy1),
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
    where it is not. Additive rather than boolean so it is valid for both the
    sdpa and the eager attention implementations.

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
            attention_mask=blocked_bias(
                positions,
                total,
                prompt.blocked_queries(scope, total),
                prompt.copy1,
                model.dtype,
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
    """Read the attention patterns back and confirm the blocked quadrant is zero.

    Needs eager attention (sdpa returns no patterns). Reports the same quantity
    unmasked too, so a zero means the mask worked rather than that the model was
    never looking at copy 1 anyway.
    """
    length = prompt.ids.shape[1]
    positions = torch.arange(length, device=prompt.ids.device)
    queries = prompt.blocked_queries(scope, length)
    masses = {}
    for label, blocked in (("unmasked", np.array([], dtype=int)), ("masked", queries)):
        outputs = model(
            input_ids=prompt.ids,
            attention_mask=blocked_bias(
                positions, length, blocked, prompt.copy1, model.dtype
            ),
            use_cache=False,
            output_attentions=True,
        )
        # [layer, head, query, key], sliced to the blocked quadrant: the largest
        # share of any blocked query's attention that still lands on copy 1.
        patterns = torch.stack([layer[0] for layer in outputs.attentions]).float()
        quadrant = patterns[:, :, queries][:, :, :, prompt.copy1]
        masses[label] = round(float(quadrant.sum(dim=-1).max()), 6)
    return {"scope": scope, "blocked_queries": len(queries), **masses}


def summarise(records: list[dict]) -> dict:
    """Accuracy per condition, plus how each masked arm moves against `BASELINE`."""
    by_condition: dict[str, dict[int, bool]] = {}
    for record in records:
        by_condition.setdefault(record["condition"], {})[record["index"]] = record[
            "correct"
        ]

    summary: dict[str, dict] = {}
    baseline = by_condition.get(BASELINE, {})
    for condition in (c.name for c in CONDITIONS):
        scores = by_condition.get(condition)
        if not scores:
            continue
        shared = sorted(set(scores) & set(baseline))
        summary[condition] = {
            "n": len(scores),
            "accuracy": round(sum(scores.values()) / len(scores), 4),
            # Paired against the unmasked 2-repeat arm on the same questions, so
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="results/cross_copy_ablation_pathways",
        help="Output directory. Prefer a name that encodes the intervention "
        "(default: pathways). Do not reuse dirs from other interventions "
        "(e.g. results/cross_copy_ablation_mask_after_*).",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Any HF causal LM. gemma-3 12b and 27b both fit one 80GB A100 in bf16.",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=10,
        help="GSM8K test questions from the start of the split; -1 runs all 1319.",
    )
    parser.add_argument(
        "--max-digits",
        type=int,
        default=8,
        help="Cap on decoded digit tokens before the object is closed anyway.",
    )
    parser.add_argument(
        "--verify-mask",
        action="store_true",
        help="Before the run, read attention patterns back on question 0 and "
        "assert the masked quadrant is zero. Requires --attn eager.",
    )
    parser.add_argument(
        "--attn",
        default="eager",
        choices=["eager", "sdpa"],
        help="Attention implementation. Both honour the 4D mask and agree to "
        "within bf16 noise; eager is the default because only it can return "
        "attention patterns for --verify-mask, and it costs ~9%% here (these "
        "prompts are short enough that attention is not the bottleneck).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (question, condition) pairs already in samples.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    load_dotenv(find_dotenv(usecwd=True))  # HF_TOKEN, if the cache is cold.

    # bf16 throughout: it is what the eval's providers served, and fp16 overflows
    # gemma. Greedy decoding on top of bf16 is stable within one attention
    # implementation but not across them -- eager and sdpa flipped 4/40 answers
    # on the 12b, always where the top two logits were within ~0.4 -- so do not
    # mix --attn settings inside one comparison, and treat that flip rate as the
    # noise floor for per-question differences between conditions.
    assert not args.verify_mask or args.attn == "eager", (
        "--verify-mask reads attention patterns, which only eager returns"
    )
    print(f"Loading {args.model} on {args.device} (bf16, {args.attn})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    ).to(args.device)
    model.eval()

    grammar = build_grammar(tokenizer, args.device)
    asked = questions(None if args.questions < 0 else args.questions)
    print(f"{len(asked)} questions x {len(CONDITIONS)} conditions")

    # --resume only ever appends, so it is safe to point a second model at an
    # --out that already holds another's records. Without it the file is
    # truncated, which would take those records with it.
    samples_path = out / "samples.jsonl"
    records, done = [], set()
    if args.resume:
        records, done = load_done(samples_path, args.model)
        print(f"resuming: {len(done)} sample-conditions already done for {args.model}")
    elif samples_path.exists():
        print(f"overwriting {samples_path} ({args.model} run; --resume to keep it)")
        samples_path.unlink()

    if args.verify_mask:
        prompt = tokenize(tokenizer, asked[0][0], 2, args.device)
        checks = [verify_mask(model, prompt, scope) for scope in ("copy2", "answer")]
        for check in checks:
            print(f"mask check {check}")
            assert check["masked"] == 0.0, f"mask leaked: {check}"
            assert check["unmasked"] > 0.0, "nothing to mask; check the spans"
        (out / "mask_check.json").write_text(json.dumps(checks, indent=2))

    with samples_path.open("a") as sink:
        for index, (question, reference) in enumerate(tqdm(asked)):
            for condition in CONDITIONS:
                if (index, condition.name) in done:
                    continue
                prompt = tokenize(tokenizer, question, condition.repeats, args.device)
                completion = constrained_answer(
                    model, tokenizer, prompt, grammar, condition.scope, args.max_digits
                )
                record = {
                    "model": args.model,
                    "index": index,
                    "condition": condition.name,
                    "completion": completion,
                    "reference": reference,
                    "correct": scored(completion, reference),
                    "prompt_tokens": int(prompt.ids.shape[1]),
                    "copy1_tokens": len(prompt.copy1),
                    "masked_queries": len(
                        prompt.blocked_queries(condition.scope, prompt.ids.shape[1])
                    ),
                }
                records.append(record)
                sink.write(json.dumps(record) + "\n")
                sink.flush()  # A 27b full-set run should survive being killed.

    report = {
        "intervention": INTERVENTION,
        "intervention_desc": INTERVENTION_DESC,
        "model": args.model,
        "questions": len(asked),
        "condition_names": [c.name for c in CONDITIONS],
        "conditions": summarise(records),
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    (out / "INTERVENTION.txt").write_text(
        "\n".join(
            [
                f"intervention: {INTERVENTION}",
                f"description: {INTERVENTION_DESC}",
                f"model: {args.model}",
                f"conditions: {', '.join(c.name for c in CONDITIONS)}",
                "mask_copy2: copy2 cannot attend to copy1; answer still can (c1 → answer only)",
                "mask_answer: answer (after copy2) cannot attend to copy1; copy2 still can (c1 → c2 only)",
                "",
            ]
        )
    )
    print(json.dumps(report, indent=2))
    print(f"wrote {samples_path} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
