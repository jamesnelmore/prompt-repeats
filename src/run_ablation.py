"""Run six white-box attention ablations for repeated GSM8K questions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import torch
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from attention import (
    Mask,
    build_grammar,
    constrained_answer,
    masked_query_count,
    tokenize,
    verify_mask,
)
from gsm8k import get_questions, scored

MODEL_NAME = "google/gemma-3-12b-it"
BASELINE = "2copies"


@dataclass(frozen=True)
class Arm:
    """One explicit ablation arm."""

    name: str
    copies: int
    mask: Mask


class SampleRecord(TypedDict):
    """One result for one arm on one pinned GSM8K test row.

    `gsm8k_test_index` is the zero-based row position in the pinned test split.
    `aligned_question_token_count` is the number of strict-inside token pairs
    shared by the two question copies. `masked_query_token_count` is the number
    of query positions whose attention is changed by the arm's mask.
    """

    model: str
    gsm8k_test_index: int
    arm: str
    completion: str
    reference_answer: str
    correct: bool
    prompt_token_count: int
    copy1_token_count: int
    aligned_question_token_count: int
    masked_query_token_count: int


ARMS = (
    Arm("1copy", 1, None),
    Arm(BASELINE, 2, None),
    Arm("block_answer_from_copy1", 2, "answer"),
    Arm("block_copy2_from_copy1", 2, "copy2"),
    Arm("copy2_strictly_past_copy1", 2, "strictly_past"),
    Arm("copy2_past_or_aligned_copy1", 2, "past_or_aligned"),
)
MASKED_ARMS = tuple(arm for arm in ARMS if arm.mask is not None)


def load_done(path: Path, model: str) -> tuple[list[SampleRecord], set[tuple[int, str]]]:
    """Load prior records for this model, keyed by question and arm."""
    if not path.exists():
        return [], set()
    records: list[SampleRecord] = [
        cast(SampleRecord, record)
        for line in path.read_text().splitlines()
        if line and (record := json.loads(line)).get("model") == model
    ]
    return records, {
        (record["gsm8k_test_index"], record["arm"]) for record in records
    }


def summarise(records: list[SampleRecord]) -> dict[str, dict[str, object]]:
    """Report accuracy and paired changes against the unmasked two-copy baseline."""
    by_arm: dict[str, dict[int, bool]] = {}
    for record in records:
        by_arm.setdefault(record["arm"], {})[record["gsm8k_test_index"]] = record[
            "correct"
        ]

    baseline = by_arm.get(BASELINE, {})
    summary: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        scores = by_arm.get(arm.name, {})
        shared = set(scores) & set(baseline)
        if scores:
            summary[arm.name] = {
                "n": len(scores),
                "accuracy": round(sum(scores.values()) / len(scores), 4),
                "vs_2copies": {
                    "broken": sum(
                        baseline[index] and not scores[index] for index in shared
                    ),
                    "fixed": sum(
                        not baseline[index] and scores[index] for index in shared
                    ),
                },
            }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/attention_ablation")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument("--max-digits", type=int, default=8)
    parser.add_argument("--verify-mask", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    load_dotenv(find_dotenv(usecwd=True))

    print(f"Loading {args.model} on {args.device} (bf16, eager)...")
    tokenizer = cast(
        PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(args.model)
    )
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="eager"
        ).to(args.device),
    )
    model.eval()

    grammar = build_grammar(tokenizer, args.device)
    questions = get_questions(None if args.questions < 0 else args.questions)
    print(f"{len(questions)} questions x {len(ARMS)} arms")

    samples_path = out / "samples.jsonl"
    records: list[SampleRecord] = []
    done: set[tuple[int, str]] = set()
    if args.resume:
        records, done = load_done(samples_path, args.model)
        print(f"resuming: {len(done)} question-arm pairs already done")
    elif samples_path.exists():
        samples_path.unlink()

    if args.verify_mask:
        prompt = tokenize(tokenizer, questions[0][0], 2, args.device)
        checks = [verify_mask(model, prompt, arm.mask) for arm in MASKED_ARMS]
        for check in checks:
            assert check["masked"] == 0.0, f"mask leaked: {check}"
        (out / "mask_check.json").write_text(json.dumps(checks, indent=2))

    with samples_path.open("a") as sink:
        for gsm8k_test_index, (question, reference_answer) in enumerate(tqdm(questions)):
            for arm in ARMS:
                if (gsm8k_test_index, arm.name) in done:
                    continue
                prompt = tokenize(tokenizer, question, arm.copies, args.device)
                completion = constrained_answer(
                    model, tokenizer, prompt, grammar, arm.mask, args.max_digits
                )
                record: SampleRecord = {
                    "model": args.model,
                    "gsm8k_test_index": gsm8k_test_index,
                    "arm": arm.name,
                    "completion": completion,
                    "reference_answer": reference_answer,
                    "correct": scored(completion, reference_answer),
                    "prompt_token_count": int(prompt.ids.shape[1]),
                    "copy1_token_count": len(prompt.copy1),
                    "aligned_question_token_count": len(prompt.aligned1),
                    "masked_query_token_count": masked_query_count(prompt, arm.mask),
                }
                records.append(record)
                sink.write(json.dumps(record) + "\n")
                sink.flush()

    report = {
        "model": args.model,
        "questions": len(questions),
        "arms": [arm.name for arm in ARMS],
        "results": summarise(records),
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
