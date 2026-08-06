"""Attention ablations of the prompt-repeat effect (HF 4D masks).

Two intervention presets share one decode/mask engine (`ablation.masks`).
Condition names use "copies": `1copy` = question once, `2copies` = twice.

  pathways
    Which path from copy 1 to the answer does the second copy buy?
    Copy 1 can reach the answer two ways: the answer attends to it directly, or
    copy 2 attends to it and the answer then reads copy 2. Sever each path alone.
      1copy / 2copies / 2copies_mask_copy2 / 2copies_mask_answer

  causal_cross_copy
    Does the second copy need to see its own future in copy 1?
    Keep two copies but make cross-copy attention causal within the aligned
    copies: copy2[i] may attend to copy1[j] only for j < i.
      1copy / 2copies / 2copies_mask_copy2 / 2copies_causal_copy2

Prompts, grammar and scorer are `gsm8k_nocot`'s (the eval's no-CoT arm).

Run with:
  python src/run_ablation.py --intervention pathways --model google/gemma-3-12b-it \
      --questions 10 --out results/cross_copy_ablation_pathways_12b
  python src/run_ablation.py --intervention causal_cross_copy --model google/gemma-3-27b-it \
      --questions -1 --out results/cross_copy_ablation_causal_27b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ablation.interventions import INTERVENTIONS, Intervention
from ablation.masks import (
    constrained_answer,
    load_done,
    masked_query_count,
    summarise,
    tokenize,
    verify_mask,
)
from gsm8k_nocot import build_grammar, questions, scored

MODEL_NAME = "google/gemma-3-12b-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intervention",
        required=True,
        choices=sorted(INTERVENTIONS),
        help="Which ablation study to run (see module docstring).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to a name that encodes the intervention. "
        "Do not reuse dirs from other interventions.",
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
        "assert the masked cells are zero.",
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
    intervention: Intervention = INTERVENTIONS[args.intervention]
    out = Path(args.out if args.out is not None else intervention.default_out)
    out.mkdir(parents=True, exist_ok=True)
    load_dotenv(find_dotenv(usecwd=True))  # HF_TOKEN, if the cache is cold.

    # bf16 throughout: it is what the eval's providers served, and fp16 overflows
    # gemma. Eager only -- sdpa cannot return patterns for --verify-mask, and the
    # two impls disagree on greedy answers when the top logits are close.
    print(f"Loading {args.model} on {args.device} (bf16, eager)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    grammar = build_grammar(tokenizer, args.device)
    asked = questions(None if args.questions < 0 else args.questions)
    conditions = list(intervention.conditions)
    print(
        f"{intervention.name}: {len(asked)} questions x {len(conditions)} conditions"
    )

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
        checks = [
            verify_mask(model, prompt, scope) for scope in intervention.verify_scopes
        ]
        for check in checks:
            print(f"mask check {check}")
            assert check["masked"] == 0.0, f"mask leaked: {check}"
            assert check["unmasked"] > 0.0, "nothing to mask; check the spans"
        (out / "mask_check.json").write_text(json.dumps(checks, indent=2))

    with samples_path.open("a") as sink:
        for index, (question, reference) in enumerate(tqdm(asked)):
            for condition in conditions:
                if (index, condition.name) in done:
                    continue
                prompt = tokenize(tokenizer, question, condition.copies, args.device)
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
                    "aligned_tokens": len(prompt.aligned1),
                    "masked_queries": masked_query_count(prompt, condition.scope),
                }
                records.append(record)
                sink.write(json.dumps(record) + "\n")
                sink.flush()  # A 27b full-set run should survive being killed.

    report = {
        "intervention": intervention.name,
        "intervention_desc": intervention.description,
        "model": args.model,
        "questions": len(asked),
        "condition_names": [c.name for c in conditions],
        "conditions": summarise(records, conditions),
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    (out / "INTERVENTION.txt").write_text(
        "\n".join(
            [
                f"intervention: {intervention.name}",
                f"description: {intervention.description}",
                f"model: {args.model}",
                f"conditions: {', '.join(c.name for c in conditions)}",
                *intervention.notes,
                "",
            ]
        )
    )
    print(json.dumps(report, indent=2))
    print(f"wrote {samples_path} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
