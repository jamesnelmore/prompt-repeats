"""Run the CoT vs no-CoT comparison across the survey model set via OpenRouter.

Builds the task arms (CoT ceiling + no-CoT at 1/2/4/8 repeats) and crosses them
with every model below. Inspect computes the {task} x {model} matrix and runs it.

Run with:  python src/gemma_eval.py --log-dir logs/some-run --limit 500

Uses inspect_ai.eval, so every invocation runs every task: there is no skipping
of already-completed work, and no automatic retry of tasks that fail. Each run
writes fresh logs into --log-dir alongside whatever is already there.
"""

import argparse

import inspect_ai
from dotenv import find_dotenv, load_dotenv
from inspect_ai.model import Model, get_model

from task import prompt_repeat_comparison

# The arms. Inspect crosses these with every model below.
TASKS = [
    prompt_repeat_comparison(use_cot=True),
    *[prompt_repeat_comparison(use_cot=False, prompt_repeats=r) for r in (1, 2, 4, 8)],
]

# (model, provider, quantization)
MODELS: list[tuple[str, str, str | None]] = [
    ("google/gemma-4-26b-a4b-it", "NextBit", "bf16"),
    ("google/gemma-4-31b-it", "CoreWeave", "bf16"),
]


def pinned_models() -> list[Model]:
    """Resolve MODELS to Inspect Models carrying their OpenRouter routing.

    Passing Model objects rather than name strings is what makes per-model pins
    possible: eval_set's `model_args` applies one dict to every model uniformly.
    """
    models = []
    for name, provider, quantization in MODELS:
        routing: dict[str, object] = {
            "order": [provider],
            "allow_fallbacks": False,  # Fail request if pinned provider is offline
        }
        if quantization is not None:
            routing["quantizations"] = [quantization]
        # Inspect's OpenRouter provider forwards `provider` as extra_body.provider,
        # and records it in the .eval log, so each log states how it was routed.
        models.append(get_model(f"openrouter/{name}", provider=routing))
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        required=True,
        help="Directory to write .eval logs to.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Samples per task, from the start of the dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv(find_dotenv(usecwd=True))

    logs = inspect_ai.eval(
        tasks=TASKS,
        model=pinned_models(),
        log_dir=args.logs,
        limit=args.limit,
    )
    print(f"eval complete; {len(logs)} logs written to {args.logs}.")


if __name__ == "__main__":
    main()
