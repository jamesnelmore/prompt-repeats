"""Test prompt-repeat effectiveness on Gemma models via OpenRouter.
Each model is evaluated on GSM8k with and without CoT at 1, 2, 4, and 8
copies of the question.

Run with: python src/openrouter_survey.py --logs logs/some-run [--limit 500]

Uses inspect_ai.eval_set, so a re-run with the same log directory resumes progress instead of overwriting.
"""

import argparse

import inspect_ai
from dotenv import find_dotenv, load_dotenv
from inspect_ai.model import Model, get_model

from openrouter_task import prompt_repeat_comparison

COPY_COUNTS = (1, 2, 4, 8)


def arms(limit: int | None) -> list:
    """The arms. Inspect crosses these with every model below."""
    return [
        *[
            prompt_repeat_comparison(use_cot=use_cot, prompt_copies=n, limit=limit)
            for use_cot in (True, False)
            for n in COPY_COUNTS
        ],
    ]


# (model, provider, quantization)
# OpenRouter providers are pinned for reproducibility.
# As of August 6, 2026, gemma-3-27b-it is not served in bf16 on Openrouter by a provider that supports structured responses.
MODELS: list[tuple[str, str, str | None]] = [
    ("google/gemma-4-26b-a4b-it", "NextBit", "bf16"),
    ("google/gemma-4-31b-it", "CoreWeave", "bf16"),
    ("google/gemma-3-4b-it", "DeepInfra", "bf16"),
    ("google/gemma-3-12b-it", "DeepInfra", "bf16"),
    ("google/gemma-3-27b-it", "DeepInfra", "fp8"),
]

# Reduces rate limiting. Pinning providers makes this wors.
MAX_CONCURRENT_TASKS = 3


def pinned_models(only: str | None = None) -> list[Model]:
    """Resolve MODELS to Inspect Models carrying their OpenRouter routing.

    Passing Model objects rather than name strings is what makes per-model pins
    possible: eval_set's `model_args` applies one dict to every model uniformly.

    Args:
        only: Substring filter over model names; None runs every model.
    """
    models = []
    for name, provider, quantization in MODELS:
        if only is not None and only not in name:
            continue
        routing: dict[str, object] = {
            "order": [provider],
            "allow_fallbacks": False,  # Fail request if pinned provider is offline
        }
        if quantization is not None:
            routing["quantizations"] = [quantization]
        # Inspect's OpenRouter provider forwards `provider` as extra_body.provider,
        # and records it in the .eval log
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
        default=None,
        help="Samples per task, from the start of the dataset. "
        "Omit to run the entire GSM8K test set.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Substring filter over MODELS, e.g. 'gemma-3-4b'. Omit to run all. "
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv(find_dotenv(usecwd=True))

    success, logs = inspect_ai.eval_set(
        tasks=arms(args.limit),
        model=pinned_models(args.models),
        log_dir=args.logs,
        log_dir_allow_dirty=False,
        max_tasks=MAX_CONCURRENT_TASKS,
    )
    scope = "full test set" if args.limit is None else f"limit={args.limit}"
    status = "complete" if success else "INCOMPLETE (some tasks still failing)"
    print(f"eval_set {status}; {len(logs)} logs in {args.logs} ({scope}).")
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
