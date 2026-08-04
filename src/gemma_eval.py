"""Run the CoT vs no-CoT comparison across the survey model set via OpenRouter.

Builds the task arms (CoT ceiling + no-CoT at 1/2/4/8 repeats) and crosses them
with every model below. Inspect computes the {task} x {model} matrix and runs it.

Run with:  python src/gemma_eval.py --log-dir logs/some-run [--limit 500]

Uses inspect_ai.eval_set, so a re-run against the same --log-dir resumes rather than
repeats: tasks that already completed successfully are skipped, and tasks that
failed are retried. Deleting or changing the log directory forces a re-run.

--limit is passed as a task arg, not as eval_set's own limit, because eval_set
hashes task args into the task identity but not its limit. Changing --limit
therefore yields new task identities and reruns the matrix, instead of silently
matching the previous run's logs and reporting everything as already done.
"""

import argparse

import inspect_ai
from dotenv import find_dotenv, load_dotenv
from inspect_ai.model import Model, get_model

from task import prompt_repeat_comparison


def arms(limit: int | None) -> list:
    """The arms. Inspect crosses these with every model below."""
    return [
        prompt_repeat_comparison(use_cot=True, limit=limit),
        *[
            prompt_repeat_comparison(use_cot=False, prompt_repeats=r, limit=limit)
            for r in (1, 2, 4, 8)
        ],
    ]


# (model, provider, quantization)
MODELS: list[tuple[str, str, str | None]] = [
    ("google/gemma-4-26b-a4b-it", "NextBit", "bf16"),
    ("google/gemma-4-31b-it", "CoreWeave", "bf16"),
    # (August 2026) DeepInfra is currently the only provider serving this one, and the no-CoT arm needs a
    # route that honours response_format.
    ("google/gemma-3-4b-it", "DeepInfra", "bf16"),
]


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
        default=None,
        help="Samples per task, from the start of the dataset. "
        "Omit to run the entire GSM8K test set.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Substring filter over MODELS, e.g. 'gemma-3-4b'. Omit to run all. "
        "Use this to add a model without re-running the ones already logged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv(find_dotenv(usecwd=True))

    success, logs = inspect_ai.eval_set(
        tasks=arms(args.limit),
        model=pinned_models(args.models),
        log_dir=args.logs,
        log_dir_allow_dirty=True,
    )
    scope = "full test set" if args.limit is None else f"limit={args.limit}"
    status = "complete" if success else "INCOMPLETE (some tasks still failing)"
    print(f"eval_set {status}; {len(logs)} logs in {args.logs} ({scope}).")
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
