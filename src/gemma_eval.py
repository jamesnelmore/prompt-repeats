"""Run the CoT vs no-CoT comparison across three sizes of Gemma via OpenRouter.

Builds the two task arms (CoT and no-CoT) and crosses them with each model.
Inspect computes the {task} x {model} matrix and schedules it, so this queues
2 tasks x 3 models = 6 effective evals and lets Inspect handle concurrency.

Run with:  python src/gemma_eval.py
Re-running resumes unfinished tasks from log_dir instead of redoing them.
"""

from inspect_ai import eval_set

from task import prompt_repeat_comparison

# The two arms. Inspect crosses these with every model below.
TASKS = [
    prompt_repeat_comparison(use_cot=True),
    prompt_repeat_comparison(use_cot=False, prompt_repeats=1),
    prompt_repeat_comparison(use_cot=False, prompt_repeats=2),
    prompt_repeat_comparison(use_cot=False, prompt_repeats=4),
    prompt_repeat_comparison(use_cot=False, prompt_repeats=8),
]

# Three sizes of Gemma 3. Gemma is not a reasoning model, so the no-CoT arm
# relies on the prompt template + `reasoning_effort="none"` baked into the
# task; there are no hidden reasoning tokens to disable here.
MODELS = [
    "openrouter/google/gemma-3-4b-it",
    "openrouter/google/gemma-3-12b-it",
    "openrouter/google/gemma-3-27b-it",
]


def main() -> None:
    success, logs = eval_set(
        tasks=TASKS,
        model=MODELS,
        log_dir="logs/gemma-repeats",
        limit=500,
    )
    print(f"eval_set complete (success={success}); {len(logs)} logs written.")


if __name__ == "__main__":
    main()
