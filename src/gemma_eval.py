"""Run the CoT vs no-CoT comparison across three sizes of Gemma via OpenRouter.

Builds the two task arms (CoT and no-CoT) and crosses them with each model.
Inspect computes the {task} x {model} matrix and schedules it, so this queues
2 tasks x 3 models = 6 effective evals and lets Inspect handle concurrency.

Run with:  python src/gemma_eval.py
Re-running resumes unfinished tasks from log_dir instead of redoing them.
"""

from inspect_ai import eval_set

from task import cot_comparision

# The two arms. Inspect crosses these with every model below.
TASKS = [
    cot_comparision(use_cot=True),
    cot_comparision(use_cot=False),
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
        log_dir="logs/gemma-cot2",  # required; enables retry & resume
        max_tasks=6,  # 2 tasks x 3 models = 6 effective tasks
        limit=50,
    )
    print(f"eval_set complete (success={success}); {len(logs)} logs written.")


if __name__ == "__main__":
    main()
