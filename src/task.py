from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import match
from inspect_ai.solver import generate, prompt_template
from inspect_evals.gsm8k.gsm8k import record_to_sample
from inspect_evals.utils.huggingface import hf_dataset

DATASET_PATH = "openai/gsm8k"
GSM8K_DATASET_REVISION = "cc7b047b6e5bb11b4f1af84efc572db110a51b3c"

COT_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

# TODO double check gsm8k answers are only numbers.
# TODO prefill ANSWER: in solver
NO_COT_PROMPT_TEMPLATE = """
Solve the following math problem by directly outputting the answer. Your entire response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem. $ANSWER must be a single number. DO NOT reason before outputting the answer, output only your final response and nothing else.

{prompt}

ANSWER:
""".strip()


@task
def prompt_repeat_comparision(
    use_cot: bool = False,
    prompt_repeats: int = 1,
) -> Task:
    """Inspect Task Definition for no chain-of-thought recovery benchmark."""
    # TODO write out argument and return definitions
    # Add fewshot if necessary
    # Write custom solver to verify lack of CoT

    prompt = COT_PROMPT_TEMPLATE if use_cot else NO_COT_PROMPT_TEMPLATE

    solver = [prompt_template(prompt), generate()]

    dataset = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="test",
        sample_fields=record_to_sample,
        revision=GSM8K_DATASET_REVISION,
    )
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=match(numeric=True),
        # Distinct name per arm so the "task" column shows the CoT setting
        # directly (instead of "cot_comparision" for both arms).
        name=f"gsm8k_{'cot' if use_cot else 'nocot'}",
        display_name=f"GSM8K ({'CoT' if use_cot else 'no CoT'})",
        config=GenerateConfig(
            reasoning_effort=None if use_cot else "none",
        ),
    )
