from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.scorer import match, pattern
from inspect_ai.solver import assistant_message, generate, prompt_template
from inspect_ai.util import JSONSchema
from inspect_evals.gsm8k.gsm8k import record_to_sample
from inspect_evals.utils.huggingface import hf_dataset

DATASET_PATH = "openai/gsm8k"
GSM8K_DATASET_REVISION = "cc7b047b6e5bb11b4f1af84efc572db110a51b3c"

COT_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()

# TODO double check gsm8k answers are only numbers.
NO_COT_PROMPT_TEMPLATE = """
Solve the following math problem by directly outputting the answer. Your entire response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem. $ANSWER must be a single number. DO NOT reason before outputting the answer, output only your final response and nothing else.

{prompt}
""".strip()

# Sent as a trailing assistant turn rather than as part of the user message, so
# the model continues from it instead of being asked to imitate it.
COT_PREFILL = "Reasoning:"

# What actually suppresses reasoning in the no-CoT arm. Prompting alone leaked
# reasoning on 22% of samples (gemma-4-26b/NextBit, n=100); constrained decoding
# leaked on 0%, because the grammar admits no token sequence that reasons first.
# `integer` (not `number`) so the grammar also rules out "18.0", which keeps the
# scorer to an exact regex. NOTE: this only suppresses reasoning on models with
# no separate reasoning channel -- providers generally exempt that channel from
# the grammar, so a thinking model would reason freely and still emit valid JSON.
ANSWER_SCHEMA = ResponseSchema(
    name="gsm8k_answer",
    json_schema=JSONSchema(
        type="object",
        properties={"answer": JSONSchema(type="integer")},
        required=["answer"],
        additionalProperties=False,
    ),
)


@task
def prompt_repeat_comparison(
    use_cot: bool = False,
    prompt_repeats: int = 1,
    no_cot_max_tokens: int = 30,
    limit: int | None = None,
) -> Task:
    """Inspect Task Definition for no chain-of-thought recovery benchmark.

    Args:
        use_cot: Run the with-CoT ceiling arm instead of the no-CoT arm.
        prompt_repeats: How many times to repeat the question in the prompt.
        no_cot_max_tokens: Output cap for the no-CoT arm (must fit the JSON
            wrapper, not just the digits).
        limit: Samples from the start of the dataset; None runs the whole test
            set. This is deliberately a *task arg* rather than eval_set's own
            `limit=`: eval_set hashes task args into the task identity but not
            its limit, so a task run at limit=100 and then at limit=500 would
            otherwise look like completed work and be skipped.
    """

    assert prompt_repeats >= 1, "Prompt repeat count must be at least 1"

    template = COT_PROMPT_TEMPLATE if use_cot else NO_COT_PROMPT_TEMPLATE

    if prompt_repeats > 1:
        repeated_placeholder = "\n\n".join(["{prompt}"] * prompt_repeats)
        template = template.replace("{prompt}", repeated_placeholder)

    # The no-CoT arm is constrained by ANSWER_SCHEMA instead, which forces the
    # first token to "{" and so needs no prefill.
    prefill = [assistant_message(COT_PREFILL)] if use_cot else []
    solver = [prompt_template(template), *prefill, generate()]

    dataset = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="test",
        sample_fields=record_to_sample,
        revision=GSM8K_DATASET_REVISION,
    )
    if limit is not None:
        dataset = dataset[:limit]

    # The no-CoT arm answers as JSON, which match() cannot read.
    scorers = [match(numeric=True)] if use_cot else [pattern(r'"answer":\s*(-?\d+)')]

    name = f"gsm8k_{'cot' if use_cot else 'nocot'}" + (
        f"_repeat{prompt_repeats}" if prompt_repeats > 1 else ""
    )

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorers,
        # Distinct name per arm so the "task" column shows the CoT setting
        # directly (instead of "cot_comparision" for both arms).
        name=name,
        display_name=f"GSM8K ({'CoT' if use_cot else 'no CoT'})",
        config=GenerateConfig(
            reasoning_effort=None if use_cot else "none",
            max_tokens=None if use_cot else no_cot_max_tokens,
            response_schema=None if use_cot else ANSWER_SCHEMA,
        ),
    )
