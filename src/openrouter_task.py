"""Inspect task for the OpenRouter GSM8K prompt-repeat experiment."""

from typing import Any, cast

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.scorer import match, pattern
from inspect_ai.solver import assistant_message, generate, prompt_template
from inspect_ai.util import JSONSchema
from inspect_evals.utils.deps_utils import create_stable_id
from inspect_evals.utils.huggingface import hf_dataset

from gsm8k import (
    ANSWER_PATTERN,
    COT_PROMPT_TEMPLATE,
    DATASET_PATH,
    GSM8K_DATASET_REVISION,
    NO_COT_PROMPT_TEMPLATE,
    normalize_answer,
    template_with_copies,
)

COT_PREFILL = "Reasoning:"
ANSWER_SCHEMA = ResponseSchema(
    name="gsm8k_answer",
    json_schema=JSONSchema(
        type="object",
        properties={"answer": JSONSchema(type="integer")},
        required=["answer"],
        additionalProperties=False,
    ),
)


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Build a Sample directly, normalizing GSM8K's comma-formatted target."""
    question = cast(str, record["question"])
    reasoning, answer = cast(str, record["answer"]).rsplit("####", maxsplit=1)
    return Sample(
        id=create_stable_id(question, prefix="gsm8k"),
        input=question,
        target=normalize_answer(answer.strip()),
        metadata={"reasoning": reasoning.strip()},
    )


@task
def prompt_repeat_comparison(
    use_cot: bool = False,
    prompt_copies: int = 1,
    no_cot_max_tokens: int = 30,
    limit: int | None = None,
) -> Task:
    """GSM8K prompt-repeat arm: no-CoT comparison or CoT (optionally with copies)."""
    template = template_with_copies(
        COT_PROMPT_TEMPLATE if use_cot else NO_COT_PROMPT_TEMPLATE, prompt_copies
    )
    prefill = [assistant_message(COT_PREFILL)] if use_cot else []
    dataset = hf_dataset(
        path=DATASET_PATH,
        data_dir="main",
        split="test",
        sample_fields=record_to_sample,
        revision=GSM8K_DATASET_REVISION,
    )
    if limit is not None:
        dataset = dataset[:limit]

    return Task(
        dataset=dataset,
        solver=[prompt_template(template), *prefill, generate()],
        scorer=[match(numeric=True)] if use_cot else [pattern(ANSWER_PATTERN.pattern)],
        name=f"gsm8k_{'cot' if use_cot else 'nocot'}"
        + (f"_copies{prompt_copies}" if prompt_copies > 1 else ""),
        display_name=(
            f"GSM8K ({'CoT' if use_cot else 'no CoT'}"
            + (f", {prompt_copies} copies" if prompt_copies > 1 else "")
            + ")"
        ),
        config=GenerateConfig(
            reasoning_effort=None if use_cot else "none",
            max_tokens=None if use_cot else no_cot_max_tokens,
            response_schema=None if use_cot else ANSWER_SCHEMA,
        ),
    )
