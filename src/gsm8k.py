"""GSM8K prompt, data, and answer-format protocol."""

import re

from datasets import load_dataset

DATASET_PATH = "openai/gsm8k"
GSM8K_DATASET_REVISION = "cc7b047b6e5bb11b4f1af84efc572db110a51b3c"

COT_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()

NO_COT_PROMPT_TEMPLATE = """
Solve the following math problem by directly outputting the answer. Your entire response must be a JSON object of the form {"answer": N} where N is a single integer. DO NOT reason before outputting the answer, output only that JSON object and nothing else.

{prompt}
""".strip()

ANSWER_PREFIX = '{"answer": '
ANSWER_CLOSE = "}"
ANSWER_PATTERN = re.compile(r'"answer":\s*(-?\d+)')


def normalize_answer(answer: str) -> str:
    """Normalize GSM8K's comma-formatted integer references."""
    return answer.replace(",", "")


def template_with_copies(template: str, copies: int) -> str:
    """Put `copies` copies of `{prompt}` into a prompt template."""
    assert copies >= 1, "copies must be at least 1"
    return template.replace("{prompt}", "\n\n".join(["{prompt}"] * copies))


def no_cot_prompt(question: str, copies: int) -> str:
    """Build the shared no-CoT user message with repeated question text."""
    return template_with_copies(NO_COT_PROMPT_TEMPLATE, copies).format(prompt=question)


def get_questions(count: int | None) -> list[tuple[str, str]]:
    """Return `(question, reference_answer)` from pinned GSM8K test rows."""
    dataset = load_dataset(
        DATASET_PATH, "main", split="test", revision=GSM8K_DATASET_REVISION
    )
    return [
        (
            dataset[i]["question"],
            normalize_answer(dataset[i]["answer"].split("####")[-1].strip()),
        )
        for i in range(len(dataset) if count is None else count)
    ]


def scored(completion: str, reference: str) -> bool:
    """Whether a JSON completion's integer answer matches the reference."""
    found = ANSWER_PATTERN.search(completion)
    return found is not None and found.group(1) == reference
