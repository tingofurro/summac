from typing import Optional
from prompts.formatters import TEMPLATE_FORMATTER
from prompts.instructions import TASK_INSTRUCTION
from prompts.roles import TASK_ROLE
import warnings


def generate_prompt(
    task: str, model: str, prompt_id: str, document: str, question: Optional[str] = None, warningon: bool = False
) -> str:

    assert task in ['question_answering', 'summarization']
    assert model in ['llama', 'mistral', 'falcon']
    assert prompt_id in ['A', 'B', 'C']

    if task == 'summarization':
        if (question is not None)and(warningon):
            warnings.warn("`question` field is not empty. Did you want to use the `question answering` task?")
        instruction = TASK_INSTRUCTION[task](
            prompt_id=prompt_id,
            document = document
        )
    else:
        assert question is not None
        instruction = TASK_INSTRUCTION[task](
            prompt_id=prompt_id,
            document = document,
            question = question
        )

    return TEMPLATE_FORMATTER[model](
        messages=[
            {
                "role": "system",
                "content": TASK_ROLE[task],
            },
            {"role": "user", "content": instruction},
        ],
    )

## Example 1
print(generate_prompt(
    task="summarization",
    model="mistral",
    prompt_id="B",
    document="I am hungry its almost lunch",
))

## Example 2
print(generate_prompt(
    task="question_answering",
    model="falcon",
    prompt_id="B",
    document="I am hungry its almost lunch",
    question="what time is it?"
))