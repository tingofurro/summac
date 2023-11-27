def summary_instructions(prompt_id: str, document: str) -> str:

    prompts = {
        "A": """Summarize in a single short paragraph the context below:

{document}

The summary is:""".format(
        document = document
        ),
        "B": """Summarize in a couple of sentences the document below:

{document}

The summary is:""".format(
            document = document
        ),
        "C": """Give me a short summary of the below:

{document}

The summary is:""".format(
            document = document
        )
    }

    return prompts[prompt_id]

def question_answering_instructions(prompt_id: str, question: str, document: str) -> str:

    prompts = {
        "A": """
    Question: {question}

Context:
{document}

Answer:""".format(
            question = question,
            document = document
        ),
        "B": """
Context:
{document}

Question:
{question}

Answer:""".format(
            question = question,
            document = document
        ),
        "C": """
Context:
{document}

Answer the following: {question}""".format(
            question = question,
            document = document
        )
    }

    return prompts[prompt_id]


TASK_INSTRUCTION = {
    'question_answering': question_answering_instructions,
    'summarization': summary_instructions,
}

__all__ = ['TASK_INSTRUCTION']