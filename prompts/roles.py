def master_role_summary() -> str:
    return "You are a helpful assistant that summarizes truthfully any document given."

def master_role_qa() -> str:
    return "You are a helpful assistant that answers truthfully any question based on the context given."


TASK_ROLE = {
    'question_answering': master_role_qa(),
    'summarization': master_role_summary(),
}

__all__ = ['TASK_ROLE']