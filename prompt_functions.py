from dataclasses import dataclass
# these are very similar to the ones you have hence why they work
# this is nothing more than a formated string (f'string)
# change the f'string as you please
def opt_prompt_template(document: str) -> str:

    return """ We have the following document which we must summarise.
    {document}
    The summary that covers these points is:""".format(
        document = document
    )

# llama and falcon were instructioned tuned on other
# sets of data. So just pass the opt prompt_template here
def instruction_prompt_template(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(
                f" [/INST] {message['content'].strip()}</s><s>[INST] "
            )

    return startPrompt + "".join(conversation) + endPrompt

# these now constructs the llama / dolly / falcon prompt
# feel free to change the "content" message. I would keep this as is
# and just modify the `opt_prompt_template`.
def llama_falon_prompt_template(document: str) -> str:

    prompt: str = opt_prompt_template(document)

    return instruction_prompt_template(
        messages=[
            {
                "role": "system",
                "content": """
                You are a helpful assistant that summarizes truthfully any document given.
                """,
            },
            {"role": "user", "content": prompt},
        ],
    )