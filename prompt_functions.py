def general_prompt(promt_id: str, document: str) -> str:

    if promt_id == None or promt_id == "A":
        return """ ### Instruction:
        Your task is to summarize concisely and truthfully. Summarize the input below:

        ### Input:
        {document}

        ### Response:
        """.format(
            document = document
        )

    if promt_id == "B":
        return """ ### Instruction:
        Summarize the article below in a few sentences:

        ### Input:
        {document}

        ### Response:
        """.format(
            document = document
        )

    if promt_id == "C":
        return """ ### Instruction:
        Please write a short summary for the text below:

        ### Input:
        {document}

        ### Response:
        """.format(
            document = document
        )

# llama and falcon were instructioned tuned on other
# sets of data. So just pass the opt prompt_template here
def convert_to_llama_chat_template(messages):
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

# these now constructs the llama
# feel free to change the "content" message. I would keep this as is
def llama_prompt(prompt_id: str, document: str) -> str:

    if prompt_id == "A":
        prompt = """Your task is to summarize concisely and truthfully. Summarize the input below:
        {document}

        The summary is:""".format(
            document = document
        )

    elif prompt_id == "B":
        prompt = """Summarize the article below in a few sentences:
        {document}

        The summary is:""".format(
            document = document
        )

    else:
        prompt = """Please write a short summary for the text below:
        {document}

        The summary is:""".format(
            document = document
        )

    return convert_to_llama_chat_template(
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