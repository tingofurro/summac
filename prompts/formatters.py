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

def convert_to_mistral_chat_template(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"{message['content']}\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(
                f" [/INST] {message['content'].strip()}</s><s>[INST] "
            )

    return startPrompt + "".join(conversation) + endPrompt

def convert_to_falcon_chat_template(messages):
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"{message['content']}\n\n")
        else:
            conversation.append(f"{message['content']}\n")

    return  "".join(conversation)

def convert_to_opt_chat_template(messages):
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            continue
        else:
            conversation.append(f"{message['content']}\n")

    return  "".join(conversation)


TEMPLATE_FORMATTER = {
    'falcon': convert_to_falcon_chat_template,
    'opt': convert_to_opt_chat_template,
    'llama': convert_to_llama_chat_template,
    'mistral': convert_to_mistral_chat_template
}

__all__ = ['TEMPLATE_FORMATTER']