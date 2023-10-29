
from transformers import AutoModelForCausalLM, AutoTokenizer




def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt

messages = [
  { "role": "system","content": "You are a helpful assistant that summarizes truthfully and concisely any document given."}
]

# define question and add to messages


text = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions.
One possible site, known as Arcadia Planitia, is covered instrange sinuous features.
The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice.
Arcadia Planitia is in Mars' northern lowlands."""


instruction = """ We have the following document which we must summarise.
{document}
The short summary that covers these points is:""".format(
    document = text
)

messages.append({"role": "user", "content": instruction})
prompt = build_llama2_prompt(messages)
print(f"==>> prompt: {prompt}")

model_name = "facebook/opt-iml-1.3b"
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="llm_weights", trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, cache_dir = "llm_weights")


input_ids = tokenizer.encode(prompt, return_tensors="pt") #.to(device)
output = model.generate(input_ids.to(model.device), num_return_sequences=1,
                                max_new_tokens=int(len(input_ids[0])*0.2), # min_new_tokens=10, 
                                )   # including one special token, origi len + 1
output_text = tokenizer.decode(output[0][int(input_ids.shape[1]):], skip_special_tokens=True)
print(f"==>> output_text: {output_text}")
            