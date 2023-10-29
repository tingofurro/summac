import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt_functions import general_prompt as generate_prompt

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", cache_dir="llm_weights")


def get_model_tokenzier(model_name, prune_method="fullmodel"):
    if prune_method == "fullmodel":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="llm_weights", trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, cache_dir = "llm_weights")
    else:  
        short_name = str(model_name).split("/")[-1]
        model_name = f'pruned_model/{short_name}/{args.prune_method}'
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, tokenizer


text = "Paul Merson has restarted his row with Andros Townsend after the Tottenham midfielder was brought on with only seven minutes remaining in his team's 0-0 draw with Burnley on Sunday. 'Just been watching the game, did you miss the coach? #RubberDub #7minutes,' Merson put on Twitter. Merson initially angered Townsend for writing in his Sky Sports column that 'if Andros Townsend can get in (the England team) then it opens it up to anybody.' Paul Merson had another dig at Andros Townsend after his appearance for Tottenham against Burnley . Townsend was brought on in the 83rd minute for Tottenham as they drew 0-0 against Burnley . Andros Townsend scores England's equaliser in their 1-1 friendly draw with Italy in Turin on Tuesday night . The former Arsenal man was proven wrong when Townsend hit a stunning equaliser for England against Italy and he duly admitted his mistake. 'It's not as though I was watching hoping he wouldn't score for England, I'm genuinely pleased for him and fair play to him \\u2013 it was a great goal,' Merson said. 'It's just a matter of opinion, and my opinion was that he got pulled off after half an hour at Manchester United in front of Roy Hodgson, so he shouldn't have been in the squad. 'When I'm wrong, I hold my hands up. I don't have a problem with doing that - I'll always be the first to admit when I'm wrong.' Townsend hit back at Merson on Twitter after scoring for England against Italy . Sky Sports pundit  Merson (centre) criticised Townsend's call-up to the England squad last week . Townsend hit back at Merson after netting for England in Turin on Wednesday, saying 'Not bad for a player that should be 'nowhere near the squad' ay @PaulMerse?' Any bad feeling between the pair seemed to have passed but Merson was unable to resist having another dig at Townsend after Tottenham drew at Turf Moor"

prompt_id="C"
document = generate_prompt(prompt_id, text)

print(f"==>> document: {document}")

def demo(model_name, document):


    print(' ')
    print("".center(50, "-"))
    print("".center(50, "-"))
    print(model_name)
    print("".center(50, "-"))
    
    model, tokenizer = get_model_tokenzier(model_name)
    original_len = len(tokenizer.encode(document, return_tensors="pt")[0])
    print(f"==>> original_len: \n {original_len}")
    generate_max_new_tokens = int(original_len*0.25)
    input_ids = tokenizer.encode(document, return_tensors="pt") 
    output = model.generate(input_ids.to(model.device), num_return_sequences=1,
                                    max_new_tokens=generate_max_new_tokens, 
                                    #device = "auto",
                                    )   # including one special token, origi len + 1
    output_text = tokenizer.decode(output[0][int(input_ids.shape[1]):], skip_special_tokens=True)
    print(f"==>> output: \n {output_text}")
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"==>> full output: \n {output_text}")
        
demo("llama-2-7b", document)
demo("NousResearch/Llama-2-7b-hf", document)