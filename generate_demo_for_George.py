import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,load_from_disk, list_metrics # metrics_list = list_metrics()
from evaluate import load
from summac.model_summac import SummaCZS, SummaCConv
from summac.benchmark import SummaCBenchmark
import nltk
import pandas as pd
import numpy as np
import os, json, argparse, time
from prompt_functions import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="decapoda-research/llama-7b-hf", type=str, help='LLaMA model', 
                        choices=["decapoda-research/llama-7b-hf", 
                                 "decapoda-research/llama-13b-hf", 
                                 "decapoda-research/llama-30b-hf", 

                                 "tiiuae/falcon-7b-instruct", 
                                 "tiiuae/falcon-40b-instruct",

                                 "facebook/opt-iml-1.3b",
                                 "facebook/opt-iml-30b"
                                 ])
parser.add_argument('--data', default="factcc", type=str, help='select a summarization dataset', 
                    choices=["cogensumm", "xsumfaith", "frank", 
                             "polytope", "factcc", "summeval",
                             ])
parser.add_argument('--seed', type=int, default=412, help='Seed for sampling the calibration data.')
parser.add_argument('--prune_method', default="fullmodel", type=str, help='if using pruned model and which to use', 
                    choices=["fullmodel", "wanda", "sparsegpt"])
parser.add_argument('--prompt_id', default=4, type=str, help='pick a prompt template from prompt list')
args = parser.parse_args()




def get_model_tokenzier(model_name):
    if args.prune_method == "fullmodel":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="llm_weights", trust_remote_code=True, device_map="auto",
                                                     torch_dtype=torch.float16, low_cpu_mem_usage=True, 
                                                     ) 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, cache_dir = "llm_weights")
    else:  
        short_name = str(args.model).split("/")[-1]
        model_name = f'pruned_model/{short_name}/{args.prune_method}'
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto",
                                                     torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                     ) 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    return model, tokenizer



########### load prompt
if "opt" in args.model:
    from prompt_functions import opt_prompt_template as generate_prompt
    example = generate_prompt(' [[[ This is a demo document to show prompt template. ]]]]')
elif "falcon" in args.model or "llama" in args.model:
    from prompt_functions import llama_falon_prompt_template as generate_prompt
    example = generate_prompt(' [[[ This is a demo document to show prompt template. ]]]]')
else:
    print("==>> No prompt template for this model")      

print(f"==>> template example: \n {example}")
print(' ')

########### load model
model, tokenizer = get_model_tokenzier(args.model)


text = "looking after elderly parents can be difficult at the best of times .\nbut this man takes caring for his alzheimer 's - suffering mother to another level .\na security guard from china has touched hearts across the country because he takes his 84-year-old mother with him to work on the back of his motorbike every single day , reported the people 's daily online .\nlu xincai , who lives in zhejiang province in eastern china , says that he is scared his mother will get lost if he leaves her at home by herself because she suffers from the degenerative disease .\ndevoted : lu xincai takes his 84-year-old mother to work with him on the back of his motorbike every day .\nhe ties a sash around both of their waists to make sure she does n't fall off\nshe would often go up to the mountains to collect firewood and there were a few occasions when she got lost after dark .\nwhen mr lu 's father passed away earlier this year , he decided to take his mother with him to work because there was no one else who could look after her .\nhis wife works in a different city and his son is still in school .\nafter helping his mother to get up at 5 am every morning , he puts her on the back seat of his motorbike and ties a sash around both of their waists to ensure that she does not fall off .\nmr lu said that he rides the four kilometres to work slowly to make sure his mother feels safe and so that they can chat along the way .\nthe whole journey takes an hour .\neven when at work he checks up on his mother , who has been given her own room by his employers , a bank , to make sure that she has not wandered off somewhere .\nhe said that his mother devoted her life to caring for her children , and now he feels like he has a duty to care for her in return .\nvulnerable : his elderly mother suffers from alzheimer 's and used to get lost when she was left alone\nhe said : ` i was an apple in my mum 's eye , and now she 's my apple . '\n` our mother carried us on her back to the fields when she went to work on the farm and collect firewood when we were young . '\nhe added : ` only if i see her will i feel relaxed .\notherwise i would be afraid is she had wandered away . '",
document = generate_prompt(text)

original_len = len(tokenizer.encode(document, return_tensors="pt")[0])
generate_max_new_tokens = int(original_len*0.25)

input_ids = tokenizer.encode(document, return_tensors="pt")
output = model.generate(input_ids, num_return_sequences=1,
                            max_new_tokens=generate_max_new_tokens, 
                            #device_map = "auto",
                            )   # including one special token, origi len + 1

output_text = tokenizer.decode(output[0][int(input_ids.shape[1]):], skip_special_tokens=True)
print(f"==>> output_text: {output_text}")
