import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,load_from_disk
from evaluate import load
from summac.model_summac import SummaCZS, SummaCConv
from summac.benchmark import SummaCBenchmark
import nltk
import pandas as pd
import os
import json
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="facebook/opt-iml-1.3b", type=str, help='LLaMA model', 
                    choices=["decapoda-research/llama-7b-hf", "tiiuae/falcon-7b-instruct", "gpt2-medium"
                            "facebook/opt-iml-1.3b",
                            ])

parser.add_argument('--seed', type=int, default=412, help='Seed for sampling the calibration data.')
parser.add_argument('--prune_method', default="wanda", type=str, help='if using pruned model and which to use', 
                    choices=["fullmodel", "wanda", "sparsegpt"])
args = parser.parse_args()


np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model_tokenzier(model_name):
    if args.prune_method == "fullmodel":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="llm_weights", trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, cache_dir = "llm_weights")
    else:  
        short_name = str(args.model_name).split("/")[-1]
        model_name = f'pruned_model/{short_name}/{args.prune_method}'
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(' model size --->', trainable_params)
    return model, tokenizer


prompt = ' Please summarize the text above in one sentence. '
bertscore = load("bertscore")
rouge = load("rouge")
model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cuda") # If you have a GPU: switch to: device="cuda"-
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")



model, tokenizer = get_model_tokenzier(args.model_name)


benchmark_val = SummaCBenchmark(benchmark_folder="data/", cut="val") 
dataset = benchmark_val.get_dataset("summeval") 
#  dict_keys(['filename', 'label', 'document', 'claim', 
# 'cnndm_id', 'annotations', 'dataset', 'origin'])





results_dict = []
for i, d in enumerate(dataset):

    document = d['document'] + prompt

    input_ids = tokenizer.encode(document, return_tensors="pt").to(model.device)
    output = model.generate(input_ids, num_return_sequences=1,
                            # max_length=1024, # 1024,
                            max_new_tokens=int(len(input_ids[0])*0.2), # min_new_tokens=10, 

                            )   # including one special token, origi len + 1
    

    output_text = tokenizer.decode(output[0][int(input_ids.shape[1]):], skip_special_tokens=True)
    
    score_bertscore = bertscore.compute(predictions=[output_text], references=[d['document']], lang="en")
    score_rouge = rouge.compute(predictions=[output_text], references=[d['document']]) #, avg=True
    score_zs = model_zs.score([d['document']], [output_text])
    score_conv = model_conv.score([d['document']], [output_text])


    dataset[i]['prompt'] = prompt
    dataset[i]['generated'] = output_text
    dataset[i]['rouge'] = score_rouge
    dataset[i]['bertscore'] = score_bertscore
    dataset[i]['summac_conv'] = score_conv["scores"][0]
    dataset[i]['summac_zs'] = score_zs["scores"][0]
    

short_model_name = str(args.model_name).split("/")[-1]
save_path = os.path.join("generated_output", short_model_name, args.prune_method, args.data)
print(f"==>> save_path: {save_path}")
os.makedirs(save_path, exist_ok=True)

json_object = json.dumps(dataset, indent=4)
with open(save_path + "/detailed_result.json", "w") as outfile:
    outfile.write(json_object)
