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
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="facebook/opt-iml-1.3b", type=str, help='LLaMA model', 
                    choices=["decapoda-research/llama-7b-hf", "tiiuae/falcon-7b-instruct", "gpt2-medium",
                            "facebook/opt-iml-1.3b",
                            ])
parser.add_argument('--data', default="polytope", type=str, help='select a summarization dataset', 
                    choices=["cogensumm", "xsumfaith", "frank", 
                             "polytope", "factcc", "summeval",
                             ])
parser.add_argument('--seed', type=int, default=412, help='Seed for sampling the calibration data.')
parser.add_argument('--prune_method', default="fullmodel", type=str, help='if using pruned model and which to use', 
                    choices=["fullmodel", "wanda", "sparsegpt"])
args = parser.parse_args()


np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model_tokenzier(model_name):
    if args.prune_method == "fullmodel":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="llm_weights", trust_remote_code=True, device_map="auto").to(device) # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, cache_dir = "llm_weights")
    else:  
        short_name = str(args.model_name).split("/")[-1]
        model_name = f'pruned_model/{short_name}/{args.prune_method}'
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto").to(device) # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
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
dataset = benchmark_val.get_dataset(args.data) 
#  dict_keys(['filename', 'label', 'document', 'claim', 
# 'cnndm_id', 'annotations', 'dataset', 'origin'])



# summeval  dict_keys(['document', 'claim', 'label', 'model_name', 'cnndm_id', 'cut', 'annotations', 'dataset', 'origin', 'error_type'])
# factcc    dict_keys(['claim', 'label', 'filepath', 'id', 'document', 'annotations', 'dataset', 'origin'])
# polytope  dict_keys(['ID', 'document', 'claim', 'errors', 'cut', 'overall_label', 'omission_label', 'addition_label', 'duplication_label', 'inaccuracy_label', 'label', 'dataset', 'annotations', 'origin'])



for i, d in enumerate(dataset):
    
    document = d['document'] + prompt

    if args.data == "summeval": 
        d['id'] = d['cnndm_id']
        print(d['id'])

    if args.data == "polytope": d['id'] = d['ID']


    if i == 0: 
        print(d.keys())
        generate_dict = {str(d['id']):None}

    if d['id'] in generate_dict.keys() and generate_dict[d['id']] is not None:
        print(d['id'])
        print(generate_dict[d['id']])
        print(dataset[i])
        dataset[i]['prompt'] = generate_dict[d['id']]['prompt']
        dataset[i]['generated'] = generate_dict[d['id']]['generated']
        dataset[i]['rouge'] = generate_dict[d['id']]['rouge']
        dataset[i]['bertscore'] = generate_dict[d['id']]['bertscore']
        dataset[i]['summac_conv'] = generate_dict[d['id']]['summac_conv']
        dataset[i]['summac_zs'] = generate_dict[d['id']]['summac_zs']
        
    else:
        
        try:
            input_ids = tokenizer.encode(document, return_tensors="pt").to(device)
            output = model.generate(input_ids, num_return_sequences=1,
                                max_new_tokens=int(len(input_ids[0])*0.2), # min_new_tokens=10, 
                                )   # including one special token, origi len + 1
            
            if (output.shape[-1] - input_ids.shape[-1]) <= 10:
                document = f"""{document}"""
                input_ids = tokenizer.encode(document, return_tensors="pt").to(device)
                output = model.generate(input_ids, num_return_sequences=1,
                                max_new_tokens=int(len(input_ids[0])*0.2), # min_new_tokens=10, 
                                )
                output_text = tokenizer.decode(output[0][int(input_ids.shape[1]):], skip_special_tokens=True)
                print(f"==>> output_text less than 10 tokens, after process: {output_text}")

        except:
            document = f"""{document}"""
            input_ids = tokenizer.encode(document, return_tensors="pt") #.to(device)
            output = model.generate(input_ids.to(device), num_return_sequences=1,
                                max_new_tokens=int(len(input_ids[0])*0.2), # min_new_tokens=10, 
                                )   # including one special token, origi len + 1
            output_text = tokenizer.decode(output[0][int(input_ids.shape[1]):], skip_special_tokens=True)
            print(f"==>> after processed output_text: {output_text}")


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

        generate_dict[d['id']] = {"prompt": prompt, 'generated': output_text, 'rouge': score_rouge, 'bertscore': score_bertscore, 
                                        'summac_conv': score_conv["scores"][0], 'summac_zs': score_zs["scores"][0]
                                        }

short_model_name = str(args.model_name).split("/")[-1]
save_path = os.path.join("generated_output", short_model_name, args.prune_method, args.data)
print(f"==>> save_path: {save_path}")
os.makedirs(save_path, exist_ok=True)

json_object = json.dumps(dataset, indent=4)
with open(save_path + "/detailed_result.json", "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(generate_dict, indent=4)
with open(save_path + "/norepeated_result.json", "w") as outfile:
    outfile.write(json_object)



