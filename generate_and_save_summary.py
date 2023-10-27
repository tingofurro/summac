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
parser.add_argument('--model', default="tiiuae/falcon-7b-instruct", type=str, help='LLaMA model', 
                        choices=[
                            #   "decapoda-research/llama-7b-hf", 

                                 "tiiuae/falcon-7b-instruct", 
                                 "tiiuae/falcon-40b-instruct",

                                 "meta-llama/Llama-2-7b-chat-hf",
                                 "meta-llama/Llama-2-13b-chat-hf",

                                #  "facebook/opt-iml-1.3b",
                                #  "facebook/opt-iml-30b",

                                #  "NousResearch/Nous-Hermes-llama-2-7b",
                                #  "NousResearch/Nous-Hermes-Llama2-13b"
                                 ])
parser.add_argument('--data', default="xsumfaith", type=str, help='select a summarization dataset', 
                    choices=[ #"cogensumm", "frank", 
                             "polytope", "factcc", "summeval", "xsumfaith",
                             ])
parser.add_argument('--seed', type=int, default=412, help='Seed for sampling the calibration data.')
parser.add_argument('--prune_method', default="fullmodel", type=str, help='if using pruned model and which to use', 
                    choices=["fullmodel", "wanda", "sparsegpt", "magnitude"])
parser.add_argument('--prompt_id', default="A", type=str, 
                    choices=["A", "B", "C"],
                    help='pick a prompt template from prompt list, A or B or None')
args = parser.parse_args()



np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

short_model_name = str(args.model).split("/")[-1]
save_path = os.path.join("generated_output", short_model_name, args.prune_method, args.data)
os.makedirs(save_path, exist_ok=True)




def get_model_tokenzier(model_name):
    if args.prune_method == "fullmodel":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="llm_weights", trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, cache_dir = "llm_weights")
    else:  
        short_name = str(args.model).split("/")[-1]
        model_name = f'pruned_model/{short_name}/{args.prune_method}'
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    #trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, tokenizer



########### load prompt
if 'falcon' in str(args.model).lower(): from prompt_functions import falcon_prompt as generate_prompt
elif 'llama' in str(args.model).lower(): from prompt_functions import llama_prompt as generate_prompt
else: print(' no prompt generator found')




########### load metrics
harim = load("NCSOFT/harim_plus")  #  using model : facebook/bart-large-cnn
rouge = load("rouge")
bertscore = load("bertscore")
model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cuda") # If you have a GPU: switch to: device="cuda"-
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")


########### load model
model, tokenizer = get_model_tokenzier(args.model)


########### load dataset
# benchmark_val = SummaCBenchmark(benchmark_folder="data/", cut="val") 
# dataset = benchmark_val.get_dataset(args.data) 
f = open(f"data/{args.data}.json")
dataset = json.load(f)
# summeval  dict_keys(['document', 'claim', 'label', 'model_name', 'cnndm_id', 'cut', 'annotations', 'dataset', 'origin', 'error_type'])
# factcc    dict_keys(['claim', 'label', 'filepath', 'id', 'document', 'annotations', 'dataset', 'origin'])
# polytope  dict_keys(['ID', 'document', 'claim', 'errors', 'cut', 'overall_label', 'omission_label', 'addition_label', 'duplication_label', 'inaccuracy_label', 'label', 'dataset', 'annotations', 'origin'])
# xsumfaith dict_keys(['document', 'claim', 'bbcid', 'model_name', 'label', 'cut', 'annotations', 'dataset', 'origin'])

key_list = list(dataset.keys())

for i, key in enumerate(key_list):
    # prepare full input based on prompt template
    if i == 0: 
        try: 
            with open(save_path + f"/prompt_{args.prompt_id}_raw_result.json", "r+") as json_file:
                generate_dict = json.load(json_file)
                print("".center(50, "-"))
                print(' countinue from last time')
        except: generate_dict = dataset.copy()

        print("\n \n", generate_prompt(args.prompt_id, dataset[key]['document']))


    if 'generated' in [generate_dict[key].keys()]:
        pass
    else: # not exist in generate_dict
        document = generate_prompt(args.prompt_id, dataset[key]['document'])
        #character_len = len(dataset[key]['document'])

        original_len = len(tokenizer.encode(document, return_tensors="pt")[0])
        generate_max_new_tokens = int(original_len*0.25)
        input_ids = tokenizer.encode(document, return_tensors="pt") 
        output = model.generate(input_ids.to(model.device), num_return_sequences=1,
                                max_new_tokens=generate_max_new_tokens, 
                                #device = "auto",
                                )   # including one special token, origi len + 1
        output_text = tokenizer.decode(output[0][int(input_ids.shape[1]):], skip_special_tokens=True)
        

        score_harim = harim.compute(predictions=[output_text], references=[document])
        score_rouge = rouge.compute(predictions=[output_text], references=[document]) #, avg=True
        score_bertscore = bertscore.compute(predictions=[output_text], references=[document], lang="en")
        score_zs = model_zs.score([document], [output_text])
        score_conv = model_conv.score([document], [output_text])


        generate_dict[key]['generated'] = output_text
        generate_dict[key]['rouge'] = score_rouge
        generate_dict[key]['bertscore'] = score_bertscore
        generate_dict[key]['harim'] = score_harim
        generate_dict[key]['summac_conv'] = score_conv["scores"][0]
        generate_dict[key]['summac_zs'] = score_zs["scores"][0]


        ######### this part is only for quick testing and saving
        json_object = json.dumps(generate_dict, indent=4)
        with open(save_path + f"/prompt_{args.prompt_id}_raw_result.json", "w") as outfile:
            outfile.write(json_object)
            outfile.close()
        ######### this part is only for quick testing and saving


json_object = json.dumps(generate_dict, indent=4)
with open(save_path + f"/prompt_{args.prompt_id}_raw_result.json", "w") as outfile:
    outfile.write(json_object)
    outfile.close()