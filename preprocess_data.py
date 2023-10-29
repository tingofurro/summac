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
parser.add_argument('--data', default="summeval", type=str, help='select a summarization dataset', 
                    choices=[ #"cogensumm", "frank", 
                             "polytope", "factcc", "summeval", "xsumfaith",
                             ])
args = parser.parse_args()



########### load dataset
benchmark_val = SummaCBenchmark(benchmark_folder="data/", cut="val") 
dataset = benchmark_val.get_dataset(args.data) 

# summeval  dict_keys(['document', 'claim', 'label', 'model_name', 'cnndm_id', 'cut', 'annotations', 'dataset', 'origin', 'error_type'])
# factcc    dict_keys(['claim', 'label', 'filepath', 'id', 'document', 'annotations', 'dataset', 'origin'])
# polytope  dict_keys(['ID', 'document', 'claim', 'errors', 'cut', 'overall_label', 'omission_label', 'addition_label', 'duplication_label', 'inaccuracy_label', 'label', 'dataset', 'annotations', 'origin'])
# xsumfaith dict_keys(['document', 'claim', 'bbcid', 'model_name', 'label', 'cut', 'annotations', 'dataset', 'origin'])


k = 0

for i, d in enumerate(dataset):

    if args.data == "summeval": temp_id = d['cnndm_id']
    if args.data == "polytope": temp_id = d['ID']
    if args.data == "xsumfaith": temp_id = d['bbcid']
    if args.data == "factcc": temp_id = d['id']
    
    if i == 0: 
        generate_dict = {f'{args.data}_{k}':d}
        temp_id_list = [temp_id]
        k += 1

        print(generate_dict)
        print(generate_dict.keys())

    if temp_id in temp_id_list: 
        pass
    else: # not exist in generate_dict
        # d['new_id'] = f"{args.data}_k"
        generate_dict[f'{args.data}_{k}'] = d
        temp_id_list.append(temp_id)
        k += 1
        

json_data = json.dumps(generate_dict, indent=4)

with open(f"data/{args.data}.json", "w") as json_file:
    json_file.write(json_data)




print(f" saving at data/{args.data}.json")