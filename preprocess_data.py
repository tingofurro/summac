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
parser.add_argument('--data', default="xsumfaith", type=str, help='select a summarization dataset', 
                    choices=["cogensumm", "frank", 
                             "polytope", "factcc", "summeval", "xsumfaith",
                             ])
parser.add_argument('--prompt_id', default=None, type=str, help='pick a prompt template from prompt list, A or B or None')
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

    if args.data == "summeval": d['id'] = d['cnndm_id']
    if args.data == "polytope": d['id'] = d['ID']
    if args.data == "xsumfaith": d['id'] = d['bbcid']
    
    
    
    if i == 0: 
        print(f"==>> document: {d.keys} \n {d}")
        ######### this part is only for quick testing and saving
        generate_dict = {str(d['id']):None}


    if d['id'] in generate_dict.keys() and generate_dict[d['id']] is not None: 
        pass
    else: # not exist in generate_dict
        generate_dict[d['id']] = d


json_object = json.dumps(dataset, indent=4)
with open(f"data/{args.data}_data.json", "w") as outfile:
    outfile.write(json_object)



