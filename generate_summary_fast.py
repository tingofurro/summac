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

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="tiiuae/falcon-7b-instruct", type=str, help='LLaMA model', 
                        choices=["decapoda-research/llama-7b-hf", 
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
parser.add_argument('--prompt_id', default=1, type=int, help='pick a prompt template from prompt list')
args = parser.parse_args()



np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model_tokenzier(model_name):
    if args.prune_method == "fullmodel":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="llm_weights", trust_remote_code=True, device_map="auto").to(device) # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, cache_dir = "llm_weights")
    else:  
        short_name = str(args.model).split("/")[-1]
        model_name = f'pruned_model/{short_name}/{args.prune_method}'
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto").to(device) # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f' model {str(args.model)} size --->', trainable_params)
    return model, tokenizer



########### load prompt
try:
    with open('./generated_output/prompt_list.json', 'r') as file:
        prompt_list = json.load(file)
except json.decoder.JSONDecodeError as e:
    print(f"Error loading JSON: {e}")
    prompt_list = {}  # Assign an empty dictionary as a default value or handle the error accordingly

prompt = prompt_list[f"prompt_{str(args.prompt_id)}"]["prompt"]
if isinstance(prompt, list): multipart_prompt = True
else: multipart_prompt = False


########### load metrics
harim = load("NCSOFT/harim_plus")
rouge = load("rouge")
model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cuda") # If you have a GPU: switch to: device="cuda"-
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")


########### load model
model, tokenizer = get_model_tokenzier(args.model)



########### load dataset
benchmark_val = SummaCBenchmark(benchmark_folder="data/", cut="val") 
dataset = benchmark_val.get_dataset(args.data) 
# summeval  dict_keys(['document', 'claim', 'label', 'model_name', 'cnndm_id', 'cut', 'annotations', 'dataset', 'origin', 'error_type'])
# factcc    dict_keys(['claim', 'label', 'filepath', 'id', 'document', 'annotations', 'dataset', 'origin'])
# polytope  dict_keys(['ID', 'document', 'claim', 'errors', 'cut', 'overall_label', 'omission_label', 'addition_label', 'duplication_label', 'inaccuracy_label', 'label', 'dataset', 'annotations', 'origin'])




for i, d in enumerate(dataset):
    if multipart_prompt: document = str(prompt[0]) + d['document'] + str(prompt[1])
    elif prompt_list[f"prompt_{str(args.prompt_id)}"]["document_before_prompt"]: document = d['document'] + prompt
    else: document = prompt + d['document']

    if args.data == "summeval": d['id'] = d['cnndm_id']
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
                                max_new_tokens=int(len(input_ids[0])*0.25), # min_new_tokens=10, 
                                )   # including one special token, origi len + 1

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

        dataset[i]['document'] = d['document']
        dataset[i]['generated'] = output_text
        dataset[i]['rouge'] = score_rouge
        dataset[i]['bertscore'] = score_bertscore
        dataset[i]['summac_conv'] = score_conv["scores"][0]
        dataset[i]['summac_zs'] = score_zs["scores"][0]

        generate_dict[d['id']] = {"document": d['document'], "prompt": prompt, 'generated': output_text, 'rouge': score_rouge, 'bertscore': score_bertscore, 
                                 'summac_conv': score_conv["scores"][0], 'summac_zs': score_zs["scores"][0]
                                 }



short_model_name = str(args.model).split("/")[-1]
save_path = os.path.join("generated_output", short_model_name, args.prune_method, args.data)
print(f"==>> save_path: {save_path}")
os.makedirs(save_path, exist_ok=True)

json_object = json.dumps(dataset, indent=4)
with open(save_path + f"/detailed_result_prompt{str(args.prompt_id)}.json", "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(generate_dict, indent=4)
with open(save_path + f"/norepeated_result_prompt{str(args.prompt_id)}.json", "w") as outfile:
    outfile.write(json_object)



