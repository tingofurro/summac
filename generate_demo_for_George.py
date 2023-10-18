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



np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
#device = "cuda" if torch.cuda.is_available() else "cpu"


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
    print(' ')
    
elif "falcon" in args.model or "llama" in args.model:
    from prompt_functions import llama_falon_prompt_template as generate_prompt
    example = generate_prompt(' [[[ This is a demo document to show prompt template. ]]]]')
    print(' ')
    
else:
    print("==>> No prompt template for this model")      

print(f"==>> example: \n {example}")
########### load metrics
harim = load("NCSOFT/harim_plus")  #  using model : facebook/bart-large-cnn
rouge = load("rouge")
bertscore = load("bertscore")
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




for i, d in enumerate(dataset[:5]):
    document = generate_prompt(d['document'])

    if args.data == "summeval": d['id'] = d['cnndm_id']
    if args.data == "polytope": d['id'] = d['ID']

    if i == 0: 
        print(d.keys())
        generate_dict = {str(d['id']):None}

    if d['id'] in generate_dict.keys() and generate_dict[d['id']] is not None:
        dataset[i]['prompt'] = generate_dict[d['id']]['prompt']
        dataset[i]['generated'] = generate_dict[d['id']]['generated']
        dataset[i]['rouge'] = generate_dict[d['id']]['rouge']
        dataset[i]['bertscore'] = generate_dict[d['id']]['bertscore']
        dataset[i]['harim'] = generate_dict[d['id']]['harim']
        dataset[i]['summac_conv'] = generate_dict[d['id']]['summac_conv']
        dataset[i]['summac_zs'] = generate_dict[d['id']]['summac_zs']
        
    else:
        original_len = len(tokenizer.encode(document, return_tensors="pt")[0])
        generate_max_new_tokens = int(original_len*0.25)
        try:
            input_ids = tokenizer.encode(document, return_tensors="pt")
            output = model.generate(input_ids, num_return_sequences=1,
                                    max_new_tokens=generate_max_new_tokens, 
                                    device = "auto",
                                    )   # including one special token, origi len + 1

        except:
            document = f"""{document}"""
            input_ids = tokenizer.encode(document, return_tensors="pt") 
            output = model.generate(input_ids, num_return_sequences=1,
                                max_new_tokens=generate_max_new_tokens, 
                                #device = "auto",
                                )   # including one special token, origi len + 1
            output_text = tokenizer.decode(output[0][int(input_ids.shape[1]):], skip_special_tokens=True)
            print(f"==>> after processed output_text: {output_text}")


        output_text = tokenizer.decode(output[0][int(input_ids.shape[1]):], skip_special_tokens=True)

        score_harim = harim.compute(predictions=[output_text], references=[d['document']])
        score_rouge = rouge.compute(predictions=[output_text], references=[d['document']]) #, avg=True
        score_bertscore = bertscore.compute(predictions=[output_text], references=[d['document']], lang="en")
        score_zs = model_zs.score([d['document']], [output_text])
        score_conv = model_conv.score([d['document']], [output_text])

        dataset[i]['document'] = d['document']
        dataset[i]['generated'] = output_text
        dataset[i]['rouge'] = score_rouge
        dataset[i]['bertscore'] = score_bertscore
        dataset[i]['harim'] = score_harim
        dataset[i]['summac_conv'] = score_conv["scores"][0]
        dataset[i]['summac_zs'] = score_zs["scores"][0]

        generate_dict[d['id']] = {"document": d['document'], "prompt": args.prompt_id, 'generated': output_text, 'rouge': score_rouge, 'bertscore': score_bertscore, 'harim': score_harim, 
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



