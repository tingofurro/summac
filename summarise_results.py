import numpy as np
import pandas as pd
import json
from summac.model_summac import SummaCZS, SummaCConv


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
parser.add_argument('--prune_method', default="fullmodel", type=str, help='if using pruned model and which to use', 
                    choices=["fullmodel", "wanda", "sparsegpt"])
parser.add_argument('--prompt_id', default=1, type=int, help='pick a prompt template from prompt list')
args = parser.parse_args()


def return_list_mean_std(score_list):
    #return "{:.3f} (".format(np.mean(score_list)), "{:.3f})".format(np.std(score_list))
    return "{:.3f}({:.3f})".format(np.mean(score_list), np.std(score_list))


def print_one_prune(data, model_name, prune_method):
    with open(f"generated_output/falcon-7b-instruct/{prune_method}/{data}/norepeated_result_prompt1.json") as json_file:
        data = json.load(json_file)

    rouge_list,bert_list, harim_list, summac_conv_list, summac_zs_list = [], [], [], [], []

    for k in data.keys():
        if data[k] is not None:
            rouge_list.append(data[k]['rouge']['rougeL'])
            bert_list.append(data[k]['bertscore']['f1'][0])
            harim_list.append(data[k]['harim']['f1'][0])
            summac_conv_list.append(data[k]['summac_conv'])
            summac_zs_list.append(data[k]['summac_zs'])
        # rouge_list = [i for i in rouge_list if i != 0]
        # bert_list = [i for i in bert_list if i != 0]
        # summac_conv_list = [i for i in summac_conv_list if i != 0]
        # summac_zs_list = [i for i in summac_zs_list if i != 0]

    return rouge_list, bert_list, harim_list, summac_conv_list, summac_zs_list


rouge_list_final, bert_list_final, harim_list_final, summac_conv_list_final, summac_zs_list_final = [], [], [], [], []




for method in ['fullmodel', 'wanda', 'sparsegpt']:
    rouge_list, bert_list, harim_list, summac_conv_list, summac_zs_list = print_one_prune(args.data, args.model, method)
    rouge_list_final.append(return_list_mean_std(rouge_list))
    bert_list_final.append(return_list_mean_std(bert_list))
    harim_list_final.append(return_list_mean_std(harim_list))
    summac_conv_list_final.append(return_list_mean_std(summac_conv_list))
    summac_zs_list_final.append(return_list_mean_std(summac_zs_list))


df = pd.DataFrame(list(zip(rouge_list_final, bert_list_final, harim_list, summac_conv_list_final, summac_zs_list_final)), 
               columns =['RougeL', 'BERTScore', 'HaRiM', 'SUMMAC_conv', 'SUMMAC_zs'])


df.to_csv(f'generated_output/{args.model_name}/{args.data}_result.csv', index=False)