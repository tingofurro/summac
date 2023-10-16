import numpy as np
import pandas as pd
import json


from summac.model_summac import SummaCZS, SummaCConv

model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cpu") # If you have a GPU: switch to: device="cuda"
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean")

def return_list_mean_std(score_list):
    #return "{:.3f} (".format(np.mean(score_list)), "{:.3f})".format(np.std(score_list))
    return "{:.3f}({:.3f})".format(np.mean(score_list), np.std(score_list))




def print_one_prune(data, model_name, prune_method):
    with open(f'generated_output/{model_name}/{prune_method}/{data}/norepeated_result.json') as json_file:
        data = json.load(json_file)

    rouge_list = []
    bert_list = []
    summac_conv_list = []
    summac_zs_list = []

    for k in data.keys():
        if data[k] is not None:
            rouge_list.append(data[k]['rouge']['rougeL'])
            bert_list.append(data[k]['bertscore']['f1'][0])
            summac_conv_list.append(data[k]['summac_conv'])
            summac_zs_list.append(data[k]['summac_zs'])
        # rouge_list = [i for i in rouge_list if i != 0]
        # bert_list = [i for i in bert_list if i != 0]
        # summac_conv_list = [i for i in summac_conv_list if i != 0]
        # summac_zs_list = [i for i in summac_zs_list if i != 0]

    return rouge_list, bert_list, summac_conv_list, summac_zs_list


rouge_list_final, bert_list_final, summac_conv_list_final, summac_zs_list_final = [], [], [], []


data = 'polytope'
model_name = "facebook/opt-iml-1.3b"  
# model_name = 'falcon-7b-instruct' #

for method in ['fullmodel', 'wanda', 'sparsegpt']:
    rouge_list, bert_list, summac_conv_list, summac_zs_list = print_one_prune(data, model_name, method)
    rouge_list_final.append(return_list_mean_std(rouge_list))
    bert_list_final.append(return_list_mean_std(bert_list))
    summac_conv_list_final.append(return_list_mean_std(summac_conv_list))
    summac_zs_list_final.append(return_list_mean_std(summac_zs_list))


df = pd.DataFrame(list(zip(rouge_list_final, bert_list_final, summac_conv_list_final, summac_zs_list_final)), 
               columns =['rouge', 'bert', 'summac_conv', 'summac_zs'])


df.to_csv(f'generated_output/{model_name}/{data}_result.csv', index=False)