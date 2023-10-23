import numpy as np
import pandas as pd
import json, argparse
from summac.model_summac import SummaCZS, SummaCConv



def return_list_mean_std(score_list):
    #return "{:.3f} (".format(np.mean(score_list)), "{:.3f})".format(np.std(score_list))
    return "{:.3f} ({:.3f})".format(np.mean(score_list), np.std(score_list))


parser = argparse.ArgumentParser()
parser.add_argument('--model', default="NousResearch/Nous-Hermes-Llama2-13b", type=str, help='LLaMA model', 
                        choices=[
                                 "tiiuae/falcon-7b-instruct", 

                                 "NousResearch/Nous-Hermes-llama-2-7b",
                                 "NousResearch/Nous-Hermes-Llama2-13b"
                                 ])
parser.add_argument('--data', default="factcc", type=str, help='select a summarization dataset', 
                    choices=["cogensumm", "xsumfaith", "frank", 
                             "polytope", "factcc", "summeval",
                             ])
parser.add_argument('--prune_method', default="fullmodel", type=str, help='if using pruned model and which to use', 
                    choices=["fullmodel", "wanda", "sparsegpt"])
parser.add_argument('--prompt_id', default=1, type=int, help='pick a prompt template from prompt list')
args = parser.parse_args()





def print_one_prune(data, model_name, prune_method):
    with open(f"generated_output/{model_name}/{prune_method}/{data}/norepeated_result_promptNone.json") as json_file:
        data = json.load(json_file)

    rouge_list, bert_list, harim_list, summac_conv_list, summac_zs_list = [], [], [], [], []
    for k in data.keys():
        #print(data[k].keys())
        if data[k] is not None:
            rouge_list.append(data[k]['rouge']['rougeL'])
            bert_list.append(data[k]['bertscore']['f1'][0])
            harim_list.append(data[k]['harim'][0])
            summac_conv_list.append(data[k]['summac_conv'])
            summac_zs_list.append(data[k]['summac_zs'])

    return rouge_list, bert_list, harim_list, summac_conv_list, summac_zs_list
   
rouge_list_final, bert_list_final, harim_list_final, summac_conv_list_final, summac_zs_list_final = [], [], [], [], []


data_loop_list = ["polytope", "factcc", "summeval"]
method_loop_list = ['fullmodel', 'wanda', 'sparsegpt']
model_loop_list = [
                "NousResearch/Nous-Hermes-llama-2-7b", 
                "NousResearch/Nous-Hermes-Llama2-13b", 
                "tiiuae/falcon-7b-instruct",
                ]

data_list, method_list, model_list = [], [], []
for data in data_loop_list:
    for model in model_loop_list:
        for method in method_loop_list:

            data_list.append(data)
            model_list.append(model)
            method_list.append(method)

            rouge_list, bert_list, harim_list, summac_conv_list, summac_zs_list = print_one_prune(data, model.split('/')[-1], method)
            rouge_list_final.append(return_list_mean_std(rouge_list))
            bert_list_final.append(return_list_mean_std(bert_list))
            harim_list_final.append(return_list_mean_std(harim_list))
            summac_conv_list_final.append(return_list_mean_std(summac_conv_list))
            summac_zs_list_final.append(return_list_mean_std(summac_zs_list))
            # rouge_list_final.append(np.mean(rouge_list))
            # bert_list_final.append(np.mean(bert_list))
            # harim_list_final.append(np.mean(harim_list))
            # summac_conv_list_final.append(np.mean(summac_conv_list))
            # summac_zs_list_final.append(np.mean(summac_zs_list))



df = pd.DataFrame(list(zip(data_list, model_list, method_list, rouge_list_final, bert_list_final, harim_list_final, summac_conv_list_final, summac_zs_list_final)), 
               columns = ['Data',      'Model',     'Prune',     'RougeL',       'BERTScore',          'HaRiM',      'SUMMAC_conv',      'SUMMAC_zs'])

df.replace(to_replace=["fullmodel", "sparsegpt", "wanda", "NousResearch/Nous-Hermes-llama-2-7b", "NousResearch/Nous-Hermes-Llama2-13b", "tiiuae/falcon-7b-instruct"],
           value=["Original", "SparseGPT", "Wanda", "Llama 7b", "Llama 13b", "Falcon 7b"], inplace=True) #


df.to_csv(f'generated_output/mean_std_result.csv', index=False)


df = pd.read_csv(f'generated_output/mean_std_result.csv')

for metrics in ['RougeL',       'BERTScore',          'HaRiM',      'SUMMAC_conv',      'SUMMAC_zs']:
    RougeL = df[['Data', 'Model', 'Prune', metrics]]
    RougeL_new = RougeL[RougeL['Data'] == 'polytope'][['Model','Prune',metrics]].rename(columns={metrics: "Polytope"})
    RougeL_new['FactCC'] = RougeL[RougeL['Data'] == 'factcc'][metrics].values
    RougeL_new['SummEval'] = RougeL[RougeL['Data'] == 'summeval'][metrics].values

    RougeL_new.to_csv(f'generated_output/{metrics}_result.csv', index=False)

