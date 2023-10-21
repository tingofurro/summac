#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=4-00:00:00


#SBATCH --job-name=spar_lama7b

module load Anaconda3/2022.10
module load CUDA/11.8.0
source activate seq


for data in "summeval" "polytope" "factcc" 
do
python generate_summary_fast.py --prune_method "sparsegpt" --data $data \
                                --model "NousResearch/Nous-Hermes-llama-2-7b"
done
# for method in "wanda" "fullmodel" "sparsegpt"
# do
# for model in "decapoda-research/llama-7b-hf" "decapoda-research/llama-13b-hf" "decapoda-research/llama-30b-hf", 
# do
# python generate_summary_fast.py --prune_method $method --data polytope \
#                                 --model $model 

# done
# done
