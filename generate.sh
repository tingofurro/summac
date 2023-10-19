#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00


#SBATCH --job-name=wad_13b

module load Anaconda3/2022.10
module load CUDA/11.8.0
source activate seq


for data in "polytope" "factcc" "summeval"
do
python generate_summary_fast.py --prune_method "sparsegpt" --data $data \
                                --model "NousResearch/Nous-Hermes-Llama2-13b"
done
# for method in "wanda" "fullmodel" "sparsegpt"
# do
# for model in "decapoda-research/llama-7b-hf" "decapoda-research/llama-13b-hf" "decapoda-research/llama-30b-hf", 
# do
# python generate_summary_fast.py --prune_method $method --data polytope \
#                                 --model $model 

# done
# done
