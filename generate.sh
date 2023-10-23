#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --time=4-00:00:00


#SBATCH --job-name=xsum_llama13

module load Anaconda3/2022.10
module load CUDA/11.8.0
source activate seq

#python generate_summary_fast.py

for prune_method in "sparsegpt" "wanda" "fullmodel" "magnitude"
do
python generate_and_save_summary.py --prune_method $prune_method --data "xsumfaith" \
                                --model "tiiuae/falcon-7b-instruct"
done
# for method in "wanda" "fullmodel" "sparsegpt"
# do
# for model in "decapoda-research/llama-7b-hf" "decapoda-research/llama-13b-hf" "decapoda-research/llama-30b-hf", 
# do
# python generate_summary_fast.py --prune_method $method --data polytope \
#                                 --model $model 

# done
# done
