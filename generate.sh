#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --time=4-00:00:00


#SBATCH --job-name=xsum-fal-magnitude

module load Anaconda3/2022.10
module load CUDA/11.8.0
source activate seq


for prune_method in "sparsegpt" "wanda" "fullmodel" "magnitude"
do
for prompt_id in "A" "B" "C"
do
python generate_and_save_summary.py --prune_method $prune_method --data "polytope" \
                                     --model "NousResearch/Nous-Hermes-llama-2-7b" --prompt_id $prompt_id
done
done



# for method in "wanda" "fullmodel" "sparsegpt"
# do
# for model in "decapoda-research/llama-7b-hf" "decapoda-research/llama-13b-hf" "decapoda-research/llama-30b-hf", 
# do
# python generate_summary_fast.py --prune_method $method --data polytope \
#                                 --model $model 

# done
# done
